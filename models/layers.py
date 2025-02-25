import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision
import math
import numpy as np
import cv2
import seaborn as sns

class TensorNormalization(nn.Module):
    def __init__(self,mean, std):
        super(TensorNormalization, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.mean = mean
        self.std = std
    def forward(self, X):
        return normalizex(X,self.mean,self.std)
    def extra_repr(self) -> str:
        return 'mean=%s, std=%s'%(self.mean, self.std)

def normalizex(tensor, mean, std):
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    if mean.device != tensor.device:
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)
    return tensor.sub(mean).div(std)

class MergeTemporalDim(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq: torch.Tensor):
        return x_seq.flatten(0, 1).contiguous()

class ExpandTemporalDim(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq: torch.Tensor):
        y_shape = [self.T, int(x_seq.shape[0]/self.T)]
        y_shape.extend(x_seq.shape[1:])
        return x_seq.view(y_shape)

class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input >= 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_output * tmp
        return grad_input, None


class RateBp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tau):
        mem = 0.
        spike_pot = []
        T = x.shape[0]
        for t in range(T):
            mem = mem*tau + x[t, ...]
            spike = ((mem - 1.) > 0).float()
            mem = (1 - spike) * mem
            spike_pot.append(spike)
        out = torch.stack(spike_pot, dim=0)
        ctx.save_for_backward(x, out, torch.tensor(tau))
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        x, out, tau = ctx.saved_tensors
        x = x.mean(0, keepdim=True)
        gamma = 0.2
        ext = 1 #
        des = 1
        grad = (x>=1-tau).float()*(x<=1+ext).float()*(des-gamma+gamma*tau)/(tau+ext) + (x<=1-tau).float()*(x>=0).float()*gamma
        grad_input = grad_output * grad
        return grad_input, None


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

floor = STE.apply

class qcfs(nn.Module):
    def __init__(self, up=8., t=8):
        super().__init__()
        # self.up = nn.Parameter(torch.tensor([up]), requires_grad=True)
        self.t = t

    def forward(self, x):
        # x = x / self.up
        x = torch.clamp(x, 0, 1)
        x = floor(x*self.t+0.5)/self.t
        # x = x * self.up
        return x

class LIFSpike(nn.Module):
    def __init__(self, T, thresh=1.0, tau=1., gama=1.0, shape = None):
        super(LIFSpike, self).__init__()
        self.act = ZIF.apply
        self.thresh = thresh
        self.tau = tau
        self.gama = gama
        self.expand = ExpandTemporalDim(T)
        self.merge = MergeTemporalDim(T)
        self.relu = qcfs()
        self.ratebp = RateBp.apply
        self.mode = 'bptt'
        self.T = T
        self.up = 1

    def forward(self, x):
        if self.mode == 'bptr' and self.T > 0:
            x = self.expand(x)
            x = self.ratebp(x, self.tau)
            x = self.merge(x)
        elif self.T > 0:
            x = self.expand(x)
            mem = 0.
            spike_pot = []
            for t in range(self.T):
                mem = mem * self.tau + x[t, ...]
                

                spike = self.act(mem - self.thresh, 1)
                mem = (1 - spike) * mem
                spike_pot.append(spike)
            x = torch.stack(spike_pot, dim=0)
            x = self.merge(x)
        else:
            x = self.relu(x)
        return x

##EL spiking neuron
class LIFSpikeTau(nn.Module):
    def __init__(self, T, thresh=1.0, tau=1., gama=1.0, shape = None):
        super(LIFSpikeTau, self).__init__()
        self.act = ZIF.apply
        self.thresh = thresh
        self.tau = tau
        self.gama = gama
        self.expand = ExpandTemporalDim(T)
        self.merge = MergeTemporalDim(T)
        self.relu = qcfs()
        self.ratebp = RateBp.apply
        self.mode = 'bptt'
        self.T = T
        self.up = 1
        self.shape = shape

        self.taus = nn.Parameter(torch.full((T,) + tuple(shape), fill_value=1.0)) ##EL

    def forward(self, x):
        if self.mode == 'bptr' and self.T > 0:
            x = self.expand(x)
            x = self.ratebp(x, self.tau)
            x = self.merge(x)
        elif self.T > 0:
            x = self.expand(x)
            #print(x.shape)
            mem = 0.
            spike_pot = []
            for t in range(self.T):
                mem = mem * (self.taus[t].to(x)) + x[t, ...]
                spike = self.act(mem - self.thresh, 1)
                mem = (1 - spike) * mem
                spike_pot.append(spike)
            x = torch.stack(spike_pot, dim=0)
            x = self.merge(x)
        else:
            x = self.relu(x)
        return x

def add_dimention(x, T):
    x = x.unsqueeze(0)
    x = x.repeat(T, 1, 1, 1, 1)
    return x

class ConvexCombination(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.comb = nn.Parameter(torch.ones(n) / n)

    def forward(self, *args):
        assert(len(args) == self.n)
        out = 0.
        for i in range(self.n):
            out += args[i] * self.comb[i]
        return out


def ft(x, freq_filter):
    """
    freq_filter: [T, H, W]
    """

    fourier_transform = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)))
    expanded_fouriers = fourier_transform.unsqueeze(0) * freq_filter.unsqueeze(1).unsqueeze(1) #[T, 1, 1, H, W]
    expanded_images = torch.stack([torch.abs(torch.fft.ifft2(torch.fft.ifftshift(expanded_fouriers[i]), dim=(-1, -2))) for i in range(expanded_fouriers.shape[0])], dim=0)

    return expanded_images

def poisson(x):
    rand = torch.rand_like(x)
    out = (x>=rand).float()
    return out

class Poi(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        rand = torch.rand_like(x)
        out = (x>=rand).float()
        ctx.save_for_backward(x, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, out = ctx.saved_tensors
        # x = x.mean(0, keepdim=True)
        # out = out.mean(0, keepdim=True)
        return grad_output

poi = Poi.apply

class Poisson(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = poi(x)
        return out



if __name__ == "__main__":
    main()