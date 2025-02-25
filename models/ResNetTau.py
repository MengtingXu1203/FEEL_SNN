import random

from models.layers import *


class conv(nn.Module):
    def __init__(self,in_plane, out_plane, kernel_size, stride,padding, bias=True):
        super(conv, self).__init__()
        self.fwd = nn.Sequential(nn.Conv2d(in_plane,out_plane,kernel_size=kernel_size,stride=stride,padding=padding, bias=bias),
        nn.BatchNorm2d(out_plane))

    def forward(self,x):
        x = self.fwd(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, T, tau, stride=1, shortcut=None, shape= None):
        super(ResidualBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride
        self.conv1 = conv(in_ch, out_ch, 3, stride, 1, bias=False)
        self.neuron1 = LIFSpike(T, tau=tau, shape=shape)
        self.conv2 = conv(out_ch, out_ch, 3, 1, 1, bias=False)
        self.neuron2 = LIFSpike(T, tau=tau, shape=shape)
        self.right = shortcut

    def forward(self, input):
        out = self.conv1(input)
        out = self.neuron1(out)
        out = self.conv2(out)
        residual = input if self.right is None else self.right(input)
        out += residual
        out = self.neuron2(out)
        return out


class ResNet19Tau(nn.Module):
    def __init__(self, T, tau=1.0, num_class=10, norm=None,args=None):
        super(ResNet19Tau, self).__init__()
        self.T = T
        self.tau = tau
        self.args = args
        if norm is not None and isinstance(norm, tuple):
            self.norm = TensorNormalization(*norm)
        else:
            self.norm = TensorNormalization((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        self.pre_conv = conv(3, 128, 3, stride=1, padding=1, bias=False)
        self.neuron1 = LIFSpikeTau(self.T, tau=self.tau, shape = [128,64,64]) #[128,64,64]
        self.layer1 = self.make_layer(128, 128, 3, stride=1, shape=[128,64,64])
        self.layer2 = self.make_layer(128, 256, 3, stride=2, shape = [256,32,32])
        self.layer3 = self.make_layer(256, 512, 2, stride=2, shape = [512,16,16])
        self.layer4 = nn.Sequential() #self.make_layer(512, 1024, 3, stride=2)
        self.pool = nn.AvgPool2d(2,2)
        #W = 4
        W = 8 ##tinyimagenet
        self.fc1 = nn.Sequential(
            nn.Linear(512*W*W, 256),
            nn.BatchNorm1d(256)
        )
        self.fc2 = nn.Linear(256, num_class)
        self.merge = MergeTemporalDim(T)
        self.expand = ExpandTemporalDim(T)
        self.encode = Poisson()
        self.poisson = False
        self.hooked_grad = None

    def make_layer(self, in_ch, out_ch, block_num, stride=1,shape = None):
        shortcut = conv(in_ch, out_ch, 1, stride, 0, bias=False)
        layers = []
        layers.append(ResidualBlock(in_ch, out_ch, self.T, self.tau, stride, shortcut))
        for i in range(1, block_num):
            layers.append(ResidualBlock(out_ch, out_ch, self.T, self.tau, shape=shape))
        return nn.Sequential(*layers)

        
    # pass T to determine whether it is an ANN or SNN
    def set_simulation_time(self, T, mode='bptt'):
        self.T = T
        for module in self.modules():
            if isinstance(module, (LIFSpike, ExpandTemporalDim)):
                module.T = T
                if isinstance(module, LIFSpike):
                    module.mode = mode
        return
    
    def forward(self, input,freq_filter=None):
        if self.T > 0:
            if self.args.encode == 'ft':
                input = ft(input, freq_filter) 
            else:
                input = add_dimention(input, self.T)
            if self.poisson:
                input = self.encode(input)
            input = self.merge(input)
        x = self.pre_conv(input)
        x = self.neuron1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        out = self.fc2(x)
        if self.T > 0:
            out = self.expand(out)
        out = out.permute(1,0,2)
        return out