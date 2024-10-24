import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn

def seed_all(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def fwrite(log_file, s):
    with open(log_file, 'a', buffering=1) as fp:
        fp.write(s)

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def BPTT_attack(freq_filter, model, image, T):
    model.module.set_simulation_time(T, mode='bptt')
    output = model(image,freq_filter)
    output = output.mean(1)
    model.module.set_simulation_time(T)
    return output

def BPTR_attack(model, image, T,freq_filter=None):
    model.set_simulation_time(T, mode='bptr')
    output = model(image,freq_filter).mean(0)
    model.set_simulation_time(T)
    return output

def Act_attack(model, image, T):
    model.set_simulation_time(0)
    output = model(image)
    model.set_simulation_time(T)
    return output



def make_filter_0(H, W, filter_windows): 
    """
    params:
        H, W
        filter_windows: list; window_size = value * 2
    """
    crow = int(H / 2)
    ccol = int(W / 2)
    result = []
    for i, length in enumerate(filter_windows):
        output = torch.zeros([H, W])
        output[crow-length:crow+length, ccol-length:ccol+length] = 1
        result.append(output)
    return torch.stack(result, 0).cuda()

def random_filter_image(image, H, W, min_size, max_size): 
    r = random.randint(min_size, max_size)
    crow = int(H / 2)
    ccol = int(W / 2)
    filt = torch.zeros([H, W]) 
    filt[crow-r:crow+r, ccol-r:ccol+r] = 1 
    fourier_transform = torch.fft.fftshift(torch.fft.fft2(image, dim=(-2, -1)))
    mask_image = fourier_transform * filt.unsqueeze(0).unsqueeze(0)
    freq_image = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(mask_image), dim=(-1, -2)))

    return freq_image

