import random

import numpy as np
import skimage.io as sio
import skimage.color as sc
import skimage.transform as st

import torch
import torch.nn.functional as F
from torchvision import transforms

def _apply(func, x):
    if isinstance(x, list):
        return list(map(lambda l: func(l), x))
    else:
        return func(x)

def crop(*args, ps=64, scale=4):
    # args[0] : hr, args[1] : lr
    if isinstance(args[1], list):
        h, w, _ = args[1][0].shape
    else:
        h, w, _ = args[1].shape

    px = random.randrange(0, w - ps + 1)
    py = random.randrange(0, h - ps + 1)

    hr_ps = ps * scale
    hr_px = px * scale
    hr_py = py * scale

    def _crop_hr(x):
        return x[hr_py:hr_py + hr_ps, hr_px:hr_px + hr_ps, :]

    def _crop_lr(x):
        return x[py:py + ps, px:px + ps, :]

    return [_apply(_crop_hr, a) for a in args[0:1]] + [_apply(_crop_lr, a) for a in args[1:]]

def np2tensor(*args, rgb=255):
    m = rgb / 255
    def _np2tensor(x):
        np_transpose = np.ascontiguousarray(x.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        return m * tensor
    
    ret = []
    for r in [_apply(_np2tensor, a) for a in args]:
        # Make 4D tensor if a sequence is given
        if isinstance(r, list):
            ret.append(torch.stack(r, dim=0))
        else:
            ret.append(r)

    return ret

def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(x):
        if hflip: x = x[:, ::-1, :]
        if vflip: x = x[::-1, :, :]
        if rot90: x = x.transpose(1, 0, 2)
        
        return x

    return [_apply(_augment, a) for a in args]

def additive_noise(*args, sigma=2, rgb=255):
    s = sigma * np.random.randn(1)
    if rgb != 255:
        s *= rgb/255
    
    def _additive_noise(x):
        n = s * np.random.randn(*x.shape)
        return x + n

    return [_apply(_additive_noise, a) for a in args]

def target_input(*args, sharp2sharp=True):
    
    sharp2sharp = sharp2sharp and random.random() < 0.1 # 10% probability
    if sharp2sharp:    # preseve sharp sample as is to suppress artifacts
        target = args[-1]   # seq_sharp
        input = [patch.copy() for patch in target]

        return [input, target]
    else:
        return [*args]


def BD_downsample(x, scale=4):
    """Downsamping with Gaussian kernel used in the DUF official code

    Args:
        x (Tensor, [T, B, C, H, W]): frames to be downsampled.
        scale (int): downsampling factor: 2 | 3 | 4.
    """

    assert scale in [2, 3, 4], 'Scale [{}] is not supported'.format(scale)

    def gkern(kernlen=13, nsig=1.6):
        import scipy.ndimage.filters as fi
        inp = np.zeros((kernlen, kernlen))
        # set element at the middle to one, a dirac delta
        inp[kernlen // 2, kernlen // 2] = 1
        # gaussian-smooth the dirac, resulting in a gaussian filter mask
        return fi.gaussian_filter(inp, nsig)

    T, B, C, H, W = x.size()
    x = x.view(-1, 1, H, W)
    pad_w, pad_h = 6 + scale * 2, 6 + scale * 2  # 6 is the pad of the gaussian filter
    r_h, r_w = 0, 0
    if scale == 3:
        r_h = 3 - (H % 3)
        r_w = 3 - (W % 3)
    x = F.pad(x, [pad_w, pad_w + r_w, pad_h, pad_h + r_h], 'reflect')

    gaussian_filter = torch.from_numpy(gkern(13, 0.4 * scale)).type_as(x).unsqueeze(0).unsqueeze(0)
    x = F.conv2d(x, gaussian_filter, stride=scale)
    x = x[:, :, 2:-2, 2:-2]
    x = x.view(T, B, C, x.size(2), x.size(3))
    return x
