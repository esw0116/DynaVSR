import random

import torch
import math
import numpy as np
import data.util as data_util
import time

def _apply_all(func, x):
    '''
    Recursively apply the function to the input.

    Args:
        func (function): A function to be applied to the input.
        x (object or a list of objects): The input object(s).

    Return:
        func(x): See the below examples.

    Example::
        >>> _apply_all(f, x)
        >>> f(x)

        >>> _apply_all(f, [x1, x2])
        >>> [f(x1), f(x2)]

        >>> _apply_all(f, [x1, x2, [x3, x4]])
        >>> [f(x1), f(x2), [f(x3), f(x4)]]
    '''
    if isinstance(x, (list, tuple)):
        if len(x) == 1:
            return func(x[0])
        else:
            return [_apply_all(func, _x) for _x in x]
    else:
        return func(x)


def set_kernel_params(sigma_x=None, sigma_y=None, theta=None):
    min_sigma = 0.2
    var_sigma = 4.8
    if sigma_x is None:
        sigma_x = min_sigma + np.random.random_sample() * var_sigma
    if sigma_y is None:
        sigma_y = min_sigma + np.random.random_sample() * var_sigma
    if theta is None:
        theta = -math.pi + np.random.random_sample() * 2 * math.pi

    return {'theta': theta, 'sigma': [sigma_x, sigma_y]}

def eval_crop(hr, lr, scale):
    h, w = lr.shape[0:2]
    h *= scale
    w *= scale
    return hr[:h, :w]

def common_crop(*args, patch_size=96):
    '''
    Crop given patches.

    Args:
        args (list of 'torch.Tensor' or 'list of torch.Tensor'):
            Images or lists of images to be cropped.
            Cropping position is fixed for all images in the single *args.

        patch_size (int, optional):
        scale (int, optional):

    Return:

    '''
    # Find the lowest resolution
    min_h = min(x.shape[-2] for x in args)
    min_w = min(x.shape[-1] for x in args)

    py = random.randrange(0, min_h - patch_size + 1)
    px = random.randrange(0, min_w - patch_size + 1)

    def _crop(x):
        h = x.shape[-2]
        s = int(h // min_h)
        x = x[..., s * py:s * (py + patch_size), s * px:s * (px + patch_size)]
        return x

    return _apply_all(_crop, args)

def np_common_crop(*args, patch_size=96):
    '''
    Crop given patches.

    Args:
        args (list of 'np.array' or 'list of np.array'):
            Images or lists of images to be cropped.
            Cropping position is fixed for all images in the single *args.

        patch_size (int, optional):
        scale (int, optional):

    Return:

    '''
    # Find the lowest resolution
    min_h = min(x.shape[0] for x in args)
    min_w = min(x.shape[0] for x in args)

    py = random.randrange(0, min_h - patch_size + 1)
    px = random.randrange(0, min_w - patch_size + 1)

    def _crop(x):
        h = x.shape[0]
        s = int(h // min_h)
        x = x[s * py:s * (py + patch_size), s * px:s * (px + patch_size), ...]
        return x

    return _apply_all(_crop, args)

def get_min_in_axis(img, s_length, direction='horizontal'):
    # img: H W T
    smooth_img = np.zeros_like(img)
    h, w = img.shape[-2], img.shape[-1]
    if direction == 'horizontal':
        for i in range(w):
            smooth_img[:, :, i] = np.amin(img[:, :, data_util.index_generation(i, w, s_length, 'replicate')], axis=2)
    elif direction == 'vertical':
        for i in range(h):
            smooth_img[:, i, :] = np.amin(img[:, data_util.index_generation(i, h, s_length, 'replicate'), :], axis=1)
    return smooth_img

def crop(img_gt, img, img_lr, scale=2, patch_size=96):
    '''
    Crop given patches.

    Args:
        args (list of 'np.array' or 'list of np.array'):
            Images or lists of images to be cropped.
            Cropping position is fixed for all images in the single *args.

        patch_size (int, optional):
        scale (int, optional):

    Return:

    '''
    import pywt

    # Find the lowest resolution
    h, w = img.shape[-2] // 2, img.shape[-1] // 2
    # T C H W  --> H W T
    convert = (img.new_tensor([65.481, 128.553, 24.966]) / 255.0 + 16.0).reshape(1,3,1,1)
    img_y = img.mul(convert).sum(dim=1)
    _, (ch, cv, _) = pywt.dwt2(img_y, 'haar')
    # ch, cv become numpy array
    ch, cv = np.abs(ch), np.abs(cv)

    ch_minned = get_min_in_axis(ch, 9, 'horizontal')
    cv_minned = get_min_in_axis(cv, 9, 'vertical')
    mean_ch = np.mean(ch_minned, axis=(1,2))  # T
    mean_cv = np.mean(cv_minned, axis=(1,2))  # T
    for i in range(50):
        py = random.randrange(0, h - patch_size//2 + 1)
        px = random.randrange(0, w - patch_size//2 + 1)
        ch_patch = ch_minned[..., py: py+(patch_size//2), px: px+(patch_size//2)]
        cv_patch = cv_minned[..., py: py+(patch_size//2), px: px+(patch_size//2)]
        mean_ch_patch, mean_cv_patch = np.mean(ch_patch, axis=(1,2)), np.mean(cv_patch, axis=(1,2))
        if (mean_ch_patch >= mean_ch).all() and (mean_cv_patch >= mean_cv).all():
            break

    patch = img[..., py*2:py*2+patch_size, px*2:px*2+patch_size]
    if img_gt is not None:
        patch_gt = img_gt[..., py*2*scale:(py*2+patch_size)*scale, px*2*scale:(px*2+patch_size)*scale]
    else:
        patch_gt = None
    if img_lr is not None:
        patch_superlr = img_lr[..., (py*2)//scale:(py*2+patch_size)//scale, (px*2)//scale:(px*2+patch_size)//scale]
    else:
        patch_superlr = None
    return patch_gt, patch, patch_superlr
        

def crop_border(*args, border=[4,4]):
    '''
    Crop given patches.

    Args:
        args (list of 'np.array' or 'list of np.array'):
            Images or lists of images to be cropped.
            Cropping position is fixed for all images in the single *args.

        border (int, optional): [h, w], crop length in height, width, respectively

    Return:

    '''
    # Find the lowest resolution
    min_h = min(x.shape[-2] for x in args)
    min_w = min(x.shape[-1] for x in args)

    if isinstance(border, (int, float)):
        border = [border, border]

    def _crop_border(x):
        h = x.shape[-2]
        s = int(h // min_h)
        x = x[..., s * border[0]: -s * border[0], s * border[1]:-s * border[1]]
        return x

    return _apply_all(_crop_border, args)

def augment(*args, hflip=True, vflip=True, rot=False):
    """
        Apply random augmentations to given patches.
        Args:
        Return:
    """
    hflip = hflip and random.random() < 0.5
    vflip = vflip and random.random() < 0.5
    rot = rot and random.random() < 0.5

    def _augment(x):
        if hflip:
            x = torch.from_numpy(np.ascontiguousarray(x.numpy()[..., :, ::-1]))
        if vflip:
            x = torch.from_numpy(np.ascontiguousarray(x.numpy()[..., ::-1, :]))
        if rot:
            # Rotating kernel,
            if x.ndim == 2:
                x = x.permute(1, 0)
            # Single image C H W
            if x.ndim == 3:
                x = x.permute(0, 2, 1)
            # Video T C H W
            elif x.ndim == 4:
                x = x.permute(0, 1, 3, 2)

        return x

    return _apply_all(_augment, args)

def np2tensor(*args):
    '''
    Convert np.array to torch.Tensor.
    Output tensors will lie in between [-1, 1] (for adversarial training).

    Args:

    Return:

    '''
    def _np2tensor(x):
        if x.ndim == 3:
            # H x W x C -> C x H x W
            x = x.transpose((2, 0, 1))
        elif x.ndim == 4:
            # H x W x C x T -> T x C x H x W
            x = x.transpose((3, 2, 0, 1))

        x = np.ascontiguousarray(x)
        x = torch.from_numpy(x).float()     # [0, 255]
        x = (x / 255.)                 # [0, 1]
        return x

    return _apply_all(_np2tensor, args)


class Transformer(object):
    '''
    A class for handling various preprocessing for easy data preperation.
    '''

    def __init__(self, patch_size, hflip=True, vflip=True, rot=True):
        self.transforms = []

    def __call__(self, x):
        if not self.transforms:
            return x
        else:
            for transform in self.transforms:
                x = transform(x)

            return x

    def register(self):
        pass