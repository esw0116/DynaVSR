import sys
from os import path
import datetime

def dump_config(cfg, name):
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S\n')
    with open(path.join(name, 'config.txt'), 'a') as f:
        f.write(now)
        f.write('python ' + ' '.join(sys.argv) + '\n\n')
        for k, v in vars(cfg).items():
            f.write('{}: {}\n'.format(k, v))

        f.write('-' * 80 + '\n\n')

def quantize(x, uint8=True):
    '''

    '''
    x = (x + 1) * 127.5
    x = x.round()
    x = x.clamp(min=0, max=255)
    if uint8:
        x = x.byte()
    else:
        x = x / 255

    return x

def tensor2np(x, uint8=True):
    '''
    Convert the given Tensor to a numpy array.

    Args:
        x (torch.Tensor): In [-1, 1]

    Return:
        x (np.array, dtype=np.uint8): In [0, 255]
    '''
    x = quantize(x, uint8=uint8)
    x = x.permute(1, 2, 0)
    x = x.cpu().numpy()
    return x

def calc_psnr(x, y, rgb=False, margin=0, force=1, temporal_ignore=None):
    '''
    Calculate the PSNR between x and y.

    Note:
        **This function requires x, y in [0, 1]**
    '''
    # To set the dynamic range of x and y to 1.
    diff = (x - y)

    # Ignore some frames when evaulating a video.
    is_video = False
    if diff.dim() == 5 or (diff.dim() == 4 and diff.size(0) > 1):
        is_video = True
        if temporal_ignore is not None:
            idx = diff.new_ones(diff.size(1))
            idx[temporal_ignore] = 0
            idx = idx.long()
            diff = diff[idx==1]

    # Boundary crop (always from center)
    if force > 1:
        # Force cropped images to be n * [force]
        # Ex) force=8: 44 x 44 -> 40 x 40
        rem_h = diff.shape[-2] % force
        rem_w = diff.shape[-1] % force
        margin_top = rem_h // 2
        margin_bottom = rem_h - margin_top
        margin_left = rem_w // 2
        margin_right = rem_w - margin_left
        if margin_bottom > 0:
            diff = diff[..., margin_top:-margin_bottom, :]
        if margin_right > 0:
            diff = diff[..., margin_left:-margin_right]
    elif margin > 0:
        # Cropping fixed amount
        diff = diff[..., margin:-margin, margin:-margin]

    if not rgb:
        rgb2ycbcr = diff.new_tensor([65.738, 129.057, 25.064]) / 256
        if diff.dim() == 3:
            # An image (C x H x W)
            rgb2ycbcr = rgb2ycbcr.view(3, 1, 1)
            idx_c = 0
        elif diff.dim() == 4:
            # An image with a batch dimension (B x C x H x W)
            # or a video (T x C x H x W)
            rgb2ycbcr = rgb2ycbcr.view(1, 3, 1, 1)
            idx_c = 1
        elif diff.dim() == 5:
            # A video with a batch dimension (T x B x C x H x W)
            rgb2ycbcr = rgb2ycbcr.view(1, 1, 3, 1, 1)
            idx_c = 2

        diff = diff.mul(rgb2ycbcr).sum(dim=idx_c)

    if is_video:
        # PSNR of each image is averaged
        diff = diff.view(diff.size(0), -1)
    else:
        diff = diff.view(-1)

    # diff = diff.view(-1)

    mse = diff.pow(2).mean(dim=-1)
    psnr = -10 * mse.log10()

    if is_video:
        return psnr.mean().item(), psnr.tolist()
    else:
        return psnr.item(), 0

