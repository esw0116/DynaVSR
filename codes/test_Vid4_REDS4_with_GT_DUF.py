"""
DUF testing script, test Vid4 (SR) and REDS4 (SR-clean) datasets
write to txt log file
"""
import argparse
import os
import os.path as osp
import glob
import logging
import numpy as np
import cv2
import torch

import utils.util as util
import data.util as data_util
import models.archs.DUF_arch as DUF_arch


def main():
    #################
    # configurations
    #################
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    save_imgs = False

    prog = argparse.ArgumentParser()
    prog.add_argument('--train_mode', '-t', type=str, default='Vimeo', help='train mode')
    prog.add_argument('--data_mode', '-m', type=str, default=None, help='data_mode')
    prog.add_argument('--degradation_mode', '-d', type=str, default='impulse', choices=('impulse', 'bicubic', 'preset'), help='path to image output directory.')
    prog.add_argument('--sigma_x', '-sx', type=float, default=1, help='sigma_x')
    prog.add_argument('--sigma_y', '-sy', type=float, default=0, help='sigma_y')
    prog.add_argument('--theta', '-th', type=float, default=0, help='theta')

    args = prog.parse_args()

    train_mode = args.train_mode
    data_mode = args.data_mode
    if data_mode is None:
        if train_mode == 'Vimeo':
            data_mode = 'Vid4'
        elif train_mode == 'REDS':
            data_mode = 'REDS'
    degradation_mode = args.degradation_mode  # impulse | bicubic | preset
    sig_x, sig_y, the = args.sigma_x, args.sigma_y, args.theta
    if sig_y == 0:
        sig_y = sig_x


    # Possible combinations: (2, 16), (3, 16), (4, 16), (4, 28), (4, 52)
    scale = 2
    layer = 16
    assert (scale, layer) in [(2, 16), (3, 16), (4, 16), (4, 28),
                              (4, 52)], 'Unrecognized (scale, layer) combination'

    # model
    N_in = 7
    # model_path = '../experiments/pretrained_models/DUF_{}L_BLIND_{}_FT_report.pth'.format(layer, train_mode[0])
    model_path = '../experiments/pretrained_models/DUF_{}L_{}_S{}.pth'.format(layer, train_mode, scale)
    # model_path = '../experiments/pretrained_models/DUF_x2_16L_official.pth'
    adapt_official = True  # if 'official' in model_path else False
    DUF_downsampling = False  # True | False
    if layer == 16:
        model = DUF_arch.DUF_16L(scale=scale, adapt_official=adapt_official)
    elif layer == 28:
        model = DUF_arch.DUF_28L(scale=scale, adapt_official=adapt_official)
    elif layer == 52:
        model = DUF_arch.DUF_52L(scale=scale, adapt_official=adapt_official)

    #### dataset
    folder_subname = 'preset' if degradation_mode == 'preset' else degradation_mode + '_' + str(
        '{:.1f}'.format(sig_x)) + '_' + str('{:.1f}'.format(sig_y)) + '_' + str('{:.1f}'.format(the))

    # folder_subname = degradation_mode + '_' + str('{:.1f}'.format(sig_x)) + '_' + str('{:.1f}'.format(sig_y)) + '_' + str('{:.1f}'.format(the))
    if data_mode == 'Vid4':
        # test_dataset_folder = '../dataset/Vid4/LR_bicubic/X{}'.format(scale)
        test_dataset_folder = '../dataset/Vid4/LR_{}/X{}'.format(folder_subname, scale)
        GT_dataset_folder = '../dataset/Vid4/HR'
    elif data_mode == 'MM522':
        test_dataset_folder = '../dataset/MM522val/LR_bicubic/X{}'.format(scale)
        GT_dataset_folder = '../dataset/MM522val/HR'
    else:
        # test_dataset_folder = '../dataset/REDS4/LR_bicubic/X{}'.format(scale)
        test_dataset_folder = '../dataset/REDS/train/LR_{}/X{}'.format(folder_subname, scale)
        GT_dataset_folder = '../dataset/REDS/train/HR'

    #### evaluation
    crop_border = 0
    border_frame = N_in // 2  # border frames when evaluate
    # temporal padding mode
    padding = 'new_info'  # different from the official testing codes, which pads zeros.
    ############################################################################
    device = torch.device('cuda')
    save_folder = '../results/{}'.format(data_mode)
    util.mkdirs(save_folder)
    util.setup_logger('base', save_folder, 'test', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')

    #### log info
    logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Save images: {}'.format(save_imgs))

    def read_image(img_path):
        '''read one image from img_path
        Return img: HWC, BGR, [0,1], numpy
        '''
        img_GT = cv2.imread(img_path)
        img = img_GT.astype(np.float32) / 255.
        return img

    def read_seq_imgs(img_seq_path):
        '''read a sequence of images'''
        img_path_l = sorted(glob.glob(img_seq_path + '/*'))
        img_l = [read_image(v) for v in img_path_l]
        # stack to TCHW, RGB, [0,1], torch
        imgs = np.stack(img_l, axis=0)
        imgs = imgs[:, :, :, [2, 1, 0]]
        imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(imgs, (0, 3, 1, 2)))).float()
        return imgs

    def index_generation(crt_i, max_n, N, padding='reflection'):
        '''
        padding: replicate | reflection | new_info | circle
        '''
        max_n = max_n - 1
        n_pad = N // 2
        return_l = []

        for i in range(crt_i - n_pad, crt_i + n_pad + 1):
            if i < 0:
                if padding == 'replicate':
                    add_idx = 0
                elif padding == 'reflection':
                    add_idx = -i
                elif padding == 'new_info':
                    add_idx = (crt_i + n_pad) + (-i)
                elif padding == 'circle':
                    add_idx = N + i
                else:
                    raise ValueError('Wrong padding mode')
            elif i > max_n:
                if padding == 'replicate':
                    add_idx = max_n
                elif padding == 'reflection':
                    add_idx = max_n * 2 - i
                elif padding == 'new_info':
                    add_idx = (crt_i - n_pad) - (i - max_n)
                elif padding == 'circle':
                    add_idx = i - N
                else:
                    raise ValueError('Wrong padding mode')
            else:
                add_idx = i
            return_l.append(add_idx)
        return return_l

    def single_forward(model, imgs_in):
        with torch.no_grad():
            model_output = model(imgs_in)
            if isinstance(model_output, list) or isinstance(model_output, tuple):
                output = model_output[0]
            else:
                output = model_output
        return output

    sub_folder_l = sorted(glob.glob(osp.join(test_dataset_folder, '*')))
    sub_folder_GT_l = sorted(glob.glob(osp.join(GT_dataset_folder, '*')))
    if data_mode == 'REDS':
        sub_folder_GT_l = [k for k in sub_folder_GT_l if
                          k.find('000') >= 0 or k.find('011') >= 0 or k.find('015') >= 0 or k.find('020') >= 0]
    #### set up the models
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    avg_psnr_l, avg_psnr_center_l, avg_psnr_border_l = [], [], []
    avg_ssim_l, avg_ssim_center_l, avg_ssim_border_l = [], [], []
    subfolder_name_l = []

    # for each sub-folder
    for sub_folder, sub_folder_GT in zip(sub_folder_l, sub_folder_GT_l):
        sub_folder_name = sub_folder.split('/')[-1]
        subfolder_name_l.append(sub_folder_name)
        save_sub_folder = osp.join(save_folder, sub_folder_name)

        img_path_l = sorted(glob.glob(sub_folder + '/*'))
        max_idx = len(img_path_l)

        if save_imgs:
            util.mkdirs(save_sub_folder)

        #### read LR images
        imgs = read_seq_imgs(sub_folder)
        #### read GT images
        img_GT_l = []

        for img_GT_path in sorted(glob.glob(osp.join(sub_folder_GT, '*'))):
            img_GT_l.append(read_image(img_GT_path))

        # When using the downsampling in DUF official code, we downsample the HR images
        if DUF_downsampling:
            sub_folder = sub_folder_GT
            img_path_l = sorted(glob.glob(sub_folder + '/*'))
            max_idx = len(img_path_l)
            imgs = read_seq_imgs(sub_folder)

        avg_psnr, avg_psnr_border, avg_psnr_center, N_border, N_center = 0, 0, 0, 0, 0
        avg_ssim, avg_ssim_border, avg_ssim_center = 0, 0, 0

        # process each image
        num_images = len(img_path_l)
        for img_idx, img_path in enumerate(img_path_l):
            img_name = osp.splitext(osp.basename(img_path))[0]
            c_idx = int(osp.splitext(osp.basename(img_path))[0])
            select_idx = index_generation(c_idx, max_idx, N_in, padding=padding)
            # get input images
            imgs_in = imgs.index_select(0, torch.LongTensor(select_idx)).unsqueeze(0).to(device)

            # Downsample the HR images
            H, W = imgs_in.size(3), imgs_in.size(4)
            if DUF_downsampling:
                imgs_in = util.DUF_downsample(imgs_in, sigma=1.3, scale=scale)

            output = single_forward(model, imgs_in)

            # Crop to the original shape
            if scale == 3:
                pad_h = scale - (H % scale)
                pad_w = scale - (W % scale)
                if pad_h > 0:
                    output = output[:, :, :-pad_h, :]
                if pad_w > 0:
                    output = output[:, :, :, :-pad_w]
            output_f = output.data.float().cpu().squeeze(0)
            output = util.tensor2img(output_f)

            # save imgs
            if save_imgs:
                cv2.imwrite(osp.join(save_sub_folder, '{:08d}.png'.format(c_idx)), output)

            #### calculate PSNR
            output = output / 255.
            GT = np.copy(img_GT_l[img_idx])
            output = (output * 255).round().astype('uint8')
            GT = (GT * 255).round().astype('uint8')
            output, GT = util.crop_border([output, GT], crop_border)
            crt_psnr = util.calculate_psnr(output, GT)
            crt_ssim = util.calculate_ssim(output, GT)


            logger.info('{:3d} - {:16} \tPSNR: {:.6f} dB \tSSIM: {:.6f}'.format(img_idx + 1, img_name, crt_psnr, crt_ssim))

            if img_idx >= border_frame and img_idx < max_idx - border_frame:  # center frames
                avg_psnr_center += crt_psnr
                avg_ssim_center += crt_ssim
                N_center += 1
            else:  # border frames
                avg_psnr_border += crt_psnr
                avg_ssim_border += crt_ssim
                N_border += 1

        avg_psnr = (avg_psnr_center + avg_psnr_border) / (N_center + N_border)
        avg_psnr_center = avg_psnr_center / N_center
        avg_psnr_border = 0 if N_border == 0 else avg_psnr_border / N_border
        avg_psnr_l.append(avg_psnr)
        avg_psnr_center_l.append(avg_psnr_center)
        avg_psnr_border_l.append(avg_psnr_border)

        avg_ssim = (avg_ssim_center + avg_ssim_border) / (N_center + N_border)
        avg_ssim_center = avg_ssim_center / N_center
        avg_ssim_border = 0 if N_border == 0 else avg_ssim_border / N_border
        avg_ssim_l.append(avg_ssim)
        avg_ssim_center_l.append(avg_ssim_center)
        avg_ssim_border_l.append(avg_ssim_border)

        logger.info('Folder {} - Average PSNR: {:.6f} dB for {} frames; '
                    'Center PSNR: {:.6f} dB for {} frames; '
                    'Border PSNR: {:.6f} dB for {} frames.'.format(sub_folder_name, avg_psnr,
                                                                   (N_center + N_border),
                                                                   avg_psnr_center, N_center,
                                                                   avg_psnr_border, N_border))
        '''
        logger.info('Folder {} - Average SSIM: {:.6f} for {} frames; '
                    'Center SSIM: {:.6f} for {} frames; '
                    'Border SSIM: {:.6f} for {} frames.'.format(sub_folder_name, avg_ssim,
                                                                (N_center + N_border),
                                                                avg_ssim_center, N_center,
                                                                avg_ssim_border, N_border))
        '''
    '''
    logger.info('################ Tidy Outputs ################')
    for subfolder_name, psnr, psnr_center, psnr_border in zip(subfolder_name_l, avg_psnr_l,
                                                              avg_psnr_center_l, avg_psnr_border_l):
        logger.info('Folder {} - Average PSNR: {:.6f} dB. '
                    'Center PSNR: {:.6f} dB. '
                    'Border PSNR: {:.6f} dB.'.format(subfolder_name, psnr, psnr_center, psnr_border))
    for subfolder_name, ssim, ssim_center, ssim_border in zip(subfolder_name_l, avg_ssim_l,
                                                              avg_ssim_center_l, avg_ssim_border_l):
        logger.info('Folder {} - Average SSIM: {:.6f}. '
                    'Center SSIM: {:.6f}. '
                    'Border SSIM: {:.6f}.'.format(subfolder_name, ssim, ssim_center, ssim_border))
    '''
    logger.info('################ Final Results ################')
    logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Total Average PSNR: {:.6f} dB for {} clips. '
                'Center PSNR: {:.6f} dB. Border PSNR: {:.6f} dB.'.format(
        sum(avg_psnr_l) / len(avg_psnr_l), len(sub_folder_l),
        sum(avg_psnr_center_l) / len(avg_psnr_center_l),
        sum(avg_psnr_border_l) / len(avg_psnr_border_l)))
    logger.info('Total Average SSIM: {:.6f} for {} clips. '
                'Center SSIM: {:.6f}. Border PSNR: {:.6f}.'.format(
        sum(avg_ssim_l) / len(avg_ssim_l), len(sub_folder_l),
        sum(avg_ssim_center_l) / len(avg_ssim_center_l),
        sum(avg_ssim_border_l) / len(avg_ssim_border_l)))

    print('\n\n\n')

if __name__ == '__main__':
    main()
