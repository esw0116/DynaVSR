'''
Test Vid4 (SR) and REDS4 (SR-clean, SR-blur, deblur-clean, deblur-compression) datasets
'''
import argparse
import os
import os.path as osp
import glob
import logging
import numpy as np
import cv2
import torch

import utils.util as util
import data.Backup.util as data_util
import models.archs.EDVR_arch as EDVR_arch
import imageio


def main():
    #################
    # configurations
    #################
    flip_test = False
    scale = 4
    N_in = 7
    predeblur, HR_in = False, False
    n_feats = 128
    back_RBs = 40

    save_imgs = False
    prog = argparse.ArgumentParser()
    prog.add_argument('--train_mode', '-t', type=str, default='REDS', help='train mode')
    prog.add_argument('--data_mode', '-m', type=str, default=None, help='data_mode')
    prog.add_argument('--degradation_mode', '-d', type=str, default='impulse', choices=('impulse', 'bicubic', 'preset'), help='path to image output directory.')
    prog.add_argument('--sigma_x', '-sx', type=float, default=1, help='sigma_x')
    prog.add_argument('--sigma_y', '-sy', type=float, default=0, help='sigma_y')
    prog.add_argument('--theta', '-th', type=float, default=0, help='theta')
    prog.add_argument('--model', type=str, default=None, help='name for subdirectory')

    args = prog.parse_args()

    train_data_mode = args.train_mode
    data_mode = args.data_mode
    if data_mode is None:
        if train_data_mode == 'Vimeo':
            data_mode = 'Vid4'
        elif train_data_mode == 'REDS':
            data_mode = 'REDS'
    degradation_mode = args.degradation_mode  # impulse | bicubic | preset
    sig_x, sig_y, the = args.sigma_x, args.sigma_y, args.theta
    if sig_y == 0:
        sig_y = sig_x

    folder_subname = 'preset' if degradation_mode == 'preset' else degradation_mode + '_' + str(
        '{:.1f}'.format(sig_x)) + '_' + str('{:.1f}'.format(sig_y)) + '_' + str('{:.1f}'.format(the))

    #### dataset
    if data_mode == 'Vid4':
        test_dataset_folder = '../dataset/Vid4/LR_{}/X1_KG_ZSSR'.format(folder_subname)
        #test_dataset_folder = '../dataset/Vid4/LR_{}/X1_CF_DBPN'.format(folder_subname)
        GT_dataset_folder = '../dataset/Vid4/HR'
    elif data_mode == 'MM522':
        test_dataset_folder = '../dataset/MM522val/LR_bicubic/X1_KG_ZSSR'
        GT_dataset_folder = '../dataset/MM522val/HR'
    else:
        # test_dataset_folder = '../dataset/REDS4/LR_bicubic/X{}'.format(scale)
        test_dataset_folder = '../dataset/REDS/train/LR_{}/X1_KG_ZSSR'.format(folder_subname)
        #test_dataset_folder = '../dataset/REDS/train/LR_{}/X1_CF_DBPN'.format(folder_subname)
        GT_dataset_folder = '../dataset/REDS/train/HR'

    #### evaluation
    crop_border = 0
    border_frame = 0 # N_in // 2  # border frames when evaluate
    # temporal padding mode
    padding = 'new_info'

    save_folder = '../results/{}'.format(data_mode)
    util.mkdirs(save_folder)
    util.setup_logger('base', save_folder, 'test', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')

    #### log info
    logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Flip test: {}'.format(flip_test))

    avg_psnr_l, avg_psnr_center_l, avg_psnr_border_l = [], [], []
    avg_ssim_l, avg_ssim_center_l, avg_ssim_border_l = [], [], []
    subfolder_name_l = []

    subfolder_l = sorted(glob.glob(osp.join(test_dataset_folder, '*')))
    subfolder_GT_l = sorted(glob.glob(osp.join(GT_dataset_folder, '*')))
    if data_mode == 'REDS':
        subfolder_GT_l = [k for k in subfolder_GT_l if
                          k.find('000') >= 0 or k.find('011') >= 0 or k.find('015') >= 0 or k.find('020') >= 0]
    # for each subfolder
    for subfolder, subfolder_GT in zip(subfolder_l, subfolder_GT_l):
        subfolder_name = osp.basename(subfolder)
        subfolder_name_l.append(subfolder_name)
        save_subfolder = osp.join(save_folder, subfolder_name)

        img_path_l = sorted(glob.glob(osp.join(subfolder_GT, '*5.png')))  ## *5.png
        max_idx = len(img_path_l)
        if save_imgs:
            util.mkdirs(save_subfolder)

        #### read LQ and GT images
        img_LQ_l = []
        img_GT_l = []
        for img_LQ_path in sorted(glob.glob(osp.join(subfolder, '*5.png.png'))):
            img_LQ_l.append(data_util.read_img(None, img_LQ_path))

        for img_GT_path in sorted(glob.glob(osp.join(subfolder_GT, '*5.png'))):   ### *5.png
            img_GT_l.append(data_util.read_img(None, img_GT_path))

        avg_psnr, avg_psnr_border, avg_psnr_center, N_border, N_center = 0, 0, 0, 0, 0
        avg_ssim, avg_ssim_border, avg_ssim_center = 0, 0, 0

        # process each image
        for img_idx, img_path in enumerate(img_path_l):
            img_name = osp.splitext(osp.basename(img_path))[0]

            if save_imgs:
                cv2.imwrite(osp.join(save_subfolder, '{}.png'.format(img_name)), output)

            # calculate PSNR
            output = np.copy(img_LQ_l[img_idx])
            GT = np.copy(img_GT_l[img_idx])
            '''
            output_tensor = torch.from_numpy(np.copy(output[:,:,::-1])).permute(2,0,1)
            GT_tensor = torch.from_numpy(np.copy(GT[:,:,::-1])).permute(2,0,1).type_as(output_tensor)
            torch.save(output_tensor.cpu(), '../results/sr_test.pt')
            torch.save(GT_tensor.cpu(), '../results/hr_test.pt')
            my_psnr = utility.calc_psnr(output_tensor, GT_tensor)
            GT_tensor = GT_tensor.cpu().numpy().transpose(1,2,0)
            imageio.imwrite('../results/hr_test.png', GT_tensor)
            print('saved', my_psnr)
            '''
            '''
            # For REDS, evaluate on RGB channels; for Vid4, evaluate on the Y channel
            if data_mode == 'Vid4' or 'sharp_bicubic' or 'MM522':  # bgr2y, [0, 1]
                GT = data_util.bgr2ycbcr(GT, only_y=True)
                output = data_util.bgr2ycbcr(output, only_y=True)
            '''
            output = (output * 255).round().astype('uint8')
            GT = (GT * 255).round().astype('uint8')
            output, GT = util.crop_border([output, GT], crop_border)
            crt_psnr = util.calculate_psnr(output, GT)
            crt_ssim = util.calculate_ssim(output, GT)

            # logger.info('{:3d} - {:16} \tPSNR: {:.6f} dB \tSSIM: {:.6f}'.format(img_idx + 1, img_name, crt_psnr, crt_ssim))

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
                    'Border PSNR: {:.6f} dB for {} frames.'.format(subfolder_name, avg_psnr,
                                                                   (N_center + N_border),
                                                                   avg_psnr_center, N_center,
                                                                   avg_psnr_border, N_border))

        logger.info('Folder {} - Average SSIM: {:.6f} for {} frames; '
                    'Center SSIM: {:.6f} for {} frames; '
                    'Border SSIM: {:.6f} for {} frames.'.format(subfolder_name, avg_ssim,
                                                                   (N_center + N_border),
                                                                   avg_ssim_center, N_center,
                                                                   avg_ssim_border, N_border))

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
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Flip test: {}'.format(flip_test))
    logger.info('Total Average PSNR: {:.6f} dB for {} clips. '
                'Center PSNR: {:.6f} dB. Border PSNR: {:.6f} dB.'.format(
                    sum(avg_psnr_l) / len(avg_psnr_l), len(subfolder_l),
                    sum(avg_psnr_center_l) / len(avg_psnr_center_l),
                    sum(avg_psnr_border_l) / len(avg_psnr_border_l)))
    logger.info('Total Average SSIM: {:.6f} for {} clips. '
                'Center SSIM: {:.6f}. Border PSNR: {:.6f}.'.format(
                    sum(avg_ssim_l) / len(avg_ssim_l), len(subfolder_l),
                    sum(avg_ssim_center_l) / len(avg_ssim_center_l),
                    sum(avg_ssim_border_l) / len(avg_ssim_border_l)))

    print('\n\n\n')

if __name__ == '__main__':
    main()
