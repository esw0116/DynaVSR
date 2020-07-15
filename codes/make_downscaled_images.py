'''
Test Vid4 (SR) and REDS4 (SR-clean, SR-blur, deblur-clean, deblur-compression) datasets
'''
import argparse
import os
import os.path as osp
import glob
import logging
import numpy as np
import torch
import math
import data.Backup.util as data_util
from data import random_kernel_generator as rkg
from data import old_kernel_generator as oldkg

import imageio


def set_kernel_params(sigma_x=None, sigma_y=None, theta=None):
    if sigma_x is None:
        sigma_x = 0.2 + np.random.random_sample() * 1.8
    if sigma_y is None:
        sigma_y = 0.2 + np.random.random_sample() * 1.8
    if theta is None:
        theta = np.random.random_sample() * math.pi *2 - math.pi

    # 0~0.2 -> 0, 0.2~0.4 -> 1, 0.4~0.6 -> 2, 0.6~0.8 -> 3, 0.8~1 -> 4
    return {'theta': theta, 'sigma': [sigma_x, sigma_y]}


def main():
    #################
    # configurations
    #################
    device = torch.device('cuda')
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    prog = argparse.ArgumentParser()
    prog.add_argument('--dataset_mode', '-m', type=str, default='Vid4+REDS', help='data_mode')
    prog.add_argument('--degradation_mode', '-d', type=str, default='impulse', choices=('impulse', 'bicubic', 'preset'), help='path to image output directory.')
    prog.add_argument('--sigma_x', '-sx', type=float, default=1, help='sigma_x')
    prog.add_argument('--sigma_y', '-sy', type=float, default=0, help='sigma_y')
    prog.add_argument('--theta', '-t', type=float, default=0, help='theta')

    args = prog.parse_args()

    data_modes = args.dataset_mode
    degradation_mode = args.degradation_mode  # impulse | bicubic
    sig_x, sig_y, the = args.sigma_x, args.sigma_y, args.theta * math.pi / 180
    if sig_y == 0:
        sig_y = sig_x

    scale = 2
    kernel_size = 21

    N_frames = 7
    padding = 'new_info'
    ############################################################################

    # model = EDVR_arch.EDVR(n_feats, N_in, 8, 5, back_RBs, predeblur=False, HR_in=False, scale=scale)
    # est_model = LR_arch.DirectKernelEstimator_CMS(nf=n_feats)
    data_mode_l = data_modes.split('+')

    for i in range(len(data_mode_l)):
        data_mode = data_mode_l[i]
        #### dataset
        if data_mode == 'Vid4':
            kernel_folder = '../experiments/pretrained_models/Vid4Gauss.npy'
            dataset_folder = '../dataset/Vid4'
        elif data_mode == 'MM522':
            kernel_folder = '../experiments/pretrained_models/MM522Gauss.npy'
            dataset_folder = '../dataset/MM522val'
        else:
            kernel_folder = '../experiments/pretrained_models/REDSGauss.npy'
            dataset_folder = '../dataset/REDS/train'

        GT_dataset_folder = osp.join(dataset_folder, 'HR')
        save_folder_name = 'preset' if degradation_mode == 'preset' else degradation_mode + '_' + str('{:.1f}'.format(sig_x)) + '_' + str('{:.1f}'.format(sig_y)) + '_' + str('{:.1f}'.format(args.theta))
        save_folder = osp.join(dataset_folder, 'LR_'+save_folder_name, 'X'+str(scale))
        if not osp.exists(save_folder):
            os.makedirs(save_folder)

        save_folder2 = osp.join(dataset_folder, 'LR_'+save_folder_name, 'X'+str(scale*scale)) #*scale*scale))
        if not osp.exists(save_folder2):
            os.makedirs(save_folder2)
        #### log info
        # logger.info('Data: {} - {}'.format(data_mode, lr_set_method))
        # logger.info('Padding mode: {}'.format(padding))
        # logger.info('Model path: {}'.format(model_path))
        # logger.info('Save images: {}'.format(save_imgs))
        # logger.info('Flip test: {}'.format(flip_test))

        #### set up the models
        # model.load_state_dict(torch.load(model_path), strict=True)
        # model.eval()
        # model = model.to(device)

        avg_psnr_l, avg_psnr_center_l, avg_psnr_border_l = [], [], []
        subfolder_name_l = []
        # subfolder_l = sorted(glob.glob(osp.join(bicubic_dataset_folder, '*')))
        subfolder_GT_l = sorted(glob.glob(osp.join(GT_dataset_folder, '*')))
        if data_mode == 'REDS':
            subfolder_GT_l = [k for k in subfolder_GT_l if
                              k.find('000') >= 0 or k.find('011') >= 0 or k.find('015') >= 0 or k.find('020') >= 0]
        elif data_mode == 'MM522':
            subfolder_GT_l = [k for k in subfolder_GT_l if
                              k.find('001') >= 0 or k.find('005') >= 0 or k.find('008') >= 0 or k.find('009') >= 0]
        # for each subfolder
        # for subfolder, subfolder_GT in zip(subfolder_l, subfolder_GT_l):

        sig_x, sig_y, the = float(sig_x), float(sig_y), float(the)

        for subfolder_GT in subfolder_GT_l:
            print(subfolder_GT)
            gen_kwargs = set_kernel_params(sigma_x=sig_x, sigma_y=sig_y, theta=the)
            if degradation_mode == 'impulse' or degradation_mode == 'preset':
                kernel_gen = rkg.Degradation(kernel_size, scale, **gen_kwargs)
                if degradation_mode == 'preset':
                    kernel_preset = np.load(kernel_folder)
            else:
                kernel_gen = oldkg.Degradation(kernel_size, scale, type=0.7, **gen_kwargs)

            subfolder_name = osp.basename(subfolder_GT)
            subfolder_name_l.append(subfolder_name)
            save_subfolder = osp.join(save_folder, subfolder_name)
            if not osp.exists(save_subfolder):
                os.mkdir(save_subfolder)

            save_subfolder2 = osp.join(save_folder2, subfolder_name)
            if not osp.exists(save_subfolder2):
                os.mkdir(save_subfolder2)

            img_GT_path_l = sorted(glob.glob(osp.join(subfolder_GT, '*')))
            seq_length = len(img_GT_path_l)

            # max_idx = len(img_GT_path_l)
            # if save_imgs:
            #    util.mkdirs(save_subfolder)
            imgs_GT = data_util.read_img_seq(subfolder_GT)  # T C H W

            if degradation_mode == 'preset':
                for index in range(seq_length):
                    save_subsubfolder = osp.join(save_subfolder, osp.splitext(osp.basename(img_GT_path_l[index]))[0])
                    save_subsubfolder2 = osp.join(save_subfolder2, osp.splitext(osp.basename(img_GT_path_l[index]))[0])
                    if not osp.exists(save_subsubfolder):
                        os.mkdir(save_subsubfolder)
                    if not osp.exists(save_subsubfolder2):
                        os.mkdir(save_subsubfolder2)
                    kernel_gen.set_kernel_directly(kernel_preset[index])
                    imgs_HR = imgs_GT[data_util.index_generation(index, seq_length, N_frames, padding)]
                    imgs_LR = kernel_gen.apply(imgs_HR)
                    imgs_LR = imgs_LR.mul(255).clamp(0, 255).round().div(255)
                    imgs_LR_np = imgs_LR.permute(0, 2, 3, 1).cpu().numpy()
                    imgs_LR_np = (imgs_LR_np * 255).astype('uint8')
                    for i, img_LR in enumerate(imgs_LR_np):
                        imageio.imwrite(osp.join(save_subsubfolder, 'img{}.png'.format(i)), img_LR)

                    imgs_SuperLR = kernel_gen.apply(imgs_LR)
                    imgs_SuperLR = imgs_SuperLR.mul(255).clamp(0, 255).round().div(255)
                    imgs_SuperLR_np = imgs_SuperLR.permute(0, 2, 3, 1).cpu().numpy()
                    imgs_SuperLR_np = (imgs_SuperLR_np * 255).astype('uint8')
                    for i, img_SuperLR in enumerate(imgs_SuperLR_np):
                        imageio.imwrite(osp.join(save_subsubfolder2, 'img{}.png'.format(i)), img_SuperLR)

            else:
                count = 0
                imgs_GT_l = imgs_GT.split(32)
                for img_batch in imgs_GT_l:
                    if degradation_mode == 'preset':
                        kernel_gen.set_kernel_directly(kernel_preset[count])
                    img_lr_batch = kernel_gen.apply(img_batch)
                    img_lr_batch = img_lr_batch.permute(0, 2, 3, 1).cpu().numpy()
                    img_lr_batch = (img_lr_batch.clip(0, 1) * 255).round()
                    img_lr_batch = img_lr_batch.astype('uint8')
                    count_temp = count
                    for img_lr in img_lr_batch:
                        filename = osp.basename(img_GT_path_l[count])
                        imageio.imwrite(osp.join(save_subfolder, filename), img_lr)
                        count += 1

                    img_lr_batch = img_lr_batch.astype('float32') / 255
                    img_lr_batch = torch.from_numpy(img_lr_batch).permute(0, 3, 1, 2)

                    img_superlr_batch = kernel_gen.apply(img_lr_batch)
                    img_superlr_batch = img_superlr_batch.permute(0, 2, 3, 1).cpu().numpy()
                    img_superlr_batch = (img_superlr_batch.clip(0, 1) * 255).round()
                    '''
                    img_superlr_batch = img_superlr_batch.astype('float32') / 255
                    img_superlr_batch = torch.from_numpy(img_superlr_batch).permute(0, 3, 1, 2)
                    img_superlr_batch = kernel_gen.apply(img_superlr_batch)
                    img_superlr_batch = img_superlr_batch.permute(0, 2, 3, 1).cpu().numpy()
                    img_superlr_batch = (img_superlr_batch.clip(0, 1) * 255).round()
                    img_superlr_batch = img_superlr_batch.astype('float32') / 255
                    img_superlr_batch = torch.from_numpy(img_superlr_batch).permute(0, 3, 1, 2)
                    img_superlr_batch = kernel_gen.apply(img_superlr_batch)
                    img_superlr_batch = img_superlr_batch.permute(0, 2, 3, 1).cpu().numpy()
                    img_superlr_batch = (img_superlr_batch.clip(0, 1) * 255).round()
                    '''
                    img_superlr_batch = img_superlr_batch.astype('uint8')
                    count = count_temp
                    for img_superlr in img_superlr_batch:
                        filename = osp.basename(img_GT_path_l[count])
                        imageio.imwrite(osp.join(save_subfolder2, filename), img_superlr)
                        count += 1


if __name__ == '__main__':
    main()
