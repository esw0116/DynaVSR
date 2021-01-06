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
import data.util as data_util
import utils.util as util

from models.archs import LRimg_estimator as LRest

import imageio


def main():
    #################
    # configurations
    #################
    device = torch.device('cuda')

    prog = argparse.ArgumentParser()
    prog.add_argument('--dataset_mode', '-m', type=str, default='Vid4+REDS', help='data_mode')
    prog.add_argument('--degradation_mode', '-d', type=str, default='bicubic', choices=('impulse', 'bicubic', 'preset'), help='path to image output directory.')
    prog.add_argument('--model', type=str, default='SFDN', choices=('SFDN', 'MFDN'))
    prog.add_argument('--sigma_x', '-sx', type=float, default=1, help='sigma_x')
    prog.add_argument('--sigma_y', '-sy', type=float, default=0, help='sigma_y')
    prog.add_argument('--theta', '-t', type=float, default=0, help='theta')
    prog.add_argument('--scale', '-sc', type=int, default=2, choices=(2, 4), help='scale factor')

    args = prog.parse_args()

    data_modes = args.dataset_mode
    degradation_mode = args.degradation_mode  # impulse | bicubic
    sig_x, sig_y, the = args.sigma_x, args.sigma_y, args.theta * math.pi / 180
    if sig_y == 0:
        sig_y = sig_x

    scale = args.scale
    kernel_size = 21
    N_in = 5

    data_mode_l = data_modes.split('+')

    for i in range(len(data_mode_l)):
        data_mode = data_mode_l[i]

        #### dataset
        if data_mode == 'Vid4':
            load_model = 'Vimeo'
            kernel_folder = '../experiments/pretrained_models/Vid4Gauss.npy'
            dataset_folder = '../dataset/Vid4'
        elif data_mode == 'REDS':
            load_model = 'REDS'
            kernel_folder = '../experiments/pretrained_models/REDSGauss.npy'
            dataset_folder = '../dataset/REDS/train'
        else:
            raise NotImplementedError('Only Vid4 and REDS are supporteds')
        
        root_folder = osp.join(dataset_folder, 'LR_'+ degradation_mode + '_' + str('{:.1f}'.format(sig_x)) + '_' + str('{:.1f}'.format(sig_y)) + '_' + str('{:.1f}'.format(args.theta)))
        LR_dataset_folder = osp.join(root_folder, 'X'+str(scale))
        save_folder = osp.join(root_folder, 'X'+str(scale*scale)+'_'+args.model)
        print(root_folder)
        print(save_folder)

        if not osp.exists(save_folder):
            os.makedirs(save_folder)

        subfolder_name_l = []
        subfolder_LR_l = sorted(glob.glob(osp.join(LR_dataset_folder, '*')))
        if data_mode == 'REDS':
            subfolder_LR_l = [k for k in subfolder_LR_l if
                              k.find('000') >= 0 or k.find('011') >= 0 or k.find('015') >= 0 or k.find('020') >= 0]
        # for each subfolder
        # for subfolder, subfolder_GT in zip(subfolder_l, subfolder_GT_l):
        sig_x, sig_y, the = float(sig_x), float(sig_y), float(the)

        if args.model == 'SFDN':
            if scale == 2:
                model = LRest.DirectKernelEstimator_CMS(nf=64)
                model.load_state_dict(torch.load('../pretrained_models/MFDN/SFDN_{}.pth'.format(load_model)), strict=True)
            else:
                raise NotImplementedError('We do not support SFDN for scale factor 4 now.')

        else:
            model = LRest.DirectKernelEstimatorVideo(in_nc=3, nf=64, scale=scale)
            if scale == 2:
                model.load_state_dict(torch.load('../pretrained_models/MFDN/MFDN_{}.pth'.format(load_model)), strict=True)
            else:
                model.load_state_dict(torch.load('../pretrained_models/MFDN/MFDN_{}_S4.pth'.format(load_model)), strict=True)

        model.eval()
        model = model.to(device)

        for subfolder_LR in subfolder_LR_l:

            subfolder_name = osp.basename(subfolder_LR)
            subfolder_name_l.append(subfolder_name)
            save_subfolder = osp.join(save_folder, subfolder_name)
            if not osp.exists(save_subfolder):
                os.mkdir(save_subfolder)

            img_LR_path_l = sorted(glob.glob(osp.join(subfolder_LR, '*')))
            max_idx = len(img_LR_path_l)

            #### read LQ and GT images
            imgs_LR = data_util.read_img_seq(subfolder_LR)  # T C H W
            if args.model == 'SFDN':
                count = 0
                imgs_LR_l = imgs_LR.split(32)
                for img_batch in imgs_LR_l:
                    img_batch = img_batch.to(device)
                    with torch.no_grad():
                        img_lr_batch = model(img_batch)
                        img_lr_batch = img_lr_batch.permute(0,2,3,1).cpu().numpy()
                        img_lr_batch = (img_lr_batch.clip(0, 1)*255).round()
                        img_lr_batch = img_lr_batch.astype('uint8')
                        for img_lr in img_lr_batch:
                            filename = osp.basename(img_LR_path_l[count])
                            imageio.imwrite(osp.join(save_subfolder, filename), img_lr)
                            count += 1
            
            else:
                 for img_idx, img_path in enumerate(img_LR_path_l):
                    img_name = osp.splitext(osp.basename(img_path))[0]
                    select_idx = data_util.index_generation(img_idx, max_idx, N_in, padding='new_info')
                    imgs_in = imgs_LR.index_select(0, torch.LongTensor(select_idx)).unsqueeze(0).to(device)
                    imgs_in = imgs_in.transpose(1,2)
                    with torch.no_grad():
                        output = model(imgs_in)  # B C T H W
                        output = output.squeeze(0)
                        output = output[:, N_in // 2]
                        output = output.permute(1,2,0).cpu().numpy()
                        output = (output.clip(0, 1)*255).round().astype('uint8')
                        imageio.imwrite(osp.join(save_subfolder, '{}.png'.format(img_name)), output)


if __name__ == '__main__':
    main()
