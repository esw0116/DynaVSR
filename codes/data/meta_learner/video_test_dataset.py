import os.path as osp
import torch
import torch.utils.data as data
from data.Backup import util
import numpy as np
import math
from data import random_kernel_generator as rkg
from data.meta_learner import preprocessing


class VideoTestDataset(data.Dataset):
    """
    A video test dataset. Support:
    Vid4
    REDS4
    Vimeo90K-Test

    no need to prepare LMDB files
    """

    def __init__(self, opt, **kwargs):
        super(VideoTestDataset, self).__init__()
        self.scale = kwargs['scale']
        self.kernel_size = kwargs['kernel_size']
        self.model_name = kwargs['model_name']
        idx = kwargs['idx'] if 'idx' in kwargs else None
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.half_N_frames = opt['N_frames'] // 2
        if idx is None:
            self.name = opt['name']
            self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
        else:
            self.name = opt['name'].split('+')[idx]
            self.GT_root, self.LQ_root = opt['dataroot_GT'].split('+')[idx], opt['dataroot_LQ'].split('+')[idx]

        self.data_type = self.opt['data_type']
        self.data_info = {'path_LQ': [], 'path_GT': [], 'folder': [], 'idx': [], 'border': []}
        if self.data_type == 'lmdb':
            raise ValueError('No need to use LMDB during validation/test.')
        #### Generate data info and cache data
        self.imgs_LQ, self.imgs_GT = {}, {}
        if self.name.lower() in ['vid4', 'reds', 'mm522']:
            if self.name.lower() == 'vid4':
                img_type = 'img'
                subfolders_GT = util.glob_file_list(self.GT_root)
            elif self.name.lower() == 'reds':
                img_type = 'img'
                list_hr_seq = util.glob_file_list(self.GT_root)
                subfolders_GT = [k for k in list_hr_seq if
                                   k.find('000') >= 0 or k.find('011') >= 0 or k.find('015') >= 0 or k.find('020') >= 0]
            else:
                img_type = 'img'
                subfolders_GT = util.glob_file_list(self.GT_root)

            for subfolder_GT in subfolders_GT:
                subfolder_name = osp.basename(subfolder_GT)
                img_paths_GT = util.glob_file_list(subfolder_GT)
                max_idx = len(img_paths_GT)

                self.data_info['path_GT'].extend(img_paths_GT)
                self.data_info['folder'].extend([subfolder_name] * max_idx)
                for i in range(max_idx):
                    self.data_info['idx'].append('{}/{}'.format(i, max_idx))
                border_l = [0] * max_idx
                for i in range(self.half_N_frames):
                    border_l[i] = 1
                    border_l[max_idx - i - 1] = 1
                self.data_info['border'].extend(border_l)

                if self.cache_data:
                    self.imgs_GT[subfolder_name] = util.read_img_seq(img_paths_GT, img_type)
        elif opt['name'].lower() in ['vimeo90k-test']:
            pass  # TODO
        else:
            raise ValueError(
                'Not support video test dataset. Support Vid4, REDS4 and Vimeo90k-Test.')

                # Generate kernel

        if opt['degradation_mode'] == 'set':
            sigma_x = float(opt['sigma_x'])
            sigma_y = float(opt['sigma_y'])
            theta = float(opt['theta'])
            gen_kwargs = preprocessing.set_kernel_params(sigma_x=sigma_x, sigma_y=sigma_y, theta=theta)
            self.kernel_gen = rkg.Degradation(self.kernel_size, self.scale, **gen_kwargs)
            self.gen_kwargs_l = [gen_kwargs['sigma'][0], gen_kwargs['sigma'][1], gen_kwargs['theta']]

        elif opt['degradation_mode'] == 'preset':
            self.kernel_gen = rkg.Degradation(self.kernel_size, self.scale)
            if self.name.lower() == 'vid4':
                self.kernel_dict = np.load('../experiments/pretrained_models/Vid4Gauss.npy')
            elif self.name.lower() == 'reds':
                self.kernel_dict = np.load('../experiments/pretrained_models/REDSGauss.npy')
            else:
                raise NotImplementedError()

    def __getitem__(self, index):
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info['border'][index]

        select_idx = util.index_generation(idx, max_idx, self.opt['N_frames'],
                                           padding=self.opt['padding'])
        imgs_GT = self.imgs_GT[folder].index_select(0, torch.LongTensor(select_idx))

        if self.opt['degradation_mode'] == 'set':
            '''
            for i in range(imgs_GT.shape[0]):
                imgs_LR_slice = self.kernel_gen.apply(imgs_GT[i])
                imgs_LR.append(imgs_LR_slice)
                imgs_SuperLR.append(self.kernel_gen.apply(imgs_LR_slice))

            imgs_LR = torch.stack(imgs_LR, dim=0)
            imgs_SuperLR = torch.stack(imgs_SuperLR, dim=0)
            '''
            imgs_LR = self.kernel_gen.apply(imgs_GT)
            imgs_LR = imgs_LR.mul(255).clamp(0, 255).round().div(255)
            imgs_SuperLR = self.kernel_gen.apply(imgs_LR)

        elif self.opt['degradation_mode'] == 'preset':
            my_kernel = self.kernel_dict[index]
            self.kernel_gen.set_kernel_directly(my_kernel)
            imgs_LR = self.kernel_gen.apply(imgs_GT)
            imgs_LR = imgs_LR.mul(255).clamp(0, 255).round().div(255)
            imgs_SuperLR = self.kernel_gen.apply(imgs_LR)

        else:
            kwargs = preprocessing.set_kernel_params()
            kernel_gen = rkg.Degradation(self.kernel_size, self.scale, **kwargs)
            '''
            for i in range(imgs_GT.shape[0]):
                kernel_gen.gen_new_noise()
                imgs_LR_slice = kernel_gen.apply(imgs_GT[i])
                imgs_LR.append(imgs_LR_slice)
                imgs_SuperLR.append(kernel_gen.apply(imgs_LR_slice))
            imgs_LR = torch.stack(imgs_LR, dim=0)
            imgs_SuperLR = torch.stack(imgs_SuperLR, dim=0)
            '''
            imgs_LR = kernel_gen.apply(imgs_GT)
            imgs_SuperLR = kernel_gen.apply(imgs_LR)

        return {
            'SuperLQs': imgs_SuperLR,
            'LQs': imgs_LR,
            'GT': imgs_GT,
            'folder': folder,
            'idx': self.data_info['idx'][index],
            'border': border
        }

    def __len__(self):
        return len(self.data_info['path_GT'])
