import os.path as osp
import torch
import torch.utils.data as data
from data import util
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
            degradation_type = opt['degradation_type']
            opt_sigma_x = opt['sigma_x']
            opt_sigma_y = opt['sigma_y']
            opt_theta = opt['theta']
        else:
            self.name = opt['name'].split('+')[idx]
            self.GT_root, self.LQ_root = opt['dataroot_GT'].split('+')[idx], opt['dataroot_LQ'].split('+')[idx]
            if '+' in opt['degradation_type']:
                degradation_type = opt['degradation_type'].split('+')[idx]
                if '+' in str(opt['sigma_x']):
                    opt_sigma_x = float(opt['sigma_x'].split('+')[idx])
                    opt_sigma_y = float(opt['sigma_y'].split('+')[idx])
                    opt_theta = float(opt['theta'].split('+')[idx])
                    
                else:
                    opt_sigma_x = opt['sigma_x']
                    opt_sigma_y = opt['sigma_y']
                    opt_theta = opt['theta']
                
            else:
                degradation_type = opt['degradation_type']
                opt_sigma_x = opt['sigma_x']
                opt_sigma_y = opt['sigma_y']
                opt_theta = opt['theta']

        self.data_type = self.opt['data_type']
        self.data_info = {'path_SLQ': [], 'path_LQ': [], 'path_GT': [], 'folder': [], 'idx': [], 'border': []}
        if self.data_type == 'lmdb':
            raise ValueError('No need to use LMDB during validation/test.')
        #### Generate data info and cache data
        self.imgs_SLQ, self.imgs_LQ, self.imgs_GT = {}, {}, {}

        if opt['degradation_mode'] == 'preset':
            self.LQ_root = self.LQ_root + '_preset'
        else:
            if isinstance(opt_sigma_x, list):
                assert len(opt_sigma_x) == len(opt_sigma_y)
                assert len(opt_sigma_x) == len(opt_theta)

                LQ_root_list = []
                for i, (sigma_x, sigma_y, theta) in enumerate(zip(opt_sigma_x, opt_sigma_y, opt_theta)):
                    LQ_root_list.append(self.LQ_root + '_' + degradation_type + '_' + str('{:.1f}'.format(opt_sigma_x[i]))\
                           + '_' + str('{:.1f}'.format(opt_sigma_y[i])) + '_' + str('{:.1f}'.format(opt_theta[i])))
                self.LQ_root = LQ_root_list

            else:
                self.LQ_root = self.LQ_root + '_' + degradation_type + '_' + str('{:.1f}'.format(opt_sigma_x))\
                           + '_' + str('{:.1f}'.format(opt_sigma_y)) + '_' + str('{:.1f}'.format(opt_theta))
        
        slr_name = '' if opt['slr_mode'] is None else '_{}'.format(opt['slr_mode'])
        
        print(self.LQ_root)

        if self.name.lower() in ['vid4', 'reds', 'mm522']:
            if self.name.lower() == 'vid4':
                img_type = 'img'
                subfolders_GT = util.glob_file_list(self.GT_root)
                if isinstance(self.LQ_root, list):
                    num_settings = len(self.LQ_root)
                    subfolders_LQ_list = [util.glob_file_list(osp.join(LQ_root, 'X{}'.format(self.scale))) for LQ_root in self.LQ_root]
                    subfolders_SLQ_list = [util.glob_file_list(osp.join(LQ_root, 'X{}{}'.format(self.scale*self.scale, slr_name))) for LQ_root in self.LQ_root]

                    subfolders_LQ = []
                    subfolders_SLQ = []
                    for i in range(len(subfolders_LQ_list[0])):
                        subfolders_LQ.append([subfolders_LQ_list[j][i] for j in range(len(subfolders_LQ_list))])
                        subfolders_SLQ.append([subfolders_SLQ_list[j][i] for j in range(len(subfolders_SLQ_list))])

                else:
                    subfolders_LQ = util.glob_file_list(osp.join(self.LQ_root, 'X{}'.format(self.scale)))
                    subfolders_SLQ = util.glob_file_list(osp.join(self.LQ_root, 'X{}{}'.format(self.scale*self.scale, slr_name)))

            elif self.name.lower() == 'reds':
                img_type = 'img'
                list_hr_seq = util.glob_file_list(self.GT_root)
                subfolders_GT = [k for k in list_hr_seq if k.find('000') >= 0 or k.find('011') >= 0 or k.find('015') >= 0 or k.find('020') >= 0]
                if isinstance(self.LQ_root, list):
                    num_settings = len(self.LQ_root)
                    subfolders_LQ_list = []
                    subfolders_SLQ_list = []

                    for i in range(num_settings):
                        list_lr_seq = util.glob_file_list(osp.join(self.LQ_root[i], 'X{}'.format(self.scale)))
                        list_slr_seq = util.glob_file_list(osp.join(self.LQ_root[i], 'X{}{}'.format(self.scale*self.scale, slr_name)))
                        subfolder_LQ = [k for k in list_lr_seq if k.find('000') >= 0 or k.find('011') >= 0 or k.find('015') >= 0 or k.find('020') >= 0]
                        subfolder_SLQ = [k for k in list_slr_seq if k.find('000') >= 0 or k.find('011') >= 0 or k.find('015') >= 0 or k.find('020') >= 0]
                        subfolders_LQ_list.append(subfolder_LQ)
                        subfolders_SLQ_list.append(subfolder_SLQ)
                    subfolders_LQ = []
                    subfolders_SLQ = []
                    for i in range(len(subfolders_LQ_list[0])):
                        subfolders_LQ.append([subfolders_LQ_list[j][i] for j in range(len(subfolders_LQ_list))])
                        subfolders_SLQ.append([subfolders_SLQ_list[j][i] for j in range(len(subfolders_SLQ_list))])

                else:
                    list_lr_seq = util.glob_file_list(osp.join(self.LQ_root, 'X{}'.format(self.scale)))
                    list_slr_seq = util.glob_file_list(osp.join(self.LQ_root, 'X{}{}'.format(self.scale*self.scale, slr_name)))
                    #subfolders_GT = [k for k in list_hr_seq if
                    #                   k.find('000') >= 0 or k.find('011') >= 0 or k.find('015') >= 0 or k.find('020') >= 0]
                    subfolders_LQ = [k for k in list_lr_seq if
                                    k.find('000') >= 0 or k.find('011') >= 0 or k.find('015') >= 0 or k.find('020') >= 0]
                    subfolders_SLQ = [k for k in list_slr_seq if
                                    k.find('000') >= 0 or k.find('011') >= 0 or k.find('015') >= 0 or k.find('020') >= 0]

            else:
                img_type = 'img'
                list_hr_seq = util.glob_file_list(self.GT_root)
                list_lr_seq = util.glob_file_list(osp.join(self.LQ_root, 'X{}'.format(self.scale)))
                list_slr_seq = util.glob_file_list(osp.join(self.LQ_root, 'X{}'.format(self.scale*self.scale)))
                subfolders_GT = [k for k in list_hr_seq if
                                   k.find('001') >= 0 or k.find('005') >= 0 or k.find('008') >= 0 or k.find('009') >= 0]
                subfolders_LQ = [k for k in list_lr_seq if
                                   k.find('001') >= 0 or k.find('005') >= 0 or k.find('008') >= 0 or k.find('009') >= 0]
                subfolders_SLQ = [k for k in list_slr_seq if
                                   k.find('001') >= 0 or k.find('005') >= 0 or k.find('008') >= 0 or k.find('009') >= 0]

            print(subfolders_GT[0], '\n', subfolders_LQ[0], '\n', subfolders_SLQ[0])

            for subfolder_SLQ, subfolder_LQ, subfolder_GT in zip(subfolders_SLQ, subfolders_LQ, subfolders_GT):
                subfolder_name = osp.basename(subfolder_GT)
                img_paths_GT = util.glob_file_list(subfolder_GT)
                if isinstance(subfolder_LQ, list):
                    img_paths_LQ_list = [util.glob_file_list(subf_LQ) for subf_LQ in subfolder_LQ]
                    img_paths_SLQ_list = [util.glob_file_list(subf_SLQ) for subf_SLQ in subfolder_SLQ]
                    img_paths_LQ = []
                    img_paths_SLQ = []
                    for i in range(len(img_paths_GT)):
                        img_paths_LQ.append(img_paths_LQ_list[i % num_settings][i])
                        img_paths_SLQ.append(img_paths_SLQ_list[i % num_settings][i])
                else:
                    img_paths_LQ = util.glob_file_list(subfolder_LQ)
                    img_paths_SLQ = util.glob_file_list(subfolder_SLQ)

                max_idx = len(img_paths_GT)
                self.data_info['path_SLQ'].extend(img_paths_SLQ)
                self.data_info['path_LQ'].extend(img_paths_LQ)
                self.data_info['path_GT'].extend(img_paths_GT)
                self.data_info['folder'].extend([subfolder_name] * max_idx)
                for i in range(max_idx):
                    self.data_info['idx'].append('{}/{}'.format(i, max_idx))
                border_l = [0] * max_idx
                for i in range(self.half_N_frames):
                    border_l[i] = 1
                    border_l[max_idx - i - 1] = 1
                self.data_info['border'].extend(border_l)
                self.imgs_GT[subfolder_name] = util.read_img_seq(img_paths_GT, img_type)
                if opt['degradation_mode'] == 'preset':
                    self.imgs_LQ[subfolder_name] = torch.stack([util.read_img_seq(util.glob_file_list(paths_LQ), img_type) for paths_LQ in img_paths_LQ], dim=0)
                    self.imgs_SLQ[subfolder_name] = torch.stack([util.read_img_seq(util.glob_file_list(paths_SLQ), img_type) for paths_SLQ in img_paths_SLQ], dim=0)
                else:
                    self.imgs_LQ[subfolder_name] = util.read_img_seq(img_paths_LQ, img_type)
                    self.imgs_SLQ[subfolder_name] = util.read_img_seq(img_paths_SLQ, img_type)
                h, w = self.imgs_SLQ[subfolder_name].shape[-2:]
                if h % 4 != 0 or w % 4 != 0:
                    self.imgs_SLQ[subfolder_name] = self.imgs_SLQ[subfolder_name][..., :h - (h%4), :w - (w%4)]
                    self.imgs_LQ[subfolder_name] = self.imgs_LQ[subfolder_name][..., :self.scale*(h - (h%4)), :self.scale*(w - (w%4))]
                    self.imgs_GT[subfolder_name] = self.imgs_GT[subfolder_name][..., :self.scale*self.scale*(h - (h%4)), :self.scale*self.scale*(w - (w%4))]

        else:
            raise ValueError(
                'Not support video test dataset. Support Vid4, REDS4 and Vimeo90k-Test.')
        '''
        if opt['degradation_mode'] == 'set':
            sigma_x = float(opt['sigma_x'])
            sigma_y = float(opt['sigma_y'])
            theta = float(opt['theta'])
            gen_kwargs = preprocessing.set_kernel_params(sigma_x=sigma_x, sigma_y=sigma_y, theta=theta)
            self.kernel_gen = rkg.Degradation(self.kernel_size, self.scale, **gen_kwargs)
            self.gen_kwargs_l = [gen_kwargs['sigma'][0], gen_kwargs['sigma'][1], gen_kwargs['theta']]
        '''
        if opt['degradation_mode'] == 'preset':
            self.kernel_gen = rkg.Degradation(self.kernel_size, self.scale)
            if self.name.lower() == 'vid4':
                self.kernel_dict = np.load('../pretrained_models/Mixed/Vid4.npy')
            elif self.name.lower() == 'reds':
                self.kernel_dict = np.load('../pretrained_models/Mixed/REDS.npy')
            else:
                raise NotImplementedError()
    
    def __getitem__(self, index):
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info['border'][index]
        # print(self.data_info['path_LQ'][index], '\n', self.data_info['path_SLQ'][index])

        select_idx = util.index_generation(idx, max_idx, self.opt['N_frames'], padding=self.opt['padding'])
        imgs_GT = self.imgs_GT[folder].index_select(0, torch.LongTensor(select_idx))
        if self.opt['degradation_mode'] == 'preset':
            if self.opt['N_frames'] == 5:
                imgs_LR = self.imgs_LQ[folder][idx, 1:-1]
                imgs_SuperLR = self.imgs_SLQ[folder][idx, 1:-1]
            else:
                imgs_LR = self.imgs_LQ[folder][idx]
                imgs_SuperLR = self.imgs_SLQ[folder][idx]
        else:
            imgs_LR = self.imgs_LQ[folder].index_select(0, torch.LongTensor(select_idx))
            imgs_SuperLR = self.imgs_SLQ[folder].index_select(0, torch.LongTensor(select_idx))

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
