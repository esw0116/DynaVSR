import os.path as osp
import numpy as np
import math

import torch
import torch.utils.data as data

from data.meta_learner import preprocessing
from data import util


class Demo(data.Dataset):
    """
    A video test dataset. Support:
    Vid4
    REDS4
    Vimeo90K-Test

    no need to prepare LMDB files
    """

    def __init__(self, opt, **kwargs):
        super(Demo, self).__init__()
        self.scale = kwargs['scale']
        self.kernel_size = kwargs['kernel_size']
        self.model_name = kwargs['model_name']
        idx = kwargs['idx'] if 'idx' in kwargs else None
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.half_N_frames = opt['N_frames'] // 2
        if idx is None:
            self.name = opt['name']
            self.root = opt['dataroot']
        else:
            self.name = opt['name'].split('+')[idx]
            self.root = opt['dataroot'].split('+')[idx]

        self.data_type = self.opt['data_type']
        self.data_info = {'path': [], 'folder': [], 'idx': [], 'border': []}
        if self.data_type == 'lmdb':
            raise ValueError('No need to use LMDB during validation/test.')
        #### Generate data info and cache data
        self.imgs = {}

        subfolder_name = osp.basename(self.root)

        self.data_info['path'] = util.glob_file_list(self.root)
        max_idx = len(self.data_info['path'])
        self.data_info['folder'] = [subfolder_name] * max_idx

        for i in range(max_idx):
            self.data_info['idx'].append('{}/{}'.format(i, max_idx))
        border_l = [0] * max_idx
        for i in range(self.half_N_frames):
            border_l[i] = 1
            border_l[max_idx - i - 1] = 1
        self.data_info['border'].extend(border_l)
        self.imgs[subfolder_name] = util.read_img_seq(self.data_info['path'], 'img')

    def __getitem__(self, index):
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info['border'][index]

        select_idx = util.index_generation(idx, max_idx, self.opt['N_frames'], padding=self.opt['padding'])
        imgs = self.imgs[folder].index_select(0, torch.LongTensor(select_idx))

        return {
            'LQs': imgs,
            'folder': folder,
            'idx': self.data_info['idx'][index],
            'border': border
        }

    def __len__(self):
        return len(self.data_info['path'])
