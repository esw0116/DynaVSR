import os
from os import path
import glob
import random

from data import common
from data.estimator import preprocessing
from data import random_kernel_generator as rkg

import numpy as np
import math, imageio

import torch
import torch.utils.data as data


class VimeoTest(data.Dataset):
    """Vimeo testset class
    """
    def __init__(self, opt, train=True, **kwargs):
        super().__init__()
        self.data_root = opt['datasets']['train']['data_root']
        self.data_root = path.join(self.data_root, 'vimeo_septuplet')
        meta = path.join(self.data_root, 'sep_testlist.txt')
        self.opt = opt

        with open(meta, 'r') as f:
            self.img_list = sorted(f.read().splitlines())
        self.scale = opt['scale']
        self.nframes = opt['datasets']['train']['N_frames']
        self.train = train

    def __getitem__(self, idx):
        name_hr = path.join(self.data_root, 'sequences', self.img_list[idx])
        names_hr = sorted(glob.glob(path.join(name_hr, '*.png')))
        names = [path.splitext(path.basename(f))[0] for f in names_hr]
        names = [path.join(self.img_list[idx], name) for name in names]
        
        seq_hr = [imageio.imread(f) for f in names_hr]
        seq_hr = np.stack(seq_hr, axis=-1)
        start_frame = (7-self.nframes)//2
        seq_hr = seq_hr[..., start_frame:start_frame+self.nframes]

        if self.train:
            seq_hr = preprocessing.crop_border(seq_hr, border=[4,4])
            seq_hr = preprocessing.crop(seq_hr, patch_size=self.opt['datasets']['train']['patch_size'])
            seq_hr = preprocessing.augment(seq_hr, rot=False)

        seq_hr = preprocessing.np2tensor(seq_hr)

        if self.opt['network_E']['which_model_E'] == 'gaussargs':
            kwargs = preprocessing.set_kernel_params(base='bicubic')
        else:
            kwargs = preprocessing.set_kernel_params()

        kernel_gen = rkg.Degradation(self.opt['datasets']['train']['kernel_size'], self.scale, **kwargs)
        seq_lr = kernel_gen.apply(seq_hr)

        return {'LQs': seq_lr, 'GT': seq_hr, 'Kernel': kernel_gen.kernel, 'Kernel_args': kwargs, 'name': names}

    def __len__(self):
        return len(self.img_list)
