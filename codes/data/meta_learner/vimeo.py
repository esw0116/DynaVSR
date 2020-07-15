import os
from os import path
import glob
import random

from data import common
from data.meta_learner import preprocessing
from data import random_kernel_generator as rkg

import numpy as np
import math, imageio

import torch
import torch.utils.data as data


class Vimeo(data.Dataset):
    """Vimeo trainset class
    """
    def __init__(self, opt, train=True, **kwargs):
        super().__init__()
        self.data_root = opt['datasets']['train']['data_root']
        self.data_root = path.join(self.data_root, 'vimeo_septuplet')
        if train:
            meta = path.join(self.data_root, 'sep_trainlist.txt')
        else:
            meta = path.join(self.data_root, 'sep_testlist.txt')

        with open(meta, 'r') as f:
            self.img_list = sorted(f.read().splitlines())
        self.opt = opt
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
        start_frame = random.randint(0, 7-self.nframes)
        seq_hr = seq_hr[..., start_frame:start_frame+self.nframes]
        seq_hr = preprocessing.np2tensor(seq_hr)

        # seq_hr = preprocessing.crop_border(seq_hr, border=[4, 4])
        # To make time efficient crop by seq_hr and make it downsample to seq_lr
        # if self.train:
            # sinc patch_size is decided in seq_LR scale, we have to make it twice larger
            # seq_hr = preprocessing.common_crop(img=seq_hr, patch_size=self.opt['datasets']['train']['patch_size']*2)

        # include random noise for each frame
        '''
        kernel_set = []
        for i in range(5):
            kwargs = preprocessing.set_kernel_params()
            kernel_set.append(rkg.Degradation(self.opt['datasets']['train']['kernel_size'], self.scale, **kwargs).kernel)
        kernel_set = np.stack(kernel_set, axis=0)
        
        kernel_temp = rkg.Degradation(self.opt['datasets']['train']['kernel_size'], self.scale)
        kernel_temp.set_kernel_directly(kernel_set)

        seq_lr = kernel_temp.apply(seq_hr)
        seq_lr = seq_lr.mul(255).clamp(0, 255).round().div(255)
        kernel_temp.set_kernel_directly(kernel_set[2])
        seq_superlr = kernel_temp.apply(seq_lr)
        '''
        kwargs = preprocessing.set_kernel_params()
        kernel_gen = rkg.Degradation(self.opt['datasets']['train']['kernel_size'], self.scale, **kwargs)

        seq_lr = kernel_gen.apply(seq_hr)
        seq_lr = seq_lr.mul(255).clamp(0, 255).round().div(255)
        seq_superlr = kernel_gen.apply(seq_lr)

        if self.train:
            # seq_hr, seq_lr, seq_superlr = preprocessing.crop(seq_hr, seq_lr, seq_superlr, patch_size=self.opt['datasets']['train']['patch_size'])
            seq_hr, seq_lr, seq_superlr = preprocessing.augment(seq_hr, seq_lr, seq_superlr)

        return {'SuperLQs': seq_superlr,
                'LQs': seq_lr,
                'GT': seq_hr,
                # 'Kernel': kernel_gen.kernel
                }

    def __len__(self):
        return len(self.img_list)
