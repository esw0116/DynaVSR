import os
from os import path
import glob
import random

from data import common
from data.estimator import preprocessing
from data import random_kernel_generator as rkg

import numpy as np
import imageio
import tqdm
import math

import torch
import torch.utils.data as data

import time

class VSRBase(data.Dataset):
    def __init__(
            self, opt, data_root='../dataset', train=True, benchmark=False):

        super(VSRBase, self).__init__()
        self.apath = ''
        self.opt = opt
        self.opt_data = opt['datasets']
        self.split = 'train' if train else 'val'
        self.ext = '.png'

        self.train = train
        self.benchmark = benchmark
        self._set_directory(data_root)
        self.scale = opt['scale']
        if benchmark:
            self.img_type = 'img'
        else:
            self.img_type = opt['datasets']['train']['img_type']
        self.stride = opt['datasets']['train']['interval_list']

        timesteps = opt['datasets']['train']['N_frames']

        self.dict_hr= self._scan()
        self.keys = sorted(self.dict_hr.keys())

        # Pre-decode png files
        if self.img_type == 'bin':
            for k in tqdm.tqdm(self.dict_hr.keys(), ncols=80):
                bin_path = path.join(self.apath, 'bin')
                for idx, v in enumerate(self.dict_hr[k]):
                    save_as = v.replace(self.apath, bin_path)
                    save_as = save_as.replace(self.ext, '')
                    # If we don't have the binary, make it.
                    if not path.isfile(save_as+'.npy'):
                        os.makedirs(path.dirname(save_as), exist_ok=True)
                        img = imageio.imread(v)
                        # Bypassing the zip archive error
                        # _, w, c = img.shape
                        # dummy = np.zeros((1,w,c))
                        # img_dummy = np.concatenate((img, dummy), axis=0)
                        # torch.save(img_dummy, save_as)
                        np.save(save_as, img)
                    # Update the dictionary
                    self.dict_hr[k][idx] = save_as + '.npy'

        # Calculate all possible sequence combinations
        if not isinstance(self.stride, (tuple, list)):
            self.stride = [self.stride]
        abs_stride = list(map(lambda x: abs(x), self.stride))

        self.is_seq = True
        self.sample_length = [(timesteps - 1) * abs_s + 1 for abs_s in abs_stride]
        self.offsets = [list(range(sample_length)) for sample_length in self.sample_length]
        self.len_dict = {}
        for k, v in self.dict_hr.items():
            self.len_dict[k] = []
            for str_idx, offset in enumerate(self.offsets):
                self.len_dict[k].append([(len(v) - o) // self.sample_length[str_idx] for o in offset])

        # Count the number of samples
        self.len_dict = {}
        for k, v in self.dict_hr.items():
            self.len_dict[k] = []
            for str_idx, offset in enumerate(self.offsets):
                self.len_dict[k].append([(len(v) - o) // self.sample_length[str_idx] for o in offset])
        self.weighted_sample = list(map(lambda x: sum([sum(y) for y in x]), self.len_dict.values()))
        self.n_samples = sum(self.weighted_sample)

    def sampling_weights(self):
        return list(map(lambda x: 1/x, self.weighted_sample))

    def _set_directory(self, data_root):
        raise NotImplementedError

    def _scan(self):
        dir_path = path.join(self.apath, self.split) if not self.benchmark else self.apath
        dir_hr = path.join(dir_path, 'HR')
        list_hr_seq = os.listdir(dir_hr)
        dict_hr = {
            k: sorted(
                glob.glob(path.join(dir_hr, k, '*' + self.ext))
            ) for k in list_hr_seq
        }
        return dict_hr

    # Find the appropriate set for given idx
    def find_set(self, idx):
        for k in self.keys:
            # length of possible subsequences for each stride per each video
            len_list = [sum(b) for b in self.len_dict[k]]
            set_length = sum(len_list)
            if idx < set_length:
                set_key = k
                for str_idx, offset_list in enumerate(self.len_dict[k]):
                    if idx < len_list[str_idx]:
                        sample_length = self.sample_length[str_idx]
                        for len_idx, length in enumerate(offset_list):
                            if idx < length:
                                if self.stride[str_idx] > 0:
                                    seq_idx = len_idx + idx * sample_length
                                else:
                                    seq_idx = len_idx + idx * sample_length + sample_length - 1
                                break
                            else:
                                idx -= length
                        break
                    else:
                        idx -= len_list[str_idx]
                break
            else:
                idx -= set_length

        return set_key, str_idx, seq_idx

    def __getitem__(self, idx):
        # Randomly choose a sequence in a video
        # key = self.keys[idx]
        # str_idx, seq_idx = self.find_set(key)
        key, str_idx, seq_idx = self.find_set(idx)
        set_hr = self.dict_hr[key]

        if self.stride[str_idx] > 0:
            seq_end = seq_idx + self.sample_length[str_idx]
        else:
            seq_end = seq_idx - self.sample_length[str_idx]

        if seq_end >= 0:
            name_hr = set_hr[seq_idx:seq_end:self.stride[str_idx]]
        else:
            name_hr = set_hr[seq_idx::self.stride[str_idx]]

        if self.img_type == 'img':
            fn_read = imageio.imread
        elif self.img_type == 'bin':
            fn_read = np.load
        else:
            raise ValueError('Wrong img type: {}'.format(self.img_type))

        name = [path.join(key, path.basename(f)) for f in name_hr]
        seq_hr = [fn_read(f) for f in name_hr]
        seq_hr = np.stack(seq_hr, axis=-1)
        
        # if self.train:
            # sinc patch_size is decided in seq_LR scale, we have to make it twice larger
            # seq_hr = preprocessing.common_crop(seq_hr, patch_size=self.opt['datasets']['train']['patch_size']*2)
        seq_hr = preprocessing.np2tensor(seq_hr)        
        seq_hr = preprocessing.common_crop(seq_hr, patch_size=self.opt['datasets']['train']['patch_size'] * 3)
        kwargs = preprocessing.set_kernel_params()
        kwargs_l = kwargs['sigma']
        kwargs_l.append(kwargs['theta'])
        kwargs_l = torch.Tensor(kwargs_l)

        base_type = random.random()

        # include random noise for each frame
        kernel_gen = rkg.Degradation(self.opt['datasets']['train']['kernel_size'], self.scale, base_type, **kwargs)
        seq_lr = []
        seq_superlr = []
        for i in range(seq_hr.shape[0]):
            # kernel_gen.gen_new_noise()
            seq_lr_slice = kernel_gen.apply(seq_hr[i])
            seq_lr.append(seq_lr_slice)
            seq_superlr.append(kernel_gen.apply(seq_lr_slice))

        seq_lr = torch.stack(seq_lr, dim=0)
        seq_superlr = torch.stack(seq_superlr, dim=0)
        '''
        if not os.path.exists(os.path.join('../result', self.opt['name'])):
            os.makedirs(os.path.join('../result', self.opt['name']), exist_ok=True)
        imageio.imwrite(os.path.join('../result', self.opt['name'], name[2].split('/')[0]+'_'+name[2].split('/')[1]+'_before.png'), seq_lr[2].numpy().transpose(1,2,0))
        imageio.imwrite(os.path.join('../result', self.opt['name'], name[2].split('/')[0]+'_'+name[2].split('/')[1]+'_after.png'), seq_hr[2].numpy().transpose(1,2,0))

        if self.train:
            seq_hr, seq_lr = preprocessing.crop(seq_hr, seq_lr, patch_size=self.opt_data['train']['patch_size'])
            # seq_hr, seq_lr = preprocessing.augment(seq_hr, seq_lr)
        # imageio.imwrite(os.path.join('../result', self.opt['name'], name[2].split('/')[0]+'_'+name[2].split('/')[1]+'_after.png'), seq_lr[2].numpy().transpose(1,2,0))
        '''

        if self.train:
            seq_hr, seq_lr, seq_superlr = preprocessing.crop(seq_hr, seq_lr, seq_superlr, patch_size=self.opt['datasets']['train']['patch_size'])
            seq_hr, seq_lr, seq_superlr = preprocessing.augment(seq_hr, seq_lr, seq_superlr)

        return {'SuperLQs': seq_superlr, 'LQs': seq_lr, 'GT': seq_hr, 'Kernel': kernel_gen.kernel, 'Kernel_args': kwargs_l, 'name': name}

    def __len__(self):
        return self.n_samples
