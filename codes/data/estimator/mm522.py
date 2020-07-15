import glob
import os
from os import path
import imageio
import numpy as np
import torch
from data import common
from data.estimator import vsrbase
from data.estimator import preprocessing
from data import random_kernel_generator as rkg


class MM522(vsrbase.VSRBase):
    """MM522 train OR test subset class
    """
    def __init__(self, *args, **kwargs):
        self.set = 'MM522'
        super(MM522, self).__init__(*args, **kwargs)

    def _set_directory(self, data_root):
        self.apath = path.join(data_root, 'MM522')

    def _scan(self):
        def _make_keys(dir_path):
            """
            :param dir_path: 'frame_lr/X4/001/0001'
            :return: 001_0001
            """
            return os.path.basename(os.path.dirname(dir_path))

        if self.split == 'train':
            dir_train = path.join(self.apath, 'train')
            list_hr_seq = sorted(glob.glob(dir_train + '/**/*/truth'))
        else:
            dir_train = path.join(self.apath, 'val')
            list_hr_seq = sorted(glob.glob(dir_train + '/**/truth'))

        dict_hr = {
            _make_keys(k): sorted(
                glob.glob(path.join(k, '*' + self.ext))
            ) for k in list_hr_seq
        }

        return dict_hr

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
        name = [path.splitext(f)[0] for f in name]
        seq_hr = [fn_read(f) for f in name_hr]
        seq_hr = np.stack(seq_hr, axis=-1)
        seq_hr = preprocessing.np2tensor(seq_hr)

        if self.opt['network_E']['which_model_E'] == 'gaussargs':
            kwargs = preprocessing.set_kernel_params(base='bicubic')
        else:
            kwargs = preprocessing.set_kernel_params()

        kwargs_l = kwargs['sigma']
        kwargs_l.append(kwargs['theta'])
        kwargs_l = torch.Tensor(kwargs_l)
        basis_label = int(5 * kwargs['type'])

        kernel_gen = rkg.Degradation(self.opt_data['train']['kernel_size'], self.scale, **kwargs)
        seq_lr = kernel_gen.apply(seq_hr)

        if self.train:
            seq_hr, seq_lr = preprocessing.crop(seq_hr, seq_lr, patch_size=self.opt_data['train']['patch_size'])
            # seq_hr, seq_lr = preprocessing.augment(seq_hr, seq_lr)

        return {'LQs': seq_lr, 'GT': seq_hr, 'Kernel': kernel_gen.kernel,
                'Kernel_type': basis_label, 'Kernel_args': kwargs_l, 'name': name}
