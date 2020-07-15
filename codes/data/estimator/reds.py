import glob
import os
from os import path
import random
from collections import OrderedDict

from data import common
from data.estimator import vsrbase

import numpy as np
import scipy.misc as misc
import imageio

import torch
import torch.utils.data as data


class REDS(vsrbase.VSRBase):
    """GOPRO_Large train OR test subset class
    """
    def __init__(self, *args, **kwargs):
        super(REDS, self).__init__(*args, **kwargs)

    def _set_directory(self, data_root):
        self.apath = path.join(data_root, 'REDS')

    def _scan(self):
        """
        Follow the train/valid split of REDS in EDVR paper.
        Trainset : Train(except 000, 011, 015, 020) + Valid (Total 266 videos)
        Validset : Train000, 011, 015, 020 (Total 4 videos)
        :return:
        """

        def _make_keys(dir_path):
            """
            :param dir_path: Path
            :return: train_000 form
            """
            dir, base = path.dirname(dir_path), path.basename(dir_path)
            tv = 'train' if dir.find('train')>=0 else 'val'
            return tv + '_' + base


        dir_path_train = path.join(self.apath, 'train')
        dir_hr_train = path.join(dir_path_train, 'HR')
        list_hr_seq = glob.glob(dir_hr_train+'/*')
        list_hr_seq_val = [k for k in list_hr_seq if k.find('000') >= 0 or k.find('011') >= 0 or k.find('015') >= 0 or k.find('020') >= 0]

        for x in list_hr_seq_val:
            list_hr_seq.remove(x)

        dir_path_valid = path.join(self.apath, 'val')
        dir_hr_valid = path.join(dir_path_valid, 'HR')
        list_hr_seq.extend(glob.glob(dir_hr_valid+'/*'))

        if self.split == 'train':
            dict_hr = {
                _make_keys(k): sorted(
                    glob.glob(path.join(k, '*' + self.ext))
                ) for k in list_hr_seq
            }
            return dict_hr

        else:
            dict_hr = {
                _make_keys(k): sorted(
                    glob.glob(path.join(k, '*' + self.ext))
                ) for k in list_hr_seq_val
            }
            return dict_hr
