from importlib import import_module
import collections

import torch
from torch.utils.data import DataLoader, ConcatDataset

def get_loader(opt, train=True, **kwargs):

    flag = 'train' if train else 'val'
    name = opt['datasets'][flag]['name']
    batch_size = opt['datasets']['train']['batch_size']
    n_threads = opt['datasets']['train']['n_workers']
    cpu = opt['cpu']

    if train:
        shuffle = True
    else:
        shuffle = True #False
        batch_size = 1

    benchmark = False
    if name.lower() in ['vid4']:
        benchmark = True
    if '+' not in name:
        dataset = import_module('data.estimator' + name.lower())
        dataset = getattr(dataset, name)(opt=opt, train=train, benchmark=benchmark, **kwargs)
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
            num_workers=n_threads,
            pin_memory=not cpu,
        )
    else:
        dataset = []
        name_list = name.split('+')
        for name_frag in name_list:
            dataset_frag = import_module('data.estimator' + name_frag.lower())
            dataset.append(getattr(dataset_frag, name_frag)(opt=opt, train=train, benchmark=benchmark, **kwargs))
        loader = DataLoader(
            dataset=ConcatDataset(dataset),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
            num_workers=n_threads,
            pin_memory=not cpu,
        )

    return loader


def get_dataset(
        opt, train=True, **kwargs):

    flag = 'train' if train else 'val'
    name = opt['datasets'][flag]['name']

    if '+' not in name:
        benchmark = False
        if name.lower() in ['vid4']:
            benchmark = True
        dataset = import_module('data.estimator.' + name.lower())
        dataset = getattr(dataset, name)(opt=opt, train=train, benchmark=benchmark, **kwargs)
    else:
        dataset = []
        name_list = name.split('+')
        for name_frag in name_list:
            benchmark = False
            if name_frag.lower() in ['vid4']:
                benchmark = True
            dataset_frag = import_module('data.estimator.' + name_frag.lower())
            dataset.append(getattr(dataset_frag, name_frag)(opt=opt, train=train, benchmark=benchmark, **kwargs))
        dataset = ConcatDataset(dataset)

    return dataset
