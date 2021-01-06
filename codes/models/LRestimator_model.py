import os
from collections import OrderedDict
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.nn.parallel import DataParallel, DistributedDataParallel

import models.networks as networks
from .base_model import BaseModel

logger = logging.getLogger('base')


def make_lr(img, kernel, scale):
    # img : BXT C H W
    weights = kernel.repeat((3, 1, 1, 1)).cuda()
    pad_func = torch.nn.ReflectionPad2d(kernel.size(-1) // 2)

    img = pad_func(img)
    lr_img = F.conv2d(img, weights, groups=3, stride=scale)

    return lr_img


class LRimgestimator_Model(BaseModel):
    def name(self):
        return 'Estimator_Model'

    def __init__(self, opt):
        super(LRimgestimator_Model, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']
        self.train_opt = train_opt
        self.kernel_size = opt['datasets']['train']['kernel_size']
        self.patch_size = opt['datasets']['train']['patch_size']
        self.batch_size = opt['datasets']['train']['batch_size']

        # define networks and load pretrained models
        self.scale = opt['scale']
        self.model_name = opt['network_E']['which_model_E']
        self.mode = opt['network_E']['mode']

        self.netE = networks.define_E(opt).to(self.device)
        if opt['dist']:
            self.netE = DistributedDataParallel(self.netE, device_ids=[torch.cuda.current_device()])
        else:
            self.netE = DataParallel(self.netE)
        self.load()

        # loss
        if train_opt['loss_ftn'] == 'l1':
            self.MyLoss = nn.L1Loss(reduction='mean').to(self.device)
        elif train_opt['loss_ftn'] == 'l2':
            self.MyLoss = nn.MSELoss(reduction='mean').to(self.device)
        else:
            self.MyLoss = None

        if self.is_train:
            self.netE.train()

            # optimizers
            self.optimizers = []
            wd_R = train_opt['weight_decay_R'] if train_opt['weight_decay_R'] else 0
            optim_params = []
            for k, v in self.netE.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    print('WARNING: params [%s] will not optimize.' % k)
            self.optimizer_E = torch.optim.Adam(optim_params, lr=train_opt['lr_C'], weight_decay=wd_R)
            print('Weight_decay:%f' % wd_R)
            self.optimizers.append(self.optimizer_E)

            # schedulers
            self.schedulers = []
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, \
                                                                    train_opt['lr_steps'], train_opt['lr_gamma']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

        #print('---------- Model initialized ------------------')
        #self.print_network()
        #print('-----------------------------------------------')

    def feed_data(self, data):
        self.real_H = data['LQs'].to(self.device)
        self.real_L = None if 'SuperLQs' not in data.keys() else data['SuperLQs'].to(self.device)
        B, T, C, H, W = self.real_H.shape
        if self.mode == 'image':
            self.var_H = self.real_H.reshape(B*T, C, H, W)
        else:
            self.var_H = self.real_H.transpose(1, 2)  # B C T H W

    def optimize_parameters(self, step=None):
        self.optimizer_E.zero_grad()
        fake_L = self.netE(self.var_H)
        if self.mode == 'image':
            H, W = fake_L.shape[-2:]
            B, T, C = self.real_H.shape[:3]
            self.fake_L = fake_L.reshape(B, T, C, H, W)
        else:
            self.fake_L = fake_L.transpose(1,2)
        LR_loss = self.MyLoss(self.fake_L, self.real_L)
        # set log
        self.log_dict['l_pix'] = LR_loss.item()
        # Show the std of real, fake kernel
        LR_loss.backward()
        self.optimizer_E.step()

    def forward_without_optim(self, step=None):
        fake_L = self.netE(self.var_H)
        if self.mode == 'image':
            H, W = fake_L.shape[-2:]
            B, T, C = self.real_H.shape[:3]
            self.fake_L = fake_L.reshape(B, T, C, H, W)
        else:
            self.fake_L = fake_L.transpose(1,2)

    def test(self):
        self.netE.eval()
        with torch.no_grad():
            fake_L = self.netE(self.var_H)
            if self.mode == 'image':
                H, W = fake_L.shape[-2:]
                B, T, C = self.real_H.shape[:3]
                self.fake_L = fake_L.reshape(B, T, C, H, W)
            else:
                self.fake_L = fake_L.transpose(1, 2)
        self.netE.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        T = self.fake_L.size(1)
        out_dict['LQ'] = self.real_L.detach()[0, T//2].float().cpu()
        out_dict['rlt'] = self.fake_L.detach()[0, T//2].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0, T//2].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netE)
        if isinstance(self.netE, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netE.__class__.__name__,
                                             self.netE.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netE.__class__.__name__)
        logger.info('Network R structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def load(self):
        load_path_E = self.opt['path']['pretrain_model_E']
        if load_path_E is not None:
            logger.info('Loading pretrained model for E [{:s}] ...'.format(load_path_E))
            self.load_network(load_path_E, self.netE)

    def save(self, iter_step):
        self.save_network(self.netE, 'E', iter_step)



