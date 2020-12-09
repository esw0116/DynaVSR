import logging
from collections import OrderedDict

import os
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import CharbonnierLoss, HuberLoss

logger = logging.getLogger('base')


class VideoBaseModel(BaseModel):
    def __init__(self, opt):
        super(VideoBaseModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        # self.print_network()
        self.load()
        
        self.log_dict = OrderedDict()

        #### loss
        loss_type = train_opt['pixel_criterion']
        if loss_type == 'l1':
            self.cri_pix = nn.L1Loss(reduction='mean').to(self.device)  # Change from sum to mean
        elif loss_type == 'l2':
            self.cri_pix = nn.MSELoss(reduction='mean').to(self.device)  # Change from sum to mean
        elif loss_type == 'cb':
            self.cri_pix = CharbonnierLoss().to(self.device)
        elif loss_type == 'huber':
            self.cri_pix = HuberLoss().to(self.device)
        else:
            raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
        self.l_pix_w = train_opt['pixel_weight']

        if self.is_train:
            self.netG.train()

            #### optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            if train_opt['ft_tsa_only']:
                normal_params = []
                tsa_fusion_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        if 'tsa_fusion' in k:
                            tsa_fusion_params.append(v)
                        else:
                            normal_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))
                optim_params = [
                    {  # add normal params first
                        'params': normal_params,
                        'lr': train_opt['lr_G']
                    },
                    {
                        'params': tsa_fusion_params,
                        'lr': train_opt['lr_G']
                    },
                ]
            if opt['train']['freeze_front']:
                normal_params = []
                freeze_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        if 'module.conv3d_1' in k or 'module.dense_block_1' in k or 'module.dense_block_2' in k or 'module.dense_block_3' in k:
                            freeze_params.append(v)
                        else:
                            normal_params.append(v)
                optim_params = [
                    {  # add normal params first
                        'params': normal_params,
                        'lr': train_opt['lr_G']
                    },
                    {
                        'params': freeze_params,
                        'lr': 0
                    },
                ]
            elif train_opt['small_offset_lr']:
                normal_params = []
                conv_offset_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        if 'pcd_align' in k or 'fea_L' in k or 'feature_extraction' in k or 'conv_first' in k:
                            conv_offset_params.append(v)
                        else:
                            normal_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))
                optim_params = [
                    {  # add normal params first
                        'params': normal_params,
                        'lr': train_opt['lr_G']
                    },
                    {
                        'params': conv_offset_params,
                        'lr': train_opt['lr_G'] * 0.1
                    },
                ]
            else:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        optim_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))

            if train_opt['optim'] == 'SGD':
                self.optimizer_G = torch.optim.SGD(optim_params, lr=train_opt['lr_G'], weight_decay=wd_G)
            else:
                self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                    weight_decay=wd_G,
                                                    betas=(train_opt['beta1'], train_opt['beta2']))
            
            self.optimizers.append(self.optimizer_G)

            #### schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError()


    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQs'].to(self.device)
        if need_GT:
            self.real_H = data['GT'].to(self.device)

    def set_params_lr_zero(self):
        # fix normal module
        self.optimizers[0].param_groups[0]['lr'] = 0

    def optimize_parameters(self, step):
        if self.opt['train']['ft_tsa_only'] and step < self.opt['train']['ft_tsa_only']:
            self.set_params_lr_zero()

        self.optimizer_G.zero_grad()
        self.fake_H = self.netG(self.var_L)

        l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
        l_pix.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()

    def optimize_by_loss(self, loss):
        if self.opt['train']['ft_tsa_only']:
            self.set_params_lr_zero()

        self.optimizer_G.zero_grad()
        loss.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_pix'] = loss.item()

    def calculate_loss(self):
        self.fake_H = self.netG(self.var_L)
        l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
        self.log_dict['l_pix'] = l_pix.item()
        return l_pix

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H = self.netG(self.var_L)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['rlt'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self, verbose=True):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            if verbose:
                logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def load_for_test(self):
        load_path_G = os.path.join(self.opt['path']['models'], 'latest_G.pth')
        if load_path_G is not None:
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
        else:
            print('No models are saved!')

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)

    def save_for_test(self):
        save_path = os.path.join(self.opt['path']['models'], 'latest_G.pth')
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            network = self.netG.module
            state_dict = network.state_dict()
        else:
            state_dict = self.netG.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)
