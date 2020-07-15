import os
import math
import argparse
import random
import logging
import imageio
import time
import pandas as pd
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from data.Backup.data_sampler import DistIterSampler

import options.options as option
from utils import util
from data.meta_learner import loader, create_dataloader, create_dataset, preprocessing
from models import create_model


def init_dist(backend='nccl', **kwargs):
    """initialization for distributed training"""
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--downsampling', '-D', type=str, default='BI')
    parser.add_argument('--exp_name', type=str, default='temp')
    args = parser.parse_args()
    if args.exp_name == 'temp':
        opt = option.parse(args.opt, is_train=True)
    else:
        opt = option.parse(args.opt, is_train=True, exp_name=args.exp_name)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)
    folder_name = opt['name']

    if args.exp_name != 'temp':
        folder_name = args.exp_name

    #### distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            print('\n\n')
            print(opt['path'])
            print('\n\n')
            util.mkdir_and_rename(
                opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt['use_tb_logger'] and 'debug' not in opt['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir='../tb_logger/' + folder_name)
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            # train_set = create_dataset(dataset_opt, scale=opt['scale'],
            #                           kernel_size=opt['datasets']['train']['kernel_size'],
            #                           model_name=opt['network_E']['which_model_E'])
            train_set = loader.get_dataset(opt, train=True)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt['dist']:
                train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
                total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
        elif phase == 'val':
            if '+' in opt['datasets']['val']['name']:
                val_set, val_loader = [], []
                valname_list = opt['datasets']['val']['name'].split('+')
                for i in range(len(valname_list)):
                    val_set.append(create_dataset(dataset_opt, scale=opt['scale'],
                                                  kernel_size=opt['datasets']['train']['kernel_size'],
                                                  model_name=opt['network_E']['which_model_E'], idx=i))
                    val_loader.append(create_dataloader(val_set[-1], dataset_opt, opt, None))
            else:
                val_set = [create_dataset(dataset_opt, scale=opt['scale'],
                                         kernel_size=opt['datasets']['train']['kernel_size'],
                                         model_name=opt['network_E']['which_model_E'])]
                val_loader = [create_dataloader(val_set, dataset_opt, opt, None)]
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    #### create model
    models = create_model(opt)
    assert len(models) == 2
    model, est_model = models[0], models[1]
    modelcp, est_modelcp = create_model(opt)
    model_fixed, est_model_fixed = create_model(opt)

    #### Define combined optimizer + scheduler
    optim_params = []
    for k, v in model.netG.named_parameters():
        if v.requires_grad:
            optim_params.append(v)
    for k, v in est_model.netE.named_parameters():
        if v.requires_grad:
            optim_params.append(v)
    if opt['train']['optim'] == 'Adam':
        optimizer = torch.optim.Adam(optim_params, lr=opt['train']['lr_G'], betas=(opt['train']['beta1'], opt['train']['beta2']))
    elif opt['train']['optim'] == 'SGD':
        optimizer = torch.optim.SGD(optim_params, lr=opt['train']['lr_G'])
    else:
        raise NotImplementedError()

    if opt['train']['lr_scheme'] == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, opt['train']['lr_steps'], opt['train']['lr_gamma'])
    else:
        raise NotImplementedError()

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
        est_model.resume_training(resume_state)

    else:
        current_step = 0
        start_epoch = 0

    #### training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    center_idx = (opt['datasets']['train']['N_frames']) // 2
    lr_alpha = opt['train']['maml']['lr_alpha']
    lr_alpha_est = opt['train']['maml']['lr_alpha_est'] if opt['train']['maml']['lr_alpha_est'] is not None else opt['train']['maml']['lr_alpha'] 
    update_step = opt['train']['maml']['adapt_iter']

    pd_log = pd.DataFrame(columns=['PSNR_Init', 'PSNR_Start', 'PSNR_Final({})'.format(update_step),
                                   'SSIM_Init', 'SSIM_Final'])

    def crop(LR_seq, HR, num_patches_for_batch=4, patch_size=44):
        """
        Crop given patches.

        Args:
            LR_seq: (B=1) x T x C x H x W
            HR: (B=1) x C x H x W

            patch_size (int, optional):

        Return:
            B(=batch_size) x T x C x H x W
        """
        # Find the lowest resolution
        cropped_lr = []
        cropped_hr = []
        assert HR.size(0) == 1
        LR_seq_ = LR_seq[0]
        #print(LR_seq[0].size())
        #print(HR.size())

        HR_ = HR[0]
        for _ in range(num_patches_for_batch):
            #patch_lr, patch_hr, _ = preprocessing.crop(LR_seq_, img_gt=HR_, patch_size=patch_size)
            patch_lr, patch_hr = preprocessing.common_crop(LR_seq_, HR_, patch_size=patch_size // 2)
            cropped_lr.append(patch_lr)
            cropped_hr.append(patch_hr)

        cropped_lr = torch.stack(cropped_lr, dim=0)
        cropped_hr = torch.stack(cropped_hr, dim=0)

        #print(cropped_lr.size(), cropped_hr.size())

        return cropped_lr, cropped_hr

    print(folder_name)

    # initialize bicubic performance
    bicubic_performance = [None] * len(val_set)
    BICUBIC_EVALUATE = [True] * len(val_set)

    termination = False
    for epoch in range(start_epoch, total_epochs + 1):
        if opt['dist']:
            train_sampler.set_epoch(epoch)



        # Main training loop
        for _, train_data in enumerate(train_loader):
            if termination is True:
                break
            current_step += 1
            if current_step > total_iters and termination is False:
                termination = True
                print('Training terminated')

            #### update learning rate
            #model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])
            scheduler.step() # no warmup for now
            optimizer.zero_grad()


            cropped_meta_train_data = {}
            meta_train_data = {}
            meta_test_data = {}
            
            '''
            # Make SuperLR seq using estimation model
            if not opt['train']['use_real']:
                est_model.feed_data(train_data)
                #est_model.test()
                est_model.forward_without_optim()
                superlr_seq = est_model.fake_L
                meta_train_data['LQs'] = superlr_seq
            else:
                meta_train_data['LQs'] = train_data['SuperLQs']
            '''
            meta_train_data['GT'] = train_data['LQs'][:, center_idx]
            meta_test_data['LQs'] = train_data['LQs']
            meta_test_data['GT'] = train_data['GT'][:, center_idx]

            # batch size = number of tasks
            total_loss_q = 0
            #batch_size = meta_train_data['LQs'].size(0)
            batch_size = train_data['LQs'].size(0)
            #model.optimizer_G.zero_grad()


            # Looping over batch dimension (due to high memory consumption)
            for batch in range(batch_size):
                #print(batch+1)
                train_data_i = {
                    'LQs': train_data['LQs'][batch:batch+1],
                    'GT': train_data['GT'][batch:batch+1],
                    'SuperLQs': train_data['SuperLQs'][batch:batch+1]
                }
                meta_train_data_i = {}
                #meta_train_data_i['LQs'] = meta_train_data['LQs'][batch:batch+1]
                meta_train_data_i['GT'] = meta_train_data['GT'][batch:batch+1]
                meta_test_data_i = {}
                meta_test_data_i['LQs'] = meta_test_data['LQs'][batch:batch+1]
                meta_test_data_i['GT'] = meta_test_data['GT'][batch:batch+1]

                if opt['network_G']['which_model_G'] == 'TOF':
                    # Bicubic upsample to match the size
                    LQs = meta_test_data_i['LQs']
                    B, T, C, H, W = LQs.shape
                    LQs = LQs.reshape(B*T, C, H, W)
                    Bic_LQs = F.interpolate(LQs, scale_factor=opt['scale'], mode='bicubic', align_corners=True)
                    meta_test_data_i['LQs'] = Bic_LQs.reshape(B, T, C, H*opt['scale'], W*opt['scale'])
                
                modelcp.netG, est_modelcp.netE = deepcopy(model.netG), deepcopy(est_model.netE)
                optim_params = []

                sr_params = []
                for k, v in modelcp.netG.named_parameters():
                    if v.requires_grad:
                        sr_params.append(v)
                
                est_params = []
                for k, v in est_modelcp.netE.named_parameters():
                    if v.requires_grad:
                        est_params.append(v)

                optim_params = [
                    {  # add normal params first
                        'params': sr_params,
                        'lr': lr_alpha
                    },
                    {
                        'params': est_params,
                        'lr': lr_alpha_est
                    },
                ]

                if opt['train']['maml']['optimizer'] == 'Adam':
                    inner_optimizer = torch.optim.Adam(optim_params, lr=lr_alpha,
                                                      betas=(
                                                      opt['train']['maml']['beta1'], opt['train']['maml']['beta2']))
                elif opt['train']['maml']['optimizer'] == 'SGD':
                    inner_optimizer = torch.optim.SGD(optim_params, lr=lr_alpha)
                else:
                    raise NotImplementedError()

                for k in range(update_step):
                    inner_optimizer.zero_grad()

                    # Make SuperLR seq using estimation model
                    if not opt['train']['use_real']:
                        est_model.feed_data(train_data_i)
                        est_model.forward_without_optim()
                        superlr_seq = est_model.fake_L
                        meta_train_data_i['LQs'] = superlr_seq
                    else:
                        meta_train_data_i['LQs'] = train_data_i['SuperLQs']
                    
                     
                    if opt['network_G']['which_model_G'] == 'TOF':
                        # Bicubic upsample to match the size
                        LQs = meta_train_data_i['LQs']
                        B, T, C, H, W = LQs.shape
                        LQs = LQs.reshape(B*T, C, H, W)
                        Bic_LQs = F.interpolate(LQs, scale_factor=opt['scale'], mode='bicubic', align_corners=True)
                        meta_train_data_i['LQs'] = Bic_LQs.reshape(B, T, C, H*opt['scale'], W*opt['scale'])

                    # Meta training
                    if opt['train']['maml']['use_patch']:
                        cropped_meta_train_data['LQs'], cropped_meta_train_data['GT'] = \
                            crop(meta_train_data_i['LQs'], meta_train_data_i['GT'],
                                 opt['train']['maml']['num_patch'],
                                 opt['train']['maml']['patch_size'])
                        model.feed_data(cropped_meta_train_data)
                    else:
                        model.feed_data(meta_train_data_i)
                    loss_train = model.calculate_loss()

                    ## Add SLR pixelwise loss while training
                    #est_model_fixed.feed_data(train_data_i)
                    #est_model_fixed.test()
                    #slr_initialized = est_model_fixed.fake_L
                    #slr_initialized = slr_initialized.to('cuda') 
                    if opt['network_G']['which_model_G'] == 'TOF':
                        loss_train += F.l1_loss(LQs.to('cuda'), train_data_i['SuperLQs'].to('cuda'))
                    else:
                        loss_train += F.l1_loss(meta_train_data_i['LQs'].to('cuda'), train_data_i['SuperLQs'].to('cuda'))

                    loss_train.backward()
                    # print('Inner Update, {}'.format(k+1))
                    inner_optimizer.step()


                # Meta testing - final forward to update the base model parameters

                model.feed_data(meta_test_data_i)
                loss_q = model.calculate_loss()
                # Copy base parameters to current model
                for param, base_param in zip(model.netG.parameters(), modelcp.netG.parameters()):
                    param.data = base_param.data

                # Calculate gradient & update meta-learner
                #print(type(model.netG.parameters()))
                #print(type(model.netG.parameters() + est_model.netE.parameters()))
                grads = torch.autograd.grad(loss_q / batch_size, model.netG.parameters())
                for j, param in enumerate(model.netG.parameters()):
                    param.grad += grads[j]

                est_model.feed_data(train_data_i)
                est_model.forward_without_optim()
                loss_e = est_model.MyLoss(est_model.fake_L, est_model.real_L) # / 100  # compare with GT SLR image to current output
                for param, base_param in zip(est_model.netE.parameters(), est_modelcp.netE.parameters()):
                    param.data = base_param.data
                # TODO: set different loss scales for loss_e and loss_q?
                gradsE = torch.autograd.grad(loss_e / (batch_size*10), est_model.netE.parameters())
                # print(gradsE)
                for j, param in enumerate(est_model.netE.parameters()):
                    param.grad += gradsE[j]


                #(loss_q/batch_size).backward()
                total_loss_q += loss_q.item() / batch_size
                # print(batch, k, loss_train.item(), loss_q.item())
                
                del modelcp.netG, est_modelcp.netE

            tb_logger.add_scalar('Train loss', total_loss_q, global_step=current_step)
            # print('Meta Update')
            #model.optimizer_G.step()
            optimizer.step()

            #### log
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '[epoch:{:3d}, iter:{:8,d}, lr:('.format(epoch, current_step)
                #for v in model.get_current_learning_rate():
                #    message += '{:.3e},'.format(v)
                #message += '{:.3e},'.format([param_group['lr'] for param_group in optimizer.param_groups][0])
                message += '{:.3e},'.format(optimizer.param_groups[0]['lr'])
                message += ')] '
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)
            
            
            
            #### validation
            #print(opt['datasets'].get('val', None))
            if opt['datasets'].get('val', None) and (current_step % opt['train']['val_freq'] == 0 or termination is True):
                if opt['model'] in ['sr', 'srgan'] and rank <= 0:  # image restoration validation
                    # does not support multi-GPU validation
                    pbar = util.ProgressBar(len(val_loader))
                    avg_psnr = 0.
                    idx = 0
                    for val_data in val_loader:
                        idx += 1
                        img_name = os.path.splitext(os.path.basename(val_data['LQ_path'][0]))[0]
                        img_dir = os.path.join(opt['path']['val_images'], img_name)
                        util.mkdir(img_dir)

                        model.feed_data(val_data)
                        model.test()

                        visuals = model.get_current_visuals()
                        sr_img = util.tensor2img(visuals['rlt'])  # uint8
                        gt_img = util.tensor2img(visuals['GT'])  # uint8

                        # Save SR images for reference
                        save_img_path = os.path.join(img_dir,
                                                     '{:s}_{:d}.png'.format(img_name, current_step))
                        util.save_img(sr_img, save_img_path)

                        # calculate PSNR
                        sr_img, gt_img = util.crop_border([sr_img, gt_img], opt['scale'])
                        avg_psnr += util.calculate_psnr(sr_img, gt_img)
                        pbar.update('Test {}'.format(img_name))

                    avg_psnr = avg_psnr / idx

                    # log
                    logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        tb_logger.add_scalar('psnr', avg_psnr, current_step)
                        
                else:  # video restoration validation <<<----Use this
                    if opt['dist']:
                        # multi-GPU testing
                        for val_idx, val_set_frag in enumerate(val_set):
                            # PSNR_rlt: psnr_init, psnr_before, psnr_after
                            psnr_rlt = [{}, {}, {}]
                            # SSIM_rlt: ssim_init, ssim_after
                            ssim_rlt = [{}, {}]
                            if rank == 0:
                                pbar = util.ProgressBar(len(val_set_frag))
                            for idx in range(rank, len(val_set_frag), world_size):
                                val_data = val_set_frag[idx]
                                if 'name' in val_data.keys():
                                    name = val_data['name'][center_idx][0]
                                else:
                                    name = '{}/{:08d}'.format(val_data['folder'], int(val_data['idx'].split('/')[0]))

                                train_folder = os.path.join('../results', folder_name, name)
                                if not os.path.isdir(train_folder):
                                    os.makedirs(train_folder, exist_ok=True)

                                val_data['SuperLQs'].unsqueeze_(0)
                                val_data['LQs'].unsqueeze_(0)
                                val_data['GT'].unsqueeze_(0)
                                folder = val_data['folder']
                                idx_d, max_idx = val_data['idx'].split('/')
                                idx_d, max_idx = int(idx_d), int(max_idx)
                                for i in range(len(psnr_rlt)):
                                    if psnr_rlt[i].get(folder, None) is None:
                                        psnr_rlt[i][folder] = torch.zeros(max_idx, dtype=torch.float32, device='cuda')
                                for i in range(len(ssim_rlt)):
                                    if ssim_rlt[i].get(folder, None) is None:
                                        ssim_rlt[i][folder] = torch.zeros(max_idx, dtype=torch.float32, device='cuda')

                                cropped_meta_train_data = {}
                                meta_train_data = {}
                                meta_test_data = {}
                                
                                # Make SuperLR seq using estimation model Later
                                meta_train_data['GT'] = val_data['LQs'][:, center_idx]
                                meta_test_data['LQs'] = val_data['LQs'][0:1]
                                meta_test_data['GT'] = val_data['GT'][0:1, center_idx]
                                # Check whether the batch size of each validation data is 1
                                assert val_data['SuperLQs'].size(0) == 1

                                if opt['network_G']['which_model_G'] == 'TOF':
                                    LQs = meta_test_data['LQs']
                                    B, T, C, H, W = LQs.shape
                                    LQs = LQs.reshape(B*T, C, H, W)
                                    Bic_LQs = F.interpolate(LQs, scale_factor=opt['scale'], mode='bicubic', align_corners=True)
                                    meta_test_data['LQs'] = Bic_LQs.reshape(B, T, C, H*opt['scale'], W*opt['scale'])

                                #modelcp.netG = deepcopy(model.netG)
                                modelcp.netG, est_modelcp.netE = deepcopy(model.netG), deepcopy(est_model.netE)
                                
                                optim_params = []

                                sr_params = []
                                for k, v in modelcp.netG.named_parameters():
                                    if v.requires_grad:
                                        sr_params.append(v)
                                
                                est_params = []
                                for k, v in est_modelcp.netE.named_parameters():
                                    if v.requires_grad:
                                        est_params.append(v)

                                optim_params = [
                                    {  # add normal params first
                                        'params': sr_params,
                                        'lr': lr_alpha
                                    },
                                    {
                                        'params': est_params,
                                        'lr': lr_alpha_est
                                    },
                                ]
                                
                                if opt['train']['maml']['optimizer'] == 'Adam':
                                    inner_optimizer = torch.optim.Adam(optim_params, lr=lr_alpha,
                                                                    betas=(
                                                                    opt['train']['maml']['beta1'], opt['train']['maml']['beta2']))
                                elif opt['train']['maml']['optimizer'] == 'SGD':
                                    inner_optimizer = torch.optim.SGD(optim_params, lr=lr_alpha)
                                else:
                                    raise NotImplementedError()

                                #if opt['train']['maml']['optimizer'] == 'Adam':
                                #    inner_optimizer = torch.optim.Adam(modelcp.netG.parameters(), lr=lr_alpha,
                                #                                      betas=(opt['train']['maml']['beta1'],
                                #                                             opt['train']['maml']['beta2']))
                                #elif opt['train']['maml']['optimizer'] == 'SGD':
                                #    inner_optimizer = torch.optim.SGD(modelcp.netG.parameters(), lr=lr_alpha)
                                #else:
                                #    raise NotImplementedError()
                                
                                if max_idx < 80 or (idx_d < max_idx/2 and max_idx >= 80):

                                    # Before start inner-update evaluate the PSNR init, PSNR start
                                    # Init (Before Meta update)
                                    if BICUBIC_EVALUATE[val_idx]:
                                        model_fixed.feed_data(meta_test_data)
                                        model_fixed.test()
                                        model_fixed_visuals = model_fixed.get_current_visuals()
                                        hr_image = util.tensor2img(model_fixed_visuals['GT'], mode='rgb')
                                        init_image = util.tensor2img(model_fixed_visuals['rlt'], mode='rgb')
                                        # Note : hr_image, init_image are RGB, [0,255], np.uint8
                                        # imageio.imwrite(os.path.join(train_folder, 'hr.png'), hr_image)
                                        # imageio.imwrite(os.path.join(train_folder, 'sr_init.png'), init_image)
                                        # Update PSNR init rlt
                                        psnr_bicubic = util.calculate_psnr(init_image, hr_image)
                                        psnr_rlt[0][folder][idx_d] = psnr_bicubic
                                        # Update SSIM init rlt
                                        # ssim_bicubic = util.calculate_ssim(init_image, hr_image)
                                        ssim_rlt[0][folder][idx_d] = 0 #ssim_bicubic
                                        
                                    else:
                                        psnr_rlt[0][folder][idx_d] = bicubic_performance[val_idx]['psnr'][folder][idx_d]
                                        ssim_rlt[0][folder][idx_d] = bicubic_performance[val_idx]['ssim'][folder][idx_d]


                                    #### Forward
                                    update_time = 0
                                    # Before (After Meta update, Before Inner update)
                                    modelcp.feed_data(meta_test_data)
                                    modelcp.test()
                                    model_start_visuals = modelcp.get_current_visuals(need_GT=True)
                                    hr_image = util.tensor2img(model_start_visuals['GT'], mode='rgb')
                                    start_image = util.tensor2img(model_start_visuals['rlt'], mode='rgb')
                                    # imageio.imwrite(os.path.join(train_folder, 'sr_start.png'), start_image)
                                    psnr_rlt[1][folder][idx_d] = util.calculate_psnr(start_image, hr_image)
                                    
                                    # Inner Loop Update
                                    st = time.time()
                                    for i in range(update_step):

                                    # Make SuperLR seq using UPDATED estimation model
                                        if not opt['train']['use_real']:
                                            est_modelcp.feed_data(val_data)
                                            #est_model.test()
                                            est_modelcp.forward_without_optim()
                                            superlr_seq = est_modelcp.fake_L
                                            meta_train_data['LQs'] = superlr_seq
                                        else:
                                            meta_train_data['LQs'] = val_data['SuperLQs']

                                        if opt['network_G']['which_model_G'] == 'TOF':
                                            # Bicubic upsample to match the size
                                            LQs = meta_train_data['LQs']
                                            B, T, C, H, W = LQs.shape
                                            LQs = LQs.reshape(B*T, C, H, W)
                                            Bic_LQs = F.interpolate(LQs, scale_factor=opt['scale'], mode='bicubic', align_corners=True)
                                            meta_train_data['LQs'] = Bic_LQs.reshape(B, T, C, H*opt['scale'], W*opt['scale'])
                                        
                                        # Update both modelcp + estmodelcp jointly
                                        inner_optimizer.zero_grad()
                                        if opt['train']['maml']['use_patch']:
                                            cropped_meta_train_data['LQs'], cropped_meta_train_data['GT'] = \
                                                crop(meta_train_data['LQs'], meta_train_data['GT'],
                                                    opt['train']['maml']['num_patch'],
                                                    opt['train']['maml']['patch_size'])
                                            modelcp.feed_data(cropped_meta_train_data)
                                        else:
                                            modelcp.feed_data(meta_train_data)

                                        loss_train = modelcp.calculate_loss()
                                        ## Add SLR pixelwise loss while training
                                        est_model_fixed.feed_data(val_data)
                                        est_model_fixed.test()
                                        slr_initialized = est_model_fixed.fake_L
                                        slr_initialized = slr_initialized.to('cuda')
                                        
                                        if opt['network_G']['which_model_G'] == 'TOF':
                                            loss_train += F.l1_loss(LQs.to('cuda'), slr_initialized)
                                        else:
                                            loss_train += F.l1_loss(meta_train_data['LQs'].to('cuda'), slr_initialized)

                                        loss_train.backward()
                                        inner_optimizer.step()

                                    et = time.time()
                                    update_time = et - st

                                    modelcp.feed_data(meta_test_data)
                                    modelcp.test()
                                    model_update_visuals = modelcp.get_current_visuals(need_GT=False)
                                    update_image = util.tensor2img(model_update_visuals['rlt'], mode='rgb')
                                    # Save and calculate final image
                                    # imageio.imwrite(os.path.join(train_folder, 'sr_finish.png'), update_image)
                                    psnr_rlt[2][folder][idx_d] = util.calculate_psnr(update_image, hr_image)
                                    ssim_rlt[1][folder][idx_d] = 0 #util.calculate_ssim(update_image, hr_image)

                                    if name in pd_log.index:
                                        pd_log.at[name, 'PSNR_Init'] = psnr_rlt[0][folder][idx_d].item()
                                        pd_log.at[name, 'PSNR_Start'] = (psnr_rlt[1][folder][idx_d] - psnr_rlt[0][folder][idx_d]).item()
                                        pd_log.at[name, 'PSNR_Final({})'.format(update_step)] = (psnr_rlt[2][folder][idx_d] - psnr_rlt[0][folder][idx_d]).item()
                                        pd_log.at[name, 'SSIM_Init'] = ssim_rlt[0][folder][idx_d].item()
                                        pd_log.at[name, 'SSIM_Final'] = ssim_rlt[1][folder][idx_d].item()
                                    else:
                                        pd_log.loc[name] = [psnr_rlt[0][folder][idx_d].item(),
                                                            psnr_rlt[1][folder][idx_d].item() - psnr_rlt[0][folder][idx_d].item(),
                                                            psnr_rlt[2][folder][idx_d].item() - psnr_rlt[0][folder][idx_d].item(),
                                                            ssim_rlt[0][folder][idx_d].item(), ssim_rlt[1][folder][idx_d].item()]

                                    pd_log.to_csv(os.path.join('../results', folder_name, 'psnr_update_{}.csv'.format(val_idx)))

                                    if rank == 0:
                                        for _ in range(world_size):
                                            pbar.update('Test {} - {}/{}: I: {:.3f}/{:.4f} \tF+: {:.3f}/{:.4f} \tTime: {:.3f}s'
                                                        .format(folder, idx_d, max_idx,
                                                                psnr_rlt[0][folder][idx_d].item(), ssim_rlt[0][folder][idx_d].item(),
                                                                psnr_rlt[2][folder][idx_d].item(), ssim_rlt[1][folder][idx_d].item(),
                                                                update_time
                                                                ))

                            
                            if BICUBIC_EVALUATE[val_idx]:
                                bicubic_performance[val_idx] = {}
                                bicubic_performance[val_idx]['psnr'] = deepcopy(psnr_rlt[0])
                                bicubic_performance[val_idx]['ssim'] = deepcopy(ssim_rlt[0])
                                BICUBIC_EVALUATE[val_idx] = False
                            
                            ## collect data
                            for i in range(len(psnr_rlt)):
                                for _, v in psnr_rlt[i].items():
                                    dist.reduce(v, 0)
                            for i in range(len(ssim_rlt)):
                                for _, v in ssim_rlt[i].items():
                                    dist.reduce(v, 0)
                            dist.barrier()

                            if rank == 0:
                                psnr_rlt_avg = {}
                                psnr_total_avg = [0., 0., 0.]
                                # 0: Init, 1: Start, 2: Final
                                #Just calculate the final value of psnr_rlt(i.e. psnr_rlt[2])
                                for k, v_init in psnr_rlt[0].items():
                                    v_start = psnr_rlt[1][k]
                                    v_final = psnr_rlt[2][k]
                                    psnr_rlt_avg[k] = [torch.sum(v_init).cpu().item() / (v_init!=0).sum().item(),
                                                        torch.sum(v_start).cpu().item() / (v_start!=0).sum().item(),
                                                        torch.sum(v_final).cpu().item() / (v_final!=0).sum().item()]
                                    for i in range(len(psnr_rlt)):
                                        psnr_total_avg[i] += psnr_rlt_avg[k][i]
                                for i in range(len(psnr_rlt)):
                                    psnr_total_avg[i] /= len(psnr_rlt[0])
                                log_s = '# Validation # Final PSNR: {:.4e}:'.format(psnr_total_avg[2])
                                for k, v in psnr_rlt_avg.items():
                                    log_s += ' {}: {:.4e}'.format(k, v[2])
                                logger.info(log_s)
                                if opt['use_tb_logger'] and 'debug' not in opt['name']:
                                    tb_logger.add_scalar('psnr_avg_{}_0:init'.format(val_idx), psnr_total_avg[0], current_step)
                                    tb_logger.add_scalar('psnr_avg_{}_1:start'.format(val_idx), psnr_total_avg[1], current_step)
                                    tb_logger.add_scalar('psnr_avg_{}_2:final'.format(val_idx), psnr_total_avg[2], current_step)

                                    for k, v in psnr_rlt_avg.items():
                                        tb_logger.add_scalar(k+'_psnr_avg_{}_0:init'.format(val_idx), v[0], current_step)
                                        tb_logger.add_scalar(k+'_psnr_avg_{}_1:start'.format(val_idx), v[1], current_step)
                                        tb_logger.add_scalar(k+'_psnr_avg_{}_2:final'.format(val_idx), v[2], current_step)
                                '''
                                ssim_rlt_avg = {}
                                ssim_total_avg = 0.
                                #Just calculate the final value of ssim_rlt(i.e. ssim_rlt[1])
                                for k, v in ssim_rlt[1].items():
                                    ssim_rlt_avg[k] = torch.sum(v).cpu().item() / (v!=0).sum().item()
                                    ssim_total_avg += ssim_rlt_avg[k]
                                ssim_total_avg /= len(ssim_rlt[1])
                                log_s = '# Validation # SSIM: {:.4e}:'.format(ssim_total_avg)
                                for k, v in ssim_rlt_avg.items():
                                    log_s += ' {}: {:.4e}'.format(k, v)
                                logger.info(log_s)
                                if opt['use_tb_logger'] and 'debug' not in opt['name']:
                                    tb_logger.add_scalar('ssim_avg', ssim_total_avg, current_step)
                                    for k, v in ssim_rlt_avg.items():
                                        tb_logger.add_scalar(k+'ssim_avg', v, current_step)
                                '''

                    else:
                        # Single GPU
                        for val_idx, val_set_frag in enumerate(val_set):
                            # PSNR_rlt: psnr_init, psnr_before, psnr_after
                            psnr_rlt = [{}, {}, {}]
                            # SSIM_rlt: ssim_init, ssim_after
                            ssim_rlt = [{}, {}]
                            pbar = util.ProgressBar(len(val_set_frag))
                            for val_data in val_loader[val_idx]:
                                folder = val_data['folder'][0]
                                idx_d, max_idx = int(val_data['idx'][0].split('/')[0]), int(val_data['idx'][0].split('/')[1])
                                if 'name' in val_data.keys():
                                    name = val_data['name'][0][center_idx][0]
                                else:
                                    name = '{}/{:08d}'.format(folder, idx_d)

                                train_folder = os.path.join('../results', opt['name'], name)
                                if not os.path.isdir(train_folder):
                                    os.makedirs(train_folder, exist_ok=True)

                                for i in range(len(psnr_rlt)):
                                    if psnr_rlt[i].get(folder, None) is None:
                                        psnr_rlt[i][folder] = torch.zeros(max_idx, dtype=torch.float32, device='cuda')
                                        #psnr_rlt[i][folder] = []
                                for i in range(len(ssim_rlt)):
                                    if ssim_rlt[i].get(folder, None) is None:
                                        ssim_rlt[i][folder] = torch.zeros(max_idx, dtype=torch.float32, device='cuda')
                                        #ssim_rlt[i][folder] = []

                                cropped_meta_train_data = {}
                                meta_train_data = {}
                                meta_test_data = {}
                                
                                # Make SuperLR seq using estimation model
                                '''if not opt['train']['use_real']:
                                    est_model.feed_data(val_data)
                                    #est_model.test()
                                    est_model.forward_without_optim()
                                    superlr_seq = est_model.fake_L
                                    meta_train_data['LQs'] = superlr_seq
                                else:
                                    meta_train_data['LQs'] = val_data['SuperLQs']
                                '''
                                meta_train_data['GT'] = val_data['LQs'][:, center_idx]
                                meta_test_data['LQs'] = val_data['LQs'][0:1]
                                meta_test_data['GT'] = val_data['GT'][0:1, center_idx]
                                # Check whether the batch size of each validation data is 1
                                assert val_data['SuperLQs'].size(0) == 1

                                if opt['network_G']['which_model_G'] == 'TOF':
                                    LQs = meta_test_data['LQs']
                                    B, T, C, H, W = LQs.shape
                                    LQs = LQs.reshape(B*T, C, H, W)
                                    Bic_LQs = F.interpolate(LQs, scale_factor=opt['scale'], mode='bicubic', align_corners=True)
                                    meta_test_data['LQs'] = Bic_LQs.reshape(B, T, C, H*opt['scale'], W*opt['scale'])

                                #modelcp.netG = deepcopy(model.netG)
                                modelcp.netG, est_modelcp.netE = deepcopy(model.netG), deepcopy(est_model.netE)
                                optim_params = []

                                sr_params = []
                                for k, v in modelcp.netG.named_parameters():
                                    if v.requires_grad:
                                        sr_params.append(v)
                                
                                est_params = []
                                for k, v in est_modelcp.netE.named_parameters():
                                    if v.requires_grad:
                                        est_params.append(v)

                                optim_params = [
                                    {  # add normal params first
                                        'params': sr_params,
                                        'lr': lr_alpha
                                    },
                                    {
                                        'params': est_params,
                                        'lr': lr_alpha_est
                                    },
                                ]

                                if opt['train']['maml']['optimizer'] == 'Adam':
                                    inner_optimizer = torch.optim.Adam(optim_params, lr=lr_alpha,
                                                                    betas=(
                                                                    opt['train']['maml']['beta1'], opt['train']['maml']['beta2']))
                                elif opt['train']['maml']['optimizer'] == 'SGD':
                                    inner_optimizer = torch.optim.SGD(optim_params, lr=lr_alpha)
                                else:
                                    raise NotImplementedError()

                                #if opt['train']['maml']['optimizer'] == 'Adam':
                                #    inner_optimizer = torch.optim.Adam(modelcp.netG.parameters(), lr=lr_alpha,
                                #                                      betas=(opt['train']['maml']['beta1'],
                                #                                             opt['train']['maml']['beta2']))
                                #elif opt['train']['maml']['optimizer'] == 'SGD':
                                #    inner_optimizer = torch.optim.SGD(modelcp.netG.parameters(), lr=lr_alpha)
                                #else:
                                #    raise NotImplementedError()
                                
                                if max_idx < 80 or (idx_d < max_idx/2 and max_idx >= 80):

                                    # Before start inner-update evaluate the PSNR init, PSNR start
                                    # Init (Before Meta update)
                                    if BICUBIC_EVALUATE[val_idx]:
                                        model_fixed.feed_data(meta_test_data)
                                        model_fixed.test()
                                        model_fixed_visuals = model_fixed.get_current_visuals()
                                        hr_image = util.tensor2img(model_fixed_visuals['GT'], mode='rgb')
                                        init_image = util.tensor2img(model_fixed_visuals['rlt'], mode='rgb')
                                        # Note : hr_image, init_image are RGB, [0,255], np.uint8
                                        # imageio.imwrite(os.path.join(train_folder, 'hr.png'), hr_image)
                                        # imageio.imwrite(os.path.join(train_folder, 'sr_init.png'), init_image)
                                        # Update PSNR init rlt
                                        psnr_bicubic = util.calculate_psnr(init_image, hr_image)
                                        psnr_rlt[0][folder][idx_d] = psnr_bicubic
                                        # Update SSIM init rlt
                                        # ssim_bicubic = util.calculate_ssim(init_image, hr_image)
                                        ssim_rlt[0][folder][idx_d] = 0.1 #ssim_bicubic
                                        
                                    else:
                                        psnr_rlt[0][folder][idx_d] = bicubic_performance[val_idx]['psnr'][folder][idx_d]
                                        ssim_rlt[0][folder][idx_d] = bicubic_performance[val_idx]['ssim'][folder][idx_d]


                                    #### Forward
                                    update_time = 0
                                    # Before (After Meta update, Before Inner update)
                                    modelcp.feed_data(meta_test_data)
                                    modelcp.test()
                                    model_start_visuals = modelcp.get_current_visuals(need_GT=True)
                                    hr_image = util.tensor2img(model_start_visuals['GT'], mode='rgb')
                                    start_image = util.tensor2img(model_start_visuals['rlt'], mode='rgb')
                                    # imageio.imwrite(os.path.join(train_folder, 'sr_start.png'), start_image)
                                    psnr_rlt[1][folder][idx_d] = util.calculate_psnr(start_image, hr_image)
                                    
                                    # Inner Loop Update
                                    st = time.time()
                                    for i in range(update_step):

                                    # Make SuperLR seq using UPDATED estimation model
                                        if not opt['train']['use_real']:
                                            est_modelcp.feed_data(val_data)
                                            #est_model.test()
                                            est_modelcp.forward_without_optim()
                                            superlr_seq = est_modelcp.fake_L
                                            meta_train_data['LQs'] = superlr_seq
                                        else:
                                            meta_train_data['LQs'] = val_data['SuperLQs']

                                        if opt['network_G']['which_model_G'] == 'TOF':
                                            # Bicubic upsample to match the size
                                            LQs = meta_train_data['LQs']
                                            B, T, C, H, W = LQs.shape
                                            LQs = LQs.reshape(B*T, C, H, W)
                                            Bic_LQs = F.interpolate(LQs, scale_factor=opt['scale'], mode='bicubic', align_corners=True)
                                            meta_train_data['LQs'] = Bic_LQs.reshape(B, T, C, H*opt['scale'], W*opt['scale'])
                                        
                                        # Update both modelcp + estmodelcp jointly
                                        inner_optimizer.zero_grad()
                                        if opt['train']['maml']['use_patch']:
                                            cropped_meta_train_data['LQs'], cropped_meta_train_data['GT'] = \
                                                crop(meta_train_data['LQs'], meta_train_data['GT'],
                                                    opt['train']['maml']['num_patch'],
                                                    opt['train']['maml']['patch_size'])
                                            modelcp.feed_data(cropped_meta_train_data)
                                        else:
                                            modelcp.feed_data(meta_train_data)

                                        loss_train = modelcp.calculate_loss()

                                        ## Add SLR pixelwise loss while training
                                        est_model_fixed.feed_data(val_data)
                                        est_model_fixed.test()
                                        slr_initialized = est_model_fixed.fake_L
                                        slr_initialized = slr_initialized.to('cuda')

                                        if opt['network_G']['which_model_G'] == 'TOF':
                                            loss_train += F.l1_loss(LQs.to('cuda'), slr_initialized)
                                        else:
                                            loss_train += F.l1_loss(meta_train_data['LQs'].to('cuda'), slr_initialized)
                                        loss_train.backward()
                                        inner_optimizer.step()

                                    et = time.time()
                                    update_time = et - st

                                    modelcp.feed_data(meta_test_data)
                                    modelcp.test()
                                    model_update_visuals = modelcp.get_current_visuals(need_GT=False)
                                    update_image = util.tensor2img(model_update_visuals['rlt'], mode='rgb')
                                    # Save and calculate final image
                                    # imageio.imwrite(os.path.join(train_folder, 'sr_finish.png'), update_image)
                                    psnr_rlt[2][folder][idx_d] = util.calculate_psnr(update_image, hr_image)
                                    ssim_rlt[1][folder][idx_d] = 0.11 #util.calculate_ssim(update_image, hr_image)

                                    if name in pd_log.index:
                                        pd_log.at[name, 'PSNR_Init'] = psnr_rlt[0][folder][idx_d].item()
                                        pd_log.at[name, 'PSNR_Start'] = (psnr_rlt[1][folder][idx_d] - psnr_rlt[0][folder][idx_d]).item()
                                        pd_log.at[name, 'PSNR_Final({})'.format(update_step)] = (psnr_rlt[2][folder][idx_d] - psnr_rlt[0][folder][idx_d]).item()
                                        pd_log.at[name, 'SSIM_Init'] = ssim_rlt[0][folder][idx_d].item()
                                        pd_log.at[name, 'SSIM_Final'] = ssim_rlt[1][folder][idx_d].item()
                                    else:
                                        pd_log.loc[name] = [psnr_rlt[0][folder][idx_d].item(),
                                                            psnr_rlt[1][folder][idx_d].item() - psnr_rlt[0][folder][idx_d].item(),
                                                            psnr_rlt[2][folder][idx_d].item() - psnr_rlt[0][folder][idx_d].item(),
                                                            ssim_rlt[0][folder][idx_d].item(), ssim_rlt[1][folder][idx_d].item()]

                                    pd_log.to_csv(os.path.join('../results', folder_name, 'psnr_update_{}.csv'.format(val_idx)))

                                    pbar.update('Test {} - {}/{}: I: {:.3f}/{:.4f} \tF+: {:.3f}/{:.4f} \tTime: {:.3f}s'
                                                .format(folder, idx_d, max_idx,
                                                        psnr_rlt[0][folder][idx_d].item(), ssim_rlt[0][folder][idx_d].item(),
                                                        psnr_rlt[2][folder][idx_d].item(), ssim_rlt[1][folder][idx_d].item(),
                                                        update_time
                                                        ))


                            if BICUBIC_EVALUATE[val_idx]:
                                bicubic_performance[val_idx] = {}
                                bicubic_performance[val_idx]['psnr'] = deepcopy(psnr_rlt[0])
                                bicubic_performance[val_idx]['ssim'] = deepcopy(ssim_rlt[0])
                                BICUBIC_EVALUATE[val_idx] = False

                            psnr_rlt_avg = {}
                            psnr_total_avg = [0., 0., 0.]
                            # 0: Init, 1: Start, 2: Final
                            #Just calculate the final value of psnr_rlt(i.e. psnr_rlt[2])
                            for k, v_init in psnr_rlt[0].items():
                                v_start = psnr_rlt[1][k]
                                v_final = psnr_rlt[2][k]
                                psnr_rlt_avg[k] = [torch.sum(v_init).cpu().item() / (v_init!=0).sum().item(),
                                                    torch.sum(v_start).cpu().item() / (v_start!=0).sum().item(),
                                                    torch.sum(v_final).cpu().item() / (v_final!=0).sum().item()]
                                for i in range(len(psnr_rlt)):
                                    psnr_total_avg[i] += psnr_rlt_avg[k][i]
                            for i in range(len(psnr_rlt)):
                                psnr_total_avg[i] /= len(psnr_rlt[0])
                            log_s = '# Validation # Final PSNR: {:.4e}:'.format(psnr_total_avg[2])
                            for k, v in psnr_rlt_avg.items():
                                log_s += ' {}: {:.4e}'.format(k, v[2])
                            logger.info(log_s)
                            if opt['use_tb_logger'] and 'debug' not in opt['name']:
                                tb_logger.add_scalar('psnr_avg_{}_0:init'.format(val_idx), psnr_total_avg[0], current_step)
                                tb_logger.add_scalar('psnr_avg_{}_1:start'.format(val_idx), psnr_total_avg[1], current_step)
                                tb_logger.add_scalar('psnr_avg_{}_2:final'.format(val_idx), psnr_total_avg[2], current_step)

                                for k, v in psnr_rlt_avg.items():
                                    tb_logger.add_scalar(k+'_psnr_avg_{}_0:init'.format(val_idx), v[0], current_step)
                                    tb_logger.add_scalar(k+'_psnr_avg_{}_1:start'.format(val_idx), v[1], current_step)
                                    tb_logger.add_scalar(k+'_psnr_avg_{}_2:final'.format(val_idx), v[2], current_step)
                            '''
                            psnr_rlt_avg = {}
                            psnr_total_avg = 0.
                            # Just calculate the final value of psnr_rlt(i.e. psnr_rlt[2])
                            for k, v in psnr_rlt[2].items():
                                psnr_rlt_avg[k] = torch.sum(v).cpu().item() / (v != 0).sum().item()
                                psnr_total_avg += psnr_rlt_avg[k]
                            psnr_total_avg /= len(psnr_rlt)
                            log_s = '# Validation # PSNR: {:.4e}:'.format(psnr_total_avg)
                            for k, v in psnr_rlt_avg.items():
                                log_s += ' {}: {:.4e}'.format(k, v)
                            logger.info(log_s)
                            if opt['use_tb_logger'] and 'debug' not in opt['name']:
                                tb_logger.add_scalar('psnr_avg', psnr_total_avg, current_step)
                                for k, v in psnr_rlt_avg.items():
                                    tb_logger.add_scalar(k, v, current_step)

                            ssim_rlt_avg = {}
                            ssim_total_avg = 0.
                            # Just calculate the final value of ssim_rlt(i.e. ssim_rlt[1])
                            for k, v in ssim_rlt[1].items():
                                ssim_rlt_avg[k] = torch.sum(v).cpu().item() / (v != 0).sum().item()
                                ssim_total_avg += ssim_rlt_avg[k]
                            ssim_total_avg /= len(ssim_rlt)
                            log_s = '# Validation # SSIM: {:.4e}:'.format(ssim_total_avg)
                            for k, v in ssim_rlt_avg.items():
                                log_s += ' {}: {:.4e}'.format(k, v)
                            logger.info(log_s)
                            if opt['use_tb_logger'] and 'debug' not in opt['name']:
                                tb_logger.add_scalar('ssim_avg', ssim_total_avg, current_step)
                                for k, v in ssim_rlt_avg.items():
                                    tb_logger.add_scalar(k, v, current_step)
                            '''
                            '''
                            pbar = util.ProgressBar(len(val_loader))
                            psnr_rlt = {}  # with border and center frames
                            psnr_rlt_avg = {}
                            psnr_total_avg = 0.
                            for val_data in val_loader:
                                folder = val_data['folder'][0]
                                idx_d = val_data['idx'].item()
                                # border = val_data['border'].item()
                                if psnr_rlt.get(folder, None) is None:
                                    psnr_rlt[folder] = []

                                model.feed_data(val_data)
                                model.test()
                                visuals = model.get_current_visuals()
                                rlt_img = util.tensor2img(visuals['rlt'])  # uint8
                                gt_img = util.tensor2img(visuals['GT'])  # uint8

                                # calculate PSNR
                                psnr = util.calculate_psnr(rlt_img, gt_img)
                                psnr_rlt[folder].append(psnr)
                                pbar.update('Test {} - {}'.format(folder, idx_d))
                            for k, v in psnr_rlt.items():
                                psnr_rlt_avg[k] = sum(v) / len(v)
                                psnr_total_avg += psnr_rlt_avg[k]
                            psnr_total_avg /= len(psnr_rlt)
                            log_s = '# Validation # PSNR: {:.4e}:'.format(psnr_total_avg)
                            for k, v in psnr_rlt_avg.items():
                                log_s += ' {}: {:.4e}'.format(k, v)
                            logger.info(log_s)
                            if opt['use_tb_logger'] and 'debug' not in opt['name']:
                                tb_logger.add_scalar('psnr_avg', psnr_total_avg, current_step)
                                for k, v in psnr_rlt_avg.items():
                                    tb_logger.add_scalar(k, v, current_step)
                            '''

            #### save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                if rank <= 0:
                    logger.info('Saving models and training states.')
                    model.save(current_step)
                    model.save_training_state(epoch, current_step, model_type='G')
                    est_model.save(current_step)
                    est_model.save_training_state(epoch, current_step, model_type='E')

    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')
        tb_logger.close()

    logger.info('Saving the final model.')
    model.save('latest')
    logger.info('End of training.')
    tb_logger.close()


if __name__ == '__main__':
    main()
