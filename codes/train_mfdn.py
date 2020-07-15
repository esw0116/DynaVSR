import os
import math
import argparse
import random
import logging
import imageio

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler

import options.options as option
from utils import util
from data.meta_learner import loader, create_dataloader, create_dataset
from data import util as data_util
from models import create_model
import utility

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
    parser.add_argument('--exp_name', type=str, default='temp')
    args = parser.parse_args()
    if args.exp_name == 'temp':
        opt = option.parse(args.opt, is_train=True)
    else:
        opt = option.parse(args.opt, is_train=True, exp_name=args.exp_name)

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
            tb_logger = SummaryWriter(log_dir='../tb_logger/' + opt['name'])
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    '''
    train_loader = loader.get_loader(opt, train=True)
    val_loader = loader.get_loader(opt, train=False)
    # train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
    total_iters = int(opt['train']['niter'])
    total_epochs = int(opt['train']['epochs'])
    '''
    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            # train_set = create_dataset(dataset_opt)
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
                                                  model_name=opt['network_E']['which_model_E'],
                                                  idx=i))
                    val_loader.append(create_dataloader(val_set[-1], dataset_opt, opt, None))
            else:
                valname_list = opt['datasets']['val']['name']
                val_set = create_dataset(dataset_opt, scale=opt['scale'],
                                         kernel_size=opt['datasets']['train']['kernel_size'],
                                         model_name=opt['network_E']['which_model_E'])
                # val_set = loader.get_dataset(opt, train=False)
                val_loader = create_dataloader(val_set, dataset_opt, opt, None)
                '''
                valname_list = [opt['datasets']['val']['name']]
                val_set = [create_dataset(dataset_opt, scale=opt['scale'],
                                         kernel_size=opt['datasets']['train']['kernel_size'],
                                         model_name=opt['network_E']['which_model_E'])]
                # val_set = loader.get_dataset(opt, train=False)
                val_loader = [create_dataloader(val_set[0], dataset_opt, opt, None)]
                '''
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    #### create model
    model = create_model(opt)

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    #### training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs + 1):
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break

            #### training
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            #### update learning rate
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

            #### log
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '[epoch:{:3d}, iter:{:8,d}, lr:('.format(epoch, current_step)
                for v in model.get_current_learning_rate():
                    message += '{:.3e},'.format(v)
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
            if opt['datasets'].get('val', None) and current_step % opt['train']['val_freq'] == 0:
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
                else:  # video restoration validation
                    if opt['dist']:
                        # multi-GPU testing
                        if isinstance(val_loader, list):
                            for valname, val_set_frag in zip(valname_list, val_set):
                                psnr_rlt = {}  # with border and center frames
                                if not os.path.exists(os.path.join('../results', opt['name'], valname)):
                                    os.makedirs(os.path.join('../results', opt['name'], valname))
                                if rank == 0:
                                    pbar = util.ProgressBar(len(val_set_frag))
                                for idx in range(rank, len(val_set_frag), world_size):

                                    val_data = val_set_frag[idx]
                                    val_data['LQs'].unsqueeze_(0)
                                    val_data['SuperLQs'].unsqueeze_(0)
                                    folder = val_data['folder']
                                    idx_d, max_idx = val_data['idx'].split('/')
                                    idx_d, max_idx = int(idx_d), int(max_idx)
                                    if psnr_rlt.get(folder, None) is None:
                                        psnr_rlt[folder] = torch.zeros(max_idx, dtype=torch.float32, device='cuda')

                                    model.feed_data(val_data)
                                    model.test()

                                    visuals = model.get_current_visuals()
                                    rlt_img = util.tensor2img(visuals['rlt'], mode='rgb')  # uint8 RGB--> RGB
                                    gt_img = util.tensor2img(visuals['LQ'], mode='rgb')  # uint8 --> RGB
                                    hr_img = util.tensor2img(visuals['GT'], mode='rgb')  # uint8 --> BGR

                                    save_img_path = os.path.join('../results', opt['name'], valname, folder)
                                    if not os.path.exists(save_img_path):
                                        os.makedirs(save_img_path)
                                    imageio.imwrite(os.path.join(save_img_path, '{:04d}_est.png'.format(idx_d)), rlt_img)
                                    imageio.imwrite(os.path.join(save_img_path, '{:04d}_gt.png'.format(idx_d)), gt_img)
                                    imageio.imwrite(os.path.join(save_img_path, '{:04d}_hr.png'.format(idx_d)), hr_img)

                                    # Convert to Ycbcr
                                    # rlt_img = rlt_img.astype('float32') / 255
                                    # gt_img = gt_img.astype('float32') / 255

                                    # rlt_img = data_util.bgr2ycbcr(rlt_img, only_y=True)
                                    # gt_img = data_util.bgr2ycbcr(gt_img, only_y=True)

                                    # calculate PSNR
                                    psnr_rlt[folder][idx_d] = util.calculate_psnr(rlt_img, gt_img)
                                    if rank == 0:
                                        for _ in range(world_size):
                                            pbar.update('Test {} - {}/{} - PSNR: {:.3f}'
                                                        .format(folder, idx_d, max_idx, psnr_rlt[folder][idx_d].item()))

                                ## collect data
                                for _, v in psnr_rlt.items():
                                    dist.reduce(v, 0)
                                dist.barrier()

                                if rank == 0:
                                    psnr_rlt_avg = {}
                                    psnr_total_avg = 0.
                                    for k, v in psnr_rlt.items():
                                        psnr_rlt_avg[k] = torch.mean(v).cpu().item()
                                        psnr_total_avg += psnr_rlt_avg[k]
                                    psnr_total_avg /= len(psnr_rlt)
                                    log_s = '# Validation # PSNR: {:.4e}'.format(psnr_total_avg)
                                    for k, v in psnr_rlt_avg.items():
                                        log_s += ' {}: {:.4e}'.format(k, v)
                                    logger.info(log_s)

                                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                                        tb_logger.add_scalar('psnr_avg', psnr_total_avg, current_step)
                                        for k, v in psnr_rlt_avg.items():
                                            tb_logger.add_scalar(k, v, current_step)


                        else:
                            psnr_rlt = {}  # with border and center frames
                            selected_srmodel_idx = []
                            if rank == 0:
                                pbar = util.ProgressBar(len(val_set))
                            for idx in range(rank, len(val_set), world_size):
                                val_data = val_set[idx]
                                valname = opt['datasets']['val']['name']
                                val_data['LQs'].unsqueeze_(0)
                                val_data['SuperLQs'].unsqueeze_(0)
                                folder = val_data['folder']
                                idx_d, max_idx = val_data['idx'].split('/')
                                idx_d, max_idx = int(idx_d), int(max_idx)
                                if psnr_rlt.get(folder, None) is None:
                                    psnr_rlt[folder] = torch.zeros(max_idx, dtype=torch.float32, device='cuda')

                                model.feed_data(val_data)
                                model.test()

                                visuals = model.get_current_visuals()
                                rlt_img = util.tensor2img(visuals['rlt'])  # uint8 RGB--> BGR
                                gt_img = util.tensor2img(visuals['LQ'])  # uint8 --> BGR
                                if not os.path.exists(os.path.join('../results', opt['name'], valname, folder)):
                                    os.makedirs(os.path.join('../results', opt['name'], valname, folder))
                                imageio.imwrite(os.path.join('../results', opt['name'], valname, folder,
                                                             '{:04d}_est.png'.format(idx_d)),
                                                rlt_img[:, :, ::-1])
                                imageio.imwrite(os.path.join('../results', opt['name'], valname, folder,
                                                             '{:04d}_gt.png'.format(idx_d)),
                                                gt_img[:, :, ::-1])

                                # Convert to Ycbcr
                                # rlt_img = rlt_img.astype('float32') / 255
                                # gt_img = gt_img.astype('float32') / 255

                                # rlt_img = data_util.bgr2ycbcr(rlt_img, only_y=True)
                                # gt_img = data_util.bgr2ycbcr(gt_img, only_y=True)

                                # calculate PSNR
                                psnr_rlt[folder][idx_d] = util.calculate_psnr(rlt_img, gt_img)
                                '''
                                #Write dataframe
                                name = val_data['folder'] + ' ' + val_data['idx']
                                if name in df.index:
                                    df.at[name, 'GT_Type'] = val_data['Kernel_type'].item()
                                    df.at[name, 'GT_sigma_x'] = val_data['Kernel_args'][0].item()
                                    df.at[name, 'GT_sigma_y'] = val_data['Kernel_args'][1].item()
                                    df.at[name, 'GT_theta'] = val_data['Kernel_args'][2].item()
                                    df.at[name, 'EST_Type'] = model.class_pred.item()
                                    df.at[name, 'EST_sigma_x'] = model.fake_kernel_args[0].item()
                                    df.at[name, 'EST_sigma_y'] = model.fake_kernel_args[1].item()
                                    df.at[name, 'EST_theta'] = model.fake_kernel_args[2].item()
                                else:
                                    df.loc[name] = [val_data['Kernel_type'].item(), val_data['Kernel_args'][0].item(),
                                                    val_data['Kernel_args'][1].item(), val_data['Kernel_args'][2].item(),
                                                    model.class_pred.item(), model.fake_kernel_args[0].item(),
                                                    model.fake_kernel_args[1].item(), model.fake_kernel_args[2].item()]
                                '''
                                if rank == 0:
                                    for _ in range(world_size):
                                        pbar.update('Test {} - {}/{} - PSNR: {:.3f}'
                                                    .format(folder, idx_d, max_idx, psnr_rlt[folder][idx_d].item()))

                                ## collect data
                            for _, v in psnr_rlt.items():
                                dist.reduce(v, 0)
                            dist.barrier()

                            if rank == 0:
                                psnr_rlt_avg = {}
                                psnr_total_avg = 0.
                                for k, v in psnr_rlt.items():
                                    psnr_rlt_avg[k] = torch.mean(v).cpu().item()
                                    psnr_total_avg += psnr_rlt_avg[k]
                                psnr_total_avg /= len(psnr_rlt)
                                log_s = '# Validation # PSNR: {:.4e}'.format(psnr_total_avg)
                                for k, v in psnr_rlt_avg.items():
                                    log_s += ' {}: {:.4e}'.format(k, v)
                                logger.info(log_s)

                                if opt['use_tb_logger'] and 'debug' not in opt['name']:
                                    tb_logger.add_scalar('psnr_avg', psnr_total_avg, current_step)
                                    for k, v in psnr_rlt_avg.items():
                                        tb_logger.add_scalar(k, v, current_step)

                    else:
                        pbar = util.ProgressBar(len(val_loader))
                        psnr_rlt = {}  # with border and center frames
                        psnr_rlt_avg = {}
                        psnr_total_avg = 0.
                        for val_data in val_loader:
                            folder = val_data['folder'][0]
                            idx_d = val_data['idx'][0]
                            # border = val_data['border'].item()
                            if psnr_rlt.get(folder, None) is None:
                                psnr_rlt[folder] = []

                            # print(val_data['name'][0][0].split('/')[0])
                            total_psnr = 0
                            video_length = val_data['LQs'].size(1)
                            for i in range(video_length):
                                val_seg = {}
                                select_idx = data_util.index_generation(i, video_length, opt['datasets']['train']['N_frames'])
                                val_seg['LQs'] = val_data['LQs'][:, select_idx]
                                val_seg['GT'] = val_data['GT'] # [:, i]
                                model.feed_data(val_seg)
                                model.test()
                                visuals = model.get_current_visuals()
                                rlt_img = util.tensor2img(visuals['rlt'])  # uint8
                                gt_img = util.tensor2img(visuals['GT'])  # uint8
                                # calculate PSNR
                                psnr, _ = utility.calc_psnr(visuals['rlt'], visuals['GT'])
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

            #### save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                if rank <= 0:
                    logger.info('Saving models and training states.')
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')
        tb_logger.close()


if __name__ == '__main__':
    main()
