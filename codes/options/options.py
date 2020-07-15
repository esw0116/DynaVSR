import os
import os.path as osp
import math
import logging
import yaml
from collections import OrderedDict
from utils.util import OrderedYaml
Loader, Dumper = OrderedYaml()


def parse(opt_path, is_train=True, exp_name=None):
    with open(opt_path, mode='r') as f:
        opt = yaml.load(f, Loader=Loader)
    # export CUDA_VISIBLE_DEVICES
    # gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    # os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    # print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
    if exp_name is not None:
        opt['name'] = exp_name
    else:
        '''
        inner_loop_name = opt['train']['maml']['optimizer'][0] + str(opt['train']['maml']['adapt_iter']) + str(math.floor(math.log10(opt['train']['maml']['lr_alpha'])))
        meta_loop_name = opt['train']['optim'][0] + str(opt['train']['lr_G'])[0] + str(math.floor(math.log10(opt['train']['lr_G'])))
        if opt['datasets']['val']['degradation_mode'] == 'set':
            degradation_name = str(opt['datasets']['val']['degradation_type'])\
                    + '_' + str(opt['datasets']['val']['sigma_x']) \
                    + '_' + str(opt['datasets']['val']['sigma_y'])\
                    + '_' + str(opt['datasets']['val']['theta'])
        else:
            degradation_name = opt['datasets']['val']['degradation_mode']
        patch_name = 'p{}x{}'.format(opt['train']['maml']['patch_size'], opt['train']['maml']['num_patch']) if opt['train']['maml']['use_patch'] else 'full'
        use_real_flag = '_ideal' if opt['train']['use_real'] else ''
        opt['name'] = opt['name'] + '_' + inner_loop_name + meta_loop_name + '_' + degradation_name + '_' + patch_name + use_real_flag
        '''
        pass
    opt['is_train'] = is_train
    if opt['distortion'] == 'sr':
        scale = opt['scale']

    # datasets
    for phase, dataset in opt['datasets'].items():
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        if opt['distortion'] == 'sr':
            dataset['scale'] = scale
        is_lmdb = False
        if dataset.get('dataroot_GT', None) is not None:
            dataset['dataroot_GT'] = osp.expanduser(dataset['dataroot_GT'])
            if dataset['dataroot_GT'].endswith('lmdb'):
                is_lmdb = True
        if dataset.get('dataroot_LQ', None) is not None:
            dataset['dataroot_LQ'] = osp.expanduser(dataset['dataroot_LQ'])
            if dataset['dataroot_LQ'].endswith('lmdb'):
                is_lmdb = True
        dataset['data_type'] = 'lmdb' if is_lmdb else 'img'
        if dataset['mode'].endswith('mc'):  # for memcached
            dataset['data_type'] = 'mc'
            dataset['mode'] = dataset['mode'].replace('_mc', '')

    # path
    for key, path in opt['path'].items():
        if path and key in opt['path'] and key != 'strict_load':
            if isinstance(path, OrderedDict):
                for subkey, subpath in opt['path'][key].items():
                    if subpath and subkey in opt['path'][key]:
                        opt['path'][key][subkey] = osp.expanduser(subpath)
            else:
                opt['path'][key] = osp.expanduser(path)
    opt['path']['root'] = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
    if is_train:
        experiments_root = osp.join(opt['path']['root'], 'experiments', opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['training_state'] = osp.join(experiments_root, 'training_state')
        opt['path']['log'] = experiments_root
        opt['path']['val_images'] = osp.join(experiments_root, 'val_images')

        # change some options for debug mode
        if 'debug' in opt['name']:
            opt['train']['val_freq'] = 8
            opt['logger']['print_freq'] = 1
            opt['logger']['save_checkpoint_freq'] = 8
    else:  # test
        results_root = osp.join(opt['path']['root'], 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root

    # network
    if opt['distortion'] == 'sr':
        opt['network_G']['scale'] = scale

    return opt


def dict2str(opt, indent_l=1):
    '''dict to string for logger'''
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


class NoneDict(dict):
    def __missing__(self, key):
        return None


# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


def check_resume(opt, resume_iter):
    '''Check resume states and pretrain_model paths'''
    logger = logging.getLogger('base')
    if opt['path']['resume_state']:
        if opt['path'].get('pretrain_model_G', None) is not None or opt['path'].get(
                'pretrain_model_D', None) is not None:
            logger.warning('pretrain_model path will be ignored when resuming training.')

        opt['path']['pretrain_model_G'] = osp.join(opt['path']['models'],
                                                   '{}_G.pth'.format(resume_iter))
        logger.info('Set [pretrain_model_G] to ' + opt['path']['pretrain_model_G'])
        if 'gan' in opt['model']:
            opt['path']['pretrain_model_D'] = osp.join(opt['path']['models'],
                                                       '{}_D.pth'.format(resume_iter))
            logger.info('Set [pretrain_model_D] to ' + opt['path']['pretrain_model_D'])
