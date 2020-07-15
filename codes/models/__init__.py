import logging
logger = logging.getLogger('base')


def create_model(opt):
    models = opt['model']

    def _create(model):
        # image restoration
        if model == 'sr':  # PSNR-oriented super resolution
            from .SR_model import SRModel as M
        elif model == 'srgan':  # GAN-based super resolution, SRGAN / ESRGAN
            from .SRGAN_model import SRGANModel as M
        # video restoration
        elif model == 'video_base':
            from .Video_base_model import VideoBaseModel as M
        elif model == 'classifier':
            from .Classifier_model import Classifier_Model as M
        elif model == 'estimator':
            from .Kernelestimator_model import Kernelestimator_Model as M
        elif model == 'lrimgestimator':
            from .LRestimator_model import LRimgestimator_Model as M
        else:
            raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
        m = M(opt)
        logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))

        return m

    if '+' in models:
        model_name_list = models.split('+')
        model_list = []
        for model in model_name_list:
            model_list.append(_create(model))
        return model_list
    else:
        return _create(models)
