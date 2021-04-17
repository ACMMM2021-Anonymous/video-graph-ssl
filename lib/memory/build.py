from .mem_bank import RGBMem, CMCMem
from .mem_moco import RGBMoCo, CMCMoCo
from .criterion import NCECriterion, NCESoftmaxLoss, D

def create_contrast(cfg, n_data):
    if cfg.CONTRAST.MEM_TYPE == 'bank':
        mem_func = RGBMem if cfg.CROSS.MODALITY == 'visual' else CMCMem
        memory = mem_func(cfg.CROSS.FEAT_DIM, n_data,
                          cfg.CONTRAST.NCE_K, cfg.CONTRAST.NCE_T, cfg.CONTRAST.NCE_M)
    elif cfg.CONTRAST.MEM_TYPE == 'moco':
        mem_func = RGBMoCo if cfg.CROSS.MODALITY == 'visual' else CMCMoCo
        memory = mem_func(cfg.CROSS.FEAT_DIM, cfg.CONTRAST.NCE_K, cfg.CONTRAST.NCE_T)
    elif cfg.CONTRAST.MEM_TYPE == 'simsiam':
        memory = None
    else:
        raise NotImplementedError(
            'mem not suported: {}'.format(cfg.CONTRAST.MEM_TYPE))

    return memory

def create_criterion(cfg, n_data):
    if cfg.CROSS.CRITERION == 'crossentropy':
        criterion = NCESoftmaxLoss()
    elif cfg.CROSS.CRITERION == 'NCE':
        criterion = NCECriterion(n_data)
    elif cfg.CROSS.CRITERION == 'simsiam_d':
        criterion = D()
    else:
        raise NotImplementedError(
            'criterion not suported: {}'.format(cfg.cfg.CROSS.CRITERION))
    
    return criterion
