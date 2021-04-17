import torch.nn as nn
from .pooling_opts import TemporalAggreModel
from .module_wrappers import TemporalGraphAug

def get_agg(agg_fun='avg', model_type='2D'):
    aggregation = TemporalAggreModel(pooling=agg_fun, model_type='2D')
    return aggregation

def build_aug_block(base_model, module_name_list, n_segments):
    for module_name in module_name_list:
        if '.' in module_name:
            module_name_splits = module_name.split('.')
            module = getattr(base_model, module_name_splits[0])
            for i in range(1, len(module_name_splits)):
                module = getattr(module, module_name_splits[i])
            else:
                module = getattr(base_model, module_name)

    in_channels = module.in_channels
    aug_model = TemporalGraphAug(in_channels=in_channels)
    new_module = nn.Sequential(aug_model, module)

    if '.' in module_name:
        module_name_splits = module_name.split('.')
        module = getattr(base_model, module_name_splits[0])
        for i in range(1, len(module_name_splits)-1):  # take care 'x.y' case 
            module = getattr(module, module_name_splits[i])
            setattr(module, module_name_splits[-1], new_module)
        else:
            setattr(base_model, module_name, new_module)

    return base_model
