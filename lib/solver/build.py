import torch 

from .lr_helper import LR_Scheduler
from .lr_scheduler import WarmupMultiStepLR

def create_optimizer(cfg, model):
    if cfg.SOLVER.USE_TRICK:
        train_params = model.get_optim_policies()
        for group in train_params:
            print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
                group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))
    else:
        train_params = model.parameters()

    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(train_params, cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY, 
                            momentum=cfg.SOLVER.MOMENTUM, nesterov=cfg.SOLVER.NESTEROV)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(train_params, cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    return optimizer



def make_optimizer(cfg, model):

    if cfg.SOLVER.USE_TRICK:
        train_params = model.get_optim_policies()
        for group in train_params:
            print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
                group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))
    else:
        train_params = []

    lr = cfg.SOLVER.BASE_LR
    weight_decay = cfg.SOLVER.WEIGHT_DECAY
    if len(train_params) == 0:
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "bias" in key:
                lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
            train_params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    else:
        for param_group in train_params:
            for param in param_group['params']:
                if not param.requires_grad:
                    continue
            param_group['lr'] = lr * param_group['lr_mult']
            param_group['weight_decay'] = weight_decay * param_group['decay_mult']

    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(train_params, momentum=cfg.SOLVER.MOMENTUM, nesterov=cfg.SOLVER.NESTEROV)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(train_params, cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)

    return optimizer


def make_lr_scheduler(cfg, optimizer):
    return WarmupMultiStepLR(
        optimizer=optimizer,
        milestones=cfg.SOLVER.STEPS,
        gamma=cfg.SOLVER.GAMMA,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD,
        mode=cfg.SOLVER.LR_SCHEDULER,
        max_epochs=cfg.SOLVER.MAX_EPOCHS,
    )