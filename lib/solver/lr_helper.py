# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
from __future__ import division
import numpy as np
import math
from torch.optim.lr_scheduler import _LRScheduler

class LR_Scheduler(object):
    """Learning Rate Scheduler

    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``

    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``

    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``

    Args:
        args:
          :attr:`args.lr_scheduler` lr scheduler mode (`cos`, `poly`),
          :attr:`args.lr` base learning rate, :attr:`args.epochs` number of epochs,
          :attr:`args.lr_step`

        iters_per_epoch: number of iterations per epoch
    """
    def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0,
                 lr_step=0, warmup_epochs=0):
        self.mode = mode
        print('Using {} LR Scheduler!'.format(self.mode))
        self.lr = base_lr
        if mode == 'step':
            assert lr_step
        self.lr_step = lr_step
        self.iters_per_epoch = iters_per_epoch
        self.N = num_epochs * iters_per_epoch
        self.epoch = -1
        self.warmup_iters = warmup_epochs * iters_per_epoch

    def __call__(self, optimizer, i, epoch, best_pred):
        T = epoch * self.iters_per_epoch + i
        if self.mode == 'cos':
            lr = 0.5 * self.lr * (1 + math.cos(1.0 * T / self.N * math.pi))
        elif self.mode == 'poly':
            lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)
        elif self.mode == 'step':
            import collections
            if isinstance(self.lr_step, int):
                lr = self.lr * (0.1 ** (epoch // self.lr_step))
            elif isinstance(self.lr_step, collections.Iterable):
                lr = self.lr * (0.1 ** sum(epoch > np.array(self.lr_step)))
        else:
            raise NotImplemented
        # warm up lr schedule
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = lr * 1.0 * T / self.warmup_iters
        if epoch > self.epoch:
            print('\n=>Epoches %i, learning rate = %.4f, \
                previous best = %.4f' % (epoch, lr, best_pred))
            self.epoch = epoch
        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            # # enlarge the lr at the head
            # optimizer.param_groups[0]['lr'] = lr
            # for i in range(1, len(optimizer.param_groups)):
            #     optimizer.param_groups[i]['lr'] = lr * 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * param_group['lr_mult']
                param_group['weight_decay'] = param_group['weight_decay'] * param_group['decay_mult']

class LRScheduler(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        if 'lr_spaces' not in self.__dict__:
            raise Exception('lr_spaces must be set in "LRSchduler"')
        super(LRScheduler, self).__init__(optimizer, last_epoch)

    def get_cur_lr(self):
        return self.lr_spaces[self.last_epoch]

    def get_lr(self):
        epoch = self.last_epoch
        return [self.lr_spaces[epoch] * pg['initial_lr'] / self.start_lr for pg in self.optimizer.param_groups]

    def __repr__(self):
        return "({}) lr spaces: \n{}".format(self.__class__.__name__, self.lr_spaces)


class LogScheduler(LRScheduler):
    def __init__(self, optimizer, start_lr=0.03, end_lr=5e-4, epochs=50, last_epoch=-1, **kwargs):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.epochs = epochs
        self.lr_spaces = np.logspace(math.log10(start_lr), math.log10(end_lr), epochs)

        super(LogScheduler, self).__init__(optimizer, last_epoch)


class StepScheduler(LRScheduler):
    def __init__(self, optimizer, start_lr=0.01, end_lr=None, step=10, mult=0.1, epochs=50, last_epoch=-1, **kwargs):
        if end_lr is not None:
            if start_lr is None:
                start_lr = end_lr / (mult ** (epochs // step))
            else:  # for warm up policy
                mult = math.pow(end_lr/start_lr, 1. / (epochs // step))
        self.start_lr = start_lr
        self.lr_spaces = self.start_lr * (mult**(np.arange(epochs) // step))
        self.mult = mult
        self._step = step

        super(StepScheduler, self).__init__(optimizer, last_epoch)


class MultiStepScheduler(LRScheduler):
    def __init__(self, optimizer, start_lr=0.01, end_lr=None, steps=[10,20,30,40], mult=0.5, epochs=50, last_epoch=-1, **kwargs):
        if end_lr is not None:
            if start_lr is None:
                start_lr = end_lr / (mult ** (len(steps)))
            else:
                mult = math.pow(end_lr/start_lr, 1. / len(steps))
        self.start_lr = start_lr
        self.lr_spaces = self._build_lr(start_lr, steps, mult, epochs)
        self.mult = mult
        self.steps = steps

        super(MultiStepScheduler, self).__init__(optimizer, last_epoch)

    def _build_lr(self, start_lr, steps, mult, epochs):
        lr = [0] * epochs
        lr[0] = start_lr
        for i in range(1, epochs):
            lr[i] = lr[i-1]
            if i in steps:
                lr[i] *= mult
        return np.array(lr, dtype=np.float32)


class LinearStepScheduler(LRScheduler):
    def __init__(self, optimizer, start_lr=0.01, end_lr=0.005, epochs=50, last_epoch=-1, **kwargs):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.lr_spaces = np.linspace(start_lr, end_lr, epochs)

        super(LinearStepScheduler, self).__init__(optimizer, last_epoch)


class CosStepScheduler(LRScheduler):
    def __init__(self, optimizer, start_lr=0.01, end_lr=0.005, epochs=50, last_epoch=-1, **kwargs):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.lr_spaces = self._build_lr(start_lr, end_lr, epochs)

        super(CosStepScheduler, self).__init__(optimizer, last_epoch)

    def _build_lr(self, start_lr, end_lr, epochs):
        index = np.arange(epochs).astype(np.float32)
        lr = end_lr + (start_lr - end_lr) * (1. + np.cos(index * np.pi/ epochs)) * 0.5
        return lr.astype(np.float32)


class WarmUPScheduler(LRScheduler):
    def __init__(self, optimizer, warmup, normal, epochs=50, last_epoch=-1):
        warmup = warmup.lr_spaces # [::-1]
        normal = normal.lr_spaces
        self.lr_spaces = np.concatenate([warmup, normal])
        self.start_lr = normal[0]

        super(WarmUPScheduler, self).__init__(optimizer, last_epoch)


LRs = {
    'log': LogScheduler,
    'step': StepScheduler,
    'multi-step': MultiStepScheduler,
    'linear': LinearStepScheduler,
    'cos': CosStepScheduler}


def _build_lr_scheduler(optimizer, cfg, epochs=50, last_epoch=-1):
    if 'type' not in cfg:
        # return LogScheduler(optimizer, last_epoch=last_epoch, epochs=epochs)
        cfg['type'] = 'log'

    if cfg['type'] not in LRs:
        raise Exception('Unknown type of LR Scheduler "%s"'%cfg['type'])

    return LRs[cfg['type']](optimizer, last_epoch=last_epoch, epochs=epochs, **cfg)


def _build_warm_up_scheduler(optimizer, cfg, epochs=50, last_epoch=-1):
    warmup_epoch = cfg['warmup']['epoch']
    sc1 = _build_lr_scheduler(optimizer, cfg['warmup'], warmup_epoch, last_epoch)
    sc2 = _build_lr_scheduler(optimizer, cfg, epochs - warmup_epoch, last_epoch)
    return WarmUPScheduler(optimizer, sc1, sc2, epochs, last_epoch)


def build_lr_scheduler(optimizer, cfg, epochs=50, last_epoch=-1):
    if 'warmup' in cfg:
        return _build_warm_up_scheduler(optimizer, cfg, epochs, last_epoch)
    else:
        return _build_lr_scheduler(optimizer, cfg, epochs, last_epoch)


if __name__ == '__main__':
    import torch.nn as nn
    from torch.optim import SGD

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(10, 10, kernel_size=3)
    net = Net().parameters()
    optimizer = SGD(net, lr=0.01)

    # test1
    step = {
            'type': 'step',
            'start_lr': 0.01,
            'step': 10,
            'mult': 0.1
            }
    lr = build_lr_scheduler(optimizer, step)
    print(lr)

    log = {
            'type': 'log',
            'start_lr': 0.03,
            'end_lr': 5e-4,
            }
    lr = build_lr_scheduler(optimizer, log)

    print(lr)

    log = {
            'type': 'multi-step',
            "start_lr": 0.01,
            "mult": 0.1,
            "steps": [10, 15, 20]
            }
    lr = build_lr_scheduler(optimizer, log)
    print(lr)

    cos = {
            "type": 'cos',
            'start_lr': 0.01,
            'end_lr': 0.0005,
            }
    lr = build_lr_scheduler(optimizer, cos)
    print(lr)

    step = {
            'type': 'step',
            'start_lr': 0.001,
            'end_lr': 0.03,
            'step': 1,
            }

    warmup = log.copy()
    warmup['warmup'] = step
    warmup['warmup']['epoch'] = 5
    lr = build_lr_scheduler(optimizer, warmup, epochs=55)
    print(lr)

    lr.step()
    print(lr.last_epoch)

    lr.step(5)
    print(lr.last_epoch)


