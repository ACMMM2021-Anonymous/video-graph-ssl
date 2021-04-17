# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from bisect import bisect_right
import math

import torch


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        mode = 'step',
        last_epoch=-1,
        max_epochs=100
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.mode = mode
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.max_epochs = max_epochs
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha

        if self.mode == 'step':
            import collections
            if isinstance(self.milestones, int):
                lr_factor = self.gamma ** (self.last_epoch // self.milestones)
            elif isinstance(self.milestones, collections.Iterable):
                lr_factor = self.gamma ** bisect_right(self.milestones, self.last_epoch)
        elif self.mode == 'poly':
            lr_factor = pow((1 - 1.0 * self.last_epoch / self.max_epochs), 0.9)
        elif self.mode == 'cos':
            lr_factor = 0.5 * (1. + math.cos(1.0 * self.last_epoch / self.max_epochs * math.pi))
        else:
            raise NotImplementedError(
                'currently not suported: {} scheduler'.format(self.mode))

        return [
            base_lr
            * warmup_factor
            * lr_factor
            for base_lr in self.base_lrs
        ]