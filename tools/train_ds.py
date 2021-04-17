import argparse
import time
from tqdm import tqdm
import os
import sys

import torch
import numpy as np
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_

sys.path.append('.')
from lib.config import cfg
from lib.modeling import create_video_model
from lib.solver import make_optimizer, make_lr_scheduler
from lib.data import make_data_loader
from lib.utils import creat_saver, creat_criterion
from lib.evaluation import AverageMeter, accuracy



class Trainer(object):

    def __init__(self, cfg):
        # -------------------------------------------------------------------------
        # basic argguments
        # -------------------------------------------------------------------------
        self.cfg = cfg
        self.best_pred = 0.0

        # -------------------------------------------------------------------------
        # logger
        # -------------------------------------------------------------------------
        self.saver, self.writer = creat_saver(cfg)

        # -------------------------------------------------------------------------
        # model, loss function and optimizer
        # -------------------------------------------------------------------------
        self.model = create_video_model(cfg)
        self.optimizer = make_optimizer(cfg, self.model)
        self._parse_model()
        self.criterion = creat_criterion(cfg)
        # -------------------------------------------------------------------------
        # basic parameters
        # -------------------------------------------------------------------------

        # -------------------------------------------------------------------------
        # train and validation dataset
        # -------------------------------------------------------------------------
        self.train_loader, self.val_loader = make_data_loader(cfg)

        # -------------------------------------------------------------------------
        # evaluator and lr schedular
        # -------------------------------------------------------------------------
        self.batch_time, self.data_time = AverageMeter(), AverageMeter()
        self.top1, self.top5 = AverageMeter(), AverageMeter()
        self.losses = AverageMeter()
        self.scheduler = make_lr_scheduler(cfg, self.optimizer)



    def _parse_model(self):
        """
        a simpler wrapper that creates the train/test models
        """

        if self.cfg.CHECKPOINT.RESUME != 'none':
            if not os.path.isfile(self.cfg.CHECKPOINT.RESUME):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(self.cfg.CHECKPOINT.RESUME))

            checkpoint = torch.load(self.cfg.CHECKPOINT.RESUME, map_location='cpu')
            check_state = checkpoint['state_dict']
            state = {k: v for k, (_, v) in zip(self.model.state_dict().keys(), check_state.items()) if 'new_fc' not in k}
            model_state = self.model.state_dict()
            model_state.update(state)
            self.model.load_state_dict(model_state)

        if self.cfg.MODEL.LINEAR_PROBE:
            for name, parameter in self.model.named_parameters():
                if 'new_fc' not in name:
                    parameter.requires_grad = False

        if self.cfg.MODEL.DEVICE == 'cuda':
            self.model = torch.nn.DataParallel(self.model).cuda()


    def train(self, epoch):
        self.batch_time.reset()
        self.data_time.reset()
        self.losses.reset()
        self.top1.reset()
        self.top5.reset()


        # switch to train mode
        self.model.train()

        end = time.time()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, (image, target) in enumerate(tbar):
             # measure data loading time
            self.data_time.update(time.time() - end)

            if self.cfg.MODEL.DEVICE == 'cuda':
                image, target = image.cuda(), target.cuda()
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)
            loss.backward()
            if self.cfg.SOLVER.CLIP_GRADIENT != 'none':
                total_norm = clip_grad_norm_(self.model.parameters(), self.cfg.SOLVER.CLIP_GRADIENT)
                if total_norm > self.cfg.SOLVER.CLIP_GRADIENT:
                    print("clipping gradient: {} with coef {}".format(total_norm, self.cfg.SOLVER.CLIP_GRADIENT / total_norm))
            self.optimizer.step()

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.detach(), target.detach(), topk=(1,5))
            self.losses.update(loss.item(), image.size(0))
            self.top1.update(prec1.item(), image.size(0))
            self.top5.update(prec5.item(), image.size(0))

            # report
            tbar.set_description(f't (l={self.losses.avg:.4f})(top1={self.top1.avg:.4f})')
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # measure elapsed time
            self.batch_time.update(time.time() - end)
            end = time.time()

            if i % self.cfg.CHECKPOINT.PRINT_FREQ == 0:
                print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(self.train_loader), batch_time=self.batch_time,
                    data_time=self.data_time, loss=self.losses, top1=self.top1, top5=self.top5, lr=self.optimizer.param_groups[-1]['lr'])))
        
        self.writer.add_scalar('train/total_loss_epoch', self.losses.sum, epoch)
        self.writer.add_scalar('train/loss', self.losses.avg, epoch)
        self.writer.add_scalar('train/top1_acc', self.top1.avg, epoch)
        self.writer.add_scalar('train/top5_acc', self.top5.avg, epoch)
        self.writer.add_scalar('lr', self.optimizer.param_groups[-1]['lr'], epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.cfg.DATALOADER.BATCH_SIZE + image.data.shape[0]))
        print('Loss: %.3f' % self.losses.sum)

        if self.cfg.CHECKPOINT.NO_VAL and ((epoch+1) % self.cfg.CHECKPOINT.CHECK_INTERVAL == 0 or epoch == cfg.SOLVER.MAX_EPOCHS-1):
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

        

    def validation(self, epoch):
        self.batch_time.reset()
        self.losses.reset()
        self.top1.reset()
        self.top5.reset()

        # switch to evaluation mode
        self.model.eval()

        end = time.time()
        tbar = tqdm(self.val_loader, desc='\r')
        for i, (image, target) in enumerate(tbar):
            if self.cfg.MODEL.DEVICE == 'cuda':
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.detach(), target.detach(), topk=(1,5))
            self.losses.update(loss.item(), image.size(0))
            self.top1.update(prec1.item(), image.size(0))
            self.top5.update(prec5.item(), image.size(0))
            tbar.set_description(f't (l={self.losses.avg:.4f})(a1={self.top1.avg:.4f})')

            # measure elapsed time
            self.batch_time.update(time.time() - end)
            end = time.time()

            if i % self.cfg.CHECKPOINT.PRINT_FREQ == 0:
                print(('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(self.val_loader), batch_time=self.batch_time, loss=self.losses,
                    top1=self.top1, top5=self.top5)))

        # Fast test during the training
        self.writer.add_scalar('val/total_loss_epoch', self.losses.sum, epoch)
        self.writer.add_scalar('val/Top_1_acc', self.top1.avg, epoch)
        self.writer.add_scalar('val/Top_5_acc', self.top5.avg, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.cfg.TEST.BATCH_SIZE + image.data.shape[0]))
        print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
            .format(top1=self.top1, top5=self.top5, loss=self.losses)))


        new_pred = self.top1.avg
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

def main():
    parser = argparse.ArgumentParser(description='Classification model training')
    parser.add_argument('--config_file', type=str, default=None, required=True,
                        help='Optional config file for params')
    parser.add_argument('opts', help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_IDS    # new add by gpu
        cudnn.benchmark = True

    print(cfg)
    torch.manual_seed(cfg.MODEL.SEED)
    trainer = Trainer(cfg)
    print('Starting Epoch:', trainer.cfg.SOLVER.START_EPOCH)
    print('Total Epochs:', trainer.cfg.SOLVER.MAX_EPOCHS)
    for epoch in range(trainer.cfg.SOLVER.START_EPOCH, trainer.cfg.SOLVER.MAX_EPOCHS):
        # -------------------------------------------------------------------------
        # training
        # -------------------------------------------------------------------------
        trainer.train(epoch)
        # adjust learning rate
        trainer.scheduler.step()
        # -------------------------------------------------------------------------
        # validation
        # -------------------------------------------------------------------------
        if not trainer.cfg.CHECKPOINT.NO_VAL and ((epoch+1) % cfg.CHECKPOINT.EVAL_INTERVAL == 0 or epoch == cfg.SOLVER.MAX_EPOCHS-1):
            trainer.validation(epoch)

    trainer.writer.close()



if __name__ == '__main__':
    main()
