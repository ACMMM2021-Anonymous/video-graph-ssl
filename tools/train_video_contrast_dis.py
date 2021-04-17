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

from apex import amp
from apex.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append('.')
from lib.config import cfg
from lib.modeling import create_visual_model
from lib.memory import create_contrast, create_criterion
from lib.solver import make_optimizer, make_lr_scheduler
from lib.data import build_video_contrastive_loader 
from lib.utils import creat_saver
from lib.evaluation import AverageMeter, accuracy



class Trainer(object):

    def __init__(self, cfg, args):
        # -------------------------------------------------------------------------
        # basic argguments
        # -------------------------------------------------------------------------
        self.cfg = cfg
        self.args = args
        self.best_pred = 0.0

    def init_ddp_enviroment(self, gpu, ngpus_per_node):
        self.args.gpu = gpu
        self.args.ngpus_per_node = ngpus_per_node
        self.args.node_rank = self.args.rank
        self.args.local_rank = gpu
        self.args.local_center = self.args.rank * ngpus_per_node

        torch.cuda.set_device(gpu)
        cudnn.benchmark = True

        if self.args.gpu is not None:
            print("Use GPU: {} for training".format(self.args.gpu))

        #if self.args.distributed:   
        self.args.rank = self.args.rank * ngpus_per_node + gpu
        os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
        dist.init_process_group(
            backend='nccl', init_method=self.args.dist_url,
            world_size=self.args.world_size, rank=self.args.rank)

        # setup local group on each node, for ShuffleBN
        local_groups = []
        for i in range(0, self.args.world_size // ngpus_per_node):
            gp = torch.distributed.new_group(
                ranks=list(range(i * ngpus_per_node, (i + 1) * ngpus_per_node)),
                backend='nccl')
            local_groups.append(gp)

        local_group = local_groups[self.args.rank // ngpus_per_node]
        if self.args.local_rank == 0:
            print("node_rank:", self.args.node_rank)
            print("local_center:", self.args.local_center)
            print("local group size:", dist.get_world_size(local_group))

        self.local_group = local_group

    def parse_trainer(self):
        # -------------------------------------------------------------------------
        # logger
        # -------------------------------------------------------------------------
        if self.args.local_rank == 0:
            self.saver, self.writer = creat_saver(self.cfg)

        # -------------------------------------------------------------------------
        # distributed training prepare
        # -------------------------------------------------------------------------
        # self.init_ddp_enviroment()

        # -------------------------------------------------------------------------
        # model, loss function and optimizer
        # -------------------------------------------------------------------------
        
        self.model, self.model_ema = create_visual_model(self.cfg)
        self.optimizer = make_optimizer(self.cfg, self.model)
        self._parse_model()
        # -------------------------------------------------------------------------
        # basic parameters
        # -------------------------------------------------------------------------

        # -------------------------------------------------------------------------
        # train and validation dataset
        # -------------------------------------------------------------------------
        self.train_loader, self.train_sampler, n_data = build_video_contrastive_loader(self.cfg)
        self.contrast = create_contrast(self.cfg, n_data)
        if self.contrast is not None:
            self.contrast = self.contrast.cuda()
        self.criterion = create_criterion(self.cfg, n_data)
        self.criterion = self.criterion.cuda()

        # -------------------------------------------------------------------------
        # evaluator and lr schedular
        # -------------------------------------------------------------------------
        self.batch_time, self.data_time = AverageMeter(), AverageMeter()
        self.top1, self.top5 = AverageMeter(), AverageMeter()
        self.losses = AverageMeter()
        self.scheduler = make_lr_scheduler(self.cfg, self.optimizer)

        # optional step: synchronize memory
        if self.contrast is not None:
            self._broadcast_memory(self.contrast)

    def _wrap_up(self):
        """
        Wrap up models with apex and DDP
        """
        args = self.args

        self.model.cuda(args.gpu)
        if isinstance(self.model_ema, torch.nn.Module):
            self.model_ema.cuda(args.gpu)

        # to amp model if needed
        if self.cfg.APEX.FLAG:
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level=self.cfg.APEX.OPT_LEVEL
            )
            if isinstance(self.model_ema, torch.nn.Module):
                self.model_ema = amp.initialize(
                    self.model_ema, opt_level=self.cfg.APEX.OPT_LEVEL
                )
        # to distributed data parallel
        self.model = DDP(self.model, device_ids=[args.gpu])

        if isinstance(self.model_ema, torch.nn.Module):
            self._momentum_update(self.model.module, self.model_ema, 0)

    def _resume(self):
        if self.cfg.CHECKPOINT.RESUME != 'none':
            if not os.path.isfile(self.cfg.CHECKPOINT.RESUME):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(self.cfg.CHECKPOINT.RESUME))
            else: 
                print("=> loading checkpoint '{}'".format(self.cfg.CHECKPOINT.RESUME))
            checkpoint = torch.load(self.cfg.CHECKPOINT.RESUME, map_location='cpu')
            self.cfg.SOLVER.START_EPOCH = checkpoint['epoch']
            if self.cfg.MODEL.DEVICE == 'cuda':
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not self.cfg.CHECKPOINT.FINETUNE:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                if self.contrast is not None:
                    self.contrast.load_state_dict(checkpoint['contrast'])
                if self.cfg.CONTRAST.MEM_TYPE == 'moco':
                    self.model_ema.load_state_dict(checkpoint['model_ema'])
                if self.cfg.APEX.FLAG:
                    print('==> resuming amp state_dict')
                    amp.load_state_dict(checkpoint['amp'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(self.cfg.CHECKPOINT.RESUME, checkpoint['epoch']))
        
        if self.cfg.CHECKPOINT.FINETUNE:
            self.cfg.SOLVER.START_EPOCH = 0

    @staticmethod
    def _momentum_update(model, model_ema, m):
        ''' model_ema = m * model_ema + (1-m) * model'''
        for p1, p2 in zip(model.parameters(), model_ema.parameters()):
            p2.data.mul_(m).add_(p1.detach().data, alpha=1-m)

    @staticmethod
    def _global_gather(x):
        all_x = [torch.ones_like(x)
                 for _ in range(dist.get_world_size())]
        dist.all_gather(all_x, x, async_op=False)
        return torch.cat(all_x, dim=0)

    def _shuffle_bn(self, x, model_ema):
        """ Shuffle BN implementation
        Args:
          x: input image on each GPU/process
          model_ema: momentum encoder on each GPU/process
        """
        args = self.args
        local_gp = self.local_group
        bsz = x.size(0)

        # gather x locally for each node
        node_x = [torch.ones_like(x)
                  for _ in range(dist.get_world_size(local_gp))]
        dist.all_gather(node_x, x.contiguous(),
                        group=local_gp, async_op=False)
        node_x = torch.cat(node_x, dim=0)

        # shuffle bn
        shuffle_ids = torch.randperm(
            bsz * dist.get_world_size(local_gp)).cuda()
        reverse_ids = torch.argsort(shuffle_ids)
        dist.broadcast(shuffle_ids, 0)
        dist.broadcast(reverse_ids, 0)

        this_ids = shuffle_ids[args.local_rank*bsz:(args.local_rank+1)*bsz]
        with torch.no_grad():
            this_x = node_x[this_ids]
            if self.cfg.CONTRAST.JIGSAW:
                k = model_ema(this_x)
            else:
                k = model_ema(this_x)

        # globally gather k
        all_k = self._global_gather(k)

        # unshuffle bn
        node_id = args.node_rank
        ngpus = args.ngpus_per_node
        node_k = all_k[node_id*ngpus*bsz:(node_id+1)*ngpus*bsz]
        this_ids = reverse_ids[args.local_rank*bsz:(args.local_rank+1)*bsz]
        k = node_k[this_ids]

        return k, all_k

    def _broadcast_memory(self, contrast):
        """Synchronize memory buffers
        Args:
          contrast: memory.
        """
        if self.cfg.CROSS.MODALITY == 'visual':
            dist.broadcast(contrast.memory, 0)
        else:
            dist.broadcast(contrast.memory_1, 0)
            dist.broadcast(contrast.memory_2, 0)

    def _parse_model(self):
        """
        a simpler wrapper that creates the train/test models
        """
        # ------------------------------------------------------------
        # set device wrap model
        if self.cfg.MODEL.DEVICE == 'cuda':
            self._wrap_up()

        # -----------------------------------------------------------------------
        # resume
        self._resume()
        


    def train(self, epoch):
        
        if self.cfg.CONTRAST.MEM_TYPE == 'moco':
            self._train_moco(epoch)
        elif self.cfg.CONTRAST.MEM_TYPE == 'bank':
            self._train_ins(epoch)
        elif self.cfg.CONTRAST.MEM_TYPE == 'simsiam':
            self._train_simsiam(epoch)
        else:
            raise NotImplementedError('Unknown Contast type {}!'.format(self.cfg.CONTRAST.MEM_TYPE))

        if self.args.local_rank == 0:
            if self.cfg.CHECKPOINT.NO_VAL and ((epoch+1) % self.cfg.CHECKPOINT.CHECKPOINT_INTERVAL == 0 or epoch == self.cfg.SOLVER.MAX_EPOCHS-1):
                # save checkpoint every epoch
                print('======>Saving Checkpoint...')
                state = {
                    'epoch': epoch + 1,
                    'state_dict': self.model.module.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'contrast': self.contrast.state_dict() if self.contrast is not None else self.contrast,
                }
                if self.cfg.CONTRAST.MEM_TYPE == 'moco':
                    state['model_ema'] = self.model_ema.state_dict()
                if self.cfg.APEX.FLAG:
                    state['amp'] = amp.state_dict()
                is_best = False
                check_filename = 'checkpoint_{}.pth.tar'.format(epoch+1)
                self.saver.save_checkpoint(state, is_best, filename=check_filename)
                del state

        if (self.local_rank == 0) and ((epoch+1) == self.cfg.SOLVER.MAX_EPOCHS):
            self.writer.close()

    def _train_ins(self, epoch):
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
        for i, (image, _, index) in enumerate(tbar):
            # measure data loading time
            self.data_time.update(time.time() - end)

            ########## key part for train ########
            ########## forward ######################
            if self.cfg.MODEL.DEVICE == 'cuda':
                image = image.float().cuda(self.args.gpu, non_blocking=True)
                index = index.cuda(self.args.gpu, non_blocking=True)

            bsz = image.size(0)
            self.optimizer.zero_grad()
            feat = self.model(image)
            all_feat = self._global_gather(feat)
            all_index = self._global_gather(index)

            output, labels = self.contrast(feat, index, None, all_feat, all_index)
            loss = self.criterion(output)

            ################ backward #############
            if self.cfg.APEX.FLAG:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            if self.cfg.SOLVER.CLIP_GRADIENT != 'none':
                total_norm = clip_grad_norm_(self.model.parameters(), self.cfg.SOLVER.CLIP_GRADIENT)
                if total_norm > self.cfg.SOLVER.CLIP_GRADIENT:
                    print("clipping gradient: {} with coef {}".format(total_norm, self.cfg.SOLVER.CLIP_GRADIENT / total_norm))
            self.optimizer.step()
            ############  end ####################

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.detach(), labels.detach(), topk=(1, 5))
            self.losses.update(loss.item(), bsz)
            self.top1.update(prec1.item(), bsz)
            self.top5.update(prec5.item(), bsz)

            # report
            print('Train loss: %.3f' % (self.losses.avg))
            tbar.set_description('Train loss: %.3f' % (self.losses.avg))
            if self.args.local_rank == 0:
                self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # measure elapsed time
            self.batch_time.update(time.time() - end)
            end = time.time()

            if self.args.local_rank == 0:
                if i % self.cfg.CHECKPOINT.PRINT_FREQ == 0:
                    print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        epoch, i, len(self.train_loader), batch_time=self.batch_time,
                        data_time=self.data_time, loss=self.losses, top1=self.top1, top5=self.top5, lr=self.optimizer.param_groups[-1]['lr'])))

        if self.args.local_rank == 0:
            self.writer.add_scalar('train/total_loss_epoch', self.losses.sum, epoch)
            self.writer.add_scalar('train/loss', self.losses.avg, epoch)
            self.writer.add_scalar('train/top1_acc', self.top1.avg, epoch)
            self.writer.add_scalar('train/top5_acc', self.top5.avg, epoch)
            self.writer.add_scalar('lr', self.optimizer.param_groups[-1]['lr'], epoch)
            print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.cfg.DATALOADER.BATCH_SIZE + image.data.shape[0]))
            print('Loss: %.3f' % self.losses.sum)   

    def _train_moco(self, epoch): 
        self.batch_time.reset()
        self.data_time.reset()
        self.losses.reset()
        self.top1.reset()
        self.top5.reset()

        # switch to train mode
        self.model.train()
        self.model_ema.eval()

        def set_bn_train(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.train()
        self.model_ema.apply(set_bn_train)


        end = time.time()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, (images, _, _) in enumerate(tbar):
            # measure data loading time
            self.data_time.update(time.time() - end)

            ########## key part for train ########
            ############ forward ##############
            if self.cfg.MODEL.DEVICE == 'cuda':
                images = images.float().cuda(self.args.gpu, non_blocking=True)
            bsz = images.size(0)
            x1, x2 = torch.chunk(images, 2, dim=1)

            # ids for ShuffleBN for momentum encoder
            feat_k, all_k = self._shuffle_bn(x2, self.model_ema)
            self.optimizer.zero_grad()
            
            feat_q = self.model(x1)
            output, labels = self.contrast(feat_q, feat_k, all_k=all_k)
            loss = self.criterion(output)

            ################ backward #############
            if self.cfg.APEX.FLAG:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            if self.cfg.SOLVER.CLIP_GRADIENT != 'none':
                total_norm = clip_grad_norm_(self.model.parameters(), self.cfg.SOLVER.CLIP_GRADIENT)
                if total_norm > self.cfg.SOLVER.CLIP_GRADIENT:
                    print("clipping gradient: {} with coef {}".format(total_norm, self.cfg.SOLVER.CLIP_GRADIENT / total_norm))
            self.optimizer.step()
            ############  end ####################

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.detach(), labels.detach(), topk=(1, 5))
            self.losses.update(loss.item(), bsz)
            self.top1.update(prec1.item(), bsz)
            self.top5.update(prec5.item(), bsz)

            # report
            print('Train loss: %.3f' % (self.losses.avg))
            tbar.set_description('Train loss: %.3f' % (self.losses.avg))
            if self.args.local_rank == 0:
                self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # update momentum encoder
            self._momentum_update(self.model, self.model_ema, self.cfg.CONTRAST.ALPHA)
            # measure elapsed time
            self.batch_time.update(time.time() - end)
            end = time.time()

            if self.args.local_rank == 0:
                if i % self.cfg.CHECKPOINT.PRINT_FREQ == 0:
                    print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        epoch, i, len(self.train_loader), batch_time=self.batch_time,
                        data_time=self.data_time, loss=self.losses, top1=self.top1, top5=self.top5, lr=self.optimizer.param_groups[-1]['lr'])))
        
        if self.args.local_rank == 0:
            self.writer.add_scalar('train/total_loss_epoch', self.losses.sum, epoch)
            self.writer.add_scalar('train/loss', self.losses.avg, epoch)
            self.writer.add_scalar('train/top1_acc', self.top1.avg, epoch)
            self.writer.add_scalar('train/top5_acc', self.top5.avg, epoch)
            self.writer.add_scalar('lr', self.optimizer.param_groups[-1]['lr'], epoch)
            print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.cfg.DATALOADER.BATCH_SIZE + images.data.shape[0]))
            print('Loss: %.3f' % self.losses.sum)

        if self.args.local_rank == 0 and epoch == self.cfg.SOLVER.MAX_EPOCHS:
            self.writer.close()

    def _train_simsiam(self, epoch):
        self.batch_time.reset()
        self.data_time.reset()
        self.losses.reset()

        # switch to train mode
        self.model.train()

        end = time.time()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, (images, _, _) in enumerate(tbar):
            # measure data loading time
            self.data_time.update(time.time() - end)

            ########## key part for train ########
            ############ forward ##############
            if self.cfg.MODEL.DEVICE == 'cuda':
                images = images.float().cuda(self.args.gpu, non_blocking=True)
            bsz = images.size(0)
            loss = self.model(images)

            ################ backward #############
            if self.cfg.APEX.FLAG:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            if self.cfg.SOLVER.CLIP_GRADIENT != 'none':
                total_norm = clip_grad_norm_(self.model.parameters(), self.cfg.SOLVER.CLIP_GRADIENT)
                if total_norm > self.cfg.SOLVER.CLIP_GRADIENT:
                    print("clipping gradient: {} with coef {}".format(total_norm, self.cfg.SOLVER.CLIP_GRADIENT / total_norm))
            self.optimizer.step()
            ############  end ####################

            # measure accuracy and record loss
            self.losses.update(loss.item(), bsz)

            # report
            print('Train loss: %.3f' % (self.losses.avg))
            tbar.set_description('Train loss: %.3f' % (self.losses.avg))
            if self.args.local_rank == 0:
                self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # measure elapsed time
            self.batch_time.update(time.time() - end)
            end = time.time()

            if self.args.local_rank == 0:
                if i % self.cfg.CHECKPOINT.PRINT_FREQ == 0:
                    print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                        epoch, i, len(self.train_loader), batch_time=self.batch_time,
                        data_time=self.data_time, loss=self.losses, lr=self.optimizer.param_groups[-1]['lr'])))
        
        if self.args.local_rank == 0:
            self.writer.add_scalar('train/total_loss_epoch', self.losses.sum, epoch)
            self.writer.add_scalar('train/loss', self.losses.avg, epoch)
            self.writer.add_scalar('lr', self.optimizer.param_groups[-1]['lr'], epoch)
            print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.cfg.DATALOADER.BATCH_SIZE + images.data.shape[0]))
            print('Loss: %.3f' % self.losses.sum)


def main():
    parser = argparse.ArgumentParser(description='Classification model training')
    parser.add_argument('--config_file', type=str, default='', required=True,
                        help='Optional config file for params')
    parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                    help='number of nodes for distributed training')
    parser.add_argument('--dist-url', default='tcp://localhost:23456', type=str,
                            help='url used to set up distributed training')
    parser.add_argument('opts', help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    print(cfg)

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_IDS   
        cudnn.benchmark = True

    torch.manual_seed(cfg.MODEL.SEED)

    # ------------------------------------------------------------------------------
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, (cfg, args)))
    # ------------------------------------------------------------------------------


def main_worker(gpu, ngpus_per_node, args):
    cfg, args = args

    # --------------------------------------------------------------------------
    # initialize trainer and ddp environment
    # --------------------------------------------------------------------------
    trainer = Trainer(cfg, args)
    trainer.init_ddp_enviroment(gpu, ngpus_per_node)

    # -------------------------------------------------------------------------
    # build model, dataset, contrast, criterion and memory
    # -------------------------------------------------------------------------
    trainer.parse_trainer()


    # -------------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------------
    print('Starting Epoch:', trainer.cfg.SOLVER.START_EPOCH)
    print('Total Epochs:', trainer.cfg.SOLVER.MAX_EPOCHS)
    for epoch in range(trainer.cfg.SOLVER.START_EPOCH, trainer.cfg.SOLVER.MAX_EPOCHS):
        # -------------------------------------------------------------------------
        # training
        # -------------------------------------------------------------------------
        trainer.train_sampler.set_epoch(epoch)
        trainer.train(epoch)
        #adjust learning rate
        trainer.scheduler.step()


if __name__ == '__main__':
    main()