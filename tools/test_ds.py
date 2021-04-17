from sklearn.metrics import confusion_matrix

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
from lib.modeling import VideoModelWrapper
from lib.data.datasets import BaseDataset
from lib.evaluation import AverageMeter, accuracy
from lib.data.transform.consistency_transforms import *


def get_parser():
    # options
    parser = argparse.ArgumentParser(
        description="Standard video-level testing")
    parser.add_argument('--dataset', type=str, default='ucf101', choices=['ucf101', 'hmdb51', 'kinetics'])
    parser.add_argument('--modality', type=str, default='RGB', choices=['RGB', 'Flow', 'RGBDiff'])
    parser.add_argument('--root', default='/', type=str)
    parser.add_argument('--test_list', default='/', type=str)
    parser.add_argument('--checkpoint', type=str, default='/')
    parser.add_argument('--arch', type=str, default="S3D")
    parser.add_argument('--save_scores', type=str, default=None)
    parser.add_argument('--test_clips', type=int, default=10)
    parser.add_argument('--max_num', type=int, default=-1)
    parser.add_argument('--test_crops', type=int, default=10)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--scale_size', type=int, default=256)
    parser.add_argument('--pool_fun', type=str, default='avg',
                        choices=['avg', 'max', 'topk'])
    parser.add_argument('--mean', type=list, nargs='+', default=[0.485, 0.456, 0.406])
    parser.add_argument('--std', type=list, nargs='+', default=[0.229, 0.224, 0.225])
    parser.add_argument('--video_length', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.7)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gpus', nargs='+', type=int, default=None)
    parser.add_argument('--softmax', default=False, action="store_true", help='use softmax')
    parser.add_argument('--csv_file', type=str, default=None)
    parser.add_argument('--model_type', type=str, default='3D')
 

    args = parser.parse_args()

    return args

def create_model(args):

    if args.dataset == 'ucf101':
        args.num_class = 101
    elif args.dataset == 'hmdb51':
        args.num_class = 51
    elif args.dataset == 'kinetics':
        args.num_class = 400
    else:
        raise ValueError('Unknown dataset '+args.dataset)


    model = VideoModelWrapper(args.num_class, args.video_length, args.modality,
                backbone_name=args.arch, backbone_type=args.model_type, agg_fun=args.pool_fun, dropout=args.dropout)
    


    checkpoint = torch.load(args.checkpoint)
    print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_pred']))

    base_dict = checkpoint['state_dict']
    model.load_state_dict(base_dict)

    if args.gpus is not None:
        devices = args.gpus
    else:
        devices = list(range(args.workers))

    model = torch.nn.DataParallel(model.cuda(devices[0]), device_ids=devices)
    model.eval()

    return model


def get_data(args):
    if args.test_crops == 1:
        cropping = torchvision.transforms.Compose([
            VideoResize(args.scale_size),
            VideoCenterCrop(args.input_size),
        ])
    elif args.test_crops == 3:  
        cropping = torchvision.transforms.Compose([
            VideoFullResSample(args.input_size, args.scale_size, flip=False)
        ])
    elif args.test_crops == 5:  
        cropping = torchvision.transforms.Compose([
            VideoOverSampleCrop(args.input_size, args.scale_size, flip=False)
        ])
    elif args.test_crops == 10:
        cropping = torchvision.transforms.Compose([
            VideoOverSampleCrop(args.input_size, args.scale_size)
        ])
    else:
        raise ValueError("Only 1, 5, 10 crops are supported while we got {}".format(args.test_crops))

    transform = torchvision.transforms.Compose([
                        cropping,
                        VideoNormalize(mean=args.mean, std=args.std),
                        VideoToTensor(backbone_type=args.model_type),
                    ])

    data_loader = torch.utils.data.DataLoader(
            BaseDataset(args.root, args.test_list, video_length=args.video_length,
                    modality=args.modality,
                    image_tmpl="img_{:05d}.jpg" if args.modality in ['RGB', 'RGBDiff'] else args.flow_prefix+"{}_{:05d}.jpg",
                    test_mode=True, num_clips=args.test_clips,
                    transform=transform),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers * 2, pin_memory=True)

    return data_loader


def eval_video(video_data, model, args):
    data, label = video_data
    num_crop = args.test_crops

    with torch.no_grad():
        input_clips = data.split(num_crop * args.video_length, dim=2)   
        input_clips = [clip.view((-1, 3, num_crop, args.video_length) + clip.shape[-2:]).contiguous() for clip in input_clips]
        results = []
        for input_clip in input_clips:
            results.extend([model(input_clip[:, :, i, :]) for i in range(num_crop)])
        results = torch.stack(results, axis=1).mean(1)
        
        if args.softmax:
            results = torch.softmax(results, dim=-1)

        return results.data.cpu().numpy().copy(), label

def main_work():

    top1 = AverageMeter()
    top5 = AverageMeter()

    args = get_parser()
    model = create_model(args)
    data_loader = get_data(args)

    total_num = len(data_loader.dataset)
    data_gen = enumerate(data_loader)
    output = []

    proc_start_time = time.time()
    max_num = args.max_num if args.max_num > 0 else len(data_loader.dataset)

    for i, (data, label) in data_gen:
        with torch.no_grad():
            if i >= max_num:
                break
            rst = eval_video((data, label), model, args)
            output.append(rst)
            cnt_time = time.time() - proc_start_time
            print('video {} done, total {}/{}, average {} sec/video'.format(i, i+1,
                                                                            total_num,
                                                                            float(cnt_time) / (i+1)))

            prec1, prec5 = accuracy(torch.from_numpy(rst[0]), rst[1], topk=(1, 5))
            top1.update(prec1.item(), label.shape[0])
            top5.update(prec5.item(), label.shape[0])
            if i % 20 == 0:
                print('video {} done, total {}/{}, average {:.3f} sec/video, '
                    'moving Prec@1 {:.3f} Prec@5 {:.3f}'.format(i * args.batch_size, i * args.batch_size, total_num,
                                                              float(cnt_time) / (i+1) / args.batch_size, top1.avg, top5.avg))

    video_pred = [np.argmax(x[0]) for x in output]
    video_labels = [x[1] for x in output]

    cf = confusion_matrix(video_labels, video_pred).astype(float)
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    cls_acc = cls_hit / cls_cnt
    print(cls_acc)

    print('-----Evaluation is finished------')
    print('Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))
    print('Overall Prec@1 {:.02f}% Prec@5 {:.02f}%'.format(top1.avg, top5.avg))

    if args.save_scores is not None:
        name_list = [x.strip().split()[0] for x in open(args.test_list)]
        order_dict = {e:i for i, e in enumerate(sorted(name_list))}

        reorder_output = [None] * len(output)
        reorder_label = [None] * len(output)

        for i in range(len(output)):
            idx = order_dict[name_list[i]]
            reorder_output[idx] = output[i]
            reorder_label[idx] = video_labels[i]

        np.savez(args.save_scores, scores=reorder_output, labels=reorder_label)

if __name__ == "__main__":
    main_work()