import argparse
# import time
from tqdm import tqdm
import os
import sys
import json
import pickle

import torch
import torchvision
import numpy as np
import torch.backends.cudnn as cudnn

from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

sys.path.append('.')
from lib.modeling import VisualModelWrapper
from lib.data.datasets import BaseDataset
from lib.data.transform.consistency_transforms import *


def _create_model(args):

    if args.dataset == 'ucf101':
        args.num_class = 101
    elif args.dataset == 'hmdb51':
        args.num_class = 51
    elif args.dataset == 'kinetics':
        args.num_class = 400
    else:
        raise ValueError('Unknown dataset '+args.dataset)

    model = VisualModelWrapper(args.video_length, args.modality,
                    backbone_name=args.arch, backbone_type=args.model_type, agg_fun=args.pool_fun, dropout=args.dropout,
                    pretrained=True, pretrain_path='/home/zhangjingran/2020/proj_1/lib/model_zoo/s3d_kinetics.pth')

    if os.path.exists(args.checkpoint):
        print('load pretrained model from ', args.checkpoint)
        checkpoint = torch.load(args.checkpoint)
        base_dict = {k.split('encoder.')[1]: v for k, v in list(checkpoint['state_dict'].items()) if 'proj_head' not in k}
        model.load_state_dict(base_dict)

    if args.gpus is not None:
        devices = args.gpus
    else:
        devices = list(range(args.workers))

    model = torch.nn.DataParallel(model.cuda(devices[0]), device_ids=devices)
    model.eval()

    return model

def _get_data(args, train=True):
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

    if train:
        train_dataset = BaseDataset(args.root, args.train_list, video_length=args.video_length,
                        modality=args.modality,
                        image_tmpl="img_{:05d}.jpg" if args.modality in ['RGB', 'RGBDiff'] else args.flow_prefix+"{}_{:05d}.jpg",
                        test_mode=True, num_clips=args.test_clips,
                        transform=transform)
        data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
             num_workers=args.workers*2, pin_memory=True)
    else:
        val_dataset = BaseDataset(args.root, args.test_list, video_length=args.video_length,
                        modality=args.modality,
                        image_tmpl="img_{:05d}.jpg" if args.modality in ['RGB', 'RGBDiff'] else args.flow_prefix+"{}_{:05d}.jpg",
                        test_mode=True, num_clips=args.test_clips,
                        transform=transform)
        data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
             num_workers=args.workers*2, pin_memory=True)

    return data_loader

def _extract_feature_single(args, model, data):
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

        return results

def extract_feature(args, train=True):
    """
    Extract and save features for train split, several clips per video.
    10 clips of 16 frames
    """
    data_loader = _get_data(args, train)
    model = _create_model(args)

    tmp_state = 'train' if train else 'val'

    features, classes = [], []
    loader_tbar = tqdm(data_loader, desc='\r')
    for i, (image, target) in enumerate(loader_tbar):
        if args.gpus is not None:
            image = image.cuda(args.gpu, non_blocking=True)

        feature = _extract_feature_single(args, model, image)    

        print('Processing video {} in {} set done!'.format(i, tmp_state))       

        features.append(feature)
        classes.append(target)
        
    features = torch.cat(features, dim=0)
    classes = torch.cat(classes, dim=0)

    print('{} features extracted done!\t Total samples {}'.format(tmp_state, len(data_loader.dataset)))

    with open(os.path.join(args.save_scores, '{}_'.format(tmp_state)+args.features_file), 'wb') as handle:
        print("Dumping {} feature as pkl file".format(tmp_state), flush=True)
        pickle.dump(features.cpu().numpy(), handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.save_scores, '{}_'.format(tmp_state)+args.classes_file), 'wb') as handle:
        print("Dumping {} class as pkl file".format(tmp_state), flush=True)
        pickle.dump(classes.cpu().numpy(), handle, protocol=pickle.HIGHEST_PROTOCOL)

    del features, classes
    

    
def topk_retrieval(args):
    """
    Extract features from test split and search on train split features.
    """
    def load_pickle(pkl_path):
        if os.path.exists(pkl_path):
            print(f"Loading pickle file: {pkl_path}", flush=True)
            with open(pkl_path, 'rb') as handle:
                result = pickle.load(handle)
                return result
        else:
            print('NO feature is founded in file {}'.format(pkl_path))

    train_features = load_pickle(args.train_feature_path)
    train_classes = load_pickle(args.train_classes_path)
    val_features = load_pickle(args.val_feature_path)
    val_classes = load_pickle(args.val_classes_path)

    if args.norm:
        val_features = torch.nn.functional.normalize(torch.from_numpy(val_features), dim=1).numpy()
        train_features = torch.nn.functional.normalize(torch.from_numpy(train_features), dim=1).numpy()  

    ks = [1, 5, 10, 20, 50]
    topk_correct = {k:0 for k in ks}

    # calculate retrieval sample similarity
    if args.distance_metric == 'cosine':
        distances = cosine_distances(val_features, train_features)
    else:
        distances = euclidean_distances(val_features, train_features)
    indices = np.argsort(distances)

    class_indexs = open(args.class_list, 'r').readlines()
    class_indexs = [x.strip().split(' ') for x in class_indexs]
    class_idx2label = {str(int(idx)-1): class_name for idx, class_name in class_indexs}

    for k in ks:
        top_k_indices = indices[:, :k]
        for video_id, (ind, val_label) in enumerate(zip(top_k_indices, val_classes)):
            retri_labels = train_classes[ind]
            if val_label in retri_labels:
                topk_correct[k] += 1

            print(('video {} done\t' 
                    'validation label {}, retrievaled label in train set {} of top_{} \n'.
                    format(video_id, class_idx2label[str(val_label)], [class_idx2label[str(retri_label)] for retri_label in retri_labels], k)))

    for k in ks:
        correct = topk_correct[k]
        total = len(val_classes)
        print('Top-{}, correct = {:.2f}, total = {}, acc = {:.3f}'.format(k, correct, total, correct/total))

    with open(os.path.join(args.save_scores, 'topk_correct.json'), 'w') as fp:
        json.dump(topk_correct, fp)

def get_parser():
    # options
    parser = argparse.ArgumentParser(
        description="Standard video-level testing")
    parser.add_argument('--dataset', type=str, default='ucf101', choices=['ucf101', 'hmdb51', 'kinetics'])
    parser.add_argument('--modality', type=str, default='RGB', choices=['RGB', 'Flow', 'RGBDiff'])
    parser.add_argument('--root', default='/', type=str)
    parser.add_argument('--train_list', default='/', type=str)
    parser.add_argument('--test_list', default='/', type=str)
    parser.add_argument('--class_list', default='/', type=str)
    parser.add_argument('--checkpoint', type=str, default='/')
    parser.add_argument('--arch', type=str, default="S3D")
    parser.add_argument('--model_type', type=str, default='3D')
    parser.add_argument('--test_clips', type=int, default=1)
    parser.add_argument('--max_num', type=int, default=-1)
    parser.add_argument('--test_crops', type=int, default=1)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--scale_size', type=int, default=256)
    parser.add_argument('--pool_fun', type=str, default='avg',
                        choices=['avg', 'max', 'topk'])
    parser.add_argument('--mean', type=list, nargs='+', default=[0.485, 0.456, 0.406])
    parser.add_argument('--std', type=list, nargs='+', default=[0.229, 0.224, 0.225])
    parser.add_argument('--video_length', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gpus', nargs='+', type=int, default=None)
    parser.add_argument('--softmax', default=False, action="store_true", help='use softmax')
    parser.add_argument('--extract_feature', action="store_true", default=False)
    parser.add_argument('--save_scores', default='/', type=str)
    parser.add_argument('--features_file', type=str, default='features.pkl')
    parser.add_argument('--classes_file', type=str, default='classes.pkl')

    ### distributed training
    parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                    help='number of nodes for distributed training')
    parser.add_argument('--dist-url', default='tcp://localhost:23456', type=str,
                            help='url used to set up distributed training')
    parser.add_argument('opts', help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)

    # retrieval_metric
    parser.add_argument('--distance_metric', type=str, default='cosine')
    parser.add_argument('--norm', action="store_true", default=False)
    parser.add_argument('--train_feature_path', type=str, default='/')
    parser.add_argument('--train_classes_path', type=str, default='/')
    parser.add_argument('--val_feature_path', type=str, default='/')
    parser.add_argument('--val_classes_path', type=str, default='/')
 
    args = parser.parse_args()

    return args

def main_worker():
    
    args = get_parser()
    print(args)

    if args.extract_feature:
        extract_feature(args, train=True)
        extract_feature(args, train=False)
    else:
        topk_retrieval(args)

if __name__ == '__main__':

    main_worker()