from .video_transforms import *
from .consistency_transforms import *
import torchvision

def create_transform_pil(cfg, is_train=True):
    normalize = GroupNormalize(mean=cfg.INPUT.MEAN, std=cfg.INPUT.STD)
    if is_train:
        augmentation = torchvision.transforms.Compose([GroupMultiScaleCrop(cfg.INPUT.BASE_SIZE, [1, .875, .75, .66]),
                                        GroupRandomHorizontalFlip(is_flow=False)])
        transform = torchvision.transforms.Compose([
                        augmentation,
                        Stack(roll=False),
                        ToTorchFormatTensor(div=(cfg.MODEL.BACKBONE in ['S3D', 'S3DG']), 
                                backbone_type=cfg.MODEL.BACKBONE_TYPE),
                        normalize,
                    ])
    else:
        transform = torchvision.transforms.Compose([
                        GroupScale(cfg.INPUT.SCALE_SIZE),
                        GroupCenterCrop(cfg.INPUT.CROP_SIZE),
                        Stack(roll=False),
                        ToTorchFormatTensor(div=(cfg.MODEL.BACKBONE in ['S3D', 'S3DG'])),
                        normalize,
                    ])
    return transform

def build_transform_cv2(cfg, is_train=True):
    
    if is_train:        
        transform = torchvision.transforms.Compose([
                        VideoMultiScaleCrop(cfg.INPUT.BASE_SIZE, [1, .875, .75, .66]),
                        VideoRandomHorizontalFlip(p=0.5),
                        VideoNormalize(mean=cfg.INPUT.MEAN, std=cfg.INPUT.STD),
                        VideoToTensor(backbone_type=cfg.MODEL.BACKBONE_TYPE),
                    ])
    else:
        transform = torchvision.transforms.Compose([
                        VideoResize(cfg.INPUT.SCALE_SIZE),
                        VideoCenterCrop(cfg.INPUT.CROP_SIZE),
                        VideoNormalize(mean=cfg.INPUT.MEAN, std=cfg.INPUT.STD),
                        VideoToTensor(backbone_type=cfg.MODEL.BACKBONE_TYPE),
                    ])
    return transform

def build_video_contrast_transform_cv2(cfg):
    
    transform = torchvision.transforms.Compose([
                    VideoRandomResizedCrop(cfg.INPUT.BASE_SIZE, scale=(0.2, 1.)),
                    VideoRandomApply(
                          VideoRandomColorJitter(brightness=0.4, contrast=0.4, 
                                    saturation=0.4, hue=0.1), p=0.8
                     ),
                    VideoRandomGrayScale(p=0.2),
                    VideoRandomApply(
                          VideoGaussianBlur(sigma_limit=(0.1, 2.)),
                    p=0.5),
                    VideoRandomHorizontalFlip(p=0.5),
                    VideoNormalize(mean=cfg.INPUT.MEAN, std=cfg.INPUT.STD),
                    VideoToTensor(backbone_type=cfg.MODEL.BACKBONE_TYPE),
                ])
    
    return transform