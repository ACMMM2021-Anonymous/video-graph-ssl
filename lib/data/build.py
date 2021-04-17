import torch
from torch.utils.data import DataLoader

from .transform import create_transform_pil, build_transform_cv2, build_video_contrast_transform_cv2
from .datasets import BaseDataset, VisualDataset

def make_data_loader(cfg):
    if cfg.INPUT.PRE_LOAD == 'cv2':
        train_transform = build_transform_cv2(cfg, is_train=True)
        val_transform = build_transform_cv2(cfg, is_train=False)
    else:
        train_transform = create_transform_pil(cfg, is_train=True)
        val_transform = create_transform_pil(cfg, is_train=False)

    train_dataset = BaseDataset(root_path=cfg.DATASET.ROOT_DIR, list_file=cfg.DATASET.TRAIN_SPLIT, 
                    video_length=cfg.INPUT.VIDEO_LENGTH, modality=cfg.INPUT.MODALITY,
                    sample_type=cfg.INPUT.SAMPLE_TYPE, pre_load=cfg.INPUT.PRE_LOAD,
                    image_tmpl=cfg.INPUT.IMG_TMP if cfg.INPUT.MODALITY in ["RGB", "RGBDiff"] else cfg.INPUT.FLOW_TMP,
                    transform=train_transform)

    val_dataset = BaseDataset(root_path=cfg.DATASET.ROOT_DIR, list_file=cfg.DATASET.VALIDATION_SPLIT, 
                    video_length=cfg.INPUT.VIDEO_LENGTH, modality=cfg.INPUT.MODALITY,
                    sample_type=cfg.INPUT.SAMPLE_TYPE, random_shift=False, pre_load=cfg.INPUT.PRE_LOAD,
                    image_tmpl=cfg.INPUT.IMG_TMP if cfg.INPUT.MODALITY in ["RGB", "RGBDiff"] else cfg.INPUT.FLOW_TMP,
                    transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=cfg.DATALOADER.BATCH_SIZE, shuffle=True,
            num_workers=cfg.DATALOADER.NUM_WORKERS, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
            num_workers=cfg.DATALOADER.NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader

def build_video_contrastive_loader(cfg):
    
    train_transform = build_video_contrast_transform_cv2(cfg)

    ngpus_per_node = len(cfg.MODEL.DEVICE_IDS.split(','))
    BATCH_SIZE = int(cfg.DATALOADER.BATCH_SIZE / ngpus_per_node)
    NUM_WORKERS = int((cfg.DATALOADER.NUM_WORKERS + ngpus_per_node - 1) / ngpus_per_node)

    train_dataset = VisualDataset(root_path=cfg.DATASET.VISUAL_ROOT_DIR, list_file=cfg.DATASET.TRAIN_SPLIT, 
                    video_length=cfg.INPUT.VIDEO_LENGTH, modality=cfg.INPUT.MODALITY,
                    sample_type=cfg.INPUT.SAMPLE_TYPE, frame_interval=cfg.INPUT.SAMPLE_RATE, mem_type=cfg.CONTRAST.MEM_TYPE,
                    image_tmpl=cfg.INPUT.IMG_TMP if cfg.INPUT.MODALITY in ["RGB", "RGBDiff"] else cfg.INPUT.FLOW_TMP,
                    transform=train_transform, temporal_jitter=cfg.INPUT.TEMPORAL_JITTER)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=(train_sampler is None),
            num_workers=NUM_WORKERS, pin_memory=True, sampler=train_sampler)

    return train_loader, train_sampler, len(train_dataset)