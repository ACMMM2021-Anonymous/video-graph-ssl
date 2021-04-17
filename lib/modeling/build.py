from .model_wrappers import VideoModelWrapper
from .visual_wrappers import VisualModelWrapper
from .graph_wrappers import GraphWrapper

def create_video_model(cfg):
    """
    a simpler wrapper that creates the train/test models
    """
        
    model = VideoModelWrapper(cfg.DATASET.NUM_CLASS, cfg.INPUT.VIDEO_LENGTH, cfg.INPUT.MODALITY,
                backbone_name=cfg.MODEL.BACKBONE, backbone_type=cfg.MODEL.BACKBONE_TYPE, agg_fun=cfg.MODEL.POOLING_TYPE, dropout=cfg.MODEL.DROPOUT, 
                partial_bn=not cfg.SOLVER.NO_PARTIALBN, pretrained=cfg.MODEL.PRETRAINED, pretrain_path=cfg.MODEL.PRETRAIN_PATH)

    return model

def create_visual_model(cfg):

    visual_encoder1 = VisualModelWrapper(cfg.INPUT.VIDEO_LENGTH, cfg.INPUT.MODALITY,
                backbone_name=cfg.MODEL.BACKBONE, backbone_type=cfg.MODEL.BACKBONE_TYPE, agg_fun=cfg.MODEL.POOLING_TYPE, dropout=cfg.MODEL.DROPOUT, 
                partial_bn=not cfg.SOLVER.NO_PARTIALBN, pretrained=cfg.MODEL.PRETRAINED, pretrain_path=cfg.MODEL.PRETRAIN_PATH)

    model = GraphWrapper(visual_encoder1, cfg.CROSS.FEAT_DIM, cfg.CROSS.HEAD_TYPE, cfg.CONTRAST.MEM_TYPE)

    if cfg.CONTRAST.MEM_TYPE == 'moco':
        visual_encoder2 = VisualModelWrapper(cfg.INPUT.VIDEO_LENGTH, cfg.INPUT.MODALITY,
                backbone_name=cfg.MODEL.BACKBONE, backbone_type=cfg.MODEL.BACKBONE_TYPE, agg_fun=cfg.MODEL.POOLING_TYPE, dropout=cfg.MODEL.DROPOUT, 
                partial_bn=not cfg.SOLVER.NO_PARTIALBN, pretrained=cfg.MODEL.PRETRAINED, pretrain_path=cfg.MODEL.PRETRAIN_PATH)
        model_ema = GraphWrapper(visual_encoder2, cfg.CROSS.FEAT_DIM, cfg.CROSS.HEAD_TYPE, cfg.CONTRAST.MEM_TYPE)
    else:
        model_ema = None

    return model, model_ema


