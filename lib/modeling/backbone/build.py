# from .backbone_2d import *
# from .backbone_3d import *
from . import backbone_2d
from . import backbone_3d

def create_backbone(cfg):
    if cfg.MODEL.BACKBONE_TYPE == '2D':
        #model = getattr(backbones_2d, cfg.MODEL.BACKBONE)()
        backbone = getattr(backbone_2d, cfg.MODEL.BACKBONE)()
    elif cfg.MODEL.BACKBONE_TYPE == '3D':
        #model = getattr(backbones_3d, cfg.MODEL.BACKBONE)()
        backbone = getattr(backbone_3d, cfg.MODEL.BACKBONE)()
    else:
        raise ValueError("Only support 2D or 3D backbone")

    # backbone = cfg.MODEL.BACKBONE() (str function) ???

    if cfg.MODEL.PRETRAINED:
        backbone.load_state_dict(cfg.MODEL.PRETRAIN_PATH)
    return backbone