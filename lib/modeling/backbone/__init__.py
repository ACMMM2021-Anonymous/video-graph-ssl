from . import backbone_2d
from . import backbone_3d 

def build_backbone(cfg, backbone):
    if cfg.MODEL.BACKBONE_TYPE == '2D':
        #model = getattr(backbones_2d, cfg.MODEL.BACKBONE)()
        model = getattr(backbone_2d, backbone)()
    elif cfg.MODEL.BACKBONE_TYPE == '3D':
        #model = getattr(backbones_3d, cfg.MODEL.BACKBONE)()
        model = getattr(backbone_3d, backbone)()
    else:
        raise ValueError("Only support 2D or 3D backbone")

    if cfg.MODEL.PRETRAINED:
        model.load_state_dict(cfg.MODEL.PRETRAIN_PATH)
    return model