from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = 'cuda'
_C.MODEL.DEVICE_IDS = '0, 1, 2, 3'
_C.MODEL.SEED = 1
_C.MODEL.BACKBONE = 'resnet101'
_C.MODEL.BACKBONE_TYPE = '2D'
_C.MODEL.PRETRAINED = True
_C.MODEL.PRETRAIN_PATH = 'none'
_C.MODEL.PRETRAIN_CHOICE = 'none'
_C.MODEL.METRIC_LOSS_TYPE = 'CrossEntropyLoss'
_C.MODEL.POOLING_TYPE = 'avg'
_C.MODEL.DROPOUT = 0.5
_C.MODEL.NO_PARTIALBN = False
_C.MODEL.DISTRIBUTED = True
_C.MODEL.REASONING_FLAG = False
_C.MODEL.AUG_FLAG = False
_C.MODEL.LINEAR_PROBE = False

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.BASE_SIZE = [224, 224]
_C.INPUT.CROP_SIZE = [224, 224]
_C.INPUT.SCALE_SIZE = [256, 256]
_C.INPUT.MEAN = [0.485, 0.456, 0.406]
_C.INPUT.STD = [0.229, 0.224, 0.225]
_C.INPUT.MODALITY = 'RGB'
_C.INPUT.SAMPLE_TYPE = 'uniform'
_C.INPUT.VIDEO_LENGTH = 16
_C.INPUT.SAMPLE_RATE = 4
_C.INPUT.IMG_TMP = 'img_{:05d}.jpg'
_C.INPUT.FLOW_TMP = 'flow_{}_{:05d}.jpg'
_C.INPUT.FLIP = True
_C.INPUT.PRE_LOAD = 'cv2'
_C.INPUT.TEMPORAL_JITTER = False

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.NAME = 'kinetics'
_C.DATASET.NUM_CLASS = 101
_C.DATASET.VISUAL_ROOT_DIR = 'path'
_C.DATASET.AUDIO_ROOT_DIR = 'path'
_C.DATASET.TRAIN_SPLIT = './'
_C.DATASET.VALIDATION_SPLIT = './'

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 8
_C.DATALOADER.BATCH_SIZE = 128

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "SGD"
_C.SOLVER.LR_SCHEDULER = 'poly'
_C.SOLVER.MAX_EPOCHS = 50
_C.SOLVER.START_EPOCH = 0
_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.BIAS_LR_FACTOR = 2
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 5e-4
_C.SOLVER.WEIGHT_DECAY_BIAS = 0
_C.SOLVER.NESTEROV = False
_C.SOLVER.USE_TRICK = False
_C.SOLVER.LR_STEP = 20
_C.SOLVER.CLIP_GRADIENT = 'none'
_C.SOLVER.NO_PARTIALBN = True

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (30, 60)
_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 5
_C.SOLVER.WARMUP_METHOD = "linear"

# ---------------------------------------------------------------------------- #
# apex
# ---------------------------------------------------------------------------- #
_C.APEX = CN()
_C.APEX.FLAG = False
_C.APEX.OPT_LEVEL = 'O1'
_C.APEX.LOCAL_RANK = -1

_C.TEST = CN()
_C.TEST.BATCH_SIZE = 128
_C.TEST.WEIGHT = ""

# ---------------------------------------------------------------------------- #
# checkpoint
# ---------------------------------------------------------------------------- #
_C.CHECKPOINT = CN()
_C.CHECKPOINT.RESUME = 'none'
_C.CHECKPOINT.CHECKNAME = 'video_model'
_C.CHECKPOINT.CHECKPOINT_INTERVAL = 20
_C.CHECKPOINT.NO_VAL = False
_C.CHECKPOINT.EVAL_INTERVAL = 5
_C.CHECKPOINT.FINETUNE = False
_C.CHECKPOINT.PRINT_FREQ = 20


# ---------------------------------------------------------------------------- #
# Contrast
# ---------------------------------------------------------------------------- #
_C.CONTRAST = CN()
_C.CONTRAST.MEM_TYPE = 'bank'
_C.CONTRAST.NCE_K = 65536
_C.CONTRAST.NCE_T = 0.07
_C.CONTRAST.NCE_M = 0.5
_C.CONTRAST.ALPHA = 0.999
_C.CONTRAST.JIGSAW = False

# ---------------------------------------------------------------------------- #
# Ccross modality
# ---------------------------------------------------------------------------- #
_C.CROSS = CN()
_C.CROSS.FEAT_DIM = 128
_C.CROSS.HEAD_TYPE = 'mlp'
_C.CROSS.MEM = None
_C.CROSS.BETA = 0.5
_C.CROSS.MODALITY = 'visual'
_C.CROSS.CRITERION = 'crossentropy'

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
