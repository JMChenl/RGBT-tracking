# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from yacs.config import CfgNode as CN

__C = CN()

cfg = __C

__C.META_ARC = "siamcar_r50"

__C.CUDA = True

# ------------------------------------------------------------------------ #
# Training options
# ------------------------------------------------------------------------ #
__C.TRAIN = CN()

# Anchor Target
__C.TRAIN.EXEMPLAR_SIZE = 127

__C.TRAIN.SEARCH_SIZE = 255

__C.TRAIN.OUTPUT_SIZE = 25

__C.TRAIN.RESUME = ''

__C.TRAIN.PRETRAINED = ''

__C.TRAIN.LOG_DIR = './logs'

__C.TRAIN.SNAPSHOT_DIR = './Ablation_Study/Fusion_att+att'   # LasHeR_testing

__C.TRAIN.EPOCH = 25

__C.TRAIN.START_EPOCH = 0

__C.TRAIN.BATCH_SIZE = 32

__C.TRAIN.NUM_WORKERS = 32

__C.TRAIN.MOMENTUM = 0.9

__C.TRAIN.WEIGHT_DECAY = 0.0001

__C.TRAIN.CLS_WEIGHT = 1.0

__C.TRAIN.LOC_WEIGHT = 2.0

__C.TRAIN.CEN_WEIGHT = 1.0

__C.TRAIN.PRINT_FREQ = 20

__C.TRAIN.LOG_GRADS = False

__C.TRAIN.GRAD_CLIP = 10.0

__C.TRAIN.BASE_LR = 0.005

__C.TRAIN.LR = CN()

__C.TRAIN.LR.TYPE = 'log'

__C.TRAIN.LR.KWARGS = CN(new_allowed=True)

__C.TRAIN.LR_WARMUP = CN()

__C.TRAIN.LR_WARMUP.WARMUP = True

__C.TRAIN.LR_WARMUP.TYPE = 'step'

__C.TRAIN.LR_WARMUP.EPOCH = 5

__C.TRAIN.LR_WARMUP.KWARGS = CN(new_allowed=True)

__C.TRAIN.NUM_CLASSES = 2

__C.TRAIN.NUM_CONVS = 4

__C.TRAIN.PRIOR_PROB = 0.01

__C.TRAIN.LOSS_ALPHA = 0.25

__C.TRAIN.LOSS_GAMMA = 2.0

# ------------------------------------------------------------------------ #
# Dataset options
# ------------------------------------------------------------------------ #
__C.DATASET = CN(new_allowed=True)

# Augmentation
# for template
__C.DATASET.TEMPLATE = CN()

# for detail discussion
__C.DATASET.TEMPLATE.SHIFT = 4

__C.DATASET.TEMPLATE.SCALE = 0.05

__C.DATASET.TEMPLATE.BLUR = 0.0

__C.DATASET.TEMPLATE.FLIP = 0.0

__C.DATASET.TEMPLATE.COLOR = 1.0

__C.DATASET.SEARCH = CN()

__C.DATASET.SEARCH.SHIFT = 64

__C.DATASET.SEARCH.SCALE = 0.18
# __C.DATASET.SEARCH.SCALE = 0

__C.DATASET.SEARCH.BLUR = 0.0

__C.DATASET.SEARCH.FLIP = 0.0

__C.DATASET.SEARCH.COLOR = 1.0

# for detail discussion
__C.DATASET.NEG = 0.0

__C.DATASET.GRAY = 0.0

__C.DATASET.NAMES = ('VID', 'COCO', 'DET', 'YOUTUBEBB')

__C.DATASET.VID = CN()
__C.DATASET.VID.ROOT = 'train_dataset/vid/crop511'          # VID dataset path
__C.DATASET.VID.ANNO = 'train_dataset/vid/train.json'
__C.DATASET.VID.FRAME_RANGE = 100
__C.DATASET.VID.NUM_USE = 100000  # repeat until reach NUM_USE

__C.DATASET.YOUTUBEBB = CN()
__C.DATASET.YOUTUBEBB.ROOT = 'train_dataset/yt_bb/crop511'  # YOUTUBEBB dataset path
__C.DATASET.YOUTUBEBB.ANNO = 'train_dataset/yt_bb/train.json'
__C.DATASET.YOUTUBEBB.FRAME_RANGE = 3
__C.DATASET.YOUTUBEBB.NUM_USE = -1  # use all not repeat

__C.DATASET.COCO = CN()
__C.DATASET.COCO.ROOT = 'train_dataset/coco/crop511'         # COCO dataset path
__C.DATASET.COCO.ANNO = 'train_dataset/coco/train2017.json'
__C.DATASET.COCO.FRAME_RANGE = 1
__C.DATASET.COCO.NUM_USE = -1

__C.DATASET.DET = CN()
__C.DATASET.DET.ROOT = 'train_dataset/det/crop511'           # DET dataset path
__C.DATASET.DET.ANNO = 'train_dataset/det/train.json'
__C.DATASET.DET.FRAME_RANGE = 1
__C.DATASET.DET.NUM_USE = -1

__C.DATASET.GTOT = CN()
__C.DATASET.GTOT.ROOT = '/root/cjm/object_tracker/My-/SiamCAR-master/tools/train_dataset/GTOT/crop511'         # GOT dataset path
__C.DATASET.GTOT.rgb_ANNO = 'train_dataset/GTOT/train_v.json'
__C.DATASET.GTOT.t_ANNO = 'train_dataset/GTOT/train_i.json'
__C.DATASET.GTOT.init_ANNO = 'train_dataset/GTOT/init.json'
__C.DATASET.GTOT.FRAME_RANGE = 1
__C.DATASET.GTOT.NUM_USE = -1

__C.DATASET.LasHeR = CN()
__C.DATASET.LasHeR.ROOT = '/root/cjm/object_tracker/My-/BAN-SiamCAR/tools/train_dataset/LasHeR/crop511'         # GOT dataset path
__C.DATASET.LasHeR.rgb_ANNO = '/root/cjm/object_tracker/My-/BAN-SiamCAR/tools/train_dataset/LasHeR/train_v.json'
__C.DATASET.LasHeR.t_ANNO = '/root/cjm/object_tracker/My-/BAN-SiamCAR/tools/train_dataset/LasHeR/train_v.json'
__C.DATASET.LasHeR.init_ANNO = '/root/cjm/object_tracker/My-/BAN-SiamCAR/tools/train_dataset/LasHeR/init.json'
# __C.DATASET.LasHeR.rgb_ANNO = '/root/cjm/object_tracker/My-/BAN-SiamCAR/tools/train_dataset/LasHeR/train979_v.json'
# __C.DATASET.LasHeR.t_ANNO = '/root/cjm/object_tracker/My-/BAN-SiamCAR/tools/train_dataset/LasHeR/train979_i.json'
# __C.DATASET.LasHeR.init_ANNO = '/root/cjm/object_tracker/My-/BAN-SiamCAR/tools/train_dataset/LasHeR/train979_init.json'
__C.DATASET.LasHeR.FRAME_RANGE = 100
__C.DATASET.LasHeR.NUM_USE = 100000

__C.DATASET.LaSOT = CN()
__C.DATASET.LaSOT.ROOT = 'train_dataset/lasot/crop511'         # LaSOT dataset path
__C.DATASET.LaSOT.ANNO = 'train_dataset/lasot/train.json'
__C.DATASET.LaSOT.FRAME_RANGE = 100
__C.DATASET.LaSOT.NUM_USE = 100000

__C.DATASET.VIDEOS_PER_EPOCH = 100000 #600000
# ------------------------------------------------------------------------ #
# Backbone options
# ------------------------------------------------------------------------ #
__C.BACKBONE = CN()

# Backbone type, current only support resnet18,34,50;alexnet;mobilenet
__C.BACKBONE.TYPE = 'res50'

__C.BACKBONE.KWARGS = CN(new_allowed=True)

# Pretrained backbone weights
__C.BACKBONE.PRETRAINED = ''

# Train layers
__C.BACKBONE.TRAIN_LAYERS = ['layer2', 'layer3', 'layer4']

# Layer LR
__C.BACKBONE.LAYERS_LR = 0.1

# Switch to train layer
__C.BACKBONE.TRAIN_EPOCH = 10

# ------------------------------------------------------------------------ #
# Adjust layer options
# ------------------------------------------------------------------------ #
__C.ADJUST = CN()

# Adjust layer
__C.ADJUST.ADJUST = True

__C.ADJUST.KWARGS = CN(new_allowed=True)

# Adjust layer type
__C.ADJUST.TYPE = "AdjustAllLayer"

# ------------------------------------------------------------------------ #
# RPN options
# ------------------------------------------------------------------------ #
__C.CAR = CN()

# RPN type
__C.CAR.TYPE = 'MultiCAR'

__C.CAR.KWARGS = CN(new_allowed=True)

# ------------------------------------------------------------------------ #
# Tracker options
# ------------------------------------------------------------------------ #
__C.TRACK = CN()

__C.TRACK.TYPE = 'SiamCARTracker'

# Scale penalty
__C.TRACK.PENALTY_K = 0.04

# Window influence
__C.TRACK.WINDOW_INFLUENCE = 0.44

# Interpolation learning rate
__C.TRACK.LR = 0.4

# Exemplar size
__C.TRACK.EXEMPLAR_SIZE = 127

# Instance size
__C.TRACK.INSTANCE_SIZE = 255

# Context amount
__C.TRACK.CONTEXT_AMOUNT = 0.5

__C.TRACK.STRIDE = 8


__C.TRACK.SCORE_SIZE = 25

__C.TRACK.hanming = True

__C.TRACK.NUM_K = 2

__C.TRACK.NUM_N = 1

__C.TRACK.REGION_S = 0.1

__C.TRACK.REGION_L = 0.44


# ------------------------------------------------------------------------ #
# HP_SEARCH parameters
# ------------------------------------------------------------------------ #
__C.HP_SEARCH = CN()

__C.HP_SEARCH.OTB100 = [0.35, 0.2, 0.45]

__C.HP_SEARCH.GOT10K = [0.7, 0.06, 0.1]

__C.HP_SEARCH.UAV123 = [0.4, 0.2, 0.3]

__C.HP_SEARCH.LaSOT = [0.33, 0.04, 0.3]

#__C.HP_SEARCH.GTOT = [0.33, 0.22, 0.4]#best
__C.HP_SEARCH.GTOT = [0.30, 0.22, 0.45]#0.928/0.74 66666666
__C.HP_SEARCH.LasHeR = [0.30, 0.22, 0.45]
__C.HP_SEARCH.data = [0.30, 0.22, 0.45]
__C.HP_SEARCH.VTUAV = [0.30, 0.22, 0.45]

#__C.HP_SEARCH.RGB_T234 = [0.33, 0.22, 0.4]
#__C.HP_SEARCH.RGB_T234 = [0.30, 0.14, 0.45]
#__C.HP_SEARCH.RGB_T234 = [0.30, 0.06, 0.45]#0.785/0.570
#__C.HP_SEARCH.RGB_T234 = [0.30, 0.22, 0.45]# 0.785/0.571 6666666666
__C.HP_SEARCH.RGB_T234 = [0.30, 0.06, 0.45] # 0.801/0.575 model_18
#__C.HP_SEARCH.RGB_T234 = [0.30, 0.22, 0.1]# 0.762/0.556
#__C.HP_SEARCH.RGB_T234 = [0.30, 0.22, 0.48]#0.783/0.567
#__C.HP_SEARCH.RGB_T234 = [0.40, 0.22, 0.45]#0.775/0.563
#__C.HP_SEARCH.RGB_T234 = [0.40, 0.04, 0.44]#0.772/0.56
#__C.HP_SEARCH.RGB_T234 = [0.38, 0.05, 0.42]#0.776/0.562
#__C.HP_SEARCH.RGB_T234 = [0.30, 0.22, 0.40]#0.782/0.570
#__C.HP_SEARCH.RGB_T234 = [0.30, 0.22, 0.45]# 0.780/0.568
#__C.HP_SEARCH.RGB_T234 = [0.30, 0.22, 0.45]
#__C.HP_SEARCH.RGB_T234 = [0.7, 0.06, 0.1]
#__C.HP_SEARCH.RGB_T234 = [0.35, 0.2, 0.45]
#__C.HP_SEARCH.RGB_T234 = [0.4, 0.2, 0.3]

#__C.HP_SEARCH.LasHeR = [0.7, 0.06, 0.1]

#__C.HP_SEARCH.LasHeR = [0.33, 0.22, 0.4]
