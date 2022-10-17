## DEFAULT CONFIGURATION USED IN OUR EXPERIMENTS ON 2 GPUs

import numpy as np

# DATASET CONFIGURATION
BATCH_SIZE = [64, 64]
META_TRAIN_PRCT = 90
NORMALISE_PARAMS = [
    1.0 / 255,  # SCALE
    np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3)),  # MEAN
    np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3)),  # STD
]
NUM_CLASSES = [3, 3]
NUM_WORKERS = 1
N_TASK0 = 8000  # store in-memory these many samples for the first task
TRAIN_DIR = "/media/dell/DATA/wy/data/GID-15/512/"
TRAIN_LIST = "/media/dell/DATA/wy/LightRS/data/list/gid15_vege_train.lst"
# TRAIN_LIST = "./data/lists/train+.lst"
VAL_BATCH_SIZE = 32
VAL_CROP_SIZE = 400
VAL_DIR = "/media/dell/DATA/wy/data/GID-15/512/"
VAL_LIST = "/media/dell/DATA/wy/LightRS/data/list/gid15_vege_val.lst"  # meta-train and meta-val learning
# VAL_LIST = "./data/lists/train+.lst"
VAL_OMIT_CLASSES = [0]  # ignore background when computing the reward
VAL_RESIZE_SIDE = 400

TEST_DIR = "/media/dell/DATA/wy/data/GID-15/512/"
TEST_LIST = "/media/dell/DATA/wy/LightRS/data/list/gid15_vege_val.lst" 

# AUGMENTATIONS CONFIGURATION
CROP_SIZE = [256, 350]
HIGH_SCALE = 1.4
LOW_SCALE = 0.7
RESIZE_SIDE = [300, 400]

# ENCODER OPTIMISATION CONFIGURATION
ENC_GRAD_CLIP = 3.0
ENC_LR = [1e-3, 1e-3]
ENC_MOM = [0.9] * 3
ENC_OPTIM = "sgd"
ENC_WD = [1e-5] * 3

# DECODER OPTIMISATION CONFIGURATION
DEC_AUX_WEIGHT = 0.15  # to disable aux, set to -1
DEC_GRAD_CLIP = 3.0
DEC_LR = [3e-3, 3e-3]
DEC_MOM = [0.9] * 3
DEC_OPTIM = "adam"
DEC_WD = [1e-5] * 3

# GENERAL OPTIMISATION CONFIGURATION
DO_KD = True
DO_POLYAK = True
FREEZE_BN = [False, False]
KD_COEFF = 0.3
NUM_EPOCHS = 20000
NUM_SEGM_EPOCHS = [5, 1]
RANDOM_SEED = 9314
VAL_EVERY = [10, 1]  # how often to record validation scores

# GENERAL DEBUGGING CONFIGURATION
CKPT_PATH = "./ckpt/rs_checkpoint.pth.tar"
PRINT_EVERY = 100
SNAPSHOT_DIR = "./ckpt/"
SUMMARY_DIR = "./tb_logs/"

# CONTROLLER CONFIGURATION: USED TO CREATE RNN-BASED CONTROLLER
CELL_MAX_REPEAT = 4
CELL_MAX_STRIDE = 2
CELL_NUM_LAYERS = 4
CTRL_AGENT = "ppo"
CTRL_BASELINE_DECAY = 0.95
CTRL_LR = 1e-4
CTRL_VERSION = "cvpr"
DEC_NUM_CELLS = 3
LSTM_HIDDEN_SIZE = 100
LSTM_NUM_LAYERS = 2
NUM_AGG_OPS = 2
NUM_OPS = 11

# DECODER CONFIGURATION: USED TO CREATE DECODER ARCHITECTURE
AGG_CELL_SIZE = 64 #48
AUX_CELL = True
SEP_REPEATS = 1
