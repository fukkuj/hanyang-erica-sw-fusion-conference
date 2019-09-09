# data path
DATA_PATH = "./data"
TRASH_TRAIN_DATA_PATH = "./data"
TRASH_VALID_DATA_PATH = "./ai/data/trash2/valid"
DETECTOR_TRAIN_DATA_PATH = "./ai/data/detector/train"
DETECTOR_VALID_DATA_PATH = "./ai/data/detector/valid"

AE_CKPT_PATH = "./ai/ckpts/ae.pth"
CLF_CKPT_PATH = "./ai/ckpts/clf.pth"

# hyper parameters
BATCH_SIZE = 27
ETA = 1e-2
EPOCHS = 100

# image information
HEIGHT = 128
WIDTH = 128
IN_CHANNEL = 6

CONTRACTIVE_AE_LAMBDA = 1e-2

CUDA_DEVICES = [0]

# categories of trash
# TRASH_CAT = [
#     "can", "glass", "plastic"
# ]
TRASH_CAT = [
    "can", "glass", "paper", "plastic"
]

DETECTOR_CAT = [
    "nothing", "trash"
]