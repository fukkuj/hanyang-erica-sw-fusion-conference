# data path
TRAIN_DATA_PATH = "./data/trash_collecting/train"
VALID_DATA_PATH = "./data/trash_collecting/valid"
TRASH_TRAIN_DATA_PATH = "./data/trash/train"
TRASH_VALID_DATA_PATH = "./data/trash/valid"
DETECTOR_TRAIN_DATA_PATH = "./data/detector/train"
DETECTOR_VALID_DATA_PATH = "./data/detector/valid"
TRASH_DATA_ALL_PATH = "./data/trash_collecting_all"

AE_CKPT_PATH = "./ai/ckpts/ae.pth"
CNN_CKPT_PATH = "./ai/ckpts/fcnn.pth"
CNN_CLF_CKPT_PATH = "./ai/ckpts/clf_cnn.pth"
VGG_CLF_CKPT_PATH = "./ai/ckpts/clf_vgg.pth"
AE_CLF_CKPT_PATH = "./ai/ckpts/clf_ae.pth"
DET_CKPT_PATH = "./ai/ckpts/det.pth"

# hyper parameters
BATCH_SIZE = 8
# BATCH_SIZE = 6
ETA = 1e-3
EPOCHS = 100

# image information
HEIGHT = 128
WIDTH = 128
IN_CHANNEL = 6

CONTRACTIVE_AE_LAMBDA = 1e-2

CUDA_DEVICES = [0]

TRASH_CAT = [
    "can", "glass", "paper", "plastic"
]

DETECTOR_CAT = [
    "nothing", "trash"
]
