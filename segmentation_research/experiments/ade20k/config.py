# ade20k data config

# paths
DATA_ROOT = "/home/connor/Dev/Data/"
DATASET_IMAGE_PATH = DATA_ROOT + "ADEChallengeData2016/images/"
TRAIN_DATA_PATH = DATASET_IMAGE_PATH + "training/"
TEST_DATA_PATH = DATASET_IMAGE_PATH + "validation/"

CLASS_DICT_PATH = "classes.csv"

# Image size that we are going to use
IMG_SIZE = 192 # 192 (small/fast) or 473 (original PSPNet)
# Our images are RGB (3 channels)
N_CHANNELS = 3
# Scene Parsing has 150 classes + `not labeled`
N_CLASSES = 151