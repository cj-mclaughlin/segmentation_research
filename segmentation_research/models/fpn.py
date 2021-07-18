from tensorflow.keras.layers import Input, UpSampling2D, AveragePooling2D, Conv2D, concatenate, add
from tensorflow.keras.models import Model
from ..backbones.drn import drn_c_105
from ..utils.blocks import conv_norm_act
from ..utils.regularizers import WEIGHT_DECAY, WS_STD

def aggregate(method, features):
    if method == "add":
        return add(features)
    elif method == "concat":
        return concatenate(features)
    else:
        raise ValueError('aggregation must be either add or concat.')

def fpn(input_shape, num_classes, backbone=drn_c_105, filters=256, aggregation="add", normalization="batchnorm", activation="relu", regularizer=WEIGHT_DECAY):
    """
    FPN using DRN
    https://arxiv.org/abs/1612.03144
    """
    pass