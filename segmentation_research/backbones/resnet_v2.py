from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, add
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.python.keras import regularizers

from ..utils.blocks import conv_norm_act, bottleneck_block

from segmentation_research.utils.regularizers import WEIGHT_DECAY

def add_blocks(inputs, filters, num_blocks, dilation_rate=1, stride_first=False, normalization="batchnorm", regularizer=WEIGHT_DECAY, activation="relu", name_base=""):
    x = bottleneck_block(inputs, filters, dilation_rate=dilation_rate, strides=2 if stride_first else 1, 
                        normalization=normalization, regularizer=regularizer, activation=activation, name_base=f"{name_base}_b1")
    for i in range(num_blocks - 1):
        x = bottleneck_block(x, filters, dilation_rate=dilation_rate, strides=1,
                            normalization=normalization, regularizer=regularizer, activation=activation, name_base=f"{name_base}_b{2+i}")
    return x


def resnet50_v2(inputs, normalization="batchnorm", regularizer=WEIGHT_DECAY, activation="relu", blocks=[3, 4, 6, 3]):
    """
    dilated-style resnet50v2, as in
    https://github.com/hszhao/semseg
    """
    x1 = conv_norm_act(inputs, filters=64, kernel_size=3, strides=2, normalization=normalization, regularizer=regularizer, activation=activation, name_base="conv1_0")
    x1 = conv_norm_act(x1, filters=64, kernel_size=3, strides=1, normalization=normalization, regularizer=regularizer, activation=activation, name_base="conv1_1")
    x1 = conv_norm_act(x1, filters=128, kernel_size=3, strides=1, normalization=normalization, regularizer=regularizer, activation=activation, name_base="conv1_2")
    pool = MaxPool2D(pool_size=(3,3), strides=(2,2), padding="same", name="conv2_pool")(x1)
    x2 = add_blocks(pool, 64, blocks[0], dilation_rate=1, stride_first=False, normalization=normalization, regularizer=regularizer, activation=activation, name_base="conv2")
    x3 = add_blocks(x2, 128, blocks[1], dilation_rate=1, stride_first=True, normalization=normalization, regularizer=regularizer, activation=activation, name_base="conv3")
    x4 = add_blocks(x3, 256, blocks[2], dilation_rate=2, stride_first=False, normalization=normalization, regularizer=regularizer, activation=activation, name_base="conv4")
    x5 = add_blocks(x4, 512, blocks[3], dilation_rate=4, stride_first=False, normalization=normalization, regularizer=regularizer, activation=activation, name_base="conv5")
    features = [x1, x2, x3, x4, x5]
    model = Model(inputs, x5)
    return model, features