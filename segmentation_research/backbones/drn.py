from ..utils.blocks import conv_norm_act, residual_block, bottleneck_block
from ..utils.regularizers import WEIGHT_DECAY, WS_STD

def drn(input, normalization="batchnorm", regularizer=WEIGHT_DECAY, activation="relu", block_count=[1, 1, 2, 2, 2, 2, 1, 1], block=bottleneck_block):
    """
    Supports DRN-C Models from: https://arxiv.org/abs/1705.09914
    See reference pytorch implementation: https://github.com/fyu/drn
    returns feature map from all levels
    """
    # level 1
    l1 = conv_norm_act(input, 16, kernel_size=(7,7), dilation_rate=1, normalization=normalization, regularizer=regularizer, activation=activation)
    l1 = residual_block(l1, 32, 1, 2, normalization, regularizer, activation)
    # level 2
    l2 = residual_block(l1, 32, 1, 2, normalization, regularizer, activation)
    # level 3
    prev = l2
    for _ in range(block_count[2]-1):
        l3 = block(prev, filters=64, dilation_rate=1, strides=1, normalization=normalization, regularizer=regularizer, activation=activation)
        prev = l3
    l3 = block(l3, filters=64, dilation_rate=1, strides=2, normalization=normalization, regularizer=regularizer, activation=activation)
    # level 4
    prev = l3
    for _ in range(block_count[3]):
        l4 = block(prev, filters=128, dilation_rate=1, strides=1, normalization=normalization, regularizer=regularizer, activation=activation)
        prev = l4
    # level 5
    prev = l4
    for _ in range(block_count[4]):
        l5 = block(prev, filters=256, dilation_rate=2, strides=1, normalization=normalization, regularizer=regularizer, activation=activation)
        prev = l5
    # level 6
    prev = l5
    for _ in range(block_count[5]):
        l6 = block(prev, filters=512, dilation_rate=4, strides=1, normalization=normalization, regularizer=regularizer, activation=activation)
        prev = l6
    # level 7
    prev = l6
    for _ in range(block_count[6]):
        l7 = conv_norm_act(prev, filters=512, kernel_size=(3,3), dilation_rate=2, normalization=normalization, regularizer=regularizer, activation=activation)
    # level 8
    prev = l7
    for _ in range(block_count[6]):
        l8 = conv_norm_act(prev, filters=512, kernel_size=(3,3), dilation_rate=1, normalization=normalization, regularizer=regularizer, activation=activation)
    return [l1, l2, l3, l4, l5, l6, l7, l8]

def drn_c_26(input, normalization="batchnorm", regularizer=WEIGHT_DECAY, activation="relu"):
    return drn(input, normalization=normalization, regularizer=regularizer, activation=activation, block_count=[1, 1, 2, 2, 2, 2, 2, 2], block=residual_block)

def drn_c_105(input, normalization="batchnorm", regularizer=WEIGHT_DECAY, activation="relu"):
    return drn(input, normalization=normalization, regularizer=regularizer, activation=activation, block_count=[1, 1, 3, 4, 23, 2, 2, 2], block=bottleneck_block)