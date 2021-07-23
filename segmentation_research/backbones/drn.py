from ..utils.blocks import conv_norm_act, residual_block, bottleneck_block
from ..utils.regularizers import WEIGHT_DECAY, WS_STD

def check_arch(arch):
    if arch not in ["B", "C"]:
        raise ValueError("Value for arch must be C or B")

def drn(
    input, normalization="batchnorm", regularizer=WEIGHT_DECAY, activation="relu", 
    block_count=[1, 1, 2, 2, 2, 2, 1, 1], arch="C", block=bottleneck_block):
    """
    Supports DRN_C Models from: https://arxiv.org/abs/1705.09914
    See reference pytorch implementation: https://github.com/fyu/drn
    returns feature map from all levels
    """
    check_arch(arch)
    # level 1
    l1 = conv_norm_act(
        input, 16, kernel_size=(7,7), dilation_rate=1, normalization=normalization, 
        regularizer=regularizer, activation=activation, name_base="level1_block1")
    l1 = residual_block(l1, 32, 1, 2, normalization, regularizer, activation, name_base="level1_block2")
    # level 2
    l2 = residual_block(l1, 32, 1, 2, normalization, regularizer, activation, name_base="level2_block2")
    # level 3
    prev = l2
    for i in range(block_count[2]-1):
        l3 = block(
            prev, filters=64, dilation_rate=1, strides=1, normalization=normalization, 
            regularizer=regularizer, activation=activation, name_base=f"level3_block{i}")
        prev = l3
    l3 = block(
        l3, filters=64, dilation_rate=1, strides=2, normalization=normalization, 
        regularizer=regularizer, activation=activation, name_base=f"level3_block{block_count[2]}")
    # level 4
    prev = l3
    for i in range(block_count[3]):
        l4 = block(
            prev, filters=128, dilation_rate=1, strides=1, normalization=normalization, 
            regularizer=regularizer, activation=activation, name_base=f"level4_block{i}")
        prev = l4
    # level 5
    prev = l4
    for i in range(block_count[4]):
        l5 = block(
            prev, filters=256, dilation_rate=2, strides=1, normalization=normalization, 
            regularizer=regularizer, activation=activation, name_base=f"level5_block{i}")
        prev = l5
    # level 6
    prev = l5
    for i in range(block_count[5]):
        l6 = block(
            prev, filters=512, dilation_rate=4, strides=1, normalization=normalization, 
            regularizer=regularizer, activation=activation, name_base=f"level6_block{i}")
        prev = l6
    # level 7
    prev = l6
    for i in range(block_count[6]):
        if arch == "C":
            l7 = conv_norm_act(
                prev, filters=512, kernel_size=(3,3), dilation_rate=2, normalization=normalization, 
                regularizer=regularizer, activation=activation, name_base=f"level7_block{i}_p1")
            l7 = conv_norm_act(
                l7, filters=512, kernel_size=(3,3), dilation_rate=2, normalization=normalization, 
                regularizer=regularizer, activation=activation, name_base=f"level7_block{i}_p2")
        elif arch == "B":
            l7 = block(
                prev, filters=512, dilation_rate=2, strides=1, normalization=normalization, 
                regularizer=regularizer, activation=activation, name_base=f"level7_block{i}")
        prev = l7
    # level 8
    prev = l7
    for i in range(block_count[7]):
        if arch == "C":
            l8 = conv_norm_act(
                prev, filters=512, kernel_size=(3,3), dilation_rate=1, normalization=normalization, 
                regularizer=regularizer, activation=activation, name_base=f"level8_block{i}_p1")
            l8 = conv_norm_act(
                l8, filters=512, kernel_size=(3,3), dilation_rate=1, normalization=normalization, 
                regularizer=regularizer, activation=activation, name_base=f"level8_block{i}_p2")
        elif arch == "B":
            l8 = block(
                prev, filters=512, dilation_rate=1, strides=1, normalization=normalization, 
                regularizer=regularizer, activation=activation, name_base=f"level8_block{i}")
        prev = l8
    return [l1, l2, l3, l4, l5, l6, l7, l8]

def drn_b_26(input, normalization="batchnorm", regularizer=WEIGHT_DECAY, activation="relu"):
    return drn(
        input, normalization=normalization, regularizer=regularizer, activation=activation, 
        block_count=[1, 1, 2, 2, 2, 2, 1, 1], block=residual_block, arch="B")

def drn_c_26(input, normalization="batchnorm", regularizer=WEIGHT_DECAY, activation="relu"):
    return drn(
        input, normalization=normalization, regularizer=regularizer, activation=activation, 
        block_count=[1, 1, 2, 2, 2, 2, 1, 1], block=residual_block, arch="C")

def drn_c_105(input, normalization="batchnorm", regularizer=WEIGHT_DECAY, activation="relu"):
    return drn(
        input, normalization=normalization, regularizer=regularizer, activation=activation, 
        block_count=[1, 1, 3, 4, 23, 2, 1, 1], block=bottleneck_block, arch="C")