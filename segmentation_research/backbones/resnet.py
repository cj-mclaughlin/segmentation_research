from tensorflow.keras.layers import MaxPool2D
from ..utils.blocks import conv_norm_act, residual_block, bottleneck_block
from ..utils.regularizers import WEIGHT_DECAY, WS_STD
from tensorflow.keras.models import Model

def resnet(
    inputs, normalization="batchnorm", regularizer=WEIGHT_DECAY, 
    activation="relu", block_count=[3, 4, 6, 3], block=bottleneck_block):
    """
    Supports ResNet Models from https://arxiv.org/abs/1512.03385
    """
    # level 1
    l1 = conv_norm_act(
        inputs, 64, kernel_size=(7,7), strides=2, normalization=normalization, 
        regularizer=regularizer, activation=activation, name_base="conv1")
    # level 2
    l2 = MaxPool2D(pool_size=(3,3,), strides=2, padding="same", name="conv2_pool")(l1)
    for i in range(block_count[0]):
        l2 = block(
            l2, filters=64, strides=1, normalization=normalization, 
            regularizer=regularizer, activation=activation, name_base=f"conv2_b{i}")
    # level 3
    l3 = block(
        l2, filters=128, strides=2, normalization=normalization, 
        regularizer=regularizer, activation=activation, name_base=f"conv3_b0")
    for i in range(block_count[1]-1):
        l3 = block(
            l3, filters=128, strides=1, normalization=normalization, 
            regularizer=regularizer, activation=activation, name_base=f"conv3_b{i+1}")
    # level 4
    l4 = block(
        l3, filters=256, strides=2, normalization=normalization, 
        regularizer=regularizer, activation=activation, name_base=f"conv4_b0")
    for i in range(block_count[1]-1):
        l4 = block(
            l4, filters=256, strides=1, normalization=normalization, 
            regularizer=regularizer, activation=activation, name_base=f"conv4_b{i+1}")
    # level 5
    l5 = block(
        l4, filters=512, strides=2, normalization=normalization, 
        regularizer=regularizer, activation=activation, name_base=f"conv5_b0")
    for i in range(block_count[1]-1):
        l5 = block(
            l5, filters=512, strides=1, normalization=normalization, 
            regularizer=regularizer, activation=activation, name_base=f"conv5_b{i+1}")
    model = Model(inputs, l5)
    features = [l1, l2, l3, l4, l5]
    return model, features

def resnet18(inputs, normalization="batchnorm", regularizer=WEIGHT_DECAY, activation="relu"):
    return resnet(
        inputs, normalization=normalization, regularizer=regularizer, 
        activation=activation, block_count=[2, 2, 2, 2], block=residual_block)

def resnet34(inputs, normalization="batchnorm", regularizer=WEIGHT_DECAY, activation="relu"):
    return resnet(
        inputs, normalization=normalization, regularizer=regularizer, 
        activation=activation, block_count=[3, 4, 6, 3], block=residual_block)

def resnet50(inputs, normalization="batchnorm", regularizer=WEIGHT_DECAY, activation="relu"):
    return resnet(
        inputs, normalization=normalization, regularizer=regularizer, 
        activation=activation, block_count=[3, 4, 6, 3], block=bottleneck_block)

def resnet101(inputs, normalization="batchnorm", regularizer=WEIGHT_DECAY, activation="relu"):
    return resnet(
        inputs, normalization=normalization, regularizer=regularizer, 
        activation=activation, block_count=[3, 4, 23, 3], block=bottleneck_block)

def resnet152(inputs, normalization="batchnorm", regularizer=WEIGHT_DECAY, activation="relu"):
    return resnet(
        inputs, normalization=normalization, regularizer=regularizer, 
        activation=activation, block_count=[3, 8, 36, 3], block=bottleneck_block)