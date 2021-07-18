from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Activation, add
import tensorflow.keras.backend as K
from .normalization import get_norm_layer
from .regularizers import WEIGHT_DECAY, WS_STD


def conv_norm_act(input, filters, dilation_rate=1, strides=1, kernel_size=(3,3), normalization='batchnorm', regularizer=WEIGHT_DECAY, activation='relu'):
    """
    conv -> normalization -> activation block
    """
    x = Conv2D(filters, kernel_size, padding="same", strides=strides, dilation_rate=dilation_rate, kernel_regularizer=regularizer, use_bias=not normalization)(input)
    if normalization is not None:
        x = get_norm_layer(normalization, filters)(x)
    x = Activation(activation)(x)
    return x

def residual_block(input, filters, dilation_rate=1, strides=1, normalization='batchnorm', regularizer=WEIGHT_DECAY, activation="relu"):
    """
    Original ResNet identity/shortcut block
    """
    # shortcut if needed
    shortcut = input
    if K.int_shape(input)[-1] != filters or strides != 1:
        shortcut = Conv2D(filters, kernel_size=(1,1), strides=strides, kernel_regularizer=regularizer)(shortcut)
    # conv 1
    x1 = Conv2D(filters, kernel_size=(3,3), strides=1, padding="same", dilation_rate=dilation_rate, use_bias=not normalization, kernel_regularizer=regularizer)(input)
    if normalization is not None: 
        x1 = get_norm_layer(normalization, filters)(x1)
    x1 = Activation(activation)(x1)
    # conv 2
    x2 = Conv2D(filters, kernel_size=(3,3), strides=strides, padding="same", dilation_rate=dilation_rate, use_bias=not normalization, kernel_regularizer=regularizer)(x1)
    if normalization is not None: 
        x2 = get_norm_layer(normalization, filters)(x2)
    # residual
    residual = add([shortcut, x2])
    residual = Activation(activation)(residual)
    return residual

def bottleneck_block(input, filters, dilation_rate=1, strides=1, normalization='batchnorm', regularizer=WEIGHT_DECAY, activation="relu"):
    """
    Original ResNet bottleneck block
    """
    # shortcut if needed
    shortcut = input
    if K.int_shape(input)[-1] != filters*4 or strides != 1:
        shortcut = Conv2D(filters*4, kernel_size=(1,1), strides=strides, kernel_regularizer=regularizer)(shortcut)
    # conv 1
    x1 = Conv2D(filters, kernel_size=(1,1), strides=1, padding="same", dilation_rate=1, use_bias=not normalization, kernel_regularizer=regularizer)(input)
    if normalization is not None: 
        x1 = get_norm_layer(normalization, filters)(x1)
    x1 = Activation(activation)(x1)
    # conv 2
    x2 = Conv2D(filters, kernel_size=(3,3), strides=1, padding="same", dilation_rate=dilation_rate, use_bias=not normalization, kernel_regularizer=regularizer)(x1)
    if normalization is not None: 
        x2 = get_norm_layer(normalization, filters)(x2)
    x2 = Activation(activation)(x2)
    # conv 3
    x3 = Conv2D(filters*4, kernel_size=(1, 1), strides=strides, padding="same", dilation_rate=dilation_rate, use_bias=not normalization, kernel_regularizer=regularizer)(x2)
    if normalization is not None: 
        x3 = get_norm_layer(normalization, filters)(x3)
    # residual
    residual = add([shortcut, x3])
    residual = Activation(activation)(residual)
    return residual

def inverted_bottleneck_block(input, filters, dilation_rate=1, strides=1, normalization='batchnorm', regularizer=WEIGHT_DECAY, activation="relu", expansion=6):
    """
    Mobilenetv2 block with fixed alpha=1
    ref: https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet_v2.py
    """
    in_channels = K.int_shape(input)[-1]
    # shortcut if needed
    shortcut = input
    if K.int_shape(input)[-1] != filters or strides != 1:
        shortcut = Conv2D(filters, kernel_size=(1,1), strides=strides, kernel_regularizer=regularizer)(shortcut)
    # conv 1
    x1 = Conv2D(expansion * in_channels, kernel_size=(1,1), strides=1, padding="same", dilation_rate=1, use_bias=not normalization, kernel_regularizer=regularizer)(input)
    if normalization is not None: 
        x1 = get_norm_layer(normalization, expansion * in_channels)(x1)
    x1 = Activation(activation)(x1)
    # conv 2
    x2 = DepthwiseConv2D(kernel_size=(3,3), strides=strides, padding="same", dilation_rate=dilation_rate, use_bias=not normalization, kernel_regularizer=regularizer)(x1)
    if normalization is not None: 
        x2 = get_norm_layer(normalization, expansion * in_channels)(x2)
    x2 = Activation(activation)(x2)
    # conv 3
    x3 = Conv2D(filters, kernel_size=(1,1), strides=1, padding="same", use_bias=not normalization, kernel_regularizer=regularizer)(x2)
    if normalization is not None: 
        x3 = get_norm_layer(normalization, filters)(x3)
    # residual
    residual = add([shortcut, x3])
    return residual