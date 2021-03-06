from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Activation, add
import tensorflow.keras.backend as K
from .normalization import get_norm_layer
from .regularizers import WEIGHT_DECAY

def Conv2dNorm(
    filters,
    kernel_size=(3,3),
    strides=(1, 1),
    padding='same',
    dilation_rate=(1, 1),
    kernel_initializer='he_normal',
    kernel_regularizer=WEIGHT_DECAY,
    normalization=None,
    name_base=""):

    def wrapper(input):
        x = Conv2D(
            filters, 
            kernel_size, 
            padding=padding, 
            strides=strides, 
            dilation_rate=dilation_rate, 
            kernel_regularizer=kernel_regularizer, 
            kernel_initializer=kernel_initializer,
            use_bias=not normalization,
            name=f"{name_base}_conv"
        )(input)
        if normalization is not None:
            x = get_norm_layer(normalization, filters, name_base)(x)
        return x

    return wrapper

def conv_norm_act(
    input, filters, dilation_rate=1, strides=1, kernel_size=(3,3), normalization='batchnorm', 
    regularizer=WEIGHT_DECAY, activation='relu', name_base=""):
    """
    conv -> normalization -> activation block
    """
    x = Conv2dNorm(
        filters, 
        kernel_size, 
        strides=strides, 
        dilation_rate=dilation_rate, 
        kernel_regularizer=regularizer,
        normalization=normalization,
        name_base=name_base)(input)
    x = Activation(activation, name=f"{name_base}_act")(x)
    return x

def residual_block(
    input, filters, dilation_rate=1, strides=1, normalization='batchnorm', 
    regularizer=WEIGHT_DECAY, activation="relu", name_base=""):
    """
    Original ResNet identity/shortcut block
    """
    # shortcut if needed
    shortcut = input
    if K.int_shape(input)[-1] != filters or strides != 1:
        shortcut = Conv2dNorm(
            filters, 
            kernel_size=(1,1), 
            strides=strides,
            kernel_regularizer=regularizer,
            normalization=normalization,
            name_base=f"{name_base}_shortcut")(shortcut)
    # conv 1
    x1 = Conv2dNorm(
        filters, 
        kernel_size=(3,3), 
        strides=1, 
        dilation_rate=dilation_rate, 
        kernel_regularizer=regularizer,
        normalization=normalization,
        name_base=f"{name_base}_p1")(input)
    x1 = Activation(activation, name=f"{name_base}_act1")(x1)
    # conv 2
    x2 = Conv2dNorm(
        filters, 
        kernel_size=(3,3), 
        strides=strides, 
        dilation_rate=dilation_rate, 
        kernel_regularizer=regularizer,
        normalization=normalization,
        name_base=f"{name_base}_p2")(input)
    # residual
    residual = add([shortcut, x2], name=f"{name_base}_add")
    residual = Activation(activation, name=f"{name_base}_final_act")(residual)
    return residual

def bottleneck_block(
    input, filters, dilation_rate=1, strides=1, normalization='batchnorm', 
    regularizer=WEIGHT_DECAY, activation="relu", name_base=""):
    """
    Original ResNet bottleneck block
    """
    # shortcut if needed
    shortcut = input
    if K.int_shape(input)[-1] != filters*4 or strides != 1:
        shortcut = Conv2dNorm(
            filters*4, 
            kernel_size=(1,1), 
            strides=strides,
            kernel_regularizer=regularizer,
            normalization=normalization,
            name_base=f"{name_base}_shortcut")(shortcut)
    # conv 1
    x1 = Conv2dNorm(
        filters, 
        kernel_size=(1,1), 
        strides=1,
        kernel_regularizer=regularizer,
        normalization=normalization,
        name_base=f"{name_base}_p1")(input)
    x1 = Activation(activation, name=f"{name_base}_act1")(x1)
    # conv 2
    x2 = Conv2dNorm(
        filters, 
        kernel_size=(3,3), 
        strides=1, 
        dilation_rate=dilation_rate, 
        kernel_regularizer=regularizer,
        normalization=normalization,
        name_base=f"{name_base}_p2")(x1)
    x2 = Activation(activation, name=f"{name_base}_act2")(x2)
    # conv 3
    x3 = Conv2dNorm(
        filters*4, 
        kernel_size=(1,1), 
        strides=strides,
        kernel_regularizer=regularizer,
        normalization=normalization,
        name_base=f"{name_base}_p3")(x2)
    # residual
    residual = add([shortcut, x3], name=f"{name_base}_add")
    residual = Activation(activation, name=f"{name_base}_final_act")(residual)
    return residual

def inverted_bottleneck_block(
    input, filters, dilation_rate=1, strides=1, normalization='batchnorm', 
    regularizer=WEIGHT_DECAY, activation="relu", expansion=6, name_base=""):
    """
    Mobilenetv2 block with fixed alpha=1
    ref: https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet_v2.py
    """
    in_channels = K.int_shape(input)[-1]
    # shortcut if needed
    shortcut = input
    if K.int_shape(input)[-1] != filters or strides != 1:
        shortcut = Conv2dNorm(
            filters, 
            kernel_size=(1,1), 
            strides=strides,
            kernel_regularizer=regularizer,
            normalization=normalization)(shortcut)
    # conv 1
    x1 = Conv2dNorm(
        filters=expansion * in_channels, 
        kernel_size=(1,1), 
        strides=1,
        kernel_regularizer=regularizer,
        normalization=normalization)(input)
    x1 = Activation(activation)(x1)
    # conv 2
    x2 = DepthwiseConv2D(
        kernel_size=(3,3), 
        strides=strides, 
        dilation_rate=dilation_rate, 
        use_bias=not normalization, 
        kernel_regularizer=regularizer)(x1)
    if normalization is not None: 
        x2 = get_norm_layer(normalization, expansion * in_channels)(x2)
    x2 = Activation(activation)(x2)
    # conv 3
    x3 = Conv2dNorm(
        filters, 
        kernel_size=(1,1), 
        strides=1,
        kernel_regularizer=regularizer,
        normalization=normalization)(x2)
    # residual
    residual = add([shortcut, x3])
    return residual