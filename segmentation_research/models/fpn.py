from tensorflow.keras.layers import Input, UpSampling2D, Conv2D, concatenate, add
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from ..backbones.resnet import resnet34, resnet50, resnet101
from ..utils.blocks import conv_norm_act, Conv2dNorm
from ..utils.regularizers import WEIGHT_DECAY, WS_STD


def aggregate(method, features):
    if method == "add":
        return add(features, name="aggregate_add")
    elif method == "concat":
        return concatenate(features, name="aggregate_concat")
    else:
        raise ValueError('aggregation must be either add or concat.')

def fpn_block(up_input, down_input, filters, interpolation="bilinear", regularizer=WEIGHT_DECAY, name_base=""):
    """
    connect upward (backbone) filter map with downstream fpn module using upsampling and 1x1 conv
    """
    # upstream branch
    shortcut = up_input
    # apply shortcut conv to input to make sure features match
    if K.int_shape(shortcut)[-1] != filters:
        shortcut = Conv2dNorm(
            filters, kernel_size=(1,1), padding="same", strides=1, 
            kernel_regularizer=regularizer, name_base="fpn_shortcut")(shortcut)
    # upsampling
    shortcut = UpSampling2D((2, 2), interpolation=interpolation, name=f"{name_base}_fpn_upsampling")(shortcut)
    # downstream branch
    pyramid = Conv2D(
        filters, kernel_size=(1,1), padding="same", strides=1, 
        kernel_regularizer=regularizer, name=f"{name_base}_conv")(down_input)
    return add([pyramid, shortcut])

def fpn_segmentation_head(input, filters=128, normalization="batchnorm", activation="relu", regularizer=WEIGHT_DECAY, name_base=""):
    """
    two conv->norm->act blocks
    """
    x = conv_norm_act(
        input, filters, normalization=normalization, activation=activation, 
        regularizer=regularizer, name_base=f"{name_base}_seg1")
    x = conv_norm_act(
        x, filters, normalization=normalization, activation=activation, 
        regularizer=regularizer, name_base=f"{name_base}_seg2")
    return x

def FPN(
    input_shape, num_classes, backbone=resnet101, interpolation="bilinear", filters=256, seg_filters=128, 
    agg_type="concat", normalization="batchnorm", activation="relu", regularizer=WEIGHT_DECAY):
    """
    FPN: https://arxiv.org/abs/1612.03144
    More implementation details for segmentation at http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf
    Aggregates features from backbone containing feature maps scale (1/4) to scale (1/32)
    Uses bilinear interpolation by default, rather than nearest as in paper
    Implementation nearly identical to https://github.com/qubvel/segmentation_models/blob/master/segmentation_models/models/fpn.py
    """
    input = Input(shape=input_shape)
    features = backbone(input, normalization=normalization, regularizer=regularizer, activation=activation)
    l1 = features[0]
    l2 = features[1]
    l3 = features[2]
    l4 = features[3]
    output_features = features[4]

    # feature pyramid aggregation
    p5 = fpn_block(output_features, l4, filters, interpolation=interpolation, name_base="p4")
    p4 = fpn_block(p5, l3, filters, interpolation=interpolation, name_base="p3")
    p3 = fpn_block(p4, l2, filters, interpolation=interpolation, name_base="p2")
    p2 = fpn_block(p3, l1, filters, interpolation=interpolation, name_base="p1")

    # add segmentation heads
    s5 = fpn_segmentation_head(p5, seg_filters, normalization=normalization, activation=activation, regularizer=regularizer, name_base="p4")
    s4 = fpn_segmentation_head(p4, seg_filters, normalization=normalization, activation=activation, regularizer=regularizer, name_base="p3")
    s3 = fpn_segmentation_head(p3, seg_filters, normalization=normalization, activation=activation, regularizer=regularizer, name_base="p2")
    s2 = fpn_segmentation_head(p2, seg_filters, normalization=normalization, activation=activation, regularizer=regularizer, name_base="p1")

    # upsample and aggregate
    s5 = UpSampling2D((8,8), interpolation=interpolation, name="agg_upsampling1")(s5)
    s4 = UpSampling2D((4,4), interpolation=interpolation, name="agg_upsampling2")(s4)
    s3 = UpSampling2D((2,2), interpolation=interpolation, name="agg_upsampling3")(s3)
    pyramid_map = aggregate(agg_type, [s5, s4, s3, s2])

    # prediction head
    combined = conv_norm_act(
        pyramid_map, seg_filters, normalization=normalization, activation=activation, 
        regularizer=regularizer, name_base="feature_combination")
    upsampled = UpSampling2D((2,2), interpolation=interpolation, name="final_upsampling")(combined)
    prediction = Conv2D(
        filters=num_classes, kernel_size=(3,3), padding="same", 
        activation="softmax", name="prediction")(upsampled)

    model = Model(input, prediction)
    return model