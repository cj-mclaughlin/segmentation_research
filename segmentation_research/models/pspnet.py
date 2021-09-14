from tensorflow.keras.layers import Input, UpSampling2D, AveragePooling2D, Conv2D, concatenate, Dropout, Activation
from tensorflow.keras.models import Model
from ..backbones.drn import drn_c_105
from ..utils.blocks import conv_norm_act
from ..utils.regularizers import WEIGHT_DECAY, WS_STD
from ..utils.losses import WeightedCCE

FEATURE_RESOLUTION_PROPORTION = 8

def check_input_shape(input_shape, pool_sizes):
    for size in pool_sizes:
        assert (input_shape[0] // FEATURE_RESOLUTION_PROPORTION) % size == 0, f"input dimension {input_shape[0]} is not divisible by {FEATURE_RESOLUTION_PROPORTION*size}"
        assert (input_shape[1] // FEATURE_RESOLUTION_PROPORTION) % size == 0, f"input dimension {input_shape[1]} is not divisible by {FEATURE_RESOLUTION_PROPORTION*size}"

def PSPNet(input_shape, num_classes, backbone=drn_c_105, pool_sizes=[1, 2, 3, 6], post_upsample_proc=False, ext_features=None, pool_filters=512, normalization="batchnorm", activation="relu", regularizer=WEIGHT_DECAY):
    """
    PSPNet using DRN
    https://arxiv.org/abs/1612.01105
    """
    check_input_shape(input_shape, pool_sizes)
    input = Input(shape=input_shape)
    features = backbone(input, normalization=normalization, regularizer=regularizer, activation=activation)
    stage4_features = features[4]  # corresponding feature map you would have in resnet_stage4  
    final_features = features[-1]
    prediction = None
    feature_map_resolution = input_shape[0] // FEATURE_RESOLUTION_PROPORTION, input_shape[1] // FEATURE_RESOLUTION_PROPORTION
    
    # four levels of pooling
    # 1x1
    pool1_size = (feature_map_resolution[0] // pool_sizes[0], feature_map_resolution[1] // pool_sizes[0])
    pool1 = AveragePooling2D(pool_size=pool1_size, strides=pool1_size, padding="same")(final_features)
    # 2x2
    pool2_size = (feature_map_resolution[0] // pool_sizes[1], feature_map_resolution[1] // pool_sizes[1])
    pool2 = AveragePooling2D(pool_size=pool2_size, strides=pool2_size, padding="same")(final_features)
    # 3x3
    pool3_size = (feature_map_resolution[0] // pool_sizes[2], feature_map_resolution[1] // pool_sizes[2])
    pool3 = AveragePooling2D(pool_size=pool3_size, strides=pool3_size, padding="same")(final_features)
    # 6x6
    pool4_size = (feature_map_resolution[0] // pool_sizes[3], feature_map_resolution[1] // pool_sizes[3])
    pool4 = AveragePooling2D(pool_size=pool4_size, strides=pool4_size, padding="same")(final_features)
    
    # follow up each pooled feature map with 1x1 convolution
    f1 = conv_norm_act(
        pool1, pool_filters, kernel_size=(1,1), normalization=normalization, 
        regularizer=regularizer, activation=activation, name_base="p1")
    f2 = conv_norm_act(
        pool2, pool_filters, kernel_size=(1,1), normalization=normalization, 
        regularizer=regularizer, activation=activation, name_base="p2")
    f3 = conv_norm_act(
        pool3, pool_filters, kernel_size=(1,1), normalization=normalization, 
        regularizer=regularizer, activation=activation, name_base="p3")
    f4 = conv_norm_act(
        pool4, pool_filters, kernel_size=(1,1), normalization=normalization, 
        regularizer=regularizer, activation=activation, name_base="p4")
    
    # upsample to original size
    f1_up = UpSampling2D(pool1_size, interpolation="bilinear")(f1)
    f2_up = UpSampling2D(pool2_size, interpolation="bilinear")(f2)
    f3_up = UpSampling2D(pool3_size, interpolation="bilinear")(f3)
    f4_up = UpSampling2D(pool4_size, interpolation="bilinear")(f4)
    
    # concat with feature map
    features = concatenate([final_features, f1_up, f2_up, f3_up, f4_up])
    # aggregate feature pyramid with another conv_norm_act
    features = conv_norm_act(features, pool_filters, kernel_size=(3,3), normalization=normalization,
                            regularizer=regularizer, activation=activation, name_base="aggregation")
    # pyramid dropout
    features = Dropout(0.1, name="pyramid_dropout")(features)
    
    # AS IN THE ORIGINAL PAPER
    if not post_upsample_proc:
        # conv w/channels for class predictions
        prediction_features = Conv2D(filters=num_classes, kernel_size=(1,1), activation=None, name="final_conv")(features)
        # upsample to original image resolution
        upsample = UpSampling2D((FEATURE_RESOLUTION_PROPORTION,FEATURE_RESOLUTION_PROPORTION), 
                                interpolation="bilinear", name="end_upsample")(prediction_features)
        # softmax for prediction
        prediction = Activation("softmax")(upsample)

    # FOR MY OWN PURPOSES
    else:
        # upsample first
        upsample = UpSampling2D((FEATURE_RESOLUTION_PROPORTION,FEATURE_RESOLUTION_PROPORTION), 
                                interpolation="bilinear", name="end_upsample")(features)
        # concat external features at original resolution, if provided
        if ext_features is not None:
            upsample = concatenate([upsample, ext_features])
        # conv + softmax
        prediction = Conv2D(filters=num_classes, kernel_size=(1,1), activation="softmax", name="final_conv")(upsample)


    # AUXILLARY LOSS
    # make prediction off of stage4 of resnet base as well
    stage4_features = conv_norm_act(stage4_features, 256, kernel_size=(3,3), normalization=normalization,
                            regularizer=regularizer, activation=activation, name_base="stage4_predict_features")
    stage4_features = Dropout(0.1, name="stage4_dropout")(stage4_features)
    stage4_features = Conv2D(filters=num_classes, kernel_size=(1,1), activation=None, name="final_stage4_conv")(stage4_features)
    stage4_upsample = UpSampling2D(
        (FEATURE_RESOLUTION_PROPORTION,FEATURE_RESOLUTION_PROPORTION), 
        interpolation="bilinear", name="stage4_upsample")(stage4_features)
    stage4_prediction = Activation("softmax")(stage4_upsample)
    model = Model(input, [prediction, stage4_prediction])
    return model


# for auxillary loss, just call model.fit() with a dictionary using the keys "prediction", "stage4_prediction"

# def auxillary_loss(class_weights, alpha=0.4):
#     """
#     example auxillary loss for pspnet
#     """
#     pspnet_loss = {
#         "stage4_prediction": WeightedCCE(class_weights, alpha=alpha),
#         "end_prediction": WeightedCCE(class_weights, alpha=1.0)
#     }
#     return pspnet_loss