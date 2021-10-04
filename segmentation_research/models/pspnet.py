from tensorflow.keras.layers import Input, UpSampling2D, AveragePooling2D, Conv2D, concatenate, Dropout, Activation
from tensorflow_addons.layers import AdaptiveAveragePooling2D
from tensorflow.keras.models import Model
from ..backbones.drn import drn_c_105
from ..backbones.resnet_v2 import resnet50_v2
from ..utils.blocks import conv_norm_act
from ..utils.regularizers import WEIGHT_DECAY, WS_STD

FEATURE_RESOLUTION_PROPORTION = 8

def check_input_shape(input_shape, bins):
    for size in bins:
        assert (input_shape[0] // FEATURE_RESOLUTION_PROPORTION) % size == 0, f"input dimension {input_shape[0]} is not divisible by {FEATURE_RESOLUTION_PROPORTION*size}"
        assert (input_shape[1] // FEATURE_RESOLUTION_PROPORTION) % size == 0, f"input dimension {input_shape[1]} is not divisible by {FEATURE_RESOLUTION_PROPORTION*size}"

def StaticPPM(x, bin, pool_size, name_base, pool_filters=512, normalization="batchnorm", activation="relu", regularizer=WEIGHT_DECAY):
    """
    as in published pspnet
    """
    pool = AveragePooling2D(pool_size=pool_size, strides=pool_size, padding="same")(x)
    pool = conv_norm_act(
        pool, pool_filters, kernel_size=(1,1), normalization=normalization, 
        regularizer=regularizer, activation=activation, name_base=name_base)
    pool = UpSampling2D(pool_size, interpolation="bilinear")(pool)
    return pool

def AdaptivePPM(x, bin, pool_size, name_base, pool_filters=512, normalization="batchnorm", activation="relu", regularizer=WEIGHT_DECAY):
    """
    as in authors git repo with updated results
    """
    pool = AdaptiveAveragePooling2D((bin, bin))(x)
    pool = conv_norm_act(
        pool, pool_filters, kernel_size=(1,1), normalization=normalization, 
        regularizer=regularizer, activation=activation, name_base=name_base)
    pool = UpSampling2D(pool_size, interpolation="bilinear")(pool)
    return pool

def PSPNet(input_shape, num_classes, backbone=resnet50_v2, bins=[1, 2, 3, 6], post_upsample_proc=False, ext_features=None, ext_model=None, pool_filters=512, normalization="batchnorm", activation="relu", regularizer=WEIGHT_DECAY):
    """
    PSPNet
    https://arxiv.org/abs/1612.01105
    https://github.com/hszhao/semseg
    """
    check_input_shape(input_shape, bins)
    input = Input(shape=input_shape)
    # features = backbone(input, normalization=normalization, regularizer=regularizer, activation=activation)
    features = backbone(input, normalization=normalization, regularizer=regularizer, activation=activation)
    stage4_features = features[0]  # corresponding feature map you would have in resnet_stage4  
    final_features = features[1]
    prediction = None
    feature_map_resolution = input_shape[0] // FEATURE_RESOLUTION_PROPORTION, input_shape[1] // FEATURE_RESOLUTION_PROPORTION
    
    # four levels of pooling
    # 1x1
    pool1_size = (feature_map_resolution[0] // bins[0], feature_map_resolution[1] // bins[0])
    f1_up = AdaptivePPM(final_features, bins[0], pool1_size, name_base="p1", normalization=normalization, 
                    regularizer=regularizer, activation=activation)
    # 2x2
    pool2_size = (feature_map_resolution[0] // bins[1], feature_map_resolution[1] // bins[1])
    f2_up = AdaptivePPM(final_features, bins[1], pool2_size, name_base="p2", normalization=normalization, 
                    regularizer=regularizer, activation=activation)
    # 3x3
    pool3_size = (feature_map_resolution[0] // bins[2], feature_map_resolution[1] // bins[2])
    f3_up = AdaptivePPM(final_features, bins[2], pool3_size, name_base="p3", normalization=normalization, 
                    regularizer=regularizer, activation=activation)
    # 6x6
    pool4_size = (feature_map_resolution[0] // bins[3], feature_map_resolution[1] // bins[3])
    f4_up = AdaptivePPM(final_features, bins[3], pool4_size, name_base="p4", normalization=normalization, 
                    regularizer=regularizer, activation=activation)
    
    # concat with feature map
    features = concatenate([final_features, f1_up, f2_up, f3_up, f4_up])
    # aggregate feature pyramid with another conv_norm_act
    features = conv_norm_act(features, pool_filters, kernel_size=(3,3), normalization=normalization,
                            regularizer=regularizer, activation=activation, name_base="aggregation")
    # pyramid dropout
    features = Dropout(0.1, name="pyramid_dropout")(features)
    
    # AS IN THE ORIGINAL PAPER - FINAL CONV BEFORE UPSAMPLE
    if not post_upsample_proc:
        if ext_model is not None:
            ext_features = ext_model(input)[-1]  # for superpoint descriptors

        if ext_features is not None:
            features = concatenate([features, ext_features])
        
        # conv w/channels for class predictions
        prediction_features = Conv2D(filters=num_classes, kernel_size=(1,1), activation=None, name="final_conv")(features)
        # upsample to original image resolution
        upsample = UpSampling2D((FEATURE_RESOLUTION_PROPORTION,FEATURE_RESOLUTION_PROPORTION), 
                                interpolation="bilinear", name="end_upsample")(prediction_features)
        # softmax for prediction
        prediction = Activation("softmax", name="end_prediction")(upsample)

    # FOR MY OWN EXPERIMENTS - FINAL CONV AFTER UPSAMPLE
    else:
        upsample = UpSampling2D((FEATURE_RESOLUTION_PROPORTION,FEATURE_RESOLUTION_PROPORTION), 
                                interpolation="bilinear", name="end_upsample")(features)
        if ext_model is not None:
            ext_features = ext_model(input)

        if ext_features is not None:
            upsample = concatenate([upsample, ext_features])
        
        # conv + softmax
        prediction = Conv2D(filters=num_classes, kernel_size=(1,1), activation="softmax", name="end_prediction")(upsample)


    # AUXILLARY LOSS
    # make prediction off of stage4 of resnet base as well
    stage4_features = conv_norm_act(stage4_features, 256, kernel_size=(3,3), normalization=normalization,
                            regularizer=regularizer, activation=activation, name_base="stage4_predict_features")
    stage4_features = Dropout(0.1, name="stage4_dropout")(stage4_features)
    stage4_features = Conv2D(filters=num_classes, kernel_size=(1,1), activation=None, name="final_stage4_conv")(stage4_features)
    stage4_upsample = UpSampling2D(
        (FEATURE_RESOLUTION_PROPORTION,FEATURE_RESOLUTION_PROPORTION), 
        interpolation="bilinear", name="stage4_upsample")(stage4_features)
    stage4_prediction = Activation("softmax", name="stage4_prediction")(stage4_upsample)
    model = Model(input, [prediction, stage4_prediction])
    return model


# for auxillary loss, just call model.fit() with a dictionary using the keys "end_prediction", "stage4_prediction"
# e.g.
# loss_weights= {
            #       "stage4_prediction": 0.4, 
            #       "end_prediction": 1.0
            #   }