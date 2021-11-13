from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from tensorflow.keras.layers import Input, Flatten, Conv2D, Dense, Reshape, Softmax, UpSampling2D, Lambda, concatenate, add, GlobalAveragePooling2D, multiply, Activation
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.models import Model
import tensorflow as tf
from datetime import datetime
from tqdm import tqdm

from dataset import VALSET_SIZE, get_dataset, TRAINSET_SIZE, TESTSET_SIZE, BATCH_SIZE
from config import N_CLASSES
from segmentation_research.experiments.ade20k.config import IMG_SIZE
from segmentation_research.utils.metrics import NonBackgroundAccuracy, NonBackgroundMIoU, IntersectionUnionTarget
from segmentation_research.utils.losses import NonBackgroundSparseCCE 

# 192x192 trained model with pooling loss
pretrained_model_path = "/home/connor/Dev/pspnet_pool_v2_refit.h5"
base_log_dir = "/home/connor/Dev/segmentation_research/segmentation_research/experiments/ade20k/logs/"

def get_metrics(y_true, y_pred):
    iut = IntersectionUnionTarget()
    iut.update_state(y_true, y_pred)
    i, u, t = iut.result()
    return tf.cast(i, tf.float32), tf.cast(u, tf.float32), tf.cast(t, tf.float32)

def eval_model(model, dataset):
    """
    return overall accuracy, mean accuracy, mean iou on dataset
    """
    init = False
    intersections = tf.zeros(())
    unions = tf.zeros(())
    targets = tf.zeros(())
    BATCHES = TESTSET_SIZE//BATCH_SIZE
    # generate BATCHES x C array
    for _ in tqdm(range(BATCHES)):
        for X, y in dataset['test'].take(1):
            pred = model.predict(X)[-1] # index -1 for full model with multiple predictions
            intersection, union, target = get_metrics(y, pred)
            intersection = tf.expand_dims(intersection, 0)
            union = tf.expand_dims(union, 0)
            target = tf.expand_dims(target, 0)
            if not init: 
                init = True
                intersections = intersection
                unions = union
                targets = target
            else:
                intersections = tf.concat([intersections, intersection], axis=0)
                unions = tf.concat([unions, union], axis=0)
                targets = tf.concat([targets, target], axis=0)
    # iou/acc per class
    iou = tf.reduce_sum(intersections, axis=0) / (tf.reduce_sum(unions, axis=0) + 1e-10)
    acc = tf.reduce_sum(intersections, axis=0) / (tf.reduce_sum(targets, axis=0) + 1e-10)
    mean_iou = tf.reduce_mean(iou)
    mean_acc = tf.reduce_mean(acc)
    overall_acc = tf.reduce_sum(intersections) / (tf.reduce_sum(targets) + 1e-10)
    return overall_acc, mean_acc, mean_iou

def common_callbacks(log_name, model_save_path, es_patience=3, lr_patience=2, min_lr=1e-5):
    early_stop = EarlyStopping(patience=es_patience, verbose=1)
    checkpoint = ModelCheckpoint(model_save_path, verbose=1, save_best_only=True)
    plateau = ReduceLROnPlateau(factor=0.5, patience=lr_patience, min_lr=min_lr, verbose=1)
    date_str = datetime.now().strftime("%d_%m_%Y_%H:%M/")
    log_dir = base_log_dir + log_name + "/" + date_str
    tensorboard = TensorBoard(log_dir, histogram_freq=1)
    return [early_stop, plateau, checkpoint, tensorboard]

def classification_gt_model(pretrained_model, scales, num_nonlinearity=1):
    """
    explore limits of accuracy improvements if we are given ground truth classification labels
    """
    image = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    pre_softmax_model = Model(pretrained_model.input, pretrained_model.get_layer("final_conv").output)
    pre_softmax_model.trainable = False

    pre_softmax = pre_softmax_model(image)

    classification = Input(shape=(scales[0], scales[0], 150))
    background = tf.expand_dims(tf.zeros_like(classification)[:,:,:,0], axis=-1)
    classification_with_background = tf.concat([background, classification], axis=-1)

    upsample = classification_with_background

    # upsample if needed
    if scales[-1] < IMG_SIZE//8:
        upsample_factor = (int((IMG_SIZE//8)/scales[-1]), int((IMG_SIZE//8)/scales[-1]))
        upsample = UpSampling2D(upsample_factor, interpolation="bilinear", name="bias_upsample")(upsample)
    
    # concat --> conv1x1
    new_feat = concatenate([pre_softmax, upsample])
    for _ in range(num_nonlinearity):
        new_feat = Conv2D(filters=151, activation="relu", kernel_size=1, padding="same", use_bias=True)(new_feat)

    new_feat = Lambda(lambda img: tf.image.resize(img, size=(IMG_SIZE, IMG_SIZE), method="bilinear", name="end_upsample"))(new_feat)
    segmentation = Softmax(name="final_prediction")(new_feat)
    return Model({"input_1":image, "input_2":classification}, segmentation)

def tune_classification_model(pretrained_model, scales, num_nonlinearity=1):
    """
    pretrained PSPNet --> train only on classification heads, and their combination with frozen pre-softmax layers
    """
    image = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    pre_softmax = pretrained_model.get_layer("final_conv").output
    pool6 = pretrained_model.get_layer("pool4_supervision").output
    base_model = Model(pretrained_model.input, [pre_softmax, pool6])
    base_model.trainable = False

    pre_softmax, pool6 = base_model(image)

    classification = Input(shape=(scales[0], scales[0], 150))
    noisy_classification = (pool6 + classification) / 2
    # background = tf.expand_dims(tf.zeros_like(pool6)[:,:,:,0], axis=-1)
    # pool6_with_background = tf.concat([background, pool6], axis=-1)

    # upsample = pool6_with_background

    # upsample if needed
    # upsample_needed = int((IMG_SIZE//8)/scales[-1])
    # if scales[-1] < IMG_SIZE//8:
    #     upsample_factor = (upsample_needed, upsample_needed)
    #     upsample = UpSampling2D(upsample_factor, interpolation="bilinear", name="bias_upsample")(noisy_classification)
    
    # squeeze-excite channel attention
    # se = GlobalAveragePooling2D()(noisy_classification)
    # se = Reshape((1, 1, 150))(se)
    se = Conv2D(10, kernel_size=1, activation="relu", use_bias=False)(noisy_classification)
    se = Conv2D(150, kernel_size=1, activation="sigmoid", use_bias=False)(se)
    se_background_factor = tf.expand_dims(tf.zeros_like(se)[:,:,:,0], axis=-1)
    se = tf.concat([se_background_factor, se], axis=-1)
    # se = UpSampling2D((upsample_needed, upsample_needed), interpolation="bilinear")(se)

    new_feat = multiply([pre_softmax, se])

    # concat --> conv1x1
    # new_feat = concatenate([pre_softmax, upsample])
    # for _ in range(num_nonlinearity):
    #     new_feat = Conv2D(filters=151, activation="relu", kernel_size=1, padding="same", use_bias=True)(new_feat)

    new_feat = Lambda(lambda img: tf.image.resize(img, size=(IMG_SIZE, IMG_SIZE), method="bilinear", name="end_upsample"))(new_feat)
    segmentation = Softmax(name="final_prediction")(new_feat)
    return Model({"input_1":image, "input_2":classification}, segmentation)

def multi_head_classification_model(pretrained_model, scales, num_nonlinearity=1):
    """
    pretrained PSPNet --> train only on classification heads, and their combination with frozen pre-softmax layers
    """
    image = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    pre_softmax = pretrained_model.get_layer("final_conv").output
    pool1 = pretrained_model.get_layer("pool1_supervision").output
    pool2 = pretrained_model.get_layer("pool2_supervision").output
    pool3 = pretrained_model.get_layer("pool3_supervision").output
    pool6 = pretrained_model.get_layer("pool4_supervision").output
    base_model = Model(pretrained_model.input, [pre_softmax, pool1, pool2, pool3, pool6])
    base_model.trainable = False

    pre_softmax, pool1, pool2, pool3, pool6 = base_model(image)

    classification_1 = Input(shape=(scales[0], scales[0], 150))
    classification_2 = Input(shape=(scales[1], scales[1], 150))
    classification_3 = Input(shape=(scales[2], scales[2], 150))
    classification_4 = Input(shape=(scales[3], scales[3], 150))
    noisy_classification_1 = (pool1 + classification_1) / 2
    noisy_classification_2 = (pool2 + classification_2) / 2
    noisy_classification_3 = (pool3 + classification_3) / 2
    noisy_classification_4 = (pool6 + classification_4) / 2
    classifications = [noisy_classification_1, noisy_classification_2, noisy_classification_3, noisy_classification_4]

    flat_classifications = [Flatten()(c) for c in classifications]
    feat = concatenate(flat_classifications)
    for i in range(num_nonlinearity):
        if i == num_nonlinearity-1:
            feat = Dense(6*6*150, activation="sigmoid")(feat)
        else:
            feat = Dense(6*6*150, activation="relu")(feat)

    # combined = Dense(6*6*150, activation="sigmoid")(feat)
    upsample_amount = int((IMG_SIZE//8)/scales[-1])
    combined = Reshape((6, 6, 150))(feat)
    combined = UpSampling2D((upsample_amount, upsample_amount), interpolation="bilinear")(combined)
    # upsampled_classifications = []
    # for s_idx in range(len(scales)):
    #     upsample_amount = int((IMG_SIZE//8)/scales[s_idx])
    #     upsampled = UpSampling2D((upsample_amount, upsample_amount), interpolation="bilinear")(classifications[s_idx])
    #     upsampled_classifications.append(upsampled)
    
    # concat = concatenate(upsampled_classifications)
    # combined = Conv2D(filters=150, kernel_size=3, padding="same", activation="sigmoid")(concat)
    background_factor = tf.expand_dims(tf.zeros_like(combined)[:,:,:,0], axis=-1)
    combined = tf.concat([background_factor, combined], axis=-1)

    new_feat = multiply([pre_softmax, combined])

    # concat --> conv1x1
    # new_feat = concatenate([pre_softmax, upsample])
    # for _ in range(num_nonlinearity):
    #     new_feat = Conv2D(filters=151, activation="relu", kernel_size=1, padding="same", use_bias=True)(new_feat)

    new_feat = Lambda(lambda img: tf.image.resize(img, size=(IMG_SIZE, IMG_SIZE), method="bilinear", name="end_upsample"))(new_feat)
    segmentation = Softmax(name="final_prediction")(new_feat)
    inputs = {"input_1":image, "input_2":classification_1, "input_3":classification_2,  "input_4":classification_3,  "input_5":classification_4}
    return Model(inputs, segmentation)

def single_classification_head_exp(classification_head_size=6):
    classification_heads = [classification_head_size, 1, 1, 1]  # as current dataset always extracts 4 heads

    dataset = get_dataset(classification_heads=classification_heads, gt_heads=True)

    base_model = load_model(pretrained_model_path, compile=False)
    model = classification_gt_model(base_model, scales=[classification_head_size])

    print(model.summary())

    model.compile(
        optimizer=Adam(2e-3),
        loss={"final_prediction": NonBackgroundSparseCCE} )
    
    model.fit(
        dataset["train"],
        validation_data=dataset["val"],
        epochs=25, 
        steps_per_epoch=TRAINSET_SIZE // BATCH_SIZE,
        validation_steps=VALSET_SIZE // BATCH_SIZE,
        callbacks=common_callbacks(
            log_name=f"single_head_gt{classification_head_size}",
            model_save_path=f"models/gt{classification_head_size}_cat_conv1x1.h5")
    )

def tune_classification_arch_exp():
    dataset = get_dataset(classification_heads=[1, 2, 3, 6], gt_heads=True)

    base_model = load_model(pretrained_model_path, compile=False)
    model = multi_head_classification_model(base_model, scales=[1, 2, 3, 6], num_nonlinearity=2)

    print(model.summary())

    model.compile(
        optimizer=Adam(2e-3),
        loss={"final_prediction": NonBackgroundSparseCCE} )
    
    model.fit(
        dataset["train"],
        validation_data=dataset["val"],
        epochs=75, 
        steps_per_epoch=TRAINSET_SIZE // BATCH_SIZE,
        validation_steps=VALSET_SIZE // BATCH_SIZE,
        callbacks=common_callbacks(
            log_name=f"all_scale_mlp_mult2",
            model_save_path=f"models/all_scale_mlp_mult2.h5")
    )

if __name__ == "__main__":
    # MODEL BASELINE (288x288)
    # ACC 0.7388, MIOU 0.3384

    # tune_classification_arch_exp()
    dataset = get_dataset(classification_heads=None, gt_heads=False, crop_aug=True)
    test_path = "/home/connor/Dev/segmentation_research/segmentation_research/experiments/ade20k/models/baseline_pspnet_473_nocrop.h5"
    model = load_model(test_path, compile=False)
    overall_acc, mean_acc, miou = eval_model(model, dataset)
    print(overall_acc, mean_acc, miou)



# SINGLE HEAD EXP RESULTS (CAT+CONV1x1)
# MODEL BASELINE
# ACC 0.7388, MIOU 0.3384
# GT6
# ACC 86.08 MIOU 0.5452
# GT12
# ACC 87.08 MIOU 0.5600
# GT36
# ACC 93.98 MIOU 69.828

# SINGLE NOISY HEAD (6x6)
# CAT+CONV1x1
# 82.67 0.4639
# 'SPATIAL' SE
# 82.67 ACC 0.4684
# SE 
# 77.77 ACC 0.3824 MIOU

# ALL SCALES NOISY HEADS
# SPATIAL CONV + MULT + SE
# 0.8331 ACC 0.475 MIOU
# MLP 1 Layer
# 0.8349 ACC 0.505 MIOU
# SPATIAL CONV + MULT
# 0.8439 ACC 0.528 MIOU

# REAL HEAD ONLY - no finetuning
# CAT CONV1x1
# 75.30 acc 34.69 miou