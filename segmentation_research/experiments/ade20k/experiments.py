from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.layers import Input, Flatten, Conv2D, Dense, Reshape, Softmax, UpSampling2D, Lambda, concatenate, add, DepthwiseConv2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
import tensorflow as tf
from datetime import datetime
from tqdm import tqdm

from dataset import VALSET_SIZE, get_dataset, TRAINSET_SIZE, TESTSET_SIZE, BATCH_SIZE
from segmentation_research.utils.metrics import NonBackgroundAccuracy, NonBackgroundMIoU
from segmentation_research.utils.losses import NonBackgroundSparseCCE 

# 192x192 trained model with pooling loss
pretrained_model_path = "/home/connor/Dev/pspnet_pool.h5"
base_log_dir = "/home/connor/Dev/segmentation_research/segmentation_research/experiments/ade20k/logs/"

def get_metrics(y_true, y_pred):
    acc = NonBackgroundAccuracy()
    miou = NonBackgroundMIoU()
    acc.update_state(y_true, y_pred)
    miou.update_state(y_true, y_pred)
    return acc.result(), miou.result()

def eval_model(model, dataset):
    mious = 0
    accs = 0 
    BATCHES = TESTSET_SIZE//BATCH_SIZE
    for _ in tqdm(range(BATCHES)):
        for X, y in dataset['test'].take(1):
            pred = model.predict(X)
            acc, miou = get_metrics(y, pred)
            accs += acc
            mious += miou
    return accs/BATCHES, mious/BATCHES

def common_callbacks(log_name, es_patience=3, lr_patience=2, min_lr=1e-5):
    early_stop = EarlyStopping(patience=es_patience, verbose=1)
    plateau = ReduceLROnPlateau(factor=0.5, patience=lr_patience, min_lr=min_lr)
    date_str = datetime.now().strftime("%d_%m_%Y_%H:%M/")
    log_dir = base_log_dir + log_name + "/" + date_str
    tensorboard = TensorBoard(log_dir, histogram_freq=1)
    return [early_stop, plateau, tensorboard]

def classification_gt_model(pretrained_model, scales, image_shape=(192, 192, 3), end_shape=(192, 192, 151), num_nonlinearity=1):
    image = Input(shape=image_shape)

    pre_softmax_model = Model(pretrained_model.input, pretrained_model.get_layer("final_conv").output)
    pre_softmax_model.trainable = False

    pre_softmax = pre_softmax_model(image)

    classification = Input(shape=(scales[0], scales[0], 150))
    background = tf.expand_dims(tf.zeros_like(classification)[:,:,:,0], axis=-1)
    classification_with_background = tf.concat([background, classification], axis=-1)

    # upsample = classification_comb
    upsample = classification_with_background

    # upsample if needed
    if scales[-1] < image_shape[0]//8:
        upsample_factor = (int((image_shape[0]//8)/scales[-1]), int((image_shape[1]//8)/scales[-1]))
        upsample = UpSampling2D(upsample_factor, interpolation="bilinear", name="bias_upsample")(upsample)
    
    # concat --> conv1x1
    new_feat = concatenate([pre_softmax, upsample])
    for _ in range(num_nonlinearity):
        new_feat = Conv2D(filters=151, activation="relu", kernel_size=1, padding="same", use_bias=True)(new_feat)

    new_feat = Lambda(lambda img: tf.image.resize(img, size=(end_shape[0], end_shape[1]), method="bilinear", name="end_upsample"))(new_feat)
    segmentation = Softmax(name="final_prediction")(new_feat)
    return Model({"input_1":image, "input_2":classification}, segmentation)

def single_classification_head_exp(classification_head_size=6):
    classification_heads = [classification_head_size, 1, 1, 1]  # as current dataset always extracts 4 heads

    dataset = get_dataset(classification_heads=classification_heads)

    base_model = load_model(pretrained_model_path, compile=False)
    model = classification_gt_model(base_model, scales=[classification_head_size])

    print(model.summary())

    model.compile(
        optimizer=Adam(1e-3),
        loss={"final_prediction": NonBackgroundSparseCCE} )
    
    model.fit(
        dataset["train"],
        validation_data=dataset["val"],
        epochs=1, 
        steps_per_epoch=TRAINSET_SIZE // BATCH_SIZE,
        validation_steps=VALSET_SIZE // BATCH_SIZE,
        callbacks=common_callbacks(f"single_head_{classification_head_size}"))

if __name__ == "__main__":
    single_classification_head_exp(6)
