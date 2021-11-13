# main pspnet train script

from segmentation_research.models.pspnet import PSPNet
from datetime import datetime
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint, LearningRateScheduler
import tensorflow as tf
from segmentation_research.utils.metrics import NonBackgroundAccuracy, NonBackgroundMIoU
from segmentation_research.utils.losses import NonBackgroundSparseCCE 

from dataset import VALSET_SIZE, get_dataset, TRAINSET_SIZE, TESTSET_SIZE, BATCH_SIZE

base_log_dir = "/home/connor/Dev/segmentation_research/segmentation_research/experiments/ade20k/logs/"

# LOSS FN
AUX_LOSS = 0.4
BASE_LOSS = 1.0

# LR SCHEDULE
EPOCHS = 100
BASE_LR = 1e-2
MIN_LR = 1e-5

def poly_schedule(epoch, lr):
    POWER = 0.9
    factor = (1 - (epoch/EPOCHS))**POWER
    new_lr = ((BASE_LR - MIN_LR) * factor) + MIN_LR
    return new_lr

def common_callbacks(log_name, model_save_path, es_patience=7, lr_patience=2, min_lr=1e-5):
    early_stop = EarlyStopping(patience=es_patience, verbose=1)
    checkpoint = ModelCheckpoint(model_save_path, verbose=1, save_best_only=True)
    date_str = datetime.now().strftime("%d_%m_%Y_%H:%M/")
    log_dir = base_log_dir + log_name + "/" + date_str
    tensorboard = TensorBoard(log_dir, histogram_freq=1)
    scheduler = LearningRateScheduler(poly_schedule, verbose=1)
    return [early_stop, checkpoint, tensorboard, scheduler]

def get_baseline_model():
    model =  PSPNet(input_shape=(473, 473, 3), num_classes=151, aux_head=True)
    model.compile(optimizer=SGD(1e-2, momentum=0.9), 
              loss={
                  "stage4_prediction":NonBackgroundSparseCCE,
                  "end_prediction":NonBackgroundSparseCCE
              },
              loss_weights= {
                  "stage4_prediction": AUX_LOSS, 
                  "end_prediction": BASE_LOSS,
              },
              metrics={
                  "stage4_prediction": NonBackgroundAccuracy(), 
                  "end_prediction": NonBackgroundAccuracy()
              })
    print(model.summary())
    return model

if __name__ == "__main__":
    pspnet = get_baseline_model()
    dataset = get_dataset(crop_aug=True, classification_heads=None)
    pspnet.fit(
        dataset["train"],
        validation_data=dataset["val"],
        epochs=100, 
        steps_per_epoch=TRAINSET_SIZE // BATCH_SIZE,
        validation_steps=VALSET_SIZE // BATCH_SIZE,
        callbacks=common_callbacks(
            log_name=f"baseline_pspnet_473",
            model_save_path=f"models/baseline_pspnet_473_crop.h5"))
