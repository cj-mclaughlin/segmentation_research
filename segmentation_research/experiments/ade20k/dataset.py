from glob import glob
import tensorflow as tf
from config import DATASET_IMAGE_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH

from preprocessing import parse_image, load_image_train, load_image_test

DS_SEED = 42
BUFFER_SIZE = 1000
BATCH_SIZE = 8
VAL_SPLIT = 0.2

AUTOTUNE = tf.data.experimental.AUTOTUNE

DATASET_SIZE = len(glob(TRAIN_DATA_PATH + "*.jpg"))
TESTSET_SIZE = len(glob(TEST_DATA_PATH + "*.jpg"))
TRAINSET_SIZE = int((1 - VAL_SPLIT) * DATASET_SIZE)
VALSET_SIZE = DATASET_SIZE - TRAINSET_SIZE

def init_tf_datasets():
    """
    return initialized train/val/test datasets which call parse_img
    """
    full_dataset = tf.data.Dataset.list_files(TRAIN_DATA_PATH + "*.jpg", seed=DS_SEED)
    full_dataset = full_dataset.map(parse_image)

    train_size = int((1 - VAL_SPLIT) * DATASET_SIZE)

    train_dataset = full_dataset.take(train_size)
    val_dataset = full_dataset.skip(train_size)

    test_dataset = tf.data.Dataset.list_files(TEST_DATA_PATH + "*.jpg", seed=DS_SEED)
    test_dataset = test_dataset.map(parse_image)

    return train_dataset, val_dataset, test_dataset

def get_dataset(crop_aug=False, classification_heads=[1, 2, 3, 6], gt_heads=False):
    """
    args:
        batch_size
        classification head (none or list of scales)
        gt_heads (whether to feed ground truth classification head along with input image)
        crop_aug 
            if True (apply random scale + crop on train, center crop on train)
            else simply resize images to input size
        val_split
        classification_heads
    return dictionary of BatchDatasets for with keys for train/val/test
    """
    train_dataset, val_dataset, test_dataset = init_tf_datasets()
    dataset = {"train": train_dataset, "val": val_dataset, "test": test_dataset}
    
    load_train_with_heads = lambda x: load_image_train(x, classification_heads=classification_heads, crop_mode=crop_aug, gt_heads=gt_heads)
    load_test_with_heads = lambda x: load_image_test(x, classification_heads=classification_heads, crop_mode=crop_aug, gt_heads=gt_heads)

    dataset['train'] = dataset['train'].map(load_train_with_heads, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset['train'] = dataset['train'].shuffle(buffer_size=BUFFER_SIZE, seed=DS_SEED)
    dataset['train'] = dataset['train'].repeat()
    dataset['train'] = dataset['train'].batch(BATCH_SIZE)
    dataset['train'] = dataset['train'].prefetch(buffer_size=AUTOTUNE)

    dataset['val'] = dataset['val'].map(load_test_with_heads)
    dataset['val'] = dataset['val'].repeat()
    dataset['val'] = dataset['val'].batch(BATCH_SIZE)
    dataset['val'] = dataset['val'].prefetch(buffer_size=AUTOTUNE)

    dataset['test'] = dataset['test'].map(load_test_with_heads)
    dataset['test'] = dataset['test'].repeat()
    dataset['test'] = dataset['test'].batch(BATCH_SIZE)
    dataset['test'] = dataset['test'].prefetch(buffer_size=AUTOTUNE)

    return dataset