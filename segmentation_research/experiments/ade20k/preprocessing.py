import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import numpy as np

from config import IMG_SIZE, N_CLASSES
from classification_utils import extract_mask_classes

# use image statistics from ImageNet
MEAN = [0.485, 0.456, 0.406]
MEAN_SCALE = [int(255 * m) for m in MEAN]

STD = [0.229, 0.224, 0.225]
STD_SCALE = [int(255 * s) for s in STD]

@tf.function
def parse_image(img_path: str) -> dict:
    """
    Initial Image Parsing Helper
    in: full path to image
    out: dict {image: (image data), segmentation_mask: (annotation data), x, y: image size}
    """
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)
    shape = tf.shape(image)
    x = tf.cast(shape[0], tf.float32) # cast to float for later use
    y = tf.cast(shape[1], tf.float32)

    # Example ADE Paths:
    # .../trainset/images/training/ADE_train_00000001.jpg
    # .../trainset/annotations/training/ADE_train_00000001.png
    mask_path = tf.strings.regex_replace(img_path, "images", "annotations")
    mask_path = tf.strings.regex_replace(mask_path, "jpg", "png")
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)

    return {'image': image, "x":x, "y":y, 'segmentation_mask': mask}

@tf.function
def normalize(input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
    """
    Takes in uint8 input image/mask
    1) Rescale image to 0-1 and convert to float
    2) Remove imagenet mean and divide by imagenet std
    return normalized image and mask
    """
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_image = input_image - MEAN
    input_image = input_image / STD
    return input_image, input_mask

# @tf.function
def center_crop(img, mask, width, height, new_width, new_height):
    """
    tf function compatible implementation of center cropping
    """
    width, height = tf.cast(width, tf.float32), tf.cast(height, tf.float32)
    new_width, new_height = tf.cast(new_width, tf.float32), tf.cast(new_height, tf.float32)
    left = int(tf.math.ceil((width - new_width) / 2))
    right = int(width - tf.math.floor((width - new_width) / 2))

    top = int(tf.math.ceil((height - new_height) / 2))
    bottom = int(height - tf.math.floor((height - new_height) / 2))

    center_cropped_mask = mask[left:right, top:bottom]
    if len(img.shape) == 2:
        center_cropped_img = img[left:right, top:bottom]
    else:
        center_cropped_img = img[left:right, top:bottom, ...]

    return center_cropped_img, center_cropped_mask

def pad_to_size(image, mask, x, y, new_x=IMG_SIZE, new_y=IMG_SIZE, pad_mode="rand"):
    # pad if needed, using constant values from imagenet mean
    padding_y = tf.zeros(shape=(2,), dtype=tf.int32)
    padding_x = tf.zeros(shape=(2,), dtype=tf.int32)
    pad_top, pad_left = 0, 0 
    if x < new_x:
        num_pad_x = int(new_x - x)
        if pad_mode == "rand":
            pad_left = tf.experimental.numpy.random.randint(low=0, high=num_pad_x, dtype=tf.int32, size=1)[0]
        else: # even padding
            pad_left = int(num_pad_x / 2)
        pad_right = num_pad_x - pad_left
        padding_x = tf.stack([pad_left, pad_right], axis=0)
    if y < new_y:
        num_pad_y = int(new_y - y)
        if pad_mode == "rand":
            pad_top = tf.experimental.numpy.random.randint(low=0, high=num_pad_y, dtype=tf.int32, size=1)[0]
        else: # even padding
            pad_top = int(num_pad_y / 2)
        pad_bottom = num_pad_y - pad_top
        padding_y = tf.stack([pad_top, pad_bottom], axis=0)
    if x < new_x or y < new_y:
        padding_chan = tf.zeros_like(padding_x)
        padding = tf.stack([padding_x, padding_y, padding_chan])
        pad_r = tf.pad(image[:,:,0:1], padding, mode="CONSTANT", constant_values=MEAN_SCALE[0])
        pad_g = tf.pad(image[:,:,1:2], padding, mode="CONSTANT", constant_values=MEAN_SCALE[1])
        pad_b = tf.pad(image[:,:,2:3], padding, mode="CONSTANT", constant_values=MEAN_SCALE[2])
        image = tf.concat([pad_r, pad_g, pad_b], axis=-1)
        mask = tf.pad(mask, padding, mode="CONSTANT", constant_values=0)
    return image, mask

@tf.function
def load_image_train(datapoint: dict, classification_heads=[1, 2, 3, 6], crop_mode=False, gt_heads=False) -> tuple:
    """
    apply image augmentation with given parameters 
    """
    if crop_mode:
        # random scale between 0.5 and 2.0 without changing aspect ratio
        scale = np.random.uniform(low=0.5, high=2.0, size=1)
        scale_x, scale_y = int(scale*datapoint["x"]), int(scale*datapoint["y"])
        new_size = tf.concat([scale_x, scale_y], axis=0)
        input_image = tf.image.resize(datapoint['image'], new_size, method="bilinear")
        input_mask = tf.image.resize(datapoint['segmentation_mask'], new_size, method="nearest")

    else:
        # no random scale
        input_image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE), method="bilinear")
        input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMG_SIZE, IMG_SIZE), method="nearest")

    # random rotate with stength ~ 0.1
    rotate = tf.random.uniform(shape=[1], minval=-0.175, maxval=0.175)
    if tf.random.uniform(()) > 0.5:
        rot_r = tf.expand_dims(tfa.image.rotate(input_image[:,:,0], angles=rotate, interpolation="bilinear", fill_mode="constant", fill_value=MEAN_SCALE[0]), axis=-1)
        rot_g = tf.expand_dims(tfa.image.rotate(input_image[:,:,1], angles=rotate, interpolation="bilinear", fill_mode="constant", fill_value=MEAN_SCALE[1]), axis=-1)
        rot_b = tf.expand_dims(tfa.image.rotate(input_image[:,:,2], angles=rotate, interpolation="bilinear", fill_mode="constant", fill_value=MEAN_SCALE[2]), axis=-1)
        input_image = tf.concat([rot_r, rot_g, rot_b], axis=-1)
        input_mask = tfa.image.rotate(input_mask, angles=rotate, interpolation="nearest", fill_mode="constant", fill_value=0)

    # color jitter contrast with strength ~ 0.1
    # input_image = tf.image.random_brightness(input_image, max_delta=0.1)
    # input_image = tf.image.random_contrast(input_image, lower=0.1, upper=1.1)
    # input_image = tf.image.random_saturation(input_image, lower=0.1, upper=1.1)
    # input_image = tf.image.random_hue(input_image, 0.02)
    # input_image = tf.clip_by_value(input_image, 0.0, 255.0)

    # random flip
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    # gaussian blur
    if tf.random.uniform(()) > 0.5:
        input_image = tfa.image.gaussian_filter2d(input_image)
    
    if crop_mode:
        # pad if needed, using constant values from imagenet mean
        input_image, input_mask = pad_to_size(input_image, input_mask, x=scale_x, y=scale_y, new_x=IMG_SIZE, new_y=IMG_SIZE, pad_mode="rand")
        
        # IMG-SIZED crop
        seed = (np.random.randint(low=0, high=100), np.random.randint(low=0, high=100))
        input_image = tf.image.stateless_random_crop(input_image, size=(IMG_SIZE, IMG_SIZE, 3), seed=seed)
        input_mask = tf.image.stateless_random_crop(input_mask, size=(IMG_SIZE, IMG_SIZE, 1), seed=seed)

    # normalize inputs
    input_image, input_mask = normalize(input_image, input_mask)

    X, y = input_image, input_mask

    if classification_heads is not None:
        # extract classification heads
        pool1_matrix, pool2_matrix, pool3_matrix, pool4_matrix = tf.numpy_function(extract_mask_classes, inp=[input_mask, classification_heads], Tout=(tf.float32, tf.float32, tf.float32, tf.float32))
        pool1_matrix = tf.reshape(pool1_matrix, (classification_heads[0], classification_heads[0], N_CLASSES-1))
        pool2_matrix = tf.reshape(pool2_matrix, (classification_heads[1], classification_heads[1], N_CLASSES-1))
        pool3_matrix = tf.reshape(pool3_matrix, (classification_heads[2], classification_heads[2], N_CLASSES-1))
        pool4_matrix = tf.reshape(pool4_matrix, (classification_heads[3], classification_heads[3], N_CLASSES-1))
        if gt_heads:
            X = {"input_1": input_image, "input_2": pool1_matrix, "input_3": pool2_matrix, "input_4": pool3_matrix, "input_5": pool4_matrix}
        y = {"stage4_prediction": input_mask, "end_prediction": input_mask, 
            'classification_head':pool1_matrix, 'classification_head_1':pool2_matrix, 
            'classification_head_2':pool3_matrix, 'classification_head_3':pool4_matrix }

    return X, y

@tf.function
def load_image_test(datapoint: dict, crop_mode=False, classification_heads=[1, 2, 3, 6], gt_heads=False) -> tuple:
    # resize or center crop
    if crop_mode:
        input_image, input_mask = pad_to_size(
            datapoint['image'], datapoint['segmentation_mask'], x=datapoint['x'], y=datapoint['y'], new_x=IMG_SIZE, new_y=IMG_SIZE, pad_mode="center")
        padded_size_x, padded_size_y = IMG_SIZE, IMG_SIZE
        if datapoint['x'] > IMG_SIZE:
            padded_size_x = tf.cast(datapoint['x'], tf.int32)
        if datapoint['y'] > IMG_SIZE:
            padded_size_y = tf.cast(datapoint['y'], tf.int32)

        input_image, input_mask = center_crop(input_image, input_mask, padded_size_x, padded_size_y, IMG_SIZE, IMG_SIZE)
    else:
        input_image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE), method="bilinear")
        input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMG_SIZE, IMG_SIZE), method="nearest")

    input_image, input_mask = normalize(input_image, input_mask)

    X = input_image
    y = input_mask

    if classification_heads is not None:
        # extract classification heads
        pool1_matrix, pool2_matrix, pool3_matrix, pool4_matrix = tf.numpy_function(extract_mask_classes, inp=[input_mask, classification_heads], Tout=(tf.float32, tf.float32, tf.float32, tf.float32))
        pool1_matrix = tf.reshape(pool1_matrix, (classification_heads[0], classification_heads[0], N_CLASSES-1))
        pool2_matrix = tf.reshape(pool2_matrix, (classification_heads[1], classification_heads[1], N_CLASSES-1))
        pool3_matrix = tf.reshape(pool3_matrix, (classification_heads[2], classification_heads[2], N_CLASSES-1))
        pool4_matrix = tf.reshape(pool4_matrix, (classification_heads[3], classification_heads[3], N_CLASSES-1))
        if gt_heads:
            X = {"input_1": input_image, "input_2": pool1_matrix, "input_3": pool2_matrix, "input_4": pool3_matrix, "input_5": pool4_matrix}
        y = {"stage4_prediction": input_mask, "end_prediction": input_mask, 
            'classification_head':pool1_matrix, 'classification_head_1':pool2_matrix, 
            'classification_head_2':pool3_matrix, 'classification_head_3':pool4_matrix }
    return X, y

if __name__ == '__main__':
    # debugging
    # original size > IMG_SIZE
    datapoint = parse_image("/home/connor/Dev/Data/ADEChallengeData2016/images/training/ADE_train_00020210.jpg")
    X, y = load_image_train(datapoint, crop_mode=True, classification_heads=None)
    plt.imshow(datapoint['image'])
    plt.show()
    plt.imshow(X)
    plt.show()
    # original size < IMG_SIZE
    datapoint = parse_image("/home/connor/Dev/Data/ADEChallengeData2016/images/training/ADE_train_00000021.jpg")
    X, y = load_image_train(datapoint, crop_mode=True, classification_heads=None)
    plt.imshow(datapoint['image'])
    plt.show()
    plt.imshow(X)
    plt.show()

    