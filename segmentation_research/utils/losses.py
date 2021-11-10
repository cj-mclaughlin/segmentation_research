from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np

def WeightedCCE(class_weights, alpha=1.0):
    """
    Cross entropy loss weighted by class weights and scaling term alpha
    modified from https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d
    """
    class_weights = K.variable(class_weights)
    a = K.variable(alpha)
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * class_weights
        loss = -K.sum(loss, -1)
        return a*loss

    return loss

def get_label_weight_mask(labels, ignore_label, num_classes, label_weights=1.0, dtype=tf.float32):
    """
    create mask to ignore background class of mask
    taken from https://github.com/tensorflow/models/blob/master/research/deeplab/core/utils.py
    """
    if not isinstance(label_weights, (float, list)):
        raise ValueError(
            'The type of label_weights is invalid, it must be a float or a list.')

    if isinstance(label_weights, list) and len(label_weights) != num_classes:
        raise ValueError(
            'Length of label_weights must be equal to num_classes if it is a list, '
            'label_weights: %s, num_classes: %d.' % (label_weights, num_classes))

    not_ignore_mask = tf.not_equal(labels, ignore_label)
    not_ignore_mask = tf.cast(not_ignore_mask, dtype)
    if isinstance(label_weights, float):
        return not_ignore_mask * label_weights

    label_weights = tf.constant(label_weights, dtype)
    weight_mask = tf.einsum('...y,y->...',
                            tf.one_hot(labels, num_classes, dtype=dtype),
                            label_weights)
    return tf.multiply(not_ignore_mask, weight_mask)

def NonBackgroundSparseCCE(y_true, y_pred, dtype=tf.float32, ignore_label=0, num_classes=151):
    """
    Sparse Categorical Cross Entropy which ignores background class
    """
    weights = get_label_weight_mask(y_true, ignore_label=ignore_label, num_classes=num_classes, label_weights=1.0, dtype=dtype)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    loss = loss_fn(y_true, y_pred)
    weighted_loss = tf.multiply(tf.squeeze(weights), loss)
    weighted_loss = tf.reduce_mean(weighted_loss)
    return weighted_loss
