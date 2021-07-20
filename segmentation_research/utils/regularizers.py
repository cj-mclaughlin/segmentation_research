from tensorflow.keras.regularizers import l2
import tensorflow as tf

def weight_standardization(kernel):
    """
    Weight standardization regularizer
    https://arxiv.org/abs/1903.10520
    """
    kernel_mean = tf.math.reduce_mean(kernel, axis=[0, 1, 2], keepdims=True, name='kernel_mean')
    kernel = kernel - kernel_mean
    kernel_std = tf.math.reduce_std(kernel, axis=[0, 1, 2], keepdims=True, name='kernel_std')
    kernel = kernel / (kernel_std + 1e-5)

WEIGHT_DECAY = l2(1e-4)
WS_STD = weight_standardization