import tensorflow as tf
import numpy as np

class NonBackgroundAccuracy(tf.keras.metrics.Metric):
    def __init__(self, ignore_index=0, name='acc', **kwargs):
        super(NonBackgroundAccuracy, self).__init__(name=name, **kwargs)
        self.acc_fn = tf.keras.metrics.SparseCategoricalAccuracy()
        self.acc = 0
        self.ignore_index = ignore_index

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float64)
        y_pred = tf.cast(y_pred, tf.float64)
        y_pred_mask = tf.where(y_true == self.ignore_index, y_true, y_pred)
        accuracy = self.acc_fn(y_true, y_pred_mask)
        self.acc = accuracy

    def result(self):
        return self.acc

    def reset_state(self):
        self.acc_fn.reset_state()
        self.acc = 0

class NonBackgroundMIoU(tf.keras.metrics.Metric):
    def __init__(self, ignore_index=0, name='miou', num_classes=150, **kwargs):
        super(NonBackgroundMIoU, self).__init__(name=name, **kwargs)
        self.miou = 0
        self.ignore_index = ignore_index
        self.num_classes = num_classes

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.math.argmax(y_pred, axis=-1).numpy()
        y_true = tf.cast(tf.squeeze(y_true, axis=-1), tf.int64).numpy()
        y_pred[np.where(y_true == self.ignore_index)] = self.ignore_index  # ignore background
        intersection = y_pred[np.where(y_pred == y_true)]
        area_intersection, _ = np.histogram(intersection, bins=np.arange(self.num_classes+1))
        area_output, _ = np.histogram(y_pred, bins=np.arange(self.num_classes+1))
        area_target, _ = np.histogram(y_true, bins=np.arange(self.num_classes+1))
        area_union = area_output + area_target - area_intersection
        self.miou = np.nanmean(area_intersection / (area_union))

    def result(self):
        return self.miou

    def reset_state(self):
        self.miou = 0

class IntersectionUnionTarget(tf.keras.metrics.Metric):
    """
    https://github.com/hszhao/semseg/blob/master/util/util.py
    """
    def __init__(self, ignore_index=0, name='miou', num_classes=150, **kwargs):
        super(IntersectionUnionTarget, self).__init__(name=name, **kwargs)
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.intersection = 0
        self.union = 0
        self.target = 0

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.math.argmax(y_pred, axis=-1).numpy()
        y_true = tf.cast(tf.squeeze(y_true, axis=-1), tf.int64).numpy()
        y_pred[np.where(y_true == self.ignore_index)] = self.ignore_index  # ignore background
        intersection = y_pred[np.where(y_pred == y_true)]
        area_intersection, _ = np.histogram(intersection, bins=np.arange(self.num_classes+1))
        area_output, _ = np.histogram(y_pred, bins=np.arange(self.num_classes+1))
        area_target, _ = np.histogram(y_true, bins=np.arange(self.num_classes+1))
        area_union = area_output + area_target - area_intersection
        self.intersection = area_intersection
        self.union = area_union
        self.target = area_target

    def result(self):
        return self.intersection, self.union, self.target

    def reset_state(self):
        self.intersection = 0
        self.union = 0
        self.target = 0