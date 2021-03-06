import tensorflow as tf
from ray import tune


class MaxErrorDistance(tf.keras.metrics.Metric):
    def __init__(self, weights, name='WeightedError', **kwargs):
        super(MaxErrorDistance, self).__init__(name=name, **kwargs)
        self.space_weights = tf.constant(weights, dtype=tf.float32)
        self.values = self.add_weight(name='WE_values', shape=weights.shape, initializer='zeros')
        self.samples = self.add_weight(name='WE_samples', initializer='zeros')


class WeightedError(tf.keras.metrics.Metric):
    def __init__(self, weights, name='WeightedError', **kwargs):
        super(WeightedError, self).__init__(name=name, **kwargs)
        self.space_weights = tf.constant(weights, dtype=tf.float32)
        self.values = self.add_weight(name='WE_values', shape=weights.shape, initializer='zeros')
        self.samples = self.add_weight(name='WE_samples', initializer='zeros')

    def reset_states(self):
        self.reset_state()

    def reset_state(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))

    def update_state(self, y_true, y_pred, sample_weight=None):
        loss_values = tf.math.subtract(y_true, y_pred)
        loss_values = tf.math.abs(loss_values)  # Absolute
        # loss_values = tf.math.square(loss_values) # Square
        self.values.assign_add(tf.reduce_sum(loss_values, axis=0))
        self.samples.assign_add(y_pred.shape[0])

    def result(self):
        values = tf.math.multiply(self.values, self.space_weights)
        return tf.divide(values, self.samples)


@tf.function
def discrete_prediction(y_pred):
    return tf.clip_by_value(tf.math.floor(tf.math.scalar_mul(2, y_pred)), 0, 1)


class AverageVolume(tf.keras.metrics.Metric):
    def __init__(self, name='AverageVolume', **kwargs):
        super(AverageVolume, self).__init__(name=name, **kwargs)
        self.volumes = self.add_weight(name='volumes', initializer='zeros')  # Sum of pred volumes / true volumes
        self.samples = self.add_weight(name='samples', initializer='zeros')  # Sample counts

    def reset_states(self):
        self.reset_state()

    def reset_state(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.reduce_sum(tf.clip_by_value(tf.math.floor(tf.math.scalar_mul(2, y_pred)), 0, 1), axis=1)
        y_true = tf.reduce_sum(y_true, axis=1)  # True Volumes
        self.volumes.assign_add(tf.reduce_sum(tf.math.divide(y_pred, y_true)))  # Volume Ratios
        self.samples.assign_add(y_pred.shape[0])  # Sample count

    def result(self):
        return tf.math.divide(self.volumes, self.samples)


class VolumeAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='VolumeAccuracy', **kwargs):
        super(VolumeAccuracy, self).__init__(name=name, **kwargs)
        self.error_volumes = self.add_weight(name='error_volumes', initializer='zeros')
        self.samples = self.add_weight(name='samples', initializer='zeros')

    def reset_states(self):
        self.reset_state()

    def reset_state(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.clip_by_value(tf.math.floor(tf.math.scalar_mul(2, y_pred)), 0, 1)
        err = tf.reduce_sum(tf.cast(tf.not_equal(y_pred, y_true), dtype=tf.float32),
                            axis=1)  # Volume of FP and FN error of each sample.
        vol = tf.reduce_sum(y_true, axis=1)  # Volume of each sample.
        vol_err = tf.math.divide(err, vol)  # Ratio of Err to True Volume

        self.error_volumes.assign_add(tf.reduce_sum(vol_err))  # Sum of Ratios
        self.samples.assign_add(y_pred.shape[0])  # Count of Samples

    def result(self):
        return 1 - tf.math.divide(self.error_volumes, self.samples)  # 1 - Avg err ratio per sample.


class FPError(tf.keras.metrics.Metric):
    def __init__(self, name='FPError', **kwargs):
        super(FPError, self).__init__(name=name, **kwargs)
        self.error_volumes = self.add_weight(name='error_volumes', initializer='zeros')
        self.samples = self.add_weight(name='samples', initializer='zeros')

    def reset_states(self):
        self.reset_state()

    def reset_state(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.clip_by_value(tf.math.floor(tf.math.scalar_mul(2, y_pred)), 0, 1)
        err = tf.reduce_sum(
            tf.cast(tf.logical_and(tf.not_equal(y_pred, y_true), tf.math.equal(y_true, 0)), dtype=tf.float32),
            axis=1)  # Volume of FP error of each sample.
        vol = tf.reduce_sum(y_true, axis=1)  # Volume of each sample.
        vol_err = tf.math.divide(err, vol)  # Ratio of Err to True Volume

        self.error_volumes.assign_add(tf.reduce_sum(vol_err))  # Sum of Ratios
        self.samples.assign_add(y_pred.shape[0])  # Count of Samples

    def result(self):
        return tf.math.divide(self.error_volumes, self.samples)  # 1 - Avg err ratio per sample.


class FNError(tf.keras.metrics.Metric):
    def __init__(self, name='FNError', **kwargs):
        super(FNError, self).__init__(name=name, **kwargs)
        self.error_volumes = self.add_weight(name='error_volumes', initializer='zeros')
        self.samples = self.add_weight(name='samples', initializer='zeros')

    def reset_states(self):
        self.reset_state()

    def reset_state(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.clip_by_value(tf.math.floor(tf.math.scalar_mul(2, y_pred)), 0, 1)
        err = tf.reduce_sum(
            tf.cast(tf.logical_and(tf.not_equal(y_pred, y_true), tf.math.equal(y_true, 1)), dtype=tf.float32),
            axis=1)  # Volume of FN error of each sample.
        vol = tf.reduce_sum(y_true, axis=1)  # Volume of each sample.
        vol_err = tf.math.divide(err, vol)  # Ratio of Err to True Volume

        self.error_volumes.assign_add(tf.reduce_sum(vol_err))  # Sum of Ratios
        self.samples.assign_add(y_pred.shape[0])  # Count of Samples

    def result(self):
        return tf.math.divide(self.error_volumes, self.samples)  # 1 - Avg err ratio per sample.


class Precision(tf.keras.metrics.Metric):
    def __init__(self, name='Precision', **kwargs):
        super(Precision, self).__init__(name=name, **kwargs)
        self.precision_sum = self.add_weight(name='precision_sum', initializer='zeros')
        self.samples = self.add_weight(name='samples', initializer='zeros')

    def reset_states(self):
        self.reset_state()

    def reset_state(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.clip_by_value(tf.math.floor(tf.math.scalar_mul(2, y_pred)), 0, 1)
        tp = tf.reduce_sum(
            tf.cast(tf.logical_and(tf.equal(y_pred, y_true), tf.math.equal(y_true, 1)), dtype=tf.float32),
            axis=1)  # Volume of TP error of each sample.
        fp = tf.reduce_sum(
            tf.cast(tf.logical_and(tf.not_equal(y_pred, y_true), tf.math.equal(y_true, 0)), dtype=tf.float32),
            axis=1)  # Volume of FP error of each sample.
        precision = tf.math.divide(tp, tf.math.add(tp, fp))

        self.precision_sum.assign_add(tf.reduce_sum(precision))  # Precision ratios
        self.samples.assign_add(y_pred.shape[0])  # Count of Samples

    def result(self):
        return tf.math.divide(self.precision_sum, self.samples)  # Precision Average


class Recall(tf.keras.metrics.Metric):
    def __init__(self, name='Recall', **kwargs):
        super(Recall, self).__init__(name=name, **kwargs)
        self.recall_sum = self.add_weight(name='recall_sum', initializer='zeros')
        self.samples = self.add_weight(name='samples', initializer='zeros')

    def reset_states(self):
        self.reset_state()

    def reset_state(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.clip_by_value(tf.math.floor(tf.math.scalar_mul(2, y_pred)), 0, 1)
        tp = tf.reduce_sum(
            tf.cast(tf.logical_and(tf.equal(y_pred, y_true), tf.math.equal(y_true, 1)), dtype=tf.float32),
            axis=1)  # Volume of TP error of each sample.
        fn = tf.reduce_sum(
            tf.cast(tf.logical_and(tf.not_equal(y_pred, y_true), tf.math.equal(y_true, 1)), dtype=tf.float32),
            axis=1)  # Volume of FN error of each sample.
        recall = tf.math.divide(tp, tf.math.add(tp, fn))

        self.recall_sum.assign_add(tf.reduce_sum(recall))  # recall ratios
        self.samples.assign_add(y_pred.shape[0])  # Count of Samples

    def result(self):
        return tf.math.divide(self.precision_sum, self.samples)  # Recall Average


class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='F1Score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.f1_sum = self.add_weight(name='f1_sum', initializer='zeros')
        self.samples = self.add_weight(name='samples', initializer='zeros')

    def reset_states(self):
        self.reset_state()

    def reset_state(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.clip_by_value(tf.math.floor(tf.math.scalar_mul(2, y_pred)), 0, 1)
        tp = tf.reduce_sum(
            tf.cast(tf.logical_and(tf.equal(y_pred, y_true), tf.math.equal(y_true, 1)), dtype=tf.float32),
            axis=1)  # Volume of TP of each sample.
        fp = tf.reduce_sum(
            tf.cast(tf.logical_and(tf.not_equal(y_pred, y_true), tf.math.equal(y_true, 0)), dtype=tf.float32),
            axis=1)  # Volume of FP error of each sample.
        fn = tf.reduce_sum(
            tf.cast(tf.logical_and(tf.not_equal(y_pred, y_true), tf.math.equal(y_true, 1)), dtype=tf.float32),
            axis=1)  # Volume of FN error of each sample.
        # precision = tf.math.divide(tp, tf.math.add(tp, fp))
        # recall = tf.math.divide(tp, tf.math.add(tp, fn))

        self.f1_sum.assign_add(tf.reduce_sum(
            tf.math.divide(tp, tf.math.add(tp, tf.math.scalar_mul(0.5, tf.math.add(fp, fn))))))  # F1Scores
        self.samples.assign_add(y_pred.shape[0])  # Count of Samples

    def result(self):
        return tf.math.divide(self.f1_sum, self.samples)  # Recall Average


def make_tensors():
    import numpy as np
    v = np.zeros((3, 8000))
    v2 = np.zeros((3, 8000))
    for i in range(3):
        v[i, i::3] = 0.5
        v2[i, i * v2.shape[1] // 3:(i + 1) * v2.shape[1] // 3] = 0.5
    return [tf.identity(tf.constant(v)), tf.identity(tf.constant(v2))]


def make_metrics():
    import train_network
    import pathlib

    data_path = pathlib.Path('/nfs/data/TapiaLab/HyperParameterOptimization/Data/')
    label = 'Kuka_14_20x20x20'
    metrics = []
    metrics.append(WeightedError(train_network.get_weights(data_path, label)))
    metrics.append(AverageVolume())
    metrics.append(VolumeAccuracy())
    metrics.append(FPError())
    metrics.append(FNError())
    return metrics
