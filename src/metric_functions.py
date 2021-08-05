import tensorflow as tf


class WeightedError(tf.keras.metrics.Metric):
    def __init__(self, weights, name='WeightedError', **kwargs):
        super(WeightedError, self).__init__(name=name, **kwargs)
        self.space_weights = tf.constant(weights)
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
        return tf.math.divide(tf.math.multiply(self.values, self.weights), self.samples)


class DiscreteAccuracy(tf.keras.metrics.Metric):
    def __init__(self, dimension, name='DiscreteAccuracy', **kwargs):
        super(DiscreteAccuracy, self).__init__(name=name, **kwargs)
        self.dimension = dimension
        self.counts = self.add_weight(name='DA_counts', initializer='zeros')
        self.totals = self.add_weight(name='DA_total', initializer='zeros')
    
    def reset_states(self):
        self.reset_state()

    def reset_state(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.clip_by_value(tf.math.floor(tf.math.scalar_mul(2, y_pred)), 0, 1)
        val = tf.reduce_sum(tf.cast(tf.equal(y_pred, y_true), dtype=tf.float32))
        self.counts.assign_add(val)
        self.totals.assign_add(y_pred.shape[0] * y_pred.shape[1])


    def result(self):
        return tf.math.divide(self.counts, self.totals)


class VolumeAccuracy(tf.keras.metrics.Metric):
    def __init__(self, dimension, name='VolumeAccuracy', **kwargs):
        super(VolumeAccuracy, self).__init__(name=name, **kwargs)
        self.dimension = dimension
        self.error_volume_ratios = self.add_weight(name='VA_EVR', initializer='zeros')
        self.samples = self.add_weight(name='VA_VS', initializer='zeros')
    
    def reset_states(self):
        self.reset_state()

    def reset_state(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.clip_by_value(tf.math.floor(tf.math.scalar_mul(2, y_pred)), 0, 1)
        err = tf.reduce_sum(tf.cast(tf.not_equal(y_pred, y_true), dtype=tf.float32), axis = 1) # Volume of FP and FN error of each sample.
        vol = tf.reduce_sum(y_true, axis = 1) # Volume of each sample.
        vol_err = tf.math.divide(err, vol) # Ratio of Err to True Volume

        self.error_volume_ratios.assign_add(tf.reduce_sum(vol_err)) # Sum of Ratios
        self.samples.assign_add(y_pred.shape[0]) # Count of Samples

    def result(self):
        return 1 - tf.math.divide(self.error_volume_ratios, self.samples) # 1 - Avg err ratio per sample.


class FPVolumeAccuracy(tf.keras.metrics.Metric):
    def __init__(self, dimension, name='FPVolumeAccuracy', **kwargs):
        super(FPVolumeAccuracy, self).__init__(name=name, **kwargs)
        self.dimension = dimension
        self.error_volume_ratios = self.add_weight(name='PVA_EVR', initializer='zeros')
        self.samples = self.add_weight(name='PVA_VS', initializer='zeros')
    
    def reset_states(self):
        self.reset_state()

    def reset_state(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.clip_by_value(tf.math.floor(tf.math.scalar_mul(2, y_pred)), 0, 1)
        err = tf.reduce_sum(tf.cast(tf.logical_and(tf.not_equal(y_pred, y_true), tf.math.equal(y_true, 0)), dtype=tf.float32), axis = 1) # Volume of FP error of each sample.
        vol = tf.reduce_sum(y_true, axis = 1) # Volume of each sample.
        vol_err = tf.math.divide(err, vol) # Ratio of Err to True Volume

        self.error_volume_ratios.assign_add(tf.reduce_sum(vol_err)) # Sum of Ratios
        self.samples.assign_add(y_pred.shape[0]) # Count of Samples

    def result(self):
        return 1 - tf.math.divide(self.error_volume_ratios, self.samples) # 1 - Avg err ratio per sample.

    
class FNVolumeAccuracy(tf.keras.metrics.Metric):
    def __init__(self, dimension, name='FNVolumeAccuracy', **kwargs):
        super(FPVolumeAccuracy, self).__init__(name=name, **kwargs)
        self.dimension = dimension
        self.error_volume_ratios = self.add_weight(name='NVA_EVR', initializer='zeros')
        self.samples = self.add_weight(name='NVA_VS', initializer='zeros')
    
    def reset_states(self):
        self.reset_state()

    def reset_state(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.clip_by_value(tf.math.floor(tf.math.scalar_mul(2, y_pred)), 0, 1)
        err = tf.reduce_sum(tf.cast(tf.logical_and(tf.not_equal(y_pred, y_true), tf.math.equal(y_true, 1)), dtype=tf.float32), axis = 1) # Volume of FP error of each sample.
        vol = tf.reduce_sum(y_true, axis = 1) # Volume of each sample.
        vol_err = tf.math.divide(err, vol) # Ratio of Err to True Volume

        self.error_volume_ratios.assign_add(tf.reduce_sum(vol_err)) # Sum of Ratios
        self.samples.assign_add(y_pred.shape[0]) # Count of Samples

    def result(self):
        return 1 - tf.math.divide(self.error_volume_ratios, self.samples) # 1 - Avg err ratio per sample.

def make_tensors():
    import numpy as np
    v = np.zeros((3, 8000))
    v2 = np.zeros((3, 8000))
    for i in range(3):
        v[i,i::3] = 0.5
        v2[i,i*v2.shape[1] // 3:(i+1) * v2.shape[1]//3] = 0.5
    return [tf.identity(tf.constant(v)), tf.identity(tf.constant(v2))]

def make_metrics():
    import train_network
    import pathlib

    data_path = pathlib.Path('/nfs/data/TapiaLab/HyperParameterOptimization/Data/')
    label = 'Kuka_14_20x20x20'
    metrics = []
    metrics.append(WeightedError(train_network.get_weights(data_path, label)))
    metrics.append(DiscreteAccuracy(8000))
    metrics.append(VolumeAccuracy(8000))
    metrics.append(FPVolumeAccuracy(8000))
    metrics.append(FNVolumeAccuracy(8000))
    return metrics

    
