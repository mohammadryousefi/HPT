import tensorflow as tf


class WeightedError(tf.keras.metrics.Metric):
    def __init__(self, weights, **kwargs):
        super(WeightedError, self).__init__(**kwargs)
        self.weights = weights
        self.values = self.add_weight(name='weighted_error', initializer='zeros')

    def reset_state(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))

    def update_state(self, y_true, y_pred, sample_weights=None):
        loss_values = tf.math.subtract(y_true, y_pred)
        loss_values = tf.math.abs(loss_values)  # Absolute
        # loss_values = tf.math.square(loss_values) # Square
        loss_values = tf.math.multiply(loss_values, self.weights)
        self.values.assign_add(tf.reduce_sum(loss_values))

    def result(self):
        return self.values
