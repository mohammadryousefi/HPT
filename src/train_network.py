'''
Revise this model to compute train the network using the standard MSE loss and monitor the custom loss function values. We want to see if there's a correlation between the values of MSE and custom loss function.
'''
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import sys
import argparse
import pathlib
import metric_functions

'''
Weights is a single matrix defining the average loss scale per position in the entire space.
NOTE: This is done without per sample scale factor.
'''


def weighted_loss_averaged(weights):
    def loss(y_true, y_pred):
        loss_values = tf.math.subtract(y_true, y_pred)
        # loss_values = tf.math.abs(loss_values) # Absolute
        loss_values = tf.math.square(loss_values)  # Square
        loss_values = tf.math.multiply(loss_values, weights)
        loss_values = tf.math.reduce_sum(loss_values, axis=-1)
        return loss_values

    return loss


def get_model(error_weights):
    features = tf.keras.Input(shape=(14,), name='features')
    pred = tf.keras.layers.Dense(units=500, activation='relu')(features)
    pred = tf.keras.layers.Dense(units=1000, activation='relu')(pred)
    pred = tf.keras.layers.Dense(units=2000, activation='relu')(pred)
    pred = tf.keras.layers.Dense(units=4000, activation='relu')(pred)
    pred = tf.keras.layers.Dense(units=8000)(pred)

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

    # loss = weighted_loss_averaged(error_weights) # Just going to record as a metric
    metrics = ['accuracy', metric_functions.WeightedError(error_weights)]

    model = tf.keras.Model(inputs=features, outputs=pred)
    model.compile(optimizer=opt, loss='mse', metrics=metrics)
    return model


def parse_args(args):
    parser = argparse.ArgumentParser(description='Train a SVG network with weighted loss function.')
    parser.add_argument('save_path', metavar='Path', help='Directory for checkpoints.')
    parser.add_argument('path', metavar='Path', help='Directory for all data files and weights.')
    parser.add_argument('experiment', metavar='Experiment',
                        help='The label of the experiment. It is used to determine data file names and checkpoints.')
    ns = parser.parse_args(args)
    ns.save_path = pathlib.Path(ns.save_path)
    ns.path = pathlib.Path(ns.path)
    return ns


def get_data(path, label):
    data_file = path.joinpath(pathlib.Path(label + '.npz'))
    data = sp.load_npz(data_file)
    split = data.shape[0] // 10
    return data[:-split, :14].toarray(), data[:-split, 14:].toarray(), data[-split:, :14].toarray(), data[-split:,
                                                                                                     14:].toarray()


def get_weights(path, label):
    data_file = path.joinpath(pathlib.Path(label + '_edt.npz'))
    data = np.load(data_file)['arr_0']
    split = data.shape[0] // 10
    return np.mean(data[:-split], axis=0)


def get_experiment_dir(path, label):
    exp_dir = path / label
    if not exp_dir.exists():
        return exp_dir
    counter = 0
    pattern = label + ' ({})'
    while True:
        counter += 1
        exp_dir = path / pattern.format(counter)
        if not exp_dir.exists():
            return exp_dir


def main(args):
    params = parse_args(args)
    x_train, y_train, x_eval, y_eval = get_data(params.path, params.experiment)
    weights = get_weights(params.path, params.experiment)
    exp_dir = get_experiment_dir(params.save_path, params.experiment)
    exp_dir.mkdir(parents=True)
    model = get_model(weights)
    callbacks = []
    callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30))
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=exp_dir / 'tensorboard'))
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(exp_dir / 'checkpoint', save_weights_only=True, monitor='val_loss',
                                           save_best_only=True))
    print(
        f'Creating model using "{params.experiment + ".npz"}" and "{params.experiment + "_edt.npz"}" located in'
        f' "{params.path}". The checkpoints will be saved in "{exp_dir}".')
    print(model.summary())
    model.fit(x_train, y_train, batch_size=100, epochs=1000, callbacks=callbacks, validation_data=(x_eval, y_eval))


if __name__ == '__main__':
    main(sys.argv[1:])
