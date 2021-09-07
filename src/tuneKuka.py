# def setup_environment(data_path, input_dim, output_dim, space, train_samples, alpha=0.5, epochs=500, batch_size=32,
#                       patience=2, max_error_factor=10, error_cutoff=2):
#     environment = dict()
#
#     # Data Parameters
#     environment['DataPath'] = data_path
#     environment['InputDim'] = input_dim
#     environment['OutputDim'] = output_dim
#     environment['OutputShape'] = space
#     environment['TrainSamples'] = train_samples
#
#     # Training Parameters
#     environment['Epochs'] = epochs
#     environment['BatchSize'] = batch_size
#     environment['Patience'] = patience
#
#     # Computation Parameter
#     environment['Alpha'] = alpha
#     environment['MaxError'] = error_cutoff
#     environment['MaxErrorFactor'] = max_error_factor
#
#     return environment


def train(config):
    import tensorflow.keras as k
    import metric_functions

    # Simple Sequential Model with 4 Adjustable Hidden Layers
    # All layers RELU.
    if config['Checkpoint']:
        model = k.models.load_model(config['Checkpoint'])
    else:
        model = k.Sequential()

        model.add(k.Input((config['InputDim'],), batch_size=config['BatchSize'], name='JointConfigs'))
        model.add(k.layers.Dense(config['layer1'], activation='relu', name='Hidden1'))
        model.add(k.layers.Dense(config['layer2'], activation='relu', name='Hidden2'))
        model.add(k.layers.Dense(config['layer3'], activation='relu', name='Hidden3'))
        model.add(k.layers.Dense(config['layer4'], activation='relu', name='Hidden4'))
        model.add(k.layers.Dense(config['OutputDim'], activation='relu', name='Voxels'))
        metrics = [metric_functions.F1Score(), metric_functions.AverageVolume()]
        model.compile(optimizer=k.optimizers.Adam(1e-4), loss='mse', metrics=metrics)

    callbacks = [k.callbacks.EarlyStopping(patience=config['Patience'])]

    # Load Data
    # Expects CSR matrix of shape N, M where N = Num Samples and M = 2 * DoF + DimX * DimY * DimZ
    from scipy.sparse import load_npz
    mtx = load_npz(config['DataPath'])
    x_train = mtx[:config['TrainSamples'], :config['InputDim']]
    y_train = mtx[:config['TrainSamples'], config['InputDim']:]
    x_val = mtx[config['TrainSamples']:, :config['InputDim']]
    y_val = mtx[config['TrainSamples']:, config['InputDim']:]
    from distance import compute_edt
    edt = compute_edt(y_val, config['OutputShape'])

    # Work with numpy.array objects
    x_train = x_train.toarray()
    y_train = y_train.toarray()
    x_val = x_val.toarray()
    y_val = y_val.toarray()

    del mtx
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=config['Epochs'],
              batch_size=config['BatchSize'], callbacks=callbacks, verbose=0)

    import numpy as np
    values = model.evaluate(x_val, y_val, batch_size=config['BatchSize'], verbose=0)
    y_pred = np.clip(np.floor(2 * model.predict(x_val, batch_size=config['BatchSize'], verbose=0).tonumpy()), 0, 1)
    max_error_distance = np.max(edt[np.where(y_pred != y_val)])
    if max_error_distance > config['MaxError']:
        fitness = 0
    else:

        values = model.evaluate(x_val, y_val, batch_size=config['BatchSize'])

        _, f1_score, volume = values
        fitness = np.exp(- max_error_distance * config['Beta']) * (
                config['Alpha'] * f1_score + (1 - config['Alpha']) * np.exp(-np.abs(np.log(volume))))
    return {'Loss': values[0], 'Fitness': fitness}


def get_params(args):
    import argparse
    _parser = argparse.ArgumentParser()
    _parser.add_argument('--local_dir', required=True, help='Path to the root directory for experiment results.')
    _parser.add_argument('--exp_name', required=True, help='Unique experiment identifier.')
    return _parser.parse_args(args)


# Requirements Ray (Tune, Defaults, Tensorboard), Tensorflow
def main():
    import sys
    import ray
    from ray import tune
    from ray.tune.suggest import ConcurrencyLimiter
    from ray.tune.suggest.hyperopt import HyperOptSearch

    params = get_params(sys.argv[1:])

    # ray.init(address='64.106.20.170', num_cpus=1, num_gpus=1)  # Initialize Local
    ray.init(address='64.106.20.170:6379')  # Initialize on cluster (Started on CLI)

    config = {
        'Checkpoint': None,
        'InputDim': 14,
        'OutputDim': 8000,
        'OutputShape': (20, 20, 20),
        'DataPath': '/nfs/data/TapiaLab/HyperParameterOptimization/Data/Kuka_14_20x20x20.npz',  # Make sure it exists.
        'TrainSamples': 90000,
        'Epochs': 500,
        'BatchSize': 32,
        'Patience': 2,
        'Alpha': 0.5,
        'Beta': 0.1,
        'MaxError': 2,
        'layer1': tune.choice([64, 128, 256, 512, 1024, 2048]),
        'layer2': tune.choice([64, 128, 256, 512, 1024, 2048]),
        'layer3': tune.choice([64, 128, 256, 512, 1024, 2048]),
        'layer4': tune.choice([64, 128, 256, 512, 1024, 2048])
    }

    search_alg = ConcurrencyLimiter(HyperOptSearch(), max_concurrent=4)

    # TODO: Add unique identifier to experiments from CLI arguments.
    analysis = tune.run(
        train,
        name=params.exp_name,  # f"{UID}"
        config=config,
        search_alg=search_alg,
        num_samples=30,
        metric="Fitness",
        mode="max",
        local_dir=params.local_dir,
        log_to_file=True,
        resources_per_trial={'cpu': 1, 'gpu': 1})

    print("Best HP found were: ", analysis.best_config)


if __name__ == '__main__':
    main()
