import tensorflow as tf
import numpy as np
import time
import multiprocessing


class SimpleHyperParametersSpace:
    def __init__(self, N, K, dtype=None, seed=None):
        self.dims = N
        self.choices = K
        self.values = np.empty(shape=(self.dims, self.choices), dtype=dtype)
        if seed is None:
            self.seed = time.time_ns()
            self.rng = np.random.default_rng(self.seed)
        else:
            self.rng = np.random.default_rng(seed)

    def choose(self, dim, probabilities=None):
        return self.rng.choice(self.values[dim], p=probabilities, shuffle=False)

    def add_choices(self, dim, choices):
        assert len(choices) == self.choices, 'Mismatch choice count'
        self.values[dim] = choices

    def add_values(self, values: np.ndarray):
        assert values.dtype == self.values.dtype, 'Type mismatch'
        assert values.shape == self.values.shape, 'Values must have the same dimensions'
        self.values[:] = values


LOCAL_PARAMS = None  # dict
RL_PARAMS = None  # dict

BUFFER_SIZE = 50000
BATCH_SIZE = 100
HYPER_PARAM_SPACE = None  # Dictionary to be used in building networks.

EVAL_METRICS = None  # List[Metric]

TRAIN_DATA = None  # dict(np.array)
EVAL_DATA = None  # dict(np.array)
TEST_DATA = None  # dict(np.array)


def build_rnn_network(space: SimpleHyperParametersSpace):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input((space.choices,), name='OneHotInput'))
    model.add(tf.keras.layers.LSTM(128), name='LSTM')
    model.add(tf.keras.layers.Dense(space.choices, activation='softmax', name='Probabilities'))
    return model


def generate_new_vector(rn_network):
    # actions = [next_prediction(rn_network)]
    # for i in range(int(actions[0])):
    #     actions.append(next_prediction(rn_network))
    # return actions
    pass


class Buffer:
    def __init__(self, size, batch_size):
        self.batch_size = batch_size
        self.max_size = size
        self.buffer_loc = 0
        self.record_index = 0
        self.buffer = np.empty(shape=(self.max_size,), dtype=object)
        self.is_locked = False

    def next_batch(self):
        batch = self.buffer[self.buffer_loc: self.buffer_loc + self.batch_size]
        self.buffer_loc += self.batch_size
        return batch

    def record(self, values):
        if self.is_locked:
            raise BufferError('Operation invoked while in Read-Only Mode.')
        if self.record_index == self.max_size:
            raise BufferError('Operation exceeds max buffer capacity.')
        self.buffer[self.record_index] = values
        self.record_index += 1

    def reset(self, unlock=False):
        if unlock:
            self.unlock()
        self.buffer_loc = 0
        if not self.is_locked:
            self.record_index = 0

    def lock(self):
        self.is_locked = True

    def unlock(self):
        self.is_locked = False

    def is_ready(self):
        return self.buffer_loc < self.record_index

    def get_last(self):
        if self.record_index:
            return self.buffer[self.record_index - 1]
        else:
            raise BufferError('Buffer is empty.')


def reinforcement_learning_outer_loop(space, reward_function, training_length):
    max_reward = -float('inf')
    max_model = None

    rn_network = build_rnn_network(len(input_space))
    actions = Buffer(BUFFER_SIZE, BATCH_SIZE)
    rewards = Buffer(BUFFER_SIZE, BATCH_SIZE)
    for i in range(training_length):
        actions.record(generate_new_vector(rn_network))
        network = build_network(actions.get_last(), HYPER_PARAM_SPACE)
        # Train model
        rewards.record(evaluate(network, TEST_DATA['features'], TEST_DATA['labels'], 'mean_value', EVAL_METRICS))
        update(rn_network, actions.get_last(), rewards.get_last())
        if rewards.get_last() > max_reward:
            max_reward = rewards.get_last()
            max_model = network
    return max_model


def select_hp_vector(policy, space):
    pass


def update(rn, action_batch, reward_batch):
    # Compute Nabla R with "Neural Architecture Search with Reinforcement Learning, arXiv:1611.01578v2"
    pass
