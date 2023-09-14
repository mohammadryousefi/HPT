import tensorflow as tf

LOCAL_PARAMS = None  # dict
RL_PARAMS = None  # dict

BUFFER_SIZE = 50000
BATCH_SIZE = 100
HYPER_PARAM_SPACE = None  # Dictionary to be used in building networks.

EVAL_METRICS = None  # List[Metric]

TRAIN_DATA = None  # dict(np.array)
EVAL_DATA = None  # dict(np.array)
TEST_DATA = None  # dict(np.array)


def normalize(values):
    pass


def evaluate(model, input_features, ground_truth, method, *metrics):
    predictions = model.predict(input_features)
    metric_values = []
    for metric in metrics:
        metric_values.append(metric.evaluate(predictions, ground_truth))

    if method == 'mean_value':
        return sum(metric_values) / len(metric_values)
    elif method == 'mean_norm':
        norm_values = sum(map(lambda x: normalize(x), metric_values)) / len(metric_values)


def build_rnn_network(dim):
    '''
    Create a network with a single LSTM layer. The inputs are one-hot encoded vectors in with size equal to the number of options.
    The output is treated as a probability distribution for selection of the next hyper-parameter.
    '''
    model = tf.keras.Sequential()
    model.add(tf.keras.Input((dim,)))
    model.add(tf.keras.layers.LSTM(128, activation='tanh'))
    model.add(tf.keras.layers.Dense(dim, activation='softmax'))

    model.compile(optimizer='adam', loss='mse')
    return model


def build_network(rnn_actions, space, model_input, model_output):
    '''
    Using one-hot row vectors in rnn_actions generate the target network.
    '''
    assert space.ndim == 1 and space.shape[0] == rnn_actions.shape[
        -1] or rnn_actions.shape == space.shape, "Action space does not match hyper-parameter space."
    model = tf.keras.Sequential()
    model.add(tf.keras.Input((model_input,)))
    if space.ndim == 1:
        for i in range(rnn_actions.shape[0]):
            model.add(tf.keras.layers.Dense(space[np.argmax(rnn_actions[i])], activation='relu'))
    else:
        for i in range(rnn_actions.shape[0]):
            model.add(tf.keras.layers.Dense(space[i, np.argmax(rnn_actions[i])], activation='relu'))
    model.add(tf.keras.layers.Dense(model_output, activation=None))
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=opt, loss='mse')
    return model


def generate_sequence(rnn_network, to_generate, dim):
    '''
    Generate a sequence of actions which takes as input a one-hot encoded vector of the last selection.
    The input and output dimension is the same.
    The initial action for the entire sequence is a zero vector.
    '''
    actions = numpmy.zeros((to_generate + 1, dim))
    for i in range(to_generate):
        selection = np.random.choice(dim, p=rnn_network.predict(actions[i]))
        actions[i + 1, selection] = 1

    return actions[1:], np.argmax(actions[1:], axis=-1)


class History:
    def __init__(self, capacity, sequence_length, choices, batch_size=1):
        self.max_size = capacity
        self.output_buffer = np.zeros((capacity, sequence_length, choices))
        self.selection_buffer = np.zeros((capacity, sequence_length))
        self.batch_size = batch_size
        self.loc = 0
        self.used = 0
        self.is_locked = False

    def next_batch(self):
        if self.loc >= self.used:
            return None, None
        end = min(self.loc + self.batch_size, self.used)
        outputs = self.output_buffer[self.loc:end]
        selections = self.selection_buffer[self.loc:end]
        self.loc = end
        return outputs, selections

    def record(self, outputs, selections):
        if self.is_locked:
            raise BufferError('Operation invoked while in Read-Only Mode.')
        if self.used == self.max_size:
            raise BufferError('Operation exceeds max buffer capacity.')
        self.output_buffer[self.used] = outputs
        self.selection_buffer[self.used] = selections
        self.used += 1

    def reset(self, unlock=False):
        if unlock:
            self.unlock()
        self.loc = 0
        if not self.is_locked:
            self.used = 0

    def lock(self):
        self.is_locked = True

    def unlock(self):
        self.is_locked = False

    def is_ready(self):
        return self.loc < self.used

    def get_last(self):
        if self.is_ready():
            return self.output_buffer[self.used - 1], self.selection_buffer[self.used - 1]
        else:
            return None, None


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


def reinforcement_learning_outer_loop(space, model_input, model_output, reward_function, iterations, num_params=None):
    max_reward = -float('inf')
    max_model = None
    assert space.ndim == 2 or num_params is not None, "Single dimensional space requires num_params to be defined."
    assert 0 < space.ndim < 3, "Space must be a 1 or 2 dimensional array"
    rnn_network = build_rnn_network(space.shape[-1])
    if space.ndim == 2:
        num_params = space.shape[0]

    # How do I handle the actions? Every so many actions has only 1 reward.
    # This buffer design does not work. I need to return the action sequence.
    # actions = Buffer(BUFFER_SIZE, BATCH_SIZE)
    rewards = Buffer(BUFFER_SIZE, BATCH_SIZE)
    history = History(BUFFER_SIZE, num_params, space.shape[-1], BATCH_SIZE)
    for i in range(iterations):
        # actions.record(generate_sequence(rnn_network, num_params, space.shape[-1]))
        history.record(generate_sequence(rnn_network, num_params, space.shape[-1]))
        network = build_network(history.get_last(), HYPER_PARAM_SPACE)
        # Train model
        rewards.record(evaluate(network, TEST_DATA['features'], TEST_DATA['labels'], 'mean_value', EVAL_METRICS))
        update(rnn_network, history.get_last(), rewards.get_last())
        if rewards.get_last() > max_reward:
            max_reward = rewards.get_last()
            max_model = network
    return max_model


def select_hp_vector(policy, space):
    pass


def update(rn, action_batch, reward_batch):
    # Compute Nabla R with "Neural Architecture Search with Reinforcement Learning, arXiv:1611.01578v2"
    pass
