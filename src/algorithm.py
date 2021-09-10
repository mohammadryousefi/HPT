LOCAL_PARAMS = None # dict
RL_PARAMS = None # dict

BUFFER_SIZE = 50000
BATCH_SIZE = 100
HYPER_PARAM_SPACE = None # Dictionary to be used in building networks.

EVAL_METRICS = None # List[Metric]

TRAIN_DATA = None # dict(np.array)
EVAL_DATA = None # dict(np.array)
TEST_DATA = None # dict(np.array)

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

def build_rn_network(space):
    # Generate a network which returns hyper parameters 1 at a time.



def build_network(actions, values_dict):
    # Generate a network for local search using parameters from a provided 
    # hyper-parameter vector. Legality of each action is determined by 
    # translation of the next vector value into an index which is applied
    # to the next values dict.
    # General loop
    # Start Model
    # For each action_value in actions -> use values_dict[PARAMETER_TYPE][action_value]
    pass

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
        self.buffer = numpy.empty(shape=(self.max_size, ), dtype=object)
        self.is_locked = False

    def next_batch():
        batch = self.buffer[self.buffer_loc: self.buffer_loc + self.batch_size]
        self.buffer_loc += self.batch_size
        return batch

    def record(values):
        if self.is_locked:
            raise BufferError('Operation invoked while in Read-Only Mode.')
        if self.record_index == self.max_size:
            raise BufferError('Operation exceeds max buffer capacity.')
        self.buffer[self.record_index] = values
        self.record_index += 1

    def reset(unlock=False):
        if unlock:
            self.unlock()
        self.buffer_loc = 0
        if not self.is_locked:
            self.record_index = 0

    def lock():
        self.is_locked = True

    def unlock():
        self.is_locked = False

    def is_ready():
        return self.buffer_loc < self.record_index

    def get_last():
        if self.record_index:
            return self.buffer[self.record_index - 1]
        else:
            raise BufferError('Buffer is empty.')




def reinforcement_learning_outer_loop(space, reward_function, training_length):
    max_reward = -float('inf')
    max_model = None

    rn_network = create_rn_network(len(input_space))
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


