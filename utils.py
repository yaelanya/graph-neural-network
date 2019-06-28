import numpy as np


class BatchGenerator(object):
    def __init__(self, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
            
    def generator(self, inputs, targets):
        data_size = len(inputs)
        num_batches = np.ceil(data_size / self.batch_size).astype(np.int)

        if self.shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            _inputs = inputs[shuffle_indices]
            _targets = targets[shuffle_indices]
        else:
            _inputs = inputs[:]
            _targets = targets[:]

        for batch_num in range(num_batches):
            start_index = batch_num * self.batch_size
            end_index = min((batch_num + 1) * self.batch_size, data_size)

            batch_inputs = _inputs[start_index:end_index]
            batch_targets = _targets[start_index:end_index]

            yield batch_inputs, batch_targets


def split_train_val(inputs, targets, val_rate, shuffle=True):
    """
    Return:
        train_inputs, train_targets, val_inputs, val_targets
    """
    data_size = len(inputs)
    split_index = np.ceil(data_size * val_rate).astype(np.int)
    
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        _inputs = inputs[shuffle_indices]
        _targets = targets[shuffle_indices]
    else:
        _inputs = inputs[:]
        _targets = targets[:]
        
    return _inputs[split_index:], _targets[split_index:], _inputs[:split_index], _targets[:split_index]