from numpy.random import default_rng
from numpy import exp, array, newaxis, argmax


def sigmoid(z):
    return 1/(1 + exp(-z))


def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


def reshape_input(data, layer_size=None):
    data = array(data)
    if len(data.shape) == 1 and layer_size != 1:
        return data[:, newaxis]
    elif layer_size is not None:
        if data.shape[0] == layer_size:
            return data
        else:
            return data.T
    else:
        return data


def random_partion(data, size, y=None):

    i = default_rng().permutation(len(data[0, :]))
    data = data[:, i]
    partitions = [data[:, k:k+size] for k in range(0, len(data[0, :]), size)]
    if y is not None:
        y = y[:, i]
        partitions_y = [y[:, k:k+size] for k in range(0, len(y[0, :]), size)]

        return partitions, partitions_y
    return partitions


def decide(data):
    return argmax(data, axis=0)[:]
