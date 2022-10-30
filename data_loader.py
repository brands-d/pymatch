from gzip import open as gopen
from pickle import load
from pathlib import Path
from numpy import array, zeros, reshape, short


def vectorize(data, out_size):
    out = zeros(len(data) * out_size, dtype=short)
    indices = array(range(len(data))) * out_size + data
    out[indices] = 1

    return reshape(out, (len(data), out_size)).T


def load_mnist_data(path, out_size):

    with gopen(Path(path), 'rb') as f:
        data = load(f, encoding='latin1')

    training_data = array(data[0][0]).T
    training_results = vectorize(data[0][1], out_size=out_size)
    validation_data = array(data[1][0]).T
    validation_results = vectorize(data[1][1], out_size=out_size)
    test_data = array(data[2][0]).T
    test_results = vectorize(data[2][1], out_size=out_size)

    return (training_data, training_results), (validation_data, validation_results), (test_data, test_results)
