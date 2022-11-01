import time
from math import sqrt
from numpy.random import default_rng
from numpy import array, newaxis, argmax, reshape


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
    return argmax(data, axis=0)


def plot_digit_data(data, i=None):
    from matplotlib.pyplot import imshow, show

    size = int(sqrt(len(data[0])))
    if i is not None:
        imshow(reshape(data[0][:, i], (size, size)))
        print(decide(data[1][:, i]))

    else:
        imshow(reshape(data[0], (size, size)))
        print(decide(data[1]))
    show()


def timeit(f):
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = f(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'{f.__name__} took {total_time:.1f} s')

        return result

    return timeit_wrapper
