from numpy import mean
from pickle import dump, load
from gzip import open as gopen

from layer import Layer
from lib import reshape_input, random_partion, decide, timeit


class NeuralNetwork:

    def __init__(self, layers, activation='sigmoid', cost='mse', prob=False):

        if isinstance(layers[0], Layer):
            # layers is already list of 'Layer'
            self.layers = layers

        else:
            # init doubly linked list of 'Layer' with sizes from 'layers'
            self.layers = [Layer(layers[0],
                                 activation=activation, cost=cost), ]
            prev = self.layers[0]

            for i, size in enumerate(layers[1:]):
                # last layer
                if prob and i == len(layers) - 2:
                    self.layers += [Layer(size, prev=prev,
                                          activation='softmax', cost=cost), ]
                else:
                    self.layers += [Layer(size, prev=prev,
                                          activation=activation, cost=cost), ]
                self.layers[-2].next = self.layers[-1]
                prev = self.layers[-1]

        self.shape = tuple(layer.size for layer in self.layers)
        self.size = sum(self.shape)

    def classify(self, data, decisive=False):
        data = reshape_input(data, self.layers[0].size)
        for layer in self.layers[:-1]:
            data = layer.forward(data, decisive=False)

        return self.layers[-1].forward(data, decisive=decisive)

    def _backward(self, y, data=None):
        y = reshape_input(y, self.layers[-1].size)
        if data is not None:
            # without data backpropagate last
            # classification (results cached in layers)
            self.classify(data)

        return self.layers[0].backward(y)

    @timeit
    def train(self, data, y, epochs=1, batch_size=1, eta=1,
              test_data=None, decisive=False, regularize=None):
        for i in range(epochs):
            partitions, partitions_y = random_partion(data, batch_size, y=y)

            for partition, partition_y in zip(partitions, partitions_y):
                deltas = self._backward(partition_y, data=partition)
                self._update_network(deltas, eta=eta, reg=regularize)

            if test_data is not None:
                is_ = self.classify(test_data[0], decisive=decisive)
                should = reshape_input(test_data[1])

                if decisive:
                    outcome = is_ == decide(should)
                    print(f'{i + 1}/{epochs} : {sum(outcome)}/{len(outcome)}')
                else:
                    outcome = self.layers[-1].cost_func.f(is_, should)
                    print(f'{i + 1}/{epochs} : {mean(outcome):.3f}')

    def _update_network(self, deltas, eta, reg):
        for layer, delta in zip(self.layers[1:], deltas):
            layer.update(delta, eta, reg)

    def save(self, path='./network'):
        with gopen(path, 'wb') as f:
            dump(self, f)

    @ staticmethod
    def load(path='./network'):
        with gopen(path, 'rb') as f:
            return load(f)

    def __str__(self):
        return f'NeuralNetwork({self.shape})'

    def __len__(self):
        return len(self.layers)
