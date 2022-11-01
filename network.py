from numpy import mean
from pickle import dump, load
from gzip import open as gopen

from layer import Layer
from lib import reshape_input, random_partion, decide


class NeuralNetwork:
    def __init__(self, layers, activation='sigmoid', cost='mse'):
        if isinstance(layers[0], Layer):
            self.layers = layers
        else:
            self.layers = [
                Layer(layers[0], activation=activation, cost=cost), ]
            prev = self.layers[0]
            for size in layers[1:]:
                self.layers += [Layer(size, prev=prev,
                                      activation=activation, cost=cost), ]
                self.layers[-2].next = self.layers[-1]
                prev = self.layers[-1]
            self.layers = tuple(self.layers)

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
            # without data backpropagate last classification (results cached in layers)
            self.classify(data)

        return self.layers[0].backward(y)

    def train(self, data, y, epochs=1, batch_size=1, eta=1, test_data=None, decisive=False):
        for i in range(epochs):
            partitions, partitions_y = random_partion(data, batch_size, y=y)

            for partition, partition_y in zip(partitions, partitions_y):
                deltas = self._backward(partition_y, data=partition)
                self._update_network(deltas, eta=eta)

            if test_data is not None:
                is_ = self.classify(test_data[0], decisive=decisive)
                should = reshape_input(test_data[1])

                if decisive:
                    outcome = is_ == decide(should)
                    print(f'{i + 1}/{epochs} : {sum(outcome)}/{len(outcome)}')
                else:
                    outcome = self.layers[-1].cost_func.f(is_, should)
                    print(f'{i + 1}/{epochs} : {mean(outcome):.3f}')

    def _update_network(self, deltas, eta):
        for layer, delta in zip(self.layers[1:], deltas):
            layer.update(delta, eta)

    def save(self, path='./network'):
        with gopen(path, 'wb') as f:
            dump(self, f)

    @ staticmethod
    def load(path='./network'):
        with gopen(path, 'rb') as f:
            return load(f)

    def __str__(self) -> str:
        return f'NeuralNetwork({self.shape})'
