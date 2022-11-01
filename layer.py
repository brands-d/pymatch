from numpy import sum as nsum
from numpy.random import standard_normal
from numpy import dot, multiply, einsum, printoptions, array2string

from lib import decide
from activation_functions import Sigmoid
from cost_functions import MeanSquaredError


class Layer:
    def __init__(self, size, next_=None, prev=None, initial='gaussian', activation='sigmoid', cost='mse'):
        self.size = size
        self.next = next_
        self.prev = prev
        self._z_cache = None
        self._a_cache = None

        if activation == 'sigmoid':
            self.act_func = Sigmoid
        else:
            raise NotImplementedError

        if cost == 'mse':
            self.cost_func = MeanSquaredError
        else:
            raise NotImplementedError

        if self.prev is None:
            self.weight = None
            self.bias = None
        else:
            if initial == 'gaussian':
                self.weight = standard_normal((size, self.prev.size))
                self.bias = standard_normal((size, 1))
            else:
                self.weight, self.bias = initial

    def forward(self, data, decisive=False):
        if self.prev is None:
            # first layer treats input as weighted input
            self._a_cache = data
        else:
            self._z_cache = dot(self.weight, data) + self.bias
            self._a_cache = self.act_func.f(self._z_cache)
        if decisive:
            i = decide(self._a_cache)
            return i
        else:
            return self._a_cache

    def backward(self, y):
        if self.prev is None:
            # no backward for first layer
            return self.next.backward(y)
        if self.next is None:
            # last layer
            deltas = []
            aux = self.cost_func.nabla(self._a_cache, y)
        else:
            deltas = self.next.backward(y)
            aux = dot(self.next.weight.T, deltas[0])
        prime = self.act_func.f_prime(self._z_cache)
        delta = multiply(aux, prime)

        return [delta, ] + deltas

    def update(self, delta, eta=1):
        scaling = eta / len(delta[0, :])
        self.weight -= scaling * einsum('kl,jl->kj', delta, self.prev._a_cache)
        self.bias -= scaling * nsum(delta, axis=1, keepdims=True)

    def __str__(self):
        with printoptions(precision=2):
            if self.prev is None:
                output = f'Layer({self.size})'
            else:
                output = f'Layer({self.size})\nb = {array2string(self.bias, prefix="b = ")}\nw = {array2string(self.weight, prefix="w = ")}'

        return output
