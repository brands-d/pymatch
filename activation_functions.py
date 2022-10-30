from numpy import exp

class ActivationFunction:

    @staticmethod
    def f(data):
        pass

    @staticmethod
    def f_prime(data):
        pass


class Sigmoid(ActivationFunction):

    @staticmethod
    def f(z):
        return 1/(1 + exp(-z))

    @staticmethod
    def f_prime(z):
        return Sigmoid.f(z)*(1-Sigmoid.f(z))
