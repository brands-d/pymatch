from numpy import exp


class ActivationFunction:
    @staticmethod
    def f(data):
        return data

    @staticmethod
    def f_prime(data):
        return data


class Sigmoid(ActivationFunction):
    @staticmethod
    def f(z):
        return 1/(1 + exp(-z))

    @staticmethod
    def f_prime(z):
        s_z = Sigmoid.f(z)
        return s_z*(1-s_z)
