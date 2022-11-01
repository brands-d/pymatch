from numpy import exp


class ActivationFunction:
    @staticmethod
    def f(data):
        raise NotImplementedError('This is an abstract class defining'
                                  'cost functions. Please use a subclass.')

    @staticmethod
    def f_prime(data):
        raise NotImplementedError('This is an abstract class defining'
                                  'cost functions. Please use a subclass.')


class Sigmoid(ActivationFunction):
    @staticmethod
    def f(z):
        return 1/(1 + exp(-z))

    @staticmethod
    def f_prime(z):
        s_z = Sigmoid.f(z)
        return s_z*(1-s_z)
