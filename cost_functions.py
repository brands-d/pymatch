from numpy import mean
from numpy import log as ln
from numpy import nansum as nsum


class CostFunction:
    @staticmethod
    def f(data):
        raise NotImplementedError('This is an abstract class defining'
                                  'cost functions. Please use a subclass.')

    @staticmethod
    def nabla(data):
        raise NotImplementedError('This is an abstract class defining'
                                  'cost functions. Please use a subclass.')


class MeanSquaredError(CostFunction):
    @staticmethod
    def f(a, y):
        return nsum((y-a)**2, axis=0) / 2

    @staticmethod
    def nabla(a, y):
        return a - y


class CostEntropy(CostFunction):
    @staticmethod
    def f(a, y):
        return -mean(y*ln(a)+(1-y)*ln(1-a), axis=0)

    @staticmethod
    def nabla(a, y):
        return (-y/a+(1-y)/(1-a))/len(a)
