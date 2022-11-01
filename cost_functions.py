from numpy import nansum as nsum


class CostFunction:
    @staticmethod
    def f(data):
        return data

    @staticmethod
    def nabla(data):
        return data


class MeanSquaredError(CostFunction):
    @staticmethod
    def f(a, y):
        return nsum((y-a)**2, axis=0) / 2

    @staticmethod
    def nabla(a, y):
        return a - y
