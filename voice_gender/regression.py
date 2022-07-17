import numpy as np
import abc


def sigmoid(s):
    r = 0
    try:
        # print(f'cal exp s: {s}')
        r = 1 / (1 + np.exp(-s))
    except:
        print(f'error with s = {s}')

    return r


def logistic_sigmoid_GD(X, y, w_init, eta, tol=1e-3, max_count=10000):
    w = w_init
    for i in range(max_count):
        grad = np.dot(X.T, sigmoid(np.dot(X, w)) - y)
        w_new = w - eta * grad
        if np.linalg.norm(grad) / len(grad) < tol:
            break
        w = w_new
    return [w, i]


class LogisticRegressionOpt:
    def __init__(self,
                 tol=1e-4,
                 max_iter=1000,
                 eta=0.05,
                 solver='gd'):
        self.tol = tol
        self.max_iter = max_iter
        self.eta = eta
        self.solver = solver

    def fit(self, X, y, w_init):
        if 'gd' == self.solver:
            return self.gd_logistic_regression(X, y, w_init,
                                               eta=self.eta,
                                               tol=self.tol,
                                               max_iter=self.max_iter)
        else:
            raise 'unsupported alg'

    def gd_logistic_regression(X, y, w_init, eta=0.05, tol=1e-3, max_iter=50000):
        count = 0
        w = w_init
        while count < max_iter:
            grad = np.dot(X.T, sigmoid(np.dot(X, w)) - y) / X.shape[0]
            w = w - eta * grad
            count += 1
            if np.linalg.norm(grad) < tol:
                return [w, count]

        return [w, count]

