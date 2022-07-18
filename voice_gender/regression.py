import numpy as np
import abc


def sigmoid(s):
    return 1 / (1 + np.exp(-s))


def logistic_sigmoid_GD(X, y, w_init, eta, tol=1e-3, max_count=10000):
    w = w_init
    for i in range(max_count):
        grad = np.dot(X.T, sigmoid(np.dot(X, w)) - y)
        w_new = w - eta * grad
        if np.linalg.norm(grad) / len(grad) < tol:
            break
        w = w_new
    return [w, i]


def predict(X, w):
    return sigmoid(np.dot(X, w))


def calc_error(y_pred, y_label):
    len_label = len(y_label)
    cost = (-y_label * np.log(y_pred) - (1 - y_label) * np.log(1 - y_pred)).sum() / len_label
    return cost


def logistic_grad(X, y, w):
    return np.dot(X.T, sigmoid(np.dot(X, w)) - y) / X.shape[0]


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
        self.grad = None
        self.w = None
        self.count = 0
        self.cost_list = []

    def fit(self, X, y, w_init):
        if 'gd' == self.solver:
            return self.gd_logistic_regression(X, y, w_init,
                                               self.eta, self.tol, self.max_iter)
        elif 'sgd' == self.solver:
            return self.sgd_logistic_regression(X, y, w_init,
                                                self.eta, self.tol, self.max_iter)
        else:
            raise 'unsupported alg'

    def gd_logistic_regression(self, X, y, w_init, eta, tol, max_iter):
        self.count = 0
        self.w = w_init
        while self.count < max_iter:
            y_pred = predict(X, self.w)
            if self.count % 100 == 0:
                cost = calc_error(y_pred, y)
                self.cost_list.append(cost)

            self.grad = logistic_grad(X, y, self.w)
            self.w = self.w - eta * self.grad
            self.count += 1

            if np.linalg.norm(self.grad) < tol:
                return [self.w, self.count, self.cost_list]

        return [self.w, self.count, self.cost_list]

    def sgd_logistic_regression(self, X, y, w_init, eta, tol, max_iter):
        self.count = 0
        self.w = w_init
        N = X.shape[0]
        d = X.shape[1]
        while self.count < max_iter:
            mix_id = np.random.permutation(N)
            for i in mix_id:
                pass



