import numpy as np

np.random.seed(2)


def sigmoid(s):
    return 1 / (1 + np.exp(-s))


def predict(X, w):
    return sigmoid(np.dot(X, w))


def calc_error(y_pred, y_label):
    len_label = len(y_label)
    cost = (-y_label * np.log(y_pred) - (1 - y_label) * np.log(1 - y_pred)).sum() / len_label
    return cost


def logistic_grad(X, y, w):
    return np.dot(X.T, sigmoid(np.dot(X, w)) - y) / X.shape[0]


def cost_function(X, y, w):
    y_pred = predict(X, w)
    cost = -y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)
    cost = cost.sum() / len(y)
    return cost


def move_with_direction(w, step_size, grad):
    return w - step_size * grad


class LogisticRegressionOpt:
    def __init__(self,
                 solver='gd',
                 tol=1e-4,
                 max_iter=1000,
                 eta=0.05,
                 check_after=10):
        self.tol = tol
        self.max_iter = max_iter
        self.eta = eta
        self.solver = solver
        self.grad = None
        self.w = None
        self.count = 0
        self.inner_count = 0
        self.cost_list = []
        self.check_after = check_after


    def back_tracking_step_size(self, X, y, w, grad):
        step_size = 1
        alpha = beta = 0.5
        count = 0
        max_iter = 10000
        while cost_function(X, y, move_with_direction(w, step_size, grad)) > \
                cost_function(X, y, w) - alpha * step_size * np.dot(grad.T, grad):
            step_size = beta * step_size
            count += 1
            self.inner_count += 1
            if count > max_iter:
                return step_size
        return step_size

    def fit(self, X, y, w_init):
        if 'gd' == self.solver:
            return self.gd_logistic_regression(X, y, w_init,
                                               self.eta, self.tol, self.max_iter)
        elif 'sgd' == self.solver:
            return self.sgd_logistic_regression(X, y, w_init,
                                                self.eta, self.tol, self.max_iter)
        elif 'bgd' == self.solver:
            return self.gd_back_tracking_logistic_regression(X, y, w_init,
                                                             self.tol, self.max_iter)
        else:
            raise 'unsupported alg'

    def gd_logistic_regression(self, X, y, w_init, step_size, tol, max_iter):
        self.count = 0
        self.w = w_init
        while self.count < max_iter:
            if self.count % 100 == 0:
                y_pred = predict(X, self.w)
                cost = calc_error(y_pred, y)
                self.cost_list.append(cost)

            self.grad = logistic_grad(X, y, self.w)
            self.w = self.w - step_size * self.grad
            self.count += 1

            if np.linalg.norm(self.grad) < tol:
                return [self.w, self.count, self.cost_list]

        return [self.w, self.count, self.cost_list]

    def sgd_logistic_regression(self, X, y, w_init, step_size, tol, max_iter):
        self.count = 0
        self.w = w_init
        N = X.shape[0]
        d = X.shape[1]

        while self.count < max_iter:
            mix_id = np.random.permutation(N)
            for i in mix_id:
                xi = X[i, :].reshape(d, 1)
                yi = y[i].reshape(1, 1)
                zi = sigmoid(np.dot(self.w.T, xi))
                self.grad = (yi - zi) * xi
                self.w = self.w + step_size * self.grad
                self.count += 1

                if self.count % self.check_after == 0:
                    y_pred = predict(X, self.w)
                    cost = calc_error(y_pred, y)
                    self.cost_list.append(cost)
                    if np.linalg.norm(self.grad) < tol:
                        return [self.w, self.count, self.cost_list]

        return [self.w, self.count, self.cost_list]

    def gd_back_tracking_logistic_regression(self, X, y, w_init, tol, max_iter):
        self.count = 0
        self.w = w_init
        while self.count < max_iter:
            if self.count % 2 == 0:
                y_pred = predict(X, self.w)
                cost = calc_error(y_pred, y)
                self.cost_list.append(cost)

            self.grad = logistic_grad(X, y, self.w)
            self.w = self.w - self.back_tracking_step_size(X, y, self.w, self.grad) * self.grad
            self.count += 1

            if np.linalg.norm(self.grad) < tol:
                return [self.w, self.count, self.cost_list]

        return [self.w, self.count, self.cost_list]
