import numpy as np
from sklearn.linear_model import LogisticRegression

np.random.seed(2)
np.seterr(divide='ignore', over='ignore', invalid='ignore')


def sigmoid(s):
    return 1 / (1 + np.exp(-s))


def predict(X, w):
    return sigmoid(np.dot(X, w))


def predict_class(X, w):
    y_hat = predict(X, w)
    y_hat[y_hat < .5] = 0
    y_hat[y_hat >= .5] = 1
    return y_hat


def accuracy_gd(X_test, y_test, w):
    return 1 - np.mean(np.abs(predict_class(X_test, w) - y_test))


def accuracy_sk(X_test, y_test, logreg: LogisticRegression):
    return 1 - np.mean(np.abs(logreg.predict(X_test) - y_test))


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
                 step_size=0.05,
                 batch_size=10,
                 check_after=10):
        self.tol = tol
        self.max_iter = max_iter
        self.step_size = step_size
        self.solver = solver
        self.grad = None
        self.grad_norm_list = []
        self.w = None
        self.count = 0
        self.inner_count = 0
        self.cost_list = []
        self.check_after = check_after
        self.batch_size = batch_size

    def back_tracking_step_size(self, X, y, w, grad):
        step_size = self.step_size
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
        # disable this in benchmark
        print(f'initial cost: {cost_function(X, y, w_init)}')
        if 'gd' == self.solver:
            return self.gd_logistic_regression(X, y, w_init,
                                               self.step_size, self.tol, self.max_iter)
        elif 'sgd' == self.solver:
            return self.sgd_logistic_regression(X, y, w_init,
                                                self.step_size, self.tol, self.max_iter)
        elif 'sgd_batch' == self.solver:
            return self.sgd_mini_batch_logistic_regression(X, y, w_init,
                                                           self.step_size, self.tol,
                                                           self.max_iter, self.batch_size)
        elif 'bgd' == self.solver:
            return self.gd_back_tracking_logistic_regression(X, y, w_init,
                                                             self.tol, self.max_iter)
        else:
            raise 'unsupported alg'

    def gd_logistic_regression(self, X, y, w_init, step_size, tol, max_iter):
        self.count = 0
        self.w = w_init
        while self.count < max_iter:
            self.handle_gd(X, step_size, y)

            if self.count % self.check_after == 0:
                grad_norm = np.linalg.norm(self.grad)
                # disable this in benchmark
                self.cal_metrics(X, grad_norm, y)
                if grad_norm < tol:
                    return [self.w, self.count, self.cost_list]

        return [self.w, self.count, self.cost_list]

    def handle_gd(self, X, step_size, y):
        self.grad = logistic_grad(X, y, self.w)
        self.w = self.w - step_size * self.grad
        self.count += 1

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
                self.grad = (zi - yi) * xi

                # self.w = self.w - self.back_tracking_step_size(X, y, self.w, self.grad) * self.grad
                self.w = self.w - step_size * self.grad
                self.count += 1

                if self.count % self.check_after == 0:
                    grad_norm = np.linalg.norm(self.grad)
                    # disable this in benchmark
                    self.cal_metrics(X, grad_norm, y)
                    if grad_norm < tol:
                        return [self.w, self.count, self.cost_list]

        return [self.w, self.count, self.cost_list]

    def sgd_mini_batch_logistic_regression(self, X, y, w_init, step_size, tol, max_iter, batch_size):
        self.count = 0
        self.w = w_init

        while self.count < max_iter:
            X_batch, y_batch = self.get_training_batch(X, batch_size, y)

            self.handle_gd(X_batch, step_size, y_batch)

            if self.count % self.check_after == 0:
                grad_norm = np.linalg.norm(self.grad)
                # disable this in benchmark
                self.cal_metrics(X, grad_norm, y)
                if grad_norm < tol:
                    return [self.w, self.count, self.cost_list]

        return [self.w, self.count, self.cost_list]

    def get_training_batch(self, X, batch_size, y):
        mix_id = np.random.permutation(X.shape[0])
        batch = mix_id[0:batch_size]
        return X[batch, :], y[batch]

    def gd_back_tracking_logistic_regression(self, X, y, w_init, tol, max_iter):
        self.count = 0
        self.w = w_init
        while self.count < max_iter:
            self.handle_gd(X, self.back_tracking_step_size(X, y, self.w, self.grad), y)

            if self.count % self.check_after == 0:
                grad_norm = np.linalg.norm(self.grad)
                # disable this in benchmark
                self.cal_metrics(X, grad_norm, y)
                if grad_norm < tol:
                    return [self.w, self.count, self.cost_list]

        return [self.w, self.count, self.cost_list]

    def cal_metrics(self, X, grad_norm, y):
        y_pred = predict(X, self.w)
        cost = calc_error(y_pred, y)
        self.cost_list.append(cost)
        # disable this in benchmark
        if self.count % (10 * self.check_after) == 0:
            print(f'count: {self.count}, cost: {cost}, grad norm: {grad_norm}')
        self.grad_norm_list.append(grad_norm)
