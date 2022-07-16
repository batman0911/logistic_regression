import numpy as np


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


def gd_logistic_regression(X, y, w_init, eta, tol=1e-3, max_count=10000):
    N = X.shape[0]
    d = X.shape[1]
    count = 0
    w = w_init
    while count < max_count:
        # sum = np.zeros((d, 1))
        # # rand_id = np.random.permutation(N)
        # for i in range(N):
        #     # xi = X[:, i].reshape(d, 1)
        #     xi = X[i, :].reshape(d, 1)
        #     yi = y[i]
        #     grad = (sigmoid(np.dot(w.T, xi)) - yi) * xi
        #     sum += eta * grad
        #
        # w = w - sum
        # if np.linalg.norm(sum) < tol:
        #     print(f'end before max count: {count}')
        #     return w
        # count += 1
        grad = np.dot(X.T, sigmoid(np.dot(X, w)) - y)
        w = w - eta * grad
        count += 1
        if np.linalg.norm(grad) < tol:
            return [w, count]

    return [w, count]



