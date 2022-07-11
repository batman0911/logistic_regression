import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2)


def sigmoid(s):
    return 1/(1 + np.exp(s))


def sgd_logistic_regression(X, y, w_init, eta, tol=1e-4, max_count=10000):
    w = [w_init]
    it = 0
    N = X.shape[1]
    d = X.shape[0]
    print(f'total pont: {N}, point dim: {d}')
    count = 0
    check_w_after = 20
    while count < max_count:
        rand_id = np.random.permutation(N)

        for i in rand_id:
            xi = X[:, i].reshape(d, 1)
            yi = y[i]
            zi = sigmoid(np.dot(w[-1].T, xi))
            w_new = w[-1] - eta*(yi - zi)*xi
            count += 1

            if count % check_w_after == 0:
                if np.linalg.norm(w_new - w[-check_w_after]) < tol:
                    print(f'count: {count}')
                    return w
            w.append(w_new)

    print(f'count: {count}')
    return w


if __name__ == '__main__':
    X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
                   2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]])
    y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
    X = np.concatenate((np.ones((1, X.shape[1])), X), axis=0)
    print(f'X: {X}')

    eta = 0.05
    d = X.shape[0]
    w_init = np.random.randn(d, 1)
    print(f'w_init: {w_init}')

    w = sgd_logistic_regression(X, y, w_init, eta)
    print(w[-1])

    X0 = X[1, np.where(y == 0)][0]
    y0 = y[np.where(y == 0)]
    X1 = X[1, np.where(y == 1)][0]
    y1 = y[np.where(y == 1)]

    plt.plot(X0, y0, 'ro', markersize=8)
    plt.plot(X1, y1, 'bs', markersize=8)

    xx = np.linspace(0, 6, 1000)
    w0 = w[-1][0][0]
    w1 = w[-1][1][0]
    threshold = -w0 / w1
    yy = sigmoid(w0 + w1 * xx)
    plt.axis([-2, 8, -1, 2])
    plt.plot(xx, yy, 'g-', linewidth=2)
    plt.plot(threshold, .5, 'y^', markersize=8)
    plt.xlabel('studying hours')
    plt.ylabel('predicted probability of pass')
    plt.show()