import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import regression as rg

np.random.seed(2)


def conv(s):
    return 0 if 'f' in s.lower() else 1


def load_data():
    global X_train, y_train
    X = np.loadtxt("../data/voice.csv", skiprows=(1), delimiter=",", usecols=(range(20)))
    y = np.loadtxt("../data/voice.csv", skiprows=(1), delimiter=",", usecols=20,
                   encoding=None, converters=conv, dtype=int)
    np.set_printoptions(precision=4)
    X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    one = np.ones((X.shape[0], 1))
    X = np.concatenate((one, X), axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, y_train, X_test, y_test


def test_sklearn(loop, X_train, X_test, y_train, y_test):
    logreg = LogisticRegression(tol=1e-4)
    t3 = time.time()
    for i in range(loop):
        logreg.fit(X_train[:, 1:21], y_train)
    t4 = time.time()
    print(f'sklearn complete in {(t4 - t3) / loop}')
    print(f'sklearn accuracy: {rg.accuracy_sk(X_test[:, 1:21], y_test, logreg)}')


def test_gd(loop, X_train, X_test, y_train, y_test):
    y_train = y_train.reshape((X_train.shape[0], 1))
    y_test = y_test.reshape((X_test.shape[0], 1))
    eta = 0.05
    # w_init = np.ones((X_train.shape[1], 1))
    w_init = np.random.randn(X_train.shape[1], 1)
    gdlogreg = rg.LogisticRegressionOpt(solver='gd',
                                        tol=1e-4,
                                        max_iter=100000,
                                        step_size=0.05,
                                        batch_size=100,
                                        check_after=1000)
    t1 = time.time()
    for i in range(loop):
        gdlogreg.fit(X_train, y_train, w_init)
    t2 = time.time()
    print(f'complete in {(t2 - t1)/loop}, count: {gdlogreg.count}, inner count: {gdlogreg.inner_count}')
    # print(f'complete in {(t2 - t1) / loop}, count: {gdlogreg.count}, final cost: {gdlogreg.cost_list[-1]}, '
    #       f'grad norm: {np.linalg.norm(gdlogreg.grad)}')
    plt.plot(range(len(gdlogreg.cost_list)), gdlogreg.cost_list)
    plt.show()
    plt.plot(range(len(gdlogreg.grad_norm_list)), gdlogreg.grad_norm_list)
    plt.show()
    print(f'gd accuracy: {rg.accuracy_gd(X_test, y_test, gdlogreg.w)}')


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data()
    loop = 20
    test_sklearn(loop, X_train, X_test, y_train, y_test)
    test_gd(loop, X_train, X_test, y_train, y_test)








