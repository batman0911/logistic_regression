import time

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
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
    # X_train = X[:3000, :]
    # X_test = X[-100:, :]
    # y = y.reshape((X.shape[0], 1))
    # y_train = y[:3000, :]
    # y_test = y[-100:]
    mix_id = np.random.permutation(X.shape[0])
    batch_train = mix_id[0:3000]
    batch_test = mix_id[3001:X.shape[0]]

    X_train = X[batch_train, :]
    y_train = y[batch_train].reshape((X_train.shape[0], 1))
    X_test = X[batch_test, :]
    y_test = y[batch_test].reshape((X_test.shape[0], 1))
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data()
    y_train = y_train.reshape((X_train.shape[0], 1))
    y_test = y_test.reshape((X_test.shape[0], 1))
    eta = 0.05
    # w_init = np.ones((X_train.shape[1], 1))
    w_init = np.random.randn(X_train.shape[1], 1)
    gdlogreg = rg.LogisticRegressionOpt(solver='bgd',
                                        tol=1e-4,
                                        max_iter=100000,
                                        step_size=20,
                                        batch_size=100,
                                        check_after=10)
    loop = 1

    t1 = time.time()
    for i in range(loop):
        gdlogreg.fit(X_train, y_train, w_init)
    t2 = time.time()

    # print(f'complete in {(t2 - t1)/loop}')

    print(f'complete in {(t2 - t1)/loop}, count: {gdlogreg.count}, final cost: {gdlogreg.cost_list[-1]}, '
          f'grad norm: {np.linalg.norm(gdlogreg.grad)}')

    plt.plot(range(len(gdlogreg.cost_list)), gdlogreg.cost_list)
    plt.show()

    plt.plot(range(len(gdlogreg.grad_norm_list)), gdlogreg.grad_norm_list)
    plt.show()

    print(f'gd accuracy: {rg.accuracy_gd(X_test, y_test, gdlogreg.w)}')

    logreg = LogisticRegression(tol=1e-4)

    t3 = time.time()
    for i in range(loop):
        logreg.fit(X_train[:, 1:21], y_train)
    t4 = time.time()
    print(f'sklearn complete in {(t4 - t3)/loop}')
    # print(f'sklearn intercept: {logreg.intercept_}')
    # print(f'sklearn coef: {logreg.coef_}')

    print(f'sklearn accuracy: {rg.accuracy_sk(X_test[:, 1:21], y_test, logreg)}')
    # print(logreg.predict(X_test))
    # print(logreg.score(X_test, y_test, logreg.class_weight))







