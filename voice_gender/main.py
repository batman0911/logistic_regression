import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import regression as rg


np.random.seed(2)


def conv(s):
    return 0 if 'f' in s.lower() else 1


if __name__ == '__main__':
    X = np.loadtxt("../data/voice.csv", skiprows=(1), delimiter=",", usecols=(range(20)))
    y = np.loadtxt("../data/voice.csv", skiprows=(1), delimiter=",", usecols=20,
                   encoding=None, converters=conv, dtype=int)
    np.set_printoptions(precision=2)
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    one = np.ones((X.shape[0], 1))
    X = np.concatenate((one, X), axis=1)
    X_train = X[:2000, :]
    X_test = X[-3:, :]
    y = y.reshape((X.shape[0], 1))
    y_train = y[:2000, :]
    y_test = y[-3:]
    print(f'X shape: {X_train.shape}')
    # print(X.head())
    eta = 1
    d = X_train.shape[1]
    # print(f'X sgd: {X_train}')
    print(f'X shape: {X_train.shape}')
    # w_init = np.random.randn(d, 1)
    w_init = np.ones((X_train.shape[1], 1))
    w_sgd = rg.gd_logistic_regression(X_train, y_train, w_init, eta)
    print(f'sgd intercept: {w_sgd}')

    # print(f'predict by sklean: {logreg.predict(X_test)}')



