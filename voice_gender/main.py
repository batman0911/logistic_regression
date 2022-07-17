import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    one = np.ones((X.shape[0], 1))
    X = np.concatenate((one, X), axis=1)
    X_train = X[:2000, :]
    X_test = X[-3:, :]
    y = y.reshape((X.shape[0], 1))
    y_train = y[:2000, :]
    y_test = y[-3:]
    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data()
    eta = 0.05
    w_init = np.ones((X_train.shape[1], 1))
    gdlogreg = rg.LogisticRegressionOpt(solver='gd', tol=1e-4, max_iter=50000, eta=0.05)
    w_sgd = gdlogreg.fit(X_train, y_train, w_init)
    print(f'sgd intercept: {w_sgd}')

    logreg = LogisticRegression(tol=1e-4)
    logreg.fit(X_train, y_train)
    print(f'sklearn intercept: {logreg.intercept_}')
    print(f'sklearn coef: {logreg.coef_}')




