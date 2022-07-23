import time

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import figure

figure(figsize=(8, 6), dpi=80)
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import regression as rg

np.random.seed(2)


def conv(s):
    return 0 if 'f' in s.lower() else 1


def load_data():
    global X_train, y_train
    X = np.loadtxt("../data/input/voice.csv", skiprows=(1), delimiter=",", usecols=(range(20)))
    y = np.loadtxt("../data/input/voice.csv", skiprows=(1), delimiter=",", usecols=20,
                   encoding=None, converters=conv, dtype=int)
    np.set_printoptions(precision=4)
    X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    one = np.ones((X.shape[0], 1))
    X = np.concatenate((one, X), axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, y_train, X_test, y_test


def test_sklearn(loop, X_train, X_test, y_train, y_test):
    logreg = LogisticRegression(tol=1e-2)
    t3 = time.time()
    for i in range(loop):
        logreg.fit(X_train[:, 1:21], y_train)
    t4 = time.time()
    print(f'sklearn complete in {(t4 - t3) / loop}')
    print(f'sklearn accuracy: {rg.accuracy_sk(X_test[:, 1:21], y_test, logreg)}')


def test_gd(loop, X_train, X_test, y_train, y_test, solver):
    y_train = y_train.reshape((X_train.shape[0], 1))
    y_test = y_test.reshape((X_test.shape[0], 1))
    eta = 0.05
    # w_init = np.ones((X_train.shape[1], 1))
    w_init = np.random.randn(X_train.shape[1], 1)
    gdlogreg = rg.LogisticRegressionOpt(solver=solver,
                                        tol=1e-2,
                                        max_iter=10000,
                                        step_size=1,
                                        batch_size=100,
                                        check_after=10)
    t1 = time.time()
    for i in range(loop):
        gdlogreg.fit(X_train, y_train, w_init)
    t2 = time.time()
    print(f'{solver} - complete in {(t2 - t1) / loop}, count: {gdlogreg.count}, inner count: {gdlogreg.inner_count}')
    # print(f'complete in {(t2 - t1) / loop}, count: {gdlogreg.count}, inner count: {gdlogreg.inner_count},'
    #       f' final cost: {gdlogreg.cost_list[-1]}, '
    #       f'grad norm: {np.linalg.norm(gdlogreg.grad)}')

    data_loss_func = {
        'count': gdlogreg.check_after * np.asarray(range(len(gdlogreg.cost_list))),
        'loss_func': gdlogreg.cost_list,
        'grad_norm': gdlogreg.grad_norm_list
    }
    df = pd.DataFrame(data_loss_func)
    df.to_csv(f'../data/output/loss_func_{gdlogreg.solver}_10k_1e-2_{gdlogreg.step_size}.csv', index=False)

    plot_data(gdlogreg)
    print(f'gd accuracy: {rg.accuracy_gd(X_test, y_test, gdlogreg.w)}')


def plot_data(gdlogreg):
    plt.plot(gdlogreg.check_after * np.asarray(range(len(gdlogreg.cost_list))), gdlogreg.cost_list)
    plt.title(f'Loss function - {gdlogreg.solver}', fontsize=16)
    plt.ylabel('Value', fontsize=16)
    plt.xlabel('Count', fontsize=16)
    plt.show()
    plt.plot(gdlogreg.check_after * np.asarray(range(len(gdlogreg.grad_norm_list))), gdlogreg.grad_norm_list)
    plt.title(f'Gradient norm - {gdlogreg.solver}', fontsize=16)
    plt.ylabel('Value', fontsize=16)
    plt.xlabel('Count', fontsize=16)
    plt.show()


def plot_multiple(col, x_label, y_label, title):
    gd = pd.read_csv('../data/output/loss_func_gd_10k_1e-2_0.05.csv')
    bgd = pd.read_csv('../data/output/loss_func_bgd.csv')
    bgd_1 = pd.read_csv('../data/output/loss_func_bgd_1.csv')
    sgd = pd.read_csv('../data/output/loss_func_sgd_10k_0.05.csv')
    sgd_batch = pd.read_csv('../data/output/loss_func_sgd_batch_10k_0.05.csv')
    plt.plot(gd['count'], gd[col], label='gd')
    plt.plot(bgd['count'], bgd[col], label='bgd_20')
    plt.plot(bgd_1['count'], bgd_1[col], label='bgd_1')
    plt.plot(sgd['count'], sgd[col], label='sgd')
    plt.plot(sgd_batch['count'], sgd_batch[col], label='sgd_batch')
    plt.xlim(0, 1000)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data()
    loop = 10
    test_sklearn(loop, X_train, X_test, y_train, y_test)
    # for solver in ['gd', 'sgd', 'sgd_batch']:
    #     test_gd(loop, X_train, X_test, y_train, y_test, solver)

    test_gd(loop, X_train, X_test, y_train, y_test, 'bgd')
    # plot_multiple('loss_func', 'Count', 'Value', 'Loss function')
