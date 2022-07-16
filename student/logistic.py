import numpy as np


def sigmoid(s):
    return 1 / (1 + np.exp(-s))


def logistic_sigmoid_GD(X, y, w_init, eta, tol=1e-2):
    w = w_init
    for i in range(1000):
        grad = np.dot(X.T, sigmoid(np.dot(X, w)) - y)
        w_new = w - eta * grad
        if np.linalg.norm(grad) / len(grad) < tol:
            break
        w = w_new
    return [w, i]


# load X,y
X = np.loadtxt("../data/voice.csv", skiprows=(1), delimiter=",", usecols=(range(3)))


def conv(s):
    return 0 if 'f' in s.lower() else 1


y = np.loadtxt("../data/voice.csv", skiprows=(1), delimiter=",", usecols=20,
               encoding=None, converters=conv, dtype=int)

one = np.ones((X.shape[0], 1))
X = np.concatenate((one, X), axis=1)
X_train = X[:3, :]
X_test = X[-2:, :]

print(X_train)

y = y.reshape((X.shape[0], 1))
y_train = y[:3, :]
y_test = y[-2:, :]

w_init = np.ones((X_train.shape[1], 1))
w, i = logistic_sigmoid_GD(X_train, y_train, w_init, 1)
print('Solution found by GD: w = ', w.T, ',\nafter %d iterations.' % (i + 1))

# testing
y_hat = sigmoid(np.dot(X_test, w))
residual = y_test - y_hat
print(y_hat)
