import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import time

def sigmoid(s):
    return 1/(1 + np.exp(-s))

def logistic_GD(X, y, w_init, t, tol = 1e-2):
    #fixed step size
    w = w_init
    for i in range(5000):
        y_hat = sigmoid(np.dot(X,w))
        grad = np.dot(X.T, y_hat-y)/X.shape[0]
        w_new = w - t * grad
        loss = -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

        if np.linalg.norm(grad) < tol:
            break
        w = w_new
    print('\nstep', i, ' with loss = ', np.linalg.norm(loss)/X.shape[1])
    return [w, i]

def logistic_GD_backtracking(X, y, w_init, t_init, tol = 1e-4):
    w = w_init
    for i in range(5000):
        #calculate step size
        t = t_init
        alpha = .5
        beta = .5
        grad = np.dot(X.T, sigmoid(np.dot(X, w)) - y)/ X.shape[0]
        a = sigmoid(w-t*grad)
        b = sigmoid(w)
        c = alpha * t * np.linalg.norm(grad)**2
        while np.linalg.norm(a) > np.linalg.norm(b - c):
            t = beta * t
            grad = np.dot(X.T, sigmoid(np.dot(X, w)) - y)/ X.shape[0]
            a = sigmoid(w - t * grad)
            b = sigmoid(w)
            c = alpha * t * np.linalg.norm(grad) ** 2
            #print('bktrack')

        #update
        y_hat = sigmoid(np.dot(X, w))
        grad = np.dot(X.T, y_hat - y) / X.shape[0]
        w_new = w - t * grad
        loss = -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

        if np.linalg.norm(grad) < tol:
            break
        w = w_new
    print('\nstep', i, ' with loss = ', np.linalg.norm(loss) / X.shape[1])
    return [w, i]

def sgrad(X, y ,w, i, shufle_id):
    true_id = shufle_id[i]
    xi = X[true_id,:]
    yi = y[true_id,:]
    yi_hat = sigmoid(np.dot(xi,w))
    g = xi * (yi_hat - yi)
    g = g.reshape((X.shape[1], 1))
    return [g, yi_hat]

def logistic_SGD(X, y, w_init, t, tol = 1e-2):
    N = X.shape[0]
    w = w_init
    #print('w_init',w_init.shape)
    count = 0
    for epoch in range(10):
        shufle_id = np.random.permutation(N)
        for i in range(N):
            count += 1
            grad, yi_hat = sgrad(X, y, w, i, shufle_id)
            #print('single grad',grad.shape)
            w_new = w - t * grad
            #print('w_new',w_new.shape)
            if np.linalg.norm(grad) < tol:
                break
            w = w_new
            loss = -(y * np.log(yi_hat) + (1 - y) * np.log(1 - yi_hat))
    print('\nstep', i, ' with loss = ', np.linalg.norm(loss))
    return [w, i]

def pred(y_hat):
    y_hat[y_hat<.5] = 0
    y_hat[y_hat>=.5] = 1
    return y_hat

#load X,y
X = np.loadtxt("voice.csv", skiprows=(1),delimiter=",", usecols=(range(20)) )
def conv(s):
    return 0 if 'f' in s.lower() else 1
y = np.loadtxt("voice.csv", skiprows=(1),delimiter=",", usecols=20,
               encoding=None, converters=conv, dtype=int)
np.set_printoptions(precision=2)
X = (X-np.min(X,axis=0))/(np.max(X,axis=0)-np.min(X,axis=0))
one = np.ones((X.shape[0],1))
X = np.concatenate((one,X), axis=1)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

#sklearn
t1 = time.time()
clf = LogisticRegression(tol=1e-2).fit(X_train,y_train)
t2 = time.time()
print('solution by sklearn: w = ', clf.coef_)
#print('predict by sklearn: y_hat =', clf.predict(X_test))
print("test accuracy: {} %".format(100 - np.mean(np.abs(clf.predict(X_test) - y_test)) * 100))
print("time execute: {:.3f}(s)".format(t2-t1))

#GD
w_init = np.ones((X_train.shape[1],1))
y = y.reshape((X.shape[0],1))
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
t1 = time.time()
w, i = logistic_GD(X_train, y_train, w_init, 0.2)
t2 = time.time()
print('Solution found by GD: w = ', w.T, ',\nafter %d iterations.' %(i+1))
y_hat = sigmoid(np.dot(X_test,w))
#print('y_hat',y_hat)
print("test accuracy: {} %".format(100 - np.mean(np.abs(pred(y_hat) - y_test)) * 100))
print("time execute: {:.3f}(s)".format(t2-t1))

#GD backtracking
#w1,i1 = logistic_GD_backtracking(X_train, y_train, w_init,1)
#print('Solution found by GD backtracking: w = ', w1.T, ',\nafter %d iterations.' %(i1+1))

#SGD
t1 = time.time()
w2,i2 = logistic_SGD(X_train, y_train, w_init,0.2)
t2 = time.time()
print('Solution found by SGD: w = ', w2.T, ',\nafter %d iterations.' %(i2+1))
y_hat = sigmoid(np.dot(X_test,w2))
print("test accuracy: {} %".format(100 - np.mean(np.abs(pred(y_hat) - y_test)) * 100))
print("time execute: {:.3f}(s)".format(t2-t1))

#mini batch