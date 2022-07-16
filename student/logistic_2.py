import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def sigmoid(s):
    return 1/(1 + np.exp(-s))

def logistic_GD(X, y, w_init, t, tol = 1e-3):
    #fixed step size
    w = w_init
    for i in range(10000):
        grad = np.dot(X.T, sigmoid(np.dot(X,w))-y)
        w_new = w - t * grad
        if np.linalg.norm(grad) < tol:
            break
        w = w_new
    return [w, i]

def logistic_GD_backtracking(X, y, w_init, t_init, tol = 1e-2):
    w = w_init
    for i in range(10000):
        #calculate step size
        t = t_init
        alpha = .5
        beta = .5
        grad = np.dot(X.T, sigmoid(np.dot(X, w)) - y)
        a = sigmoid(w-t*grad)
        b = sigmoid(w)
        c = alpha * t * np.linalg.norm(grad)**2
        while np.linalg.norm(a) > np.linalg.norm(b - c):
            t = beta * t
            grad = np.dot(X.T, sigmoid(np.dot(X, w)) - y)
            a = sigmoid(w - t * grad)
            b = sigmoid(w)
            c = alpha * t * np.linalg.norm(grad) ** 2
            #print('bktrack')

        #update
        grad = np.dot(X.T, sigmoid(np.dot(X,w))-y)
        w_new = w - t * grad
        if np.linalg.norm(grad) < tol:
            break
        w = w_new
    return [w, i]

#load X,y
X = np.loadtxt("../data/voice.csv", skiprows=(1),delimiter=",", usecols=(range(20)) )
def conv(s):
    return 0 if 'f' in s.lower() else 1
y = np.loadtxt("../data/voice.csv", skiprows=(1),delimiter=",", usecols=20,
               encoding=None, converters=conv, dtype=int)
np.set_printoptions(precision=2)
X = (X-np.mean(X,axis=0))/np.std(X,axis=0)
one = np.ones((X.shape[0],1))
X = np.concatenate((one,X), axis=1)
X_train = X[:2000,:]
X_test = X[-3:,:]

#y = y.reshape((X.shape[0],1))
y_train = y[:2000]
y_test = y[-3:]

#sklearn
clf = LogisticRegression(tol=1e-2).fit(X_train,y_train)
print('solution by sklearn: w = ', clf.coef_)
print('predict by sklearn: y_hat =', clf.predict(X_test))

w_init = np.ones((X_train.shape[1],1))
y = y.reshape((X.shape[0],1))
y_train = y[:2000,:]
y_test = y[-3:,:]
w, i = logistic_GD(X_train, y_train, w_init, 1)
print('Solution found by GD: w = ', w.T, ',\nafter %d iterations.' %(i+1))

#w1,i1 = logistic_GD_backtracking(X_train, y_train, w_init,1)
#print('Solution found by GD backtracking: w = ', w1.T, ',\nafter %d iterations.' %(i1+1))

#testing
y_hat = sigmoid(np.dot(X_test,w))
residual = y_test - y_hat
print('y_hat',y_hat)






