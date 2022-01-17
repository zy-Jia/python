# -*- coding: utf-8 -*-
import numpy as np
from .metrics import r2_score


class Gradient_Descent():
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self._theta = None


    def fit_gd(self, X_train, y_train, lr=0.01, n_iters=1e4):
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        def loss_function(theta, X_b, y):
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(X_b)
            except:
                return float('inf')
            
        
        def derivative(theta, X_b, y):
            res = np.empty(len(theta))
            res[0] = np.sum(X_b.dot(theta) - y)
            for i in range(1, len(theta)):
                res[i] = (X_b.dot(theta) - y).dot(X_b[:,i])
                
            return res * 2 / len(X_b)
        
        
        def gradient_descent(X_b, y, init_theta, lr, n_iters = 1e4, epsilon = 1e-8):
            theta = init_theta
            cur_iter = 0
            while cur_iter < n_iters:
                gradient = derivative(theta, X_b, y)
                last_theta = theta
                theta = theta - lr * gradient
                if(abs(loss_function(theta, X_b, y) - loss_function(last_theta, X_b, y)) < epsilon):
                    break
                cur_iter += 1
        
            return theta
        
        
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        init_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b, y_train, init_theta, lr, n_iters)

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self


    def predict(self, X_test):
        assert self.intercept_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert X_test.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"

        X_b = np.hstack([np.ones((len(X_test), 1)), X_test])
        return X_b.dot(self._theta)
    
    
    def score(self, X_test, y_test):
        y_hat = self.predict(X_test)
        return r2_score(y_hat, y_test)
        

    def __repr__(self):
        return "gradient_descent()"

if __name__ == '__main__':
    lr = Gradient_Descent()
    x = np.array([1,2,3,4,5])
    y = np.array([1,3,2,3,5])
    lr.fit(x, y)