# -*- coding: utf-8 -*-
import numpy as np
from .metrics import r2_score


class MultivarateRegression():
    def __init__(self):
        self.__theta = None
        self.interception_ = None
        self.coff_ = None


    def fit_normal(self, X_train, y_train):
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self.__theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        self.interception = self.__theta[0]
        self.coff_ = self.__theta[1:]
        return self
    
    
    def predict(self, X_test):
        X_b = np.hstack([np.ones((len(X_test), 1)), X_test])
        return X_b.dot(self.__theta)


    def score(self, X_test, y_test):
        y_hat = self.predict(X_test)
        return r2_score(y_hat, y_test)
        

    def __repr__(self):
        return "MultivarateRegression()"

if __name__ == '__main__':
    lr = MultivarateRegression()
    x = np.array([1,2,3,4,5])
    y = np.array([1,3,2,3,5])
    lr.fit(x, y)