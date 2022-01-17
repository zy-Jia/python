# -*- coding: utf-8 -*-
import numpy as np
from .metrics import r2_score


class LinearRegression():
    def __init__(self):
        self.a_ = None
        self.b_ = None


    def fit_dot(self, X_train, y_train):
        x_mean = np.mean(X_train)
        y_mean = np.mean(y_train)
        
        self.a_ = (X_train - x_mean).dot(y_train - y_mean) / (X_train - x_mean).dot(X_train - x_mean)
        self.b_ = y_mean - self.a_ * x_mean
        print(self.a_, self.b_)
        return self
    
    def fit(self, X_train, y_train):
        x_mean = np.mean(X_train)
        y_mean = np.mean(y_train)

        d = 0.0
        m = 0.0
        for x_i, y_i in zip(X_train, y_train):
            d += (x_i - x_mean) * (y_i - y_mean)
            m += (x_i - x_mean) * (x_i - x_mean)
        self.a_ = d / m
        self.b_ = y_mean - self.a_ * x_mean
        return self


    def predict(self, X_test):
        return np.array([self._predict(x) for x in X_test])
    

    def _predict(self, xsingle):
        return xsingle * self.a_ + self.b_


    def score(self, X_test, y_test):
        y_hat = self.predict(X_test)
        return r2_score(y_hat, y_test)
        

    def __repr__(self):
        return "LinearRegression()"

if __name__ == '__main__':
    lr = LinearRegression()
    x = np.array([1,2,3,4,5])
    y = np.array([1,3,2,3,5])
    lr.fit(x, y)