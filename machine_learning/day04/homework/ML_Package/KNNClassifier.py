# -*- coding: utf-8 -*-
from math import sqrt
import numpy as np
from collections import Counter
from .metrics import accuary_score


class KNNClassifier():
    def __init__(self,k = 14):
        self.k = k
        self.X_train = None
        self.y_train = None


    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        return self


    def predict(self, X_test):
        # return self._predict(X_test)
        return np.array([self._predict(x) for x in X_test])
    

    def _predict(self, xsingle):
        distances = [sqrt(np.sum((x_i - xsingle) ** 2)) for x_i in self.X_train]
        nearst = np.argsort(distances)
        res = [self.y_train[i] for i in nearst[:self.k]]
        votes = Counter(res)
        return votes.most_common(1)[0][0]


    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return accuary_score(y_pred, y_test)


    def __repr__(self):
        return "KNNClassifier()"
