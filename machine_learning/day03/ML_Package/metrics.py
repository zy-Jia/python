import numpy as np


def accuary_score(y_pred, y_test):
    return sum(y_pred==y_test)/len(y_test)


def mse(y_hat, y_test):
    return np.sum((y_hat - y_test) ** 2) / len(y_test)


def r2_score(y_hat, y_test):
    return 1 - mse(y_hat, y_test) / np.var(y_test)