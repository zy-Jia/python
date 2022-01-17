import numpy as np


def train_test_split(X, y, train_radio=0.2, seed=2048):
    if(seed):
        np.random.seed(seed)
        
    shuffle_indexes = np.random.permutation(len(X))
    
    test_size = int(len(X)*train_radio)
    test_index = shuffle_indexes[:test_size]
    train_index = shuffle_indexes[test_size:]
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    return X_train, X_test, y_train, y_test