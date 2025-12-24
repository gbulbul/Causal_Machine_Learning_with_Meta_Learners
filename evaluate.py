import numpy as np

def rmse(x, y):
    return np.sqrt(np.mean((x - y) ** 2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
