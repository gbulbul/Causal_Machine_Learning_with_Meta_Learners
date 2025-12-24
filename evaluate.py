import numpy as np
from simulate_data import simulate_causal_data
from meta_learners import s_learner, t_learner
import matplotlib.pyplot as plt

def rmse(x, y):
    return np.sqrt(np.mean((x - y) ** 2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
