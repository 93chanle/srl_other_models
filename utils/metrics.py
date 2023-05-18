import numpy as np

def MSE(pred, true):
    return np.mean((pred-true)**2).round(2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def weighted_RMSE(pred, true, w_rmse_weight):
    diff = pred - true
    weighted_diff = np.where(diff > 0, diff*w_rmse_weight, diff)
    return np.sqrt((weighted_diff**2).mean()).round(2)

def LinEx(pred, true, linex_weight):
    diff = pred - true # this order matters
    loss = (2/np.power(linex_weight, 2))*(np.exp(linex_weight*diff)- linex_weight*diff - 1)
    return np.sqrt((loss).mean()).round(2)

def LinLin(pred, true, linlin_weight):
    diff = true - pred # positive = underestimation, negative = overestimation
    loss = np.where(diff < 0, -diff*linlin_weight, diff*(1-linlin_weight))
    return loss.mean().round(2)

