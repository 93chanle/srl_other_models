import numpy as np
import torch
import torch.nn as nn

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return round(np.sqrt(MSE(pred, true)), 2)

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

def predicted_revenue(pred, true):
    pred_non_neg = np.where(pred < 0, 0, pred)
    return np.nansum(np.where(pred_non_neg > true, 0, pred_non_neg)).round(2)

def linex_objective(pred, true, linex_weight):
    diff = pred - true # this order matters
    grad = -(2/linex_weight) * ( np.exp(linex_weight * diff) - 1)
    hess = 2 * np.exp(linex_weight * diff)
    return grad, hess

def linex_eval_metric(pred, true, linex_weight):
    diff = pred - true # this order matters
    loss = (2/np.power(linex_weight, 2))*(np.exp(linex_weight*diff)- linex_weight*diff - 1)
    return np.sqrt((loss).mean()).round(2)

def w_rmse_objective(pred, true, w_rmse_weight):
    diff = pred - true
    weights = np.where(diff >= 0, w_rmse_weight, 1)
    
    # Calculate the weighted RMSE
    weighted_squared_errors = weights * diff**2
    grad = diff * weights
    hess = np.ones_like(grad) * weights
    
    return grad, hess

def w_rmse_eval_metric(pred, true, w_rmse_weight):
    diff = pred - true
    weights = np.where(diff >= 0, w_rmse_weight, 1)
    weighted_squared_errors = weights * diff**2
    weighted_rmse = np.sqrt(np.mean(weighted_squared_errors))
    return  weighted_rmse


def linlin_objective(pred, true, linlin_weight):
    
    diff = pred - true
    weights = np.where(diff >= 0, linlin_weight, 1)
    
    # Calculate the weighted RMSE
    weighted_squared_errors = weights * diff**2
    grad = np.where(diff < 0, -linlin_weight, 1-linlin_weight)
    hess = np.ones_like(grad)
    
    return grad, hess

def linlin_eval_metric(pred, true, linlin_weight):
    diff = pred - true
    loss = np.where(diff < 0, -diff*linlin_weight, diff*(1-linlin_weight))
    return loss.mean()
        
class WeightedRMSE(nn.Module):
    '''
    alpha: weight parameter to penalize more when pred > true
    '''
    def __init__(self, wrmse_weight):
        super(WeightedRMSE, self).__init__()
        self.wrmse_weight = wrmse_weight
    
    def forward(self, pred, true):
        diff = pred - true
        weighted_diff = torch.where(diff > 0, diff*self.wrmse_weight, diff)
        return torch.sqrt((weighted_diff**2).mean())
    
class LinLinLoss(nn.Module):
    '''
    alpha: weight parameter to penalize more when pred > true
    '''
    def __init__(self, linlin_weight):
        super(LinLinLoss, self).__init__()
        self.linlin_weight = linlin_weight # prob of underforcast, 1- : overforecast
    
    def forward(self, pred, true):
        diff = pred - true
        weighted_diff = torch.where(diff < 0, -diff*self.linlin_weight, diff*(1-self.linlin_weight))
        return weighted_diff.mean() # MAE?
     
class LinExLoss(nn.Module):
    def __init__(self, linex_weight):
        super(LinExLoss, self).__init__()
        self.linex_weight = linex_weight
    
    def forward(self, pred, true):
        diff = pred - true # this order matters
        linex_weight = torch.tensor(self.linex_weight)
        a = (2/torch.pow(linex_weight, 2))
        b = (torch.exp(linex_weight*diff)- linex_weight*diff - 1)
        loss = (2/torch.pow(linex_weight, 2))*(torch.exp(linex_weight*diff)- linex_weight*diff - 1)
        return torch.sqrt((loss).mean())

class PositiveMSE(nn.Module):
    def __init__(self):
        super(PositiveMSE, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, true):
        mse_loss = self.mse(pred, true)
        neg_loss = torch.mean(torch.abs(torch.min(pred, torch.zeros_like(pred))))
        return mse_loss + neg_loss
    
class MSLELoss(nn.Module):
    def __init__(self):
        super(MSLELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, true):
        return self.mse(torch.log(pred + 1), torch.log(true + 1))

