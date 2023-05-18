import argparse
import os
import torch
import yaml
import pickle as pkl

from model.embed import DataEmbedding

import xgboost as xgb

from data.argparser import args_parsing

import numpy as np
from datetime import datetime

now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
from data.data_loader import Dataset_XGB
from utils.postprocessing import ProcessedResult

from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error

from model.embed import DataEmbedding

def linex_obj_and_eval(linex_weight):
    """Linear-exponential loss function with "weight" parameter a
    The larger a is the more positive errors are penalized
    When a is small, loss function looks like normal quadratic loss
    Args:
        a (float): positive, range from 0.001
    """
    def objective(pred, true):
        diff = pred - true # this order matters
        grad = -(2/linex_weight) * ( np.exp(linex_weight * diff) - 1)
        hess = 2 * np.exp(linex_weight * diff)
        return grad, hess
    
    def eval_metric(pred, true):
        diff = pred - true # this order matters
        loss = (2/np.power(linex_weight, 2))*(np.exp(linex_weight*diff)- linex_weight*diff - 1)
        return np.sqrt((loss).mean()).round(2)
    
    return objective, eval_metric

def w_rmse_obj_and_eval(w_rmse_weight):
    
    def objective(pred, true):
        
        diff = pred - true
        weights = np.where(diff >= 0, w_rmse_weight, 1)
        
        # Calculate the weighted RMSE
        weighted_squared_errors = weights * diff**2
        grad = diff * weights
        hess = np.ones_like(grad) * weights
        
        return grad, hess
    
    def eval_metric(pred, true):
        diff = pred - true
        weights = np.where(diff >= 0, w_rmse_weight, 1)
        weighted_squared_errors = weights * diff**2
        weighted_rmse = np.sqrt(np.mean(weighted_squared_errors))
        return  weighted_rmse
        
    return objective, eval_metric

def linlin_obj_and_eval(linlin_weight):
    
    def objective(pred, true):
        
        diff = pred - true
        weights = np.where(diff >= 0, linlin_weight, 1)
        
        # Calculate the weighted RMSE
        weighted_squared_errors = weights * diff**2
        grad = np.where(diff < 0, -linlin_weight, 1-linlin_weight)
        hess = np.ones_like(grad)
        
        return grad, hess
    
    def eval_metric(pred, true):
        diff = pred - true
        loss = np.where(diff < 0, -diff*linlin_weight, diff*(1-linlin_weight))
        return loss.mean()
        
    return objective, eval_metric

# objective = w_rmse_objective(5)

class Exp_XGBoost():
    def __init__(self, args):
        self.args = args
        self.model = self._build_model()
        
    def _select_objective_and_eval_metric(self):
        match self.args.loss:
            case 'rmse':
                objective='reg:squarederror'
                eval_metric=mean_absolute_error
            case 'w_rmse':
                objective, eval_metric=w_rmse_obj_and_eval(self.args.w_rmse_weight)
            case 'linex':
                objective, eval_metric=linex_obj_and_eval(self.args.linex_weight)
            case 'linlin':
                objective, eval_metric=linlin_obj_and_eval(self.args.linlin_weight)
        return objective, eval_metric
        
    def _build_model(self):
        
        objective, eval_metric = self._select_objective_and_eval_metric()
        
        model = xgb.XGBRegressor(
            n_estimators=self.args.n_estimators,
            max_depth=self.args.max_depth,
            subsample=self.args.subsample,
            min_child_weight=self.args.min_child_weight,
            learning_rate=self.args.learning_rate,
            objective=objective, # either 'reg:squarederror' or custom objective
            eval_metric=eval_metric, # function, e.g. from sk.metrics
            tree_method="hist"
        )
        
        return MultiOutputRegressor(model)
    
    def _get_data(self, flag):
        
        data_set = Dataset_XGB(
            root_path=self.args.root_path,
            data_path=self.args.data_path,
            flag=flag,
            size=[self.args.input_len,self.args.target_len],
            features='S',
            target='capacity_price',
            timeenc=1,
            freq='d',
            scale='standard',
            cols=None
        )
        
        return data_set
    
    def train(self):
        train_data = self._get_data('train')

        trained_model = self.model.fit(
            # np.concatenate((train_data.matrix_x, train_data.matrix_mark), 1), 
            train_data.matrix_x, 
            train_data.matrix_y)
        return trained_model
    
    def vali(self):
        vali_data = self._get_data('val')
        
        # preds = self.model.predict(
        #     np.concatenate((vali_data.matrix_x, vali_data.matrix_mark), 1)
        # )
        preds = self.model.predict(vali_data.matrix_x)
        trues = vali_data.matrix_y
        
        result = ProcessedResult(preds, trues, args=self.args, data=vali_data)
        
        # Folder for saving result
        folder_path = './results/' + self.args.timestamp + "_" + self.args.data +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        with open('processed_result_test.pkl', 'wb') as f:
            pkl.dump(result, f)
                
        fig = result.plot_pred_vs_true(result.pred)
        fig.savefig(folder_path + 'xgb_result.png', bbox_inches='tight')
        
        print('Finished vali!')
        return