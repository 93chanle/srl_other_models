import argparse
import os
import yaml
import torch

import numpy as np
from datetime import datetime

def args_parsing():

    now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    parser = argparse.ArgumentParser(description='[XGBoost] Forecasting')

    # Main arguements
    parser.add_argument('--data', type=str, required=False, default='SRL_NEG_00_04', help='data')
    parser.add_argument('--model', type=str, required=False, default='xgboost',help='model of experiment, options: []')

    parser.add_argument('--loss', type=str, default='rmse',help='customized loss functions, one of [w_rmse, linex, linlin, rmse]')

    parser.add_argument('--w_rmse_weight', type=float, default=5,help='weighted parameter for weighted rmse loss function')
    parser.add_argument('--linex_weight', type=float, default=0.05,help='weighted parameter for linear-exponential loss function')
    parser.add_argument('--linlin_weight', type=float, default=0.1,help='weighted parameter for linlin / pinball loss function')

    parser.add_argument('--input_len', type=int, default=30, help='input sequence length')
    parser.add_argument('--target_len', type=int, default=1, help='prediction length')

    parser.add_argument('--timestamp', type=str, default=now)
    
    parser.add_argument('--root_path', type=str, default= 'data\\processed\\SRL\\', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='SRL_NEG_00_04.csv', help='data file')    
    parser.add_argument('--features', type=str, default='S', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--cols', type=str, nargs='+', help='external col names from the data files as the additional input features (not including target)')
    
    parser.add_argument('--scale', type=str, default='standard', help='forecasting task, options: [standard, minmax, none]')
    parser.add_argument('--target', type=str, default='capacity_price', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='d', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

    # Model arguments
    
    parser.add_argument('--n_estimators', type=int, default=100, help='The number of gradient boosted trees.')
    parser.add_argument('--max_depth', type=int, default=3, help='Maximum depth of a tree.')
    parser.add_argument('--min_child_weight', type=float, default=1.0, help='Minimum sum of instance weight needed in a child.')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Boosting learning rate.')
    parser.add_argument('--subsample', type=float, default=1.0, help='Subsample ratio of the training instances.')
    parser.add_argument('--colsample_bytree', type=float, default=1.0, help='Subsample ratio of columns when constructing each tree.')
    parser.add_argument('--reg_gamma', type=float, default=0.0, help='L2 regularization.')
    parser.add_argument('--reg_alpha', type=float, default=0.0, help='L1 regularization.')

    # FOR TUNING
    parser.add_argument('--tune_num_samples', type=int, default=5, help='Number of sample interations in hyperparameter tuning')

    args = parser.parse_args()
    
    args.root_path = os.path.normpath(args.root_path)

    args.freq = args.freq[-1:]

    # Pass default values for external data incorporation
    if args.features == 'MS' and args.cols is None:
        args.cols = ['gas', 'coal']

    print('Args in experiment:')
    print(args)
    print('')
    return args

# with open("args.yaml", "w") as f:
#     yaml.dump(args, f)

