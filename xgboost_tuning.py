import argparse
import os
import yaml
from ray import tune, air
from ray.air import session, Checkpoint
from ray.tune.search.optuna import OptunaSearch

import numpy as np
from datetime import datetime

from utils.postprocessing import ProcessedResultXGB

from darts.models.forecasting.xgboost import XGBModel

import pickle as pkl

from data.data_loader import Dataset_SRL_XGBoost

now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

parser = argparse.ArgumentParser(description='[Tuning] XGBoost')


# non-tuneable

parser.add_argument('--product_type', type=str, required=False, default='SRL_NEG_00_04', help='data')

args = parser.parse_args()
print('Args in experiment:')
print(args)

### TUNING 

# Define search space
# Create Tuner object
# Define objective function

# Define search space 
search_space = {'max_depth': tune.randint(3, 12), 
                'eta': tune.uniform(0.01, 0.5),
                'n_estimators': tune.randint(50, 150),
                'subsample': tune.uniform(0.5, 1),
                'colsample_bytree': tune.uniform(0.5, 1),
                'alpha': tune.uniform(0, 2), 
                'lambda': tune.uniform(1, 2),
                'lags': tune.randint(10, 50), 
                }

# Define trainable
# exp.tune() serves as objective function for Ray
def trainable(config):
        
    args.lags = config["lags"]
    args.max_depth = config["max_depth"]
    args.eta = config["eta"]
    args.n_estimators = config['n_estimators']
    args.subsample = config['subsample']
    args.colsample_bytree = config['colsample_bytree']
    args.reg_alpha = config['alpha']
    args.reg_lambda = config['lambda']

    print(f'--------------Start new run-------------------')
    
    # print(f'Tune learning rate: {args.learning_rate}')
    # print(f'Tune train epochs: {args.train_epochs}')
    # print(f'Tune alpha: {args.alpha}')
    # print(f'Tune seq_len: {args.seq_len}')
    # print(f'Tune pred_len: {args.pred_len}')
    
    dataset = Dataset_SRL_XGBoost(root_path='C:/codes/srl_informer/data/processed/', product_type=args.product_type)
    model = XGBModel(
        lags=args.lags,
        max_depth=args.max_depth,
        eta=args.eta,
        n_estimators=args.n_estimators,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
    )
    
    model.fit(dataset.train)
    pred = model.predict(len(dataset.val))
    
    pred = pred.pd_dataframe()['capacity_price']
    true = dataset.val.pd_dataframe()['capacity_price']

    result = ProcessedResultXGB(pred, true, args, dataset)
    
    revenue = result.predict_revenue(result.pred)
    loss = result.rmse(result.pred)
    fig = result.plot_pred_vs_true(result.pred)
    fig.savefig(f'xgboost.png', bbox_inches = 'tight')
    fig = result.plot_pred_vs_true(result.pred_naive)
    fig.savefig(f'naive.png', bbox_inches = 'tight')

        # Dump result object
    with open('processed_result.pickle', 'wb') as f:
        pkl.dump(result, f)
    
    session.report({"revenue": revenue, 
                    "loss": loss,
                    }
                   )  # Report to Tune
    
    print(f'Predicted revenue: {revenue}') 
    print(f'Loss: {loss}') 
    print(f'--------------End run-------------------')
    
# Define search algorithm
algo = OptunaSearch(metric=["loss", "revenue"], mode=["min", "max"])

# Custom trial name creator
def my_trial_dirname_creator(trial):
    trial_name = trial.trainable_name
    trial_id = trial.trial_id
    return f"my_prefix_{trial_name}_{trial_id}"

# Start Tune run
tuner = tune.Tuner(
    trainable,
    tune_config=tune.TuneConfig(
        search_alg=algo,
        num_samples=300,
        trial_dirname_creator=my_trial_dirname_creator,
    ),
    run_config=air.RunConfig(
        name=f'tune_XGB_{args.product_type}_{now}',
        local_dir='ray_tune/',
    ),
    param_space=search_space,
)

results = tuner.fit()

best = results.get_best_result(metric='loss', mode='min')

best.config

# Delete all other trials
folder_path = f'C:/codes/srl_informer/ray_tune/tune_XGB_{args.product_type}_{now}'

trainable_to_keep = best.log_dir.parts[-1]

# Get a list of all subfolders in the folder
subfolders = os.listdir(folder_path)

# Iterate over the subfolders and delete each one except for the folder to keep
for subfolder in subfolders:
    subfolder_path = os.path.join(folder_path, subfolder)
    if os.path.isdir(subfolder_path) and subfolder != trainable_to_keep and subfolder.find('.') == -1:
        os.system('rm -rf {}'.format(subfolder_path))
