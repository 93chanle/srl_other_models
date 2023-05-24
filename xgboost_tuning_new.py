import argparse
import os
import yaml
from ray import tune, air
import numpy as np
from ray.air import session, Checkpoint
from ray.tune.search.optuna import OptunaSearch

from datetime import datetime
from experiment.exp import Exp_XGBoost
from data.argparser import args_parsing

import pickle as pkl

# now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

# parser = argparse.ArgumentParser(description='[Tuning] XGBoost')

args = args_parsing()

# print('Args in experiment:')
# print(args)

### TUNING 

# Define search space
# Create Tuner object
# Define objective function

# Define search space 
search_space = {'learning_rate ': tune.randint(3, 12), 
                'eta': tune.uniform(0.01, 0.5),
                'n_estimators': tune.randint(50, 150),
                'subsample': tune.uniform(0.3, 0.9),
                'colsample_bytree': tune.uniform(0.5, 1),
                'alpha': tune.uniform(0, 2), 
                'lambda': tune.uniform(1, 2),
                'lags': tune.randint(10, 50), 
                'input_len': tune.choice([56, 70, 84, 98, 112])}

match args.loss:
    case 'linex':
        search_space['linex_weight'] = tune.quniform(0.01, 3, 0.005)
    case 'w_rmse':
        search_space['w_rmse_weight'] = tune.quniform(1, 10, 0.1)
    case 'linlin':
        search_space['linlin_weight'] = tune.quniform(0.05, 0.45, 0.025)

# Define trainable
# exp.tune() serves as objective function for Ray
def trainable(config):
        
    args.learning_rate  = config["learning_rate "]
    args.eta = config["eta"]
    args.n_estimators = config['n_estimators']
    args.subsample = config['subsample']
    args.colsample_bytree = config['colsample_bytree']
    args.reg_alpha = config['alpha']
    args.reg_lambda = config['lambda']
    args.input_len = config['input_len']
    
    match args.loss:
        case 'linex':
            args.linex_weight = config['linex_weight']
        case 'w_rmse':
            args.w_rmse_weight = config['w_rmse_weight']
        case 'linlin':
            args.linlin_weight = config['linlin_weight']

    print(f'--------------Start new run-------------------')
    
    # print(f'Tune learning rate: {args.learning_rate}')
    # print(f'Tune train epochs: {args.train_epochs}')
    # print(f'Tune alpha: {args.alpha}')
    # print(f'Tune seq_len: {args.seq_len}')
    # print(f'Tune pred_len: {args.pred_len}')
    
    exp = Exp_XGBoost(args)
    
    exp.train()
    
    loss, revenue, result = exp.tune()
    
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
    return f"{trial_name}_{trial_id}"

# Start Tune run
tuner = tune.Tuner(
    trainable,
    tune_config=tune.TuneConfig(
        search_alg=algo,
        num_samples=args.tune_num_samples,
        trial_dirname_creator=my_trial_dirname_creator,
    ),
    run_config=air.RunConfig(
        name=f'tune_XGB_{args.data}_{args.timestamp}',
        local_dir='ray_tune/',
    ),
    param_space=search_space,
)

results = tuner.fit()

best = results.get_best_result(metric='loss', mode='min')

best.config


# Delete all other trials
# folder_path = f'C:/codes/srl_informer/ray_tune/tune_XGB_{args.data}_{now}'
folder_path = os.path.abspath(f'ray_tune\\tune_XGB_{args.data}_{args.timestamp}')

trainable_to_keep = best.log_dir.parts[-1]

# Get a list of all subfolders in the folder
subfolders = os.listdir(folder_path)

# Iterate over the subfolders and delete each one except for the folder to keep
for subfolder in subfolders:
    subfolder_path = os.path.join(folder_path, subfolder)
    if os.path.isdir(subfolder_path) and subfolder != trainable_to_keep and subfolder.find('.') == -1:
        os.system('rm -rf {}'.format(subfolder_path))
