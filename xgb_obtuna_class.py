from data.argparser import args_parsing
from experiment.exp import Exp_XGBoost
import optuna
from datetime import datetime
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice
from functools import partial

# now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

# args = args_parsing()

class XGB_Optuna():
    def objective(trial, args, search_space):
    
        # param = {
        #     "objective": "binary",
        #     "metric": "auc",
        #     "verbosity": -1,
        #     "boosting_type": "gbdt",
        #     "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        #     "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        #     "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        # }
        
        # SEARCH SPACE
        
        # match args.loss:
        #     case 'linex':
        #         args.linex_weight = trial.suggest_float('linex_weight', 0.001, 1, step=0.01)
        #     case 'w_rmse':
        #         args.w_rmse_weight = trial.suggest_float('w_rmse_weight', 1.0, 10.0, step=0.1)
        #     case 'linlin':
        #         args.linlin_weight = trial.suggest_float('linlin_weight', 0.05, 0.45, step=0.005)
            
        # args.n_estimators = trial.suggest_int("n_estimators", 50, 150)
        # args.max_depth = trial.suggest_int("max_depth", 3, 12)
        # args.learning_rate = trial.suggest_float("learning_rate", 0.05, 0.5)
        # args.min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
        # args.gamma = trial.suggest_float('gamma', 0, 1, step=0.1)
        # args.subsample = trial.suggest_float('subsample', 0.5, 1.0, step=0.1)
        # args.colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0, step=0.1)
        # args.reg_alpha = trial.suggest_loguniform('reg_alpha', 1e-5, 1.0)
        # args.reg_lambda = trial.suggest_loguniform('reg_lambda', 1e-5, 1.0)
        
        match args.loss:
            case 'linex':
                args.linex_weight = trial.suggest_float("linex_weight", 
                                                        search_space['linex_weight'][0], 
                                                        search_space['linex_weight'][1], 
                                                        step=search_space['linex_weight'][2])
            case 'w_rmse':
                args.w_rmse_weight = trial.suggest_float("w_rmse_weight", 
                                                         search_space['w_rmse_weight'][0], 
                                                         search_space['w_rmse_weight'][1],
                                                         step=search_space['w_rmse_weight'][2])
            case 'linlin':
                args.linlin_weight = trial.suggest_float("linlin_weight", 
                                                         search_space['linlin_weight'][0], 
                                                         search_space['linlin_weight'][1],
                                                         step=search_space['linlin_weight'][2])
            
        args.n_estimators = trial.suggest_int("n_estimators", search_space['n_estimators'][0], search_space['n_estimators'][1])
        args.min_child_weight = trial.suggest_int("min_child_weight", search_space['min_child_weight'][0], search_space['min_child_weight'][1])
        args.max_depth = trial.suggest_int("max_depth", search_space['max_depth'][0], search_space['max_depth'][1])
        
        args.learning_rate = trial.suggest_float("learning_rate", search_space['learning_rate'][0], search_space['learning_rate'][1])
        args.gamma = trial.suggest_float("gamma", search_space['gamma'][0], search_space['gamma'][1])
        
        args.subsample = trial.suggest_float("subsample", 
                                             search_space['subsample'][0], 
                                             search_space['subsample'][1],
                                             step=search_space['subsample'][2])
        
        args.colsample_bytree = trial.suggest_float("colsample_bytree", 
                                                    search_space['colsample_bytree'][0], 
                                                    search_space['colsample_bytree'][1],
                                                    step=search_space['colsample_bytree'][2])

        args.reg_alpha = trial.suggest_loguniform("reg_alpha", search_space['reg_alpha'][0], search_space['reg_alpha'][1])
        args.reg_lambda = trial.suggest_loguniform("reg_lambda", search_space['reg_lambda'][0], search_space['reg_lambda'][1])
        
        exp = Exp_XGBoost(args)
        
        exp.train()
        
        loss, revenue, _ = exp.tune()
        
        # # Add a callback for pruning.
        # pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "auc")
        # gbm = lgb.train(param, dtrain, valid_sets=[dvalid], callbacks=[pruning_callback])

        # preds = gbm.predict(valid_x)
        # pred_labels = np.rint(preds)
        # accuracy = sklearn.metrics.accuracy_score(valid_y, pred_labels)
        return loss, revenue
    
    def __init__(self, args, search_space, timestamp, save_path='optuna_studies/xgboost/', 
                 objective=objective, n_trials=100, seed=1993):
        self.args=args
        self.objective=partial(objective, args=args, search_space=search_space)
        self.timestamp=timestamp
        self.seed=seed
        self.save_path=save_path
        self.n_trials=n_trials
        self.set_target(target=0)
        self.study = self._optimize()
    
    def _create_study(self):
        study_name = f"tune_xgboost_{self.args.data}_{self.args.loss}_{self.timestamp}"  # Unique identifier of the study
        storage_name = f"sqlite:///{self.save_path}{study_name}.db"
        study = optuna.create_study(study_name=study_name, storage=storage_name,
                                    load_if_exists=True,
                            # directions=['minimize', 'maximize'],
                            # direction='minimize',
                            sampler=optuna.samplers.TPESampler(seed=self.seed),
                            )
        return study
        
    def _optimize(self):
        study = self._create_study()
        study.optimize(self.objective, n_trials=self.n_trials)
        return study
    
    def set_target(self, target=0):
        # 0 for loss, 1 for revenue, used in plotting results
        targets = {0:'Loss', 1:'Revenue'}
        self.target=target        
        self.target_name = targets[self.target]
    
    def plot_validation_set(self): 
        res = self.study.trials_dataframe()
        try:
            id = res[f'values_{self.target}'].idxmax()
        except KeyError:
            id = res[f'value'].idxmax()
        best_params = self.study.trials[id].params

        # Copy configuration of best model

        for key in best_params.keys():
            vars(self.args)[key] = best_params[key]
            
        exp = Exp_XGBoost(self.args)

        exp.train()

        fig, result = exp.plot_vali()
        
        return fig, result