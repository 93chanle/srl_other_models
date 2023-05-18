from data.data_loader import Dataset_XGB
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

def xgboost_model(args, objective='reg:squarederror'):

    model = xgb.XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        subsample=args.subsample,
        min_child_weight=args.min_child_weight,
        learning_rate=args.learning_rate,
        objective=objective, # either 'reg:squarederror' or custom objective
        eval_metric=mean_absolute_error, # function, e.g. from sk.metrics
        tree_method="hist"
        )
    
    return model