import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.metrics import weighted_RMSE, RMSE, MSE, LinEx, LinLin
import matplotlib.dates as mdates

from utils.tools import add_line_breaks_to_args_string


class ProcessedResult():
    def __init__(self, preds, trues, args, data):
        self.args = args
        self.num_pred = preds.shape[0]
        self.data = data
        self.pred_raw = self.convert_seq(preds, inverse=False)
        self.true_raw = self.convert_seq(trues, inverse=False)
        self.pred = self.convert_seq(preds, inverse=True)
        self.true = self.convert_seq(trues, inverse=True)
        self.pred_naive = self.true.shift(1)
        
    
    def convert_seq(self, seq_raw, inverse=True):
        if inverse: 
            seq = self.data.scaler.inverse_transform(seq_raw)
        else: seq = seq_raw
        
        if seq.shape[1] == 1:
            return pd.Series(seq.mean(1).squeeze())
            
        else:
            array = seq.squeeze()
            array = np.array([np.concatenate([np.repeat(np.nan, i), array[i], np.repeat(np.nan, self.num_pred-i-1)]) for i in np.arange(self.num_pred)])
            df = pd.DataFrame(array.transpose())
            average = df.mean(axis=1)
            
            return average[:self.num_pred] # so that

    def plot_pred_vs_true(self, pred):
        
        fig, ax = plt.subplots(figsize=(15,5))
        pred_non_neg = np.where(pred < 0, 0, pred)

        ax.plot(self.data.target_date_range, self.true, label='True', color='mediumturquoise')
        ax.plot(self.data.target_date_range, pred, label ='Raw prediction', linestyle ='--', alpha = 0.3, color='green')
        ax.plot(self.data.target_date_range, pred_non_neg, label ='Predicted SRL price', color='tomato')

        # Plot where revenues are made
        range = self.data.target_date_range.reset_index(drop=True)
        diff = pred - self.true
        query = (diff < 0) & (pred > 0) # Positive predictions which are lower than trues
        ax.plot(range[query], pred_non_neg[query], '.', alpha=0.4, color='black', label ='Revenue made')

        # plt.annotate(f'Predicted revenue: {self.predict_revenue(pred)}€', 
        #              xy=(0.05, 0.9), xycoords='axes fraction',
        #              bbox=dict(boxstyle="round,pad=0.3", fc="cyan", ec="b", lw=1, alpha=0.5))
        # plt.annotate(f'Weighted RMSE (alpha={self.args.alpha}): {self.weighted_rmse(pred)}', 
        #              xy=(0.05, 0.8), xycoords='axes fraction',
        #              bbox=dict(boxstyle="round,pad=0.3", fc="cyan", ec="b", lw=1, alpha=0.5))

        plt.annotate(f'Predicted revenue: {self.predict_revenue(pred)}€, Baseline revenue: {self.predict_revenue(self.pred_naive)}€, Loss: {self.loss(pred)}', 
                        xy=(0.1, -0.2), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="cyan", ec="b", lw=1, alpha=0.2))


        args_dict = self.args.__dict__

        # loss_weights = ['linex_weight', 'w_rmse_weight', 'linlin_weight']

        # if f'{self.args.loss}_weight' in loss_weights:
        #     for loss_weight in loss_weights:
        #         if loss_weight != f'{self.args.loss}_weight' and f'{self.args.loss}_weight' in args_dict.keys():
        #             # del args_dict[loss_weight]
        #             try:
        #                 args_dict.pop(loss_weight)
        #             except KeyError:
        #                 print('Debug')

        args = add_line_breaks_to_args_string(args_dict, max_len=120)

        plt.annotate(args, 
                        xy=(0.1, -0.8), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="gray", ec="b", lw=1, alpha=0.1))

        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # Title
        ax.set_title(f'Product {self.args.data}, {self.data.target_date_range.iloc[0]._date_repr} to {self.data.target_date_range.iloc[-1]._date_repr} (Validation Set)') # Get string representation
        plt.close()
        return(fig)
    
    def predict_revenue(self, pred):
        pred_non_neg = np.where(pred < 0, 0, pred)
        return np.nansum(np.where(pred_non_neg > self.true, 0, pred_non_neg)).round(2)
    
    def loss(self, pred):
        match self.args.loss:
            case 'linex':
                result = LinEx(pred, self.true, self.args.linex_weight)
            case 'w_rmse':
                result = weighted_RMSE(pred, self.true, self.args.w_rmse_weight)
            case 'rmse':
                result = RMSE(pred, self.true)
            case 'linlin':
                result = LinLin(pred, self.true, self.args.linlin_weight)
        return result