import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler

from utils.tools import StandardScaler, MinMaxScaler
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')

class Dataset_XGB(Dataset):
    def __init__(self, root_path, flag='train', input_len=48, target_len=1, train_val_test_split = [0.7, 0.15, 0.15],
                 features='S', data_path='SRL_NEG_00_04.csv', 
                 target='capacity_price', scale='standard', inverse=False, timeenc=0, freq='d', cols=None):
        # size [seq_len, label_len, pred_len]
        # info

        self.input_len = input_len
        self.target_len = target_len

        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        
        assert scale in ['minmax', 'standard', None]
        
        self.scale = scale        
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.train_val_test_split = train_val_test_split
        self.__read_data__()

    def __read_data__(self):
        
        if self.scale == 'minmax':
            self.scaler = MinMaxScaler()
          
        elif self.scale == 'standard':
            self.scaler = StandardScaler()
        
        elif self.scale is None:
            self.scaler = None
        
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        
        if self.features=='M' or self.features=='MS':
            df_data = df_raw[self.cols+[self.target]]
            
            # df_raw[['date']+self.cols+[self.target]]
        elif self.features=='S':
            df_data = df_raw[[self.target]]
            
        # cols = list(df_raw.columns); 
        # if self.cols:
        #     cols=self.cols.copy()
        #     cols.remove(self.target)
        # else:
        #     cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')
        # df_raw = df_raw[['date']+self.cols+[self.target]]
        
        # This is the number of sequences we are getting from the input time series
        assert sum(self.train_val_test_split) == 1.0
        
        # This is the number of sequences we are getting from the input time series
        num_test = int(len(df_raw)*self.train_val_test_split[2]) - self.target_len
        num_vali = int(len(df_raw)*self.train_val_test_split[1])
        num_train = len(df_raw) - num_vali - num_test - self.input_len - self.target_len
        
        if self.scale == 'standard':
            train_data = df_data[0:num_train+self.input_len]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        elif self.scale is None:
            data = df_data.values #np.array
            
        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        # Create sequences based on the input, label and predict lengths
        self.seqs_x, self.seqs_y = self.generate_sequences(data)
        self.seqs_x_mark, self.seqs_y_mark = self.generate_sequences(data_stamp)   
        self.seqs_x_date, self.seqs_y_date = self.generate_sequences(df_stamp.date)
        
        # Set train-val-test borders (for sequences)
        # Here number of test and val are calculated first,
        # to make sure they stay the same regardless of input/label/pred len
        
        idx_seqs = [0, num_train, num_train + num_vali, len(self.seqs_x)]
        idx_data = [idx + self.input_len for idx in idx_seqs] 
        idx_data[0] = 0
        
        # Subset sequences according to data type (train val test)
        self.seqs_x = self.seqs_x[idx_seqs[self.set_type]:idx_seqs[self.set_type+1]]
        self.seqs_y  = self.seqs_y[idx_seqs[self.set_type]:idx_seqs[self.set_type+1]]
        
        self.matrix_x = np.array(self.seqs_x).reshape(-1, self.input_len)
        self.matrix_y = np.array(self.seqs_y).reshape(-1, self.target_len)
        
        self.seqs_x_mark = self.seqs_x_mark[idx_seqs[self.set_type]:idx_seqs[self.set_type+1]]
        self.seqs_y_mark = self.seqs_y_mark[idx_seqs[self.set_type]:idx_seqs[self.set_type+1]]
        self.seqs_x_date = self.seqs_x_date[idx_seqs[self.set_type]:idx_seqs[self.set_type+1]]
        self.seqs_y_date = self.seqs_y_date[idx_seqs[self.set_type]:idx_seqs[self.set_type+1]]
        
        self.matrix_mark = np.array(self.seqs_x_mark).reshape(len(self.seqs_x_mark), -1)
        
        self.start_date_x = self.seqs_x_date[0].iloc[0]
        self.end_date_x = self.seqs_x_date[-1].iloc[-1]
        self.start_date_y = self.seqs_y_date[0].iloc[0]
        self.end_date_y = self.seqs_y_date[-1].iloc[-1]
        
        # if flag=='train':
        #     self.target_data_x = df_raw[[self.target]][]
        
        self.date_index_x = pd.date_range(self.start_date_x, self.end_date_x, freq='D')
        self.date_index_y = pd.date_range(self.start_date_y, self.end_date_y, freq='D')
        
        # Raw data for plotting
        self.target_data = df_raw[idx_data[self.set_type]: idx_data[self.set_type + 1]]
        self.target_date_range = pd.to_datetime(self.target_data['date'], format = '%Y-%m-%d')

    # def __getitem__(self, index):
    #     s_begin = index
    #     s_end = s_begin + self.seq_len
    #     r_begin = s_end - self.label_len 
    #     r_end = r_begin + self.label_len + self.pred_len
    
    #     seq_x = self.data_x[s_begin:s_end]
    #     if self.inverse:
    #         seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
    #     else:
    #         seq_y = self.data_y[r_begin:r_end]
    #     seq_x_mark = self.data_stamp[s_begin:s_end]
    #     seq_y_mark = self.data_stamp[r_begin:r_end]

    #     return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __getitem__(self, index):
        
        seq_x = self.seqs_x[index]
        seq_y  = self.seqs_y[index]
        seq_x_mark = self.seqs_x_mark[index]
        seq_y_mark = self.seqs_y_mark[index]
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.seqs_x)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    def generate_sequences(self, data):
        """_summary_

        Args:
            data (np.array): input data (row idx: time)

        Returns:
            list: 2 list of encoder & decoder inputs (with appropriate input, label & pred seq len)
        """
        inputs=[]
        targets=[]
        
        
        for i in range(len(data) - self.input_len - self.target_len + 1): # Index purposes (otherwise error)
            input = data[i: i + self.input_len]
            inputs.append(input)
            
            target_start_idx = i + self.input_len
            # print(f'{dec_start_idx=}')
            target = data[target_start_idx:target_start_idx+self.target_len]
            targets.append(target)
        
        return inputs, targets