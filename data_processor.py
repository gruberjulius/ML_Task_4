import os
import numpy as np
import pandas as pd
import datetime as dt

from scipy import stats

class DataReader:
    
    def read_intraday_data(self, intraday_data_path):
        
        intraday_df = pd.concat([pd.read_csv(os.path.join(intraday_data_path, f)) for f in os.listdir(intraday_data_path)])
        intraday_df.Date = pd.to_datetime(intraday_df.Date, format='%Y%m%d')
        intraday_df.Time = pd.to_datetime(intraday_df.Time, format='%H:%M:%S.%f').dt.time 
    
        return intraday_df
    
    def read_daily_data(self, daily_data_path):
        
        daily_df = pd.concat([pd.read_csv(os.path.join(daily_data_path, f)) for f in os.listdir(daily_data_path)])
        daily_df.Date = pd.to_datetime(daily_df.Date, format='%Y%m%d')
        daily_df.rename(columns= {'ID':'Id'}, inplace= True)

        return daily_df
        
class DataPrep:
    
    def __init__(self, intraday_data, daily_data) -> None:
        self.intraday_data = intraday_data.copy()
        self.daily_data = daily_data.copy()
        
    def get_features(self):
        
        self.daily_data['MDV_63_sqrt'] = np.sqrt(self.daily_data.MDV_63)        
        self.daily_data.eval('Stock_Split = SharesAdjFactor != 1', inplace= True)
        self.daily_data['Dividend'] =  (self.daily_data['PxAdjFactor'] != 1) &  (~self.daily_data['Stock_Split'])
        
        last_time = dt.time(15, 30)
        intraday_data = self.intraday_df.query('Time == @last_time')
        
        cols_to_merge = ['MDV_63_sqrt', 'Stock_Split', 'Dividend']
        merge_on = ['Date', 'Id']
        intraday_data = intraday_data.merge(self.daily_data[merge_on + cols_to_merge], on = merge_on, how = 'left')
        
        return intraday_data
        
    def get_target(self, clip_MAD = False, normalize = False):
        
        def target(intraday_per_id):
            #calculate the residual return over the next 24 hrs     
            data_3_30 = intraday_per_id[intraday_per_id.Time== dt.time(9, 45)].copy()
            eod_data = intraday_per_id[intraday_per_id.Time == dt.time(16, 0)].copy()
            eod_data.rename(columns= {'CumReturnResid':'CumReturnResid_EOD'}, inplace = True)
            data_3_30 = data_3_30.merge(eod_data[['Date', 'CumReturnResid_EOD']], on = 'Date')
            data_3_30['Resid_t_1'] = data_3_30.CumReturnResid.shift(-1)
            data_3_30.eval('y = (1 + Resid_t_1)* (1+ CumReturnResid_EOD) / (1 + CumReturnResid) -1', inplace= True)
            return data_3_30[['Date', 'y']]
        
        target_df = self.intraday_data[['Date', 'Time', 'Id', 'CumReturnResid' ]].groupby('Id').apply(target).droplevel(-1)
        target_df.reset_index(inplace = True)
        target_df = target_df.merge(self.daily_data[['Date', 'Id', 'EST_VOL']], on = ['Date', 'Id'], how = 'left')
        target_df.dropna(subset='y', inplace = True)
        
        if clip_MAD:
            
            MAD_by_Date = target_df.groupby('Date').apply(lambda x: stats.median_abs_deviation(x['y'])).to_frame()
            MAD_by_Date.reset_index(inplace=True)
            MAD_by_Date.columns = ['Date', 'MAD']
            
            target_df = target_df.merge(MAD_by_Date,on= 'Date')
            target_df['y'] = np.clip(target_df.y, -5 *target_df.MAD, 5 *target_df.MAD )
        
        if normalize:
            target_df['y'] = target_df['y'] / target_df['EST_VOL']
       
        return target_df        