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
        intraday_df.set_index(['Date', 'Id'], inplace=True)

        return intraday_df
    
    def read_daily_data(self, daily_data_path):
        
        daily_df = pd.concat([pd.read_csv(os.path.join(daily_data_path, f)) for f in os.listdir(daily_data_path)])
        daily_df.Date = pd.to_datetime(daily_df.Date, format='%Y%m%d')
        daily_df.rename(columns= {'ID':'Id'}, inplace= True)
        daily_df.set_index(['Date', 'Id'], inplace=True)
    
        return daily_df
        
class DataPrep:
    
    def __init__(self, intraday_data, daily_data) -> None:
        self.intraday_data = intraday_data.copy()
        self.daily_data = daily_data.copy()

        
    def get_features(self):
        
        self.daily_data['MDV_63_sqrt'] = np.sqrt(self.daily_data.MDV_63)
        
        #corp actions
        self.daily_data['PxAdjFactorRatio'] = self.daily_data.groupby('Id', as_index=False).apply(lambda x: x.PxAdjFactor.div(x.PxAdjFactor.shift())).droplevel(0)
        self.daily_data['SharesAdjFactorRatio'] = self.daily_data.groupby('Id', as_index=False).apply(lambda x: x.SharesAdjFactor.div(x.SharesAdjFactor.shift())).droplevel(0)    
        self.daily_data.PxAdjFactorRatio.fillna(1, inplace= True)
        self.daily_data.SharesAdjFactorRatio.fillna(1, inplace= True)
        self.daily_data.eval('Stock_Split = SharesAdjFactorRatio != 1', inplace= True)
        self.daily_data.eval('Dividend =  PxAdjFactorRatio != 1/SharesAdjFactorRatio', inplace= True)
        
        #rolling returns
        eod_data = self.intraday_data[self.intraday_data.Time == dt.time(16, 0)][['CumReturnResid']].copy() 
        
        rolling_periods = [5, 10, 20]
        for period in rolling_periods:
            eod_data[f'Rolling_Return_{period}d'] = eod_data.groupby('Id')['CumReturnResid'].rolling(window = period, min_periods= 1).sum().droplevel(0)

        #delte
        #eod_data[f'Rolling_Return_{period}d'] = eod_data.groupby('Id')['CumReturnResid'].rolling(window = period, min_periods= 1).sum().droplevel(0)
        rolling_return_cols =  [f'Rolling_Return_{i}d' for i in rolling_periods]
        rolling_prev_returns = eod_data[rolling_return_cols].groupby('Id').shift(1)
        
        last_time = dt.time(15, 30)
        intraday_data = self.intraday_data.query('Time == @last_time')
        cols_to_merge = ['MDV_63_sqrt', 'Stock_Split', 'Dividend'] + ['PxAdjFactorRatio', 'SharesAdjFactorRatio' ]
        intraday_data = intraday_data.join(self.daily_data[cols_to_merge])
        
        #join rolling returns until t- 1 and append return asof 3:30 from prediction day
        intraday_data = intraday_data.join(rolling_prev_returns)
        for col in rolling_return_cols:
            intraday_data[col] = intraday_data['CumReturnResid'] + intraday_data[col] 
            
        return intraday_data
        
    def get_target(self, clip_MAD = False, normalize = False):
        
        data_3_30 = self.intraday_data[self.intraday_data.Time== dt.time(15, 30)].copy()
        eod_data = self.intraday_data[self.intraday_data.Time == dt.time(16, 0)].copy() 
        eod_data.rename(columns= {'CumReturnResid':'CumReturnResid_EOD'}, inplace = True)
        target_df = data_3_30.join(eod_data[['CumReturnResid_EOD']])
        
        #residual return from 4:00 to 3:30 tmr (t + 1 return)
        target_df['Resid_t_1'] = target_df.groupby('Id')['CumReturnResid'].shift(-1)
        target_df.eval('y = Resid_t_1 + CumReturnResid_EOD - CumReturnResid', inplace= True)
        
        #target_df.eval('y = (1 + Resid_t_1)* (1+ CumReturnResid_EOD) / (1 + CumReturnResid) -1', inplace= True)
        target_df.dropna(subset='y', inplace = True)
        
        if clip_MAD:
            
            MAD_by_date = target_df.groupby('Date').apply(lambda x: stats.median_abs_deviation(x['y']))
            MAD_by_date.name = 'MAD'
            target_df = target_df.join(MAD_by_date, on='Date')
            target_df['y'] = np.clip(target_df.y, -5 *target_df.MAD, 5 *target_df.MAD )
        
        if normalize:
            target_df = target_df.join(self.daily_data[['EST_VOL']])
            target_df['y'] = target_df['y'] / target_df['EST_VOL']
       
        return target_df        
    
if __name__ == '__main__':
    
    daily_data_path = r'data/daily_data'
    intraday_data_path = r'data/intraday_data'

    reader = DataReader()
    intraday_df = reader.read_intraday_data(intraday_data_path)
    daily_df = reader.read_daily_data(daily_data_path)
    intraday_df.dropna(subset= 'CumReturnResid', inplace=True)
    data_prep = DataPrep(intraday_df, daily_df)
    target_df = data_prep.get_target(clip_MAD=True, normalize= True)
