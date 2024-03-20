import os
import pandas as pd
import datetime as dt

class DataReader:
    
    def read_intraday_data(self, intraday_data_path):
        
        intraday_df = pd.concat([pd.read_csv(os.path.join(intraday_data_path, f)) for f in os.listdir(intraday_data_path)])
        intraday_df.Date = pd.to_datetime(intraday_df.Date, format='%Y%m%d')
        intraday_df.Time = pd.to_datetime(intraday_df.Time, format='%H:%M:%S.%f').dt.time 
        
        #intraday_df['Timestamp'] = pd.to_datetime(intraday_df.Date, format='%Y%m%d') +  pd.to_timedelta(intraday_df.Time)
        #intraday_df.drop(columns= ['Date', 'Time'], inplace= True)
        #intraday_df = intraday_df.reindex(columns= [intraday_df.columns[-1]] + list(intraday_df.columns[:-1]))
        #intraday_df.dropna(subset = ['CumReturnResid'], inplace= True )
        return intraday_df
    
    def read_daily_data(self, daily_data_path):
        
        daily_df = pd.concat([pd.read_csv(os.path.join(daily_data_path, f)) for f in os.listdir(daily_data_path)])
        daily_df.Date = pd.to_datetime(daily_df.Date, format='%Y%m%d')
        daily_df.rename(columns= {'ID':'Id'}, inplace= True)

        return daily_df
        
class DataPrep:
    
    def __init__(self, intraday_data, daily_data) -> None:
        self.intraday_data = intraday_data
        self.daily_data = daily_data
        
    def get_target(self, clip = True):
        
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
        
        return target_df        