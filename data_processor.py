import os
import numpy as np
import pandas as pd
import datetime as dt
import pandas_market_calendars as mcal
from scipy import stats
from typing import Callable, Dict,List

class DataReader:
    
    DATEFORMAT = '%Y%m%d'
    
    @staticmethod
    def read_intraday_data(intraday_data_path:str, end_date = dt.datetime.max)->pd.DataFrame:
        
        #if start_date != dt.datetime.min: start_date += dt.timedelta(days = -30) #provide some buffer for lookback features  
        print(f'Reading intraday data up to {end_date.strftime(DataReader.DATEFORMAT)}')
        parser = lambda f: dt.datetime.strptime(f[:-4], DataReader.DATEFORMAT)
        intraday_df = pd.concat([pd.read_csv(os.path.join(intraday_data_path, f)) for f in os.listdir(intraday_data_path) if parser(f) <= end_date])
        intraday_df.Date = pd.to_datetime(intraday_df.Date, format= DataReader.DATEFORMAT)
        intraday_df.Time = pd.to_datetime(intraday_df.Time, format='%H:%M:%S.%f').dt.time 
        intraday_df.set_index(['Date', 'Id'], inplace=True)
        intraday_df.dropna(subset= 'CumReturnResid', inplace=True)
        
        return intraday_df
    
    @staticmethod
    def read_daily_data(daily_data_path:str,  end_date = dt.datetime.max)->pd.DataFrame:

        print(f'Reading daily data up to {end_date.strftime(DataReader.DATEFORMAT)}')
        parser = lambda f: dt.datetime.strptime(f[4:-4], DataReader.DATEFORMAT)
        daily_df = pd.concat([pd.read_csv(os.path.join(daily_data_path, f)) for f in os.listdir(daily_data_path) if parser(f) <= end_date])
        daily_df.Date = pd.to_datetime(daily_df.Date, format=DataReader.DATEFORMAT)
        daily_df.rename(columns= {'ID':'Id'}, inplace= True)
        daily_df.set_index(['Date', 'Id'], inplace=True)

        return daily_df
        
class DataPrep:
    
    def __init__(self, intraday_data:pd.DataFrame, daily_data:pd.DataFrame, start_date = dt.datetime.min, features:List[str] = None)->pd.DataFrame:
        self.intraday_data = intraday_data.copy()
        self.daily_data = daily_data.copy()
        
        #data available asof t-1 
        cols_same_day = ['Open', 'PxAdjFactor', 'SharesAdjFactor', 'SYMBOL', 'MIC']
        shift_cols = [col for col in self.daily_data.columns if col not in cols_same_day]
        self.daily_data[shift_cols]= self.daily_data.groupby('Id')[shift_cols].shift()
        self.daily_data.rename(columns = {col: f'{col}_preday' for col in shift_cols}, inplace = True)
        #convert annualized vol to daily vol
        self.daily_data.eval('EST_VOL_preday = EST_VOL_preday / sqrt(252)', inplace= True)
        
        #ffill nan values
        self.daily_data[['MDV_63_preday', 'EST_VOL_preday']] = self.daily_data[['MDV_63_preday', 'EST_VOL_preday']].groupby('Id').ffill()
        
        self.start_date = start_date
        self.features = features

        
    def get_features(self, rolling_periods = range(1, 21), indicators:Dict[str, Callable] = {}, save_to:str = None)->pd.DataFrame:
        
        self.daily_data['MDV_63_sqrt'] = np.sqrt(self.daily_data.MDV_63_preday)
        
        #corp actions
        self.daily_data['PxAdjFactorRatio'] = self.daily_data.groupby('Id', as_index=False).apply(lambda x: x.PxAdjFactor.div(x.PxAdjFactor.shift())).droplevel(0)
        self.daily_data['SharesAdjFactorRatio'] = self.daily_data.groupby('Id', as_index=False).apply(lambda x: x.SharesAdjFactor.div(x.SharesAdjFactor.shift())).droplevel(0)    
        self.daily_data.PxAdjFactorRatio.fillna(1, inplace= True)
        self.daily_data.SharesAdjFactorRatio.fillna(1, inplace= True)
        self.daily_data.eval('Stock_Split = SharesAdjFactorRatio != 1', inplace= True)
        self.daily_data.eval('Dividend =  PxAdjFactorRatio != 1/SharesAdjFactorRatio', inplace= True)
        
        #trading calendar
        first_date = self.daily_data.index.get_level_values('Date')[0]
        last_date = self.daily_data.index.get_level_values('Date')[-1]
        schedule = self.get_schedule('NYSE',first_date, last_date )
        self.daily_data = self.daily_data.join(schedule[['EarlyClose', 'NextHoliday']], on = 'Date')
        
        #rolling returns asof t-1
        eod_data = self.intraday_data[self.intraday_data.Time == dt.time(16, 0)][['CumReturnResid']].copy() 
        for period in rolling_periods:
            eod_data[f'Rolling_Return_{period}d'] = eod_data.groupby('Id')['CumReturnResid'].rolling(window = period, min_periods= 1).sum().droplevel(0)
            
        rolling_return_cols =  [f'Rolling_Return_{i}d' for i in rolling_periods]
        rolling_prev_returns = eod_data[rolling_return_cols].groupby('Id').shift(1)
        
        #trading indicators
        for name, func in indicators.items():
            self.daily_data = self.calc_indicator(self.daily_data, func, name)

        #intraday RSI
        last_time = dt.time(15, 30)
        #filter out data after 15:30 and compute the raw resid return
        returns = self.intraday_data.query('Time <= @last_time').groupby(['Date', 'Id'])['CumReturnResid'].diff()
        gains = np.maximum(returns, 0)
        losses = -np.minimum(returns, 0)
        avg_gains = gains.groupby(['Date', 'Id']).mean()
        avg_losses = losses.groupby(['Date', 'Id']).mean()
        daily_rsi = 100 - 100 / (1 + avg_gains / avg_losses)
        daily_rsi.name = 'IntradayRSI'

        #merge features from daily data
        intraday_data = self.intraday_data.query('Time == @last_time')
        cols_to_merge = ['MDV_63_sqrt','MIC', 'Stock_Split', 'Dividend', 'Volume_preday', 'Open', 'Close_preday', 'EST_VOL_preday', 'PxAdjFactor', 'SharesAdjFactor', 'EarlyClose', 'NextHoliday'] +  list(indicators.keys())
        intraday_data = intraday_data.join(self.daily_data[cols_to_merge])
        
        #volume feature
        intraday_data['CumVolume'] = intraday_data.CumVolume.where(intraday_data.CumVolume > 0, np.nan)
        intraday_data['CumVolume'] = intraday_data.groupby('Id')['CumVolume'].ffill()
        intraday_data['logCumVolume_Adj'] = np.log(1 + intraday_data.CumVolume * intraday_data.SharesAdjFactor) #adjust for corp action
        intraday_data['VolumeChange'] = intraday_data.groupby('Id')['logCumVolume_Adj'].diff()
        
        #normalize cross-sectionally
        intraday_data['VolumeChangeNormalize'] = self.cross_section_normalization(intraday_data['VolumeChange'], normalize=True, clip_std= 4)
        
        #join rolling returns until t- 1
        intraday_data = intraday_data.join(rolling_prev_returns)
        
        #standardize and clip all return data
        for col in rolling_return_cols + ['CumReturnResid']:
            intraday_data[f'{col}_clipped'] = self.clip_by_MAD(intraday_data[col], est_vol = intraday_data['EST_VOL_preday'])
        
        intraday_data = intraday_data.join(daily_rsi)

        #exchange info
        intraday_data['NYSE'] = intraday_data.MIC.isin(['XNYS', 'XASE'])
        intraday_data.NYSE = intraday_data.NYSE.astype('float')
        
        intraday_data = intraday_data[intraday_data.index.get_level_values(0) >= self.start_date]
        intraday_data.dropna(subset= self.features, inplace=True, how = 'any')    
        if self.features is not None:
            intraday_data = intraday_data[self.features]
            
        #export features to csv - create one csv file per date
        if save_to:
            for date in intraday_data.index.get_level_values('Date'):
                intraday_data.loc[date:date][self.features].to_csv(os.path.join(save_to, f'features.{date.strftime(DataReader.DATEFORMAT)}.csv'))
                
        return intraday_data
    
    @staticmethod
    def cross_section_normalization(series:pd.Series, normalize:bool,  clip_std:int = None)->pd.Series:
        
        series_avg = series.groupby('Date').mean()
        series_std = series.groupby('Date').std()
        
        if clip_std:
            series = np.clip(series, series_avg - clip_std * series_std, series_avg + clip_std * series_std) 
        
        if normalize:
            series = (series - series_avg) / series_std

        return series
        
    
    @staticmethod
    def calc_indicator(daily_df:pd.DataFrame, indicator_func:Callable, name:str)->pd.DataFrame:
        
        dfs = []
        for _, df in daily_df.groupby('Id'):
            dfs.append(indicator_func(df.Close))
            
        indicators = pd.concat(dfs)
        indicators.name = name
        daily_df = daily_df.join(indicators)

        return daily_df
        
    def get_target(self, clip_MAD = False, normalize = False)->pd.DataFrame:
        
        data_3_30 = self.intraday_data[self.intraday_data.Time== dt.time(15, 30)].copy()
        eod_data = self.intraday_data[self.intraday_data.Time == dt.time(16, 0)].copy() 
        eod_data.rename(columns= {'CumReturnResid':'CumReturnResid_EOD'}, inplace = True)
        target_df = data_3_30.join(eod_data[['CumReturnResid_EOD']])
        
        #residual return from 4:00 to 3:30 tmr (t + 1 return)
        target_df['Resid_t_1'] = target_df.groupby('Id')['CumReturnResid'].shift(-1)
        target_df.dropna(subset = 'Resid_t_1', inplace = True) #no return data available on t+1
        target_df.eval('y_actual = CumReturnResid_EOD - CumReturnResid + Resid_t_1 ', inplace= True)
        
        #compute y_actual_clipped to calc model R^2
        target_df['y_actual_clipped'] =  self.clip_by_MAD(target_df['y_actual'])
        
        #y is the transformed y_actual - target var for fitting
        target_df['y'] = target_df['y_actual'].copy()
        if normalize:
            target_df = target_df.join(self.daily_data[['EST_VOL_preday']])
            target_df.dropna(subset = 'EST_VOL_preday', inplace = True)
            target_df['y'] = target_df['y'] / target_df['EST_VOL_preday']
            
        if clip_MAD:
            #clip the normalized y for training model
            target_df['y'] =  self.clip_by_MAD(target_df['y'])
        
        target_df = target_df[target_df.index.get_level_values(0) >= self.start_date]
        return target_df        

    
    @staticmethod
    def clip_by_MAD(series:pd.Series, est_vol:pd.Series = None, clip_n = 5)->pd.Series:

        #standardize by est_vol
        if est_vol is not None:
            series = series / est_vol
        
        median_by_date= series.groupby('Date').median()
        MAD_by_date = np.abs(series - median_by_date).groupby('Date').median()
        return np.clip(series, median_by_date - clip_n *MAD_by_date , median_by_date + clip_n *MAD_by_date )
    
    @staticmethod
    def get_schedule(exchange:str, start_date:dt.datetime, end_date:dt.datetime)->pd.DataFrame:
        
        nyse = mcal.get_calendar(exchange)
        trading_schedule = nyse.schedule(start_date= start_date, end_date=end_date + pd.Timedelta(days = 10))
        trading_schedule['market_open'] = trading_schedule['market_open'].dt.tz_convert('America/New_York')
        trading_schedule['market_close'] = trading_schedule['market_close'].dt.tz_convert('America/New_York')
        trading_schedule['EarlyClose'] = trading_schedule['market_close'].dt.hour < 16
        trading_schedule.reset_index(inplace= True)
        trading_schedule.rename(columns= {'index':'Date'}, inplace=True)
        trading_schedule['NextDate'] = trading_schedule.Date.shift(-1)
        trading_schedule.dropna(inplace=True)
        trading_schedule['DaysToNext'] = (trading_schedule.NextDate - trading_schedule.Date).dt.days
        trading_schedule.eval('NextHoliday =DaysToNext > 1', inplace=True)
        
        trading_schedule.set_index('Date', inplace= True)
        
        return trading_schedule
    
if __name__ == '__main__':
    
    start_date = dt.datetime(2015, 1, 1)
    daily_data_path = r'oos_data/daily_data'
    intraday_data_path = r'oos_data/intraday_data'
    intraday_df = DataReader.read_intraday_data(intraday_data_path)
    daily_df = DataReader.read_daily_data(daily_data_path)
    feature_cols = ['Rolling_Return_5d_clipped', 'Rolling_Return_10d_clipped', 'CumReturnResid', 'IntradayRSI', 'NYSE']
    data_prep = DataPrep(intraday_df, daily_df, start_date, features= feature_cols + ['EST_VOL_preday'])
    X_df = data_prep.get_features()
    
    