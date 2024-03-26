import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

def train_val_split(input_data, train_size):
    years = list(input_data.index.get_level_values('Date').year.drop_duplicates())
    split_idx = int(np.ceil(len(years) * train_size))
    validation_years = years[split_idx:]
    train_data = input_data[input_data.index.get_level_values('Date').year < validation_years[0]] #2010 - 2013
    val_data = input_data[input_data.index.get_level_values('Date').year >= validation_years[0]] # 2014
    
    #remove the first day to avoid overlap with train data 
    val_dates = val_data.index.get_level_values('Date')
    val_data = val_data[val_dates > val_dates[0]]
    
    return train_data, val_data

class ExpandingWindowCV:
    
    def __init__(self, regressor, param_grid) -> None:
        
        self.grid_search = GridSearchCV(regressor, param_grid, verbose=1)  
        
    
    def fit(self, x_train:pd.DataFrame, y_train:pd.DataFrame, sample_weight, overlap_period = 1):
        
        custom_cv_index = self._cv_index(x_train, overlap_period)
        self.grid_search.cv = custom_cv_index
        self.grid_search.fit(x_train.to_numpy(), y_train.to_numpy(), sample_weight= sample_weight)
    
    def _cv_index(self, train_data, overlap_period):
        
        cv = []
        date_index = train_data.index.get_level_values('Date')
        date_index_unique = date_index.drop_duplicates()
        for year in date_index.year.unique()[:-1]:
            last_train_idx = date_index.get_loc(date_index[date_index.year <= year][-1]).stop
            first_test_date = date_index_unique[date_index_unique.get_loc(date_index[last_train_idx]) + overlap_period]
            first_test_idx = date_index.get_loc(first_test_date).start
            last_test_idx = date_index.get_loc(date_index[date_index.year <= year + 1][-1]).stop
            cv.append((np.arange(last_train_idx), np.arange(first_test_idx, last_test_idx)))
            
        return cv 
                