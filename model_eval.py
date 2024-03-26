import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from typing import Dict

def load_model(name):
    with open(os.path.join('models', f'{name}.pickle'), 'rb') as file:
        loaded_object = pickle.load(file)
    return loaded_object
    

class ModelEval:
    
    def __init__(self, regressor, feature_names,  x_train:np.ndarray, y_train:np.array, y_train_weight:np.array, x_val:np.ndarray, y_val_clipped:np.array, 
                 y_val_vol:np.array, y_val_weight:np.array, grid_search:GridSearchCV = None, best_params = None) -> None:
        #saving for reference
        self.grid_cv = grid_search 
        self.feature_names = feature_names
        
        assert (grid_search is None) ^ (best_params is None)
        self.regressor = regressor.set_params(**grid_search.best_params_ if grid_search else best_params)
        self.regressor.fit(x_train, y_train, sample_weight = y_train_weight)
        
        #val data
        self.x_val = x_val
        self.y_val = y_val_clipped
        self.y_val_vol = y_val_vol
        self.y_val_weight = y_val_weight

    def weighted_r2(self, x_val = None)->float:
        
        y_pred = self.regressor.predict(self.x_val if x_val is None else x_val)
        return r2_score(self.y_val, y_pred * self.y_val_vol, sample_weight=self.y_val_weight)
    
    def feature_importance(self)->Dict[str, float]:
        importance =  {k:v for k, v in zip(self.feature_names, self.regressor.feature_importances_)}
        return dict(sorted(importance.items(), key = lambda x: x[1], reverse=True))
    
    
    def feature_importance_MDA(self, n_repeats=30, random_state=7):

        rng = np.random.RandomState(random_state)
        
        # Baseline score with all features
        baseline_score = self.weighted_r2()
        
        # Store feature importances
        importances = np.zeros(self.x_val.shape[1])
        
        # Calculate importance for each feature
        for i in range(self.x_val.shape[1]):
            feature_scores = []
            for _ in range(n_repeats):
                X_permuted = self.x_val.copy()
                X_permuted[:, i] = rng.permutation(self.x_val[:, i])
                permuted_score = self.weighted_r2(X_permuted)
                feature_scores.append(baseline_score - permuted_score)
            
            importances[i] = np.mean(feature_scores)
        
        importance =  {k:v for k, v in zip(self.feature_names, importances)}
        return dict(sorted(importance.items(), key = lambda x: x[1], reverse=True))

    
    def to_pickle(self, name:str):
        
        filepath = os.path.join('models', f'{name}.pickle')
        if os.path.exists(filepath):
            print(f'The file {filepath} already exists')
        else:
            with open(filepath, 'wb') as file:
                pickle.dump(self, file)
            print(f'model save as {filepath}')
    
    def feature_bin_plot(self):
        
        # Bin plot for Square Footage vs Price
        sns.regplot(x='CumReturnResid', y='y_actual_clipped', x_bins=20, fit_reg=None)
        # axs[0].set_title('Binned Scatter Plot of Square Footage vs. Price')

        # # Bin plot for Age vs Price
        # sns.regplot(x='Rolling_Return_5d', y='y_actual_clipped', data=val_data, ax=axs[1], x_bins=20, fit_reg=None)
        # axs[1].set_title('Binned Scatter Plot of Age vs. Price')

        plt.tight_layout()
        plt.show()