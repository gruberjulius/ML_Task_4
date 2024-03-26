import os
import pickle
import argparse
import pandas as pd
import datetime as dt
from data_processor import DataReader, DataPrep
from sklearn.metrics import r2_score

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' was created.")
    else:
        raise FileExistsError(f"Folder '{folder_path}' already exists.")
    

def main():
    parser = argparse.ArgumentParser(description='Process some dates.')
    parser.add_argument('-i', required=True, help='Input directory')
    parser.add_argument('-o', required=True, help='Output directory')
    parser.add_argument('-p', help='Directory containing the model (for Mode 2)')
    parser.add_argument('-s', required=True, help='Start date in YYYYMMDD format')
    parser.add_argument('-e', required=True, help='End date in YYYYMMDD format')
    parser.add_argument('-m', required=True, type=int, choices=[1, 2], help='Mode to run')

    args = parser.parse_args()

    start_date = dt.datetime.strptime(args.s, '%Y%m%d')
    end_date = dt.datetime.strptime(args.e, '%Y%m%d')
    feature_cols = ['Rolling_Return_5d_clipped', 'Rolling_Return_10d_clipped', 'CumReturnResid','AD_preday', 'IntradayRSI', 'NYSE']

    if args.m == 1:
        #create features using data froms start to end dates from the input directory
        daily_df = DataReader.read_daily_data(os.path.join(args.i, 'daily_data'), end_date = end_date)
        intraday_df = DataReader.read_intraday_data(os.path.join(args.i, 'intraday_data'), end_date = end_date)

        #create and output features
        data_prep = DataPrep(intraday_df, daily_df, start_date = start_date, features=feature_cols)
        create_folder(args.o)
        features = data_prep.get_features(save_to=args.o)     
        targets = data_prep.get_target(clip_MAD=True, normalize=False, save_to=args.o)
           
    elif args.m == 2:
        if not args.p:
            raise ValueError('Model path must be provided for mode 2')
        
        with open(os.path.join(args.p, os.listdir(args.p)[0]), 'rb') as file:
            loaded_model = pickle.load(file)

        #read features data from input dir
        parser = lambda f, beg: dt.datetime.strptime(f[beg:-4], DataReader.DATEFORMAT)
        features = pd.concat([pd.read_csv(os.path.join(args.i, f), parse_dates=['Date']) for f in os.listdir(args.i) if f[0] == 'f' and (start_date <= parser(f, 9) <= end_date)])
        features.set_index(['Date', 'Time','Id'], inplace=True)
        features.sort_index(level = 'Date', inplace=True)
        
        #predict 
        features['Pred'] = loaded_model.predict(features[feature_cols])
        
        #read target data from input dir
        targets = pd.concat([pd.read_csv(os.path.join(args.i, f), parse_dates=['Date']) for f in os.listdir(args.i) if f[0] == 't' and (start_date <= parser(f, 8) <= end_date)])
        targets.set_index(['Date', 'Time', 'Id'], inplace= True)
        targets = targets.join(features['Pred'])
        
        #and scale back to return space
        targets.Pred = targets.Pred * targets.EST_VOL_preday
        #targets.Pred.fillna(0, inplace=True)
        create_folder(args.o)
        print('exporting predictions...')
        targets.Pred.to_csv(f'{args.o}/predictions.csv')
        targets.to_csv(f'{args.o}/targets.csv')
        print(targets.isna().describe())
        targets.dropna(inplace=True)
        print(f'r2_score on the test set is: {r2_score(targets.y, targets.Pred, sample_weight=targets.MDV_63_sqrt):8f}')    
        
    print('Done')

if __name__ == '__main__':
    main()
