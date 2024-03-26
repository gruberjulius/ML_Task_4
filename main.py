import os
import pickle
import argparse
import pandas as pd
import datetime as dt
from data_processor import DataReader, DataPrep

# Dummy function to make predictions. Replace with actual prediction logic.
def make_predictions(features, model_path):
    # Load the model from the specified path
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    # Replace with actual prediction logic
    predictions = model.predict(features)
    return predictions

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' was created.")
    else:
        print(f"Folder '{folder_path}' already exists.")

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

    if args.m == 1:
        #create features using data froms start to end dates from the input directory
        daily_df = DataReader.read_daily_data(os.path.join(args.i, 'daily_data'), start_date, end_date)
        intraday_df = DataReader.read_intraday_data(os.path.join(args.i, 'intraday_data'), start_date, end_date)

        #create and output features
        data_prep = DataPrep(intraday_df, daily_df)
        create_folder(args.o)
        features = data_prep.get_features(save_to=args.o)     
           
    elif args.m == 2:
        if not args.p:
            raise ValueError('Model path must be provided for mode 2')
        features = pd.read_csv(f'{args.i}/features.csv')
        predictions = make_predictions(features, args.p)
        predictions_df = pd.DataFrame(predictions, columns=['Date', 'Time', 'Id', 'Pred'])
        predictions_df.to_csv(f'{args.o}/predictions.csv', index=False)

if __name__ == '__main__':
    main()
