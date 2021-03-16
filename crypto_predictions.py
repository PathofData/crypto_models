import os
import argparse
import numpy as np
import pandas as pd
from crypto_modules import fetch_ohlc, preprocess_ohlcv_data, predict_classifier


FEATURES_PATH = 'saved_models/feature_list_v3.joblib'
MODEL_PATH = 'saved_models/classification_model_BTC_v3.joblib'
PREDICTIONS_FN = 'saved_models/saved_predictions.csv'
RAW_DATA_FN = 'saved_models/BTC_raw.csv'


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Parser for crypto predictions. Arguments coming soon')

    ts = pd.to_datetime('today').strftime("%Y-%m-%d %H:%M:%S")

    data, current_mean = fetch_ohlc()
    
    last_ts = data.index.max()
    archive_ts = last_ts - pd.Timedelta('3h')
    archive_data = data[data.index >= archive_ts]
    
    if not os.path.isfile(RAW_DATA_FN):
        archive_data.to_csv(RAW_DATA_FN)
    else:
        archive_data.to_csv(RAW_DATA_FN, mode='a', header=False)

    data = preprocess_ohlcv_data(raw_data=data)
    prediction = predict_classifier(preprocessed_data=data,
                                    feature_names_path=FEATURES_PATH,
                                    model_path=MODEL_PATH)

    prediction_df = pd.DataFrame({
        'time': [pd.to_datetime('today')],
        'current_mean': [current_mean],
        'prediction': [prediction[0]]
    })
    prediction_df.set_index('time', inplace=True, drop=True)

    if not os.path.isfile(PREDICTIONS_FN):
        prediction_df.to_csv(PREDICTIONS_FN)
    else:
        prediction_df.to_csv(PREDICTIONS_FN, mode='a', header=False)

    print(f'{ts}: Model Prediction: {prediction[0]}.')
