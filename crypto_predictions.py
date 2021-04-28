import os
import argparse
import numpy as np
import pandas as pd
from crypto_modules import fetch_ohlc, preprocess_ohlcv_data, predict_classifier

SCALE_PATH = 'saved_models/column_scale_v1.joblib'
FEATURES_PATH = 'saved_models/feature_list_v9.joblib'
MODEL_PATH = 'saved_models/classification_model_BTC_v9.joblib'
PREDICTIONS_FN = 'saved_models/saved_predictions.csv'
RAW_DATA_FN = 'saved_models/BTC_raw.csv'
#
BASE_DIR = os.getenv('BASE_DIR', '/usr/local/airflow/dags')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Parser for crypto predictions. Arguments coming soon')

    ts = pd.Timestamp.today(tz='UTC').floor('5min').strftime("%Y-%m-%d %H:%M:%S")

    data, current_mean = fetch_ohlc()

    last_ts = data.index.max()
    archive_ts = last_ts - pd.Timedelta('4h')
    archive_data = data[data.index >= archive_ts]

    if not os.path.isfile(os.path.join(BASE_DIR, RAW_DATA_FN)):
        archive_data.to_csv(os.path.join(BASE_DIR, RAW_DATA_FN))
    else:
        archive_data.to_csv(os.path.join(
            BASE_DIR, RAW_DATA_FN), mode='a', header=False)

    data = preprocess_ohlcv_data(
        raw_data=data, scale_fn=os.path.join(BASE_DIR, SCALE_PATH))
    prediction = predict_classifier(preprocessed_data=data,
                                    feature_names_path=os.path.join(
                                        BASE_DIR, FEATURES_PATH),
                                    model_path=os.path.join(BASE_DIR, MODEL_PATH))

    prediction_df = pd.DataFrame({
        'time': [pd.Timestamp.today(tz='UTC').floor('5min')],
        'current_mean': [current_mean],
        'prediction': [prediction[0]]
    })
    prediction_df.set_index('time', inplace=True, drop=True)

    # Create write path if not exists
    os.makedirs(os.path.basename(os.path.join(BASE_DIR, PREDICTIONS_FN)), exist_ok=True)

    if not os.path.isfile(os.path.join(BASE_DIR, PREDICTIONS_FN)):
        prediction_df.to_csv(os.path.join(BASE_DIR, PREDICTIONS_FN))
    else:
        prediction_df.to_csv(os.path.join(
            BASE_DIR, PREDICTIONS_FN), mode='a', header=False)

    print(f'{ts}: Model Prediction: {prediction[0]}.')
