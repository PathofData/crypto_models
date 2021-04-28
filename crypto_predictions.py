import os
import argparse
import numpy as np
import pandas as pd
from crypto_modules import fetch_ohlc, preprocess_ohlcv_data, predict_classifier, str2UTCms
import ccxt
from datetime import datetime, timedelta

SCALE_PATH = 'saved_models/column_scale_v1.joblib'
FEATURES_PATH = 'saved_models/feature_list_v9.joblib'
MODEL_PATH = 'saved_models/classification_model_BTC_v9.joblib'

PREDICTIONS_FN = 'saved_predictions.csv'
RAW_DATA_FN = 'BTC_raw.csv'
#
BASE_DIR = os.getenv('BASE_DIR', '/usr/local/airflow/dags')


def main():

    ts_now = datetime.utcnow()

    parser = argparse.ArgumentParser(
        description='Parser for crypto predictions. Arguments coming soon')

    parser.add_argument(
        "--output",
        "-o",
        help="Output path for the predictions logging",
        default=".",
        type=str
    )

    parser.add_argument(
        "--pair",
        "-p",
        help="Trading pair to predict",
        default="BTC/USDT",
        type=str
    )

    parser.add_argument(
        "--start_date",
        "-s",
        help="Days before the end date",
        default=1,
        type=int
    )

    parser.add_argument(
        "--end_date",
        "-e",
        help="End date to fetch data format (YYYY-MM-DD-HH-MM-SS)",
        type=str
    )

    args = parser.parse_args()

    # Create output dirs
    os.makedirs(os.path.abspath(args.output), exist_ok=True)
    # Set Output Dir
    OUTPUT_DIR = os.path.abspath(args.output)

    # Check pair validity
    pairs = [pair.lower() for pair in ccxt.binance().fetch_tickers()]
    #
    if args.pair.lower() not in pairs:
        print(f"Cant find pair {args.pair} at the exchange list.")
        return

    # Get end date
    if not args.end_date:
        until_ms = int(ts_now.timestamp() * 1000)
    else:
        until_ms = str2UTCms(args.end_date)

    # Get Start date
    # 86400000 No of ms per day
    since_ms = until_ms - 86400000*args.start_date

    # Fetch pair Data
    data, current_mean = fetch_ohlc(args.pair,until_ms,since_ms)
    
    #TODO Floor (5 min intervals) dataset here?

    last_ts = data.index.max()
    archive_ts = last_ts - pd.Timedelta('4h')
    archive_data = data[data.index >= archive_ts]

    if not os.path.isfile(os.path.join(OUTPUT_DIR, RAW_DATA_FN)):
        archive_data.to_csv(os.path.join(OUTPUT_DIR, RAW_DATA_FN))
    else:
        archive_data.to_csv(os.path.join(
            OUTPUT_DIR, RAW_DATA_FN), mode='a', header=False)

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

    if not os.path.isfile(os.path.join(OUTPUT_DIR, PREDICTIONS_FN)):
        prediction_df.to_csv(os.path.join(OUTPUT_DIR, PREDICTIONS_FN))
    else:
        prediction_df.to_csv(os.path.join(
            OUTPUT_DIR, PREDICTIONS_FN), mode='a', header=False)

    print(f'{ts_now}: Model Prediction: {prediction[0]}.')


if __name__ == '__main__':
    main()
