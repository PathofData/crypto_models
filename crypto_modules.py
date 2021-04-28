from time import sleep
from typing import List
import ccxt
from ta import add_all_ta_features
import pandas as pd
import numpy as np
from joblib import load
from datetime import datetime
import pytz
from feature_generation import create_features, get_feature_names


def str2UTCms(date_time_str):
    """
    Get string in Format 'YYYY-MM-DD HH:MM:SS'
    and return the respected ms time in UTC
    """
    date_obj = datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')
    date_obj_utc = date_obj.replace(tzinfo=pytz.utc)
    return int(date_obj_utc.timestamp() * 1000)


def fetch_ohlc(coin_name:'str'='BTC/USDT',
               until_ms:int=None, 
               since_ms:int=None,
               t_frame:'str'='5m',
               limit:int=288) -> pd.DataFrame:

    exchange = ccxt.binance()
    t_frame = t_frame
    limit = limit
    symbol = coin_name
    
    date_interval_ms = until_ms - since_ms

    date_rate_ms = int(limit * pd.Timedelta(t_frame).total_seconds() * 1000)

    num_calls = np.ceil(date_interval_ms / 
                    date_rate_ms).astype(int)

    data = []
    total_calls = 0
    
    while since_ms < until_ms:
        if total_calls > num_calls:
            print(f'Maximum number of expected calls exceeded.')
            print(f'Expected number of calls: {num_calls}')
            print(f'Performed calls: {total_calls}')
        
        sleep(np.maximum(exchange.rateLimit / 1000, 2))
        
        try:
            total_calls += 1
            response = exchange.fetch_ohlcv(symbol, timeframe=t_frame, since=since_ms, limit=limit)

            if response:
                if len(response) < limit:
                    print(f'Warning call {total_calls} has packet loss:')
                    print(f'Expected {limit} packets but {len(response)} were received.')
                data.extend(response)
                since_ms += date_rate_ms
            else:
                print(f'Call {total_calls} resulted in empty response. Exiting.')
                break

        except ccxt.NetworkError as e:
            print(exchange.id, 'fetch_ohlcv failed due to a network error:', str(e))
            break
        
        except ccxt.ExchangeError as e:
            print(exchange.id, 'fetch_ohlcv failed due to exchange error:', str(e))
            break
        
        except Exception as e:
            print(exchange.id, 'fetch_ohlcv failed with:', str(e))
            break

    if data:
        header = ['date', 'Open', 'High', 'Low', 'Close', 'Volume']
        data = pd.DataFrame(data, columns=header)
        data['date'] = pd.to_datetime(data['date'], unit='ms')
        data.set_index('date', inplace=True, drop=True)
        data.sort_index(inplace=True)
        mean_value = data['Close'].mean()
    else:
        print(exchange.id, 'fetch_ohlcv returned no data')
        data = pd.DataFrame()
        mean_value = np.nan
    
    return data, mean_value
    

def preprocess_ohlcv_data(raw_data:pd.DataFrame,
                          scale_fn:'str',
                          data_freq:'str'='5min',
                          window_wd:int=288) -> pd.DataFrame:
    
    # np.seterr(invalid='raise')
    
    raw_data = (raw_data
                .resample(rule=data_freq)
                .asfreq()
                .interpolate(method='time', limit=None))

    raw_data = add_all_ta_features(raw_data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)

    cols_to_scale = load(scale_fn)

    for col in cols_to_scale:
        raw_data[col] = np.log(raw_data[col] + 1)
    
    X_data = raw_data.reset_index(drop=True)
    
    X_features, _, _ = create_features(data_raw=X_data,
                                       fs=1,
                                       segment_window=window_wd,
                                       partitioning=False,
                                       window_length=12*24,
                                       label_length=12*4,
                                       stride= 1,
                                       subsample_factor= 1,
                                       binary_delta_labels= True,
                                       binary_delta_value= 'Close')
        
    if np.where(np.isnan(X_features))[0].shape[0] > 0:
        print('Warning: Nan values encountered in features')

    feature_names = get_feature_names(raw_names=X_data.columns)

    X_features = pd.DataFrame(X_features, columns=feature_names)

    
    
    return X_features


def predict_classifier(preprocessed_data:pd.DataFrame, feature_names_path:str, model_path:str) -> List[float]:

    try:
        feature_list = load(feature_names_path)
    except Exception as e:
        print('Could not find the feature list:', str(e))

    try:
        model = load(model_path)
    except Exception as e:
        print('Could not find the model:', str(e))

    try:
        preprocessed_data = preprocessed_data[feature_list]
    except KeyError as e:
        print('Required features are not present in the preprocessed data:', str(e))

    try:
        if len(preprocessed_data.shape) == 1:
            preprocessed_data = preprocessed_data.reshape(1, -1)
        model_predictions = model.predict_proba(preprocessed_data)[:, 1]
    except Exception as e:
        print('Model prediction failed with: ', str(e))
        model_predictions = []

    return model_predictions
