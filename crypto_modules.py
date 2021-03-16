from time import sleep
from typing import List
import ccxt
import pandas as pd
import numpy as np
from joblib import load

from feature_generation import create_features, get_feature_names


def fetch_ohlc(coin_name:'str'='BTC/USDT',
               date_until:'str'='today', 
               date_since:'str'=None,
               t_frame:'str'='5m',
               limit:int=288) -> pd.DataFrame:

    exchange = ccxt.binance()
    t_frame = t_frame
    limit = limit
    until = date_until
    symbol = coin_name

    if not date_since:
        since = since = (pd.to_datetime('today') - pd.Timedelta('24h')).strftime("%Y-%m-%d %H:%M:%S.%f")
    else:
        since = date_since

    date_interval_ms = int((pd.to_datetime(until) -
                            pd.to_datetime(since)).total_seconds() * 1000)

    date_rate_ms = int(limit * pd.Timedelta(t_frame).total_seconds() * 1000)

    num_calls = np.ceil(date_interval_ms / 
                    date_rate_ms).astype(int)

    data = []
    total_calls = 0
    since_ms = int(pd.to_datetime(since).timestamp() * 1000)
    until_ms = int(pd.to_datetime(until).timestamp() * 1000)

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
                          data_freq:'str'='5min',
                          window_wd:int=288) -> pd.DataFrame:
    
    np.seterr(invalid='raise')
    
    raw_data = (raw_data
                .resample(rule=data_freq)
                .asfreq()
                .interpolate(method='time', limit=None))

    raw_data['TR'] = np.maximum((raw_data['High'] - raw_data['Low']).abs(), (raw_data['High'] - raw_data['Close'].shift(1).abs()))
    raw_data['TR'] = np.maximum(raw_data['TR'], (raw_data['Low'] - raw_data['Close'].shift(1).abs()))
    raw_data['ATR'] = raw_data['TR'].rolling(6, min_periods=1).mean()
    raw_data['DM'] = (raw_data['High'] - raw_data['Low']) / raw_data['TR'] - ( (raw_data['High'].shift(1) - raw_data['Low'].shift(1)) / raw_data['TR'].shift(1) )

    raw_data['DI_plus'] = raw_data['High'] - raw_data['High'].shift(1)
    raw_data['DI_minus'] = raw_data['Low'] - raw_data['Low'].shift(1)

    raw_data['DI_plus_ind'] = raw_data[(raw_data['DI_plus'] > 0)
                        & (raw_data['DI_plus'].abs() > raw_data['DI_minus'].abs())
                        ]['DI_plus']
    raw_data['DI_plus_ind'] = raw_data['DI_plus_ind'].fillna(0)

    raw_data['DI_minus_ind'] = -raw_data[(raw_data['DI_minus'] < 0)
                            & (raw_data['DI_plus'].abs() < raw_data['DI_minus'].abs())
                        ]['DI_minus']
    raw_data['DI_minus_ind'] = raw_data['DI_minus_ind'].fillna(0)
    raw_data['DI_plus_ind'] = raw_data['DI_plus_ind'].rolling(6, min_periods=1).sum() / raw_data['TR'].rolling(6, min_periods=1).sum()
    raw_data['DI_minus_ind'] = raw_data['DI_minus_ind'].rolling(6, min_periods=1).sum() / raw_data['TR'].rolling(6, min_periods=1).sum()
    raw_data['DI'] = raw_data['DI_plus_ind'] - raw_data['DI_minus_ind']

    raw_data['MOMENTUM'] = raw_data['Close'] - raw_data['Close'].shift(2)

    raw_data['SUM_up_close'] = raw_data['Close'] - raw_data['Close'].shift(1)
    raw_data['SUM_up_close'] = raw_data[raw_data['SUM_up_close'] > 0]['SUM_up_close']
    raw_data['SUM_up_close'] = raw_data['SUM_up_close'].fillna(0).rolling(6, min_periods=1).mean()
    raw_data['SUM_down_close'] = raw_data['Close'] - raw_data['Close'].shift(1)
    raw_data['SUM_down_close'] = -raw_data[raw_data['SUM_down_close'] < 0]['SUM_down_close']
    raw_data['SUM_down_close'] = raw_data['SUM_down_close'].fillna(0).rolling(6, min_periods=1).mean()

    raw_data['RSI'] = 100 - 100/(1 + (raw_data['SUM_up_close'] / raw_data['SUM_down_close']))

    raw_data['X_m'] = (raw_data['High'] + raw_data['Low'] + raw_data['Close']) / 3
    raw_data['Beta_1'] = 2 * raw_data['X_m'] - raw_data['High']
    raw_data['Sigma_1'] = 2 * raw_data['X_m'] - raw_data['Low']
    raw_data['HBOP'] = 2 * raw_data['X_m'] - 2 * raw_data['Low'] + raw_data['High']
    raw_data['LBOP'] = 2 * raw_data['X_m'] - 2 * raw_data['High'] + raw_data['Low']

    raw_data['SI'] = raw_data['Close'] - raw_data['Close'].shift(1) + 0.5 * ( raw_data['Close'] - raw_data['Open'] ) + 0.25 * ( raw_data['Close'].shift(1) - raw_data['Open'].shift(1))
    raw_data['K'] = np.maximum((raw_data['High'] - raw_data['Close'].shift(1)).abs(), (raw_data['Low'] - raw_data['Close'].shift(1).abs()))
    raw_data['R1'] = (raw_data['High'] - raw_data['Close'].shift(1)).abs() \
                - 0.5 * (raw_data['Low'] - raw_data['Close'].shift(1)).abs() \
                + 0.25 * (raw_data['Close'].shift(1) - raw_data['Open'].shift(1)).abs()
    raw_data['R2'] = (raw_data['Low'] - raw_data['Close'].shift(1)).abs() \
                - 0.5 * (raw_data['High'] - raw_data['Close'].shift(1)).abs() \
                + 0.25 * (raw_data['Close'].shift(1) - raw_data['Open'].shift(1)).abs()
    raw_data['R3'] = (raw_data['High'] - raw_data['Low']).abs() \
                + 0.25 * (raw_data['Close'].shift(1) - raw_data['Open'].shift(1)).abs()
    raw_data['R_ind_1'] = (raw_data['High'] - raw_data['Close'].shift(1)).abs()
    raw_data['R_ind_2'] = (raw_data['Low'] - raw_data['Close'].shift(1)).abs()
    raw_data['R_ind_3'] = (raw_data['High'] - raw_data['Low']).abs()

    def match_row(frame):
        idx = frame.name

        if frame.idxmax() == 'R_ind_1':
            return raw_data.loc[idx, 'R1']
        elif frame.idxmax() == 'R_ind_2':
            return raw_data.loc[idx, 'R2']
        elif frame.idxmax() == 'R_ind_3':
            return raw_data.loc[idx, 'R3']
        
    raw_data['R'] = raw_data[['R_ind_1', 'R_ind_2', 'R_ind_3']].apply(match_row, axis=1)

    raw_data['SI'] = (50 * raw_data['SI'] / raw_data['R'] * raw_data['K']).round()

    raw_data.drop(['DI_plus', 'DI_minus', 'DI_plus_ind', 'DI_minus_ind', 'SUM_up_close',
     'SUM_down_close', 'K', 'R1', 'R2', 'R3', 'R_ind_1', 'R_ind_2', 'R_ind_3', 'R'], axis=1, inplace=True)
    raw_data.fillna(0, inplace=True)

    cols_to_scale = ['Open', 'High', 'Low', 'Close', 'Volume', 'X_m', 'Beta_1', 'Sigma_1', 'HBOP', 'LBOP']
    for col in cols_to_scale:
        raw_data[col] = np.log(raw_data[col] + 1)
    
    X_data = raw_data.reset_index(drop=True)

    try:
        X_features, _ = create_features(data_raw=X_data,
                                        fs=1,
                                        segment_window=window_wd,
                                        partitioning=False,
                                        window_length=12*24,
                                        label_length=12*3,
                                        stride= 1,
                                        subsample_factor= 1,
                                        binary_delta_labels= True,
                                        binary_delta_value= 'Close')
        
        if np.where(np.isnan(X_features))[0].shape[0] > 0:
            print('Warning: Nan values encountered in features')

        feature_names = get_feature_names(raw_names=X_data.columns)

        X_features = pd.DataFrame(X_features, columns=feature_names)

    except Exception as e:
        print('Feature creation failed with:', str(e))
        print(str(e.__class__.__name__))
        print(str(e.__context__))
        X_features = pd.DataFrame()
    
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
