from decimal import ROUND_HALF_UP, Decimal
import math
import traceback
from typing import List, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import norm, inv
from scipy import stats
from scipy.fftpack import fft
from scipy.signal import butter, filtfilt
from statsmodels.distributions.empirical_distribution import ECDF
from statsmodels.regression.linear_model import burg


class Preprocess:
    def __init__(self, fs: int = 2, seg_window: int= 48) -> None:
        """
        Args:
            fs (int, default=1): Sampling frequency of signals
            seg_window (int, default 48): Window size of sliding window to segment raw signals.
        """
        self.fs = fs
        self.seg_window = seg_window

    def apply_filter(
        self, signal: pd.DataFrame, filter: str = "median", window: int = 5) -> pd.DataFrame:
        """A denosing filter is applied to remove noise in signals.
        Args:
            signal (pd.DataFrame): Raw signal
            filter (str, default='median'): Filter name is chosen from 'mean', 'median', or 'butterworth'
            window (int, default=5): Length of filter
        Returns:
            signal (pd.DataFrame): Filtered signal
        See Also:
            'butterworth' applies a 3rd order low-pass Butterworth filter with a corner frequency of 20 Hz.
        """
        if filter == "mean":
            signal = signal.rolling(window=window, center=True, min_periods=1).mean()
        elif filter == "median":
            signal = signal.rolling(window=window, center=True, min_periods=1).median()
        elif filter == "butterworth":
            fc = 20  # cutoff frequency
            w = fc / (self.fs / 2)  # Normalize the frequency
            b, a = butter(3, w, "low")  # 3rd order low-pass Butterworth filter
            signal = pd.DataFrame(filtfilt(b, a, signal, axis=0), columns=signal.columns)
        else:
            raise ValueError("Not defined filter. See Args.")
            
        return signal

    def normalize(self, signal: pd.DataFrame) -> pd.DataFrame:
        """Apply normalization
        Args:
            signal (pd.DataFrame): Raw signal
        Returns:
            signal (pd.DataFrame): Normalized signal
        """
        df_mean = signal.mean()
        df_std = signal.std()
        signal = (signal - df_mean) / df_std
        return signal

    def segment_signal(
        self,
        signal: pd.DataFrame,
        overlap_rate: int = 0.5,
        res_type: str = "dataframe",
    ) -> List[pd.DataFrame]:
        """Sample signals in fixed-width sliding windows and 50% overlap.
        Args:
            signal (pandas.DataFrame): Raw signal
            overlap_rate (float, default=0.5): Overlap rate of sliding window to segment raw signals.
            res_type (str, default='dataframe'): Type of return value; 'array' or 'dataframe'
        Returns:
            signal_seg (list of pandas.DataFrame): List of segmented sigmal.
        """
        signal_seg = []
        window_size = self.seg_window

        if len(signal) <= window_size:
            signal_seg = [signal]
        else:
            for start_idx in range(0, len(signal) - window_size, int(window_size * overlap_rate)):
                seg = signal.iloc[start_idx : start_idx + window_size].reset_index(drop=True)
                if res_type == "array":
                    seg = seg.values
                signal_seg.append(seg)

        if res_type == "array":
            signal_seg = np.array(signal_seg)

        return signal_seg


    def partition_time_series(
        self,
        signal: pd.DataFrame,
        window_length: int,
        label_length: int,
        stride: int = 1,
        subsample_factor: int = 1,
        binary_delta_labels: bool = True,
        binary_delta_value: str = 'Close'
    ) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
        """Partition a time series signal into a train/test subset of rolling window segments
        of variable window size, test window size, stride and subsample factor. If specified
        the label subset is the outcome of the binary delta in the average value of test set
        minus the train set variable.

        Args:
            signal (pd.DataFrame): The input time series
            window_length (int): The resulting length of each train segment
            label_length (int): The resulting length of each test segment
            stride (int, optional): How many data points to skip per iteration. Defaults to 1.
            subsample_factor (int, optional): Divides the total number of samples in each segment
            by factor. Defaults to 1.

        Returns:
            Tuple(List[pd.DataFrame], List[pd.DataFrame]): Returns a lists of train/test segments
        """
        x_dataset = []
        y_dataset = []

        if len(signal) <= window_length:
            x_dataset = [signal]
            y_dataset = []
        else:
            for i in range(0, len(signal) - (window_length + label_length), stride):
                x_data = signal.iloc[i : i+window_length : subsample_factor].reset_index(drop=True)
                y_data = signal.iloc[i+window_length : 
                                     i+window_length+label_length : 
                                     subsample_factor].reset_index(drop=True)
                if binary_delta_labels:
                    y_data = (y_data[binary_delta_value].mean() - 
                              x_data[binary_delta_value].mean() > 0).astype(int)

                x_dataset.append(x_data)
                y_dataset.append(y_data)

        return x_dataset, y_dataset

    def separate_bias(self, signal: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Separate signal into body and bias signal.
        Another low pass Butterworth filter with a corner frequency of 0.3 Hz is applied.
        Args:
            signal (pd.DataFrame): Segmented signal
        Returns:
            signal_body (pd.DataFrame): Body signal
            rsignal_bias (pd.DataFrame): Bias signal
        """
        fc = 0.3  # cutoff frequency
        w = fc / (self.fs / 2)  # Normalize the frequency
        b, a = butter(3, w, "low")  # 3rd order low pass Butterworth filter
        signal_bias = pd.DataFrame(
            filtfilt(b, a, signal, axis=0), columns=signal.columns
        )  # Apply Butterworth filter

        # Substract bias from sigal.
        signal_body = signal - signal_bias
        return signal_body, signal_bias

    def obtain_jerk_signal(self, signal: pd.DataFrame) -> pd.DataFrame:
        """Derive signal to obtain Jerk signals
        Args:
            signal (pd.DataFrame)
        Returns:
            jerk_signal (pd.DataFrame):
        """
        jerk_signal = signal.diff(periods=1)  # Calculate difference
        jerk_signal.iloc[0] = jerk_signal.iloc[1]  # Fillna
        jerk_signal = jerk_signal / (1 / self.fs)  # Derive in time (1 / sampling frequency)
        return jerk_signal

    def obtain_magnitude(self, signal):
        """Calculate the magnitude of multi-dimensional signals using the Euclidean norm
        Args:
            signal (pandas.DataFrame): multi-dimensional signals
        Returns:
            res (pandas.DataFrame): Magnitude of multi-dimensional signals
        """
        return pd.DataFrame(norm(signal, ord=2, axis=1))

    def obtain_spectrum(self, signal):
        """Obtain spectrum using Fast Fourier Transform (FFT).
        Args:
            signal (pandas.DataFrame): Time domain signals
        Returns:
            amp (pandas.DataFrame): Amplitude spectrum
            phase (pandas.DataFrame): Phase spectrum
        """
        N = len(signal)
        columns = signal.columns

        for col in columns:
            signal[col] = signal[col] * np.hamming(N)  # hamming window

        F = fft(signal, axis=0)  # Apply FFT
        F = F[: N // 2, :]  # Remove the overlapping part

        amp = np.abs(F)  # Obtain the amplitude spectrum
        amp = amp / N * 2
        amp[0] = amp[0] / 2
        amp = pd.DataFrame(amp, columns=columns)  # Convert array to DataFrame
        phase = np.angle(F)
        phase = pd.DataFrame(phase, columns=columns)  # Convert array to DataFrame

        return amp, phase

    def obtain_ecdf_percentile(self, signal, n_bins=10):
        """Obtain ECDF (empirical cumulative distribution function) percentile values.
        Args:
            signal (DataFrame): Time domain signals
            n_bins (int, default: 10): How many percentiles to use as a feature
        Returns:
            features (array): ECDF percentile values.
        """
        idx = np.linspace(0, signal.shape[0] - 1, n_bins)  # Take n_bins linspace percentile.
        idx = [int(Decimal(str(ix)).quantize(Decimal("0"), rounding=ROUND_HALF_UP)) for ix in idx]
        features = np.array([])
        for col in signal.columns:
            ecdf = ECDF(signal[col].values)  # fit
            x = ecdf.x[1:]  # Remove -inf
            feat = x[idx]
            features = np.hstack([features, feat])

        return features

    def obtain_mean(self, signal) -> np.ndarray:
        return signal.mean().values

    def obtain_std(self, signal) -> np.ndarray:
        return signal.std().values

    def obtain_mad(self, signal) -> np.ndarray:
        return stats.median_absolute_deviation(signal, axis=0)

    def obtain_max(self, signal) -> np.ndarray:
        return signal.max().values

    def obtain_min(self, signal) -> np.ndarray:
        return signal.min().values

    def obtain_sma(self, signal) -> np.ndarray:
        window_size = self.seg_window
        window_second = window_size / self.fs
        return sum(signal.sum().values - self.obtain_min(signal) * len(signal)) / window_second

    def obtain_energy(self, signal) -> np.ndarray:
        return norm(signal, ord=2, axis=0) ** 2 / max(1e-7, len(signal))

    def obtain_iqr(self, signal) -> np.ndarray:
        return signal.quantile(0.75).values - signal.quantile(0.25).values

    def obtain_entropy(self, signal) -> np.ndarray:
        signal = signal - signal.min()
        _entropy = np.array([])
        for col in signal.columns:
            try:
                val = stats.entropy(signal[col])
            except FloatingPointError:
                val = 0.0
            _entropy = np.hstack([_entropy, val])
        return _entropy

    def obtain_arCoeff(self, signal) -> np.ndarray:
        arCoeff = np.array([])
        for col in signal.columns:
            try:
                val, _ = burg(signal[col], order=4)
            except FloatingPointError:
                val = [0.0]*4
            arCoeff = np.hstack([arCoeff, val])
        return arCoeff

    def obtain_correlation(self, signal) -> np.ndarray:
        if signal.shape[1] == 1:  # Signal dimension is 1
            correlation = np.array([])
        else:  # Signal dimension is > 1
            cols = signal.columns
            correlation = np.array([])
            for i,_ in enumerate(cols):
                for j,_ in enumerate(cols):
                    if j > i:
                        try:
                            partial = np.corrcoef(signal[cols[i]], signal[cols[j]])[0][1]
                        except FloatingPointError:
                            partial = 0.0
                        correlation = np.hstack([correlation, partial])
        return correlation

    def obtain_maxInds(self, signal) -> np.ndarray:
        return signal.idxmax().values

    def obtain_meanFreq(self, signal) -> np.ndarray:
        meanFreq = np.array([])
        for col in signal.columns:
            val = np.mean(signal[col] * np.arange(len(signal)))
            meanFreq = np.hstack([meanFreq, val])
        return meanFreq

    def obtain_skewness(self, signal) -> np.ndarray:
        return signal.skew().values

    def obtain_kurtosis(self, signal) -> np.ndarray:
        return signal.kurt().values

    def obtain_bandsEnergy(self, signal) -> np.ndarray:
        bandsEnergy = np.array([])
        bins = [0, 4, 8, 12, 16, 20, 24, 29, 34, 39, 44, 49, 54, 59, 64]
        for i in range(len(bins) - 1):
            df = signal.iloc[bins[i] : bins[i + 1]]
            arr = self.obtain_energy(df)
            bandsEnergy = np.hstack((bandsEnergy, arr))
        return bandsEnergy

    def obtain_angle(self, v1, v2) -> np.ndarray:
        length = lambda v: math.sqrt(np.dot(v, v))
        return math.acos(np.dot(v1, v2) / (length(v1) * length(v2)))
    
    def obtain_trend(self, signal) -> np.ndarray:
        trends = np.array([])
        mses = np.array([])
        for col in signal.columns:
            sig = signal[col]
            time_array = np.arange(len(sig))
            window_size = self.seg_window
            Zn = np.concatenate((np.ones((window_size,1)),
                                 time_array[:,np.newaxis]),axis=1)
            MM = inv(Zn.T @ Zn)
            P0 = MM @ (Zn.T @ sig.values)
            trend = P0[1]
            mse = np.mean((sig.values - Zn @ P0)**2)
            trends = np.hstack((trends, trend))
            mses = np.hstack((mses, mse))
        res = np.hstack((trends, mses))
        return res

    def obtain_pct_change(self, signal) -> np.ndarray:
        outliers = np.array([])
        for col in signal.columns:
            sig = signal[col].pct_change()
            sig_m = sig.mean()
            sig_std = sig.std()
            ci_u = sig_m + 6.5 * sig_std
            ci_l = sig_m - 6.5 * sig_std

            if any(sig > ci_u):
                result = 1
            elif any(sig < ci_l):
                result = -1
            else:
                result = 0
            outliers = np.hstack((outliers, result))
        return outliers


def create_features(data_raw: pd.DataFrame, 
                    fs: int=1, 
                    segment_window: int=48, 
                    partitioning: bool=False,
                    window_length: int=12*24,
                    label_length: int=12*3,
                    stride: int = 1,
                    subsample_factor: int = 1,
                    binary_delta_labels: bool = True,
                    binary_delta_value: str = 'Close') -> Tuple[np.ndarray, np.ndarray]:
    """Create features from raw data
    Args:
        data_raw (pd.DataFrame): Raw signals.
    Returns:
        features (np.ndarray): Created features corresponding args with columns denoting feature names.
    """
    of = Preprocess(fs=fs, seg_window=segment_window)  # Create an instance.

    # Remove noises by median filter & Butterworth filter
    # data_raw = of.apply_filter(signal=data_raw, filter="median", window=5)
    # data_raw = of.apply_filter(signal=data_raw, filter="butterworth")

    # Sample signals in fixed-width sliding windows
    if partitioning:
        tRawXYZ, labels = of.partition_time_series(signal=data_raw, 
                                                   window_length=window_length, 
                                                   label_length=label_length, 
                                                   stride=stride, 
                                                   subsample_factor=subsample_factor, 
                                                   binary_delta_labels=binary_delta_labels, 
                                                   binary_delta_value=binary_delta_value)
    else:
        tRawXYZ = of.segment_signal(data_raw, overlap_rate=0.5, res_type="dataframe")
        labels = []

    # Separate signal into body and bias signal
    tBodyRawXYZ, tBiasRawXYZ = [], []
    for raw in tRawXYZ:
        body_raw, bias_raw = of.separate_bias(raw.copy())
        tBodyRawXYZ.append(body_raw)
        tBiasRawXYZ.append(bias_raw)

    # Obtain Jerk signals
    tBodyRawJerkXYZ = []
    for body_raw in tBodyRawXYZ:
        body_raw_jerk = of.obtain_jerk_signal(body_raw.copy())

        tBodyRawJerkXYZ.append(body_raw_jerk)

    # Calculate the magnitude of signals using the Euclidean norm
    tBodyRawMag, tBiasRawMag, tBodyRawJerkMag = (
        [],
        [],
        [],
    )
    for body_raw, bias_raw, body_raw_jerk in zip(
        tBodyRawXYZ, tBiasRawXYZ, tBodyRawJerkXYZ):
        body_raw_mag = of.obtain_magnitude(body_raw.copy())
        bias_raw_mag = of.obtain_magnitude(bias_raw.copy())
        body_raw_jerk_mag = of.obtain_magnitude(body_raw_jerk.copy())

        tBodyRawMag.append(body_raw_mag)
        tBiasRawMag.append(bias_raw_mag)
        tBodyRawJerkMag.append(body_raw_jerk_mag)

    # Obtain amplitude spectrum and Phase using Fast Fourier Transform (FFT).
    (
        fBodyRawXYZAmp,
        fBodyRawJerkXYZAmp,
        fBodyRawMagAmp,
        fBodyRawJerkMagAmp,
    ) = ([], [], [], [])
    (
        fBodyRawXYZPhs,
        fBodyRawJerkXYZPhs,
        fBodyRawMagPhs,
        fBodyRawJerkMagPhs,
    ) = ([], [], [], [])
    for (
        body_raw,
        body_raw_jerk,
        body_raw_mag,
        body_raw_jerk_mag,
    ) in zip(
        tBodyRawXYZ,
        tBodyRawJerkXYZ,
        tBodyRawMag,
        tBodyRawJerkMag,
    ):
        body_raw_amp, body_raw_phase = of.obtain_spectrum(body_raw.copy())
        body_raw_jerk_amp, body_raw_jerk_phase = of.obtain_spectrum(body_raw_jerk.copy())
        body_raw_mag_amp, body_raw_mag_phase = of.obtain_spectrum(body_raw_mag.copy())
        body_raw_jerk_mag_amp, body_raw_jerk_mag_phase = of.obtain_spectrum(
            body_raw_jerk_mag.copy()
        )

        fBodyRawXYZAmp.append(body_raw_amp)
        fBodyRawJerkXYZAmp.append(body_raw_jerk_amp)
        fBodyRawMagAmp.append(body_raw_mag_amp)
        fBodyRawJerkMagAmp.append(body_raw_jerk_mag_amp)

        fBodyRawXYZPhs.append(body_raw_phase)
        fBodyRawJerkXYZPhs.append(body_raw_jerk_phase)
        fBodyRawMagPhs.append(body_raw_mag_phase)
        fBodyRawJerkMagPhs.append(body_raw_jerk_mag_phase)

    #  Following signals are obtained by implementing above functions.
    time_signals = [
        tBodyRawXYZ,
        tBiasRawXYZ,
        tBodyRawJerkXYZ,
        tBodyRawMag,
        tBiasRawMag,
        tBodyRawJerkMag,
    ]
    freq_signals = [
        fBodyRawXYZAmp,
        fBodyRawJerkXYZAmp,
        fBodyRawMagAmp,
        fBodyRawJerkMagAmp,
        fBodyRawXYZPhs,
        fBodyRawJerkXYZPhs,
        fBodyRawMagPhs,
        fBodyRawJerkMagPhs,
    ]

    all_signals = time_signals + freq_signals

    # Calculate feature vectors by using signals
    features = []

    for i in range(len(tBodyRawXYZ)):
        feature_vector = np.array([])

        # mean, std, mad, max, min, sma, energy, iqr, entropy
        for t_signal in all_signals:
            sig = t_signal[i].copy()
            mean = of.obtain_mean(sig)
            std = of.obtain_std(sig)
            mad = of.obtain_mad(sig)
            max_val = of.obtain_max(sig)
            min_val = of.obtain_min(sig)
            sma = of.obtain_sma(sig)
            energy = of.obtain_energy(sig)
            iqr = of.obtain_iqr(sig)
            entropy = of.obtain_entropy(sig)
            feature_vector = np.hstack(
                (feature_vector, mean, std, mad, max_val, min_val, sma, energy, iqr, entropy)
            )

        # arCoeff
        for t_signal in time_signals:
            sig = t_signal[i].copy()
            arCoeff = of.obtain_arCoeff(sig)
            feature_vector = np.hstack((feature_vector, arCoeff))

        # correlation
        for t_signal in [
            tBodyRawXYZ,
            tBiasRawXYZ,
            tBodyRawJerkXYZ,
        ]:
            sig = t_signal[i].copy()
            correlation = of.obtain_correlation(sig)
            feature_vector = np.hstack((feature_vector, correlation))

        # maxInds, meanFreq, skewness, kurtosis
        for t_signal in freq_signals:
            sig = t_signal[i].copy()
            maxInds = of.obtain_maxInds(sig)
            meanFreq = of.obtain_meanFreq(sig)
            skewness = of.obtain_skewness(sig)
            kurtosis = of.obtain_kurtosis(sig)
            feature_vector = np.hstack((feature_vector, maxInds, meanFreq, skewness, kurtosis))

        # bandsEnergy
        for t_signal in [tBodyRawXYZ, tBodyRawJerkXYZ]:
            sig = t_signal[i].copy()
            bandsEnergy = of.obtain_bandsEnergy(sig)
            feature_vector = np.hstack((feature_vector, bandsEnergy))

        # angle
        biasMean = tBiasRawXYZ[i].mean()
        tBodyRawMean = tBodyRawXYZ[i].mean()
        tBodyRawJerkMean = tBodyRawJerkXYZ[i].mean()

        tBodyRawWRTBias = of.obtain_angle(tBodyRawMean, biasMean)
        tBodyRawJerkWRTBias = of.obtain_angle(tBodyRawJerkMean, biasMean)

        feature_vector = np.hstack(
            (
                feature_vector,
                tBodyRawWRTBias,
                tBodyRawJerkWRTBias,
            )
        )

        # ECDF
        for t_signal in [tBodyRawXYZ]:
            sig = t_signal[i].copy()
            ecdf = of.obtain_ecdf_percentile(sig)
            feature_vector = np.hstack((feature_vector, ecdf))
        
        # trend
        for t_signal in [tBodyRawXYZ]:
            sig = t_signal[i].copy()
            trend = of.obtain_trend(sig)
            feature_vector = np.hstack((feature_vector, trend))

        # pct_change
        for t_signal in [tBodyRawXYZ]:
            sig = t_signal[i].copy()
            outlier_s = of.obtain_pct_change(sig)
            feature_vector = np.hstack((feature_vector, outlier_s))
            
        features.append(feature_vector)

    return np.array(features), np.array(labels)


def get_feature_names(raw_names: List[str]) -> List[str]:
    """Get feature names
    Returns:
        feature_names (List[str]): Title of features
    """
    time_signal_names = [
        "tBodyRawXYZ",
        "tBiasRawXYZ",
        "tBodyRawJerkXYZ",
        "tBodyRawMag",
        "tBiasRawMag",
        "tBodyRawJerkMag",
    ]
    freq_signal_names = [
        "fBodyRawXYZAmp",
        "fBodyRawJerkXYZAmp",
        "fBodyRawMagAmp",
        "fBodyRawJerkMagAmp",
        "fBodyRawXYZPhs",
        "fBodyRawJerkXYZPhs",
        "fBodyRawMagPhs",
        "fBodyRawJerkMagPhs",
    ]
    all_signal_names = time_signal_names + freq_signal_names
    feature_names = []

    for name in all_signal_names:
        for s in ["Mean", "Std", "Mad", "Max", "Min", "Sma", "Energy", "Iqr", "Entropy"]:
            if s == "Sma":
                feature_names.append(f"{name}{s}")
                continue
            if "XYZ" in name:
                n = name.replace("XYZ", "")
                feature_names += [f"{n}{s}-{ax}" for ax in raw_names]
            else:
                feature_names.append(f"{name}{s}")

    for name in time_signal_names:
        if "XYZ" in name:
            n = name.replace("XYZ", "")
            feature_names += [f"{n}ArCoeff-{ax}{i}" for ax in raw_names for i in range(4)]
        else:
            feature_names += [f"{name}ArCoeff{i}" for i in range(4)]

    for name in [
        "tBodyRawXYZ",
        "tBiasRawXYZ",
        "tBodyRawJerkXYZ",
    ]:
        n = name.replace("XYZ", "")
        for i, i_name in enumerate(raw_names):
            for j, j_name in enumerate(raw_names):
                if j > i:
                    feature_names += [f"{n}-Correlation-{i_name}-{j_name}"]

    for name in freq_signal_names:
        for s in ["MaxInds", "MeanFreq", "Skewness", "Kurtosis"]:
            if "XYZ" in name:
                n = name.replace("XYZ", "")
                feature_names += [f"{n}{s}-{ax}" for ax in raw_names]
            else:
                feature_names.append(f"{name}{s}")

    for name in ["tBodyRawXYZ", "tBodyRawJerkXYZ"]:
        n = name.replace("XYZ", "")
        feature_names += [f"{n}BandsEnergy-{ax}{i}" for i in range(14) for ax in raw_names]

    feature_names += [
        "tBodyRawWRTBias",
        "tBodyRawJerkWRTBias",
    ]

    feature_names += [
        f"tBody{sensor}ECDF-{axis}{i}"
        for sensor in ["Raw"]
        for axis in raw_names
        for i in range(10)
    ]
    
    for name in ["tBodyRawXYZ"]:
        n = name.replace("XYZ", "")
        feature_names += [f"{n}Trend-{ax}" for ax in raw_names]
        
    for name in ["tBodyRawXYZ"]:
        n = name.replace("XYZ", "")
        feature_names += [f"{n}MSE-{ax}" for ax in raw_names]

    for name in ["tBodyRawXYZ"]:
        n = name.replace("XYZ", "")
        feature_names += [f"{n}PCT_CHNG-{ax}" for ax in raw_names]
    
    return feature_names