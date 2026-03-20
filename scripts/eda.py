import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from typing import Dict, List, Tuple, Optional, Any, Callable
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from tabulate import tabulate

def _validate_df(df: pd.DataFrame) -> None:
    missing = {'timestamp', 'value'} - set(df.columns)

    if missing:
        raise ValueError(f'DataFrame is missing required columns: {missing}')
    
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        raise TypeError('Column "timestamp" must be datetime64.')

def _validate_anomaly_windows(anomaly_windows: List[Tuple[object, object]]) -> None:
    if len(anomaly_windows) == 0:
        raise ValueError('anomalous_windows must not be empty')

    for i, (start, end) in enumerate(anomaly_windows):
        try:
            start_ts = pd.to_datetime(start)
            end_ts = pd.to_datetime(end)
        except Exception as e:
            raise TypeError(
                f'Window {i} contains a non-datetime value: start={start!r}, end={end!r}'
            ) from e

        if pd.isna(start_ts) or pd.isna(end_ts):
            raise ValueError(
                f'Window {i} contains a null datetime: start={start!r}, end={end!r}'
            )

        if start_ts > end_ts:
            raise ValueError(
                f'Window {i} has start after end: start={start_ts}, end={end_ts}'
            )

def basic_summary(df: pd.DataFrame, should_print: bool = False) -> Dict:
    """
    Prints basic information for the series, including min, max, sampling interval, missing intervals (based on median), and basic summary statistics.
    """

    _validate_df(df)

    # Preliminary information (rows, start, end)
    n_rows = len(df['timestamp'])
    start_time = df['timestamp'].min()
    end_time = df['timestamp'].max()

    # The sampling interval will help determine missing values.
    time_diffs = df['timestamp'].diff().dropna()
    median_interval = time_diffs.median()
    mode_interval = time_diffs.mode()

    # The missing intervals will be important. 
    # We compute this by generating a sequence of evenly-spaced timestamps using the median interval.
    # Then we take the timestamps which actually exist and check which are absent in the observed data.
    expected_index = pd.date_range(start=start_time, end=end_time, freq=median_interval)
    observed_index = pd.DatetimeIndex(df['timestamp'])
    missing_timestamps = expected_index.difference(observed_index)
    n_missing = len(missing_timestamps)
    n_duplicate_timestamps = observed_index.duplicated().sum()

    # Summary statistics.
    mean_val = df['value'].mean()
    std_val = df['value'].std()

    summary = {
        'n_rows': n_rows,
        'start_time': start_time,
        'end_time': end_time,
        'median_interval': median_interval,
        'mode_interval': mode_interval,
        'n_missing_timestamps': n_missing,
        'n_duplicate_timestamps': n_duplicate_timestamps,
        'min_value': df['value'].min(),
        'max_value': df['value'].max(),
        'mean_value': mean_val,
        'std_value': std_val,
    }

    if should_print:
        print(tabulate(summary.items(), headers=['Metric', 'Value'], tablefmt='simple'))

    return summary

def plot_series(df: pd.DataFrame, anomaly_windows: List[Tuple[pd.Timestamp, pd.Timestamp]], title: str = 'Time Series Plot') -> None:
    """
    Creates a basic plot of the series with anomaly windows outlined.
    """

    _validate_df(df)

    _, ax = plt.subplots(figsize=(12, 6))

    # We first draw the anomaly windows.
    for i, (start, end) in enumerate(anomaly_windows):
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)

        # Creating a span in the graph for better visibility.
        ax.axvspan(
            start,
            end,
            color='red',
            alpha=0.35,
            zorder=1,
            label='Anomaly window' if i == 0 else None
        )

    ax.plot(df['timestamp'], df['value'], color='blue', linewidth=1.0, zorder=3, label='Value')
    ax.set_title(title)
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Value')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_distribution(df: pd.DataFrame, bins: int = 40, title: str = 'Distribution') -> None:
    """
    Plots basic histogram and box plot to outline distribution.
    """

    _validate_df(df)

    _, axes = plt.subplots(2, 1, figsize=(12, 6))

    axes[0].hist(df['value'], bins=bins)
    axes[0].set_title(f'{title} - Histogram')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Frequency')

    axes[1].boxplot(df['value'], vert=False)
    axes[1].set_title(f'{title} - Boxplot')
    axes[1].set_xlabel('Value')

    plt.tight_layout()
    plt.show()

def plot_rolling_statistics(df: pd.DataFrame, window: int = 24, title: str = 'Rolling Statistics') -> None:
    """
    Plots the rolling mean and standard deviation.
    """

    _validate_df(df)

    df = df.copy()

    rolling_mean = df['value'].rolling(window=window).mean()
    rolling_std = df['value'].rolling(window=window).std()

    _, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(df['timestamp'], df['value'], linewidth=0.8, label='Value')
    axes[0].plot(df['timestamp'], rolling_mean, linewidth=2, label=f'Rolling Mean ({window})')
    axes[0].set_title(f'{title} - Rolling Mean')
    axes[0].set_ylabel('Value')
    axes[0].legend()

    axes[1].plot(df['timestamp'], rolling_std, linewidth=1.5, label=f'Rolling Std ({window})')
    axes[1].set_title(f'{title} - Rolling Std')
    axes[1].set_xlabel('Timestamp')
    axes[1].set_ylabel('Std')
    axes[1].legend()

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_acf_series(df: pd.DataFrame, lags: int = 100, title: str = 'ACF') -> None:
    _validate_df(df)

    df = df.copy()

    _, ax = plt.subplots(figsize=(12, 4))
    plot_acf(df['value'].dropna(), lags=lags, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def plot_pacf_series(df: pd.DataFrame, lags: int = 50, title: str = 'PACF') -> None:
    _validate_df(df)

    df = df.copy()

    _, ax = plt.subplots(figsize=(12, 4))
    plot_pacf(df['value'].dropna(), lags=lags, ax=ax, method='ywm')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def plot_hourly_mean(df: pd.DataFrame, title: str = 'Hourly Pattern') -> None:
    _validate_df(df)

    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour

    hourly_mean = df.groupby('hour')['value'].mean()

    _, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hourly_mean.index, hourly_mean.values, marker='o')
    ax.set_title(title)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Average Value')
    ax.set_xticks(range(24))
    plt.tight_layout()
    plt.show()

def plot_dayofweek_mean(df: pd.DataFrame, title: str = 'Day-of-Week Pattern') -> None:
    _validate_df(df)

    df = df.copy()

    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['day_name'] = pd.Categorical(df['timestamp'].dt.day_name(), categories=day_order, ordered=True)

    dow_mean = df.groupby('day_name')['value'].mean()

    _, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dow_mean.index, dow_mean.values, marker='o')
    ax.set_title(title)
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Average Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_boxplot_by_hour(df: pd.DataFrame, title: str = 'Boxplot by Hour of Day') -> None:
    _validate_df(df)

    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour

    data = [df.loc[df['hour'] == h, 'value'].dropna().values for h in range(24)]

    _, ax = plt.subplots(figsize=(12, 5))
    ax.boxplot(data, labels=list(range(24)), showfliers=True)
    ax.set_title(title)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Value')
    plt.tight_layout()
    plt.show()

def plot_boxplot_by_dayofweek(df: pd.DataFrame, title: str = 'Boxplot by Day of Week') -> None:
    _validate_df(df)

    df = df.copy()

    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['day_name'] = pd.Categorical(df['timestamp'].dt.day_name(), categories=day_order, ordered=True)

    data = [df.loc[df['day_name'] == d, 'value'].dropna().values for d in day_order]

    _, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data, labels=day_order, showfliers=True)
    ax.set_title(title)
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def adf_test(df: pd.DataFrame, value_col: str = 'value', should_print: bool = False) -> Dict:
    """
    The Augmented Dickey-Fuller Test helps us determine whether a time series behaves like a Random Walk model.
        - Random Walk is a time series where each value equals the previous value plus a random shock (increment).
    
    This persistence induces non-stationarity, but this test is also sensitive to inclusion of trend component, which needs to be handled.
    """

    series = df[value_col].dropna().astype(float)

    result = adfuller(series, autolag='AIC')

    summary = {
        'adf_statistic': result[0],
        'adf_p_value': result[1],
        'adf_used_lag': result[2],
        'adf_n_obs': result[3],
        'adf_critical_value_1pct': result[4]['1%'],
        'adf_critical_value_5pct': result[4]['5%'],
        'adf_critical_value_10pct': result[4]['10%'],
        'adf_is_stationary_at_5pct': result[1] < 0.05,
    }

    if should_print:
        print(tabulate(summary.items(), headers=['Property', 'Value'], tablefmt='simple'))

    return summary

def compute_anomaly_statistics(
        df: pd.DataFrame, 
        anomaly_windows: List[Tuple[pd.Timestamp, pd.Timestamp]], 
        timestamp_col: str = 'timestamp',
        should_print: bool = False
) -> Dict:
    """
    Given that Exploratory Data Analysis (in this case) contains information about anomalous windows, we can extract some data from this.
    
    In any domain, we should know how frequent anomalous behavior is in order to assess how.
    """

    _validate_df(df)
    _validate_anomaly_windows(anomaly_windows=anomaly_windows)

    df = df.copy()

    durations = [(end - start) for start, end in anomaly_windows]
    total_duration = sum(durations, pd.Timedelta(0))
    avg_duration = total_duration / len(anomaly_windows) if anomaly_windows else pd.Timedelta(0)

    # The easiest way to assign row membership to anomaly sections is with a boolean mask.
    timestamps = df[timestamp_col]
    in_anomaly = np.zeros(len(df), dtype=bool)

    for start, end in anomaly_windows:
        in_anomaly |= (timestamps >= start) & (timestamps <= end)

    n_points = len(df)
    n_anomalous_points = int(in_anomaly.sum())
    anomalous_fraction = n_anomalous_points / n_points if n_points > 0 else np.nan

    summary = {
        'n_anomaly_windows': len(anomaly_windows),
        'total_anomaly_duration': total_duration,
        'avg_anomaly_duration': avg_duration,
        'n_points': n_points,
        'n_anomalous_points': n_anomalous_points,
        'anomalous_point_fraction': anomalous_fraction,
    }

    if should_print:
        print(tabulate(summary.items(), headers=['Property', 'Value'], tablefmt='simple'))

    return summary

def plot_lag(
        df: pd.DataFrame, 
        anomaly_windows: List[Tuple[pd.Timestamp, pd.Timestamp]],
        lags: List[int] = [1],
        value_col: str = 'value',
        time_col: str = 'timestamp',
        title: str = 'Lag Plot'
) -> None:
    """
    Lag Plots are a useful way to understand some properties of a time series, particularly seasonality when a repeating structure is seen at a particular lag.

    This lag plot also includes different colors for different cases (neither anomalous, anomalous at t, anomalous at t-lag, both anomalous).
    """

    _validate_df(df)
    _validate_anomaly_windows(anomaly_windows)

    df = df.copy()

    in_anomaly = np.zeros(len(df), dtype=bool)
    for start, end in anomaly_windows:
        in_anomaly |= (df[time_col] >= start) & (df[time_col] <= end)

    df['in_anomaly'] = in_anomaly

    n_lags = len(lags)
    n_cols = min(3, n_lags)
    n_rows = math.ceil(n_lags / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, lag in zip(axes, lags):
        lagged = pd.DataFrame({
            f'{value_col}_t': df[value_col],
            f'{value_col}_t_minus_{lag}': df[value_col].shift(lag),
            'in_anomaly_t': df['in_anomaly'],
            'in_anomaly_t_minus_lag': df['in_anomaly'].shift(lag)
        }).dropna()

        lagged['in_anomaly_t'] = lagged['in_anomaly_t'].astype(bool)
        lagged['in_anomaly_t_minus_lag'] = lagged['in_anomaly_t_minus_lag'].astype(bool)

        # An important aspect to note - if we want to outline anomalous points, do we count both pairs as anomalous if one is an anomaly?
        # In this case, I will keep it as specific as possible, four groups:
        # 0 - neither anomalous.
        # 1 - lagged value anomalous.
        # 2 - current timestamp anomalous.
        # 3 - both anomalous.

        lagged['anomaly_pair_type'] = 0
        lagged.loc[
            (~lagged['in_anomaly_t']) & (lagged['in_anomaly_t_minus_lag']),
            'anomaly_pair_type'
        ] = 1
        lagged.loc[
            (lagged['in_anomaly_t']) & (~lagged['in_anomaly_t_minus_lag']),
            'anomaly_pair_type'
        ] = 2
        lagged.loc[
            (lagged['in_anomaly_t']) & (lagged['in_anomaly_t_minus_lag']),
            'anomaly_pair_type'
        ] = 3

        normal = lagged[lagged['anomaly_pair_type'] == 0]
        lag_only_anomalous = lagged[lagged['anomaly_pair_type'] == 1]
        t_only_anomalous = lagged[lagged['anomaly_pair_type'] == 2]
        both_anomalous = lagged[lagged['anomaly_pair_type'] == 3]

        ax.scatter(
            normal[f'{value_col}_t_minus_{lag}'],
            normal[f'{value_col}_t'],
            alpha=0.10,
            s=16,
            label='Neither anomalous'
        )

        ax.scatter(
            lag_only_anomalous[f'{value_col}_t_minus_{lag}'],
            lag_only_anomalous[f'{value_col}_t'],
            alpha=0.55,
            s=22,
            label=f'Anomalous at t-{lag} only'
        )

        ax.scatter(
            t_only_anomalous[f'{value_col}_t_minus_{lag}'],
            t_only_anomalous[f'{value_col}_t'],
            alpha=0.55,
            s=22,
            label='Anomalous at t only'
        )

        ax.scatter(
            both_anomalous[f'{value_col}_t_minus_{lag}'],
            both_anomalous[f'{value_col}_t'],
            alpha=0.95,
            s=30,
            label='Both anomalous'
        )

        ax.set_xlabel(f'{value_col}(t-{lag})')
        ax.set_ylabel(f'{value_col}(t)')
        ax.set_title(f'{title} (lag={lag})')

    # Hide axes not being used.
    for ax in axes[n_lags:]:
        ax.axis('off')

    # We don't need to repeat the legend multiple times.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4)
    plt.tight_layout()
    plt.show()

    lagged['anomaly_pair_type'] = 0
    lagged.loc[
        (~lagged['in_anomaly_t']) & (lagged['in_anomaly_t_minus_lag']),
        'anomaly_pair_type'
    ] = 1
    lagged.loc[
        (lagged['in_anomaly_t']) & (~lagged['in_anomaly_t_minus_lag']),
        'anomaly_pair_type'
    ] = 2
    lagged.loc[
        (lagged['in_anomaly_t']) & (lagged['in_anomaly_t_minus_lag']),
        'anomaly_pair_type'
    ] = 3

    normal = lagged[lagged['anomaly_pair_type'] == 0]
    lag_only_anomalous = lagged[lagged['anomaly_pair_type'] == 1]
    t_only_anomalous = lagged[lagged['anomaly_pair_type'] == 2]
    both_anomalous = lagged[lagged['anomaly_pair_type'] == 3]

    _, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(
        normal[f'{value_col}_t_minus_{lag}'],
        normal[f'{value_col}_t'],
        alpha=0.10,
        s=16,
        label='Neither anomalous'
    )

    ax.scatter(
        lag_only_anomalous[f'{value_col}_t_minus_{lag}'],
        lag_only_anomalous[f'{value_col}_t'],
        alpha=0.55,
        s=22,
        label=f'Anomalous at t-{lag} only'
    )

    ax.scatter(
        t_only_anomalous[f'{value_col}_t_minus_{lag}'],
        t_only_anomalous[f'{value_col}_t'],
        alpha=0.55,
        s=22,
        label='Anomalous at t only'
    )

    ax.scatter(
        both_anomalous[f'{value_col}_t_minus_{lag}'],
        both_anomalous[f'{value_col}_t'],
        alpha=0.95,
        s=30,
        label='Both anomalous'
    )

    ax.set_xlabel(f'{value_col}(t-{lag})')
    ax.set_ylabel(f'{value_col}(t)')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()

def fast_fourier_transform_estimation(
        df: pd.DataFrame,
        value_col: str = 'value',
        time_col: str = 'timestamp',
        sampling_interval: pd.Timedelta | None = None,
        should_plot: bool = False
) -> Dict:
    """
    A Fourier Transform helps convert a time series into frequency components to identify underlying patterns like seasonality.
        - Alternatives include Wavelet Transform, EMB, etc.

    FFT is an algorithm to compute Discrete Fourier Transform and its inverse to transform the signal form time domain to frequency domain.
        - The Fourier series is known as a set of peridic functions (sine, cosine).

    The goal is to represent any periodic function by the sum of sinusoidal functions of different frequencies.

    Preconditions:
        - Data should be made stationary for best results.
    """

    _validate_df(df)

    df = df.copy()

    x = df[value_col].astype(float).values
    
    # Centering
    x = x - np.mean(x)
    n = len(x)

    if sampling_interval is None:
        diffs = df[time_col].diff().dropna()

        if len(diffs) == 0:
            raise ValueError('Series is not suitable for FFT, insufficient length.')

        sampling_interval = diffs.mode().iloc[0]
    else:
        sampling_interval = pd.to_timedelta(sampling_interval)

    # Now, we compute frequency in cycles per sample.
    fft_values = np.fft.rfft(x)

    # Our frequency unit will be cycles per five minutes, as our sample spacing is 5 minutes. This makes frequency unit cycles/sample, which we can convert to hours later.
    # If we encounter a signal with a period of 12 samples, the repetition occurs every hour in our time domain.
    freqs_per_sample = np.fft.rfftfreq(n, d=1.0)
    
    magnitude = np.abs(fft_values)
    power = magnitude ** 2

    # This is to ignore the zero-frequency components.
    if len(power) > 0:
        power[0] = 0
        magnitude[0] = 0

    peak_idx = int(np.argmax(power))
    dominant_freq_per_sample = freqs_per_sample[peak_idx]

    if dominant_freq_per_sample <= 0:
        dominant_period_samples = np.nan
        dominant_period_timedelta = pd.NaT
    else:
        dominant_period_samples = 1.0 / dominant_freq_per_sample
        dominant_period_timedelta = sampling_interval * dominant_period_samples

    summary = {
        "dominant_frequency_per_sample": dominant_freq_per_sample,
        "dominant_period_in_samples": dominant_period_samples,
        "dominant_period_timedelta": dominant_period_timedelta,
        "sampling_interval": sampling_interval,
        "n_used_points": n,
    }

    if should_plot:
        _, ax = plt.subplots(figsize=(12, 6))
        ax.plot(freqs_per_sample, magnitude)

        if dominant_freq_per_sample > 0:
            ax.axvline(
                dominant_freq_per_sample,
                linestyle='--',
                alpha=0.8,
                label=(f'Dominant Frequency = {dominant_freq_per_sample} cycles/sample')
            )
            ax.legend()
        
        ax.set_title('FFT Magnitude Spectrum')
        ax.set_xlabel('Frequency (cycles per sample)')
        ax.set_ylabel('Magnitude')
        plt.tight_layout()
        plt.show()

        print(tabulate(summary.items(), headers=['Property', 'Value'], tablefmt='simple'))

    return summary
