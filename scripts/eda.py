import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
from typing import Dict, List, Tuple, Optional
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from tabulate import tabulate

# Validates the structure of the DataFrame.
def _validate_df(df: pd.DataFrame) -> None:
    missing = {'timestamp', 'value'} - set(df.columns)

    if missing:
        raise ValueError(f'DataFrame is missing required columns: {missing}')
    
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        raise TypeError('Column "timestamp" must be datetime64.')

# Prints basic information for the series, including min, max, sampling interval, missing intervals (based on median), and basic summary statistics.
def basic_summary(df: pd.DataFrame, should_print: bool = False) -> Dict:
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

    print(tabulate(summary.items(), headers=["Metric", "Value"], tablefmt="simple"))

    return summary

# Creates a basic plot of the series with anomaly windows outlined.
def plot_series(df: pd.DataFrame, anomaly_windows: List[Tuple[pd.Timestamp, pd.Timestamp]], title: str = 'Time Series Plot') -> None:
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

# Plots basic histogram and box plot to outline distribution.
def plot_distribution(df: pd.DataFrame, bins: int = 40, title: str = 'Distribution') -> None:
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

# Plots the rolling mean and standard deviation.
def plot_rolling_statistics(df: pd.DataFrame, window: int = 24, title: str = 'Rolling Statistics') -> None:
    _validate_df(df)

    df = df.copy()

    rolling_mean = df['value'].rolling(window=window).mean()
    rolling_std = df['value'].rolling(window=window).std()

    _, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(df["timestamp"], df["value"], linewidth=0.8, label="Value")
    axes[0].plot(df["timestamp"], rolling_mean, linewidth=2, label=f"Rolling Mean ({window})")
    axes[0].set_title(f"{title} - Rolling Mean")
    axes[0].set_ylabel("Value")
    axes[0].legend()

    axes[1].plot(df["timestamp"], rolling_std, linewidth=1.5, label=f"Rolling Std ({window})")
    axes[1].set_title(f"{title} - Rolling Std")
    axes[1].set_xlabel("Timestamp")
    axes[1].set_ylabel("Std")
    axes[1].legend()

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Plots the auto-correlation graph for the series.
def plot_acf_series(df: pd.DataFrame, lags: int = 100, title: str = "ACF") -> None:
    _validate_df(df)

    df = df.copy()

    _, ax = plt.subplots(figsize=(12, 4))
    plot_acf(df["value"].dropna(), lags=lags, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

# Plots the partial auto-correlation graph for the series.
def plot_pacf_series(df: pd.DataFrame, lags: int = 50, title: str = "PACF") -> None:
    _validate_df(df)

    df = df.copy()

    _, ax = plt.subplots(figsize=(12, 4))
    plot_pacf(df["value"].dropna(), lags=lags, ax=ax, method="ywm")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

# Plots the mean value by hour of the day.
def plot_hourly_pattern(df: pd.DataFrame, title: str = "Hourly Pattern") -> None:
    _validate_df(df)

    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour

    hourly_mean = df.groupby("hour")["value"].mean()

    _, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hourly_mean.index, hourly_mean.values, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Value")
    ax.set_xticks(range(24))
    plt.tight_layout()
    plt.show()

# Plots the mean value by day of the week.
def plot_dayofweek_pattern(df: pd.DataFrame, title: str = "Day-of-Week Pattern") -> None:
    _validate_df(df)

    df = df.copy()

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df["day_name"] = pd.Categorical(df["timestamp"].dt.day_name(), categories=day_order, ordered=True)

    dow_mean = df.groupby("day_name")["value"].mean()

    _, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dow_mean.index, dow_mean.values, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Average Value")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Plots the box-and-whisker for hour of day.
def plot_boxplot_by_hour(df: pd.DataFrame, title: str = "Boxplot by Hour of Day") -> None:
    _validate_df(df)

    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour

    data = [df.loc[df["hour"] == h, "value"].dropna().values for h in range(24)]

    _, ax = plt.subplots(figsize=(12, 5))
    ax.boxplot(data, labels=list(range(24)), showfliers=True)
    ax.set_title(title)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Value")
    plt.tight_layout()
    plt.show()

# Plots the box-and-whisker for hour of day.
def plot_boxplot_by_dayofweek(df: pd.DataFrame, title: str = "Boxplot by Day of Week") -> None:
    _validate_df(df)

    df = df.copy()

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df["day_name"] = pd.Categorical(df["timestamp"].dt.day_name(), categories=day_order, ordered=True)

    data = [df.loc[df["day_name"] == d, "value"].dropna().values for d in day_order]

    _, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data, labels=day_order, showfliers=True)
    ax.set_title(title)
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Value")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()