import numpy as np
import pandas as pd
from scripts.utils import _validate_df, _validate_anomaly_windows
from typing import List, Tuple, Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def make_anomaly_mask(
    df: pd.DataFrame,
    anomaly_windows: List[Tuple[pd.Timestamp, pd.Timestamp]],
    time_col: str = "timestamp",
) -> np.ndarray:
    """This method creates a mask to identify which rows are in the provided anomaly
    windows. This will help determine valid training-test splits.

    Returns:

    in_anomaly: np.ndarray, shape (len(df), ).
    """
    _validate_df(df=df)
    _validate_anomaly_windows(anomaly_windows=anomaly_windows)

    timestamps = df[time_col]
    in_anomaly = np.zeros(len(df), dtype=bool)

    for start, end in anomaly_windows:
        in_anomaly |= (timestamps >= start) & (timestamps <= end)

    return in_anomaly


def make_sliding_window_dataset(
    df: pd.DataFrame,
    anomaly_windows: List[Tuple[pd.Timestamp, pd.Timestamp]],
    W: int,
    H: int,
    time_col: str = "timestamp",
    value_col: str = "value",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """This method will partition the time series into a supervised sliding window
    dataset for binary anomaly detection.

    We create this dataset by using reference points p, for which we retrieve the previous W frames. Additionally, we create a target column y (binary) representing whether an anomalous point is detected in the subsequent H frames.

    Note, we implemented this with overlap, so consecutive windows overlap by W-1 samples.

    Returns:

    X : np.ndarray, shape (n_samples, W), representing the feature matrix.
    y : np.ndarray, shape (n_samples,), representing the binary label (0 or 1).
    window_end_times : np.ndarray containing pd.Timestamp, shape (n_samples,).
    """
    _validate_df(df=df)
    _validate_anomaly_windows(anomaly_windows=anomaly_windows)

    if W < 1:
        raise ValueError(f"W must be >= 1, received {W}")
    if H < 1:
        raise ValueError(f"H must be >= 1, received {H}")

    values = df[value_col].to_numpy(dtype=float)
    timestamps = df[time_col].to_numpy()
    n = len(values)

    if n < W + H:
        raise ValueError(f"Series length ({n}) is too short for W={W} and H={H}. ")

    anomaly_mask = make_anomaly_mask(
        df=df, anomaly_windows=anomaly_windows, time_col=time_col
    )

    # p is the prediction point, occurring at the first index of the horizon H (we cannot read further any data, must predict based on W).
    # We will have a prediction point for every value in the range [W, n-H].
    # This makes the number of windows in our dataset equivalent to n - W - H + 1.
    n_samples = n - W - H + 1

    X = np.empty((n_samples, W), dtype=float)

    # Int type selected since models may be incompatible with bool.
    y = np.empty(n_samples, dtype=np.int8)

    # We need to store the end-times for each window so that we can make a more reasonable split based on time-stamp for anomaly windows.
    # Additionally, when we make a prediction, we can use window_end_times to make plots to evaluate predictive performance.
    window_end_times = np.empty(n_samples, dtype=timestamps.dtype)

    for idx, p in enumerate(range(W, n - H + 1)):
        X[idx] = values[p - W : p]

        # Here, we have a few choices, either we want to determine how many anomalous points, but in our case, we just need one.
        y[idx] = int(anomaly_mask[p : p + H].any())
        window_end_times[idx] = timestamps[p]

    return X, y, window_end_times


def chronological_split(
    X: np.ndarray,
    y: np.ndarray,
    end_times: pd.Series,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Dict[str, Any]:
    """This method will produce a chronological split of the time-series."""

    n = len(X)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    return {
        "X_train": X[:train_end],
        "y_train": y[:train_end],
        "t_train": end_times.iloc[:train_end].reset_index(drop=True),
        "X_val": X[train_end:val_end],
        "y_val": y[train_end:val_end],
        "t_val": end_times.iloc[train_end:val_end].reset_index(drop=True),
        "X_test": X[val_end:],
        "y_test": y[val_end:],
        "t_test": end_times.iloc[val_end:].reset_index(drop=True),
    }


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_iter: int = 1000,
) -> Pipeline:
    """Trains a Logistic Regression classifier on the flattened sliding window features.

    We include the StandardScaler in this pipeline, along with 'class_weight=balanced'
    to compensate for class mbalance in anomaly detection.
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=max_iter)),
        ]
    )

    pipeline.fit(X_train, y_train)

    return pipeline
