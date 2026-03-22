import pandas as pd
from typing import List, Tuple

def _validate_df(df: pd.DataFrame) -> None:
    missing = {"timestamp", "value"} - set(df.columns)

    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        raise TypeError('Column "timestamp" must be datetime64.')


def _validate_anomaly_windows(anomaly_windows: List[Tuple[object, object]]) -> None:
    if len(anomaly_windows) == 0:
        raise ValueError("anomalous_windows must not be empty")

    for i, (start, end) in enumerate(anomaly_windows):
        try:
            start_ts = pd.to_datetime(start)
            end_ts = pd.to_datetime(end)
        except Exception as e:
            raise TypeError(
                f"Window {i} contains a non-datetime value: start={start!r}, end={end!r}"
            ) from e

        if pd.isna(start_ts) or pd.isna(end_ts):
            raise ValueError(
                f"Window {i} contains a null datetime: start={start!r}, end={end!r}"
            )

        if start_ts > end_ts:
            raise ValueError(
                f"Window {i} has start after end: start={start_ts}, end={end_ts}"
            )
