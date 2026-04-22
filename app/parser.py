from typing import Optional

import numpy as np
import pandas as pd

REQUIRED_BASE_COLUMNS = ["time"]
AXIS_COLUMNS = ["gyro_x", "gyro_y", "gyro_z"]
OPTIONAL_COLUMNS = ["throttle", "setpoint_roll", "setpoint_pitch", "setpoint_yaw"]

COLUMN_ALIASES = {
    "t": "time",
    "timestamp": "time",
    "time_s": "time",
    "time_sec": "time",
    "seconds": "time",
    "roll": "gyro_x",
    "pitch": "gyro_y",
    "yaw": "gyro_z",
    "gyro_roll": "gyro_x",
    "gyro_pitch": "gyro_y",
    "gyro_yaw": "gyro_z",
    "gx": "gyro_x",
    "gy": "gyro_y",
    "gz": "gyro_z",
    "rc_roll": "setpoint_roll",
    "rc_pitch": "setpoint_pitch",
    "rc_yaw": "setpoint_yaw",
    "roll_setpoint": "setpoint_roll",
    "pitch_setpoint": "setpoint_pitch",
    "yaw_setpoint": "setpoint_yaw",
    "setpoint_x": "setpoint_roll",
    "setpoint_y": "setpoint_pitch",
    "setpoint_z": "setpoint_yaw",
    "thr": "throttle",
}

MIN_SAMPLES = 128


def _normalize_name(name):
    name = str(name).strip().lower()
    for ch in [" ", "-", "/", "\\", "(", ")", "[", "]", "{", "}", ":"]:
        name = name.replace(ch, "_")
    while "__" in name:
        name = name.replace("__", "_")
    return name.strip("_")


def _coerce_numeric(df):
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _dedupe_columns(df):
    cols = []
    seen = set()
    for col in df.columns:
        if col not in seen:
            cols.append(col)
            seen.add(col)
    return df.loc[:, cols]


def _sort_and_fix_time(df):
    if "time" not in df.columns:
        return df

    df = df.copy()
    df = df.dropna(subset=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    df = df.loc[~df["time"].duplicated(keep="first")].reset_index(drop=True)

    if len(df) >= 2:
        df["time"] = df["time"] - float(df["time"].iloc[0])

    return df


def _repair_time_if_needed(df):
    if "time" not in df.columns or df["time"].isna().all():
        df = df.copy()
        df["time"] = np.arange(len(df), dtype=float) * 0.001
        return df

    if len(df) < 3:
        return df

    dt = np.diff(df["time"].to_numpy(dtype=float))
    finite_dt = dt[np.isfinite(dt)]
    if len(finite_dt) == 0:
        df = df.copy()
        df["time"] = np.arange(len(df), dtype=float) * 0.001
        return df

    positive_dt = finite_dt[finite_dt > 0]
    if len(positive_dt) == 0:
        df = df.copy()
        df["time"] = np.arange(len(df), dtype=float) * 0.001
        return df

    median_dt = float(np.median(positive_dt))
    if median_dt <= 0 or not np.isfinite(median_dt):
        df = df.copy()
        df["time"] = np.arange(len(df), dtype=float) * 0.001

    return df


def _clip_throttle(df):
    if "throttle" in df.columns:
        df = df.copy()
        col = df["throttle"].to_numpy(dtype=float)
        finite = col[np.isfinite(col)]
        if len(finite) and np.nanmax(finite) > 5:
            col = (col - 1000.0) / 1000.0
        df["throttle"] = np.clip(col, 0.0, 1.0)
    return df


def _drop_sparse_rows(df):
    signal_cols = [c for c in REQUIRED_BASE_COLUMNS + AXIS_COLUMNS + OPTIONAL_COLUMNS if c in df.columns]
    if not signal_cols:
        return df
    return df.dropna(thresh=2, subset=signal_cols).reset_index(drop=True)


def _fill_small_gaps(df):
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return df
    df = df.copy()
    df[numeric_cols] = df[numeric_cols].interpolate(
        method="linear",
        limit=5,
        limit_direction="both",
    )
    return df


def _validate(df):
    if df is None or df.empty:
        return None
    if "time" not in df.columns:
        return None

    present_axes = [c for c in AXIS_COLUMNS if c in df.columns]
    if not present_axes:
        return None
    if len(df) < MIN_SAMPLES:
        return None

    time = df["time"].to_numpy(dtype=float)
    if np.isnan(time).all():
        return None

    if len(time) >= 2:
        dt = np.diff(time)
        good_dt = dt[np.isfinite(dt) & (dt > 0)]
        if len(good_dt) == 0:
            return None
        median_dt = float(np.median(good_dt))
        if median_dt <= 0 or not np.isfinite(median_dt):
            return None
        sampling_rate = 1.0 / median_dt
        if sampling_rate < 20 or sampling_rate > 10000:
            return None

    for axis in present_axes:
        series = df[axis].to_numpy(dtype=float)
        finite_ratio = np.isfinite(series).mean()
        if finite_ratio < 0.8:
            return None
        usable = series[np.isfinite(series)]
        if len(usable) == 0:
            return None
        if float(np.nanstd(usable)) < 1e-9:
            return None

    return df.reset_index(drop=True)


def parse_log(file_path):
    try:
        df = pd.read_csv(file_path)
        if df is None or df.empty:
            return None

        normalized = [_normalize_name(c) for c in df.columns]
        normalized = [COLUMN_ALIASES.get(c, c) for c in normalized]
        df.columns = normalized

        df = _dedupe_columns(df)
        df = _coerce_numeric(df)
        df = _drop_sparse_rows(df)
        df = _repair_time_if_needed(df)
        df = _sort_and_fix_time(df)
        df = _fill_small_gaps(df)
        df = _clip_throttle(df)

        ordered_cols = []
        for col in REQUIRED_BASE_COLUMNS + AXIS_COLUMNS + OPTIONAL_COLUMNS:
            if col in df.columns:
                ordered_cols.append(col)

        other_numeric = [c for c in df.columns if c not in ordered_cols and pd.api.types.is_numeric_dtype(df[c])]
        df = df[ordered_cols + other_numeric]

        return _validate(df)
    except Exception:
        return None
