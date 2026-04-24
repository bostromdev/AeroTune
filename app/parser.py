from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


CANONICAL_COLUMNS = [
    "time",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "setpoint_roll",
    "setpoint_pitch",
    "setpoint_yaw",
    "throttle",
]

ALIASES: Dict[str, List[str]] = {
    "time": ["time", "timestamp", "t", "time_s", "time_sec", "seconds", "time_ms", "time_us", "looptime"],
    "gyro_x": ["gyro_x", "gx", "gyro_roll", "roll_gyro", "roll"],
    "gyro_y": ["gyro_y", "gy", "gyro_pitch", "pitch_gyro", "pitch"],
    "gyro_z": ["gyro_z", "gz", "gyro_yaw", "yaw_gyro", "yaw"],
    "setpoint_roll": ["setpoint_roll", "roll_setpoint", "roll_sp", "rc_roll", "command_roll"],
    "setpoint_pitch": ["setpoint_pitch", "pitch_setpoint", "pitch_sp", "rc_pitch", "command_pitch"],
    "setpoint_yaw": ["setpoint_yaw", "yaw_setpoint", "yaw_sp", "rc_yaw", "command_yaw"],
    "throttle": ["throttle", "thr", "motor_throttle", "rc_throttle", "command_throttle"],
}


def _clean_name(name: object) -> str:
    value = str(name).strip().lower()
    for char in [" ", "-", "/", "\\", "(", ")", "[", "]", "{", "}", ":", ".", "%"]:
        value = value.replace(char, "_")
    while "__" in value:
        value = value.replace("__", "_")
    return value.strip("_")


def _first_existing(columns: Iterable[str], candidates: List[str]) -> Optional[str]:
    available = set(columns)
    compact = {c.replace("_", ""): c for c in columns}

    for candidate in candidates:
        if candidate in available:
            return candidate

        key = candidate.replace("_", "")
        if key in compact:
            return compact[key]

    return None


def _normalize_time(values: pd.Series, source_name: str = "time") -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)

    finite = np.isfinite(arr)
    if not finite.any():
        return arr

    first = arr[finite][0]
    arr = arr - first

    finite_arr = arr[np.isfinite(arr)]
    if len(finite_arr) < 2:
        return arr

    span = float(finite_arr[-1] - finite_arr[0])

    source = source_name.lower()
    if "us" in source or span > 1_000_000:
        return arr / 1_000_000.0

    if "ms" in source or span > 1_000:
        return arr / 1_000.0

    return arr


def _normalize_throttle(values: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]

    if len(finite) == 0:
        return np.zeros(len(arr), dtype=float)

    max_value = float(np.nanmax(finite))
    min_value = float(np.nanmin(finite))

    if max_value > 900:
        arr = (arr - 1000.0) / 1000.0
    elif max_value > 5:
        arr = arr / 100.0
    elif min_value < -0.05 and max_value <= 1.05:
        arr = (arr + 1.0) / 2.0

    return np.clip(arr, 0.0, 1.0)


def optimize_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        raise ValueError("CSV is empty.")

    df = raw.copy()
    df.columns = [_clean_name(c) for c in df.columns]

    output = pd.DataFrame()

    for canonical in CANONICAL_COLUMNS:
        source = _first_existing(df.columns, ALIASES[canonical])
        if source is None:
            continue

        if canonical == "time":
            output[canonical] = _normalize_time(df[source], source)
        elif canonical == "throttle":
            output[canonical] = _normalize_throttle(df[source])
        else:
            output[canonical] = pd.to_numeric(df[source], errors="coerce")

    if "time" not in output.columns:
        raise ValueError("Could not find a usable time column.")

    gyro_cols = [c for c in ["gyro_x", "gyro_y", "gyro_z"] if c in output.columns]
    if not gyro_cols:
        raise ValueError("Could not find usable gyro columns.")

    output = output.dropna(subset=["time"]).copy()
    output = output.sort_values("time").drop_duplicates(subset=["time"], keep="first").reset_index(drop=True)

    for col in output.columns:
        output[col] = pd.to_numeric(output[col], errors="coerce")
        if col != "time":
            output[col] = output[col].interpolate(limit=10, limit_direction="both")

    output = output.dropna(subset=gyro_cols).reset_index(drop=True)

    if len(output) < 128:
        raise ValueError("Log is too short after cleanup. Need at least 128 usable rows.")

    time = output["time"].to_numpy(dtype=float)
    dt = np.diff(time)
    good_dt = dt[np.isfinite(dt) & (dt > 0)]

    if len(good_dt) == 0:
        raise ValueError("Timestamps are invalid or not increasing.")

    sample_rate = 1.0 / float(np.median(good_dt))
    if sample_rate < 20 or sample_rate > 10000:
        raise ValueError(f"Unusual sample rate detected: {sample_rate:.1f} Hz.")

    for col in ["gyro_x", "gyro_y", "gyro_z", "setpoint_roll", "setpoint_pitch", "setpoint_yaw"]:
        if col not in output.columns:
            output[col] = 0.0

    if "throttle" not in output.columns:
        output["throttle"] = 0.0

    return output[CANONICAL_COLUMNS]


def parse_log(file_path: str | Path) -> Optional[pd.DataFrame]:
    try:
        raw = pd.read_csv(file_path)
        return optimize_dataframe(raw)
    except Exception:
        return None


def optimize_csv_file(file_path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(file_path)
    return optimize_dataframe(raw)
