from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


MIN_ROWS = 128

COLUMN_ALIASES: Dict[str, List[str]] = {
    "time": ["time", "timestamp", "t", "time_us", "time_ms", "seconds"],
    "gyro_x": ["gyro_x", "gyro_roll", "roll_gyro", "gx", "roll"],
    "gyro_y": ["gyro_y", "gyro_pitch", "pitch_gyro", "gy", "pitch"],
    "gyro_z": ["gyro_z", "gyro_yaw", "yaw_gyro", "gz", "yaw"],
    "setpoint_roll": ["setpoint_roll", "roll_setpoint", "rc_command_roll", "rc_roll", "roll_sp"],
    "setpoint_pitch": ["setpoint_pitch", "pitch_setpoint", "rc_command_pitch", "rc_pitch", "pitch_sp"],
    "setpoint_yaw": ["setpoint_yaw", "yaw_setpoint", "rc_command_yaw", "rc_yaw", "yaw_sp"],
    "throttle": ["throttle", "thr", "motor_throttle"],
}


def _clean_column_name(name: object) -> str:
    cleaned = str(name).strip().lower()
    for ch in [" ", "-", "/", "\\", "(", ")", "[", "]", "{", "}", ":", ".", "%"]:
        cleaned = cleaned.replace(ch, "_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _normalize_time(values: np.ndarray) -> np.ndarray:
    time = np.asarray(values, dtype=float)

    if len(time) == 0:
        return time

    time = time - time[0]

    if len(time) < 2:
        return time

    span = float(time[-1] - time[0])

    if span > 1_000_000:
        return time / 1_000_000.0

    if span > 1_000:
        return time / 1_000.0

    return time


def _normalize_throttle(values: np.ndarray) -> np.ndarray:
    throttle = np.asarray(values, dtype=float)
    finite = throttle[np.isfinite(throttle)]

    if finite.size == 0:
        return np.zeros_like(throttle)

    if float(np.nanmax(finite)) > 5.0:
        throttle = (throttle - 1000.0) / 1000.0

    return np.clip(throttle, 0.0, 1.0)


def parse_log(file_path: str) -> Optional[pd.DataFrame]:
    try:
        raw = pd.read_csv(file_path)
    except Exception:
        return None

    if raw is None or raw.empty:
        return None

    raw = raw.copy()
    raw.columns = [_clean_column_name(c) for c in raw.columns]

    out = pd.DataFrame()

    for target, aliases in COLUMN_ALIASES.items():
        source = _find_column(raw, aliases)
        if source is not None:
            out[target] = pd.to_numeric(raw[source], errors="coerce")

    if "time" not in out.columns:
        return None

    gyro_cols = [c for c in ("gyro_x", "gyro_y", "gyro_z") if c in out.columns]
    if not gyro_cols:
        return None

    out = out.dropna(subset=["time"]).copy()

    for col in gyro_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=gyro_cols).copy()

    if len(out) < MIN_ROWS:
        return None

    out = out.sort_values("time").drop_duplicates(subset=["time"], keep="first").reset_index(drop=True)
    out["time"] = _normalize_time(out["time"].to_numpy(dtype=float))

    dt = np.diff(out["time"].to_numpy(dtype=float))
    if len(dt) == 0 or np.any(~np.isfinite(dt)) or np.any(dt <= 0):
        return None

    for axis, setpoint_col in [
        ("gyro_x", "setpoint_roll"),
        ("gyro_y", "setpoint_pitch"),
        ("gyro_z", "setpoint_yaw"),
    ]:
        if axis in out.columns and setpoint_col not in out.columns:
            out[setpoint_col] = 0.0

    if "throttle" in out.columns:
        out["throttle"] = _normalize_throttle(out["throttle"].fillna(0.0).to_numpy(dtype=float))

    keep_order = [
        "time",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "setpoint_roll",
        "setpoint_pitch",
        "setpoint_yaw",
        "throttle",
    ]

    existing = [c for c in keep_order if c in out.columns]
    return out[existing].reset_index(drop=True)
