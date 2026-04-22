from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


MIN_ROWS = 128


COLUMN_ALIASES: Dict[str, List[str]] = {
    "time": ["time", "timestamp", "t"],

    # gyro aliases
    "gyro_roll": ["gyro_roll", "roll_gyro", "gyro_x", "gx", "roll"],
    "gyro_pitch": ["gyro_pitch", "pitch_gyro", "gyro_y", "gy", "pitch"],
    "gyro_yaw": ["gyro_yaw", "yaw_gyro", "gyro_z", "gz", "yaw"],

    # setpoint aliases
    "roll_setpoint": ["roll_setpoint", "setpoint_roll", "rc_roll", "roll_sp"],
    "pitch_setpoint": ["pitch_setpoint", "setpoint_pitch", "rc_pitch", "pitch_sp"],
    "yaw_setpoint": ["yaw_setpoint", "setpoint_yaw", "rc_yaw", "yaw_sp"],
}


def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    columns = set(df.columns)
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _normalize_time_to_seconds(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values - values[0]
    span = float(values[-1] - values[0]) if len(values) > 1 else 0.0

    # likely microseconds
    if span > 1e6:
        return values / 1_000_000.0

    # likely milliseconds
    if span > 1e3:
        return values / 1_000.0

    return values


def parse_log(file_path: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(file_path)
    except Exception:
        return None

    if df is None or df.empty:
        return None

    df.columns = [str(c).strip().lower() for c in df.columns]

    normalized = pd.DataFrame()

    for target, aliases in COLUMN_ALIASES.items():
        source = _find_column(df, aliases)
        if source is not None:
            normalized[target] = pd.to_numeric(df[source], errors="coerce")

    if "time" not in normalized.columns:
        return None

    gyro_targets = ["gyro_roll", "gyro_pitch", "gyro_yaw"]
    available_gyro = [c for c in gyro_targets if c in normalized.columns]
    if not available_gyro:
        return None

    normalized = normalized.dropna(subset=["time"]).copy()
    if normalized.empty:
        return None

    normalized["time"] = _normalize_time_to_seconds(normalized["time"].to_numpy(dtype=float))

    for col in available_gyro:
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce")

    normalized = normalized.dropna(subset=available_gyro).copy()
    if len(normalized) < MIN_ROWS:
        return None

    time_values = normalized["time"].to_numpy(dtype=float)
    if len(time_values) < 2:
        return None

    dt = np.diff(time_values)
    if np.any(~np.isfinite(dt)) or np.any(dt <= 0):
        return None

    standardized = pd.DataFrame({
        "time": normalized["time"].to_numpy(dtype=float),
        "gyro_x": normalized["gyro_roll"].to_numpy(dtype=float) if "gyro_roll" in normalized.columns else np.zeros(len(normalized)),
        "gyro_y": normalized["gyro_pitch"].to_numpy(dtype=float) if "gyro_pitch" in normalized.columns else np.zeros(len(normalized)),
        "gyro_z": normalized["gyro_yaw"].to_numpy(dtype=float) if "gyro_yaw" in normalized.columns else np.zeros(len(normalized)),
    })

    standardized["setpoint_roll"] = (
        pd.to_numeric(normalized["roll_setpoint"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if "roll_setpoint" in normalized.columns
        else np.zeros(len(normalized))
    )
    standardized["setpoint_pitch"] = (
        pd.to_numeric(normalized["pitch_setpoint"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if "pitch_setpoint" in normalized.columns
        else np.zeros(len(normalized))
    )
    standardized["setpoint_yaw"] = (
        pd.to_numeric(normalized["yaw_setpoint"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if "yaw_setpoint" in normalized.columns
        else np.zeros(len(normalized))
    )

    return standardized