from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

MIN_ROWS = 128
OPTIMIZED_COLUMNS = [
    "time",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "setpoint_roll",
    "setpoint_pitch",
    "setpoint_yaw",
    "throttle",
]

COLUMN_ALIASES: Dict[str, List[str]] = {
    "time": ["time", "timestamp", "t", "time_us", "time_ms", "seconds", "sec"],
    "gyro_x": ["gyro_x", "gx", "gyro_roll", "roll_gyro", "gyro roll", "gyro[0]", "roll"],
    "gyro_y": ["gyro_y", "gy", "gyro_pitch", "pitch_gyro", "gyro pitch", "gyro[1]", "pitch"],
    "gyro_z": ["gyro_z", "gz", "gyro_yaw", "yaw_gyro", "gyro yaw", "gyro[2]", "yaw"],
    "setpoint_roll": ["setpoint_roll", "roll_setpoint", "rc_roll", "roll_sp", "setpoint[0]", "setpoint roll"],
    "setpoint_pitch": ["setpoint_pitch", "pitch_setpoint", "rc_pitch", "pitch_sp", "setpoint[1]", "setpoint pitch"],
    "setpoint_yaw": ["setpoint_yaw", "yaw_setpoint", "rc_yaw", "yaw_sp", "setpoint[2]", "setpoint yaw"],
    "throttle": ["throttle", "thr", "motor_throttle", "rc_throttle", "throttle_percent"],
}


def _clean_column_name(name: object) -> str:
    value = str(name).strip().lower()
    for ch in [" ", "-", "/", "\\", "(", ")", "[", "]", "{", "}", ":", ".", "%"]:
        value = value.replace(ch, "_")
    while "__" in value:
        value = value.replace("__", "_")
    return value.strip("_")


def _alias_forms(name: str) -> set[str]:
    cleaned = _clean_column_name(name)
    return {cleaned, cleaned.replace("_", ""), name.strip().lower()}


def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    candidate_forms = set()
    for item in candidates:
        candidate_forms.update(_alias_forms(item))

    for col in df.columns:
        forms = _alias_forms(col)
        if forms & candidate_forms:
            return col
    return None


def _normalize_time(values: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(arr)
    if not np.any(mask):
        return arr
    first = arr[mask][0]
    arr = arr - first
    span = float(np.nanmax(arr) - np.nanmin(arr)) if len(arr) else 0.0
    if span > 1_000_000:
        arr = arr / 1_000_000.0
    elif span > 1_000:
        arr = arr / 1_000.0
    return arr


def _normalize_throttle(values: pd.Series, rows: int) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return np.zeros(rows, dtype=float)
    mx = float(np.nanmax(finite))
    mn = float(np.nanmin(finite))
    if mx > 500:
        arr = (arr - 1000.0) / 1000.0
    elif mx > 5:
        arr = arr / 100.0 if mx <= 100 else arr / mx
    elif mn < 0 or mx > 1:
        arr = (arr - mn) / max(mx - mn, 1e-9)
    return np.clip(arr, 0.0, 1.0)


def _read_csv(file_path: str | Path) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path)
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding="latin-1")


def optimize_csv_file(file_path: str | Path) -> pd.DataFrame:
    raw = _read_csv(file_path)
    if raw is None or raw.empty:
        raise ValueError("CSV is empty or unreadable.")

    raw.columns = [_clean_column_name(c) for c in raw.columns]
    out = pd.DataFrame()

    for target, aliases in COLUMN_ALIASES.items():
        source = _find_column(raw, aliases)
        if source is not None:
            out[target] = pd.to_numeric(raw[source], errors="coerce")

    if "time" not in out.columns:
        raise ValueError("Missing usable time column.")

    out["time"] = _normalize_time(out["time"])
    out = out.dropna(subset=["time"]).copy()
    out = out.sort_values("time").drop_duplicates(subset=["time"], keep="first")

    gyro_cols = [c for c in ["gyro_x", "gyro_y", "gyro_z"] if c in out.columns]
    if not gyro_cols:
        raise ValueError("Missing usable gyro columns.")

    out = out.dropna(subset=gyro_cols).copy()
    rows = len(out)

    for col in ["gyro_x", "gyro_y", "gyro_z", "setpoint_roll", "setpoint_pitch", "setpoint_yaw"]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").interpolate(limit=5, limit_direction="both").fillna(0.0)

    if "throttle" in out.columns:
        out["throttle"] = _normalize_throttle(out["throttle"], rows)
    else:
        out["throttle"] = np.zeros(rows, dtype=float)

    out = out[OPTIMIZED_COLUMNS].reset_index(drop=True)

    if len(out) < MIN_ROWS:
        raise ValueError(f"CSV is too short after cleaning. Need at least {MIN_ROWS} rows.")

    dt = np.diff(out["time"].to_numpy(dtype=float))
    if len(dt) == 0 or np.any(~np.isfinite(dt)) or np.any(dt <= 0):
        raise ValueError("Time column is not strictly increasing after optimization.")

    return out


def parse_log(file_path: str | Path) -> Optional[pd.DataFrame]:
    try:
        return optimize_csv_file(file_path)
    except Exception:
        return None
