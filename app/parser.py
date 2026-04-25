from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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

MIN_USABLE_ROWS = 128

# Canonical AeroTune column names mapped to common Betaflight / Blackbox Explorer exports.
# Betaflight arrays:
#   gyroADC[0..2]      roll/pitch/yaw gyro
#   setpoint[0..2]     roll/pitch/yaw setpoint
#   rcCommand[0..3]    roll/pitch/yaw/throttle command
ALIASES: Dict[str, List[str]] = {
    "time": [
        "time",
        "timestamp",
        "t",
        "seconds",
        "time_s",
        "time_sec",
        "time_ms",
        "time_us",
        "looptime",
    ],
    "gyro_x": [
        "gyro_x",
        "gx",
        "gyro_roll",
        "roll_gyro",
        "roll",
        "gyroadc_0",
        "gyro_adc_0",
        "gyro_0",
        "gyro_0_axis",
        "gyroscaled_0",
        "gyro_scaled_0",
        "gyro_unfilt_0",
        "gyrounfilt_0",
        "gyrodebug_0",
        "debug_0",
    ],
    "gyro_y": [
        "gyro_y",
        "gy",
        "gyro_pitch",
        "pitch_gyro",
        "pitch",
        "gyroadc_1",
        "gyro_adc_1",
        "gyro_1",
        "gyroscaled_1",
        "gyro_scaled_1",
        "gyro_unfilt_1",
        "gyrounfilt_1",
        "gyrodebug_1",
        "debug_1",
    ],
    "gyro_z": [
        "gyro_z",
        "gz",
        "gyro_yaw",
        "yaw_gyro",
        "yaw",
        "gyroadc_2",
        "gyro_adc_2",
        "gyro_2",
        "gyroscaled_2",
        "gyro_scaled_2",
        "gyro_unfilt_2",
        "gyrounfilt_2",
        "gyrodebug_2",
        "debug_2",
    ],
    "setpoint_roll": [
        "setpoint_roll",
        "roll_setpoint",
        "roll_sp",
        "sp_roll",
        "command_roll",
        "rc_roll",
        "setpoint_0",
        "setpoint0",
        "rccommand_0",
        "rc_command_0",
        "rccommand0",
    ],
    "setpoint_pitch": [
        "setpoint_pitch",
        "pitch_setpoint",
        "pitch_sp",
        "sp_pitch",
        "command_pitch",
        "rc_pitch",
        "setpoint_1",
        "setpoint1",
        "rccommand_1",
        "rc_command_1",
        "rccommand1",
    ],
    "setpoint_yaw": [
        "setpoint_yaw",
        "yaw_setpoint",
        "yaw_sp",
        "sp_yaw",
        "command_yaw",
        "rc_yaw",
        "setpoint_2",
        "setpoint2",
        "rccommand_2",
        "rc_command_2",
        "rccommand2",
    ],
    "throttle": [
        "throttle",
        "thr",
        "motor_throttle",
        "rc_throttle",
        "command_throttle",
        "rccommand_3",
        "rc_command_3",
        "rccommand3",
        "setpoint_3",
        "setpoint3",
        "motor_0",
        "motor_1",
    ],
}


def _clean_name(name: object) -> str:
    """Normalize messy CSV headers into stable snake_case-like names."""
    value = str(name).strip().lower()

    # Remove common unit wrappers while preserving array indexes.
    value = value.replace('"', "").replace("'", "")
    value = value.replace("[", "_").replace("]", "")
    value = value.replace("(", "_").replace(")", "")
    value = value.replace("{", "_").replace("}", "")
    value = value.replace("/", "_").replace("\\", "_")
    value = value.replace("-", "_").replace(" ", "_")
    value = value.replace(".", "_").replace(":", "_").replace("%", "")

    while "__" in value:
        value = value.replace("__", "_")

    return value.strip("_")


def _dedupe_columns(columns: Sequence[str]) -> List[str]:
    seen: Dict[str, int] = {}
    out: List[str] = []

    for col in columns:
        base = _clean_name(col)
        if base not in seen:
            seen[base] = 0
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base}_{seen[base]}")

    return out


def _compact(name: str) -> str:
    return _clean_name(name).replace("_", "")


def _first_existing(columns: Iterable[str], candidates: List[str]) -> Optional[str]:
    cols = list(columns)
    available = set(cols)
    compact_lookup = {_compact(c): c for c in cols}

    for candidate in candidates:
        clean = _clean_name(candidate)

        if clean in available:
            return clean

        key = _compact(clean)
        if key in compact_lookup:
            return compact_lookup[key]

    return None


def _line_looks_like_blackbox_header(fields: List[str]) -> bool:
    cleaned = [_clean_name(f) for f in fields]
    compacted = {_compact(f) for f in cleaned}

    has_time = "time" in compacted or "timestamp" in compacted
    has_loop = "loopiteration" in compacted
    has_gyro = any(
        key in compacted
        for key in (
            "gyroadc0",
            "gyroadc1",
            "gyroadc2",
            "gyro0",
            "gyro1",
            "gyro2",
            "gyrox",
            "gyroy",
            "gyroz",
            "gyroroll",
            "gyropitch",
            "gyroyaw",
        )
    )
    has_setpoint_or_rc = any(
        key in compacted
        for key in (
            "setpoint0",
            "setpoint1",
            "setpoint2",
            "rccommand0",
            "rccommand1",
            "rccommand2",
            "setpointroll",
            "setpointpitch",
            "setpointyaw",
        )
    )

    # Raw Blackbox exports usually have loopIteration + time + gyroADC.
    # AeroTune-ready CSVs usually have time + gyro_x/y/z.
    return has_time and has_gyro and (has_loop or has_setpoint_or_rc or len(fields) >= 6)


def _find_header_row(file_path: str | Path, max_scan_lines: int = 5000) -> int:
    """Find the real CSV header in files with Betaflight metadata at the top."""
    path = Path(file_path)

    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.reader(handle)
        for index, fields in enumerate(reader):
            if index >= max_scan_lines:
                break

            if not fields:
                continue

            if _line_looks_like_blackbox_header(fields):
                return index

    # If no metadata header is found, assume row 0. This keeps normal CSVs working.
    return 0


def read_blackbox_csv(file_path: str | Path) -> pd.DataFrame:
    """Read normal CSVs and raw Betaflight Blackbox Explorer CSV exports."""
    header_row = _find_header_row(file_path)

    try:
        df = pd.read_csv(
            file_path,
            skiprows=header_row,
            engine="python",
            on_bad_lines="skip",
        )
    except TypeError:
        # Compatibility with older pandas versions.
        df = pd.read_csv(
            file_path,
            skiprows=header_row,
            engine="python",
            error_bad_lines=False,  # type: ignore[call-arg]
            warn_bad_lines=False,  # type: ignore[call-arg]
        )

    if df is None or df.empty:
        raise ValueError("CSV is empty or no usable data rows were found.")

    df.columns = _dedupe_columns(df.columns)
    return df


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

    # Betaflight Blackbox "time" is usually microseconds.
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

    # RC command throttle usually ranges 1000..2000.
    if max_value > 900:
        arr = (arr - 1000.0) / 1000.0
    # Percent-style throttle.
    elif max_value > 5:
        arr = arr / 100.0
    # Some normalized controls can be -1..1.
    elif min_value < -0.05 and max_value <= 1.05:
        arr = (arr + 1.0) / 2.0

    return np.clip(arr, 0.0, 1.0)


def _numeric_series(df: pd.DataFrame, source: str) -> pd.Series:
    if source not in df.columns:
        return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)

    return pd.to_numeric(df[source], errors="coerce")


def _drop_non_data_rows(df: pd.DataFrame) -> pd.DataFrame:
    # Some exports can contain repeated headers or metadata fragments later in the file.
    time_source = _first_existing(df.columns, ALIASES["time"])
    if time_source is None:
        return df

    time_numeric = pd.to_numeric(df[time_source], errors="coerce")
    return df.loc[time_numeric.notna()].copy()


def optimize_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert supported Blackbox CSV variants into AeroTune canonical columns."""
    if raw is None or raw.empty:
        raise ValueError("CSV is empty.")

    df = raw.copy()
    df.columns = _dedupe_columns(df.columns)
    df = _drop_non_data_rows(df)

    if df.empty:
        raise ValueError("No numeric data rows found after reading CSV.")

    output = pd.DataFrame(index=df.index)

    for canonical in CANONICAL_COLUMNS:
        source = _first_existing(df.columns, ALIASES[canonical])

        if source is None:
            continue

        if canonical == "time":
            output[canonical] = _normalize_time(df[source], source)
        elif canonical == "throttle":
            output[canonical] = _normalize_throttle(df[source])
        else:
            output[canonical] = _numeric_series(df, source)

    if "time" not in output.columns:
        raise ValueError(
            "Could not find a usable time column. Expected time, timestamp, time_us, or Betaflight time."
        )

    gyro_cols = [c for c in ["gyro_x", "gyro_y", "gyro_z"] if c in output.columns]
    if not gyro_cols:
        raise ValueError(
            "Could not find usable gyro columns. Expected gyro_x/y/z or Betaflight gyroADC[0..2]."
        )

    output = output.dropna(subset=["time"]).copy()
    output = output.sort_values("time").drop_duplicates(subset=["time"], keep="first")
    output = output.reset_index(drop=True)

    for col in output.columns:
        output[col] = pd.to_numeric(output[col], errors="coerce")

    # Interpolate small gaps but never create fake gyro if an axis is truly missing.
    for col in list(output.columns):
        if col != "time":
            output[col] = output[col].interpolate(limit=10, limit_direction="both")

    gyro_cols = [c for c in ["gyro_x", "gyro_y", "gyro_z"] if c in output.columns]
    output = output.dropna(subset=gyro_cols).reset_index(drop=True)

    if len(output) < MIN_USABLE_ROWS:
        raise ValueError(f"Log is too short after cleanup. Need at least {MIN_USABLE_ROWS} usable rows.")

    time = output["time"].to_numpy(dtype=float)
    dt = np.diff(time)
    good_dt = dt[np.isfinite(dt) & (dt > 0)]

    if len(good_dt) == 0:
        raise ValueError("Timestamps are invalid or not increasing.")

    sample_rate = 1.0 / float(np.median(good_dt))
    if sample_rate < 10 or sample_rate > 20000:
        raise ValueError(f"Unusual sample rate detected: {sample_rate:.1f} Hz.")

    # Fill optional fields with zeros so the analyzer/UI always gets stable columns.
    for col in ["gyro_x", "gyro_y", "gyro_z"]:
        if col not in output.columns:
            output[col] = 0.0

    for col in ["setpoint_roll", "setpoint_pitch", "setpoint_yaw"]:
        if col not in output.columns:
            output[col] = 0.0

    if "throttle" not in output.columns:
        output["throttle"] = 0.0

    # Final cleanup: no inf, stable column order, numeric only.
    output = output.replace([np.inf, -np.inf], np.nan)

    for col in CANONICAL_COLUMNS:
        output[col] = pd.to_numeric(output[col], errors="coerce").fillna(0.0)

    return output[CANONICAL_COLUMNS]


def parse_log(file_path: str | Path) -> Optional[pd.DataFrame]:
    """Parse a user upload for analysis. Returns None instead of raising for API friendliness."""
    try:
        raw = read_blackbox_csv(file_path)
        return optimize_dataframe(raw)
    except Exception:
        return None


def optimize_csv_file(file_path: str | Path) -> pd.DataFrame:
    """Parse and normalize a CSV. Raises clear errors for the optimizer endpoint."""
    raw = read_blackbox_csv(file_path)
    return optimize_dataframe(raw)
