from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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

REQUIRED_COLUMNS = ["time", "gyro_x", "gyro_y", "gyro_z"]
OPTIONAL_COLUMNS = ["setpoint_roll", "setpoint_pitch", "setpoint_yaw", "throttle"]

MIN_USABLE_ROWS = 128
MIN_REASONABLE_SAMPLE_RATE_HZ = 10.0
MAX_REASONABLE_SAMPLE_RATE_HZ = 20_000.0

# Canonical AeroTune column names mapped to common Betaflight / Blackbox Explorer exports.
# Betaflight arrays:
#   gyroADC[0..2]      roll/pitch/yaw gyro
#   gyro_scaled[0..2]  roll/pitch/yaw gyro
#   gyro_unfilt[0..2]  roll/pitch/yaw gyro
#   setpoint[0..2]     roll/pitch/yaw setpoint
#   rcCommand[0..3]    roll/pitch/yaw/throttle command
ALIASES: Dict[str, List[str]] = {
    "time": [
        "time",
        "timestamp",
        "t",
        "seconds",
        "sec",
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


@dataclass
class ParseResult:
    df: Optional[pd.DataFrame]
    report: Dict[str, Any]


def _empty_report(file_path: str | Path) -> Dict[str, Any]:
    path = Path(file_path)
    return {
        "ok": False,
        "filename": path.name,
        "file_type": path.suffix.lower() or "unknown",
        "stage": "not_started",
        "message": "Parser has not started.",
        "error_code": None,
        "header_row": None,
        "metadata_rows_skipped": 0,
        "repeated_header_rows_removed": 0,
        "raw_columns_count": 0,
        "raw_columns_preview": [],
        "detected_columns": {
            "time": None,
            "gyro": {"roll": None, "pitch": None, "yaw": None},
            "setpoint": {"roll": None, "pitch": None, "yaw": None},
            "throttle": None,
        },
        "missing_required": [],
        "missing_optional": [],
        "sample_rate_hz": None,
        "duration_seconds": None,
        "usable_rows": 0,
        "total_rows_read": 0,
        "warnings": [],
        "suggestions": [],
    }


def _clean_name(name: object) -> str:
    """Normalize messy CSV headers into stable snake_case-like names."""
    value = str(name).strip().lower()
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
            "gyroscaled0",
            "gyroscaled1",
            "gyroscaled2",
            "gyrounfilt0",
            "gyrounfilt1",
            "gyrounfilt2",
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

    # Raw Blackbox CSV exports usually have loopIteration + time + gyroADC.
    # AeroTune-ready CSVs usually have time + gyro_x/y/z.
    return has_time and has_gyro and (has_loop or has_setpoint_or_rc or len(fields) >= 6)


def _find_header_row(file_path: str | Path, max_scan_lines: int = 5000) -> Tuple[int, int]:
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
                return index, index

    # If no metadata header is found, assume row 0. This keeps normal CSVs working.
    return 0, 0


def _count_repeated_header_rows(df: pd.DataFrame, time_source: Optional[str]) -> int:
    if time_source is None or time_source not in df.columns:
        return 0

    raw_values = df[time_source].astype(str).str.strip().str.lower()
    return int(raw_values.isin({"time", "timestamp", "time_us", "time_ms", "seconds"}).sum())


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


def _detect_sources(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    return {
        canonical: _first_existing(df.columns, ALIASES[canonical])
        for canonical in CANONICAL_COLUMNS
    }


def _format_detected_columns(sources: Dict[str, Optional[str]]) -> Dict[str, Any]:
    return {
        "time": sources.get("time"),
        "gyro": {
            "roll": sources.get("gyro_x"),
            "pitch": sources.get("gyro_y"),
            "yaw": sources.get("gyro_z"),
        },
        "setpoint": {
            "roll": sources.get("setpoint_roll"),
            "pitch": sources.get("setpoint_pitch"),
            "yaw": sources.get("setpoint_yaw"),
        },
        "throttle": sources.get("throttle"),
    }


def read_blackbox_csv_with_report(file_path: str | Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Read normal CSVs and raw Betaflight Blackbox Explorer CSV exports."""
    report = _empty_report(file_path)
    path = Path(file_path)

    if path.suffix.lower() != ".csv":
        report.update(
            {
                "stage": "file_type",
                "message": "AeroTune V1.1/V1.2 only accepts CSV exports. Raw .bbl/.bfl support is planned for V1.3.",
                "error_code": "not_csv",
                "suggestions": [
                    "Export the log as CSV from Betaflight Blackbox Explorer.",
                    "Raw .bbl/.bfl upload support belongs to the V1.3 converter step.",
                ],
            }
        )
        raise ValueError(report["message"])

    report["stage"] = "header_scan"
    header_row, metadata_rows = _find_header_row(path)
    report["header_row"] = int(header_row)
    report["metadata_rows_skipped"] = int(metadata_rows)

    try:
        df = pd.read_csv(path, skiprows=header_row, engine="python", on_bad_lines="skip")
    except TypeError:
        # Compatibility with older pandas versions.
        df = pd.read_csv(
            path,
            skiprows=header_row,
            engine="python",
            error_bad_lines=False,  # type: ignore[call-arg]
            warn_bad_lines=False,  # type: ignore[call-arg]
        )
    except Exception as exc:
        report.update(
            {
                "stage": "csv_read",
                "message": f"CSV could not be read: {exc}",
                "error_code": "csv_read_failed",
                "suggestions": [
                    "Re-export the log from Betaflight Blackbox Explorer as CSV.",
                    "Try the CSV optimizer after confirming the file opens normally.",
                ],
            }
        )
        raise ValueError(report["message"]) from exc

    if df is None or df.empty:
        report.update(
            {
                "stage": "csv_read",
                "message": "CSV is empty or no usable data rows were found.",
                "error_code": "empty_csv",
                "suggestions": ["Check that the uploaded file is a real Betaflight CSV export."],
            }
        )
        raise ValueError(report["message"])

    df.columns = _dedupe_columns(df.columns)
    report["total_rows_read"] = int(len(df))
    report["raw_columns_count"] = int(len(df.columns))
    report["raw_columns_preview"] = [str(c) for c in list(df.columns[:30])]

    return df, report


def read_blackbox_csv(file_path: str | Path) -> pd.DataFrame:
    df, _report = read_blackbox_csv_with_report(file_path)
    return df


def optimize_dataframe_with_report(raw: pd.DataFrame, report: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Convert supported Blackbox CSV variants into AeroTune canonical columns."""
    if report is None:
        report = {
            "ok": False,
            "stage": "optimize",
            "message": "Optimizing dataframe.",
            "warnings": [],
            "suggestions": [],
        }

    if raw is None or raw.empty:
        report.update(
            {
                "stage": "raw_dataframe",
                "message": "CSV is empty.",
                "error_code": "empty_dataframe",
            }
        )
        raise ValueError(report["message"])

    df = raw.copy()
    df.columns = _dedupe_columns(df.columns)

    sources_before_drop = _detect_sources(df)
    report["detected_columns"] = _format_detected_columns(sources_before_drop)
    report["missing_required"] = [col for col in REQUIRED_COLUMNS if sources_before_drop.get(col) is None]
    report["missing_optional"] = [col for col in OPTIONAL_COLUMNS if sources_before_drop.get(col) is None]

    if "time" in report["missing_required"]:
        report.update(
            {
                "stage": "column_detection",
                "message": "Missing required time column.",
                "error_code": "missing_time_column",
                "suggestions": [
                    "Expected time, timestamp, time_us, time_ms, or Betaflight time.",
                    "Re-export from Betaflight Blackbox Explorer with the time column included.",
                ],
            }
        )
        raise ValueError(report["message"])

    missing_gyro = [col for col in ["gyro_x", "gyro_y", "gyro_z"] if sources_before_drop.get(col) is None]
    if missing_gyro:
        report.update(
            {
                "stage": "column_detection",
                "message": f"Missing required gyro columns: {', '.join(missing_gyro)}.",
                "error_code": "missing_gyro_columns",
                "suggestions": [
                    "Expected gyro_x/gyro_y/gyro_z or Betaflight gyroADC[0..2].",
                    "Make sure the CSV export includes gyro traces.",
                ],
            }
        )
        raise ValueError(report["message"])

    if report["missing_optional"]:
        report["warnings"].append(
            "Some optional columns were missing. AeroTune will fill missing setpoint/throttle fields with safe zeros."
        )

    time_source = sources_before_drop.get("time")
    repeated_header_rows = _count_repeated_header_rows(df, time_source)
    report["repeated_header_rows_removed"] = int(repeated_header_rows)
    if repeated_header_rows > 0:
        report["warnings"].append(
            f"Removed {repeated_header_rows} repeated header/metadata rows inside the CSV."
        )

    # Remove non-data rows using the detected time column.
    time_numeric = pd.to_numeric(df[time_source], errors="coerce") if time_source else pd.Series([], dtype=float)
    before_rows = len(df)
    df = df.loc[time_numeric.notna()].copy()
    removed_non_data = before_rows - len(df)

    if removed_non_data > 0 and repeated_header_rows == 0:
        report["warnings"].append(
            f"Removed {removed_non_data} non-numeric metadata rows after the header."
        )

    if df.empty:
        report.update(
            {
                "stage": "row_cleanup",
                "message": "No numeric data rows found after reading CSV.",
                "error_code": "no_numeric_rows",
                "suggestions": [
                    "Confirm this is a Betaflight Blackbox CSV export, not a raw .bbl/.bfl file.",
                ],
            }
        )
        raise ValueError(report["message"])

    sources = _detect_sources(df)
    report["detected_columns"] = _format_detected_columns(sources)

    output = pd.DataFrame(index=df.index)

    for canonical in CANONICAL_COLUMNS:
        source = sources.get(canonical)
        if source is None:
            continue

        if canonical == "time":
            output[canonical] = _normalize_time(df[source], source)
        elif canonical == "throttle":
            output[canonical] = _normalize_throttle(df[source])
        else:
            output[canonical] = _numeric_series(df, source)

    output = output.dropna(subset=["time"]).copy()
    output = output.sort_values("time").drop_duplicates(subset=["time"], keep="first")
    output = output.reset_index(drop=True)

    for col in output.columns:
        output[col] = pd.to_numeric(output[col], errors="coerce")

    for col in list(output.columns):
        if col != "time":
            output[col] = output[col].interpolate(limit=10, limit_direction="both")

    gyro_cols = [c for c in ["gyro_x", "gyro_y", "gyro_z"] if c in output.columns]
    output = output.dropna(subset=gyro_cols).reset_index(drop=True)

    report["usable_rows"] = int(len(output))

    if len(output) < MIN_USABLE_ROWS:
        report.update(
            {
                "stage": "row_cleanup",
                "message": f"Log is too short after cleanup. Need at least {MIN_USABLE_ROWS} usable rows.",
                "error_code": "log_too_short",
                "suggestions": [
                    "Use a longer Blackbox recording.",
                    "Try 30–120 seconds of normal flight with turns, throttle changes, and some propwash recovery.",
                ],
            }
        )
        raise ValueError(report["message"])

    time = output["time"].to_numpy(dtype=float)
    dt = np.diff(time)
    good_dt = dt[np.isfinite(dt) & (dt > 0)]

    if len(good_dt) == 0:
        report.update(
            {
                "stage": "time_validation",
                "message": "Timestamps are invalid or not increasing.",
                "error_code": "bad_timestamps",
                "suggestions": [
                    "Re-export the CSV from Betaflight Blackbox Explorer.",
                    "Make sure the time column was not edited manually.",
                ],
            }
        )
        raise ValueError(report["message"])

    sample_rate = 1.0 / float(np.median(good_dt))
    duration = float(time[-1] - time[0]) if len(time) > 1 else 0.0

    report["sample_rate_hz"] = round(float(sample_rate), 2)
    report["duration_seconds"] = round(float(duration), 3)

    if sample_rate < MIN_REASONABLE_SAMPLE_RATE_HZ or sample_rate > MAX_REASONABLE_SAMPLE_RATE_HZ:
        report.update(
            {
                "stage": "time_validation",
                "message": f"Unusual sample rate detected: {sample_rate:.1f} Hz.",
                "error_code": "sample_rate_weird",
                "suggestions": [
                    "Check whether the time column is in seconds, milliseconds, or microseconds.",
                    "Re-export the log from Betaflight Blackbox Explorer and try again.",
                ],
            }
        )
        raise ValueError(report["message"])

    if duration < 1.0:
        report["warnings"].append(
            "Log duration is under 1 second. Analysis may be weak even if the row count is valid."
        )

    # Fill optional fields with zeros so the analyzer/UI always gets stable columns.
    for col in ["gyro_x", "gyro_y", "gyro_z"]:
        if col not in output.columns:
            output[col] = 0.0

    for col in ["setpoint_roll", "setpoint_pitch", "setpoint_yaw"]:
        if col not in output.columns:
            output[col] = 0.0

    if "throttle" not in output.columns:
        output["throttle"] = 0.0

    output = output.replace([np.inf, -np.inf], np.nan)

    for col in CANONICAL_COLUMNS:
        output[col] = pd.to_numeric(output[col], errors="coerce").fillna(0.0)

    output = output[CANONICAL_COLUMNS]
    report.update(
        {
            "ok": True,
            "stage": "complete",
            "message": "CSV parsed successfully.",
            "error_code": None,
            "optimized_columns": CANONICAL_COLUMNS,
            "usable_rows": int(len(output)),
        }
    )

    return output, report


def optimize_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    df, _report = optimize_dataframe_with_report(raw)
    return df


def parse_log_with_report(file_path: str | Path) -> ParseResult:
    """Parse a user upload and always return a parser report."""
    report = _empty_report(file_path)

    try:
        raw, report = read_blackbox_csv_with_report(file_path)
        df, report = optimize_dataframe_with_report(raw, report)
        return ParseResult(df=df, report=report)

    except Exception as exc:
        if not report.get("message") or report.get("stage") == "not_started":
            report.update(
                {
                    "stage": "unexpected",
                    "message": str(exc),
                    "error_code": "unexpected_parser_error",
                    "suggestions": [
                        "Try re-exporting the log as CSV from Betaflight Blackbox Explorer.",
                        "If it still fails, send the CSV so AeroTune can learn this format.",
                    ],
                }
            )

        report["ok"] = False
        return ParseResult(df=None, report=report)


def parse_log(file_path: str | Path) -> Optional[pd.DataFrame]:
    """Backwards-compatible parser used by older app code."""
    result = parse_log_with_report(file_path)
    return result.df


def optimize_csv_file_with_report(file_path: str | Path) -> ParseResult:
    """Parse and normalize a CSV while returning detailed diagnostics."""
    return parse_log_with_report(file_path)


def optimize_csv_file(file_path: str | Path) -> pd.DataFrame:
    """Parse and normalize a CSV. Raises clear errors for the optimizer endpoint."""
    result = parse_log_with_report(file_path)
    if result.df is None:
        raise ValueError(result.report.get("message", "Could not optimize CSV."))
    return result.df
