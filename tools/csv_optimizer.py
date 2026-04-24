from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


KEEP_TARGETS: Dict[str, List[str]] = {
    "time": ["time", "timestamp", "t", "looptime"],
    "gyro_x": ["gyro_x", "gyro_roll", "roll_gyro", "gx", "gyroADC[0]", "gyroadc[0]", "roll"],
    "gyro_y": ["gyro_y", "gyro_pitch", "pitch_gyro", "gy", "gyroADC[1]", "gyroadc[1]", "pitch"],
    "gyro_z": ["gyro_z", "gyro_yaw", "yaw_gyro", "gz", "gyroADC[2]", "gyroadc[2]", "yaw"],
    "setpoint_roll": ["setpoint_roll", "roll_setpoint", "rc_roll", "roll_sp", "setpoint[0]"],
    "setpoint_pitch": ["setpoint_pitch", "pitch_setpoint", "rc_pitch", "pitch_sp", "setpoint[1]"],
    "setpoint_yaw": ["setpoint_yaw", "yaw_setpoint", "rc_yaw", "yaw_sp", "setpoint[2]"],
    "throttle": ["throttle", "thr", "rc_throttle", "motor_throttle"],
}


def normalize_column_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def find_column(columns: List[str], aliases: List[str]) -> Optional[str]:
    normalized_map = {normalize_column_name(c): c for c in columns}
    normalized_aliases = [normalize_column_name(a) for a in aliases]

    for alias in normalized_aliases:
        if alias in normalized_map:
            return normalized_map[alias]

    for normalized, original in normalized_map.items():
        if any(alias in normalized for alias in normalized_aliases):
            return original

    return None


def normalize_time_to_seconds(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    values = values - values.iloc[0]
    span = float(values.iloc[-1] - values.iloc[0]) if len(values) > 1 else 0.0

    if span > 1_000_000:
        return values / 1_000_000.0

    if span > 1_000:
        return values / 1_000.0

    return values


def choose_best_window(df: pd.DataFrame, duration_sec: float) -> pd.DataFrame:
    if "time" not in df.columns or len(df) < 2:
        return df

    total_duration = float(df["time"].iloc[-1] - df["time"].iloc[0])
    if total_duration <= duration_sec:
        return df

    if "throttle" not in df.columns:
        start = total_duration * 0.10
        end = start + duration_sec
        return df[(df["time"] >= start) & (df["time"] <= end)].copy()

    # Pick the window with the most throttle activity. That usually captures real flight movement.
    step = max(duration_sec / 4.0, 5.0)
    best_score = -1.0
    best_start = 0.0

    starts = np.arange(0, max(total_duration - duration_sec, 0.0), step)
    if len(starts) == 0:
        starts = np.array([0.0])

    for start in starts:
        end = start + duration_sec
        window = df[(df["time"] >= start) & (df["time"] <= end)]
        if len(window) < 100:
            continue

        throttle = pd.to_numeric(window["throttle"], errors="coerce").dropna()
        if throttle.empty:
            score = float(len(window))
        else:
            score = float(throttle.std() + throttle.diff().abs().mean() * 4.0)

        if score > best_score:
            best_score = score
            best_start = float(start)

    best_end = best_start + duration_sec
    return df[(df["time"] >= best_start) & (df["time"] <= best_end)].copy()


def downsample_to_rate(df: pd.DataFrame, target_hz: float) -> pd.DataFrame:
    if "time" not in df.columns or len(df) < 2:
        return df

    duration = float(df["time"].iloc[-1] - df["time"].iloc[0])
    if duration <= 0:
        return df

    current_hz = len(df) / duration
    if current_hz <= target_hz:
        return df

    stride = max(1, int(round(current_hz / target_hz)))
    return df.iloc[::stride].copy()


def optimize_csv(
    input_path: Path,
    output_path: Path,
    max_duration_sec: float = 60.0,
    target_hz: float = 500.0,
) -> dict:
    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError("Input CSV is empty.")

    out = pd.DataFrame()
    columns = list(df.columns)

    for target, aliases in KEEP_TARGETS.items():
        source = find_column(columns, aliases)
        if source is not None:
            out[target] = pd.to_numeric(df[source], errors="coerce")

    if "time" not in out.columns:
        raise ValueError("Could not find a usable time column.")

    gyro_cols = [c for c in ["gyro_x", "gyro_y", "gyro_z"] if c in out.columns]
    if not gyro_cols:
        raise ValueError("Could not find gyro columns. Need at least one gyro axis.")

    out = out.dropna(subset=["time"]).copy()
    out["time"] = normalize_time_to_seconds(out["time"])

    out = out.dropna(subset=gyro_cols).copy()
    out = out.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)

    # Fill missing setpoints/throttle with zero if not present so AeroTune can still parse it.
    for col in ["gyro_x", "gyro_y", "gyro_z", "setpoint_roll", "setpoint_pitch", "setpoint_yaw", "throttle"]:
        if col not in out.columns:
            out[col] = 0.0

    out = choose_best_window(out, max_duration_sec)
    out["time"] = out["time"] - out["time"].iloc[0]
    out = downsample_to_rate(out, target_hz)

    final_cols = [
        "time",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "setpoint_roll",
        "setpoint_pitch",
        "setpoint_yaw",
        "throttle",
    ]
    out = out[final_cols].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    input_mb = input_path.stat().st_size / (1024 * 1024)
    output_mb = output_path.stat().st_size / (1024 * 1024)
    duration = float(out["time"].iloc[-1] - out["time"].iloc[0]) if len(out) > 1 else 0.0
    sample_rate = len(out) / duration if duration > 0 else 0.0

    return {
        "input": str(input_path),
        "output": str(output_path),
        "input_mb": round(input_mb, 2),
        "output_mb": round(output_mb, 2),
        "rows": int(len(out)),
        "duration_sec": round(duration, 2),
        "estimated_hz": round(sample_rate, 2),
        "columns": final_cols,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize large FPV Blackbox CSV logs for AeroTune.")
    parser.add_argument("input", help="Path to large input CSV")
    parser.add_argument("-o", "--output", default=None, help="Output CSV path")
    parser.add_argument("--seconds", type=float, default=60.0, help="Target best-window duration, default 60")
    parser.add_argument("--hz", type=float, default=500.0, help="Target downsample rate, default 500 Hz")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = input_path.with_name(f"{input_path.stem}_aerotune_ready.csv")

    result = optimize_csv(
        input_path=input_path,
        output_path=output_path,
        max_duration_sec=args.seconds,
        target_hz=args.hz,
    )

    print("✅ AeroTune CSV optimized")
    print(f"Input:  {result['input']}")
    print(f"Output: {result['output']}")
    print(f"Size:   {result['input_mb']} MB → {result['output_mb']} MB")
    print(f"Rows:   {result['rows']}")
    print(f"Time:   {result['duration_sec']} sec")
    print(f"Rate:   ~{result['estimated_hz']} Hz")
    print("Columns kept:", ", ".join(result["columns"]))


if __name__ == "__main__":
    main()
