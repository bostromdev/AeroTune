from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


AXIS_MAP = {
    "roll": ("gyro_x", "setpoint_roll"),
    "pitch": ("gyro_y", "setpoint_pitch"),
    "yaw": ("gyro_z", "setpoint_yaw"),
}

GOALS = {"efficiency", "snappy", "floaty"}

DRONE_BANDS = {
    "5": {"low": (0, 35), "mid": (35, 90), "high": (90, 180)},
    "7": {"low": (0, 25), "mid": (25, 70), "high": (70, 140)},
}


@dataclass
class AxisStats:
    dominant_freq_hz: Optional[float]
    low_energy: float
    mid_energy: float
    high_energy: float
    total_energy: float
    corr: Optional[float]
    lag_ms: Optional[float]
    rms: float
    peak_to_peak: float


def _band_energy(freqs: np.ndarray, mags: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask):
        return 0.0
    return float(np.sum(np.square(mags[mask])))


def _estimate_lag_ms(setpoint: np.ndarray, gyro: np.ndarray, sample_rate_hz: float) -> Optional[float]:
    if len(setpoint) < 64 or np.std(setpoint) < 1e-6 or np.std(gyro) < 1e-6:
        return None

    sp = setpoint - np.mean(setpoint)
    gy = gyro - np.mean(gyro)

    max_lag = min(int(sample_rate_hz * 0.2), len(sp) // 4)
    if max_lag < 1:
        return None

    xcorr = np.correlate(gy, sp, mode="full")
    lags = np.arange(-len(sp) + 1, len(sp))
    mask = (lags >= -max_lag) & (lags <= max_lag)

    local_xcorr = xcorr[mask]
    local_lags = lags[mask]
    best_lag = int(local_lags[np.argmax(local_xcorr)])

    return float(best_lag / sample_rate_hz * 1000.0)


def _compute_axis_stats(
    gyro: np.ndarray,
    setpoint: np.ndarray,
    sample_rate_hz: float,
    bands: Dict[str, Tuple[float, float]],
) -> AxisStats:
    centered = gyro - np.mean(gyro)
    window = np.hanning(len(centered))
    fft_vals = np.fft.rfft(centered * window)
    freqs = np.fft.rfftfreq(len(centered), d=1.0 / sample_rate_hz)
    mags = np.abs(fft_vals)

    if len(mags) > 0:
        mags[0] = 0.0

    dominant_freq_hz = None
    if len(mags) > 1 and np.max(mags) > 0:
        dominant_freq_hz = float(freqs[int(np.argmax(mags))])

    low_energy = _band_energy(freqs, mags, *bands["low"])
    mid_energy = _band_energy(freqs, mags, *bands["mid"])
    high_energy = _band_energy(freqs, mags, *bands["high"])
    total_energy = low_energy + mid_energy + high_energy

    corr = None
    if np.std(setpoint) > 1e-6 and np.std(gyro) > 1e-6:
        matrix = np.corrcoef(setpoint, gyro)
        corr_val = matrix[0, 1]
        corr = float(corr_val) if np.isfinite(corr_val) else None

    lag_ms = _estimate_lag_ms(setpoint, gyro, sample_rate_hz)

    return AxisStats(
        dominant_freq_hz=dominant_freq_hz,
        low_energy=low_energy,
        mid_energy=mid_energy,
        high_energy=high_energy,
        total_energy=total_energy,
        corr=corr,
        lag_ms=lag_ms,
        rms=float(np.sqrt(np.mean(np.square(centered)))),
        peak_to_peak=float(np.max(gyro) - np.min(gyro)),
    )


def _classify_issue(stats: AxisStats) -> Tuple[str, float]:
    total = max(stats.total_energy, 1e-9)
    low_ratio = stats.low_energy / total
    mid_ratio = stats.mid_energy / total
    high_ratio = stats.high_energy / total

    confidence = 0.35

    if stats.dominant_freq_hz is not None:
        confidence += 0.15
    if stats.corr is not None:
        confidence += 0.10
    if stats.lag_ms is not None:
        confidence += 0.10

    if high_ratio > 0.50:
        return "high_frequency_noise", min(confidence + 0.20, 1.0)
    if mid_ratio > 0.45:
        return "mid_frequency_vibration", min(confidence + 0.20, 1.0)
    if low_ratio > 0.45:
        return "low_frequency_oscillation", min(confidence + 0.20, 1.0)
    if stats.corr is not None and stats.corr < 0.35:
        return "poor_tracking", min(confidence + 0.15, 1.0)
    if stats.lag_ms is not None and stats.lag_ms > 35:
        return "slow_response", min(confidence + 0.15, 1.0)

    if stats.rms > 0 and stats.peak_to_peak > 4.0 * stats.rms and stats.corr is not None and stats.corr > 0.45:
        return "propwash_or_rebound", min(confidence + 0.15, 1.0)

    return "mostly_clean", min(confidence, 1.0)


def _base_delta_for_issue(axis: str, issue: str) -> Dict[str, float]:
    delta = {"p": 0.0, "d": 0.0, "ff": 0.0}

    if issue == "high_frequency_noise":
        delta["p"] = -0.01
        delta["d"] = -0.02 if axis != "yaw" else 0.0

    elif issue == "mid_frequency_vibration":
        delta["p"] = -0.015
        delta["d"] = -0.02 if axis != "yaw" else 0.0

    elif issue == "low_frequency_oscillation":
        delta["p"] = -0.025
        delta["d"] = -0.01 if axis != "yaw" else 0.0

    elif issue == "poor_tracking":
        delta["p"] = 0.01
        delta["ff"] = 0.015

    elif issue == "slow_response":
        delta["p"] = 0.005
        delta["ff"] = 0.02

    elif issue == "propwash_or_rebound":
        delta["p"] = -0.005
        delta["d"] = 0.02 if axis != "yaw" else 0.0

    return delta


def _apply_goal_bias(delta: Dict[str, float], axis: str, issue: str, goal: str) -> Dict[str, float]:
    result = dict(delta)

    if goal == "efficiency":
        result["p"] *= 0.8
        result["d"] *= 0.7
        result["ff"] *= 0.8

    elif goal == "snappy":
        # sharper feel / faster response
        if issue in {"poor_tracking", "slow_response", "mostly_clean"}:
            result["p"] += 0.01 if axis != "yaw" else 0.005
            result["ff"] += 0.015
        if issue == "propwash_or_rebound" and axis != "yaw":
            result["d"] += 0.005
        if issue == "high_frequency_noise":
            result["d"] = min(result["d"], 0.0)

    elif goal == "floaty":
        # softer / less locked-in
        if issue in {"mostly_clean", "poor_tracking", "slow_response"}:
            result["p"] -= 0.01 if axis != "yaw" else 0.005
            result["ff"] -= 0.015
        if issue == "propwash_or_rebound" and axis != "yaw":
            result["d"] -= 0.005

    result["p"] = float(np.clip(result["p"], -0.06, 0.06))
    result["d"] = float(np.clip(result["d"], -0.06, 0.06 if axis != "yaw" else 0.0))
    result["ff"] = float(np.clip(result["ff"], -0.06, 0.06))

    if axis == "yaw":
        result["d"] = 0.0

    return result


def _feel_text(issue: str, goal: str) -> str:
    if issue == "slow_response":
        return "Delayed / soft"
    if issue == "poor_tracking":
        return "Soft / not following stick well"
    if issue == "low_frequency_oscillation":
        return "Loose / bouncing"
    if issue == "mid_frequency_vibration":
        return "Shaky / rough"
    if issue == "high_frequency_noise":
        return "Harsh / noisy"
    if issue == "propwash_or_rebound":
        return "Washy / reboundy"
    if goal == "snappy":
        return "Mostly clean, but can be sharper"
    if goal == "floaty":
        return "Mostly clean, but can be softer"
    return "Mostly clean / efficient"


def _actions_for_issue(issue: str, goal: str) -> List[str]:
    base = []

    if issue == "high_frequency_noise":
        base.extend([
            "Noise looks high. Fix props, motors, filtering, or frame problems before pushing PID harder.",
            "Keep D conservative here."
        ])
    elif issue == "mid_frequency_vibration":
        base.extend([
            "Back off P and D a little.",
            "Check prop balance and frame vibration."
        ])
    elif issue == "low_frequency_oscillation":
        base.extend([
            "Reduce P a little first.",
            "If it still bounces, reduce D a little too."
        ])
    elif issue == "poor_tracking":
        base.extend([
            "Response looks soft.",
            "A small P and FF increase can help."
        ])
    elif issue == "slow_response":
        base.extend([
            "Response looks delayed.",
            "A small FF increase is the safest first move."
        ])
    elif issue == "propwash_or_rebound":
        base.extend([
            "This looks like propwash or rebound.",
            "A tiny D increase may help, but watch motor heat."
        ])
    else:
        base.append("Log looks mostly clean. Only tiny changes make sense.")

    if goal == "efficiency":
        base.append("Goal is efficiency, so changes stay smaller and safer.")
    elif goal == "snappy":
        base.append("Goal is snappy feel, so response and lock-in get more priority.")
    elif goal == "floaty":
        base.append("Goal is floatier feel, so the tune is softened slightly.")

    return base


def _recommendation_text(axis: str, issue: str, goal: str, delta: Dict[str, float], valid_for_pid: bool) -> str:
    if not valid_for_pid:
        return f"{axis.upper()}: diagnosis only. Data confidence is too low for safe PID moves."

    parts = []
    for key in ("p", "d", "ff"):
        val = delta[key] * 100.0
        if abs(val) >= 0.05:
            parts.append(f"{key.upper()} {'+' if val > 0 else ''}{val:.1f}%")

    if not parts:
        return f"{axis.upper()}: {issue.replace('_', ' ')} detected, but no change is recommended."

    return f"{axis.upper()}: {issue.replace('_', ' ')} for {goal} -> " + ", ".join(parts)


def detect_oscillation(
    df: pd.DataFrame,
    drone_size: str = "7",
    tuning_goal: str = "efficiency",
) -> Dict[str, Any]:
    if df is None or df.empty:
        return {
            "status": "invalid",
            "valid_for_pid": False,
            "summary": "No usable data.",
            "warnings": ["Parsed log is empty."],
            "global_actions": [],
            "axes": {},
            "recommendations": [],
            "tuning_goal": tuning_goal,
        }

    if tuning_goal not in GOALS:
        tuning_goal = "efficiency"

    if "time" not in df.columns:
        return {
            "status": "invalid",
            "valid_for_pid": False,
            "summary": "Missing time column.",
            "warnings": ["AeroTune requires real timestamps."],
            "global_actions": [],
            "axes": {},
            "recommendations": [],
            "tuning_goal": tuning_goal,
        }

    time = df["time"].to_numpy(dtype=float)
    if len(time) < 128:
        return {
            "status": "invalid",
            "valid_for_pid": False,
            "summary": "Not enough log samples.",
            "warnings": ["Need at least 128 rows for usable analysis."],
            "global_actions": [],
            "axes": {},
            "recommendations": [],
            "tuning_goal": tuning_goal,
        }

    dt = np.diff(time)
    if np.any(dt <= 0) or not np.isfinite(np.mean(dt)):
        return {
            "status": "invalid",
            "valid_for_pid": False,
            "summary": "Time spacing is invalid.",
            "warnings": ["Timestamps must be increasing and real."],
            "global_actions": [],
            "axes": {},
            "recommendations": [],
            "tuning_goal": tuning_goal,
        }

    sample_rate_hz = float(1.0 / np.median(dt))
    duration_s = float(time[-1] - time[0])

    bands = DRONE_BANDS.get(str(drone_size), DRONE_BANDS["7"])

    axes = {}
    recommendations = []
    global_actions: List[str] = []
    valid_axis_count = 0

    for axis_name, (gyro_col, setpoint_col) in AXIS_MAP.items():
        if gyro_col not in df.columns:
            continue

        gyro = df[gyro_col].to_numpy(dtype=float)
        setpoint = df[setpoint_col].to_numpy(dtype=float) if setpoint_col in df.columns else np.zeros_like(gyro)

        stats = _compute_axis_stats(gyro, setpoint, sample_rate_hz, bands)
        issue, confidence = _classify_issue(stats)

        base_delta = _base_delta_for_issue(axis_name, issue)
        pid_delta = _apply_goal_bias(base_delta, axis_name, issue, tuning_goal)

        valid_for_pid = bool(confidence >= 0.50 and stats.total_energy > 0)
        if valid_for_pid:
            valid_axis_count += 1

        actions = _actions_for_issue(issue, tuning_goal)
        recommendation_text = _recommendation_text(axis_name, issue, tuning_goal, pid_delta, valid_for_pid)
        feel = _feel_text(issue, tuning_goal)

        axes[axis_name] = {
            "axis": axis_name,
            "axis_name": axis_name.upper(),
            "status": "ok",
            "issue": issue,
            "feel": feel,
            "confidence": round(confidence, 3),
            "valid_for_pid": valid_for_pid,
            "signal": {
                "dominant_freq_hz": None if stats.dominant_freq_hz is None else round(stats.dominant_freq_hz, 2),
                "low_band_energy": round(stats.low_energy, 4),
                "mid_band_energy": round(stats.mid_energy, 4),
                "high_band_energy": round(stats.high_energy, 4),
                "total_band_energy": round(stats.total_energy, 4),
                "rms": round(stats.rms, 4),
                "peak_to_peak": round(stats.peak_to_peak, 4),
            },
            "tracking": {
                "corr": None if stats.corr is None else round(stats.corr, 4),
                "lag_ms": None if stats.lag_ms is None else round(stats.lag_ms, 2),
                "usable": stats.corr is not None,
            },
            "pid_delta_pct": {
                "p": round(pid_delta["p"], 4),
                "d": round(pid_delta["d"], 4),
                "ff": round(pid_delta["ff"], 4),
            },
            "recommendation_text": recommendation_text,
            "actions": actions,
            "warnings": [],
        }

        recommendations.append({
            "axis": axis_name,
            "issue": issue,
            "feel": feel,
            "valid_for_pid": valid_for_pid,
            "confidence": round(confidence, 3),
            "pid_delta_pct": axes[axis_name]["pid_delta_pct"],
            "recommendation_text": recommendation_text,
        })

    if not axes:
        return {
            "status": "invalid",
            "valid_for_pid": False,
            "summary": "No usable gyro axes found.",
            "warnings": ["Parser did not produce gyro_x / gyro_y / gyro_z columns."],
            "global_actions": [],
            "axes": {},
            "recommendations": [],
            "sample_rate_hz": round(sample_rate_hz, 2),
            "duration_s": round(duration_s, 3),
            "drone_size": str(drone_size),
            "tuning_goal": tuning_goal,
        }

    if valid_axis_count == 0:
        summary = "Diagnosis available, but confidence is too low for safe PID changes."
    else:
        summary = f"Analysis complete for {tuning_goal}. Conservative PID deltas are available."

    global_actions.append("Use small changes, test one step at a time, and verify motor temps after D increases.")
    global_actions.append("Final real PID numbers come from applying these deltas to your current tune.")

    return {
        "status": "ok",
        "valid_for_pid": valid_axis_count > 0,
        "summary": summary,
        "warnings": [],
        "global_actions": global_actions,
        "axes": axes,
        "recommendations": recommendations,
        "sample_rate_hz": round(sample_rate_hz, 2),
        "duration_s": round(duration_s, 3),
        "drone_size": str(drone_size),
        "tuning_goal": tuning_goal,
    }