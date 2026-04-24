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

GOALS = {
    "efficient",
    "locked_in",
    "floaty",
    # Backward-compatible aliases from earlier AeroTune UI versions.
    "efficiency",
    "efficiency_snappy",
    "efficiency_floaty",
    "snappy",
}

GOAL_ALIASES = {
    "efficiency": "efficient",
    "efficiency_snappy": "locked_in",
    "efficiency_floaty": "floaty",
    "snappy": "locked_in",
}

DRONE_BANDS = {
    "5": {"low": (0, 35), "mid": (35, 90), "high": (90, 180)},
    "7": {"low": (0, 25), "mid": (25, 70), "high": (70, 140)},
}

SOURCE_REFERENCES = [
    "Betaflight PID Tuning Guide: https://betaflight.com/docs/wiki/guides/current/PID-Tuning-Guide",
    "Betaflight tuning basis: P sharpens response, I improves hold, D damps bounceback/propwash, FF improves stick response.",
    "Safety basis: keep D conservative and check motor temperature after D/filter changes.",
]


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
    tracking_error_rms: float
    setpoint_rms: float
    error_ratio: float


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

    tracking_error = setpoint - gyro
    tracking_error_rms = float(np.sqrt(np.mean(np.square(tracking_error))))
    setpoint_rms = float(np.sqrt(np.mean(np.square(setpoint)))) if len(setpoint) else 0.0
    error_ratio = tracking_error_rms / max(setpoint_rms, 1e-6)

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
        tracking_error_rms=tracking_error_rms,
        setpoint_rms=setpoint_rms,
        error_ratio=float(error_ratio),
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

    # Noise / oscillation first
    if high_ratio > 0.50:
        return "high_frequency_noise", min(confidence + 0.20, 1.0)

    if mid_ratio > 0.45:
        return "mid_frequency_vibration", min(confidence + 0.20, 1.0)

    if low_ratio > 0.45:
        return "low_frequency_oscillation", min(confidence + 0.20, 1.0)

    # Bounceback / propwash-like behavior
    if (
        stats.rms > 0
        and stats.peak_to_peak > 4.0 * stats.rms
        and stats.corr is not None
        and stats.corr > 0.45
    ):
        return "propwash_or_bounceback", min(confidence + 0.15, 1.0)

    # Drift / weak hold maps closer to I guidance
    if (
        stats.error_ratio > 0.55
        and (stats.lag_ms is None or abs(stats.lag_ms) < 20.0)
        and (stats.corr is not None and stats.corr > 0.40)
    ):
        return "drift_or_weak_hold", min(confidence + 0.15, 1.0)

    # Soft / delayed tracking maps to P and FF guidance
    if stats.corr is not None and stats.corr < 0.35:
        return "poor_tracking", min(confidence + 0.15, 1.0)

    if stats.lag_ms is not None and stats.lag_ms > 35:
        return "slow_response", min(confidence + 0.15, 1.0)

    return "mostly_clean", min(confidence, 1.0)


def _base_delta_for_issue(axis: str, issue: str) -> Dict[str, float]:
    delta = {"p": 0.0, "i": 0.0, "d": 0.0, "ff": 0.0}

    if issue == "high_frequency_noise":
        delta["p"] = -0.02
        delta["d"] = -0.03 if axis != "yaw" else 0.0

    elif issue == "mid_frequency_vibration":
        delta["p"] = -0.02
        delta["d"] = -0.02 if axis != "yaw" else 0.0

    elif issue == "low_frequency_oscillation":
        delta["p"] = -0.03
        delta["d"] = -0.01 if axis != "yaw" else 0.0

    elif issue == "drift_or_weak_hold":
        delta["i"] = 0.04

    elif issue == "poor_tracking":
        delta["p"] = 0.02
        delta["ff"] = 0.03

    elif issue == "slow_response":
        delta["p"] = 0.01
        delta["ff"] = 0.04

    elif issue == "propwash_or_bounceback":
        delta["d"] = 0.03 if axis != "yaw" else 0.0

    return delta


def _apply_goal_bias(delta: Dict[str, float], axis: str, issue: str, goal: str) -> Dict[str, float]:
    """
    Converts detected flight behavior into conservative percentage deltas.

    Betaflight-based model:
    - P sharpens authority/tracking but can oscillate if pushed too far.
    - I improves attitude hold, especially through throttle changes, but too much can feel stiff.
    - D damps bounceback/propwash, but too much D can heat motors or amplify noise.
    - FF improves stick response without using P as the only way to make the quad feel sharp.
    """
    result = dict(delta)

    if goal == "efficient":
        result["p"] *= 0.80
        result["i"] *= 0.90
        result["d"] *= 0.70
        result["ff"] *= 0.80

    elif goal == "locked_in":
        if issue in {"poor_tracking", "slow_response", "mostly_clean"}:
            result["p"] += 0.010 if axis != "yaw" else 0.005
            result["ff"] += 0.020

        if issue == "drift_or_weak_hold":
            result["i"] += 0.010

        if issue == "propwash_or_bounceback" and axis != "yaw":
            result["d"] += 0.005

        if issue in {"high_frequency_noise", "mid_frequency_vibration"}:
            result["d"] = min(result["d"], 0.0)

    elif goal == "floaty":
        if issue in {"mostly_clean", "poor_tracking", "slow_response"}:
            result["p"] -= 0.020 if axis != "yaw" else 0.010
            result["ff"] -= 0.020

        if issue == "propwash_or_bounceback" and axis != "yaw":
            result["d"] -= 0.005

        if issue == "drift_or_weak_hold":
            result["i"] += 0.005

    result["p"] = float(np.clip(result["p"], -0.06, 0.06))
    result["i"] = float(np.clip(result["i"], -0.06, 0.06))
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
    if issue == "drift_or_weak_hold":
        return "Weak hold / drifting"
    if issue == "low_frequency_oscillation":
        return "Loose / bouncing"
    if issue == "mid_frequency_vibration":
        return "Shaky / rough"
    if issue == "high_frequency_noise":
        return "Harsh / noisy"
    if issue == "propwash_or_bounceback":
        return "Washy / bounceback"
    if goal == "locked_in":
        return "Mostly clean, but can be sharper"
    if goal == "floaty":
        return "Mostly clean, but can be softer"
    return "Mostly clean / efficient"

def _actions_for_issue(issue: str, goal: str) -> List[str]:
    base: List[str] = []

    if issue == "high_frequency_noise":
        base.extend([
            "This would probably feel buzzy, harsh, or nervous in the sticks.",
            "The drone may sound rough and motors may come down warmer than they should.",
            "Before chasing tune, check props, motors, frame screws, stack mounting, and filtering.",
            "Tune direction: calm it down. Do not push D harder until the build is mechanically clean.",
        ])
    elif issue == "mid_frequency_vibration":
        base.extend([
            "This would probably feel shaky or rough instead of smooth and planted.",
            "It often points to prop/motor/frame vibration getting into the gyro.",
            "Tune direction: soften the tune slightly and inspect the build before trying to make it sharper.",
        ])
    elif issue == "low_frequency_oscillation":
        base.extend([
            "This would probably feel like a slow wobble, bounce, or loose rocking motion.",
            "The quad is likely correcting too hard and then swinging back the other way.",
            "Tune direction: make the control loop less aggressive before adding more sharpness.",
        ])
    elif issue == "drift_or_weak_hold":
        base.extend([
            "This would probably feel like the quad does not hold attitude well when throttle changes.",
            "It may feel like it slowly leans, floats away, or needs constant correction.",
            "Tune direction: improve hold so it stays more planted through throttle changes.",
        ])
    elif issue == "poor_tracking":
        base.extend([
            "This would probably feel disconnected from your sticks.",
            "You move the stick, but the quad does not follow cleanly or immediately.",
            "Tune direction: make the response more direct, but only if the log is clean enough to support it.",
        ])
    elif issue == "slow_response":
        base.extend([
            "This would probably feel lazy, delayed, or floaty even when you are asking it to move.",
            "The quad is following, but it is late getting there.",
            "Tune direction: add response carefully instead of making the whole tune aggressive.",
        ])
    elif issue == "propwash_or_bounceback":
        base.extend([
            "This would probably show up after flips, drops, hard turns, or throttle chops.",
            "You may feel a bounce, shake, or washout when the props hit dirty air.",
            "Tune direction: add damping carefully, then check motor temps after a short test flight.",
        ])
    else:
        base.extend([
            "This log looks mostly clean.",
            "The quad should already feel fairly predictable.",
            "Tune direction: make only tiny feel-based changes instead of chasing numbers.",
        ])

    if goal == "efficient":
        base.append("Because you picked Efficient, the safest move is keeping the quad smooth, cool, and predictable.")
    elif goal == "locked_in":
        base.append("Because you picked Locked-In, the goal is a more connected stick feel without making it twitchy or hot.")
    elif goal == "floaty":
        base.append("Because you picked Floaty, the goal is softer movement and less twitch without making it sloppy.")

    return base

def _warnings_for_goal(goal: str) -> List[str]:
    if goal == "locked_in":
        return ["Locked-in tuning can increase heat, twitchiness, or overshoot if P/D/FF are pushed too far."]
    if goal == "floaty":
        return ["Floaty tuning reduces sharpness and response speed in exchange for a smoother feel."]
    if goal == "efficient":
        return ["Efficient tuning prioritizes conservative changes, lower heat risk, and smoother response over maximum sharpness."]
    return []

def _recommendation_text(axis: str, issue: str, goal: str, delta: Dict[str, float], valid_for_pid: bool) -> str:
    axis_label = axis.upper()

    if not valid_for_pid:
        return f"{axis_label}: I can read the behavior, but this log is not clean enough for confident tuning advice. Retake a better log before trusting changes."

    feel_map = {
        "high_frequency_noise": f"{axis_label}: This looks like a noisy/harsh axis. In the air it would probably feel buzzy, nervous, or rough instead of smooth.",
        "mid_frequency_vibration": f"{axis_label}: This looks like vibration getting into the gyro. It would probably feel shaky or unsettled, especially when throttle changes.",
        "low_frequency_oscillation": f"{axis_label}: This looks like a slower wobble or bounce. It may feel like the quad is over-correcting instead of holding steady.",
        "drift_or_weak_hold": f"{axis_label}: This looks like weak hold. It may feel like the quad slowly leans or needs extra correction when throttle changes.",
        "poor_tracking": f"{axis_label}: This looks like soft stick tracking. It may feel like your stick input and the quad are not fully connected.",
        "slow_response": f"{axis_label}: This looks delayed. It may feel lazy, floaty, or late when you try to snap into a move.",
        "propwash_or_bounceback": f"{axis_label}: This looks like propwash or bounceback. You may notice shake after flips, drops, hard turns, or throttle chops.",
        "mostly_clean": f"{axis_label}: This looks mostly clean. Do not chase big changes; tune by feel from here.",
    }

    base = feel_map.get(issue, f"{axis_label}: This axis has a behavior worth reviewing by feel.")

    if goal == "efficient":
        goal_text = "For an efficient feel, keep it smooth and calm instead of chasing a super sharp response."
    elif goal == "locked_in":
        goal_text = "For a locked-in feel, aim for stronger stick connection while watching for heat, twitchiness, or bounceback."
    elif goal == "floaty":
        goal_text = "For a floaty feel, soften the response so it flows more, but do not let it become delayed or sloppy."
    else:
        goal_text = "Tune this in small steps and judge the next flight by feel."

    return f"{base} {goal_text}"


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
            "source_references": SOURCE_REFERENCES,
        }

    if tuning_goal not in GOALS:
        tuning_goal = "efficient"
    tuning_goal = GOAL_ALIASES.get(tuning_goal, tuning_goal)

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
            "source_references": SOURCE_REFERENCES,
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
            "source_references": SOURCE_REFERENCES,
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
            "source_references": SOURCE_REFERENCES,
        }

    sample_rate_hz = float(1.0 / np.median(dt))
    duration_s = float(time[-1] - time[0])

    bands = DRONE_BANDS.get(str(drone_size), DRONE_BANDS["7"])

    axes = {}
    recommendations = []
    global_actions: List[str] = []
    global_warnings: List[str] = _warnings_for_goal(tuning_goal)
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
                "tracking_error_rms": round(stats.tracking_error_rms, 4),
                "error_ratio": round(stats.error_ratio, 4),
                "usable": stats.corr is not None,
            },
            "pid_delta_pct": {
                "p": round(pid_delta["p"], 4),
                "i": round(pid_delta["i"], 4),
                "d": round(pid_delta["d"], 4),
                "ff": round(pid_delta["ff"], 4),
            },
            "recommendation_text": recommendation_text,
            "actions": actions,
            "warnings": _warnings_for_goal(tuning_goal),
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
            "source_references": SOURCE_REFERENCES,
        }

    if valid_axis_count == 0:
        summary = "Diagnosis available, but confidence is too low for safe PID changes."
    else:
        summary = f"Analysis complete using Betaflight-based tuning logic for a {tuning_goal.replace('_', ' ')} feel. Review the feel-based notes for each axis."

    global_actions.append("Use small changes and test one step at a time.")
    global_actions.append("If D goes up, do a short flight and check motor temperature before continuing.")
    global_actions.append("For bounceback or propwash on a clean build, consider D Max 20–30% above base D instead of making base D too high.")
    global_actions.append("Use the feel notes to decide what to test next; do not chase numbers blindly.")

    return {
        "status": "ok",
        "valid_for_pid": valid_axis_count > 0,
        "summary": summary,
        "warnings": global_warnings,
        "global_actions": global_actions,
        "axes": axes,
        "recommendations": recommendations,
        "sample_rate_hz": round(sample_rate_hz, 2),
        "duration_s": round(duration_s, 3),
        "drone_size": str(drone_size),
        "tuning_goal": tuning_goal,
        "source_references": SOURCE_REFERENCES,
    }