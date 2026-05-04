"""
AeroTune tuning advisor.

Purpose:
- Convert Blackbox signal behavior into conservative Betaflight tuning suggestions.
- Output DELTAS, not fake final PID numbers.
- Focus on real-world pilot symptoms:
  overshoot, bounceback, propwash tendency, noisy D-term risk, tracking feel.

This module is intentionally conservative. It does not auto-tune the quad.
It gives repeatable, explainable, manual tuning advice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math

import numpy as np
import pandas as pd


AXES = ("roll", "pitch", "yaw")


GYRO_CANDIDATES = {
    "roll": [
        "gyro_roll", "gyro[roll]", "gyro_roll_axis", "gyro_x", "gyroADC[0]",
        "gyro_x_debug", "roll_gyro", "gyroRoll"
    ],
    "pitch": [
        "gyro_pitch", "gyro[pitch]", "gyro_pitch_axis", "gyro_y", "gyroADC[1]",
        "gyro_y_debug", "pitch_gyro", "gyroPitch"
    ],
    "yaw": [
        "gyro_yaw", "gyro[yaw]", "gyro_yaw_axis", "gyro_z", "gyroADC[2]",
        "gyro_z_debug", "yaw_gyro", "gyroYaw"
    ],
}

SETPOINT_CANDIDATES = {
    "roll": [
        "setpoint_roll", "setpoint[roll]", "rcCommand[0]", "axisP[0]",
        "roll_setpoint", "debug[roll_setpoint]", "setpointRoll"
    ],
    "pitch": [
        "setpoint_pitch", "setpoint[pitch]", "rcCommand[1]", "axisP[1]",
        "pitch_setpoint", "debug[pitch_setpoint]", "setpointPitch"
    ],
    "yaw": [
        "setpoint_yaw", "setpoint[yaw]", "rcCommand[2]", "axisP[2]",
        "yaw_setpoint", "debug[yaw_setpoint]", "setpointYaw"
    ],
}

DTERM_CANDIDATES = {
    "roll": ["dterm_roll", "dterm[roll]", "axisD[0]", "D_roll", "d_roll"],
    "pitch": ["dterm_pitch", "dterm[pitch]", "axisD[1]", "D_pitch", "d_pitch"],
    "yaw": ["dterm_yaw", "dterm[yaw]", "axisD[2]", "D_yaw", "d_yaw"],
}

TIME_CANDIDATES = ["time", "timestamp", "time_us", "time(ms)", "looptime", "Time"]


@dataclass
class AxisMetrics:
    axis: str
    available: bool
    gyro_column: Optional[str] = None
    setpoint_column: Optional[str] = None
    dterm_column: Optional[str] = None
    sample_count: int = 0
    movement_count: int = 0
    overshoot_score: float = 0.0
    bounceback_score: float = 0.0
    tracking_error_score: float = 0.0
    noise_score: float = 0.0
    confidence: str = "low"
    notes: Optional[List[str]] = None


def _clean_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None

    original = list(df.columns)
    cleaned_map = {_clean_name(c): c for c in original}

    for candidate in candidates:
        key = _clean_name(candidate)
        if key in cleaned_map:
            return cleaned_map[key]

    # Fuzzy fallback for weird Betaflight export labels.
    for candidate in candidates:
        key = _clean_name(candidate)
        for cleaned, original_name in cleaned_map.items():
            if key in cleaned or cleaned in key:
                return original_name

    return None


def _numeric_series(df: pd.DataFrame, column: Optional[str]) -> Optional[np.ndarray]:
    if not column or column not in df.columns:
        return None

    s = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 50:
        return None

    arr = s.to_numpy(dtype=float)

    # Remove impossible garbage spikes from broken parses.
    finite = arr[np.isfinite(arr)]
    if len(finite) < 50:
        return None

    p1, p99 = np.percentile(finite, [1, 99])
    span = max(abs(p1), abs(p99), 1.0)
    arr = np.clip(arr, -span * 3.0, span * 3.0)

    return arr


def _estimate_sample_rate(df: pd.DataFrame) -> Optional[float]:
    time_col = _find_column(df, TIME_CANDIDATES)
    if not time_col:
        return None

    t = pd.to_numeric(df[time_col], errors="coerce").dropna().to_numpy(dtype=float)
    if len(t) < 100:
        return None

    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    dt = dt[dt > 0]

    if len(dt) < 50:
        return None

    median_dt = float(np.median(dt))

    # Betaflight CSV time often appears in microseconds.
    if median_dt > 100:
        rate = 1_000_000.0 / median_dt
    # Sometimes milliseconds.
    elif median_dt > 0.1:
        rate = 1000.0 / median_dt
    # Sometimes seconds.
    else:
        rate = 1.0 / median_dt

    if rate < 10 or rate > 10000:
        return None

    return rate


def _moving_mask(setpoint: np.ndarray) -> np.ndarray:
    """
    Detect real stick movement using adaptive scale.

    Betaflight exports are not always in the same units. Some logs show
    degrees/sec-like setpoint values, while others are scaled smaller.
    This avoids missing valid logs just because the setpoint magnitude is low.
    """
    abs_sp = np.abs(setpoint)
    if len(abs_sp) == 0:
        return np.zeros(0, dtype=bool)

    p50 = float(np.percentile(abs_sp, 50))
    p75 = float(np.percentile(abs_sp, 75))
    p90 = float(np.percentile(abs_sp, 90))
    p95 = float(np.percentile(abs_sp, 95))
    max_v = float(np.max(abs_sp))

    # Adaptive threshold: high enough to ignore idle noise, low enough to catch
    # scaled Betaflight exports.
    threshold = max(
        3.0,
        p50 * 1.8,
        p75 * 0.65,
        p95 * 0.12,
        max_v * 0.08,
    )

    # Never let the threshold become larger than the useful movement band.
    threshold = min(threshold, max(3.0, p90 * 0.75))

    return abs_sp > threshold


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0 or not math.isfinite(denominator):
        return 0.0
    return float(numerator / denominator)


def _high_frequency_noise_score(signal: np.ndarray, sample_rate: Optional[float]) -> float:
    if sample_rate is None or len(signal) < 512:
        return 0.0

    x = np.asarray(signal, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 512:
        return 0.0

    # Limit workload for large logs while keeping representative behavior.
    if len(x) > 12000:
        step = max(1, len(x) // 12000)
        x = x[::step]

    x = x - np.median(x)
    std = float(np.std(x))
    if std < 1e-9:
        return 0.0

    window = np.hanning(len(x))
    spectrum = np.abs(np.fft.rfft(x * window)) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sample_rate)

    total = float(np.sum(spectrum))
    if total <= 0:
        return 0.0

    # 7-inch quads and action cameras can show mechanical energy lower than 5-inch racers.
    high_band = spectrum[(freqs >= 90) & (freqs <= min(500, sample_rate / 2))]
    return float(np.clip(np.sum(high_band) / total, 0.0, 1.0))


def _analyze_axis(df: pd.DataFrame, axis: str, sample_rate: Optional[float]) -> AxisMetrics:
    gyro_col = _find_column(df, GYRO_CANDIDATES[axis])
    sp_col = _find_column(df, SETPOINT_CANDIDATES[axis])
    d_col = _find_column(df, DTERM_CANDIDATES[axis])

    gyro = _numeric_series(df, gyro_col)
    sp = _numeric_series(df, sp_col)

    notes: List[str] = []

    if gyro is None or sp is None:
        return AxisMetrics(
            axis=axis,
            available=False,
            gyro_column=gyro_col,
            setpoint_column=sp_col,
            dterm_column=d_col,
            notes=["Missing usable gyro or setpoint column for this axis."],
        )

    n = min(len(gyro), len(sp))
    gyro = gyro[:n]
    sp = sp[:n]

    moving = _moving_mask(sp)
    movement_count = int(np.sum(moving))

    if movement_count < max(40, int(0.02 * n)):
        return AxisMetrics(
            axis=axis,
            available=True,
            gyro_column=gyro_col,
            setpoint_column=sp_col,
            dterm_column=d_col,
            sample_count=n,
            movement_count=movement_count,
            confidence="low",
            notes=["Not enough stick movement on this axis for high-confidence tuning."],
        )

    abs_sp = np.abs(sp[moving])
    abs_gyro = np.abs(gyro[moving])
    error = gyro[moving] - sp[moving]

    sp_ref = max(float(np.percentile(abs_sp, 90)), 1.0)
    gyro_ref = max(float(np.percentile(abs_gyro, 90)), 1.0)
    error_ref = max(float(np.percentile(np.abs(error), 75)), 1.0)

    # Overshoot method 1:
    # Gyro exceeds setpoint by a relative margin. Margin is adaptive because
    # Betaflight logs can be scaled differently depending on export/parser path.
    overshoot_margin = np.maximum(abs_sp * 1.04, abs_sp + max(2.0, sp_ref * 0.035))
    overshoot_events = abs_gyro > overshoot_margin
    overshoot_score_a = _safe_ratio(float(np.sum(overshoot_events)), float(len(abs_sp)))

    # Overshoot method 2:
    # Error is large while setpoint is active. This catches cases where gyro
    # visibly trails/crosses setpoint but absolute units are small.
    high_error = np.abs(error) > max(2.0, sp_ref * 0.12)
    overshoot_score_b = _safe_ratio(float(np.sum(high_error)), float(len(error)))

    # Bounceback:
    # Error sign flips during movement, meaning the craft crossed past target
    # and corrected back. This is the log behavior we were manually tuning.
    err = error
    err_abs = np.abs(err)
    meaningful_err = err_abs > max(2.0, error_ref * 0.45, sp_ref * 0.06)
    sign_flip = np.zeros_like(meaningful_err, dtype=bool)
    sign_flip[1:] = (np.sign(err[1:]) != np.sign(err[:-1])) & meaningful_err[1:] & meaningful_err[:-1]
    bounceback_score = _safe_ratio(float(np.sum(sign_flip)), float(len(err)))

    # Tracking error:
    # Higher values mean gyro is not following setpoint cleanly.
    tracking_error_score = float(np.clip(np.percentile(np.abs(error), 75) / max(sp_ref, gyro_ref, 1.0), 0.0, 2.0))

    # Blend overshoot evidence. Do not let tiny noise produce big advice, but
    # do not require cartoon-level overshoot either.
    overshoot_score = max(float(overshoot_score_a), float(overshoot_score_b) * 0.55)

    # Prefer actual D-term if available; otherwise use gyro noise proxy.
    d_arr = _numeric_series(df, d_col)
    if d_arr is not None:
        noise_score = _high_frequency_noise_score(d_arr[: min(len(d_arr), n)], sample_rate)
    else:
        noise_score = _high_frequency_noise_score(gyro, sample_rate)
        notes.append("No D-term column found; using gyro high-frequency energy as noise proxy.")

    raw_conf = 0
    if movement_count > 200:
        raw_conf += 1
    if sample_rate is not None:
        raw_conf += 1
    if d_arr is not None:
        raw_conf += 1

    confidence = "high" if raw_conf >= 3 else "medium" if raw_conf == 2 else "low"

    return AxisMetrics(
        axis=axis,
        available=True,
        gyro_column=gyro_col,
        setpoint_column=sp_col,
        dterm_column=d_col,
        sample_count=n,
        movement_count=movement_count,
        overshoot_score=round(float(overshoot_score), 4),
        bounceback_score=round(float(bounceback_score), 4),
        tracking_error_score=round(float(tracking_error_score), 4),
        noise_score=round(float(noise_score), 4),
        confidence=confidence,
        notes=notes,
    )


def _severity(metrics: AxisMetrics) -> str:
    if not metrics.available:
        return "unknown"

    score = (
        metrics.overshoot_score * 1.00
        + metrics.bounceback_score * 0.75
        + max(0.0, metrics.tracking_error_score - 0.22) * 0.30
    )

    if score >= 0.16:
        return "high"
    if score >= 0.075:
        return "medium"
    if score >= 0.030:
        return "mild"
    return "low"


def _axis_recommendation(metrics: AxisMetrics, drone_size: str, axis: str) -> Dict[str, Any]:
    zero = {
        "p_percent": 0,
        "i_percent": 0,
        "d_percent": 0,
        "dmax_percent": 0,
        "ff_percent": 0,
    }

    if not metrics.available:
        return {
            "axis": axis,
            "action": "no_change",
            "confidence": "low",
            "severity": "unknown",
            "deltas": zero,
            "reason": "No usable gyro/setpoint data found for this axis.",
            "notes": metrics.notes or [],
        }

    if axis == "yaw":
        return {
            "axis": axis,
            "action": "no_change",
            "confidence": metrics.confidence,
            "severity": "not_targeted",
            "deltas": zero,
            "reason": "Yaw was not part of the roll/pitch overshoot correction. Leave yaw alone unless yaw itself feels twitchy or delayed.",
            "notes": metrics.notes or [],
        }

    sev = _severity(metrics)

    # If noise is already elevated, avoid blindly adding more D.
    noisy = metrics.noise_score >= 0.34

    size = str(drone_size).replace('"', "").strip()
    large_quad = size in {"6", "7", "8", "9", "10"}

    # Conservative base advice.
    if sev == "high":
        deltas = {"p_percent": -5, "i_percent": 0, "d_percent": 10, "dmax_percent": 12, "ff_percent": -5}
    elif sev == "medium":
        deltas = {"p_percent": -3, "i_percent": 0, "d_percent": 8, "dmax_percent": 10, "ff_percent": -4}
    elif sev == "mild":
        deltas = {"p_percent": -2, "i_percent": 0, "d_percent": 5, "dmax_percent": 7, "ff_percent": -3}
    else:
        deltas = zero.copy()

    if large_quad and sev in {"medium", "high"}:
        # Bigger props heat motors faster and show more mechanical resonance.
        # Keep recommendations useful, but slightly less aggressive.
        deltas["d_percent"] = max(0, deltas["d_percent"] - 1)
        deltas["dmax_percent"] = max(0, deltas["dmax_percent"] - 1)

    if noisy and sev != "low":
        deltas["d_percent"] = max(0, deltas["d_percent"] - 3)
        deltas["dmax_percent"] = max(0, deltas["dmax_percent"] - 3)
        note = "High-frequency noise is elevated; secure GoPro/props/frame before pushing D much higher."
    else:
        note = "D increase targets overshoot/bounceback; small P/FF reduction prevents snapping past target."

    action = "adjust" if any(v != 0 for v in deltas.values()) else "hold"

    return {
        "axis": axis,
        "action": action,
        "confidence": metrics.confidence,
        "severity": sev,
        "deltas": deltas,
        "reason": note if action == "adjust" else "No strong overshoot correction needed from this log.",
        "evidence": {
            "overshoot_score": metrics.overshoot_score,
            "bounceback_score": metrics.bounceback_score,
            "tracking_error_score": metrics.tracking_error_score,
            "noise_score": metrics.noise_score,
            "gyro_column": metrics.gyro_column,
            "setpoint_column": metrics.setpoint_column,
            "dterm_column": metrics.dterm_column,
            "movement_count": metrics.movement_count,
        },
        "notes": metrics.notes or [],
    }


def _format_delta(value: int) -> str:
    if value > 0:
        return f"+{value}%"
    if value < 0:
        return f"{value}%"
    return "0%"


def _build_plain_english_steps(advice: Dict[str, Any]) -> List[str]:
    steps: List[str] = []

    for axis_name in ("roll", "pitch"):
        rec = advice["axes"].get(axis_name, {})
        deltas = rec.get("deltas", {})
        if rec.get("action") != "adjust":
            steps.append(f"{axis_name.title()}: leave P/I/D/D Max/FF unchanged from this log.")
            continue

        pieces = []
        if deltas.get("d_percent", 0):
            pieces.append(f"D { _format_delta(deltas['d_percent']) }")
        if deltas.get("dmax_percent", 0):
            pieces.append(f"D Max { _format_delta(deltas['dmax_percent']) }")
        if deltas.get("p_percent", 0):
            pieces.append(f"P { _format_delta(deltas['p_percent']) }")
        if deltas.get("ff_percent", 0):
            pieces.append(f"FF { _format_delta(deltas['ff_percent']) }")
        if deltas.get("i_percent", 0):
            pieces.append(f"I { _format_delta(deltas['i_percent']) }")

        steps.append(f"{axis_name.title()}: " + ", ".join(pieces) + ".")

    steps.append("Yaw: leave unchanged unless yaw itself is twitchy, delayed, or overshooting.")
    steps.append("Change one tune pass at a time, save, fly 30-90 seconds, land, and check motor temperature.")
    return steps


def build_tuning_advice(
    df: Optional[pd.DataFrame],
    existing_analysis: Optional[Dict[str, Any]] = None,
    drone_size: str = "5",
    tuning_goal: str = "balanced",
) -> Dict[str, Any]:
    """
    Main public function.

    Returns:
        tuning_advice dict suitable for API responses and frontend display.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {
            "version": "AeroTune tuning advisor v1.0",
            "mode": "delta_percent",
            "summary": "No usable dataframe was provided, so no PID advice was generated.",
            "axes": {},
            "betaflight_steps": [],
            "slider_translation": {},
            "test_plan": [],
            "safety": [],
            "confidence": "low",
        }

    sample_rate = _estimate_sample_rate(df)

    metrics = {
        axis: _analyze_axis(df, axis, sample_rate)
        for axis in AXES
    }

    axes = {
        axis: _axis_recommendation(metrics[axis], drone_size=drone_size, axis=axis)
        for axis in AXES
    }

    adjustable = [a for a in ("roll", "pitch") if axes[a].get("action") == "adjust"]

    if adjustable:
        summary = (
            "Overshoot/bounceback behavior detected. Recommended correction: increase D/D Max on affected roll/pitch axes, "
            "slightly reduce P, and slightly reduce Feedforward. Leave I and yaw alone for this pass."
        )
    else:
        summary = (
            "No strong roll/pitch overshoot correction was detected in this specific log. If the flight felt good, hold the tune. "
            "If this was only hover/smooth flight, collect a stronger validation log with snap roll, snap pitch, flip, and dirty-air recovery."
        )

    confidence_values = [axes[a].get("confidence", "low") for a in axes]
    confidence = "high" if "high" in confidence_values else "medium" if "medium" in confidence_values else "low"

    advice = {
        "version": "AeroTune tuning advisor v1.0",
        "mode": "delta_percent",
        "drone_size": str(drone_size),
        "tuning_goal": str(tuning_goal),
        "sample_rate_hz": round(sample_rate, 2) if sample_rate else None,
        "summary": summary,
        "confidence": confidence,
        "axes": axes,
        "betaflight_steps": [],
        "slider_translation": {
            "Damping": "Maps mostly to D gain. Increase when overshoot/bounceback is present and motors are not hot.",
            "Dynamic Damping": "Maps mostly to D Max. Increase slightly with D when aggressive moves bounce back.",
            "Tracking": "Maps mostly to P/I. Lower slightly if the quad snaps past the target; raise only if it feels mushy.",
            "Stick Response": "Maps mostly to Feedforward. Lower slightly if stick release feels twitchy or jumpy.",
        },
        "test_plan": [
            "Save the tune in Betaflight and confirm the numbers stayed after reconnecting.",
            "Fly a 30-90 second validation log.",
            "Include hover, smooth forward flight, snap roll, snap pitch, one flip, and dirty-air recovery if safe.",
            "Land and touch motors. Warm is okay. Too hot to hold means reduce D/D Max before more testing.",
            "Compare the next log against this one before making another change.",
        ],
        "safety": [
            "Do not tune in low light, around people, near cars, or with loose hardware.",
            "Secure GoPro/action camera before trusting high-frequency noise readings.",
            "Only change one pass at a time. Do not stack multiple blind changes.",
            "If motors get hot, back down D/D Max.",
        ],
    }

    advice["betaflight_steps"] = _build_plain_english_steps(advice)
    return advice


def attach_tuning_advice(
    analysis: Any,
    df: Optional[pd.DataFrame],
    drone_size: str = "5",
    tuning_goal: str = "balanced",
) -> Any:
    """
    Attach tuning advice to whatever the existing analyzer returns.

    Keeps backwards compatibility:
    - If existing analyzer returns a dict, add tuning_advice.
    - If it returns another object, wrap it safely.
    """
    advice = build_tuning_advice(
        df=df,
        existing_analysis=analysis if isinstance(analysis, dict) else None,
        drone_size=drone_size,
        tuning_goal=tuning_goal,
    )

    if isinstance(analysis, dict):
        analysis["tuning_advice"] = advice
        return analysis

    return {
        "analysis": analysis,
        "tuning_advice": advice,
    }
