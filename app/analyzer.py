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

GOAL_ALIASES = {
    "efficiency": "efficient",
    "efficiency_snappy": "locked_in",
    "efficiency_floaty": "floaty",
    "snappy": "locked_in",
    "smooth": "efficient",
    "cinematic": "floaty",
}

VALID_GOALS = {"efficient", "locked_in", "floaty"}

DRONE_BANDS = {
    "5": {"low": (0.0, 35.0), "mid": (35.0, 90.0), "high": (90.0, 180.0), "propwash": (30.0, 120.0)},
    "7": {"low": (0.0, 25.0), "mid": (25.0, 70.0), "high": (70.0, 140.0), "propwash": (20.0, 100.0)},
}

SOURCE_REFERENCES = [
    "Betaflight PID Tuning Guide: https://betaflight.com/docs/wiki/guides/current/PID-Tuning-Guide",
    "P controls how tightly the quad tracks setpoint; too much P can overshoot or oscillate.",
    "I controls hold against drift, wind, CG bias, and throttle changes.",
    "D damps bounceback and propwash but amplifies noise and can heat motors.",
    "FF improves stick response without forcing P to do all the work.",
    "Yaw normally keeps D at 0; yaw is tuned mainly with P, I, and FF.",
]


@dataclass(frozen=True)
class AxisStats:
    dominant_freq_hz: Optional[float]
    low_ratio: float
    mid_ratio: float
    high_ratio: float
    propwash_ratio: float
    residual_low_ratio: float
    residual_mid_ratio: float
    residual_high_ratio: float
    rms: float
    peak_to_peak: float
    residual_rms: float
    corr: Optional[float]
    lag_ms: Optional[float]
    tracking_error_rms: float
    setpoint_rms: float
    error_ratio: float
    throttle_activity: float
    propwash_score: float
    high_throttle_score: float
    low_input_hold_error: float
    has_setpoint: bool


def _normalize_goal(goal: str) -> str:
    raw = (goal or "efficient").strip().lower().replace("-", "_").replace(" ", "_")
    raw = GOAL_ALIASES.get(raw, raw)
    return raw if raw in VALID_GOALS else "efficient"


def _finite(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        return np.array([], dtype=float)
    return arr


def _band_energy(freqs: np.ndarray, mags: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask):
        return 0.0
    return float(np.sum(np.square(mags[mask])))


def _spectral_ratios(signal: np.ndarray, sample_rate_hz: float, bands: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
    if len(signal) < 128:
        return {
            "dominant_freq_hz": None,
            "low_ratio": 0.0,
            "mid_ratio": 0.0,
            "high_ratio": 0.0,
            "propwash_ratio": 0.0,
        }

    centered = signal - np.mean(signal)
    window = np.hanning(len(centered))
    mags = np.abs(np.fft.rfft(centered * window))
    freqs = np.fft.rfftfreq(len(centered), d=1.0 / sample_rate_hz)

    if len(mags):
        mags[0] = 0.0

    low = _band_energy(freqs, mags, *bands["low"])
    mid = _band_energy(freqs, mags, *bands["mid"])
    high = _band_energy(freqs, mags, *bands["high"])
    propwash = _band_energy(freqs, mags, *bands["propwash"])
    total = max(low + mid + high, 1e-12)

    dominant_freq_hz = None
    if len(mags) > 1 and float(np.max(mags)) > 0.0:
        dominant_freq_hz = float(freqs[int(np.argmax(mags))])

    return {
        "dominant_freq_hz": dominant_freq_hz,
        "low_ratio": float(low / total),
        "mid_ratio": float(mid / total),
        "high_ratio": float(high / total),
        "propwash_ratio": float(propwash / total),
    }


def _estimate_lag_ms(setpoint: np.ndarray, gyro: np.ndarray, sample_rate_hz: float) -> Optional[float]:
    if len(setpoint) < 128 or np.std(setpoint) < 1e-6 or np.std(gyro) < 1e-6:
        return None

    max_points = 1800
    if len(setpoint) > max_points:
        step = int(np.ceil(len(setpoint) / max_points))
        setpoint = setpoint[::step]
        gyro = gyro[::step]
        effective_rate = sample_rate_hz / step
    else:
        effective_rate = sample_rate_hz

    sp = setpoint - np.mean(setpoint)
    gy = gyro - np.mean(gyro)

    max_lag = min(int(effective_rate * 0.12), len(sp) // 4)
    if max_lag < 1:
        return None

    best_score = -np.inf
    best_lag = 0

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a = gy[:lag]
            b = sp[-lag:]
        elif lag > 0:
            a = gy[lag:]
            b = sp[:-lag]
        else:
            a = gy
            b = sp

        if len(a) < 64:
            continue

        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 0:
            continue

        score = float(np.dot(a, b) / denom)
        if score > best_score:
            best_score = score
            best_lag = lag

    return float(best_lag / effective_rate * 1000.0)


def _throttle_scores(residual_abs: np.ndarray, throttle: Optional[np.ndarray]) -> Tuple[float, float, float]:
    if throttle is None or len(throttle) != len(residual_abs):
        return 0.0, 0.0, 0.0

    th = np.asarray(throttle, dtype=float)
    if not np.all(np.isfinite(th)) or np.std(th) < 1e-6:
        return 0.0, 0.0, 0.0

    dth = np.abs(np.gradient(th))
    throttle_activity = float(np.std(th) + np.mean(dth) * 8.0)

    transition_mask = dth >= np.quantile(dth, 0.80)
    calm_mask = dth <= np.quantile(dth, 0.45)

    propwash_score = 0.0
    if np.any(transition_mask) and np.any(calm_mask):
        transition_rms = float(np.sqrt(np.mean(np.square(residual_abs[transition_mask]))))
        calm_rms = float(np.sqrt(np.mean(np.square(residual_abs[calm_mask]))))
        propwash_score = transition_rms / max(calm_rms, 1e-9)

    high_mask = th >= np.quantile(th, 0.82)
    mid_mask = (th >= np.quantile(th, 0.35)) & (th <= np.quantile(th, 0.65))

    high_throttle_score = 0.0
    if np.any(high_mask) and np.any(mid_mask):
        high_rms = float(np.sqrt(np.mean(np.square(residual_abs[high_mask]))))
        mid_rms = float(np.sqrt(np.mean(np.square(residual_abs[mid_mask]))))
        high_throttle_score = high_rms / max(mid_rms, 1e-9)

    return throttle_activity, propwash_score, high_throttle_score


def _compute_axis_stats(
    gyro: np.ndarray,
    setpoint: np.ndarray,
    sample_rate_hz: float,
    bands: Dict[str, Tuple[float, float]],
    throttle: Optional[np.ndarray],
) -> AxisStats:
    gyro = _finite(gyro)
    setpoint = _finite(setpoint)
    n = min(len(gyro), len(setpoint))
    if throttle is not None:
        throttle = _finite(throttle)
        n = min(n, len(throttle))

    gyro = gyro[:n]
    setpoint = setpoint[:n]
    throttle = throttle[:n] if throttle is not None else None

    mask = np.isfinite(gyro) & np.isfinite(setpoint)
    if throttle is not None:
        mask = mask & np.isfinite(throttle)

    gyro = gyro[mask]
    setpoint = setpoint[mask]
    throttle = throttle[mask] if throttle is not None else None

    has_setpoint = bool(np.std(setpoint) > 1e-6)
    residual = gyro - setpoint if has_setpoint else gyro - np.mean(gyro)
    residual_abs = np.abs(residual - np.median(residual))

    gyro_spec = _spectral_ratios(gyro, sample_rate_hz, bands)
    residual_spec = _spectral_ratios(residual, sample_rate_hz, bands)

    corr = None
    if has_setpoint and np.std(gyro) > 1e-6:
        value = float(np.corrcoef(setpoint, gyro)[0, 1])
        corr = value if np.isfinite(value) else None

    lag_ms = _estimate_lag_ms(setpoint, gyro, sample_rate_hz) if has_setpoint else None

    tracking_error_rms = float(np.sqrt(np.mean(np.square(residual))))
    setpoint_rms = float(np.sqrt(np.mean(np.square(setpoint)))) if has_setpoint else 0.0
    error_ratio = tracking_error_rms / max(setpoint_rms, 1e-6) if has_setpoint else 1.0

    low_input_hold_error = 0.0
    if has_setpoint:
        quiet_mask = np.abs(setpoint) <= max(0.12, np.quantile(np.abs(setpoint), 0.35))
        if np.any(quiet_mask):
            low_input_hold_error = float(np.sqrt(np.mean(np.square(residual[quiet_mask]))))

    throttle_activity, propwash_score, high_throttle_score = _throttle_scores(residual_abs, throttle)

    return AxisStats(
        dominant_freq_hz=gyro_spec["dominant_freq_hz"],
        low_ratio=float(gyro_spec["low_ratio"]),
        mid_ratio=float(gyro_spec["mid_ratio"]),
        high_ratio=float(gyro_spec["high_ratio"]),
        propwash_ratio=float(gyro_spec["propwash_ratio"]),
        residual_low_ratio=float(residual_spec["low_ratio"]),
        residual_mid_ratio=float(residual_spec["mid_ratio"]),
        residual_high_ratio=float(residual_spec["high_ratio"]),
        rms=float(np.sqrt(np.mean(np.square(gyro - np.mean(gyro))))),
        peak_to_peak=float(np.max(gyro) - np.min(gyro)),
        residual_rms=tracking_error_rms,
        corr=corr,
        lag_ms=lag_ms,
        tracking_error_rms=tracking_error_rms,
        setpoint_rms=setpoint_rms,
        error_ratio=float(error_ratio),
        throttle_activity=float(throttle_activity),
        propwash_score=float(propwash_score),
        high_throttle_score=float(high_throttle_score),
        low_input_hold_error=float(low_input_hold_error),
        has_setpoint=has_setpoint,
    )


def _classify_axis(axis: str, stats: AxisStats) -> Tuple[str, float, str]:
    corr = stats.corr if stats.corr is not None else 0.0
    lag = abs(stats.lag_ms) if stats.lag_ms is not None else 0.0

    # Clean is intentionally strict. High correlation alone is not enough.
    clean = (
        stats.has_setpoint
        and corr >= 0.995
        and stats.error_ratio <= 0.100
        and stats.low_input_hold_error <= 0.060
        and stats.residual_high_ratio <= 0.30
        and stats.propwash_score <= 1.12
        and stats.high_throttle_score <= 1.12
        and lag <= 22.0
    )
    if clean:
        return "clean_tune", 0.95, "High setpoint tracking, low tracking error, no throttle-linked disturbance."

    high_noise = (
        stats.residual_high_ratio >= 0.36 and stats.error_ratio >= 0.055
    ) or (
        stats.high_ratio >= 0.22 and stats.residual_rms > 0.05
    )
    if high_noise:
        return "high_frequency_noise", 0.86, "Fast residual vibration dominates the gyro error."

    high_throttle = (
        stats.throttle_activity >= 0.08
        and stats.high_throttle_score >= 1.22
        and stats.error_ratio >= 0.055
    )
    if high_throttle:
        return "high_throttle_oscillation", 0.84, "Error rises mostly at high throttle."

    propwash = (
        axis != "yaw"
        and stats.throttle_activity >= 0.10
        and stats.error_ratio >= 0.055
        and corr >= 0.94
        and (stats.propwash_score >= 1.05 or stats.low_input_hold_error >= 0.060 or stats.propwash_ratio >= 0.00015)
    )
    if propwash:
        return "propwash_or_bounceback", 0.88, "Tracking error rises during active throttle / disturbed-air sections."

    yaw_throttle_roughness = (
        axis == "yaw"
        and stats.throttle_activity >= 0.10
        and stats.error_ratio >= 0.120
        and corr >= 0.94
    )
    if yaw_throttle_roughness:
        return "yaw_throttle_roughness", 0.82, "Yaw error rises with throttle activity; yaw D still stays at 0."

    slow_response = (
        stats.has_setpoint
        and corr >= 0.85
        and stats.lag_ms is not None
        and stats.lag_ms > 24.0
        and stats.error_ratio >= 0.045
    )
    if slow_response:
        return "slow_response", 0.83, "Gyro follows the command, but the response is delayed."

    weak_hold = (
        stats.has_setpoint
        and stats.low_input_hold_error >= 0.075
        and stats.error_ratio >= 0.055
        and corr >= 0.80
    )
    if weak_hold:
        return "drift_or_weak_hold", 0.82, "Error remains when stick input is low, which points toward weak hold."

    poor_tracking = (
        stats.has_setpoint
        and ((corr < 0.85 and stats.error_ratio >= 0.12) or stats.error_ratio >= 0.22)
    )
    if poor_tracking:
        return "poor_tracking", 0.80, "Gyro does not follow setpoint tightly enough."

    low_wobble = (
        stats.residual_low_ratio >= 0.70
        and stats.error_ratio >= 0.12
        and (corr < 0.93 or stats.low_input_hold_error >= 0.10)
    )
    if low_wobble:
        return "low_frequency_oscillation", 0.80, "Slow residual movement suggests bounce or wobble instead of clean setpoint tracking."

    mid_vibration = (
        stats.residual_mid_ratio >= 0.18 and stats.error_ratio >= 0.060
    ) or (
        stats.mid_ratio >= 0.10 and stats.residual_rms > 0.055
    )
    if mid_vibration:
        return "mid_frequency_vibration", 0.78, "Mid-band vibration is present in the gyro error."

    return "mostly_clean", 0.72, "No strong instability pattern found; changes are optional feel tuning only."


def _issue_label(issue: str, goal: str) -> str:
    labels = {
        "clean_tune": "Clean tune",
        "mostly_clean": "Mostly clean",
        "propwash_or_bounceback": "Propwash / bounceback",
        "yaw_throttle_roughness": "Yaw throttle roughness",
        "high_frequency_noise": "High-frequency noise",
        "mid_frequency_vibration": "Mid-frequency vibration",
        "high_throttle_oscillation": "High-throttle oscillation",
        "low_frequency_oscillation": "Loose / bouncing",
        "drift_or_weak_hold": "Weak hold / drift",
        "poor_tracking": "Soft tracking",
        "slow_response": "Slow stick response",
    }
    return labels.get(issue, "Unknown")


def _base_delta(axis: str, issue: str, goal: str) -> Dict[str, float]:
    delta = {"p": 0.0, "i": 0.0, "d": 0.0, "ff": 0.0}

    if issue == "clean_tune":
        pass
    elif issue == "mostly_clean":
        if goal == "locked_in":
            delta["p"] = 0.01
            delta["ff"] = 0.015
        elif goal == "floaty":
            delta["p"] = -0.015
            delta["ff"] = -0.015
    elif issue == "propwash_or_bounceback":
        delta["d"] = 0.025 if axis != "yaw" else 0.0
        delta["p"] = -0.005 if axis != "yaw" else 0.0
    elif issue == "yaw_throttle_roughness":
        delta["p"] = -0.015
        delta["i"] = 0.010
    elif issue == "high_frequency_noise":
        delta["d"] = -0.030 if axis != "yaw" else 0.0
        delta["p"] = -0.010
    elif issue == "mid_frequency_vibration":
        delta["p"] = -0.015
        delta["d"] = -0.015 if axis != "yaw" else 0.0
    elif issue == "high_throttle_oscillation":
        delta["p"] = -0.020
        delta["d"] = -0.010 if axis != "yaw" else 0.0
    elif issue == "low_frequency_oscillation":
        delta["p"] = -0.030
    elif issue == "drift_or_weak_hold":
        delta["i"] = 0.030
    elif issue == "poor_tracking":
        delta["p"] = 0.020
        delta["ff"] = 0.010
    elif issue == "slow_response":
        delta["ff"] = 0.030
        delta["p"] = 0.005

    if goal == "efficient":
        for key in delta:
            delta[key] *= 0.85
    elif goal == "locked_in":
        if issue not in {"clean_tune", "high_frequency_noise", "mid_frequency_vibration", "high_throttle_oscillation"}:
            delta["p"] += 0.005 if delta["p"] >= 0 else 0.0
            delta["ff"] += 0.005 if delta["ff"] >= 0 else 0.0
    elif goal == "floaty":
        delta["p"] *= 0.75
        delta["ff"] *= 0.70
        delta["d"] *= 0.85

    for key in delta:
        delta[key] = float(np.clip(delta[key], -0.05, 0.05))

    if axis == "yaw":
        delta["d"] = 0.0

    return delta


def _move_text(delta: Dict[str, float]) -> str:
    parts = []
    for key, label in (("p", "P"), ("i", "I"), ("d", "D"), ("ff", "FF")):
        value = delta.get(key, 0.0)
        if abs(value) >= 0.0005:
            parts.append(f"{label} {'+' if value > 0 else ''}{value * 100:.1f}%")
    return " / ".join(parts) if parts else "No PID change"


def _tuning_moves(axis: str, issue: str, goal: str) -> List[str]:
    if issue == "clean_tune":
        if goal == "locked_in":
            return [
                "No fix needed. Only change if you personally want a more locked-in feel.",
                "Optional: add a tiny amount of P or FF, then stop if it gets twitchy.",
            ]
        if goal == "floaty":
            return [
                "No fix needed. Only change if you personally want a softer cinematic feel.",
                "Optional: reduce P or FF slightly if it feels too sharp.",
            ]
        return ["No PID fix needed. Leave it alone unless flight feel gives you a reason."]

    if issue == "mostly_clean":
        if goal == "locked_in":
            return ["Mostly clean. Optional tiny P/FF increase if you want more stick connection."]
        if goal == "floaty":
            return ["Mostly clean. Optional tiny P/FF decrease if you want softer movement."]
        return ["Mostly clean. No required PID fix."]

    if issue == "propwash_or_bounceback":
        return [
            "D: raise slightly on roll/pitch. D is the damping tool for propwash and bounceback.",
            "P: do not raise P first. Lower P slightly only if it feels like overshoot or over-correction.",
            "I: leave alone unless attitude drifts during throttle changes.",
            "FF: leave alone unless stick response feels delayed.",
            "After raising D, do a short test flight and check motor temperature.",
        ]

    if issue == "yaw_throttle_roughness":
        return [
            "D: keep yaw D at 0.",
            "P: lower yaw P slightly if yaw gets rough during punch-outs or fast forward flight.",
            "I: raise yaw I slightly only if heading will not hold through throttle changes.",
            "FF: use yaw FF only for stick feel, not to fix throttle roughness.",
        ]

    if issue == "high_frequency_noise":
        return [
            "D: lower or do not raise. D amplifies high-frequency gyro noise and can heat motors.",
            "P: lower slightly if the quad sounds harsh or twitchy.",
            "Mechanical check: props, motors, frame screws, stack mounting, filtering.",
            "Do not tune around a mechanical vibration problem with more D.",
        ]

    if issue == "mid_frequency_vibration":
        return [
            "P: lower slightly to calm roughness.",
            "D: lower slightly if motors sound rough or come down warm.",
            "Mechanical check first: props, motors, frame, stack mounting.",
            "I/FF: leave alone unless hold or stick response is the actual complaint.",
        ]

    if issue == "high_throttle_oscillation":
        return [
            "P: lower slightly if oscillation mainly appears at high throttle.",
            "D: lower slightly if motors are hot or the gyro is noisy.",
            "TPA: consider more TPA / lower TPA breakpoint later, but keep PID change small first.",
            "Yaw: if yaw roughness appears at full throttle, lower yaw P slightly.",
        ]

    if issue == "low_frequency_oscillation":
        if axis == "yaw":
            return [
                "P: lower yaw P slightly.",
                "D: keep yaw D at 0.",
                "I: leave alone unless heading hold is weak.",
                "FF: leave alone unless yaw stick response is delayed.",
            ]
        return [
            "P: lower slightly first. Slow bounce often means the loop is pushing too hard or swinging past center.",
            "D: leave alone at first. Add D only later if flight feel proves bounceback/propwash remains.",
            "I: leave alone unless attitude hold is weak.",
            "FF: leave alone unless stick response is delayed.",
        ]

    if issue == "drift_or_weak_hold":
        return [
            "I: raise slightly. I helps hold attitude against wind, CG bias, and throttle changes.",
            "P: leave mostly alone unless the quad also feels soft.",
            "D: leave alone unless propwash or bounceback is present.",
            "FF: leave alone unless stick response feels delayed.",
        ]

    if issue == "poor_tracking":
        return [
            "P: raise slightly if the quad feels soft and does not follow the sticks.",
            "FF: raise slightly if the main problem is delayed stick feel.",
            "D: leave alone unless bounceback or propwash appears.",
            "I: leave alone unless attitude hold is weak.",
        ]

    if issue == "slow_response":
        return [
            "FF: raise slightly first. FF improves stick response without making P do all the work.",
            "P: raise slightly only if it still feels soft after FF.",
            "D: leave alone unless bounceback or propwash appears.",
            "I: leave alone unless attitude hold is weak.",
        ]

    return ["No clear PID issue found. Retake a better log before changing PID."]


def _recommendation(axis: str, issue: str, delta: Dict[str, float], goal: str) -> str:
    axis_label = axis.upper()
    move = _move_text(delta)

    if issue == "clean_tune":
        if goal == "locked_in":
            return f"{axis_label}: clean tune. No fix needed. Only add tiny P/FF if you want more locked-in feel."
        if goal == "floaty":
            return f"{axis_label}: clean tune. No fix needed. Only soften P/FF if you want more cinematic feel."
        return f"{axis_label}: clean tune. No PID change needed."

    if issue == "mostly_clean":
        return f"{axis_label}: mostly clean. No required fix." if move == "No PID change" else f"{axis_label}: mostly clean. {move} is optional feel tuning, not a required fix."

    issue_text = {
        "propwash_or_bounceback": "propwash / bounceback detected",
        "yaw_throttle_roughness": "yaw throttle roughness detected",
        "high_frequency_noise": "high-frequency noise detected",
        "mid_frequency_vibration": "mid-frequency vibration detected",
        "high_throttle_oscillation": "high-throttle oscillation detected",
        "low_frequency_oscillation": "slow bounce / wobble detected",
        "drift_or_weak_hold": "weak hold / drift detected",
        "poor_tracking": "soft tracking detected",
        "slow_response": "slow stick response detected",
    }.get(issue, "issue detected")

    return f"{axis_label}: {issue_text}. Suggested change: {move}."


def _axis_payload(axis: str, stats: AxisStats, issue: str, confidence: float, reason: str, goal: str) -> Dict[str, Any]:
    delta = _base_delta(axis, issue, goal)
    moves = _tuning_moves(axis, issue, goal)

    return {
        "axis": axis,
        "axis_name": axis.upper(),
        "status": "ok",
        "issue": issue,
        "issue_label": _issue_label(issue, goal),
        "propwash_detected": issue == "propwash_or_bounceback",
        "feel": _issue_label(issue, goal),
        "confidence": round(confidence, 3),
        "confidence_reason": reason,
        "valid_for_pid": issue not in {"unknown"},
        "pid_delta_pct": {key: round(value, 4) for key, value in delta.items()},
        "recommendation_text": _recommendation(axis, issue, delta, goal),
        "actions": moves,
        "tuning_moves": moves,
        "warnings": _warnings(issue, axis),
        "signal": {
            "dominant_freq_hz": None if stats.dominant_freq_hz is None else round(stats.dominant_freq_hz, 2),
            "low_ratio": round(stats.low_ratio, 4),
            "mid_ratio": round(stats.mid_ratio, 4),
            "high_ratio": round(stats.high_ratio, 4),
            "propwash_ratio": round(stats.propwash_ratio, 4),
            "residual_low_ratio": round(stats.residual_low_ratio, 4),
            "residual_mid_ratio": round(stats.residual_mid_ratio, 4),
            "residual_high_ratio": round(stats.residual_high_ratio, 4),
            "rms": round(stats.rms, 4),
            "peak_to_peak": round(stats.peak_to_peak, 4),
            "residual_rms": round(stats.residual_rms, 4),
            "propwash_score": round(stats.propwash_score, 4),
            "high_throttle_score": round(stats.high_throttle_score, 4),
            "throttle_activity": round(stats.throttle_activity, 4),
        },
        "tracking": {
            "usable": stats.has_setpoint,
            "corr": None if stats.corr is None else round(stats.corr, 4),
            "lag_ms": None if stats.lag_ms is None else round(stats.lag_ms, 2),
            "tracking_error_rms": round(stats.tracking_error_rms, 4),
            "error_ratio": round(stats.error_ratio, 4),
            "low_input_hold_error": round(stats.low_input_hold_error, 4),
        },
    }


def _warnings(issue: str, axis: str) -> List[str]:
    warnings: List[str] = []
    if issue in {"propwash_or_bounceback", "high_frequency_noise", "mid_frequency_vibration"} and axis != "yaw":
        warnings.append("Check motor temperature after any D change.")
    if axis == "yaw":
        warnings.append("Yaw D should normally stay at 0; tune yaw with P, I, and FF.")
    if issue in {"high_frequency_noise", "mid_frequency_vibration"}:
        warnings.append("Fix mechanical vibration before chasing PID values.")
    return warnings


def _global_summary(axes: Dict[str, Any], goal: str) -> str:
    if not axes:
        return "No usable axes found."
    issues = [a["issue"] for a in axes.values()]
    if all(issue == "clean_tune" for issue in issues):
        return "Clean tune detected. No PID fix needed unless you want a different feel."
    if all(issue in {"clean_tune", "mostly_clean"} for issue in issues):
        return "Mostly clean tune. Any changes are optional feel tuning, not required fixes."

    problem_labels = [f"{a['axis_name']}: {a['issue_label']}" for a in axes.values() if a["issue"] not in {"clean_tune", "mostly_clean"}]
    return "Problems detected — " + " | ".join(problem_labels)


def detect_oscillation(df: pd.DataFrame, drone_size: str = "7", tuning_goal: str = "efficient") -> Dict[str, Any]:
    tuning_goal = _normalize_goal(tuning_goal)

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
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

    if "time" not in df.columns:
        return {
            "status": "invalid",
            "valid_for_pid": False,
            "summary": "Missing time column.",
            "warnings": ["AeroTune requires increasing timestamps."],
            "global_actions": [],
            "axes": {},
            "recommendations": [],
            "tuning_goal": tuning_goal,
            "source_references": SOURCE_REFERENCES,
        }

    time = np.asarray(df["time"], dtype=float)
    if len(time) < 128:
        return {
            "status": "invalid",
            "valid_for_pid": False,
            "summary": "Not enough log samples.",
            "warnings": ["Use a longer Blackbox CSV."],
            "global_actions": [],
            "axes": {},
            "recommendations": [],
            "tuning_goal": tuning_goal,
            "source_references": SOURCE_REFERENCES,
        }

    dt = np.diff(time)
    good_dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(good_dt) == 0:
        return {
            "status": "invalid",
            "valid_for_pid": False,
            "summary": "Invalid timestamps.",
            "warnings": ["Timestamps must be increasing."],
            "global_actions": [],
            "axes": {},
            "recommendations": [],
            "tuning_goal": tuning_goal,
            "source_references": SOURCE_REFERENCES,
        }

    sample_rate_hz = float(1.0 / np.median(good_dt))
    duration_s = float(time[-1] - time[0])
    bands = DRONE_BANDS.get(str(drone_size), DRONE_BANDS["7"])

    throttle = np.asarray(df["throttle"], dtype=float) if "throttle" in df.columns else None
    axes: Dict[str, Any] = {}

    for axis, (gyro_col, setpoint_col) in AXIS_MAP.items():
        if gyro_col not in df.columns:
            continue

        gyro = np.asarray(df[gyro_col], dtype=float)
        setpoint = np.asarray(df[setpoint_col], dtype=float) if setpoint_col in df.columns else np.zeros_like(gyro)

        try:
            stats = _compute_axis_stats(gyro, setpoint, sample_rate_hz, bands, throttle)
            issue, confidence, reason = _classify_axis(axis, stats)
            axes[axis] = _axis_payload(axis, stats, issue, confidence, reason, tuning_goal)
        except Exception as exc:
            axes[axis] = {
                "axis": axis,
                "axis_name": axis.upper(),
                "status": "error",
                "issue": "unknown",
                "issue_label": "Analysis error",
                "propwash_detected": False,
                "feel": "Unknown",
                "confidence": 0.0,
                "confidence_reason": str(exc),
                "valid_for_pid": False,
                "pid_delta_pct": {"p": 0.0, "i": 0.0, "d": 0.0, "ff": 0.0},
                "recommendation_text": f"{axis.upper()}: analysis failed. Retake/export the log.",
                "actions": ["Retake/export the log before changing PID."],
                "tuning_moves": ["Retake/export the log before changing PID."],
                "warnings": [str(exc)],
            }

    recommendations = []
    for axis in ("roll", "pitch", "yaw"):
        if axis in axes:
            item = axes[axis]
            recommendations.append({
                "axis": axis,
                "issue": item["issue"],
                "issue_label": item["issue_label"],
                "propwash_detected": item["propwash_detected"],
                "feel": item["feel"],
                "valid_for_pid": item["valid_for_pid"],
                "confidence": item["confidence"],
                "confidence_reason": item.get("confidence_reason", ""),
                "pid_delta_pct": item["pid_delta_pct"],
                "tuning_moves": item["tuning_moves"],
                "recommendation_text": item["recommendation_text"],
            })

    global_warnings = [
        "Use small changes and test one axis/change at a time.",
        "After raising D, do a short test flight and check motor temperature.",
        "If a log looks clean, do not chase PID changes just because a tool exists.",
    ]

    return {
        "status": "ok" if axes else "invalid",
        "valid_for_pid": bool(axes),
        "summary": _global_summary(axes, tuning_goal),
        "warnings": global_warnings,
        "global_actions": [
            "Match the recommendation to real flight feel before changing values.",
            "Fix mechanical noise before increasing D.",
            "Yaw D should normally remain 0.",
        ],
        "axes": axes,
        "recommendations": recommendations,
        "sample_rate_hz": round(sample_rate_hz, 2),
        "duration_s": round(duration_s, 3),
        "drone_size": str(drone_size),
        "tuning_goal": tuning_goal,
        "source_references": SOURCE_REFERENCES,
    }
