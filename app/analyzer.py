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
    "5": {
        "low": (0.0, 35.0),
        "mid": (35.0, 90.0),
        "high": (90.0, 180.0),
        "propwash": (30.0, 120.0),
    },
    "7": {
        "low": (0.0, 25.0),
        "mid": (25.0, 70.0),
        "high": (70.0, 140.0),
        "propwash": (20.0, 100.0),
    },
}

SOURCE_REFERENCES = [
    "Betaflight PID Tuning Guide: https://betaflight.com/docs/wiki/guides/current/PID-Tuning-Guide",
    "Betaflight PID Tuning Tab: https://betaflight.com/docs/wiki/app/pid-tuning-tab",
    "Betaflight 4.2 tuning notes: https://betaflight.com/docs/wiki/tuning/4-2-Tuning-Notes",
    "Rules used: P improves tracking but can oscillate; I improves hold; D damps bounceback/propwash but can heat motors/noise; FF improves stick response; yaw D normally stays 0.",
]


@dataclass(frozen=True)
class AxisStats:
    dominant_freq_hz: Optional[float]
    low_ratio: float
    mid_ratio: float
    high_ratio: float
    propwash_ratio: float
    rms: float
    peak_to_peak: float
    corr: Optional[float]
    lag_ms: Optional[float]
    tracking_error_rms: float
    setpoint_rms: float
    error_ratio: float
    residual_rms: float
    residual_low_ratio: float
    residual_mid_ratio: float
    residual_high_ratio: float
    propwash_score: float
    throttle_activity: float
    high_throttle_score: float
    low_input_hold_error: float
    has_setpoint: bool


def _normalize_goal(goal: str) -> str:
    raw = (goal or "efficient").strip().lower().replace("-", "_").replace(" ", "_")
    raw = GOAL_ALIASES.get(raw, raw)
    return raw if raw in VALID_GOALS else "efficient"


def _safe_array(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)] if arr.ndim == 1 else np.array([], dtype=float)


def _band_energy(freqs: np.ndarray, mags: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask):
        return 0.0
    return float(np.sum(np.square(mags[mask])))


def _spectral_ratios(signal: np.ndarray, sample_rate_hz: float, bands: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
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
        "propwash_ratio": float(propwash / max(total, 1e-12)),
    }


def _estimate_lag_ms(setpoint: np.ndarray, gyro: np.ndarray, sample_rate_hz: float) -> Optional[float]:
    if len(setpoint) < 128 or np.std(setpoint) < 1e-6 or np.std(gyro) < 1e-6:
        return None

    max_points = 2500
    if len(setpoint) > max_points:
        step = int(np.ceil(len(setpoint) / max_points))
        setpoint = setpoint[::step]
        gyro = gyro[::step]
        effective_sample_rate = sample_rate_hz / step
    else:
        effective_sample_rate = sample_rate_hz

    sp = setpoint - np.mean(setpoint)
    gy = gyro - np.mean(gyro)

    max_lag = min(int(effective_sample_rate * 0.15), len(sp) // 4)
    if max_lag < 1:
        return None

    lags = np.arange(-max_lag, max_lag + 1)
    scores = np.full(len(lags), -np.inf, dtype=float)

    for idx, lag in enumerate(lags):
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
        if denom > 0:
            scores[idx] = float(np.dot(a, b) / denom)

    if not np.isfinite(scores).any():
        return None

    best_lag = int(lags[int(np.argmax(scores))])
    return float(best_lag / effective_sample_rate * 1000.0)


def _compute_throttle_scores(
    residual_abs: np.ndarray,
    throttle: Optional[np.ndarray],
) -> Tuple[float, float, float]:
    if throttle is None or len(throttle) != len(residual_abs):
        return 0.0, 0.0, 0.0

    th = np.asarray(throttle, dtype=float)
    if not np.all(np.isfinite(th)) or np.std(th) < 1e-6:
        return 0.0, 0.0, 0.0

    dth = np.abs(np.gradient(th))
    throttle_activity = float(np.std(th) + np.mean(dth) * 8.0)

    transition_cut = float(np.quantile(dth, 0.82))
    calm_cut = float(np.quantile(dth, 0.45))
    transition_mask = dth >= transition_cut
    calm_mask = dth <= calm_cut

    propwash_score = 0.0
    if np.any(transition_mask) and np.any(calm_mask):
        transition_rms = float(np.sqrt(np.mean(np.square(residual_abs[transition_mask]))))
        calm_rms = float(np.sqrt(np.mean(np.square(residual_abs[calm_mask]))))
        propwash_score = transition_rms / max(calm_rms, 1e-6)

    high_mask = th >= np.quantile(th, 0.80)
    mid_mask = (th >= np.quantile(th, 0.35)) & (th <= np.quantile(th, 0.65))

    high_throttle_score = 0.0
    if np.any(high_mask) and np.any(mid_mask):
        high_rms = float(np.sqrt(np.mean(np.square(residual_abs[high_mask]))))
        mid_rms = float(np.sqrt(np.mean(np.square(residual_abs[mid_mask]))))
        high_throttle_score = high_rms / max(mid_rms, 1e-6)

    return throttle_activity, propwash_score, high_throttle_score


def _compute_axis_stats(
    gyro: np.ndarray,
    setpoint: np.ndarray,
    sample_rate_hz: float,
    bands: Dict[str, Tuple[float, float]],
    throttle: Optional[np.ndarray],
) -> AxisStats:
    gyro = np.asarray(gyro, dtype=float)
    setpoint = np.asarray(setpoint, dtype=float)

    n = min(len(gyro), len(setpoint))
    if throttle is not None:
        n = min(n, len(throttle))

    gyro = gyro[:n]
    setpoint = setpoint[:n]
    throttle = throttle[:n] if throttle is not None else None

    has_setpoint = bool(np.std(setpoint) > 1e-6)

    tracking_error = setpoint - gyro if has_setpoint else gyro - np.mean(gyro)
    residual = tracking_error - np.mean(tracking_error)

    gyro_spec = _spectral_ratios(gyro, sample_rate_hz, bands)
    residual_spec = _spectral_ratios(residual, sample_rate_hz, bands)

    corr = None
    if has_setpoint and np.std(gyro) > 1e-6:
        value = float(np.corrcoef(setpoint, gyro)[0, 1])
        corr = value if np.isfinite(value) else None

    lag_ms = _estimate_lag_ms(setpoint, gyro, sample_rate_hz) if has_setpoint else None

    tracking_error_rms = float(np.sqrt(np.mean(np.square(tracking_error))))
    setpoint_rms = float(np.sqrt(np.mean(np.square(setpoint)))) if has_setpoint else 0.0
    error_ratio = tracking_error_rms / max(setpoint_rms, float(np.sqrt(np.mean(np.square(gyro)))) * 2.0, 1e-6)

    residual_abs = np.abs(residual)
    throttle_activity, propwash_score, high_throttle_score = _compute_throttle_scores(residual_abs, throttle)

    low_input_hold_error = 0.0
    if has_setpoint:
        quiet_cut = max(0.10 * max(float(np.max(np.abs(setpoint))), 1e-6), float(np.quantile(np.abs(setpoint), 0.35)))
        quiet_mask = np.abs(setpoint) <= quiet_cut
        if np.any(quiet_mask):
            low_input_hold_error = float(np.sqrt(np.mean(np.square(tracking_error[quiet_mask]))))

    return AxisStats(
        dominant_freq_hz=gyro_spec["dominant_freq_hz"],
        low_ratio=gyro_spec["low_ratio"],
        mid_ratio=gyro_spec["mid_ratio"],
        high_ratio=gyro_spec["high_ratio"],
        propwash_ratio=gyro_spec["propwash_ratio"],
        rms=float(np.sqrt(np.mean(np.square(gyro - np.mean(gyro))))),
        peak_to_peak=float(np.max(gyro) - np.min(gyro)),
        corr=corr,
        lag_ms=lag_ms,
        tracking_error_rms=tracking_error_rms,
        setpoint_rms=setpoint_rms,
        error_ratio=float(error_ratio),
        residual_rms=float(np.sqrt(np.mean(np.square(residual)))),
        residual_low_ratio=residual_spec["low_ratio"],
        residual_mid_ratio=residual_spec["mid_ratio"],
        residual_high_ratio=residual_spec["high_ratio"],
        propwash_score=float(propwash_score),
        throttle_activity=float(throttle_activity),
        high_throttle_score=float(high_throttle_score),
        low_input_hold_error=low_input_hold_error,
        has_setpoint=has_setpoint,
    )


def _is_clean(stats: AxisStats) -> bool:
    if stats.has_setpoint and stats.corr is not None:
        return (
            stats.corr >= 0.94
            and stats.error_ratio <= 0.16
            and abs(stats.lag_ms or 0.0) <= 30.0
            and stats.residual_mid_ratio < 0.35
            and stats.residual_high_ratio < 0.35
            and stats.propwash_score < 1.45
            and stats.high_throttle_score < 1.45
        )

    return (
        stats.rms > 1e-6
        and stats.mid_ratio < 0.30
        and stats.high_ratio < 0.30
        and stats.propwash_score < 1.35
        and stats.high_throttle_score < 1.35
    )


def _classify_issue(stats: AxisStats) -> Tuple[str, float, List[str]]:
    reasons: List[str] = []

    confidence = 0.42
    if stats.has_setpoint:
        confidence += 0.12
        reasons.append("setpoint data present")
    if stats.corr is not None:
        confidence += 0.10
        reasons.append(f"gyro/setpoint correlation {stats.corr:.2f}")
    if stats.lag_ms is not None:
        confidence += 0.06
        reasons.append(f"response lag {stats.lag_ms:.1f}ms")
    if stats.throttle_activity > 0:
        confidence += 0.05
        reasons.append("throttle data present")

    if _is_clean(stats):
        reasons.append("high tracking quality and no dominant residual disturbance")
        return "clean_tune", min(confidence + 0.20, 1.0), reasons

    if (
        stats.residual_high_ratio >= 0.42
        or (stats.high_ratio >= 0.42 and stats.error_ratio > 0.12)
    ):
        reasons.append("high-frequency residual/noise energy is dominant")
        return "high_frequency_noise", min(confidence + 0.20, 1.0), reasons

    if (
        stats.high_throttle_score >= 1.55
        and stats.throttle_activity > 0.015
        and (stats.residual_mid_ratio + stats.residual_high_ratio) >= 0.30
    ):
        reasons.append("residual motion increases during high throttle")
        return "high_throttle_oscillation", min(confidence + 0.20, 1.0), reasons

    if (
        stats.propwash_score >= 1.55
        and stats.throttle_activity > 0.015
        and (stats.residual_mid_ratio + stats.residual_high_ratio) >= 0.28
        and stats.error_ratio > 0.08
    ):
        reasons.append("residual error rises during throttle transitions / dirty-air style events")
        return "propwash", min(confidence + 0.22, 1.0), reasons

    if (
        stats.error_ratio >= 0.18
        and stats.residual_low_ratio >= 0.45
        and stats.peak_to_peak > 3.5 * max(stats.rms, 1e-6)
        and (stats.corr is None or stats.corr >= 0.45)
    ):
        reasons.append("low-frequency residual error with overshoot-like peaks")
        return "bounceback", min(confidence + 0.18, 1.0), reasons

    if (
        stats.residual_mid_ratio >= 0.45
        or (stats.mid_ratio >= 0.45 and stats.error_ratio > 0.12)
    ):
        reasons.append("mid-frequency vibration energy is dominant")
        return "mid_frequency_vibration", min(confidence + 0.18, 1.0), reasons

    if stats.has_setpoint and stats.low_input_hold_error > max(0.20 * max(stats.setpoint_rms, 1e-6), 0.08):
        if stats.error_ratio >= 0.20:
            reasons.append("holding error remains during low stick input")
            return "weak_hold_or_drift", min(confidence + 0.16, 1.0), reasons

    if stats.has_setpoint and stats.corr is not None and stats.corr < 0.70 and stats.error_ratio > 0.20:
        reasons.append("gyro does not follow setpoint well")
        return "poor_tracking", min(confidence + 0.16, 1.0), reasons

    if stats.has_setpoint and stats.lag_ms is not None and stats.lag_ms > 35.0 and stats.error_ratio > 0.10:
        reasons.append("gyro response lags behind setpoint")
        return "slow_response", min(confidence + 0.16, 1.0), reasons

    if (
        stats.residual_low_ratio >= 0.65
        and stats.error_ratio > 0.22
        and not (stats.corr is not None and stats.corr >= 0.90)
    ):
        reasons.append("low-frequency uncommanded wobble remains after tracking checks")
        return "low_frequency_wobble", min(confidence + 0.14, 1.0), reasons

    reasons.append("minor movement only; no safe PID problem identified")
    return "clean_tune", min(confidence, 0.82), reasons


def _base_delta_for_issue(axis: str, issue: str, goal: str) -> Dict[str, float]:
    delta = {"p": 0.0, "i": 0.0, "d": 0.0, "ff": 0.0}

    if issue == "clean_tune":
        if goal == "locked_in":
            delta["p"] = 0.008 if axis != "yaw" else 0.004
            delta["ff"] = 0.012
        elif goal == "floaty":
            delta["p"] = -0.010 if axis != "yaw" else -0.006
            delta["ff"] = -0.010
        return delta

    if issue == "propwash":
        if axis != "yaw":
            delta["d"] = 0.025
        else:
            delta["p"] = -0.006

    elif issue == "bounceback":
        delta["p"] = -0.020
        if axis != "yaw":
            delta["d"] = 0.010

    elif issue == "high_frequency_noise":
        delta["p"] = -0.015
        if axis != "yaw":
            delta["d"] = -0.025

    elif issue == "mid_frequency_vibration":
        delta["p"] = -0.015
        if axis != "yaw":
            delta["d"] = -0.015

    elif issue == "high_throttle_oscillation":
        delta["p"] = -0.015
        if axis != "yaw":
            delta["d"] = -0.015

    elif issue == "low_frequency_wobble":
        delta["p"] = -0.025
        # Do not automatically move D here. Slow wobble is P-first.

    elif issue == "weak_hold_or_drift":
        delta["i"] = 0.025

    elif issue == "poor_tracking":
        delta["p"] = 0.018
        delta["ff"] = 0.012

    elif issue == "slow_response":
        delta["ff"] = 0.030
        delta["p"] = 0.006

    if goal == "efficient":
        delta["p"] *= 0.80
        delta["i"] *= 0.85
        delta["d"] *= 0.75
        delta["ff"] *= 0.80
    elif goal == "locked_in":
        if issue in {"poor_tracking", "slow_response"}:
            delta["p"] += 0.006 if axis != "yaw" else 0.003
            delta["ff"] += 0.010
        elif issue == "propwash" and axis != "yaw":
            delta["d"] += 0.004
    elif goal == "floaty":
        if issue in {"poor_tracking", "slow_response"}:
            delta["p"] *= 0.70
            delta["ff"] *= 0.70
        elif issue == "bounceback":
            delta["p"] *= 1.05

    for key in delta:
        delta[key] = float(np.clip(delta[key], -0.04, 0.04))

    if axis == "yaw":
        delta["d"] = 0.0

    return delta


def _issue_label(issue: str, goal: str) -> str:
    labels = {
        "clean_tune": {
            "efficient": "Clean tune",
            "locked_in": "Clean tune — optional sharper feel",
            "floaty": "Clean tune — optional softer feel",
        },
        "propwash": "Propwash",
        "bounceback": "Bounceback / overshoot",
        "high_frequency_noise": "Harsh / noisy",
        "mid_frequency_vibration": "Shaky / rough",
        "high_throttle_oscillation": "High-throttle oscillation",
        "low_frequency_wobble": "Loose / slow wobble",
        "weak_hold_or_drift": "Weak hold / drifting",
        "poor_tracking": "Soft tracking",
        "slow_response": "Delayed / soft response",
    }

    value = labels.get(issue, "Clean tune")
    if isinstance(value, dict):
        return value.get(goal, "Clean tune")
    return value


def _tuning_moves(axis: str, issue: str, goal: str) -> List[str]:
    if issue == "clean_tune":
        if goal == "locked_in":
            return [
                "No required PID fix.",
                "Optional feel change: add a tiny amount of P or FF only if you want a more locked-in response.",
                "Stop if it gets twitchy, rough, or motors get warmer.",
            ]
        if goal == "floaty":
            return [
                "No required PID fix.",
                "Optional feel change: lower P or FF slightly only if you want smoother cinematic movement.",
                "Stop if it starts feeling loose or delayed.",
            ]
        return [
            "No PID change needed.",
            "Only change the tune if the actual flight feel gives you a reason.",
        ]

    if issue == "propwash":
        if axis == "yaw":
            return [
                "D: keep at 0 on yaw.",
                "P: lower slightly only if yaw shakes on punch-outs or fast forward flight.",
                "I: raise slightly only if yaw will not hold heading through throttle changes.",
                "FF: leave alone unless yaw stick response feels delayed.",
            ]
        return [
            "D: raise slightly first. D is the main damping tool for propwash.",
            "P: do not raise P first.",
            "I: leave alone unless the quad also drifts or loses attitude.",
            "FF: leave alone unless stick response itself feels lazy.",
            "After raising D, do a short test and check motor temperature.",
        ]

    if issue == "bounceback":
        if axis == "yaw":
            return [
                "D: keep at 0 on yaw.",
                "P: lower yaw P slightly if it snaps past center or shakes after yaw inputs.",
                "I: raise only if yaw heading will not hold.",
                "FF: lower slightly only if yaw snaps too hard from stick input.",
            ]
        return [
            "P: lower slightly first if it snaps past center or bounces after a move.",
            "D: add slightly only if it still needs damping after P is calmer.",
            "I: leave alone unless there is slow drift or windup-like bounce.",
            "FF: lower slightly only if bounce is caused by aggressive stick input.",
        ]

    if issue == "high_frequency_noise":
        return [
            "D: lower or do not raise it. D amplifies gyro noise and can heat motors.",
            "P: lower slightly if the quad sounds harsh or twitchy.",
            "I: leave alone unless the quad drifts.",
            "FF: leave alone unless stick response is the issue.",
            "Check props, motors, frame screws, filters, and stack mounting.",
        ]

    if issue == "mid_frequency_vibration":
        return [
            "P: lower slightly if the quad feels rough.",
            "D: lower slightly if motors sound rough or come down warm.",
            "I: leave alone unless attitude hold is weak.",
            "FF: leave alone unless stick response is the complaint.",
            "Check props, motors, frame screws, and stack mounting before blaming only PID.",
        ]

    if issue == "high_throttle_oscillation":
        return [
            "P/D: reduce slightly at high throttle or use TPA-style reduction.",
            "D: do not raise until high-throttle roughness is gone.",
            "I: leave alone unless throttle changes cause attitude drift.",
            "FF: leave alone unless stick response is delayed.",
        ]

    if issue == "low_frequency_wobble":
        if axis == "yaw":
            return [
                "P: lower yaw P slightly.",
                "D: keep at 0 on yaw.",
                "I: leave alone unless yaw heading drifts.",
                "FF: leave alone unless yaw stick response is delayed.",
            ]
        return [
            "P: lower slightly first.",
            "D: leave alone at first. Add D carefully only if flight tests still show bounceback or propwash.",
            "I: leave alone unless the quad drifts or will not hold attitude.",
            "FF: leave alone unless stick feel is delayed.",
        ]

    if issue == "weak_hold_or_drift":
        return [
            "I: raise slightly. I helps hold attitude through wind, CG bias, and throttle changes.",
            "P: leave mostly alone unless the quad also feels soft.",
            "D: leave alone unless bounceback or propwash appears.",
            "FF: leave alone unless stick response feels delayed.",
        ]

    if issue == "poor_tracking":
        return [
            "P: raise slightly if the quad feels soft and does not follow the sticks.",
            "FF: raise slightly if it mainly feels delayed.",
            "D: leave alone unless bounceback or propwash appears.",
            "I: leave alone unless it will not hold attitude through throttle changes.",
        ]

    if issue == "slow_response":
        return [
            "FF: raise slightly first. FF improves stick response without making P do all the work.",
            "P: raise slightly only if it still feels soft after FF.",
            "D: leave alone unless bounceback or propwash appears.",
            "I: leave alone unless attitude hold is weak.",
        ]

    return ["No safe PID recommendation from this log."]


def _recommendation_text(axis: str, issue: str, goal: str) -> str:
    axis_label = axis.upper()

    if issue == "clean_tune":
        if goal == "locked_in":
            return f"{axis_label}: clean tune. No fix needed. Optional: make it slightly more locked-in only if that is the feel you want."
        if goal == "floaty":
            return f"{axis_label}: clean tune. No fix needed. Optional: soften it slightly only if you want smoother cinematic movement."
        return f"{axis_label}: clean tune. No PID change needed."

    if issue == "propwash":
        if axis == "yaw":
            return f"{axis_label}: dirty-air / yaw shake behavior detected. Keep yaw D at 0; only adjust yaw P/I/FF if flight feel confirms it."
        return f"{axis_label}: propwash detected. Add a small amount of D damping, then check motor temperature."

    if issue == "bounceback":
        if axis == "yaw":
            return f"{axis_label}: yaw bounceback detected. Keep yaw D at 0 and lower yaw P slightly if flight feel confirms it."
        return f"{axis_label}: bounceback detected. Lower P slightly first; add D only if damping is still needed."

    if issue == "high_frequency_noise":
        return f"{axis_label}: high-frequency noise detected. Do not raise D; reduce noise/mechanical vibration first."

    if issue == "mid_frequency_vibration":
        return f"{axis_label}: vibration detected. Check mechanical sources first; soften P/D slightly only if the build is clean."

    if issue == "high_throttle_oscillation":
        return f"{axis_label}: high-throttle oscillation detected. Reduce high-throttle P/D influence or use TPA-style logic."

    if issue == "low_frequency_wobble":
        if axis == "yaw":
            return f"{axis_label}: slow yaw wobble detected. Lower yaw P slightly and keep yaw D at 0."
        return f"{axis_label}: slow wobble detected. Lower P slightly first and leave D alone unless bounceback/propwash remains."

    if issue == "weak_hold_or_drift":
        return f"{axis_label}: weak hold detected. Raise I slightly if the quad drifts or loses attitude through throttle changes."

    if issue == "poor_tracking":
        return f"{axis_label}: soft tracking detected. Raise P slightly for authority, or FF if it mainly feels delayed."

    if issue == "slow_response":
        return f"{axis_label}: slow response detected. Raise FF first; add P only if it still feels soft."

    return f"{axis_label}: no clear PID problem detected."


def _summary_from_axes(axes: Dict[str, Any], goal: str) -> str:
    issues = [axis["issue"] for axis in axes.values()]
    if issues and all(issue == "clean_tune" for issue in issues):
        if goal == "locked_in":
            return "Clean tune detected. No fix needed; only make optional small changes if you want a more locked-in feel."
        if goal == "floaty":
            return "Clean tune detected. No fix needed; only make optional small changes if you want a softer cinematic feel."
        return "Clean tune detected. No PID change needed."

    problem_labels = []
    for name, axis in axes.items():
        if axis["issue"] != "clean_tune":
            problem_labels.append(f"{name.upper()}: {axis['feel']}")

    return "Analysis complete. " + " | ".join(problem_labels)


def _global_actions(goal: str, axes: Dict[str, Any]) -> List[str]:
    actions = ["Make one small change at a time and test again."]

    if any(axis["issue"] in {"propwash", "bounceback"} for axis in axes.values()):
        actions.append("If D goes up, do a short test flight and check motor temperature.")

    if any(axis["issue"] in {"high_frequency_noise", "mid_frequency_vibration"} for axis in axes.values()):
        actions.append("Fix mechanical noise before chasing PID changes.")

    if all(axis["issue"] == "clean_tune" for axis in axes.values()):
        if goal == "locked_in":
            actions.append("Optional: add tiny P/FF only for more locked-in feel.")
        elif goal == "floaty":
            actions.append("Optional: reduce tiny P/FF only for smoother cinematic feel.")
        else:
            actions.append("Leave the tune alone unless flight feel gives a reason.")

    return actions


def _warnings(goal: str, axes: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []

    if any(axis["pid_delta_pct"]["d"] > 0 for axis in axes.values()):
        warnings.append("D increases can heat motors. Land quickly and check motor temperature.")

    if any(axis["issue"] in {"high_frequency_noise", "mid_frequency_vibration"} for axis in axes.values()):
        warnings.append("Vibration/noise can be mechanical. Props, motors, frame, and stack mounting matter before PID.")

    if goal == "locked_in":
        warnings.append("Locked-in feel can become twitchy if P/FF are pushed too far.")
    elif goal == "floaty":
        warnings.append("Floaty feel can become loose if P/FF are lowered too far.")

    return warnings


def detect_oscillation(
    df: pd.DataFrame,
    drone_size: str = "7",
    tuning_goal: str = "efficient",
) -> Dict[str, Any]:
    goal = _normalize_goal(tuning_goal)

    if df is None or df.empty:
        return {
            "status": "invalid",
            "valid_for_pid": False,
            "summary": "No usable data.",
            "warnings": ["Parsed log is empty."],
            "global_actions": [],
            "axes": {},
            "recommendations": [],
            "tuning_goal": goal,
            "source_references": SOURCE_REFERENCES,
        }

    if "time" not in df.columns:
        return {
            "status": "invalid",
            "valid_for_pid": False,
            "summary": "Missing time column.",
            "warnings": ["AeroTune requires a time column."],
            "global_actions": [],
            "axes": {},
            "recommendations": [],
            "tuning_goal": goal,
            "source_references": SOURCE_REFERENCES,
        }

    time = np.asarray(df["time"], dtype=float)
    if len(time) < 128:
        return {
            "status": "invalid",
            "valid_for_pid": False,
            "summary": "Not enough log samples.",
            "warnings": ["Need at least 128 rows for usable analysis."],
            "global_actions": [],
            "axes": {},
            "recommendations": [],
            "tuning_goal": goal,
            "source_references": SOURCE_REFERENCES,
        }

    dt = np.diff(time)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return {
            "status": "invalid",
            "valid_for_pid": False,
            "summary": "Time spacing is invalid.",
            "warnings": ["Timestamps must be increasing."],
            "global_actions": [],
            "axes": {},
            "recommendations": [],
            "tuning_goal": goal,
            "source_references": SOURCE_REFERENCES,
        }

    sample_rate_hz = float(1.0 / np.median(dt))
    duration_s = float(time[-1] - time[0])
    bands = DRONE_BANDS.get(str(drone_size), DRONE_BANDS["7"])

    axes: Dict[str, Any] = {}
    recommendations: List[Dict[str, Any]] = []

    for axis_name, (gyro_col, setpoint_col) in AXIS_MAP.items():
        if gyro_col not in df.columns:
            continue

        gyro = np.asarray(df[gyro_col], dtype=float)
        setpoint = np.asarray(df[setpoint_col], dtype=float) if setpoint_col in df.columns else np.zeros_like(gyro)
        throttle = np.asarray(df["throttle"], dtype=float) if "throttle" in df.columns else None

        n = min(len(gyro), len(setpoint), len(time))
        if throttle is not None:
            n = min(n, len(throttle))

        mask = np.isfinite(gyro[:n]) & np.isfinite(setpoint[:n])
        if throttle is not None:
            mask = mask & np.isfinite(throttle[:n])

        if int(np.sum(mask)) < 128:
            continue

        gyro_clean = gyro[:n][mask]
        setpoint_clean = setpoint[:n][mask]
        throttle_clean = throttle[:n][mask] if throttle is not None else None

        stats = _compute_axis_stats(
            gyro_clean,
            setpoint_clean,
            sample_rate_hz,
            bands,
            throttle_clean,
        )

        issue, confidence, reasons = _classify_issue(stats)
        pid_delta = _base_delta_for_issue(axis_name, issue, goal)

        valid_for_pid = confidence >= 0.50
        feel = _issue_label(issue, goal)
        tuning_moves = _tuning_moves(axis_name, issue, goal)
        recommendation_text = _recommendation_text(axis_name, issue, goal)

        axis_payload = {
            "axis": axis_name,
            "axis_name": axis_name.upper(),
            "status": "ok",
            "issue": issue,
            "feel": feel,
            "propwash_detected": issue == "propwash",
            "confidence": round(confidence, 3),
            "confidence_reason": reasons,
            "valid_for_pid": valid_for_pid,
            "pid_delta_pct": {
                "p": round(pid_delta["p"], 4),
                "i": round(pid_delta["i"], 4),
                "d": round(pid_delta["d"], 4),
                "ff": round(pid_delta["ff"], 4),
            },
            "recommendation_text": recommendation_text,
            "tuning_moves": tuning_moves,
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

        axes[axis_name] = axis_payload
        recommendations.append({
            "axis": axis_name,
            "issue": issue,
            "feel": feel,
            "propwash_detected": issue == "propwash",
            "valid_for_pid": valid_for_pid,
            "confidence": round(confidence, 3),
            "pid_delta_pct": axis_payload["pid_delta_pct"],
            "recommendation_text": recommendation_text,
            "tuning_moves": tuning_moves,
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
            "tuning_goal": goal,
            "source_references": SOURCE_REFERENCES,
        }

    return {
        "status": "ok",
        "valid_for_pid": any(axis["valid_for_pid"] for axis in axes.values()),
        "summary": _summary_from_axes(axes, goal),
        "warnings": _warnings(goal, axes),
        "global_actions": _global_actions(goal, axes),
        "axes": axes,
        "recommendations": recommendations,
        "sample_rate_hz": round(sample_rate_hz, 2),
        "duration_s": round(duration_s, 3),
        "drone_size": str(drone_size),
        "tuning_goal": goal,
        "source_references": SOURCE_REFERENCES,
    }
