
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
    "efficiency",
    "efficiency_snappy",
    "efficiency_floaty",
    "snappy",
    "smooth",
    "cinematic",
}

GOAL_ALIASES = {
    "efficiency": "efficient",
    "efficiency_snappy": "locked_in",
    "efficiency_floaty": "floaty",
    "snappy": "locked_in",
    "smooth": "efficient",
    "cinematic": "floaty",
}

DRONE_BANDS = {
    "5": {
        "gyro_low": (0, 35),
        "gyro_mid": (35, 90),
        "gyro_high": (90, 180),
        "residual_low": (0, 35),
        "residual_mid": (35, 90),
        "residual_high": (90, 180),
        "residual_ultra": (180, 330),
    },
    "7": {
        "gyro_low": (0, 25),
        "gyro_mid": (25, 70),
        "gyro_high": (70, 140),
        "residual_low": (0, 25),
        "residual_mid": (25, 70),
        "residual_high": (70, 140),
        "residual_ultra": (140, 300),
    },
}

SOURCE_REFERENCES = [
    "Betaflight PID Tuning Guide: https://betaflight.com/docs/wiki/guides/current/PID-Tuning-Guide",
    "P controls setpoint tracking and sharpness; too much P can overshoot or oscillate.",
    "I improves attitude hold against wind, CG bias, and throttle changes.",
    "D damps bounceback and propwash, but it also amplifies gyro noise and can heat motors.",
    "Feedforward improves stick response without forcing P to do all the response work.",
    "Yaw D is normally 0; yaw is tuned mainly with P, I, and FF.",
]


@dataclass
class AxisStats:
    dominant_gyro_freq_hz: Optional[float]
    dominant_error_freq_hz: Optional[float]

    gyro_low_ratio: float
    gyro_mid_ratio: float
    gyro_high_ratio: float

    residual_low_ratio: float
    residual_mid_ratio: float
    residual_high_ratio: float
    residual_ultra_ratio: float
    residual_noise_ratio: float

    corr: Optional[float]
    lag_ms: Optional[float]

    gyro_rms: float
    setpoint_rms: float
    error_rms: float
    error_ratio: float
    abs_error_p95: float
    abs_error_p99: float
    spike_ratio: float

    propwash_score: float
    throttle_activity: float
    throttle_error_ratio: float
    high_throttle_error_ratio: float
    stop_overshoot_ratio: float
    quiet_drift_ratio: float



def normalize_goal(goal: str) -> str:
    """Public wrapper used by app.main."""
    return _normalize_goal(goal)


def _normalize_goal(goal: str) -> str:
    raw = (goal or "efficient").strip().lower().replace("-", "_").replace(" ", "_")
    mapped = GOAL_ALIASES.get(raw, raw)
    return mapped if mapped in {"efficient", "locked_in", "floaty"} else "efficient"


def _safe_array(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return arr


def _band_energy(freqs: np.ndarray, mags: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask):
        return 0.0
    return float(np.sum(np.square(mags[mask])))


def _spectrum_ratios(signal: np.ndarray, sample_rate_hz: float, bands: Dict[str, Tuple[float, float]]) -> Tuple[Optional[float], Dict[str, float]]:
    centered = signal - np.mean(signal)
    if len(centered) < 16 or np.std(centered) < 1e-12:
        return None, {key: 0.0 for key in bands}

    window = np.hanning(len(centered))
    fft_vals = np.fft.rfft(centered * window)
    freqs = np.fft.rfftfreq(len(centered), d=1.0 / sample_rate_hz)
    mags = np.abs(fft_vals)

    if len(mags):
        mags[0] = 0.0

    dominant = None
    if len(mags) > 1 and float(np.max(mags)) > 0:
        dominant = float(freqs[int(np.argmax(mags))])

    energies = {key: _band_energy(freqs, mags, low, high) for key, (low, high) in bands.items()}
    total = max(float(sum(energies.values())), 1e-12)
    ratios = {key: float(value / total) for key, value in energies.items()}

    return dominant, ratios


def _estimate_lag_ms(setpoint: np.ndarray, gyro: np.ndarray, sample_rate_hz: float) -> Optional[float]:
    if len(setpoint) < 64 or np.std(setpoint) < 1e-6 or np.std(gyro) < 1e-6:
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

    max_lag = min(int(effective_sample_rate * 0.20), len(sp) // 4)
    if max_lag < 1:
        return None

    lags = np.arange(-max_lag, max_lag + 1)
    scores = np.empty(len(lags), dtype=float)

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

        if len(a) < 32:
            scores[idx] = -np.inf
            continue

        denom = np.linalg.norm(a) * np.linalg.norm(b)
        scores[idx] = float(np.dot(a, b) / denom) if denom > 0 else -np.inf

    best_lag = int(lags[int(np.argmax(scores))])
    return float(best_lag / effective_sample_rate * 1000.0)


def _windowed_ratio(numerator: np.ndarray, reference: np.ndarray, top_quantile: float, low_quantile: float) -> float:
    if len(numerator) == 0 or len(reference) != len(numerator):
        return 1.0

    hi_cut = float(np.quantile(reference, top_quantile))
    lo_cut = float(np.quantile(reference, low_quantile))

    hi_mask = reference >= hi_cut
    lo_mask = reference <= lo_cut

    if not np.any(hi_mask) or not np.any(lo_mask):
        return 1.0

    hi = float(np.sqrt(np.mean(np.square(numerator[hi_mask]))))
    lo = float(np.sqrt(np.mean(np.square(numerator[lo_mask]))))
    return hi / max(lo, 1e-9)


def _compute_axis_stats(
    gyro: np.ndarray,
    setpoint: np.ndarray,
    sample_rate_hz: float,
    band_profile: Dict[str, Tuple[float, float]],
    throttle: Optional[np.ndarray],
) -> AxisStats:
    n = min(len(gyro), len(setpoint))
    gyro = np.asarray(gyro[:n], dtype=float)
    setpoint = np.asarray(setpoint[:n], dtype=float)

    finite_mask = np.isfinite(gyro) & np.isfinite(setpoint)
    gyro = gyro[finite_mask]
    setpoint = setpoint[finite_mask]

    error = setpoint - gyro
    abs_error = np.abs(error)
    residual = error - np.mean(error)

    gyro_bands = {
        "low": band_profile["gyro_low"],
        "mid": band_profile["gyro_mid"],
        "high": band_profile["gyro_high"],
    }
    residual_bands = {
        "low": band_profile["residual_low"],
        "mid": band_profile["residual_mid"],
        "high": band_profile["residual_high"],
        "ultra": band_profile["residual_ultra"],
    }

    dominant_gyro_freq, gyro_ratios = _spectrum_ratios(gyro, sample_rate_hz, gyro_bands)
    dominant_error_freq, residual_ratios = _spectrum_ratios(residual, sample_rate_hz, residual_bands)

    corr = None
    if np.std(setpoint) > 1e-6 and np.std(gyro) > 1e-6:
        value = float(np.corrcoef(setpoint, gyro)[0, 1])
        corr = value if np.isfinite(value) else None

    lag_ms = _estimate_lag_ms(setpoint, gyro, sample_rate_hz)

    gyro_rms = float(np.sqrt(np.mean(np.square(gyro - np.mean(gyro)))))
    setpoint_rms = float(np.sqrt(np.mean(np.square(setpoint))))
    error_rms = float(np.sqrt(np.mean(np.square(error))))
    error_ratio = error_rms / max(setpoint_rms, 1e-9)

    abs_error_p95 = float(np.quantile(abs_error, 0.95))
    abs_error_p99 = float(np.quantile(abs_error, 0.99))
    median_abs = float(np.median(abs_error))
    spike_ratio = abs_error_p99 / max(median_abs, 1e-9)

    propwash_score = 1.0
    throttle_activity = 0.0
    throttle_error_ratio = 1.0
    high_throttle_error_ratio = 1.0

    if throttle is not None and len(throttle) >= n:
        th_full = np.asarray(throttle[:n], dtype=float)
        th = th_full[finite_mask]
        if len(th) == len(error) and np.all(np.isfinite(th)) and np.std(th) > 1e-7:
            dth = np.abs(np.gradient(th))
            throttle_activity = float(np.std(th) + np.mean(dth) * 8.0)
            residual_abs = np.abs(residual)

            propwash_score = _windowed_ratio(residual_abs, dth, 0.82, 0.45)
            throttle_error_ratio = _windowed_ratio(residual_abs, dth, 0.80, 0.40)
            high_throttle_error_ratio = _windowed_ratio(residual_abs, th, 0.80, 0.40)

    quiet_setpoint = np.abs(setpoint) <= max(0.05, float(np.quantile(np.abs(setpoint), 0.35)))
    if np.any(quiet_setpoint):
        quiet_drift_ratio = float(np.sqrt(np.mean(np.square(error[quiet_setpoint]))) / max(error_rms, 1e-9))
    else:
        quiet_drift_ratio = 1.0

    # Overshoot after the stick is released / command slows down.
    d_setpoint = np.abs(np.gradient(setpoint))
    stop_mask = (np.abs(setpoint) <= float(np.quantile(np.abs(setpoint), 0.50))) & (
        d_setpoint <= float(np.quantile(d_setpoint, 0.35))
    )
    if np.any(stop_mask):
        stop_overshoot_ratio = float(np.sqrt(np.mean(np.square(error[stop_mask]))) / max(error_rms, 1e-9))
    else:
        stop_overshoot_ratio = 1.0

    return AxisStats(
        dominant_gyro_freq_hz=dominant_gyro_freq,
        dominant_error_freq_hz=dominant_error_freq,
        gyro_low_ratio=gyro_ratios["low"],
        gyro_mid_ratio=gyro_ratios["mid"],
        gyro_high_ratio=gyro_ratios["high"],
        residual_low_ratio=residual_ratios["low"],
        residual_mid_ratio=residual_ratios["mid"],
        residual_high_ratio=residual_ratios["high"],
        residual_ultra_ratio=residual_ratios["ultra"],
        residual_noise_ratio=residual_ratios["high"] + residual_ratios["ultra"],
        corr=corr,
        lag_ms=lag_ms,
        gyro_rms=gyro_rms,
        setpoint_rms=setpoint_rms,
        error_rms=error_rms,
        error_ratio=float(error_ratio),
        abs_error_p95=abs_error_p95,
        abs_error_p99=abs_error_p99,
        spike_ratio=float(spike_ratio),
        propwash_score=float(propwash_score),
        throttle_activity=float(throttle_activity),
        throttle_error_ratio=float(throttle_error_ratio),
        high_throttle_error_ratio=float(high_throttle_error_ratio),
        stop_overshoot_ratio=float(stop_overshoot_ratio),
        quiet_drift_ratio=float(quiet_drift_ratio),
    )


def _classify_issue(axis: str, stats: AxisStats) -> Tuple[str, float, str]:
    corr = stats.corr if stats.corr is not None else 0.0
    lag = abs(stats.lag_ms) if stats.lag_ms is not None else 0.0

    clean_tracking = corr >= 0.985 and stats.error_ratio <= 0.115 and lag <= 12.0
    very_clean_tracking = corr >= 0.992 and stats.error_ratio <= 0.085 and lag <= 8.0

    # IMPORTANT: noise must be checked before clean.
    # A log can track setpoint well while still showing high-frequency residual error.
    if stats.residual_noise_ratio >= 0.24 and stats.error_ratio >= 0.055:
        return "high_frequency_noise", 0.94, "high-frequency tracking residual is dominant"

    if axis == "yaw" and stats.residual_noise_ratio >= 0.16 and stats.error_ratio >= 0.075:
        return "yaw_noise_or_roughness", 0.90, "yaw residual has high-frequency roughness"

    if stats.gyro_high_ratio >= 0.34 and stats.error_ratio >= 0.08:
        return "high_frequency_noise", 0.90, "gyro high-frequency energy is elevated"

    # Mid-frequency vibration / mechanical roughness.
    if stats.residual_mid_ratio >= 0.30 and stats.error_ratio >= 0.09:
        return "mid_frequency_vibration", 0.88, "mid-frequency residual vibration is elevated"

    if stats.gyro_mid_ratio >= 0.42 and stats.error_ratio >= 0.09:
        return "mid_frequency_vibration", 0.86, "gyro mid-frequency vibration is elevated"

    # Dirty-air and bounceback are bursty and usually tied to throttle/stops.
    dirty_air = (
        stats.propwash_score >= 1.32
        and stats.throttle_activity > 0.015
        and stats.throttle_error_ratio >= 1.18
        and stats.error_ratio >= 0.10
    )
    stop_bounce = (
        stats.stop_overshoot_ratio >= 1.25
        and stats.error_ratio >= 0.12
        and stats.residual_noise_ratio < 0.24
    )

    if dirty_air or stop_bounce:
        if axis == "yaw":
            return "yaw_throttle_roughness", 0.88, "yaw error rises during disturbed or high-load moments"
        return "propwash_or_bounceback", 0.92, "error rises during throttle/stops/dirty-air moments"

    # Weak hold: low frequency error with quiet-stick / throttle bias, not noise.
    if (
        stats.quiet_drift_ratio >= 1.22
        and stats.error_ratio >= 0.14
        and stats.residual_noise_ratio < 0.20
    ):
        return "drift_or_weak_hold", 0.84, "residual error persists during quiet-stick sections"

    # Low-frequency oscillation: only call it bounce if tracking is NOT clean and residual is really low-band.
    if (
        stats.residual_low_ratio >= 0.72
        and stats.error_ratio >= 0.16
        and not clean_tracking
        and stats.residual_noise_ratio < 0.16
    ):
        return "low_frequency_oscillation", 0.84, "dominant low-frequency residual motion"

    if stats.corr is not None and stats.corr < 0.78 and stats.error_ratio >= 0.18:
        return "poor_tracking", 0.82, "low gyro/setpoint correlation"

    if stats.lag_ms is not None and stats.lag_ms > 28.0 and stats.error_ratio >= 0.10:
        return "slow_response", 0.80, "gyro response lags behind setpoint"

    if clean_tracking:
        return "clean", 0.98 if very_clean_tracking else 0.92, "high setpoint tracking and low residual error"

    # Borderline: do not force a fake problem. Give optional feel-based advice only.
    return "borderline_clean", 0.72, "minor residual movement but no strong Betaflight tuning fault"


def _base_delta_for_issue(axis: str, issue: str) -> Dict[str, float]:
    delta = {"p": 0.0, "i": 0.0, "d": 0.0, "ff": 0.0}

    if issue == "high_frequency_noise":
        delta["p"] = -0.015
        delta["d"] = -0.035 if axis != "yaw" else 0.0
    elif issue == "yaw_noise_or_roughness":
        delta["p"] = -0.018
        delta["i"] = 0.0
        delta["d"] = 0.0
    elif issue == "mid_frequency_vibration":
        delta["p"] = -0.020
        delta["d"] = -0.020 if axis != "yaw" else 0.0
    elif issue == "propwash_or_bounceback":
        delta["p"] = -0.005
        delta["d"] = 0.030 if axis != "yaw" else 0.0
    elif issue == "yaw_throttle_roughness":
        delta["p"] = -0.012
        delta["i"] = 0.010
        delta["d"] = 0.0
    elif issue == "low_frequency_oscillation":
        delta["p"] = -0.030
        delta["d"] = 0.0
    elif issue == "drift_or_weak_hold":
        delta["i"] = 0.035
    elif issue == "poor_tracking":
        delta["p"] = 0.020
        delta["ff"] = 0.020
    elif issue == "slow_response":
        delta["ff"] = 0.035
        delta["p"] = 0.008

    return delta


def _apply_goal_bias(delta: Dict[str, float], axis: str, issue: str, goal: str) -> Dict[str, float]:
    result = dict(delta)

    if goal == "efficient":
        result["p"] *= 0.80
        result["i"] *= 0.90
        result["d"] *= 0.70
        result["ff"] *= 0.80
    elif goal == "locked_in":
        if issue in {"clean", "borderline_clean"}:
            result["p"] += 0.008 if axis != "yaw" else 0.004
            result["ff"] += 0.015
        elif issue in {"poor_tracking", "slow_response"}:
            result["p"] *= 1.10
            result["ff"] *= 1.15
        elif issue == "propwash_or_bounceback" and axis != "yaw":
            result["d"] += 0.005
        elif issue in {"high_frequency_noise", "mid_frequency_vibration", "yaw_noise_or_roughness"}:
            result["d"] = min(result["d"], 0.0)
    elif goal == "floaty":
        if issue in {"clean", "borderline_clean"}:
            result["p"] -= 0.015 if axis != "yaw" else 0.008
            result["ff"] -= 0.015
        elif issue in {"poor_tracking", "slow_response"}:
            result["ff"] *= 0.75
        elif issue == "propwash_or_bounceback" and axis != "yaw":
            result["d"] -= 0.004

    for key in result:
        result[key] = float(np.clip(result[key], -0.06, 0.06))

    if axis == "yaw":
        result["d"] = 0.0

    return result


def _issue_label(issue: str, goal: str) -> str:
    labels = {
        "clean": "Clean tune",
        "borderline_clean": "Mostly clean",
        "high_frequency_noise": "High-frequency noise",
        "yaw_noise_or_roughness": "Yaw roughness / noise",
        "mid_frequency_vibration": "Mid-frequency vibration",
        "propwash_or_bounceback": "Propwash / bounceback",
        "yaw_throttle_roughness": "Yaw throttle roughness",
        "low_frequency_oscillation": "Loose / bouncing",
        "drift_or_weak_hold": "Weak hold / drift",
        "poor_tracking": "Soft tracking",
        "slow_response": "Slow stick response",
    }
    return labels.get(issue, "Unknown")


def _pid_moves_for_issue(axis: str, issue: str, goal: str) -> List[str]:
    if issue == "clean":
        if goal == "locked_in":
            return [
                "No repair needed.",
                "Optional feel change only: add a tiny amount of P/FF if you want more locked-in response.",
                "Do not change D unless propwash or bounceback is actually present.",
            ]
        if goal == "floaty":
            return [
                "No repair needed.",
                "Optional feel change only: lower P/FF slightly if you want softer cinematic movement.",
                "Do not change D unless propwash or bounceback is actually present.",
            ]
        return [
            "No PID change needed.",
            "Only change something if the flight feel gives you a reason.",
            "For Efficient/Smooth, leave it alone.",
        ]

    if issue == "borderline_clean":
        return [
            "No required PID repair from this log.",
            "If you want more locked-in feel, try a tiny P/FF increase.",
            "If you want smoother cinematic feel, try a tiny P/FF decrease.",
            "Do not change D unless the flight shows propwash, bounceback, or noise.",
        ]

    if issue == "high_frequency_noise":
        return [
            "D: lower or do not raise it. D amplifies high-frequency gyro/residual noise and can heat motors.",
            "P: lower slightly only if the quad sounds harsh, twitchy, or nervous.",
            "I: leave alone unless the quad is drifting.",
            "FF: leave alone unless stick response is the issue.",
            "Fix mechanical/filtering causes first: props, motors, frame screws, stack mounting, RPM/dynamic filtering.",
        ]

    if issue == "yaw_noise_or_roughness":
        return [
            "D: keep yaw D at 0.",
            "P: lower yaw P slightly if yaw looks rough in fast forward flight or punch-outs.",
            "I: leave yaw I alone unless heading hold is weak.",
            "FF: leave alone unless yaw stick response feels delayed.",
            "Check mechanical noise first before tuning around it.",
        ]

    if issue == "mid_frequency_vibration":
        return [
            "P: lower slightly to calm roughness.",
            "D: lower slightly if motors sound rough or come down warm.",
            "I: leave alone unless attitude hold is weak.",
            "FF: leave alone unless stick response is the actual complaint.",
            "Also check props, motors, frame screws, and stack mounting before blaming only PID.",
        ]

    if issue == "propwash_or_bounceback":
        return [
            "D: raise slightly first. This is the main Betaflight-style fix for propwash and bounceback.",
            "P: do not raise P yet. If the bounce is slow or over-correcting, lower P slightly.",
            "I: leave alone unless the quad drifts or loses attitude on throttle changes.",
            "FF: leave alone unless stick response itself feels lazy.",
            "After raising D, do a short test flight and check motor temperature.",
        ]

    if issue == "yaw_throttle_roughness":
        return [
            "D: keep yaw D at 0.",
            "P: lower slightly if yaw shakes on punch-outs or fast forward flight.",
            "I: raise slightly only if yaw will not hold heading through throttle changes.",
            "FF: raise only if yaw stick response feels delayed.",
            "Yaw is tuned mainly with P/I/FF, not D.",
        ]

    if issue == "low_frequency_oscillation":
        if axis == "yaw":
            return [
                "P: lower yaw P slightly first.",
                "D: keep yaw D at 0.",
                "I: leave alone unless yaw drifts or will not hold heading.",
                "FF: leave alone unless yaw stick feel is delayed.",
            ]
        return [
            "P: lower slightly first. Slow bounce usually means the loop is over-correcting.",
            "D: leave alone at first. Add D only later if bounceback/propwash remains.",
            "I: leave alone unless the quad drifts or will not hold attitude.",
            "FF: leave alone unless stick response is delayed.",
        ]

    if issue == "drift_or_weak_hold":
        return [
            "I: raise slightly. I helps the quad hold attitude against wind, CG bias, and throttle changes.",
            "P: leave mostly alone unless the quad also feels soft or sloppy.",
            "D: leave alone unless bounceback or propwash appears.",
            "FF: leave alone unless stick response feels delayed.",
        ]

    if issue == "poor_tracking":
        return [
            "P: raise slightly if the quad feels soft and does not follow the sticks well.",
            "FF: raise slightly if stick response feels delayed but the quad is otherwise clean.",
            "D: leave alone unless bounceback or propwash appears.",
            "I: leave alone unless it will not hold attitude through throttle changes.",
        ]

    if issue == "slow_response":
        return [
            "FF: raise slightly first. FF improves stick response without making P do all the work.",
            "P: raise slightly only if it still feels soft after FF.",
            "D: leave alone unless bounceback or propwash is showing up.",
            "I: leave alone unless attitude hold is weak.",
        ]

    return ["No clear issue detected. Retake a cleaner log before changing PID values."]


def _recommendation_text(axis: str, issue: str, goal: str, valid_for_pid: bool) -> str:
    axis_label = axis.upper()

    if not valid_for_pid:
        return f"{axis_label}: data is not clean enough to trust PID advice yet. Retake a cleaner 30–120 second log."

    if issue == "clean":
        if goal == "locked_in":
            return f"{axis_label}: clean tune. No repair needed; only add tiny P/FF if you want more locked-in feel."
        if goal == "floaty":
            return f"{axis_label}: clean tune. No repair needed; only lower tiny P/FF if you want softer cinematic feel."
        return f"{axis_label}: clean tune. No PID change needed."

    if issue == "borderline_clean":
        return f"{axis_label}: mostly clean. No required repair; make only small feel-based changes if the quad does not match your goal."

    if issue == "high_frequency_noise":
        return f"{axis_label}: high-frequency noise detected. Do not raise D. Check mechanical/filter noise first; reduce D/P only if it sounds harsh or motors warm up."

    if issue == "yaw_noise_or_roughness":
        return f"{axis_label}: yaw roughness/noise detected. Keep yaw D at 0; reduce yaw P slightly only if flight footage or punch-outs confirm yaw shake."

    if issue == "mid_frequency_vibration":
        return f"{axis_label}: mid-frequency vibration detected. Check props/motors/frame first; if the build is clean, soften P/D slightly."

    if issue == "propwash_or_bounceback":
        return f"{axis_label}: propwash/bounceback detected. Add a small amount of D or D Max style damping; do not raise P first. Check motor heat."

    if issue == "yaw_throttle_roughness":
        return f"{axis_label}: yaw roughness during throttle/high-load moments. Keep yaw D at 0; adjust yaw P/I only if flight feel confirms it."

    if issue == "low_frequency_oscillation":
        if axis == "yaw":
            return f"{axis_label}: slow yaw bounce detected. Lower yaw P slightly first. Keep yaw D at 0."
        return f"{axis_label}: slow bounce detected. Lower P slightly first. Leave D alone unless propwash/bounceback remains."

    if issue == "drift_or_weak_hold":
        return f"{axis_label}: weak hold detected. Raise I slightly because I helps attitude hold through wind, CG bias, and throttle changes."

    if issue == "poor_tracking":
        return f"{axis_label}: soft tracking detected. Raise P slightly for tighter tracking, or FF if the main complaint is delayed stick response."

    if issue == "slow_response":
        return f"{axis_label}: slow stick response detected. Raise FF first; add P only if it still feels soft."

    return f"{axis_label}: no clear PID recommendation."


def _warnings_for_goal(goal: str) -> List[str]:
    warnings = [
        "Recommendations are conservative percentage deltas, not final PID numbers.",
        "Retest after every change.",
    ]

    if goal == "efficient":
        warnings.append("Efficient tuning prioritizes smoothness, motor safety, and conservative changes.")
    elif goal == "locked_in":
        warnings.append("Locked-In tuning can increase heat, twitchiness, or overshoot if pushed too far.")
    elif goal == "floaty":
        warnings.append("Floaty tuning trades sharp response for smoother cinematic movement.")

    return warnings


def _global_actions(goal: str) -> List[str]:
    actions = [
        "Use one small change at a time, then retest.",
        "If D goes up, do a short flight and check motor temperature.",
        "Fix mechanical noise before using PID to hide vibration.",
    ]

    if goal == "efficient":
        actions.append("Efficient mode keeps changes conservative and heat-safe.")
    elif goal == "locked_in":
        actions.append("Locked-In mode allows tiny P/FF increases only when the log is clean enough.")
    elif goal == "floaty":
        actions.append("Floaty mode softens response only when the log is clean enough.")

    return actions


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
            "sample_rate_hz": None,
            "duration_s": None,
            "drone_size": str(drone_size),
            "tuning_goal": goal,
            "source_references": SOURCE_REFERENCES,
        }

    if "time" not in df.columns:
        return {
            "status": "invalid",
            "valid_for_pid": False,
            "summary": "Missing time column.",
            "warnings": ["AeroTune requires real timestamps."],
            "global_actions": [],
            "axes": {},
            "recommendations": [],
            "sample_rate_hz": None,
            "duration_s": None,
            "drone_size": str(drone_size),
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
            "sample_rate_hz": None,
            "duration_s": None,
            "drone_size": str(drone_size),
            "tuning_goal": goal,
            "source_references": SOURCE_REFERENCES,
        }

    dt = np.diff(time)
    good_dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(good_dt) == 0:
        return {
            "status": "invalid",
            "valid_for_pid": False,
            "summary": "Time spacing is invalid.",
            "warnings": ["Timestamps must be increasing and real."],
            "global_actions": [],
            "axes": {},
            "recommendations": [],
            "sample_rate_hz": None,
            "duration_s": None,
            "drone_size": str(drone_size),
            "tuning_goal": goal,
            "source_references": SOURCE_REFERENCES,
        }

    sample_rate_hz = float(1.0 / np.median(good_dt))
    duration_s = float(time[-1] - time[0])
    band_profile = DRONE_BANDS.get(str(drone_size), DRONE_BANDS["7"])

    axes: Dict[str, Any] = {}
    recommendations: List[Dict[str, Any]] = []

    for axis_name, (gyro_col, setpoint_col) in AXIS_MAP.items():
        if gyro_col not in df.columns:
            continue

        gyro = np.asarray(df[gyro_col], dtype=float)
        setpoint = np.asarray(df[setpoint_col], dtype=float) if setpoint_col in df.columns else np.zeros_like(gyro)
        throttle = np.asarray(df["throttle"], dtype=float) if "throttle" in df.columns else None

        stats = _compute_axis_stats(gyro, setpoint, sample_rate_hz, band_profile, throttle)
        issue, confidence, confidence_reason = _classify_issue(axis_name, stats)
        delta = _apply_goal_bias(_base_delta_for_issue(axis_name, issue), axis_name, issue, goal)

        valid_for_pid = confidence >= 0.70
        issue_label = _issue_label(issue, goal)
        tuning_moves = _pid_moves_for_issue(axis_name, issue, goal)
        recommendation_text = _recommendation_text(axis_name, issue, goal, valid_for_pid)

        axis_payload = {
            "axis": axis_name,
            "axis_name": axis_name.upper(),
            "status": "ok",
            "issue": issue,
            "issue_label": issue_label,
            "propwash_detected": issue in {"propwash_or_bounceback", "yaw_throttle_roughness"},
            "confidence": round(confidence, 3),
            "confidence_reason": confidence_reason,
            "valid_for_pid": valid_for_pid,
            "pid_delta_pct": {
                "p": round(delta["p"], 4),
                "i": round(delta["i"], 4),
                "d": round(delta["d"], 4),
                "ff": round(delta["ff"], 4),
            },
            "recommendation_text": recommendation_text,
            "tuning_moves": tuning_moves,
            "signal": {
                "dominant_gyro_freq_hz": None if stats.dominant_gyro_freq_hz is None else round(stats.dominant_gyro_freq_hz, 2),
                "dominant_error_freq_hz": None if stats.dominant_error_freq_hz is None else round(stats.dominant_error_freq_hz, 2),
                "gyro_low_ratio": round(stats.gyro_low_ratio, 4),
                "gyro_mid_ratio": round(stats.gyro_mid_ratio, 4),
                "gyro_high_ratio": round(stats.gyro_high_ratio, 4),
                "residual_low_ratio": round(stats.residual_low_ratio, 4),
                "residual_mid_ratio": round(stats.residual_mid_ratio, 4),
                "residual_high_ratio": round(stats.residual_high_ratio, 4),
                "residual_ultra_ratio": round(stats.residual_ultra_ratio, 4),
                "residual_noise_ratio": round(stats.residual_noise_ratio, 4),
                "gyro_rms": round(stats.gyro_rms, 4),
                "error_rms": round(stats.error_rms, 4),
                "error_ratio": round(stats.error_ratio, 4),
                "abs_error_p95": round(stats.abs_error_p95, 4),
                "abs_error_p99": round(stats.abs_error_p99, 4),
                "spike_ratio": round(stats.spike_ratio, 4),
                "propwash_score": round(stats.propwash_score, 4),
                "throttle_activity": round(stats.throttle_activity, 4),
                "throttle_error_ratio": round(stats.throttle_error_ratio, 4),
                "high_throttle_error_ratio": round(stats.high_throttle_error_ratio, 4),
                "stop_overshoot_ratio": round(stats.stop_overshoot_ratio, 4),
                "quiet_drift_ratio": round(stats.quiet_drift_ratio, 4),
            },
            "tracking": {
                "corr": None if stats.corr is None else round(stats.corr, 4),
                "lag_ms": None if stats.lag_ms is None else round(stats.lag_ms, 2),
                "setpoint_rms": round(stats.setpoint_rms, 4),
            },
            "warnings": _warnings_for_goal(goal),
        }

        axes[axis_name] = axis_payload
        recommendations.append({
            "axis": axis_name,
            "issue": issue,
            "issue_label": issue_label,
            "propwash_detected": axis_payload["propwash_detected"],
            "confidence": axis_payload["confidence"],
            "confidence_reason": confidence_reason,
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

    problem_parts = [
        f"{payload['axis_name']}: {payload['issue_label']}"
        for payload in axes.values()
        if payload["issue"] not in {"clean", "borderline_clean"}
    ]

    if problem_parts:
        summary = "Analysis complete. " + " | ".join(problem_parts)
    else:
        summary = "Analysis complete. Clean tune — no PID repair needed."

    return {
        "status": "ok",
        "valid_for_pid": any(axis["valid_for_pid"] for axis in axes.values()),
        "summary": summary,
        "warnings": _warnings_for_goal(goal),
        "global_actions": _global_actions(goal),
        "axes": axes,
        "recommendations": recommendations,
        "sample_rate_hz": round(sample_rate_hz, 2),
        "duration_s": round(duration_s, 3),
        "drone_size": str(drone_size),
        "tuning_goal": goal,
        "source_references": SOURCE_REFERENCES,
    }
