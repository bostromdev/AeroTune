from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


AXES = {
    "roll": ("gyro_x", "setpoint_roll"),
    "pitch": ("gyro_y", "setpoint_pitch"),
    "yaw": ("gyro_z", "setpoint_yaw"),
}

GOAL_ALIASES = {
    "efficient": "efficient",
    "smooth": "efficient",
    "efficiency": "efficient",
    "locked": "locked_in",
    "locked_in": "locked_in",
    "snappy": "locked_in",
    "efficiency_snappy": "locked_in",
    "floaty": "floaty",
    "cinematic": "floaty",
    "efficiency_floaty": "floaty",
}

VALID_GOALS = {"efficient", "locked_in", "floaty"}

DRONE_BANDS = {
    "5": {
        "low": (0.0, 35.0),
        "mid": (35.0, 90.0),
        "high": (90.0, 180.0),
        "propwash": (45.0, 130.0),
    },
    "7": {
        "low": (0.0, 25.0),
        "mid": (25.0, 70.0),
        "high": (70.0, 140.0),
        "propwash": (35.0, 110.0),
    },
}

SOURCE_REFERENCES = [
    "Betaflight PID Tuning Guide: https://betaflight.com/docs/wiki/guides/current/PID-Tuning-Guide",
    "P controls how tightly the machine tracks setpoint; higher P is sharper but can overshoot or oscillate.",
    "I improves attitude hold against wind, CG bias, and throttle changes; too much can feel stiff.",
    "D adds damping and helps propwash/bounceback, but amplifies gyro noise and can heat motors.",
    "Feedforward improves stick response without using P as the only way to make the quad feel sharp.",
    "Yaw D is normally 0; yaw is mainly tuned with P, I, and FF.",
]


@dataclass(frozen=True)
class AxisStats:
    sample_rate_hz: float
    duration_s: float
    corr: Optional[float]
    lag_ms: Optional[float]
    gyro_rms: float
    setpoint_rms: float
    error_rms: float
    error_ratio: float
    abs_error_p95: float
    abs_error_p99: float
    spike_ratio: float
    low_ratio: float
    mid_ratio: float
    high_ratio: float
    propwash_ratio: float
    dominant_freq_hz: Optional[float]
    throttle_activity: float
    throttle_error_ratio: float
    high_throttle_error_ratio: float
    stop_overshoot_ratio: float
    quiet_drift_ratio: float


def normalize_goal(goal: str) -> str:
    raw = str(goal or "efficient").strip().lower().replace("-", "_").replace(" ", "_")
    return GOAL_ALIASES.get(raw, "efficient")


def _safe_array(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return arr


def _rms(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(values))))


def _downsample_pair(a: np.ndarray, b: np.ndarray, max_points: int = 2500) -> Tuple[np.ndarray, np.ndarray, int]:
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    step = max(1, int(np.ceil(n / max_points)))
    return a[::step], b[::step], step


def _estimate_lag_ms(setpoint: np.ndarray, gyro: np.ndarray, sample_rate_hz: float) -> Optional[float]:
    if len(setpoint) < 64 or np.std(setpoint) < 1e-9 or np.std(gyro) < 1e-9:
        return None

    sp, gy, step = _downsample_pair(setpoint, gyro, max_points=2400)
    effective_rate = sample_rate_hz / step

    sp = sp - np.mean(sp)
    gy = gy - np.mean(gy)

    max_lag = min(int(effective_rate * 0.15), len(sp) // 4)
    if max_lag < 1:
        return None

    lags = np.arange(-max_lag, max_lag + 1)
    scores = np.empty(len(lags), dtype=float)

    for i, lag in enumerate(lags):
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
            scores[i] = -np.inf
            continue

        denom = np.linalg.norm(a) * np.linalg.norm(b)
        scores[i] = float(np.dot(a, b) / denom) if denom > 0 else -np.inf

    best_lag = int(lags[int(np.argmax(scores))])
    return float(best_lag / effective_rate * 1000.0)


def _band_energy(freqs: np.ndarray, spectrum: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask):
        return 0.0
    return float(np.sum(np.square(spectrum[mask])))


def _frequency_stats(error: np.ndarray, sample_rate_hz: float, bands: Dict[str, Tuple[float, float]]) -> Tuple[float, float, float, float, Optional[float]]:
    if len(error) < 128:
        return 0.0, 0.0, 0.0, 0.0, None

    max_points = 8192
    step = max(1, int(np.ceil(len(error) / max_points)))
    signal = error[::step]
    effective_rate = sample_rate_hz / step

    centered = signal - np.mean(signal)
    window = np.hanning(len(centered))
    spectrum = np.abs(np.fft.rfft(centered * window))
    freqs = np.fft.rfftfreq(len(centered), d=1.0 / effective_rate)

    if len(spectrum) > 0:
        spectrum[0] = 0.0

    low = _band_energy(freqs, spectrum, *bands["low"])
    mid = _band_energy(freqs, spectrum, *bands["mid"])
    high = _band_energy(freqs, spectrum, *bands["high"])
    propwash = _band_energy(freqs, spectrum, *bands["propwash"])
    total = max(low + mid + high, 1e-12)

    dominant = None
    if len(spectrum) > 1 and float(np.max(spectrum)) > 0:
        dominant = float(freqs[int(np.argmax(spectrum))])

    return float(low / total), float(mid / total), float(high / total), float(propwash / total), dominant


def _safe_corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if len(a) < 32 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return None
    value = float(np.corrcoef(a, b)[0, 1])
    return value if np.isfinite(value) else None


def _axis_stats(
    time: np.ndarray,
    gyro: np.ndarray,
    setpoint: np.ndarray,
    throttle: Optional[np.ndarray],
    bands: Dict[str, Tuple[float, float]],
) -> AxisStats:
    n = min(len(time), len(gyro), len(setpoint))
    time = time[:n]
    gyro = gyro[:n]
    setpoint = setpoint[:n]

    dt = np.diff(time)
    good_dt = dt[np.isfinite(dt) & (dt > 0)]
    sample_rate_hz = float(1.0 / np.median(good_dt)) if len(good_dt) else 1000.0
    duration_s = float(time[-1] - time[0]) if len(time) > 1 else 0.0

    error = gyro - setpoint
    abs_error = np.abs(error)

    gyro_centered = gyro - np.mean(gyro)
    gyro_rms = _rms(gyro_centered)
    setpoint_rms = _rms(setpoint)
    error_rms = _rms(error)
    error_ratio = float(error_rms / max(setpoint_rms, 0.05))

    abs_error_p95 = float(np.percentile(abs_error, 95)) if len(abs_error) else 0.0
    abs_error_p99 = float(np.percentile(abs_error, 99)) if len(abs_error) else 0.0
    spike_ratio = float(abs_error_p99 / max(abs_error_p95, 1e-9))

    corr = _safe_corr(setpoint, gyro)
    lag_ms = _estimate_lag_ms(setpoint, gyro, sample_rate_hz)
    low_ratio, mid_ratio, high_ratio, propwash_ratio, dominant_freq = _frequency_stats(error, sample_rate_hz, bands)

    throttle_activity = 0.0
    throttle_error_ratio = 0.0
    high_throttle_error_ratio = 0.0
    quiet_drift_ratio = 0.0

    if throttle is not None and len(throttle) >= n:
        th = np.asarray(throttle[:n], dtype=float)
        if np.all(np.isfinite(th)):
            dth = np.abs(np.gradient(th))
            throttle_activity = float(np.std(th) + np.mean(dth) * 10.0)

            if np.std(dth) > 1e-9:
                transition_mask = dth >= np.quantile(dth, 0.85)
                calm_mask = dth <= np.quantile(dth, 0.45)
                if np.any(transition_mask) and np.any(calm_mask):
                    trans_error = _rms(error[transition_mask])
                    calm_error = _rms(error[calm_mask])
                    throttle_error_ratio = float(trans_error / max(calm_error, 1e-9))

            high_mask = th >= np.quantile(th, 0.80)
            mid_mask = (th >= np.quantile(th, 0.35)) & (th <= np.quantile(th, 0.65))
            if np.any(high_mask) and np.any(mid_mask):
                high_throttle_error_ratio = float(_rms(error[high_mask]) / max(_rms(error[mid_mask]), 1e-9))

    quiet_mask = np.abs(setpoint) <= max(0.12, float(np.percentile(np.abs(setpoint), 35)))
    active_mask = np.abs(setpoint) >= max(0.20, float(np.percentile(np.abs(setpoint), 65)))
    if np.any(quiet_mask) and np.any(active_mask):
        quiet_drift_ratio = float(_rms(error[quiet_mask]) / max(_rms(error[active_mask]), 1e-9))

    stop_overshoot_ratio = _stop_overshoot_ratio(setpoint, gyro, sample_rate_hz)

    return AxisStats(
        sample_rate_hz=sample_rate_hz,
        duration_s=duration_s,
        corr=corr,
        lag_ms=lag_ms,
        gyro_rms=gyro_rms,
        setpoint_rms=setpoint_rms,
        error_rms=error_rms,
        error_ratio=error_ratio,
        abs_error_p95=abs_error_p95,
        abs_error_p99=abs_error_p99,
        spike_ratio=spike_ratio,
        low_ratio=low_ratio,
        mid_ratio=mid_ratio,
        high_ratio=high_ratio,
        propwash_ratio=propwash_ratio,
        dominant_freq_hz=dominant_freq,
        throttle_activity=throttle_activity,
        throttle_error_ratio=throttle_error_ratio,
        high_throttle_error_ratio=high_throttle_error_ratio,
        stop_overshoot_ratio=stop_overshoot_ratio,
        quiet_drift_ratio=quiet_drift_ratio,
    )


def _stop_overshoot_ratio(setpoint: np.ndarray, gyro: np.ndarray, sample_rate_hz: float) -> float:
    if len(setpoint) < 128:
        return 0.0

    max_points = 6000
    step = max(1, int(np.ceil(len(setpoint) / max_points)))
    sp = setpoint[::step]
    gy = gyro[::step]
    rate = sample_rate_hz / step

    dsp = np.abs(np.gradient(sp))
    if np.std(dsp) < 1e-9:
        return 0.0

    event_mask = dsp >= np.quantile(dsp, 0.92)
    event_indices = np.where(event_mask)[0]
    if len(event_indices) == 0:
        return 0.0

    window = max(4, int(0.12 * rate))
    ratios: List[float] = []

    for idx in event_indices[:: max(1, len(event_indices) // 60)]:
        lo = max(0, idx - window)
        hi = min(len(sp), idx + window)
        if hi - lo < 8:
            continue

        local_sp = sp[lo:hi]
        local_gy = gy[lo:hi]
        command_span = float(np.max(local_sp) - np.min(local_sp))
        if command_span < 0.10:
            continue

        local_error = np.abs(local_gy - local_sp)
        ratios.append(float(np.max(local_error) / max(command_span, 1e-9)))

    if not ratios:
        return 0.0

    return float(np.percentile(ratios, 80))


def _is_clean(stats: AxisStats) -> bool:
    corr_ok = stats.corr is None or stats.corr >= 0.985
    lag_ok = stats.lag_ms is None or abs(stats.lag_ms) <= 18.0

    return (
        corr_ok
        and lag_ok
        and stats.error_ratio <= 0.095
        and stats.abs_error_p99 <= max(0.16, stats.setpoint_rms * 0.22)
        and stats.throttle_error_ratio <= 1.18
        and stats.high_throttle_error_ratio <= 1.20
        and stats.stop_overshoot_ratio <= 0.24
        and stats.mid_ratio <= 0.32
        and stats.high_ratio <= 0.34
    )


def _classify(axis: str, stats: AxisStats) -> Tuple[str, float, str]:
    evidence: List[str] = []

    # Hard problems first. Clean is intentionally last-ish and strict.
    if stats.high_ratio >= 0.48 and stats.error_ratio > 0.055:
        evidence.append("high residual frequency energy")
        return "high_frequency_noise", _confidence(stats, 0.78), "; ".join(evidence)

    if stats.high_throttle_error_ratio >= 1.35 and stats.error_ratio > 0.08:
        evidence.append("error rises at high throttle")
        return "high_throttle_oscillation", _confidence(stats, 0.76), "; ".join(evidence)

    dirty_air = (
        stats.throttle_error_ratio >= 1.22
        and stats.error_ratio > 0.085
        and (stats.propwash_ratio >= 0.15 or stats.mid_ratio >= 0.16 or stats.spike_ratio >= 1.18)
    )
    bounceback = (
        stats.stop_overshoot_ratio >= 0.28
        and stats.error_ratio > 0.085
        and (stats.corr is None or stats.corr >= 0.70)
    )
    elevated_clean_following_error = (
        stats.error_ratio >= 0.115
        and (stats.corr is not None and stats.corr >= 0.985)
        and stats.low_ratio >= 0.55
    )

    if dirty_air or bounceback or elevated_clean_following_error:
        if dirty_air:
            evidence.append("error increases around throttle movement")
        if bounceback:
            evidence.append("stop/overshoot behavior detected")
        if elevated_clean_following_error:
            evidence.append("setpoint follows, but residual error is too high")
        return "propwash_or_bounceback", _confidence(stats, 0.82), "; ".join(evidence)

    if stats.mid_ratio >= 0.42 and stats.error_ratio > 0.055:
        evidence.append("mid-band residual vibration")
        return "mid_frequency_vibration", _confidence(stats, 0.74), "; ".join(evidence)

    if (
        stats.low_ratio >= 0.62
        and stats.error_ratio > 0.105
        and not _is_clean(stats)
    ):
        evidence.append("dominant low-frequency residual motion")
        return "low_frequency_oscillation", _confidence(stats, 0.72), "; ".join(evidence)

    if stats.quiet_drift_ratio >= 1.35 and stats.error_ratio > 0.075:
        evidence.append("error persists during low stick input")
        return "drift_or_weak_hold", _confidence(stats, 0.70), "; ".join(evidence)

    if stats.corr is not None and stats.corr < 0.86 and stats.error_ratio > 0.08:
        evidence.append("gyro/setpoint correlation is low")
        return "poor_tracking", _confidence(stats, 0.70), "; ".join(evidence)

    if stats.lag_ms is not None and stats.lag_ms > 32 and stats.error_ratio > 0.065:
        evidence.append("gyro response lags setpoint")
        return "slow_response", _confidence(stats, 0.68), "; ".join(evidence)

    if _is_clean(stats):
        evidence.append("high setpoint tracking and low residual error")
        return "clean", _confidence(stats, 0.90), "; ".join(evidence)

    if stats.error_ratio > 0.095:
        evidence.append("residual tracking error is elevated")
        return "soft_tracking_error", _confidence(stats, 0.66), "; ".join(evidence)

    evidence.append("no major PID problem detected")
    return "clean", _confidence(stats, 0.80), "; ".join(evidence)


def _confidence(stats: AxisStats, base: float) -> float:
    value = base
    if stats.duration_s >= 20:
        value += 0.06
    if stats.corr is not None:
        value += 0.04
    if stats.sample_rate_hz >= 250:
        value += 0.04
    return float(np.clip(value, 0.0, 0.98))


ISSUE_LABELS = {
    "clean": "Clean tune",
    "propwash_or_bounceback": "Propwash / bounceback",
    "low_frequency_oscillation": "Loose / bouncing",
    "mid_frequency_vibration": "Shaky / rough",
    "high_frequency_noise": "Harsh / noisy",
    "high_throttle_oscillation": "High-throttle oscillation",
    "drift_or_weak_hold": "Weak hold / drifting",
    "poor_tracking": "Soft tracking",
    "soft_tracking_error": "Soft tracking",
    "slow_response": "Delayed / soft",
}


def _base_delta(axis: str, issue: str) -> Dict[str, float]:
    d = {"p": 0.0, "i": 0.0, "d": 0.0, "ff": 0.0}

    if issue == "propwash_or_bounceback":
        if axis == "yaw":
            d["p"] = -0.012
            d["i"] = 0.010
        else:
            d["p"] = -0.006
            d["d"] = 0.030

    elif issue == "low_frequency_oscillation":
        d["p"] = -0.030
        d["d"] = 0.0

    elif issue == "mid_frequency_vibration":
        d["p"] = -0.018
        d["d"] = -0.018 if axis != "yaw" else 0.0

    elif issue == "high_frequency_noise":
        d["p"] = -0.015
        d["d"] = -0.030 if axis != "yaw" else 0.0

    elif issue == "high_throttle_oscillation":
        d["p"] = -0.025
        d["d"] = -0.010 if axis != "yaw" else 0.0

    elif issue == "drift_or_weak_hold":
        d["i"] = 0.030

    elif issue == "poor_tracking":
        d["p"] = 0.018
        d["ff"] = 0.012

    elif issue == "soft_tracking_error":
        d["p"] = 0.012
        d["ff"] = 0.008

    elif issue == "slow_response":
        d["ff"] = 0.030
        d["p"] = 0.008

    return d


def _apply_goal(axis: str, issue: str, delta: Dict[str, float], goal: str) -> Dict[str, float]:
    result = dict(delta)

    if issue == "clean":
        if goal == "locked_in":
            result["p"] = 0.008 if axis != "yaw" else 0.004
            result["ff"] = 0.012
        elif goal == "floaty":
            result["p"] = -0.010 if axis != "yaw" else -0.006
            result["ff"] = -0.012
        else:
            result = {"p": 0.0, "i": 0.0, "d": 0.0, "ff": 0.0}
    elif goal == "efficient":
        result["p"] *= 0.80
        result["i"] *= 0.85
        result["d"] *= 0.70
        result["ff"] *= 0.80
    elif goal == "locked_in":
        if issue in {"poor_tracking", "soft_tracking_error", "slow_response"}:
            result["p"] += 0.006 if axis != "yaw" else 0.003
            result["ff"] += 0.010
        if issue == "propwash_or_bounceback" and axis != "yaw":
            result["d"] += 0.004
        if issue in {"high_frequency_noise", "mid_frequency_vibration", "high_throttle_oscillation"}:
            result["d"] = min(result["d"], 0.0)
    elif goal == "floaty":
        if issue in {"poor_tracking", "soft_tracking_error", "slow_response"}:
            result["p"] -= 0.010 if axis != "yaw" else 0.006
            result["ff"] -= 0.010
        if issue == "propwash_or_bounceback" and axis != "yaw":
            result["d"] -= 0.004

    for key in result:
        result[key] = float(np.clip(result[key], -0.06, 0.06))

    if axis == "yaw":
        result["d"] = 0.0

    return result


def _moves(issue: str, axis: str, goal: str) -> List[str]:
    yaw_note = "D: keep yaw D at 0. Tune yaw with P/I/FF only."

    if issue == "clean":
        if goal == "locked_in":
            return [
                "No fix required. Optional feel change only.",
                "P/FF: raise very slightly only if you want a more locked-in stick feel.",
                "Do not change D just because the log is clean.",
            ]
        if goal == "floaty":
            return [
                "No fix required. Optional feel change only.",
                "P/FF: lower very slightly only if you want smoother cinematic movement.",
                "Do not reduce too far or it may feel sloppy.",
            ]
        return [
            "No PID change needed.",
            "Only change something if the flight feel gives you a reason.",
            "For Efficient/Smooth, leave it alone.",
        ]

    if issue == "propwash_or_bounceback":
        if axis == "yaw":
            return [
                "P: lower slightly if yaw shakes during punch-outs or fast forward flight.",
                "I: raise slightly only if yaw will not hold heading through throttle changes.",
                "FF: raise only if yaw stick response feels delayed.",
                yaw_note,
            ]
        return [
            "D: raise slightly for damping. This is the main Betaflight-style propwash/bounceback fix.",
            "P: lower slightly only if it looks like over-correction or slow bounce.",
            "I: leave alone unless attitude hold is weak during throttle changes.",
            "FF: leave alone unless stick response feels delayed.",
            "After any D increase, do a short flight and check motor temperature.",
        ]

    if issue == "low_frequency_oscillation":
        if axis == "yaw":
            return [
                "P: lower slightly. Slow yaw bounce usually means yaw P is pushing too hard.",
                "I: leave alone unless yaw will not hold heading.",
                "FF: leave alone unless yaw feels delayed.",
                yaw_note,
            ]
        return [
            "P: lower slightly first. Slow bounce usually means the loop is over-correcting.",
            "D: leave alone at first. Add D only later if bounceback/propwash remains.",
            "I: leave alone unless the quad drifts or will not hold attitude.",
            "FF: leave alone unless stick response is delayed.",
        ]

    if issue == "mid_frequency_vibration":
        return [
            "Mechanical check first: props, motors, frame screws, stack mounting.",
            "P: lower slightly if the build is clean but roughness remains.",
            "D: lower slightly if motors sound rough or come down warm.",
            "I/FF: leave alone unless hold or stick feel is the actual problem.",
            yaw_note if axis == "yaw" else "Do not raise D into vibration.",
        ]

    if issue == "high_frequency_noise":
        return [
            "D: lower or do not raise. D-term amplifies gyro noise and can heat motors.",
            "Mechanical/filter check first: props, motors, frame, stack, filtering.",
            "P: lower slightly only if the quad sounds harsh or twitchy.",
            "I/FF: leave alone unless hold or stick feel is the actual problem.",
            yaw_note if axis == "yaw" else "Check motor temperature before pushing D higher.",
        ]

    if issue == "high_throttle_oscillation":
        return [
            "P: lower slightly, especially if oscillation appears on punch-outs or high throttle.",
            "Consider TPA-style reduction if high-throttle-only roughness is confirmed.",
            "D: lower slightly only if motors are warm/noisy.",
            "I/FF: leave alone unless hold or stick response is the complaint.",
            yaw_note if axis == "yaw" else "Retest with a high-throttle punch-out.",
        ]

    if issue == "drift_or_weak_hold":
        return [
            "I: raise slightly. I-term helps hold attitude against throttle changes, wind, and bias.",
            "P: leave mostly alone unless the quad also feels soft.",
            "D: leave alone unless bounceback/propwash appears.",
            "FF: leave alone unless stick response feels delayed.",
        ]

    if issue in {"poor_tracking", "soft_tracking_error"}:
        return [
            "P: raise slightly if the quad feels soft and does not follow the sticks.",
            "FF: raise slightly if the response feels delayed but otherwise clean.",
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

    return ["No clear PID problem detected. Retake the log if the flight feel disagrees."]


def _recommendation(axis: str, issue: str, goal: str) -> str:
    label = axis.upper()

    if issue == "clean":
        if goal == "locked_in":
            return f"{label}: clean tune. No fix needed. Optional tiny P/FF increase only if you want it more locked-in."
        if goal == "floaty":
            return f"{label}: clean tune. No fix needed. Optional tiny P/FF decrease only if you want smoother cinematic feel."
        return f"{label}: clean tune. No PID change needed."

    if issue == "propwash_or_bounceback":
        if axis == "yaw":
            return f"{label}: yaw roughness during disturbed/high-load moments. Keep yaw D at 0; adjust yaw P/I/FF only if flight feel confirms it."
        return f"{label}: propwash or bounceback detected. Add a little D for damping, avoid raising P first, and check motor heat."

    if issue == "low_frequency_oscillation":
        if axis == "yaw":
            return f"{label}: slow yaw bounce detected. Lower yaw P slightly and keep yaw D at 0."
        return f"{label}: slow bounce detected. Lower P slightly first. Leave D alone unless propwash/bounceback remains."

    if issue == "mid_frequency_vibration":
        return f"{label}: mid-frequency vibration detected. Check mechanical noise first; then soften P/D if needed."

    if issue == "high_frequency_noise":
        return f"{label}: high-frequency noise detected. Do not raise D. Clean noise first; reduce D/P if needed."

    if issue == "high_throttle_oscillation":
        return f"{label}: high-throttle oscillation detected. Lower P slightly and consider TPA-style behavior if it only happens on punch-outs."

    if issue == "drift_or_weak_hold":
        return f"{label}: weak hold detected. Raise I slightly if it drifts or loses attitude during throttle changes."

    if issue in {"poor_tracking", "soft_tracking_error"}:
        return f"{label}: soft tracking detected. Raise P slightly for authority or FF if the delay is stick-response related."

    if issue == "slow_response":
        return f"{label}: slow response detected. Raise FF first; add P only if it still feels soft."

    return f"{label}: no clear PID change recommended."


def _summary_from_axes(axes: Dict[str, Dict[str, Any]], goal: str) -> str:
    problems = [
        f"{axis.upper()}: {payload['issue_label']}"
        for axis, payload in axes.items()
        if payload["issue"] != "clean"
    ]

    if not problems:
        if goal == "locked_in":
            return "Clean tune detected. No fix needed; optional small P/FF increase only for a more locked-in feel."
        if goal == "floaty":
            return "Clean tune detected. No fix needed; optional small P/FF decrease only for smoother cinematic feel."
        return "Clean tune detected. No PID change needed."

    return "Analysis complete. " + " | ".join(problems)


def _global_actions(goal: str) -> List[str]:
    actions = [
        "Use one small change at a time, then retest.",
        "If D goes up, do a short flight and check motor temperature.",
        "Fix mechanical noise before using PID to hide vibration.",
    ]

    if goal == "locked_in":
        actions.append("Locked-In mode allows small P/FF increases only when the log is clean or tracking is soft.")
    elif goal == "floaty":
        actions.append("Floaty mode favors smoother motion and avoids unnecessary sharpness.")
    else:
        actions.append("Efficient mode keeps changes conservative and heat-safe.")

    return actions


def _axis_payload(axis: str, issue: str, confidence: float, reason: str, stats: AxisStats, goal: str) -> Dict[str, Any]:
    delta = _apply_goal(axis, issue, _base_delta(axis, issue), goal)

    return {
        "axis": axis,
        "axis_name": axis.upper(),
        "status": "ok",
        "issue": issue,
        "issue_label": ISSUE_LABELS.get(issue, issue.replace("_", " ").title()),
        "propwash_detected": issue == "propwash_or_bounceback",
        "confidence": round(confidence, 3),
        "confidence_reason": reason,
        "valid_for_pid": confidence >= 0.55,
        "pid_delta_pct": {k: round(v, 4) for k, v in delta.items()},
        "recommendation_text": _recommendation(axis, issue, goal),
        "tuning_moves": _moves(issue, axis, goal),
        "signal": {
            "dominant_freq_hz": None if stats.dominant_freq_hz is None else round(stats.dominant_freq_hz, 2),
            "low_ratio": round(stats.low_ratio, 4),
            "mid_ratio": round(stats.mid_ratio, 4),
            "high_ratio": round(stats.high_ratio, 4),
            "propwash_ratio": round(stats.propwash_ratio, 4),
            "gyro_rms": round(stats.gyro_rms, 4),
            "error_rms": round(stats.error_rms, 4),
            "error_ratio": round(stats.error_ratio, 4),
            "abs_error_p99": round(stats.abs_error_p99, 4),
            "spike_ratio": round(stats.spike_ratio, 4),
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
    }


def detect_oscillation(
    df: pd.DataFrame,
    drone_size: str = "7",
    tuning_goal: str = "efficient",
) -> Dict[str, Any]:
    goal = normalize_goal(tuning_goal)

    if df is None or df.empty:
        return _invalid("No usable data.", goal, ["Parsed log is empty."])

    required = {"time"}
    if not required.issubset(df.columns):
        return _invalid("Missing time column.", goal, ["AeroTune requires a time column."])

    available_axes = [axis for axis, (gyro_col, _) in AXES.items() if gyro_col in df.columns]
    if not available_axes:
        return _invalid("No usable gyro axes found.", goal, ["Need gyro_x, gyro_y, or gyro_z data."])

    time = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=float)
    finite_time = np.isfinite(time)
    if finite_time.mean() < 0.95 or len(time) < 128:
        return _invalid("Log is too short or timestamps are invalid.", goal, ["Use a 30–120 second Blackbox CSV when possible."])

    dt = np.diff(time)
    good_dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(good_dt) == 0:
        return _invalid("Timestamps are not increasing.", goal, ["Time must increase throughout the log."])

    sample_rate_hz = float(1.0 / np.median(good_dt))
    duration_s = float(time[-1] - time[0])
    bands = DRONE_BANDS.get(str(drone_size), DRONE_BANDS["7"])

    throttle = None
    if "throttle" in df.columns:
        throttle = pd.to_numeric(df["throttle"], errors="coerce").interpolate(limit_direction="both").fillna(0.0).to_numpy(dtype=float)

    axes: Dict[str, Dict[str, Any]] = {}

    for axis, (gyro_col, setpoint_col) in AXES.items():
        if gyro_col not in df.columns:
            continue

        gyro = pd.to_numeric(df[gyro_col], errors="coerce").interpolate(limit_direction="both").fillna(0.0).to_numpy(dtype=float)
        if setpoint_col in df.columns:
            setpoint = pd.to_numeric(df[setpoint_col], errors="coerce").interpolate(limit_direction="both").fillna(0.0).to_numpy(dtype=float)
        else:
            setpoint = np.zeros_like(gyro)

        stats = _axis_stats(time, gyro, setpoint, throttle, bands)
        issue, confidence, reason = _classify(axis, stats)
        axes[axis] = _axis_payload(axis, issue, confidence, reason, stats, goal)

    return {
        "status": "ok",
        "valid_for_pid": any(payload["valid_for_pid"] for payload in axes.values()),
        "summary": _summary_from_axes(axes, goal),
        "warnings": _warnings(goal),
        "global_actions": _global_actions(goal),
        "axes": axes,
        "recommendations": [
            {
                "axis": payload["axis"],
                "issue": payload["issue"],
                "issue_label": payload["issue_label"],
                "propwash_detected": payload["propwash_detected"],
                "confidence": payload["confidence"],
                "confidence_reason": payload["confidence_reason"],
                "pid_delta_pct": payload["pid_delta_pct"],
                "recommendation_text": payload["recommendation_text"],
                "tuning_moves": payload["tuning_moves"],
            }
            for payload in axes.values()
        ],
        "sample_rate_hz": round(sample_rate_hz, 2),
        "duration_s": round(duration_s, 3),
        "drone_size": str(drone_size),
        "tuning_goal": goal,
        "source_references": SOURCE_REFERENCES,
    }


def _warnings(goal: str) -> List[str]:
    common = [
        "Recommendations are conservative percentage deltas, not final PID numbers.",
        "Retest after every change.",
    ]
    if goal == "locked_in":
        common.append("Locked-in tuning can increase heat or twitchiness if P/D/FF are pushed too far.")
    elif goal == "floaty":
        common.append("Floaty tuning can feel softer and less precise if P/FF are reduced too far.")
    else:
        common.append("Efficient tuning prioritizes smoothness, motor safety, and conservative changes.")
    return common


def _invalid(summary: str, goal: str, warnings: List[str]) -> Dict[str, Any]:
    return {
        "status": "invalid",
        "valid_for_pid": False,
        "summary": summary,
        "warnings": warnings,
        "global_actions": ["Export a Blackbox CSV with time, gyro, setpoint, and throttle columns."],
        "axes": {},
        "recommendations": [],
        "sample_rate_hz": None,
        "duration_s": None,
        "drone_size": None,
        "tuning_goal": goal,
        "source_references": SOURCE_REFERENCES,
    }
