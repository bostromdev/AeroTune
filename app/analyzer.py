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
    "aggressive": "locked_in",
    "efficiency_snappy": "locked_in",
    "floaty": "floaty",
    "cinematic": "floaty",
    "efficiency_floaty": "floaty",
}

VALID_GOALS = {"efficient", "locked_in", "floaty"}

DRONE_SIZE_PROFILES = {
    # Frequency bands are intentionally higher for smaller props.
    # Smaller props/motors spin faster, so using 7" bands on micros can misread normal
    # small-prop vibration as low/mid-band behavior and give oversized PID deltas.
    "3": {
        "label": "3 inch cinewhoop / compact freestyle",
        "bands": {
            "low": (0.0, 55.0),
            "mid": (55.0, 140.0),
            "high": (140.0, 260.0),
            "ultra": (260.0, 490.0),
            "propwash": (75.0, 190.0),
        },
        "pid_scale": 0.55,
        "clean_goal_scale": 0.50,
        "pid_caps": {
            "p": {"negative": 0.035, "positive": 0.022},
            "i": {"negative": 0.020, "positive": 0.022},
            "d": {"negative": 0.026, "positive": 0.014},
            "ff": {"negative": 0.028, "positive": 0.032},
        },
    },
    "3.5": {
        "label": "3.5 inch cinewhoop / compact freestyle",
        "bands": {
            "low": (0.0, 50.0),
            "mid": (50.0, 125.0),
            "high": (125.0, 240.0),
            "ultra": (240.0, 490.0),
            "propwash": (65.0, 175.0),
        },
        "pid_scale": 0.65,
        "clean_goal_scale": 0.55,
        "pid_caps": {
            "p": {"negative": 0.038, "positive": 0.026},
            "i": {"negative": 0.022, "positive": 0.025},
            "d": {"negative": 0.028, "positive": 0.017},
            "ff": {"negative": 0.030, "positive": 0.035},
        },
    },
    "4": {
        "label": "4 inch sub-250 / compact long-range",
        "bands": {
            "low": (0.0, 45.0),
            "mid": (45.0, 110.0),
            "high": (110.0, 220.0),
            "ultra": (220.0, 490.0),
            "propwash": (55.0, 160.0),
        },
        "pid_scale": 0.78,
        "clean_goal_scale": 0.65,
        "pid_caps": {
            "p": {"negative": 0.045, "positive": 0.032},
            "i": {"negative": 0.025, "positive": 0.028},
            "d": {"negative": 0.032, "positive": 0.020},
            "ff": {"negative": 0.035, "positive": 0.042},
        },
    },
    "5": {
        "label": "5 inch freestyle / racing",
        "bands": {
            "low": (0.0, 35.0),
            "mid": (35.0, 90.0),
            "high": (90.0, 180.0),
            "ultra": (180.0, 490.0),
            "propwash": (45.0, 130.0),
        },
        "pid_scale": 1.00,
        "clean_goal_scale": 1.00,
        "pid_caps": {
            "p": {"negative": 0.055, "positive": 0.045},
            "i": {"negative": 0.035, "positive": 0.040},
            "d": {"negative": 0.045, "positive": 0.032},
            "ff": {"negative": 0.050, "positive": 0.055},
        },
    },
    "7": {
        "label": "7 inch long-range / heavy freestyle",
        "bands": {
            "low": (0.0, 25.0),
            "mid": (25.0, 70.0),
            "high": (70.0, 140.0),
            "ultra": (140.0, 490.0),
            "propwash": (35.0, 110.0),
        },
        "pid_scale": 1.00,
        "clean_goal_scale": 0.90,
        "pid_caps": {
            "p": {"negative": 0.060, "positive": 0.050},
            "i": {"negative": 0.040, "positive": 0.045},
            "d": {"negative": 0.050, "positive": 0.035},
            "ff": {"negative": 0.055, "positive": 0.060},
        },
    },
}

PUBLIC_DRONE_SIZE_OPTIONS = tuple(DRONE_SIZE_PROFILES.keys())

DRONE_SIZE_ALIASES = {
    "3": "3",
    "3.0": "3",
    "3inch": "3",
    "3in": "3",
    "3.5": "3.5",
    "3.5inch": "3.5",
    "3.5in": "3.5",
    "4": "4",
    "4.0": "4",
    "4inch": "4",
    "4in": "4",
    "5": "5",
    "5.0": "5",
    "5inch": "5",
    "5in": "5",
    "7": "7",
    "7.0": "7",
    "7inch": "7",
    "7in": "7",
}

VALID_ANALYZER_DRONE_SIZES = set(DRONE_SIZE_ALIASES.keys())

# Backward-compatible name for any code that still imports DRONE_BANDS directly.
DRONE_BANDS = {size: profile["bands"] for size, profile in DRONE_SIZE_PROFILES.items()}


def normalize_drone_size(drone_size: str) -> Optional[str]:
    raw = str(drone_size or "").strip().lower()
    raw = raw.replace('"', "").replace("'", "").replace(" ", "")
    raw = raw.replace("-inch", "inch").replace("_inch", "inch")
    return DRONE_SIZE_ALIASES.get(raw)


def get_drone_profile(drone_size: str) -> Tuple[str, Dict[str, Any]]:
    size_key = normalize_drone_size(drone_size)
    if size_key is None:
        size_key = "7"
    return size_key, DRONE_SIZE_PROFILES[size_key]


def _serializable_bands(profile: Dict[str, Any]) -> Dict[str, List[float]]:
    return {
        name: [float(bounds[0]), float(bounds[1])]
        for name, bounds in profile["bands"].items()
    }


SOURCE_REFERENCES = [
    "Betaflight PID Tuning Guide: https://betaflight.com/docs/wiki/guides/current/PID-Tuning-Guide",
    "Betaflight Freestyle Tuning Principles: https://betaflight.com/docs/wiki/guides/current/Freestyle-Tuning-Principles",
    "P controls tracking authority; too much can oscillate or bounce.",
    "I improves attitude hold against wind, CG bias, and throttle changes.",
    "D damps propwash/bounceback, but amplifies noise and can heat motors.",
    "Feedforward improves stick response without using P as the only sharpness tool.",
    "Yaw D is normally 0; tune yaw mainly with P, I, and FF.",
]

ISSUE_LABELS = {
    "clean": "Clean tune",
    "propwash": "Propwash",
    "bounceback": "Bounceback / overshoot",
    "high_frequency_noise": "Harsh / noisy",
    "mid_frequency_vibration": "Shaky / rough",
    "high_throttle_oscillation": "High-throttle oscillation",
    "low_frequency_oscillation": "Loose / bouncing",
    "drift_or_weak_hold": "Weak hold / drifting",
    "poor_tracking": "Soft tracking",
    "slow_response": "Delayed / soft",
    "mechanical_noise_suspected": "Mechanical noise suspected",
    "soft_tracking_error": "Soft tracking",
}

COMMON_PROBLEM_LIBRARY = [
    {
        "id": "props_motors_frame",
        "label": "Mechanical vibration",
        "symptoms": ["jello", "rough motors", "random high-frequency noise", "noise on one axis"],
        "checks": ["props bent/chipped", "motor screws loose", "bell damage", "frame delam/crack", "stack soft mounting"],
    },
    {
        "id": "pid_gain_too_high",
        "label": "PID gain too high",
        "symptoms": ["fast oscillation", "hot motors", "rough punch-outs", "bounce at stops"],
        "checks": ["P too high", "D too high", "TPA needed at high throttle"],
    },
    {
        "id": "dirty_air",
        "label": "Dirty-air propwash",
        "symptoms": ["shake on dives", "shake after split-S", "wobble in hard turns", "recovery shake"],
        "checks": ["D damping", "D Max", "filter/noise health", "props suited to frame"],
    },
    {
        "id": "hold_drift",
        "label": "Weak hold / drift",
        "symptoms": ["won't hold angle/rate", "wind drift", "throttle-change attitude slip"],
        "checks": ["I gain", "anti-gravity behavior", "CG imbalance"],
    },
    {
        "id": "parts_mismatch",
        "label": "Random-parts build mismatch",
        "symptoms": ["tune works on one build but not another", "heavy 7 inch feels loose", "high KV/prop combo noisy"],
        "checks": ["prop pitch", "motor KV", "frame stiffness", "ESC protocol", "filtering", "battery sag"],
    },
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
    ultra_ratio: float
    propwash_ratio: float
    dominant_freq_hz: Optional[float]
    gyro_high_ratio: float
    throttle_activity: float
    throttle_error_ratio: float
    high_throttle_error_ratio: float
    stop_overshoot_ratio: float
    quiet_drift_ratio: float


def normalize_goal(goal: str) -> str:
    raw = str(goal or "efficient").strip().lower().replace("-", "_").replace(" ", "_")
    return GOAL_ALIASES.get(raw, "efficient")


def _rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(values))))


def _safe_corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    n = min(len(a), len(b))
    if n < 64:
        return None
    a = np.asarray(a[:n], dtype=float)
    b = np.asarray(b[:n], dtype=float)
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return None
    value = float(np.corrcoef(a, b)[0, 1])
    return value if np.isfinite(value) else None


def _estimate_lag_ms(setpoint: np.ndarray, gyro: np.ndarray, sample_rate_hz: float) -> Optional[float]:
    n = min(len(setpoint), len(gyro))
    if n < 128 or np.std(setpoint[:n]) < 1e-9 or np.std(gyro[:n]) < 1e-9:
        return None

    max_points = 2400
    step = max(1, int(np.ceil(n / max_points)))
    sp = setpoint[:n:step] - np.mean(setpoint[:n:step])
    gy = gyro[:n:step] - np.mean(gyro[:n:step])
    rate = sample_rate_hz / step

    max_lag = min(int(rate * 0.15), len(sp) // 4)
    if max_lag < 1:
        return None

    lags = np.arange(-max_lag, max_lag + 1)
    best_score = -np.inf
    best_lag = 0

    for lag in lags:
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
            continue
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        score = float(np.dot(a, b) / denom) if denom > 0 else -np.inf
        if score > best_score:
            best_score = score
            best_lag = int(lag)

    return float(best_lag / rate * 1000.0)


def _band_energy(freqs: np.ndarray, spectrum: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask):
        return 0.0
    return float(np.sum(np.square(spectrum[mask])))


def _frequency_stats(signal: np.ndarray, sample_rate_hz: float, bands: Dict[str, Tuple[float, float]]) -> Tuple[float, float, float, float, float, Optional[float]]:
    """
    Frequency analysis must preserve high-frequency content.
    Do not simple-downsample here; downsampling can hide/alias D-term noise.
    Instead, sample several original-rate chunks and average their band energy.
    """
    signal = np.asarray(signal, dtype=float)
    signal = signal[np.isfinite(signal)]

    if len(signal) < 128:
        return 0.0, 0.0, 0.0, 0.0, 0.0, None

    chunk_size = min(8192, len(signal))
    if len(signal) <= chunk_size:
        starts = [0]
    else:
        max_chunks = 8
        starts = np.linspace(0, len(signal) - chunk_size, num=min(max_chunks, max(2, len(signal) // chunk_size)), dtype=int).tolist()

    energies = {"low": 0.0, "mid": 0.0, "high": 0.0, "ultra": 0.0, "propwash": 0.0}
    dominant_candidates: List[Tuple[float, float]] = []

    for start in starts:
        chunk = signal[start:start + chunk_size]
        centered = chunk - np.mean(chunk)
        if np.std(centered) < 1e-12:
            continue

        window = np.hanning(len(centered))
        spectrum = np.abs(np.fft.rfft(centered * window))
        freqs = np.fft.rfftfreq(len(centered), d=1.0 / sample_rate_hz)

        if len(spectrum):
            spectrum[0] = 0.0

        for name in energies:
            energies[name] += _band_energy(freqs, spectrum, *bands[name])

        if len(spectrum) > 1 and float(np.max(spectrum)) > 0:
            idx = int(np.argmax(spectrum))
            dominant_candidates.append((float(spectrum[idx]), float(freqs[idx])))

    total = max(energies["low"] + energies["mid"] + energies["high"] + energies["ultra"], 1e-12)
    dominant = None
    if dominant_candidates:
        dominant = max(dominant_candidates, key=lambda item: item[0])[1]

    return (
        float(energies["low"] / total),
        float(energies["mid"] / total),
        float(energies["high"] / total),
        float(energies["ultra"] / total),
        float(energies["propwash"] / total),
        dominant,
    )

def _stop_overshoot_ratio(setpoint: np.ndarray, gyro: np.ndarray, sample_rate_hz: float) -> float:
    n = min(len(setpoint), len(gyro))
    if n < 256:
        return 0.0

    max_points = 6000
    step = max(1, int(np.ceil(n / max_points)))
    sp = setpoint[:n:step]
    gy = gyro[:n:step]
    rate = sample_rate_hz / step

    dsp = np.abs(np.gradient(sp))
    if np.std(dsp) < 1e-9:
        return 0.0

    events = np.where(dsp >= np.quantile(dsp, 0.92))[0]
    if len(events) == 0:
        return 0.0

    window = max(5, int(0.12 * rate))
    ratios: List[float] = []

    for idx in events[:: max(1, len(events) // 80)]:
        lo = max(0, idx - window)
        hi = min(len(sp), idx + window)
        if hi - lo < 10:
            continue
        command_span = float(np.max(sp[lo:hi]) - np.min(sp[lo:hi]))
        if command_span < 0.10:
            continue
        local_error = np.abs(gy[lo:hi] - sp[lo:hi])
        ratios.append(float(np.max(local_error) / max(command_span, 1e-9)))

    if not ratios:
        return 0.0
    return float(np.percentile(ratios, 80))


def _axis_stats(time: np.ndarray, gyro: np.ndarray, setpoint: np.ndarray, throttle: Optional[np.ndarray], bands: Dict[str, Tuple[float, float]]) -> AxisStats:
    n = min(len(time), len(gyro), len(setpoint))
    time = np.asarray(time[:n], dtype=float)
    gyro = np.asarray(gyro[:n], dtype=float)
    setpoint = np.asarray(setpoint[:n], dtype=float)

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

    low_ratio, mid_ratio, high_ratio, ultra_ratio, propwash_ratio, dominant_freq = _frequency_stats(error, sample_rate_hz, bands)
    _, _, gyro_high_ratio, gyro_ultra_ratio, _, _ = _frequency_stats(gyro_centered, sample_rate_hz, bands)

    corr = _safe_corr(setpoint, gyro)
    lag_ms = _estimate_lag_ms(setpoint, gyro, sample_rate_hz)

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
                    throttle_error_ratio = float(_rms(error[transition_mask]) / max(_rms(error[calm_mask]), 1e-9))

            high_mask = th >= np.quantile(th, 0.80)
            mid_mask = (th >= np.quantile(th, 0.35)) & (th <= np.quantile(th, 0.65))
            if np.any(high_mask) and np.any(mid_mask):
                high_throttle_error_ratio = float(_rms(error[high_mask]) / max(_rms(error[mid_mask]), 1e-9))

    quiet_mask = np.abs(setpoint) <= max(0.12, float(np.percentile(np.abs(setpoint), 35)))
    active_mask = np.abs(setpoint) >= max(0.20, float(np.percentile(np.abs(setpoint), 65)))
    if np.any(quiet_mask) and np.any(active_mask):
        quiet_drift_ratio = float(_rms(error[quiet_mask]) / max(_rms(error[active_mask]), 1e-9))

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
        ultra_ratio=ultra_ratio,
        propwash_ratio=propwash_ratio,
        dominant_freq_hz=dominant_freq,
        gyro_high_ratio=float(gyro_high_ratio + gyro_ultra_ratio),
        throttle_activity=throttle_activity,
        throttle_error_ratio=throttle_error_ratio,
        high_throttle_error_ratio=high_throttle_error_ratio,
        stop_overshoot_ratio=_stop_overshoot_ratio(setpoint, gyro, sample_rate_hz),
        quiet_drift_ratio=quiet_drift_ratio,
    )


def _is_clean(stats: AxisStats) -> bool:
    corr_ok = stats.corr is None or stats.corr >= 0.987
    lag_ok = stats.lag_ms is None or abs(stats.lag_ms) <= 18.0
    return (
        corr_ok
        and lag_ok
        and stats.error_ratio <= 0.090
        and stats.abs_error_p99 <= max(0.15, stats.setpoint_rms * 0.20)
        and stats.throttle_error_ratio <= 1.16
        and stats.high_throttle_error_ratio <= 1.18
        and stats.stop_overshoot_ratio <= 0.22
        and stats.high_ratio <= 0.30
        and stats.ultra_ratio <= 0.22
        and stats.mid_ratio <= 0.34
    )


def _confidence(stats: AxisStats, base: float) -> float:
    value = base
    if stats.duration_s >= 20:
        value += 0.05
    if stats.sample_rate_hz >= 250:
        value += 0.04
    if stats.corr is not None:
        value += 0.03
    return float(np.clip(value, 0.0, 0.98))


def _classify(axis: str, stats: AxisStats) -> Tuple[str, float, str]:
    # Clean is NOT first. Problem evidence must win before a log is called clean.
    noise_band = stats.high_ratio + stats.ultra_ratio
    residual_noise = (noise_band >= 0.38 and stats.error_ratio > 0.045) or (axis == "yaw" and noise_band >= 0.25 and stats.error_ratio > 0.075)
    gyro_noise = stats.gyro_high_ratio >= 0.50 and stats.gyro_rms > 0.02
    if residual_noise or gyro_noise:
        reason = "high-frequency residual noise" if residual_noise else "high-frequency gyro vibration"
        return "high_frequency_noise", _confidence(stats, 0.80), reason

    if stats.high_throttle_error_ratio >= 1.32 and stats.error_ratio > 0.075:
        return "high_throttle_oscillation", _confidence(stats, 0.77), "error rises mainly at high throttle"

    dirty_air = (
        stats.throttle_error_ratio >= 1.20
        and stats.error_ratio > 0.080
        and (stats.propwash_ratio >= 0.14 or stats.mid_ratio >= 0.16 or stats.spike_ratio >= 1.15)
    )
    bounceback = stats.stop_overshoot_ratio >= 0.27 and stats.error_ratio > 0.080
    if dirty_air:
        return "propwash", _confidence(stats, 0.82), "tracking error spikes around throttle movement / disturbed air"
    if bounceback:
        return "bounceback", _confidence(stats, 0.80), "overshoot after command stops"

    if stats.mid_ratio >= 0.42 and stats.error_ratio > 0.055:
        return "mid_frequency_vibration", _confidence(stats, 0.74), "mid-band residual vibration"

    if stats.low_ratio >= 0.66 and stats.error_ratio > 0.115 and not _is_clean(stats):
        return "low_frequency_oscillation", _confidence(stats, 0.72), "slow residual wobble with elevated error"

    if stats.quiet_drift_ratio >= 1.35 and stats.error_ratio > 0.075:
        return "drift_or_weak_hold", _confidence(stats, 0.70), "error persists during low stick input"

    if stats.corr is not None and stats.corr < 0.86 and stats.error_ratio > 0.080:
        return "poor_tracking", _confidence(stats, 0.70), "gyro/setpoint correlation is low"

    if stats.lag_ms is not None and stats.lag_ms > 32 and stats.error_ratio > 0.065:
        return "slow_response", _confidence(stats, 0.68), "gyro response lags setpoint"

    if _is_clean(stats):
        return "clean", _confidence(stats, 0.90), "high tracking, low residual error, no dominant problem band"

    if stats.error_ratio > 0.095:
        return "soft_tracking_error", _confidence(stats, 0.66), "residual tracking error is elevated"

    return "clean", _confidence(stats, 0.82), "no major PID problem detected"


def _base_delta(axis: str, issue: str) -> Dict[str, float]:
    delta = {"p": 0.0, "i": 0.0, "d": 0.0, "ff": 0.0}

    if issue == "propwash":
        if axis == "yaw":
            delta["p"] = -0.010
            delta["i"] = 0.010
        else:
            delta["p"] = -0.004
            delta["d"] = 0.030
    elif issue == "bounceback":
        if axis == "yaw":
            delta["p"] = -0.012
        else:
            delta["d"] = 0.025
            delta["p"] = -0.006
    elif issue == "high_frequency_noise":
        delta["p"] = -0.015
        delta["d"] = -0.030 if axis != "yaw" else 0.0
    elif issue == "mid_frequency_vibration":
        delta["p"] = -0.018
        delta["d"] = -0.018 if axis != "yaw" else 0.0
    elif issue == "high_throttle_oscillation":
        delta["p"] = -0.025
        delta["d"] = -0.010 if axis != "yaw" else 0.0
    elif issue == "low_frequency_oscillation":
        delta["p"] = -0.030
    elif issue == "drift_or_weak_hold":
        delta["i"] = 0.030
    elif issue == "poor_tracking":
        delta["p"] = 0.018
        delta["ff"] = 0.012
    elif issue == "soft_tracking_error":
        delta["p"] = 0.012
        delta["ff"] = 0.008
    elif issue == "slow_response":
        delta["ff"] = 0.030
        delta["p"] = 0.008

    if axis == "yaw":
        delta["d"] = 0.0
    return delta


def _apply_goal(axis: str, issue: str, delta: Dict[str, float], goal: str) -> Dict[str, float]:
    result = dict(delta)

    if issue == "clean":
        if goal == "locked_in":
            result = {"p": 0.008 if axis != "yaw" else 0.004, "i": 0.0, "d": 0.0, "ff": 0.012}
        elif goal == "floaty":
            result = {"p": -0.010 if axis != "yaw" else -0.006, "i": 0.0, "d": 0.0, "ff": -0.012}
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
        if issue in {"propwash", "bounceback"} and axis != "yaw":
            result["d"] += 0.004
        if issue in {"high_frequency_noise", "mid_frequency_vibration", "high_throttle_oscillation"}:
            result["d"] = min(result["d"], 0.0)
    elif goal == "floaty":
        if issue in {"poor_tracking", "soft_tracking_error", "slow_response"}:
            result["p"] -= 0.010 if axis != "yaw" else 0.006
            result["ff"] -= 0.010
        if issue in {"propwash", "bounceback"} and axis != "yaw":
            result["d"] -= 0.004

    for key in result:
        result[key] = float(np.clip(result[key], -0.06, 0.06))
    if axis == "yaw":
        result["d"] = 0.0
    return result


def _cap_signed(value: float, negative_cap: float, positive_cap: float) -> float:
    return float(np.clip(value, -abs(negative_cap), abs(positive_cap)))


def _apply_size_pid_limits(
    axis: str,
    issue: str,
    delta: Dict[str, float],
    profile: Dict[str, Any],
) -> Dict[str, float]:
    # Apply size-specific PID guardrails after issue and goal logic.
    #
    # This prevents small-prop builds from receiving the same magnitude of PID move as a
    # 5"/7" build. D increases are especially capped on 3", 3.5", and 4" profiles
    # because small motors/props can heat up or sound rough quickly when D is pushed.
    scale = float(profile.get("pid_scale", 1.0))

    if issue == "clean":
        scale *= float(profile.get("clean_goal_scale", 1.0))

    caps = profile.get("pid_caps", {})
    result: Dict[str, float] = {}

    for term, raw_value in delta.items():
        term_caps = caps.get(term, {"negative": 0.06, "positive": 0.06})
        scaled = float(raw_value) * scale
        result[term] = _cap_signed(
            scaled,
            float(term_caps.get("negative", 0.06)),
            float(term_caps.get("positive", 0.06)),
        )

    if axis == "yaw":
        result["d"] = 0.0

    return result



def _moves(issue: str, axis: str, goal: str) -> List[str]:
    yaw_note = "D: keep yaw D at 0. Tune yaw with P/I/FF only."

    if issue == "clean":
        if goal == "locked_in":
            return ["No fix required.", "Optional: raise P/FF very slightly only if you want sharper stick connection.", "Do not change D just because the log is clean."]
        if goal == "floaty":
            return ["No fix required.", "Optional: lower P/FF very slightly only if you want smoother cinematic movement.", "Do not lower too far or it may feel sloppy."]
        return ["No PID change needed.", "Only change something if the flight feel gives you a reason.", "For Efficient/Smooth, leave it alone."]

    if issue == "propwash":
        if axis == "yaw":
            return ["P: lower slightly only if yaw shakes during punch-outs or fast forward flight.", "I: raise slightly only if yaw will not hold heading through throttle changes.", yaw_note]
        return ["D: raise slightly for damping. This is the main propwash fix.", "P: do not raise P first; lower slightly only if it is over-correcting.", "I/FF: leave alone unless hold or stick response is also wrong.", "Check motor temperature after any D increase."]

    if issue == "bounceback":
        if axis == "yaw":
            return ["P: lower slightly if yaw bounces back after stick release.", "FF: leave alone unless yaw feels delayed.", yaw_note]
        return ["D: raise slightly to damp the stop.", "P: lower slightly if the bounce looks like overshoot.", "FF: leave alone unless stick response feels delayed.", "Check motor temperature after any D increase."]

    if issue == "high_frequency_noise":
        return ["D: lower or do not raise. D-term amplifies noise and can heat motors.", "Mechanical/filter check: props, motors, frame, stack, RPM/dynamic notch filtering.", "P: lower slightly only if the quad sounds harsh or twitchy.", yaw_note if axis == "yaw" else "Do not chase propwash by raising D until noise is clean."]

    if issue == "mid_frequency_vibration":
        return ["Mechanical check first: props, motors, frame screws, stack mounting.", "P/D: soften slightly if the build is clean but vibration remains.", "I/FF: leave alone unless hold or stick feel is the real complaint.", yaw_note if axis == "yaw" else "Do not raise D into vibration."]

    if issue == "high_throttle_oscillation":
        return ["P: lower slightly if roughness appears on punch-outs/high throttle.", "Consider TPA-style behavior if it only happens at high throttle.", "D: lower slightly only if motors are warm/noisy.", yaw_note if axis == "yaw" else "Retest with a controlled punch-out."]

    if issue == "low_frequency_oscillation":
        if axis == "yaw":
            return ["P: lower slightly. Slow yaw bounce usually means yaw P is pushing too hard.", "I: leave alone unless yaw will not hold heading.", yaw_note]
        return ["P: lower slightly first. Slow bounce usually means over-correction.", "D: leave alone at first. Add D only later if bounceback/propwash remains.", "I/FF: leave alone unless hold or stick response is wrong."]

    if issue == "drift_or_weak_hold":
        return ["I: raise slightly. I-term helps hold attitude against throttle changes, wind, and CG bias.", "P: leave mostly alone unless it also feels soft.", "D/FF: leave alone unless bounceback or stick delay appears."]

    if issue in {"poor_tracking", "soft_tracking_error"}:
        return ["P: raise slightly if the quad feels soft and does not follow sticks.", "FF: raise slightly if the response feels delayed but otherwise clean.", "D: leave alone unless bounceback/propwash appears.", "I: leave alone unless it will not hold attitude."]

    if issue == "slow_response":
        return ["FF: raise slightly first. FF improves stick response without making P do all the work.", "P: raise slightly only if it still feels soft after FF.", "D/I: leave alone unless bounceback or weak hold appears."]

    return ["No clear PID move. Retake the log if flight feel disagrees."]


def _recommendation(axis: str, issue: str, goal: str) -> str:
    label = axis.upper()
    if issue == "clean":
        if goal == "locked_in":
            return f"{label}: clean tune. No fix needed; optional tiny P/FF increase only for more locked-in feel."
        if goal == "floaty":
            return f"{label}: clean tune. No fix needed; optional tiny P/FF decrease only for smoother cinematic feel."
        return f"{label}: clean tune. No PID change needed."
    if issue == "propwash":
        return f"{label}: propwash detected. Add damping carefully; on yaw keep D at 0."
    if issue == "bounceback":
        return f"{label}: bounceback/overshoot detected. Add damping carefully; lower P slightly if it is over-correcting."
    if issue == "high_frequency_noise":
        return f"{label}: high-frequency noise detected. Do not raise D; clean noise first and reduce D/P if needed."
    if issue == "mid_frequency_vibration":
        return f"{label}: mid-frequency vibration detected. Check mechanical noise first; then soften P/D if needed."
    if issue == "high_throttle_oscillation":
        return f"{label}: high-throttle oscillation detected. Lower P slightly and consider TPA-style behavior if it only happens on punch-outs."
    if issue == "low_frequency_oscillation":
        return f"{label}: slow bounce detected. Lower P slightly first. Leave D alone unless propwash/bounceback remains."
    if issue == "drift_or_weak_hold":
        return f"{label}: weak hold detected. Raise I slightly if it drifts or loses attitude during throttle changes."
    if issue in {"poor_tracking", "soft_tracking_error"}:
        return f"{label}: soft tracking detected. Raise P for authority or FF if the delay is stick-response related."
    if issue == "slow_response":
        return f"{label}: slow response detected. Raise FF first; add P only if it still feels soft."
    return f"{label}: no clear PID change recommended."


def _severity(issue: str, stats: AxisStats) -> str:
    if issue == "clean":
        return "good"
    if issue in {"high_frequency_noise", "high_throttle_oscillation"} or stats.error_ratio > 0.18:
        return "warning"
    if issue in {"propwash", "bounceback", "mid_frequency_vibration"}:
        return "attention"
    return "watch"


def _summary_from_axes(axes: Dict[str, Dict[str, Any]], goal: str) -> str:
    problems = [f"{axis.upper()}: {payload['issue_label']}" for axis, payload in axes.items() if payload["issue"] != "clean"]
    if not problems:
        if goal == "locked_in":
            return "Clean tune detected. No fix needed; optional small P/FF increase only for a more locked-in feel."
        if goal == "floaty":
            return "Clean tune detected. No fix needed; optional small P/FF decrease only for smoother cinematic feel."
        return "Clean tune detected. No PID change needed."
    return "Analysis complete. " + " | ".join(problems)


def _common_flags(issue: str, stats: AxisStats) -> List[str]:
    flags: List[str] = []
    if issue in {"high_frequency_noise", "mid_frequency_vibration"}:
        flags.append("Inspect props/motors/frame/stack before PID-chasing.")
    if issue == "high_throttle_oscillation":
        flags.append("High-throttle-only roughness often points to TPA/filter/noise interaction.")
    if issue in {"propwash", "bounceback"}:
        flags.append("Retest with dives, split-S, hard turns, and throttle chops.")
    if stats.gyro_high_ratio > 0.40:
        flags.append("Gyro itself has high-frequency content; check filtering/mechanical noise.")
    return flags


def detect_oscillation(df: pd.DataFrame, drone_size: str = "7", tuning_goal: str = "efficient") -> Dict[str, Any]:
    goal = normalize_goal(tuning_goal)

    if df is None or df.empty or "time" not in df.columns:
        return {
            "status": "invalid",
            "valid_for_pid": False,
            "summary": "No usable time/gyro data found.",
            "warnings": ["Upload a parsed Betaflight CSV or run it through the optimizer first."],
            "global_actions": [],
            "axes": {},
            "recommendations": [],
            "tuning_goal": goal,
            "source_references": SOURCE_REFERENCES,
            "common_problem_library": COMMON_PROBLEM_LIBRARY,
        }

    size_key, drone_profile = get_drone_profile(drone_size)
    time = df["time"].to_numpy(dtype=float)
    bands = drone_profile["bands"]
    throttle = df["throttle"].to_numpy(dtype=float) if "throttle" in df.columns else None

    axes: Dict[str, Dict[str, Any]] = {}
    recommendations: List[Dict[str, Any]] = []
    global_warnings = [
        f"Using {drone_profile['label']} size profile: frequency bands and PID delta caps are size-specific.",
        "Recommendations are conservative percentage deltas, not final PID numbers.",
        "Retest after every change.",
        "Check motor temperature after any D increase or filter reduction.",
    ]
    global_actions = [
        "Use one small change at a time, then retest.",
        "Fix mechanical noise before using PID to hide vibration.",
        "For random-parts builds, verify props, motor KV, frame stiffness, ESC protocol, filtering, and battery health.",
        "Use the CSV optimizer when logs are too large or have messy column names.",
    ]

    valid_axis_count = 0

    for axis_name, (gyro_col, setpoint_col) in AXES.items():
        if gyro_col not in df.columns:
            continue

        gyro = df[gyro_col].to_numpy(dtype=float)
        setpoint = df[setpoint_col].to_numpy(dtype=float) if setpoint_col in df.columns else np.zeros_like(gyro)

        stats = _axis_stats(time, gyro, setpoint, throttle, bands)
        issue, confidence, reason = _classify(axis_name, stats)
        base_delta = _base_delta(axis_name, issue)
        pid_delta = _apply_goal(axis_name, issue, base_delta, goal)
        pid_delta = _apply_size_pid_limits(axis_name, issue, pid_delta, drone_profile)
        valid_for_pid = confidence >= 0.55

        if valid_for_pid:
            valid_axis_count += 1

        payload = {
            "axis": axis_name,
            "axis_name": axis_name.upper(),
            "status": "ok",
            "issue": issue,
            "issue_label": ISSUE_LABELS.get(issue, issue),
            "severity": _severity(issue, stats),
            "propwash_detected": issue == "propwash",
            "confidence": round(confidence, 3),
            "confidence_reason": reason,
            "valid_for_pid": valid_for_pid,
            "size_profile": size_key,
            "size_label": drone_profile["label"],
            "pid_delta_pct": {k: round(v, 4) for k, v in pid_delta.items()},
            "recommendation_text": _recommendation(axis_name, issue, goal),
            "tuning_moves": _moves(issue, axis_name, goal),
            "common_flags": _common_flags(issue, stats),
            "signal": {
                "dominant_freq_hz": None if stats.dominant_freq_hz is None else round(stats.dominant_freq_hz, 2),
                "low_ratio": round(stats.low_ratio, 4),
                "mid_ratio": round(stats.mid_ratio, 4),
                "high_ratio": round(stats.high_ratio, 4),
                "ultra_ratio": round(stats.ultra_ratio, 4),
                "propwash_ratio": round(stats.propwash_ratio, 4),
                "gyro_high_ratio": round(stats.gyro_high_ratio, 4),
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

        axes[axis_name] = payload
        recommendations.append({
            "axis": axis_name,
            "issue": issue,
            "issue_label": payload["issue_label"],
            "severity": payload["severity"],
            "propwash_detected": payload["propwash_detected"],
            "confidence": payload["confidence"],
            "confidence_reason": reason,
            "pid_delta_pct": payload["pid_delta_pct"],
            "recommendation_text": payload["recommendation_text"],
            "tuning_moves": payload["tuning_moves"],
        })

    return {
        "status": "ok" if axes else "invalid",
        "valid_for_pid": valid_axis_count > 0,
        "summary": _summary_from_axes(axes, goal) if axes else "No usable gyro axes found.",
        "warnings": global_warnings,
        "global_actions": global_actions,
        "axes": axes,
        "recommendations": recommendations,
        "sample_rate_hz": round(float(1.0 / np.median(np.diff(time))), 2) if len(time) > 2 else None,
        "duration_s": round(float(time[-1] - time[0]), 3) if len(time) > 1 else None,
        "drone_size": size_key,
        "requested_drone_size": str(drone_size),
        "drone_size_profile": size_key,
        "drone_size_label": drone_profile["label"],
        "frequency_bands_hz": _serializable_bands(drone_profile),
        "pid_size_profile": {
            "pid_scale": drone_profile["pid_scale"],
            "clean_goal_scale": drone_profile["clean_goal_scale"],
            "pid_caps": drone_profile["pid_caps"],
        },
        "tuning_goal": goal,
        "common_problem_library": COMMON_PROBLEM_LIBRARY,
        "source_references": SOURCE_REFERENCES,
    }
