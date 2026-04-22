from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Config
# ----------------------------

MIN_SAMPLES = 400
MIN_VALID_DURATION_SEC = 0.8
MAX_REASONABLE_DT_SEC = 0.02   # 50 Hz minimum usable sample rate
MIN_REASONABLE_DT_SEC = 0.0002 # 5 kHz upper bound guard

AXES = ("roll", "pitch", "yaw")

# Conservative frequency bands.
# 7" typically lives a bit lower than 5", so we separate them.
DRONE_PROFILES = {
    "5": {
        "bands": {
            "low": (0, 35),
            "mid": (35, 90),
            "high": (90, 180),
        },
        "axis_limits": {
            "roll": {"p": 0.05, "d": 0.06, "ff": 0.04},
            "pitch": {"p": 0.05, "d": 0.06, "ff": 0.04},
            "yaw": {"p": 0.05, "d": 0.00, "ff": 0.04},
        },
    },
    "7": {
        "bands": {
            "low": (0, 25),
            "mid": (25, 70),
            "high": (70, 140),
        },
        "axis_limits": {
            "roll": {"p": 0.04, "d": 0.05, "ff": 0.03},
            "pitch": {"p": 0.04, "d": 0.05, "ff": 0.03},
            "yaw": {"p": 0.04, "d": 0.00, "ff": 0.03},
        },
    },
}


# ----------------------------
# Data containers
# ----------------------------

@dataclass
class SignalStats:
    rms: float
    std: float
    peak_to_peak: float
    dominant_freq_hz: Optional[float]
    dominant_amp: float
    low_band_energy: float
    mid_band_energy: float
    high_band_energy: float
    total_band_energy: float
    spectral_centroid_hz: Optional[float]


@dataclass
class TrackingStats:
    corr: Optional[float]
    lag_ms: Optional[float]
    tracking_error_rms: Optional[float]
    usable: bool


# ----------------------------
# Public entry point
# ----------------------------

def detect_oscillation(df: pd.DataFrame, drone_size: str = "7") -> Dict[str, Any]:
    """
    Main analyzer entry point.

    Returns conservative, delta-based PID guidance only when the log is
    actually suitable for tuning. Otherwise returns diagnosis-only output.
    """
    profile = DRONE_PROFILES.get(str(drone_size), DRONE_PROFILES["7"])

    validation = _validate_dataframe(df)
    if not validation["valid"]:
        return {
            "status": "invalid",
            "valid_for_pid": False,
            "summary": validation["reason"],
            "warnings": validation["warnings"],
            "recommendations": [],
            "axes": {},
            "global_actions": [
                "Use a real log with valid monotonic timestamps.",
                "Do not trust FFT-based PID advice from repaired or synthetic timing.",
            ],
        }

    clean_df = validation["df"]
    time_s = clean_df["time"].to_numpy(dtype=float)
    sample_rate_hz = validation["sample_rate_hz"]

    axis_results: Dict[str, Any] = {}
    global_warnings: List[str] = []
    global_actions: List[str] = []

    for axis in AXES:
        axis_result = _analyze_axis(clean_df, axis, time_s, sample_rate_hz, profile)
        axis_results[axis] = axis_result
        global_warnings.extend(axis_result["warnings"])
        global_actions.extend(axis_result["actions"])

    deduped_actions = _dedupe_preserve_order(global_actions)
    deduped_warnings = _dedupe_preserve_order(global_warnings)

    valid_axes_for_pid = [a for a in AXES if axis_results[a]["valid_for_pid"]]
    overall_valid_for_pid = len(valid_axes_for_pid) > 0

    summary = _build_summary(axis_results, overall_valid_for_pid)

    return {
        "status": "ok",
        "valid_for_pid": overall_valid_for_pid,
        "summary": summary,
        "warnings": deduped_warnings,
        "global_actions": deduped_actions,
        "sample_rate_hz": round(sample_rate_hz, 2),
        "duration_s": round(float(time_s[-1] - time_s[0]), 3),
        "drone_size": str(drone_size),
        "axes": axis_results,
        "recommendations": _flatten_axis_recommendations(axis_results),
    }


# ----------------------------
# Validation
# ----------------------------

def _validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    warnings: List[str] = []

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {
            "valid": False,
            "reason": "No usable log data found.",
            "warnings": warnings,
        }

    working = df.copy()
    working.columns = [str(c).strip().lower() for c in working.columns]

    if "time" not in working.columns:
        return {
            "valid": False,
            "reason": "Log is missing a real time column. PID analysis requires true timestamps.",
            "warnings": warnings,
        }

    working["time"] = pd.to_numeric(working["time"], errors="coerce")
    working = working.dropna(subset=["time"]).copy()

    if len(working) < MIN_SAMPLES:
        return {
            "valid": False,
            "reason": f"Log is too short for tuning analysis. Need at least {MIN_SAMPLES} samples.",
            "warnings": warnings,
        }

    time_values = working["time"].to_numpy(dtype=float)

    # Normalize time to seconds if the log looks like microseconds or milliseconds.
    inferred_time = _normalize_time_to_seconds(time_values)
    time_s = inferred_time["time_s"]
    warnings.extend(inferred_time["warnings"])

    if len(time_s) < MIN_SAMPLES:
        return {
            "valid": False,
            "reason": "Not enough valid timestamp samples after cleaning.",
            "warnings": warnings,
        }

    dt = np.diff(time_s)
    finite_dt = dt[np.isfinite(dt)]

    if len(finite_dt) == 0:
        return {
            "valid": False,
            "reason": "Timestamp deltas are invalid.",
            "warnings": warnings,
        }

    if np.any(finite_dt <= 0):
        return {
            "valid": False,
            "reason": "Timestamps are not strictly increasing.",
            "warnings": warnings,
        }

    median_dt = float(np.median(finite_dt))
    if median_dt < MIN_REASONABLE_DT_SEC or median_dt > MAX_REASONABLE_DT_SEC:
        return {
            "valid": False,
            "reason": (
                f"Timestamp spacing looks unrealistic for analysis "
                f"(median dt={median_dt:.6f}s)."
            ),
            "warnings": warnings,
        }

    duration_s = float(time_s[-1] - time_s[0])
    if duration_s < MIN_VALID_DURATION_SEC:
        return {
            "valid": False,
            "reason": (
                f"Log duration is too short ({duration_s:.3f}s). "
                f"Need at least {MIN_VALID_DURATION_SEC:.1f}s."
            ),
            "warnings": warnings,
        }

    working = working.copy()
    working["time"] = time_s

    return {
        "valid": True,
        "reason": "OK",
        "warnings": _dedupe_preserve_order(warnings),
        "df": working,
        "sample_rate_hz": 1.0 / median_dt,
    }


def _normalize_time_to_seconds(time_values: np.ndarray) -> Dict[str, Any]:
    warnings: List[str] = []

    span = float(time_values[-1] - time_values[0])
    if span <= 0:
        return {"time_s": time_values, "warnings": warnings}

    # Heuristic unit normalization:
    # - > 10,000 seconds span for a short flight log is suspicious
    # - convert likely ms or us to seconds
    if span > 1e6:
        warnings.append("Time column appears to be in microseconds; converted to seconds.")
        return {"time_s": (time_values - time_values[0]) / 1_000_000.0, "warnings": warnings}
    if span > 1e3:
        warnings.append("Time column appears to be in milliseconds; converted to seconds.")
        return {"time_s": (time_values - time_values[0]) / 1_000.0, "warnings": warnings}

    # Already likely seconds; just zero-base it.
    return {"time_s": time_values - time_values[0], "warnings": warnings}


# ----------------------------
# Axis analysis
# ----------------------------

def _analyze_axis(
    df: pd.DataFrame,
    axis: str,
    time_s: np.ndarray,
    sample_rate_hz: float,
    profile: Dict[str, Any],
) -> Dict[str, Any]:
    gyro_col = _find_first_present(df, [f"{axis}_gyro", f"gyro_{axis}", axis])
    setpoint_col = _find_first_present(df, [f"{axis}_setpoint", f"setpoint_{axis}", f"rc_{axis}"])

    warnings: List[str] = []
    actions: List[str] = []

    if gyro_col is None:
        return {
            "axis": axis,
            "status": "missing",
            "valid_for_pid": False,
            "issue": "missing_signal",
            "confidence": 0.0,
            "warnings": [f"No gyro signal found for {axis} axis."],
            "actions": [f"Provide a real gyro trace for {axis} before tuning this axis."],
            "signal": {},
            "tracking": {},
            "pid_delta_pct": {"p": 0.0, "d": 0.0, "ff": 0.0},
            "recommendation_text": f"{axis.upper()}: no PID recommendation due to missing gyro data.",
        }

    signal = pd.to_numeric(df[gyro_col], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(signal) & np.isfinite(time_s)
    signal = signal[mask]
    t = time_s[mask]

    if len(signal) < MIN_SAMPLES:
        return {
            "axis": axis,
            "status": "too_short",
            "valid_for_pid": False,
            "issue": "insufficient_data",
            "confidence": 0.0,
            "warnings": [f"{axis.upper()} signal too short after cleaning."],
            "actions": [f"Capture a longer {axis} maneuver segment."],
            "signal": {},
            "tracking": {},
            "pid_delta_pct": {"p": 0.0, "d": 0.0, "ff": 0.0},
            "recommendation_text": f"{axis.upper()}: no PID recommendation due to insufficient data.",
        }

    signal_stats = _compute_signal_stats(signal, sample_rate_hz, profile["bands"])
    tracking_stats = _compute_tracking_stats(df, setpoint_col, gyro_col, sample_rate_hz)

    issue, confidence = _classify_issue(axis, signal_stats, tracking_stats, profile)
    pid_delta_pct, issue_actions, issue_warnings = _recommend_pid_delta(
        axis=axis,
        issue=issue,
        confidence=confidence,
        tracking=tracking_stats,
        signal=signal_stats,
        limits=profile["axis_limits"][axis],
    )

    warnings.extend(issue_warnings)
    actions.extend(issue_actions)

    valid_for_pid = _is_axis_valid_for_pid(signal_stats, tracking_stats, confidence, issue)
    if not valid_for_pid:
        pid_delta_pct = {"p": 0.0, "d": 0.0, "ff": 0.0}
        actions.append(
            f"{axis.upper()}: diagnosis only. Data confidence is too low for safe PID changes."
        )

    recommendation_text = _build_axis_recommendation_text(
        axis=axis,
        issue=issue,
        confidence=confidence,
        pid_delta_pct=pid_delta_pct,
        valid_for_pid=valid_for_pid,
    )

    return {
        "axis": axis,
        "status": "ok",
        "valid_for_pid": valid_for_pid,
        "issue": issue,
        "confidence": round(confidence, 3),
        "warnings": _dedupe_preserve_order(warnings),
        "actions": _dedupe_preserve_order(actions),
        "signal": {
            "rms": round(signal_stats.rms, 4),
            "std": round(signal_stats.std, 4),
            "peak_to_peak": round(signal_stats.peak_to_peak, 4),
            "dominant_freq_hz": None if signal_stats.dominant_freq_hz is None else round(signal_stats.dominant_freq_hz, 2),
            "dominant_amp": round(signal_stats.dominant_amp, 4),
            "low_band_energy": round(signal_stats.low_band_energy, 4),
            "mid_band_energy": round(signal_stats.mid_band_energy, 4),
            "high_band_energy": round(signal_stats.high_band_energy, 4),
            "total_band_energy": round(signal_stats.total_band_energy, 4),
            "spectral_centroid_hz": None if signal_stats.spectral_centroid_hz is None else round(signal_stats.spectral_centroid_hz, 2),
        },
        "tracking": {
            "corr": None if tracking_stats.corr is None else round(tracking_stats.corr, 4),
            "lag_ms": None if tracking_stats.lag_ms is None else round(tracking_stats.lag_ms, 2),
            "tracking_error_rms": None if tracking_stats.tracking_error_rms is None else round(tracking_stats.tracking_error_rms, 4),
            "usable": tracking_stats.usable,
        },
        "pid_delta_pct": {
            "p": round(pid_delta_pct["p"], 4),
            "d": round(pid_delta_pct["d"], 4),
            "ff": round(pid_delta_pct["ff"], 4),
        },
        "recommendation_text": recommendation_text,
    }


# ----------------------------
# Signal features
# ----------------------------

def _compute_signal_stats(
    signal: np.ndarray,
    sample_rate_hz: float,
    bands: Dict[str, Tuple[float, float]],
) -> SignalStats:
    detrended = signal - np.mean(signal)
    n = len(detrended)

    window = np.hanning(n)
    windowed = detrended * window

    fft_vals = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)
    mags = np.abs(fft_vals)

    if len(mags) > 0:
        mags[0] = 0.0  # remove DC dominance

    dominant_idx = int(np.argmax(mags)) if len(mags) else 0
    dominant_freq = float(freqs[dominant_idx]) if len(freqs) and mags[dominant_idx] > 0 else None
    dominant_amp = float(mags[dominant_idx]) if len(mags) else 0.0

    low_energy = _band_energy(freqs, mags, *bands["low"])
    mid_energy = _band_energy(freqs, mags, *bands["mid"])
    high_energy = _band_energy(freqs, mags, *bands["high"])
    total_energy = low_energy + mid_energy + high_energy

    spectral_centroid = None
    if np.sum(mags) > 0:
        spectral_centroid = float(np.sum(freqs * mags) / np.sum(mags))

    return SignalStats(
        rms=float(np.sqrt(np.mean(np.square(detrended)))),
        std=float(np.std(detrended)),
        peak_to_peak=float(np.max(signal) - np.min(signal)),
        dominant_freq_hz=dominant_freq,
        dominant_amp=dominant_amp,
        low_band_energy=float(low_energy),
        mid_band_energy=float(mid_energy),
        high_band_energy=float(high_energy),
        total_band_energy=float(total_energy),
        spectral_centroid_hz=spectral_centroid,
    )


def _band_energy(freqs: np.ndarray, mags: np.ndarray, low_hz: float, high_hz: float) -> float:
    mask = (freqs >= low_hz) & (freqs < high_hz)
    if not np.any(mask):
        return 0.0
    return float(np.sum(np.square(mags[mask])))


# ----------------------------
# Tracking features
# ----------------------------

def _compute_tracking_stats(
    df: pd.DataFrame,
    setpoint_col: Optional[str],
    gyro_col: str,
    sample_rate_hz: float,
) -> TrackingStats:
    if setpoint_col is None:
        return TrackingStats(
            corr=None,
            lag_ms=None,
            tracking_error_rms=None,
            usable=False,
        )

    sp = pd.to_numeric(df[setpoint_col], errors="coerce").to_numpy(dtype=float)
    gy = pd.to_numeric(df[gyro_col], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(sp) & np.isfinite(gy)
    sp = sp[mask]
    gy = gy[mask]

    if len(sp) < MIN_SAMPLES or np.std(sp) < 1e-6 or np.std(gy) < 1e-6:
        return TrackingStats(
            corr=None,
            lag_ms=None,
            tracking_error_rms=None,
            usable=False,
        )

    sp_z = sp - np.mean(sp)
    gy_z = gy - np.mean(gy)

    corrcoef = np.corrcoef(sp_z, gy_z)[0, 1]
    corrcoef = float(corrcoef) if np.isfinite(corrcoef) else None

    max_lag_samples = min(int(0.2 * sample_rate_hz), len(sp_z) // 4)
    if max_lag_samples < 1:
        return TrackingStats(
            corr=corrcoef,
            lag_ms=None,
            tracking_error_rms=float(np.sqrt(np.mean((sp - gy) ** 2))),
            usable=corrcoef is not None,
        )

    xcorr = np.correlate(gy_z, sp_z, mode="full")
    lags = np.arange(-len(sp_z) + 1, len(sp_z))
    lag_mask = (lags >= -max_lag_samples) & (lags <= max_lag_samples)

    local_xcorr = xcorr[lag_mask]
    local_lags = lags[lag_mask]

    best_lag = int(local_lags[np.argmax(local_xcorr)])
    lag_ms = float(best_lag / sample_rate_hz * 1000.0)

    return TrackingStats(
        corr=corrcoef,
        lag_ms=lag_ms,
        tracking_error_rms=float(np.sqrt(np.mean((sp - gy) ** 2))),
        usable=corrcoef is not None,
    )


# ----------------------------
# Classification
# ----------------------------

def _classify_issue(
    axis: str,
    signal: SignalStats,
    tracking: TrackingStats,
    profile: Dict[str, Any],
) -> Tuple[str, float]:
    total = max(signal.total_band_energy, 1e-9)
    low_ratio = signal.low_band_energy / total
    mid_ratio = signal.mid_band_energy / total
    high_ratio = signal.high_band_energy / total

    confidence = 0.35

    if signal.dominant_freq_hz is not None:
        confidence += 0.10
    if total > 0:
        confidence += 0.10
    if tracking.usable:
        confidence += 0.15
    if tracking.corr is not None and tracking.corr > 0.6:
        confidence += 0.15

    # Main issue decision
    if high_ratio > 0.50 and (signal.dominant_freq_hz or 0) >= profile["bands"]["high"][0]:
        issue = "high_frequency_noise"
        confidence += 0.15
    elif mid_ratio > 0.45:
        issue = "mid_frequency_vibration"
        confidence += 0.15
    elif low_ratio > 0.45:
        issue = "low_frequency_oscillation"
        confidence += 0.15
    elif tracking.usable and tracking.corr is not None and tracking.corr < 0.35:
        issue = "poor_tracking"
        confidence += 0.15
    elif tracking.usable and tracking.lag_ms is not None and tracking.lag_ms > 40:
        issue = "slow_response"
        confidence += 0.10
    else:
        issue = "mostly_clean"

    # Propwash-like behavior:
    # low/mid energy elevated + tracking usable but noisy
    if issue in {"low_frequency_oscillation", "mid_frequency_vibration"}:
        if tracking.usable and tracking.corr is not None and tracking.corr > 0.45:
            if signal.rms > 0 and signal.peak_to_peak > (4.0 * signal.rms):
                issue = "propwash_or_rebound"
                confidence += 0.10

    confidence = float(np.clip(confidence, 0.0, 1.0))
    return issue, confidence


# ----------------------------
# Recommendations
# ----------------------------

def _recommend_pid_delta(
    axis: str,
    issue: str,
    confidence: float,
    tracking: TrackingStats,
    signal: SignalStats,
    limits: Dict[str, float],
) -> Tuple[Dict[str, float], List[str], List[str]]:
    """
    Returns conservative percentage deltas as fractions.
    Example: -0.03 means reduce by 3%.
    """
    delta = {"p": 0.0, "d": 0.0, "ff": 0.0}
    actions: List[str] = []
    warnings: List[str] = []

    # Scale all actions by confidence, but keep them conservative.
    strength = _confidence_bucket(confidence)

    if issue == "high_frequency_noise":
        delta["d"] = -0.02 * strength if limits["d"] > 0 else 0.0
        delta["p"] = -0.01 * strength
        delta["ff"] = 0.0
        actions.extend([
            f"{axis.upper()}: reduce D slightly first.",
            f"{axis.upper()}: inspect filtering, motor noise, props, and frame resonance before further PID changes.",
        ])
        warnings.append(f"{axis.upper()}: high-frequency noise usually needs filtering/mechanical fixes, not aggressive PID moves.")

    elif issue == "mid_frequency_vibration":
        delta["d"] = -0.02 * strength if limits["d"] > 0 else 0.0
        delta["p"] = -0.015 * strength
        actions.extend([
            f"{axis.upper()}: back off P and D slightly.",
            f"{axis.upper()}: check prop balance, arm stiffness, motor mounting, and filtering.",
        ])

    elif issue == "low_frequency_oscillation":
        delta["p"] = -0.025 * strength
        delta["d"] = -0.01 * strength if limits["d"] > 0 else 0.0
        actions.extend([
            f"{axis.upper()}: reduce P a little.",
            f"{axis.upper()}: if oscillation persists, reduce D slightly too.",
        ])

    elif issue == "propwash_or_rebound":
        delta["d"] = +0.02 * strength if limits["d"] > 0 else 0.0
        delta["p"] = -0.01 * strength
        actions.extend([
            f"{axis.upper()}: try a very small D increase for propwash/rebound control.",
            f"{axis.upper()}: keep changes small and monitor motor heat.",
        ])
        warnings.append(f"{axis.upper()}: propwash fixes can overheat motors if D is pushed too far.")

    elif issue == "poor_tracking":
        if tracking.lag_ms is not None and tracking.lag_ms > 25:
            delta["ff"] = +0.02 * strength
            actions.append(f"{axis.upper()}: try a small feedforward increase for better stick response.")
        else:
            delta["p"] = +0.01 * strength
            actions.append(f"{axis.upper()}: try a very small P increase if tracking feels soft.")
        if axis == "yaw":
            delta["d"] = 0.0

    elif issue == "slow_response":
        delta["ff"] = +0.015 * strength
        actions.append(f"{axis.upper()}: try a slight feedforward increase.")

    elif issue == "mostly_clean":
        actions.append(f"{axis.upper()}: no meaningful PID change recommended from this log.")

    # Clamp to per-axis maximums
    delta["p"] = float(np.clip(delta["p"], -limits["p"], limits["p"]))
    delta["d"] = float(np.clip(delta["d"], -limits["d"], limits["d"]))
    delta["ff"] = float(np.clip(delta["ff"], -limits["ff"], limits["ff"]))

    # Yaw safety
    if axis == "yaw":
        delta["d"] = 0.0

    return delta, actions, warnings


def _confidence_bucket(confidence: float) -> float:
    """
    Maps confidence to 0.5 / 0.75 / 1.0 multiplier.
    """
    if confidence < 0.45:
        return 0.5
    if confidence < 0.70:
        return 0.75
    return 1.0


def _is_axis_valid_for_pid(
    signal: SignalStats,
    tracking: TrackingStats,
    confidence: float,
    issue: str,
) -> bool:
    if confidence < 0.50:
        return False
    if signal.total_band_energy <= 0:
        return False

    # Allow diagnosis-only when no setpoint exists.
    # PID advice should be more cautious without tracking.
    if not tracking.usable and issue not in {
        "high_frequency_noise",
        "mid_frequency_vibration",
        "low_frequency_oscillation",
    }:
        return False

    return True


# ----------------------------
# Helpers for output
# ----------------------------

def _build_axis_recommendation_text(
    axis: str,
    issue: str,
    confidence: float,
    pid_delta_pct: Dict[str, float],
    valid_for_pid: bool,
) -> str:
    if not valid_for_pid:
        return (
            f"{axis.upper()}: {issue.replace('_', ' ')} detected, "
            f"but confidence is too low for safe PID changes."
        )

    parts = []
    if abs(pid_delta_pct["p"]) > 1e-9:
        parts.append(f"P {'+' if pid_delta_pct['p'] > 0 else ''}{pid_delta_pct['p'] * 100:.1f}%")
    if abs(pid_delta_pct["d"]) > 1e-9:
        parts.append(f"D {'+' if pid_delta_pct['d'] > 0 else ''}{pid_delta_pct['d'] * 100:.1f}%")
    if abs(pid_delta_pct["ff"]) > 1e-9:
        parts.append(f"FF {'+' if pid_delta_pct['ff'] > 0 else ''}{pid_delta_pct['ff'] * 100:.1f}%")

    if not parts:
        return f"{axis.upper()}: {issue.replace('_', ' ')} detected, but no PID change is recommended."

    return (
        f"{axis.upper()}: {issue.replace('_', ' ')} "
        f"(confidence {confidence:.2f}) → " + ", ".join(parts)
    )


def _build_summary(axis_results: Dict[str, Any], valid_for_pid: bool) -> str:
    issues = []
    for axis in AXES:
        if axis in axis_results:
            issues.append(f"{axis.upper()}: {axis_results[axis]['issue'].replace('_', ' ')}")

    base = " | ".join(issues)
    if valid_for_pid:
        return f"Analysis complete. Conservative PID deltas available. {base}"
    return f"Analysis complete. Diagnosis available, but PID deltas are gated or limited. {base}"


def _flatten_axis_recommendations(axis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    recs = []
    for axis in AXES:
        if axis not in axis_results:
            continue
        item = axis_results[axis]
        recs.append({
            "axis": axis,
            "valid_for_pid": item["valid_for_pid"],
            "issue": item["issue"],
            "confidence": item["confidence"],
            "pid_delta_pct": item["pid_delta_pct"],
            "recommendation_text": item["recommendation_text"],
            "actions": item["actions"],
            "warnings": item["warnings"],
        })
    return recs


def _find_first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = set(df.columns)
    for col in candidates:
        if col in cols:
            return col
    return None


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out