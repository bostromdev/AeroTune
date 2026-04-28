# Copyright © 2026 Christopher Bostrom. All Rights Reserved.
# Source-available for personal evaluation only. See LICENSE and NOTICE.

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

AXIS_COLUMNS = {
    "roll": ("gyro_x", "setpoint_roll"),
    "pitch": ("gyro_y", "setpoint_pitch"),
    "yaw": ("gyro_z", "setpoint_yaw"),
}

DRONE_COMPARISON_BANDS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "3": {"propwash": (75.0, 190.0), "high_noise": (140.0, 490.0)},
    "3.5": {"propwash": (65.0, 175.0), "high_noise": (125.0, 490.0)},
    "4": {"propwash": (55.0, 160.0), "high_noise": (110.0, 490.0)},
    "5": {"propwash": (45.0, 130.0), "high_noise": (90.0, 490.0)},
    "7": {"propwash": (35.0, 110.0), "high_noise": (70.0, 490.0)},
}

METRIC_WEIGHTS = {
    "tracking_error_ratio": 0.34,
    "abs_error_p95": 0.18,
    "propwash_ratio": 0.22,
    "high_noise_ratio": 0.18,
    "lag_abs_ms": 0.08,
}


class ComparisonError(ValueError):
    """Raised when two logs cannot be compared safely."""


def _finite(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(number):
        return default
    return number


def _round(value: Any, digits: int = 4) -> Optional[float]:
    number = _safe_float(value)
    if number is None:
        return None
    return round(number, digits)


def _rms(values: np.ndarray) -> float:
    arr = values[np.isfinite(values)]
    if len(arr) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(arr))))


def _series(df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in df.columns:
        return np.zeros(len(df), dtype=float)
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)


def _sample_rate_and_duration(df: pd.DataFrame) -> Tuple[float, float]:
    if "time" not in df.columns or len(df) < 2:
        return 0.0, 0.0

    time = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=float)
    time = time[np.isfinite(time)]
    if len(time) < 2:
        return 0.0, 0.0

    time = np.sort(time)
    dt = np.diff(time)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return 0.0, float(time[-1] - time[0])

    sample_rate = 1.0 / float(np.median(dt))
    duration = float(time[-1] - time[0])
    return sample_rate, duration


def _band_ratio(signal: np.ndarray, sample_rate_hz: float, low_hz: float, high_hz: float) -> float:
    arr = signal[np.isfinite(signal)]
    if len(arr) < 128 or sample_rate_hz <= 0:
        return 0.0

    max_points = 8192
    step = max(1, int(np.ceil(len(arr) / max_points)))
    arr = arr[::step]
    effective_rate = sample_rate_hz / step

    arr = arr - np.mean(arr)
    if np.std(arr) < 1e-9:
        return 0.0

    window = np.hanning(len(arr))
    spectrum = np.abs(np.fft.rfft(arr * window))
    freqs = np.fft.rfftfreq(len(arr), d=1.0 / effective_rate)
    power = np.square(spectrum)
    total = float(np.sum(power[freqs > 0]))
    if total <= 1e-12:
        return 0.0

    mask = (freqs >= low_hz) & (freqs < high_hz)
    return float(np.sum(power[mask]) / total)


def _dominant_frequency(signal: np.ndarray, sample_rate_hz: float) -> Optional[float]:
    arr = signal[np.isfinite(signal)]
    if len(arr) < 128 or sample_rate_hz <= 0:
        return None

    max_points = 8192
    step = max(1, int(np.ceil(len(arr) / max_points)))
    arr = arr[::step]
    effective_rate = sample_rate_hz / step

    arr = arr - np.mean(arr)
    if np.std(arr) < 1e-9:
        return None

    window = np.hanning(len(arr))
    spectrum = np.abs(np.fft.rfft(arr * window))
    freqs = np.fft.rfftfreq(len(arr), d=1.0 / effective_rate)
    valid = freqs > 0
    if not np.any(valid):
        return None

    freqs = freqs[valid]
    spectrum = spectrum[valid]
    return float(freqs[int(np.argmax(spectrum))])


def _estimate_lag_ms(setpoint: np.ndarray, gyro: np.ndarray, sample_rate_hz: float) -> Optional[float]:
    n = min(len(setpoint), len(gyro))
    if n < 128 or sample_rate_hz <= 0:
        return None

    sp = setpoint[:n]
    gy = gyro[:n]
    valid = np.isfinite(sp) & np.isfinite(gy)
    sp = sp[valid]
    gy = gy[valid]

    if len(sp) < 128 or np.std(sp) < 1e-9 or np.std(gy) < 1e-9:
        return None

    max_points = 2400
    step = max(1, int(np.ceil(len(sp) / max_points)))
    sp = sp[::step] - np.mean(sp[::step])
    gy = gy[::step] - np.mean(gy[::step])
    rate = sample_rate_hz / step

    max_lag = min(int(rate * 0.15), len(sp) // 4)
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

        if len(a) < 32:
            continue

        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        score = float(np.dot(a, b) / denom) if denom > 0 else -np.inf
        if score > best_score:
            best_score = score
            best_lag = lag

    return float(best_lag / rate * 1000.0)


def _axis_from_analysis(analysis: Dict[str, Any], axis: str) -> Dict[str, Any]:
    axes = analysis.get("axes") if isinstance(analysis, dict) else None
    if isinstance(axes, dict) and isinstance(axes.get(axis), dict):
        return axes[axis]
    if isinstance(analysis.get(axis), dict):
        return analysis[axis]
    return {}


def _axis_metrics(
    df: pd.DataFrame,
    analysis: Dict[str, Any],
    axis: str,
    drone_size: str,
) -> Dict[str, Any]:
    gyro_col, setpoint_col = AXIS_COLUMNS[axis]
    sample_rate_hz, duration_s = _sample_rate_and_duration(df)
    bands = DRONE_COMPARISON_BANDS.get(str(drone_size), DRONE_COMPARISON_BANDS["7"])

    gyro = _series(df, gyro_col)
    setpoint = _series(df, setpoint_col)
    n = min(len(gyro), len(setpoint))
    gyro = gyro[:n]
    setpoint = setpoint[:n]
    valid = np.isfinite(gyro) & np.isfinite(setpoint)
    gyro = gyro[valid]
    setpoint = setpoint[valid]

    if len(gyro) < 64:
        raise ComparisonError(f"{axis} does not have enough usable samples for comparison.")

    error = gyro - setpoint
    gyro_rms = _rms(gyro)
    setpoint_rms = _rms(setpoint)
    error_rms = _rms(error)
    abs_error = np.abs(error[np.isfinite(error)])
    abs_error_p95 = float(np.percentile(abs_error, 95)) if len(abs_error) else 0.0

    denominator = max(setpoint_rms, 1.0)
    tracking_error_ratio = float(error_rms / denominator)
    lag_ms = _estimate_lag_ms(setpoint, gyro, sample_rate_hz)

    propwash_ratio = _band_ratio(error, sample_rate_hz, *bands["propwash"])
    high_noise_ratio = _band_ratio(gyro, sample_rate_hz, *bands["high_noise"])
    dominant_freq_hz = _dominant_frequency(gyro, sample_rate_hz)

    axis_analysis = _axis_from_analysis(analysis, axis)
    signal = axis_analysis.get("signal", {}) if isinstance(axis_analysis.get("signal"), dict) else {}
    tracking = axis_analysis.get("tracking", {}) if isinstance(axis_analysis.get("tracking"), dict) else {}

    # Prefer the analyzer's own values when available so V1.4 matches the existing V1 tune read.
    if _safe_float(signal.get("error_ratio")) is not None:
        tracking_error_ratio = float(signal["error_ratio"])
    if _safe_float(signal.get("propwash_ratio")) is not None:
        propwash_ratio = float(signal["propwash_ratio"])
    if _safe_float(signal.get("high_ratio")) is not None:
        high_noise_ratio = float(signal["high_ratio"])
    if _safe_float(signal.get("dominant_freq_hz")) is not None:
        dominant_freq_hz = float(signal["dominant_freq_hz"])
    if _safe_float(tracking.get("lag_ms")) is not None:
        lag_ms = float(tracking["lag_ms"])

    return {
        "axis": axis,
        "issue": axis_analysis.get("issue"),
        "issue_label": axis_analysis.get("issue_label"),
        "severity": axis_analysis.get("severity"),
        "confidence": _round(axis_analysis.get("confidence"), 3),
        "pid_delta_pct": axis_analysis.get("pid_delta_pct", {}),
        "sample_rate_hz": _round(sample_rate_hz, 2),
        "duration_seconds": _round(duration_s, 3),
        "usable_rows": int(len(df)),
        "gyro_rms": _round(gyro_rms, 4),
        "setpoint_rms": _round(setpoint_rms, 4),
        "error_rms": _round(error_rms, 4),
        "tracking_error_ratio": _round(tracking_error_ratio, 5),
        "abs_error_p95": _round(abs_error_p95, 4),
        "propwash_ratio": _round(propwash_ratio, 5),
        "high_noise_ratio": _round(high_noise_ratio, 5),
        "dominant_freq_hz": _round(dominant_freq_hz, 2),
        "lag_ms": _round(lag_ms, 3),
        "lag_abs_ms": _round(abs(lag_ms), 3) if lag_ms is not None else None,
    }


def _lower_is_better_change(before: Optional[float], after: Optional[float]) -> Dict[str, Any]:
    b = _safe_float(before)
    a = _safe_float(after)

    if b is None or a is None:
        return {
            "before": before,
            "after": after,
            "delta": None,
            "change_pct": None,
            "improvement_pct": None,
            "direction": "unknown",
        }

    delta = a - b
    if abs(b) < 1e-9:
        change_pct = None
        improvement_pct = 0.0 if abs(a) < 1e-9 else -100.0
    else:
        change_pct = (delta / abs(b)) * 100.0
        improvement_pct = -change_pct

    if improvement_pct > 5:
        direction = "improved"
    elif improvement_pct < -5:
        direction = "worse"
    else:
        direction = "unchanged"

    return {
        "before": _round(b, 5),
        "after": _round(a, 5),
        "delta": _round(delta, 5),
        "change_pct": _round(change_pct, 2),
        "improvement_pct": _round(improvement_pct, 2),
        "direction": direction,
    }


def _score_axis(metric_changes: Dict[str, Dict[str, Any]]) -> float:
    score = 0.0
    used_weight = 0.0

    for metric, weight in METRIC_WEIGHTS.items():
        improvement = _safe_float(metric_changes.get(metric, {}).get("improvement_pct"))
        if improvement is None:
            continue
        score += float(np.clip(improvement, -100.0, 100.0)) * weight
        used_weight += weight

    if used_weight <= 0:
        return 0.0
    return score / used_weight


def _verdict_from_score(score: float) -> str:
    if score >= 8.0:
        return "improved"
    if score <= -8.0:
        return "worse"
    return "mostly unchanged"


def _confidence_notes(before: Dict[str, Any], after: Dict[str, Any]) -> Tuple[float, List[str]]:
    confidence = 0.72
    notes: List[str] = []

    before_duration = _safe_float(before.get("duration_seconds"), 0.0) or 0.0
    after_duration = _safe_float(after.get("duration_seconds"), 0.0) or 0.0
    if min(before_duration, after_duration) < 10.0:
        confidence -= 0.18
        notes.append("One or both logs are short. Use 60–90 second comparison flights for stronger proof.")

    if max(before_duration, after_duration) > 0:
        duration_ratio = min(before_duration, after_duration) / max(before_duration, after_duration)
        if duration_ratio < 0.65:
            confidence -= 0.12
            notes.append("Before/after log durations are not very similar.")

    before_rate = _safe_float(before.get("sample_rate_hz"), 0.0) or 0.0
    after_rate = _safe_float(after.get("sample_rate_hz"), 0.0) or 0.0
    if min(before_rate, after_rate) <= 0:
        confidence -= 0.12
        notes.append("Sample rate could not be read from one log.")
    elif min(before_rate, after_rate) / max(before_rate, after_rate) < 0.70:
        confidence -= 0.10
        notes.append("Before/after sample rates are very different.")

    before_activity = _safe_float(before.get("setpoint_rms"), 0.0) or 0.0
    after_activity = _safe_float(after.get("setpoint_rms"), 0.0) or 0.0
    if min(before_activity, after_activity) < 1.0:
        confidence -= 0.10
        notes.append("Stick/setpoint activity is low in one log, so tracking comparison may be weak.")

    confidence = float(np.clip(confidence, 0.25, 0.95))
    if not notes:
        notes.append("Logs are similar enough for a useful first-pass before/after comparison.")
    return round(confidence, 3), notes


def build_multilog_comparison(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    before_analysis: Dict[str, Any],
    after_analysis: Dict[str, Any],
    drone_size: str,
    tuning_goal: str,
) -> Dict[str, Any]:
    """
    Compare two already-parsed AeroTune dataframes and analyzer results.

    Positive improvement scores mean the after log looks better. Negative scores mean the
    after log looks worse. This is intentionally conservative and should be treated as
    validation guidance, not an automatic final tune decision.
    """
    if before_df is None or before_df.empty:
        raise ComparisonError("Before log is empty after parsing.")
    if after_df is None or after_df.empty:
        raise ComparisonError("After log is empty after parsing.")

    axis_results: Dict[str, Any] = {}
    score_values: List[float] = []
    confidence_values: List[float] = []
    warnings: List[str] = []

    for axis in AXIS_COLUMNS:
        before_metrics = _axis_metrics(before_df, before_analysis or {}, axis, drone_size)
        after_metrics = _axis_metrics(after_df, after_analysis or {}, axis, drone_size)

        metric_changes = {
            metric: _lower_is_better_change(before_metrics.get(metric), after_metrics.get(metric))
            for metric in METRIC_WEIGHTS
        }

        axis_score = _score_axis(metric_changes)
        axis_verdict = _verdict_from_score(axis_score)
        axis_confidence, axis_notes = _confidence_notes(before_metrics, after_metrics)

        score_values.append(axis_score)
        confidence_values.append(axis_confidence)
        warnings.extend(axis_notes)

        axis_results[axis] = {
            "axis": axis,
            "verdict": axis_verdict,
            "score_pct": round(axis_score, 2),
            "confidence": axis_confidence,
            "before": before_metrics,
            "after": after_metrics,
            "metric_changes": metric_changes,
            "notes": axis_notes,
            "read": _axis_read(axis, axis_verdict, axis_score, metric_changes),
        }

    overall_score = float(np.mean(score_values)) if score_values else 0.0
    overall_confidence = float(np.mean(confidence_values)) if confidence_values else 0.25
    overall_verdict = _verdict_from_score(overall_score)

    best_axis = max(axis_results.values(), key=lambda item: item["score_pct"])
    worst_axis = min(axis_results.values(), key=lambda item: item["score_pct"])

    before_clean = bool((before_analysis or {}).get("clean_baseline", False))
    after_clean = bool((after_analysis or {}).get("clean_baseline", False))
    after_gate = (after_analysis or {}).get("style_gate", {}) if isinstance(after_analysis, dict) else {}
    requested_goal = str((after_analysis or {}).get("requested_tuning_goal", tuning_goal))
    effective_goal = str((after_analysis or {}).get("effective_tuning_goal", tuning_goal))

    if after_clean:
        baseline_message = (
            "After log is clean enough to move from baseline cleanup into style tuning. "
            "Locked-In or Cinematic changes should still be small and verified with another similar log."
        )
    else:
        baseline_message = (
            "After log is not clean enough for style tuning yet. Keep working in Efficient/Smooth baseline cleanup mode."
        )
        warnings.append("Do not switch to Locked-In/Cinematic style tuning until the after log is clean.")

    next_step = _next_step(overall_verdict, worst_axis)
    if after_clean:
        next_step += " Clean baseline confirmed; you may now choose Locked-In or Cinematic if you want a different feel."
    else:
        next_step += " Baseline is still not clean, so do not chase Locked-In/Cinematic feel yet."

    baseline_workflow = {
        "before_clean_baseline": before_clean,
        "after_clean_baseline": after_clean,
        "ready_for_style_tuning": after_clean,
        "requested_tuning_goal": requested_goal,
        "effective_tuning_goal": effective_goal,
        "after_style_gate": after_gate,
        "message": baseline_message,
        "rule": "Clean baseline first; style tuning second.",
    }

    return {
        "version": "V1.4",
        "type": "before_after_log_comparison",
        "drone_size": str(drone_size),
        "tuning_goal": str(tuning_goal),
        "overall_verdict": overall_verdict,
        "overall_score_pct": round(overall_score, 2),
        "confidence": round(overall_confidence, 3),
        "summary": _overall_summary(overall_verdict, overall_score, best_axis, worst_axis),
        "best_axis": best_axis["axis"],
        "worst_axis": worst_axis["axis"],
        "axes": axis_results,
        "warnings": sorted(set(warnings)),
        "next_step": next_step,
        "baseline_workflow": baseline_workflow,
        "ready_for_style_tuning": after_clean,
        "interpretation": {
            "positive_score": "After log improved versus before log.",
            "negative_score": "After log worsened versus before log.",
            "thresholds": "+8% or better = improved, -8% or worse = worse, otherwise mostly unchanged.",
            "lower_is_better_metrics": list(METRIC_WEIGHTS.keys()),
        },
    }


def _axis_read(axis: str, verdict: str, score: float, changes: Dict[str, Dict[str, Any]]) -> str:
    improved = [name for name, change in changes.items() if change.get("direction") == "improved"]
    worse = [name for name, change in changes.items() if change.get("direction") == "worse"]

    if verdict == "improved":
        if improved:
            return f"{axis.upper()} improved mainly in {', '.join(improved[:2])}."
        return f"{axis.upper()} improved overall."

    if verdict == "worse":
        if worse:
            return f"{axis.upper()} got worse mainly in {', '.join(worse[:2])}."
        return f"{axis.upper()} got worse overall."

    return f"{axis.upper()} is mostly unchanged; score {score:.1f}%."


def _overall_summary(verdict: str, score: float, best_axis: Dict[str, Any], worst_axis: Dict[str, Any]) -> str:
    if verdict == "improved":
        return (
            f"After log looks better overall by {score:.1f}%. "
            f"Best improvement was on {best_axis['axis'].upper()}; weakest axis is {worst_axis['axis'].upper()}."
        )
    if verdict == "worse":
        return (
            f"After log looks worse overall by {abs(score):.1f}%. "
            f"Worst change was on {worst_axis['axis'].upper()}; review that axis before continuing."
        )
    return (
        f"After log is mostly unchanged overall ({score:.1f}%). "
        f"The tune change may have been too small, or the two flights may not be similar enough."
    )


def _next_step(verdict: str, worst_axis: Dict[str, Any]) -> str:
    axis = worst_axis.get("axis", "axis").upper()
    if verdict == "improved":
        return (
            "Keep the change small and confirm with one more similar flight. "
            f"Watch {axis}, because it is still the weakest comparison axis."
        )
    if verdict == "worse":
        return (
            f"Do not stack more PID changes yet. Revert or halve the last change affecting {axis}, "
            "then fly the same test again."
        )
    return (
        "Do one clear, isolated tune change and repeat the same 60–90 second test route. "
        f"Prioritize {axis}, because it showed the weakest comparison result."
    )
