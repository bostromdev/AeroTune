import numpy as np

DRONE_PROFILES = {
    "1": {"low": (10, 80), "mid": (80, 200), "high": (200, 500)},
    "2": {"low": (10, 60), "mid": (60, 160), "high": (160, 400)},
    "3": {"low": (8, 50), "mid": (50, 140), "high": (140, 350)},
    "4": {"low": (5, 40), "mid": (40, 120), "high": (120, 300)},
    "5": {"low": (5, 30), "mid": (30, 100), "high": (100, 250)},
    "7": {"low": (5, 30), "mid": (30, 100), "high": (100, 250)},
}

MAX_REASONABLE_TRACKING_DELAY_MS = 120.0
MIN_TRACKING_ACTIVITY_STD = 0.02
MIN_TRACKING_CORRELATION = 0.25


def to_python_types(obj):
    if isinstance(obj, dict):
        return {k: to_python_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python_types(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_python_types(v) for v in obj)
    if isinstance(obj, np.ndarray):
        return [to_python_types(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def _safe_array(values):
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def _median_dt(time_array):
    diffs = np.diff(time_array)
    diffs = diffs[np.isfinite(diffs)]
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        raise ValueError("Invalid or non-increasing time column")
    return float(np.median(diffs))


def _validate_log(df):
    if "time" not in df.columns:
        raise ValueError("Missing required column: time")
    if len(df) < 256:
        raise ValueError("Log too short for reliable spectral analysis (need at least 256 samples)")

    time = _safe_array(df["time"].values)
    if not np.all(np.isfinite(time)):
        raise ValueError("Time column contains invalid values")

    dt = _median_dt(time)
    sample_rate = 1.0 / dt
    if sample_rate < 50:
        raise ValueError("Sampling rate too low for gyro frequency analysis")

    axes = [c for c in ("gyro_x", "gyro_y", "gyro_z") if c in df.columns]
    if not axes:
        raise ValueError("No gyro axis columns found (expected gyro_x / gyro_y / gyro_z)")

    return time, dt, sample_rate, axes


def _windowed_spectrum(signal, dt):
    signal = _safe_array(signal)
    signal = signal[np.isfinite(signal)]
    n = len(signal)
    if n < 64:
        return np.array([]), np.array([]), np.array([])

    centered = signal - np.mean(signal)
    window = np.hanning(n)
    fft = np.fft.rfft(centered * window)
    freqs = np.fft.rfftfreq(n, d=dt)
    amplitude = np.abs(fft)
    power = amplitude ** 2
    return freqs, amplitude, power


def _band_energy(freqs, power, bounds):
    lo, hi = bounds
    mask = (freqs >= lo) & (freqs < hi)
    if not np.any(mask):
        return 0.0
    return float(np.sum(power[mask]))


def _band_metrics(freqs, power, profile):
    low_energy = _band_energy(freqs, power, profile["low"])
    mid_energy = _band_energy(freqs, power, profile["mid"])
    high_energy = _band_energy(freqs, power, profile["high"])
    total_energy = low_energy + mid_energy + high_energy + 1e-12

    ratios = {
        "low": low_energy / total_energy,
        "mid": mid_energy / total_energy,
        "high": high_energy / total_energy,
    }
    dominant = max(ratios, key=ratios.get)
    return {
        "low_energy": low_energy,
        "mid_energy": mid_energy,
        "high_energy": high_energy,
        "low_ratio": ratios["low"],
        "mid_ratio": ratios["mid"],
        "high_ratio": ratios["high"],
        "dominant_band": dominant,
        "total_energy": total_energy,
    }


def _dominant_peak(freqs, amplitude, low_hz=5.0, high_hz=500.0):
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(mask):
        return 0.0, 0.0, 0.0

    f = freqs[mask]
    a = amplitude[mask]
    idx = int(np.argmax(a))
    peak_freq = float(f[idx])
    peak_amp = float(a[idx])
    floor = float(np.median(a)) + 1e-12
    significance = peak_amp / floor
    return peak_freq, peak_amp, significance


def _dominant_peak_in_band(freqs, amplitude, bounds):
    lo, hi = bounds
    mask = (freqs >= lo) & (freqs < hi)
    if not np.any(mask):
        return None

    local_f = freqs[mask]
    local_a = amplitude[mask]
    idx = int(np.argmax(local_a))
    return float(local_f[idx])


def _detect_harmonics(freqs, amplitude, fundamental, significance_floor=3.0):
    if not fundamental or fundamental <= 0:
        return []

    harmonics = []
    floor = float(np.median(amplitude)) + 1e-12

    for multiple in (2, 3, 4):
        target = fundamental * multiple
        if target > freqs[-1]:
            break
        mask = np.abs(freqs - target) <= max(3.0, fundamental * 0.03)
        if not np.any(mask):
            continue
        local_amp = float(np.max(amplitude[mask]))
        if local_amp / floor >= significance_floor:
            harmonics.append(round(target, 1))
    return harmonics


def _best_tracking_delay_ms(setpoint, gyro, dt):
    max_lag_samples = max(1, int(round((MAX_REASONABLE_TRACKING_DELAY_MS / 1000.0) / dt)))
    s = setpoint - np.mean(setpoint)
    g = gyro - np.mean(gyro)

    if np.allclose(s, 0) or np.allclose(g, 0):
        return 0.0, 0.0

    corr = np.correlate(g, s, mode="full")
    lags = np.arange(-len(s) + 1, len(g))
    center_mask = (lags >= -max_lag_samples) & (lags <= max_lag_samples)
    corr_local = corr[center_mask]
    lags_local = lags[center_mask]

    if len(corr_local) == 0:
        return 0.0, 0.0

    idx = int(np.argmax(corr_local))
    lag_idx = int(lags_local[idx])
    delay_ms = float(lag_idx * dt * 1000.0)

    denom = np.linalg.norm(g) * np.linalg.norm(s) + 1e-12
    corr_strength = float(corr_local[idx] / denom)
    return delay_ms, corr_strength


def analyze_setpoint_response(setpoint, gyro, dt):
    setpoint = _safe_array(setpoint)
    gyro = _safe_array(gyro)
    n = min(len(setpoint), len(gyro))

    if n < 32:
        return {
            "available": False,
            "valid_for_pid": False,
            "mean_tracking_error": None,
            "overshoot": None,
            "oscillation_rate": None,
            "delay_ms": None,
            "tracking_score": None,
            "tuning_notes": ["Not enough setpoint data for tracking analysis"],
        }

    setpoint = setpoint[:n]
    gyro = gyro[:n]

    valid = np.isfinite(setpoint) & np.isfinite(gyro)
    setpoint = setpoint[valid]
    gyro = gyro[valid]

    if len(setpoint) < 32:
        return {
            "available": False,
            "valid_for_pid": False,
            "mean_tracking_error": None,
            "overshoot": None,
            "oscillation_rate": None,
            "delay_ms": None,
            "tracking_score": None,
            "tuning_notes": ["Not enough valid setpoint data for tracking analysis"],
        }

    if max(float(np.std(setpoint)), float(np.std(gyro))) < MIN_TRACKING_ACTIVITY_STD:
        return {
            "available": True,
            "valid_for_pid": False,
            "mean_tracking_error": None,
            "overshoot": None,
            "oscillation_rate": None,
            "delay_ms": None,
            "tracking_score": None,
            "tuning_notes": ["Tracking activity too small for reliable PID guidance"],
        }

    error = setpoint - gyro
    max_setpoint = float(np.max(np.abs(setpoint))) + 1e-9
    max_gyro = float(np.max(np.abs(gyro))) + 1e-9

    mean_error = float(np.mean(np.abs(error)))
    norm_error = mean_error / max(max_setpoint, max_gyro)

    overshoot = float(max(0.0, np.max(np.abs(gyro)) - np.max(np.abs(setpoint))))
    overshoot_ratio = overshoot / max(max_setpoint, 1e-9)

    error_centered = error - np.mean(error)
    zero_crossings = np.where(np.diff(np.signbit(error_centered)))[0]
    oscillation_rate = float(len(zero_crossings) / max(len(error_centered), 1))

    delay_ms, corr_strength = _best_tracking_delay_ms(setpoint, gyro, dt)

    valid_for_pid = True
    notes = []

    if abs(delay_ms) > MAX_REASONABLE_TRACKING_DELAY_MS:
        valid_for_pid = False
        notes.append("Tracking invalid → possible log misalignment")
    if corr_strength < MIN_TRACKING_CORRELATION:
        valid_for_pid = False
        notes.append("Weak gyro/setpoint correlation → tracking not trustworthy for PID advice")

    tracking_score = 100.0
    tracking_score -= min(50.0, norm_error * 100.0)
    tracking_score -= min(20.0, overshoot_ratio * 45.0)
    tracking_score -= min(16.0, abs(delay_ms) * 0.20)
    tracking_score -= min(18.0, max(0.0, 0.35 - corr_strength) * 90.0)
    tracking_score = max(0.0, min(100.0, tracking_score))

    if overshoot_ratio > 0.12:
        notes.append("Overshoot present → reduce P slightly or add damping")
    if oscillation_rate > 0.08:
        notes.append("Rapid error sign changes → oscillation or noisy control response")
    if valid_for_pid:
        if delay_ms > 12:
            notes.append("Gyro lagging setpoint → consider more P or feedforward")
        elif delay_ms < -12:
            notes.append("Gyro appears to lead setpoint → inspect filtering or aggressive feedforward")
    if not notes:
        notes.append("Tracking looks reasonably controlled")

    return {
        "available": True,
        "valid_for_pid": valid_for_pid,
        "mean_tracking_error": round(mean_error, 4),
        "overshoot": round(overshoot, 4),
        "oscillation_rate": round(oscillation_rate, 4),
        "delay_ms": round(delay_ms, 2),
        "tracking_score": round(tracking_score, 1),
        "tuning_notes": notes,
    }


def _axis_mapping(col):
    return {
        "gyro_x": ("ROLL", "setpoint_roll", {"P": 60, "D": 30, "FF": 80}),
        "gyro_y": ("PITCH", "setpoint_pitch", {"P": 65, "D": 32, "FF": 85}),
        "gyro_z": ("YAW", "setpoint_yaw", {"P": 50, "D": 0, "FF": 70}),
    }[col]


def _clamp(value, lo, hi):
    return int(max(lo, min(hi, value)))


def analyze_axis(df, col, dt, drone_size="7"):
    profile = DRONE_PROFILES.get(str(drone_size), DRONE_PROFILES["7"])
    axis_name, setpoint_col, base_pid = _axis_mapping(col)

    raw_signal = _safe_array(df[col].values)
    raw_signal = raw_signal[np.isfinite(raw_signal)]

    freqs, amplitude, power = _windowed_spectrum(raw_signal, dt)
    if len(freqs) == 0:
        raise ValueError("Axis %s does not have enough valid samples" % col)

    global_peak_hz, _, peak_significance = _dominant_peak(freqs, amplitude, 5.0, min(500.0, freqs[-1]))
    bands = _band_metrics(freqs, power, profile)
    low_ratio = bands["low_ratio"]
    mid_ratio = bands["mid_ratio"]
    high_ratio = bands["high_ratio"]
    dominant_band = bands["dominant_band"]

    displayed_peak_hz = _dominant_peak_in_band(freqs, amplitude, profile[dominant_band])
    if displayed_peak_hz is None:
        displayed_peak_hz = global_peak_hz

    harmonics = _detect_harmonics(freqs, amplitude, global_peak_hz, significance_floor=3.2)
    rpm_estimate = int(round(displayed_peak_hz * 60)) if displayed_peak_hz and displayed_peak_hz >= 45 else None

    if setpoint_col in df.columns:
        tracking = analyze_setpoint_response(df[setpoint_col].values, df[col].values, dt)
    else:
        tracking = {
            "available": False,
            "valid_for_pid": False,
            "mean_tracking_error": None,
            "overshoot": None,
            "oscillation_rate": None,
            "delay_ms": None,
            "tracking_score": None,
            "tuning_notes": ["No matching setpoint column found"],
        }

    issue = "Mixed vibration signature"
    severity = "low"
    explanation = []

    if dominant_band == "low" and low_ratio >= 0.34:
        issue = "Low-frequency oscillation (tune / frame / propwash region)"
        severity = "high" if low_ratio >= 0.55 else "medium"
        explanation.append("Low-band energy dominates the axis response")
    elif dominant_band == "mid" and mid_ratio >= 0.30:
        issue = "Mid-frequency vibration (prop / motor / mechanical resonance)"
        severity = "high" if mid_ratio >= 0.50 else "medium"
        explanation.append("Mid-band energy dominates, consistent with mechanical vibration")
    elif dominant_band == "high" and high_ratio >= 0.28:
        issue = "High-frequency noise (motor / filtering / D-term region)"
        severity = "high" if high_ratio >= 0.48 else "medium"
        explanation.append("High-band energy dominates, suggesting motor-order or filtering noise")
    else:
        issue = "No strong vibration signature"
        severity = "low"
        explanation.append("No single band dominates strongly enough to call a clear fault")

    propwash_detected = bool(low_ratio >= 0.26 and mid_ratio >= 0.22)
    if propwash_detected:
        explanation.append("Low + mid energy mix suggests propwash or turbulent airflow interaction")
    if harmonics:
        explanation.append("Motor-order harmonics are present in the spectrum")

    p_adj = 0
    d_adj = 0
    ff_adj = 0

    if issue.startswith("Low-frequency"):
        p_adj -= 4
        d_adj += 5
    elif issue.startswith("Mid-frequency"):
        p_adj -= 2
        d_adj -= 2
    elif issue.startswith("High-frequency"):
        d_adj -= 7
        p_adj -= 1
        ff_adj += 1

    if tracking.get("available") and tracking.get("valid_for_pid"):
        overshoot = tracking.get("overshoot") or 0.0
        delay_ms = tracking.get("delay_ms") or 0.0
        tracking_score = tracking.get("tracking_score")
        oscillation_rate = tracking.get("oscillation_rate") or 0.0

        if overshoot > 0.12:
            p_adj -= 3
            if base_pid["D"] > 0 and dominant_band != "high":
                d_adj += 2
            explanation.append("Tracking overshoot supports less P and a little more damping")
        elif overshoot < 0.035 and delay_ms > 10:
            p_adj += 3
            ff_adj += 2
            explanation.append("Tracking is soft and laggy, so more P/FF is reasonable")

        if delay_ms > 16:
            ff_adj += 4
            if dominant_band != "high":
                p_adj += 1
            explanation.append("Noticeable delay supports more feedforward")
        elif delay_ms < -14:
            ff_adj -= 2
            explanation.append("Unexpected lead suggests feedforward may be too aggressive")

        if oscillation_rate > 0.09 and base_pid["D"] > 0:
            if dominant_band == "high":
                d_adj -= 2
            elif dominant_band == "low":
                d_adj += 2

        if tracking_score is not None and tracking_score >= 84 and issue == "No strong vibration signature":
            p_adj += 2
            ff_adj += 1
            if base_pid["D"] > 0 and high_ratio < 0.20:
                d_adj += 1
            explanation.append("Clean tracking allows a slightly more responsive tune")
        elif tracking_score is not None and tracking_score < 55 and issue == "No strong vibration signature":
            p_adj -= 2
            ff_adj -= 1
            explanation.append("Poor tracking without a strong vibration peak suggests backing off aggressiveness")
    elif tracking.get("available") and not tracking.get("valid_for_pid"):
        explanation.append("Tracking data was excluded from PID decisions due to misalignment or weak correlation")

    if high_ratio >= 0.52:
        d_adj -= 2
    if low_ratio >= 0.48:
        d_adj += 1
    if mid_ratio >= 0.45:
        p_adj -= 1
    if propwash_detected:
        p_adj -= 1
        if base_pid["D"] > 0:
            d_adj += 1

    p_adj = _clamp(p_adj, -12, 8)
    d_adj = _clamp(d_adj, -12, 8) if base_pid["D"] > 0 else 0
    ff_adj = _clamp(ff_adj, -6, 8)

    recommended_pid = {
        "P": int(round(base_pid["P"] * (1 + p_adj / 100.0))),
        "D": int(round(base_pid["D"] * (1 + d_adj / 100.0))),
        "FF": int(round(base_pid["FF"] * (1 + ff_adj / 100.0))),
    }

    spectral_penalty = 0.0
    spectral_penalty += low_ratio * 24.0
    spectral_penalty += mid_ratio * 30.0
    spectral_penalty += high_ratio * 22.0
    if peak_significance >= 3.0:
        spectral_penalty += min(10.0, (peak_significance - 3.0) * 1.8)
    if propwash_detected:
        spectral_penalty += 6.0
    if harmonics and high_ratio > 0.34:
        spectral_penalty += min(5.0, len(harmonics) * 1.2)

    axis_score = max(0.0, min(100.0, 100.0 - spectral_penalty))
    if tracking.get("available") and tracking.get("tracking_score") is not None and tracking.get("valid_for_pid"):
        axis_score = 0.58 * axis_score + 0.42 * tracking["tracking_score"]
    axis_score = max(0.0, min(100.0, axis_score))

    improvement = abs(p_adj) * 0.9 + abs(d_adj) * 0.9 + abs(ff_adj) * 0.7
    if propwash_detected:
        improvement += 1.5
    expected_score = max(axis_score, min(100.0, axis_score + improvement))

    tuning_objective = "Maximize usable speed and efficiency by reducing wasted vibration energy while keeping tracking crisp."

    return {
        "axis_name": axis_name,
        "issue": issue,
        "severity": severity,
        "dominant_band": dominant_band,
        "peak_frequency_hz": round(displayed_peak_hz, 1) if displayed_peak_hz else 0.0,
        "global_peak_hz": round(global_peak_hz, 1) if global_peak_hz else 0.0,
        "peak_significance": round(peak_significance, 2),
        "rpm_estimate": rpm_estimate,
        "harmonics_detected": harmonics,
        "propwash_detected": propwash_detected,
        "band_energy_ratio": {
            "low": round(low_ratio, 4),
            "mid": round(mid_ratio, 4),
            "high": round(high_ratio, 4),
        },
        "tracking": tracking,
        "explanation": explanation,
        "tuning_objective": tuning_objective,
        "pid_suggestions": {
            "p_adjustment_percent": p_adj,
            "d_adjustment_percent": d_adj,
            "ff_adjustment_percent": ff_adj,
        },
        "recommended_pid": recommended_pid,
        "axis_score": round(axis_score, 1),
        "expected_score": round(expected_score, 1),
    }


def detect_oscillation(df, drone_size="7"):
    _, dt, sample_rate, axes = _validate_log(df)

    results = {}
    scores = []

    for col in axes:
        try:
            axis_result = analyze_axis(df, col, dt, drone_size=drone_size)
            results[col] = axis_result
            scores.append(axis_result["axis_score"])
        except Exception as exc:
            results[col] = {
                "axis_name": col.upper(),
                "issue": "Analysis failed: %s" % exc,
                "severity": "high",
                "dominant_band": "unknown",
                "peak_frequency_hz": 0.0,
                "global_peak_hz": 0.0,
                "peak_significance": 0.0,
                "rpm_estimate": None,
                "harmonics_detected": [],
                "propwash_detected": False,
                "band_energy_ratio": {"low": 0.0, "mid": 0.0, "high": 0.0},
                "tracking": {
                    "available": False,
                    "valid_for_pid": False,
                    "mean_tracking_error": None,
                    "overshoot": None,
                    "oscillation_rate": None,
                    "delay_ms": None,
                    "tracking_score": None,
                    "tuning_notes": ["Axis analysis failed"],
                },
                "explanation": ["Axis could not be analyzed reliably"],
                "tuning_objective": "Stabilize analysis before tune changes.",
                "pid_suggestions": {
                    "p_adjustment_percent": 0,
                    "d_adjustment_percent": 0,
                    "ff_adjustment_percent": 0,
                },
                "recommended_pid": {"P": 0, "D": 0, "FF": 0},
                "axis_score": 0.0,
                "expected_score": 0.0,
            }
            scores.append(0.0)

    overall_score = round(float(np.mean(scores)), 1) if scores else 0.0
    if overall_score >= 85:
        rating = "Excellent"
    elif overall_score >= 70:
        rating = "Good"
    elif overall_score >= 50:
        rating = "Needs Improvement"
    else:
        rating = "Poor"

    summary_notes = []
    severe_axes = [k for k, v in results.items() if v.get("severity") == "high"]
    if severe_axes:
        summary_notes.append("Primary vibration band detected across: %s" % ", ".join(severe_axes))
    else:
        summary_notes.append("No axis showed a dominant severe vibration signature")

    invalid_tracking_axes = [
        k for k, v in results.items()
        if isinstance(v.get("tracking"), dict) and v["tracking"].get("available") and not v["tracking"].get("valid_for_pid")
    ]
    if invalid_tracking_axes:
        summary_notes.append(
            "Tracking data excluded on: %s due to misalignment or weak correlation"
            % ", ".join(invalid_tracking_axes)
        )

    summary_notes.append("AeroTune aims for the fastest clean response, not the highest noisy motor speed.")

    return to_python_types({
        "sampling_rate_hz": int(round(sample_rate)),
        "overall_score": overall_score,
        "rating": rating,
        "summary_notes": summary_notes,
        "analysis": results,
    })
