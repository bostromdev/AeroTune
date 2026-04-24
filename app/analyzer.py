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
    propwash_score: float
    throttle_activity: float


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
    throttle: Optional[np.ndarray] = None,
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

    # Propwash/bounceback is usually bursty and appears around throttle movement,
    # drops, turns, and dirty-air recovery rather than as one steady wobble.
    propwash_score = 0.0
    throttle_activity = 0.0
    if throttle is not None and len(throttle) == len(gyro):
        th = np.asarray(throttle, dtype=float)
        if np.all(np.isfinite(th)) and np.std(th) > 1e-6:
            dth = np.abs(np.gradient(th))
            throttle_activity = float(np.std(th) + np.mean(dth) * 8.0)
            residual = np.abs(tracking_error - np.median(tracking_error))
            transition_cut = np.quantile(dth, 0.82)
            calm_cut = np.quantile(dth, 0.45)
            transition_mask = dth >= transition_cut
            calm_mask = dth <= calm_cut

            if np.any(transition_mask) and np.any(calm_mask):
                transition_rms = float(np.sqrt(np.mean(np.square(residual[transition_mask]))))
                calm_rms = float(np.sqrt(np.mean(np.square(residual[calm_mask]))))
                propwash_score = transition_rms / max(calm_rms, 1e-6)

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
        propwash_score=float(propwash_score),
        throttle_activity=float(throttle_activity),
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
    if stats.propwash_score > 1.0:
        confidence += 0.10

    # Propwash / bounceback needs to be checked before generic low-frequency wobble.
    # Otherwise dirty-air logs get mislabeled as only "loose / bouncing."
    bursty_dirty_air = (
        stats.propwash_score >= 1.35
        and stats.throttle_activity > 0.02
        and stats.peak_to_peak > 3.0 * max(stats.rms, 1e-6)
    )
    mixed_low_mid_energy = low_ratio > 0.25 and (mid_ratio > 0.18 or high_ratio > 0.12)

    if bursty_dirty_air or (
        stats.propwash_score >= 1.20
        and mixed_low_mid_energy
        and stats.error_ratio > 0.12
    ):
        return "propwash_or_bounceback", min(confidence + 0.25, 1.0)

    if high_ratio > 0.50:
        return "high_frequency_noise", min(confidence + 0.20, 1.0)

    if mid_ratio > 0.45:
        return "mid_frequency_vibration", min(confidence + 0.20, 1.0)

    if low_ratio > 0.55:
        return "low_frequency_oscillation", min(confidence + 0.18, 1.0)

    # Fallback catch for sharp bounceback spikes when throttle data is weak.
    if (
        stats.rms > 0
        and stats.peak_to_peak > 4.5 * stats.rms
        and stats.error_ratio > 0.18
        and stats.corr is not None
        and stats.corr > 0.35
    ):
        return "propwash_or_bounceback", min(confidence + 0.15, 1.0)

    if (
        stats.error_ratio > 0.55
        and (stats.lag_ms is None or abs(stats.lag_ms) < 20.0)
        and (stats.corr is not None and stats.corr > 0.40)
    ):
        return "drift_or_weak_hold", min(confidence + 0.15, 1.0)

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

def _pid_moves_for_issue(axis: str, issue: str, goal: str) -> List[str]:
    """
    Human-facing tuning directions. No percentages here.
    Numeric deltas stay in pid_delta_pct for machine/API use only.
    """
    yaw_note = "Yaw usually uses D = 0, so do not chase yaw propwash with D."

    if issue == "propwash_or_bounceback":
        if axis == "yaw":
            return [
                "D: leave alone / keep at 0 on yaw.",
                "P: lower a little if yaw is visibly shaking on punch-outs or fast forward flight.",
                "I: raise a little only if yaw will not hold heading through throttle changes.",
                "FF: leave alone unless yaw feels delayed from stick input.",
                yaw_note,
            ]
        return [
            "D: raise a little first. This is the main Betaflight-style fix for propwash and bounceback.",
            "P: do not raise P yet. If the bounce is slow or over-correcting, lower P a little instead.",
            "I: leave I alone unless the quad is drifting or losing attitude on throttle changes.",
            "FF: leave FF alone unless stick response itself feels lazy.",
            "After raising D, do a short test flight and check motor temperature right away.",
        ]

    if issue == "low_frequency_oscillation":
        return [
            "P: lower a little. Slow bounce usually means the loop is pushing too hard or swinging past center.",
            "D: raise a little only if it still bounces after P is calmer, but watch motor temperature.",
            "I: leave alone unless the quad drifts or will not hold attitude.",
            "FF: leave alone unless the stick feel is delayed.",
        ]

    if issue == "mid_frequency_vibration":
        return [
            "P: lower a little to calm the roughness.",
            "D: lower a little if motors sound rough or come down warm.",
            "I: leave alone unless attitude hold is weak.",
            "FF: leave alone unless stick response is the actual complaint.",
            "Also check props, motors, frame screws, and stack mounting before blaming only PID.",
        ]

    if issue == "high_frequency_noise":
        return [
            "D: lower or do not raise it. D amplifies gyro noise and can heat motors fast.",
            "P: lower a little if the quad sounds harsh or twitchy.",
            "I: leave alone unless the quad is drifting.",
            "FF: leave alone unless stick response is the issue.",
            "Fix mechanical noise first: props, motors, frame, filtering, and stack mounting.",
        ]

    if issue == "drift_or_weak_hold":
        return [
            "I: raise a little. I is what helps the quad hold attitude against throttle changes, wind, and bias.",
            "P: leave mostly alone unless the quad also feels soft or sloppy.",
            "D: leave alone unless there is bounceback or propwash.",
            "FF: leave alone unless stick response feels delayed.",
        ]

    if issue == "poor_tracking":
        return [
            "P: raise a little if the quad feels soft and does not follow the sticks well.",
            "FF: raise a little if the stick response feels delayed but the quad is otherwise clean.",
            "D: leave alone unless bounceback or propwash appears.",
            "I: leave alone unless it will not hold attitude through throttle changes.",
        ]

    if issue == "slow_response":
        return [
            "FF: raise a little first. FF improves stick response without making P do all the work.",
            "P: raise a little only if it still feels soft after FF.",
            "D: leave alone unless bounceback or propwash is showing up.",
            "I: leave alone unless attitude hold is weak.",
        ]

    return [
        "No major PID move needed from this log.",
        "If it already flies good, do not chase changes just because the tool found small movement.",
        "Make one tiny change at a time only if the feel matches what you see in the notes.",
    ]


def _actions_for_issue(issue: str, goal: str) -> List[str]:
    base: List[str] = []

    if issue == "high_frequency_noise":
        base.extend([
            "High-frequency noise detected. This usually feels buzzy, harsh, or nervous.",
            "Why: the gyro is seeing fast vibration, and D-term can amplify that noise.",
            "Fix: clean the build first, then calm P/D if it still sounds rough.",
        ])
    elif issue == "mid_frequency_vibration":
        base.extend([
            "Mid-frequency vibration detected. This usually feels shaky or rough instead of planted.",
            "Why: prop, motor, frame, or stack vibration is likely getting into the gyro.",
            "Fix: inspect the build and soften P/D slightly if the log stays rough.",
        ])
    elif issue == "low_frequency_oscillation":
        base.extend([
            "Slow wobble detected. This usually feels like bouncing, rocking, or over-correction.",
            "Why: the control loop may be pushing past the target and swinging back.",
            "Fix: reduce P first; only add D carefully if the bounce needs damping.",
        ])
    elif issue == "drift_or_weak_hold":
        base.extend([
            "Weak hold detected. This usually feels like the quad slowly leans, slips, or needs correction during throttle changes.",
            "Why: I-term may not be holding attitude strongly enough.",
            "Fix: raise I a little if the same behavior shows up in flight.",
        ])
    elif issue == "poor_tracking":
        base.extend([
            "Soft tracking detected. This usually feels like the quad is not connected to your sticks.",
            "Why: the gyro is not following setpoint tightly enough.",
            "Fix: add a little P or FF depending on whether it feels soft or delayed.",
        ])
    elif issue == "slow_response":
        base.extend([
            "Slow response detected. This usually feels lazy or late when you ask the quad to move.",
            "Why: the quad follows the command, but not quickly enough.",
            "Fix: raise FF first, then P only if it still feels soft.",
        ])
    elif issue == "propwash_or_bounceback":
        base.extend([
            "Propwash detected. This is dirty-air shake, not just a normal steady wobble.",
            "When you feel it: throttle chops, split-S moves, dives, hard turns, drops, or recovering into your own disturbed air.",
            "Why: the props are hitting turbulent air and the quad is not damping the disturbance fast enough.",
            "Fix: add damping carefully. On roll/pitch that usually means a small D increase or D Max style help, then check motor temperature.",
        ])
    else:
        base.extend([
            "This log looks mostly clean.",
            "Fix: do not chase big changes unless you also feel a problem in the air.",
        ])

    if goal == "efficient":
        base.append("Because you picked Efficient, keep changes small and avoid heat or twitchiness.")
    elif goal == "locked_in":
        base.append("Because you picked Locked-In, prioritize stick connection but stop if motors get hot or the quad gets twitchy.")
    elif goal == "floaty":
        base.append("Because you picked Floaty, keep the tune softer and avoid turning every issue into a sharpness chase.")

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
        return f"{axis_label}: I can see something in the log, but the data is not clean enough to trust tuning advice yet. Retake a cleaner 30–120 second log."

    if issue == "propwash_or_bounceback":
        if axis == "yaw":
            return (
                f"{axis_label}: propwash / bounceback behavior detected, but yaw is different. "
                "Do not solve yaw shake by adding D. Check yaw P/I behavior and only adjust if the quad actually feels loose or shakes on punch-outs."
            )
        return (
            f"{axis_label}: propwash detected. This usually feels like shake after throttle chops, drops, hard turns, or dirty-air recovery. "
            "The Betaflight-style fix is more damping, so try a small D increase or D Max style help on this axis. "
            "Do not raise P first; if the bounce is slow or over-correcting, P may need to come down. Check motor heat after every D change."
        )

    if issue == "low_frequency_oscillation":
        return (
            f"{axis_label}: slow bounce detected. This feels like the quad is rocking or over-correcting. "
            "Try lowering P a little first. If it still bounces after P is calmer, add D carefully for damping."
        )

    if issue == "mid_frequency_vibration":
        return (
            f"{axis_label}: vibration detected. This feels rough or shaky rather than locked in. "
            "Check props, motors, frame, and stack mounting. If the build is clean, soften P/D slightly."
        )

    if issue == "high_frequency_noise":
        return (
            f"{axis_label}: high-frequency noise detected. This can sound harsh and heat motors. "
            "Do not raise D here. Clean mechanical noise first, then reduce D or P if needed."
        )

    if issue == "drift_or_weak_hold":
        return (
            f"{axis_label}: weak hold detected. This feels like the quad slips or drifts through throttle changes. "
            "Try raising I a little because I-term is what helps hold attitude against outside forces."
        )

    if issue == "poor_tracking":
        return (
            f"{axis_label}: soft tracking detected. This feels disconnected from your sticks. "
            "Try a little more P for tracking, or a little more FF if it mainly feels delayed."
        )

    if issue == "slow_response":
        return (
            f"{axis_label}: slow response detected. This feels lazy or late. "
            "Try raising FF first; add P only if it still feels soft after that."
        )

    return f"{axis_label}: mostly clean. Do not make a PID change unless the flight feel gives you a reason."



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
        throttle = df["throttle"].to_numpy(dtype=float) if "throttle" in df.columns else None

        stats = _compute_axis_stats(gyro, setpoint, sample_rate_hz, bands, throttle=throttle)
        issue, confidence = _classify_issue(stats)

        base_delta = _base_delta_for_issue(axis_name, issue)
        pid_delta = _apply_goal_bias(base_delta, axis_name, issue, tuning_goal)

        valid_for_pid = bool(confidence >= 0.50 and stats.total_energy > 0)
        if valid_for_pid:
            valid_axis_count += 1

        actions = _actions_for_issue(issue, tuning_goal)
        tuning_moves = _pid_moves_for_issue(axis_name, issue, tuning_goal)
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
                "propwash_score": round(stats.propwash_score, 4),
                "throttle_activity": round(stats.throttle_activity, 4),
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
            "tuning_moves": tuning_moves,
            "warnings": _warnings_for_goal(tuning_goal),
        }

        recommendations.append({
            "axis": axis_name,
            "issue": issue,
            "feel": feel,
            "valid_for_pid": valid_for_pid,
            "confidence": round(confidence, 3),
            "pid_delta_pct": axes[axis_name]["pid_delta_pct"],  # machine/API only; hidden from pilot UI
            "tuning_moves": tuning_moves,
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