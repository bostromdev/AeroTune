"""
AeroTune V1.8 Pilot-Feel Engine.

Code label: HUMAN-CONTEXT LAYER
--------------------------------
This module is the bridge between two things AeroTune needs to make a real plan:

1. Blackbox log evidence: gyro, setpoint, throttle, frequency bands, tracking error.
2. Pilot-reported flight feel: motors hot/cool, loose feel, propwash, bounceback,
   low-throttle drift, RPM-filter trouble, dynamic-notch status, and hardware changes.

Why this code lives here:
- app/main.py should only handle upload/API plumbing.
- app/analyzer.py should stay focused on signal analysis.
- app/tuning_advisor.py should stay focused on PID delta translation.
- This file owns pilot checkbox normalization, filtering policy, D-term safety gates,
  and the final human-readable plan that combines log data with what the pilot felt.

Important tuning philosophy:
AeroTune does not chase perfect graphs. If the quad flew well, motors stayed cool,
and the log does not show a dangerous pattern, the safest advice is often to hold the
current tune and save the log as a baseline.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# Code label: PILOT INPUT CATALOG
# These IDs are what the frontend sends as repeated form fields named `pilot_feel`.
# Keep IDs stable because saved reports and future comparisons may reference them.
PILOT_FEEL_OPTIONS: Tuple[Dict[str, str], ...] = (
    # Baseline / thermal state
    {"id": "felt_good", "label": "Felt good / no major issue", "category": "Baseline"},
    {"id": "good_baseline_hold", "label": "Use this as a good baseline / hold current tune", "category": "Baseline"},
    {"id": "motors_cool", "label": "Motors stayed cool", "category": "Motor temperature"},
    {"id": "motors_warm", "label": "Motors were warm", "category": "Motor temperature"},
    {"id": "motors_hot", "label": "Motors were hot", "category": "Motor temperature"},

    # Pilot feel / handling
    {"id": "felt_loose_floaty", "label": "Felt loose / floaty", "category": "Flight feel"},
    {"id": "felt_locked_in", "label": "Felt locked-in", "category": "Flight feel"},
    {"id": "twitchy_too_sharp", "label": "Felt twitchy / too sharp", "category": "Flight feel"},
    {"id": "slow_throttle_response", "label": "Slow throttle response", "category": "Throttle feel"},
    {"id": "center_stick_too_sensitive", "label": "Too sensitive around center stick", "category": "Stick feel"},
    {"id": "not_enough_center_authority", "label": "Not enough center-stick authority", "category": "Stick feel"},
    {"id": "roll_weak", "label": "Roll feels weak", "category": "Axis feel"},
    {"id": "pitch_weak", "label": "Pitch feels weak", "category": "Axis feel"},
    {"id": "yaw_weak", "label": "Yaw feels weak", "category": "Axis feel"},
    {"id": "tracking_delayed", "label": "Roll/pitch tracking feels delayed", "category": "Tracking"},
    {"id": "recovery_slow", "label": "Recovery feels slow after maneuvers", "category": "Tracking"},

    # Maneuver symptoms
    {"id": "bounceback", "label": "Bounceback after flips or rolls", "category": "Maneuver symptoms"},
    {"id": "propwash", "label": "Propwash after hard turns or drops", "category": "Maneuver symptoms"},
    {"id": "low_throttle_wobbles", "label": "Wobbles during low throttle", "category": "Throttle symptoms"},
    {"id": "high_throttle_wobbles", "label": "Wobbles during high throttle", "category": "Throttle symptoms"},
    {"id": "throttle_punch_oscillation", "label": "Oscillation during throttle punches", "category": "Throttle symptoms"},
    {"id": "fast_forward_shake", "label": "Shaking during fast forward flight", "category": "Vibration / noise"},
    {"id": "low_throttle_drift_or_fall", "label": "Drifts or falls at low throttle", "category": "Low throttle"},
    {"id": "low_throttle_climb", "label": "Climbs too easily at low throttle", "category": "Low throttle"},
    {"id": "underpowered", "label": "Feels underpowered", "category": "Power feel"},
    {"id": "overpowered", "label": "Feels overpowered / hard to manage", "category": "Power feel"},

    # Noise / mechanical flags
    {"id": "motors_rough_buzzy", "label": "Motors sounded rough or buzzy", "category": "Vibration / noise"},
    {"id": "one_motor_different", "label": "One motor sounded different", "category": "Vibration / noise"},
    {"id": "video_jello", "label": "Video had vibration / jello", "category": "Vibration / noise"},

    # Config and event flags
    {"id": "rx_failsafe_or_arming_warnings", "label": "Had RX failsafe or arming warnings", "category": "Config / safety"},
    {"id": "rpm_filter_caused_issues", "label": "RPM filter caused issues", "category": "Filtering"},
    {"id": "dynamic_notch_enabled", "label": "Dynamic notch was enabled", "category": "Filtering"},
    {"id": "dynamic_notch_disabled", "label": "Dynamic notch was disabled", "category": "Filtering"},
    {"id": "rpm_filter_enabled", "label": "RPM filter was enabled", "category": "Filtering"},
    {"id": "pids_changed_before_flight", "label": "I changed PIDs before this flight", "category": "Changed before flight"},
    {"id": "filters_changed_before_flight", "label": "I changed filters before this flight", "category": "Changed before flight"},
    {"id": "props_changed_before_flight", "label": "I changed props before this flight", "category": "Changed before flight"},
    {"id": "crash_or_hit_something", "label": "I crashed or hit something during this flight", "category": "Changed before flight"},
)

OPTION_BY_ID: Dict[str, Dict[str, str]] = {item["id"]: item for item in PILOT_FEEL_OPTIONS}

THERMAL_HOT_IDS = {"motors_hot", "motors_rough_buzzy"}
THERMAL_SAFE_IDS = {"motors_cool"}
GOOD_BASELINE_IDS = {"felt_good", "good_baseline_hold", "motors_cool", "felt_locked_in"}
LOW_THROTTLE_IDS = {"low_throttle_drift_or_fall", "low_throttle_climb", "low_throttle_wobbles"}
MECHANICAL_RISK_IDS = {"motors_rough_buzzy", "one_motor_different", "video_jello", "crash_or_hit_something"}
THROTTLE_LINKED_IDS = {"high_throttle_wobbles", "throttle_punch_oscillation", "fast_forward_shake", "propwash"}
RPM_RISK_IDS = {"rpm_filter_caused_issues", "rx_failsafe_or_arming_warnings"}
DYNAMIC_NOTCH_IDS = {"dynamic_notch_enabled", "dynamic_notch_disabled"}

# Code label: BASELINE-HOLD CONFLICT MAP
# These options mean "the flight had a real tuning symptom." If any are selected,
# they override the good-baseline hotfix so AeroTune does not hide useful advice.
# Low-throttle altitude feel is intentionally not in this set because it usually
# needs throttle/idle/hover-point guidance, not PID deltas.
ACTIVE_TUNING_SYMPTOM_IDS = {
    "felt_loose_floaty",
    "twitchy_too_sharp",
    "slow_throttle_response",
    "center_stick_too_sensitive",
    "not_enough_center_authority",
    "roll_weak",
    "pitch_weak",
    "yaw_weak",
    "tracking_delayed",
    "recovery_slow",
    "bounceback",
    "propwash",
    "high_throttle_wobbles",
    "throttle_punch_oscillation",
    "fast_forward_shake",
    "underpowered",
    "overpowered",
}

# Code label: PILOT-FEEL OUTPUT MAP
# Every checkbox should create a visible interpretation in the API response. These
# outputs are intentionally not additive PID deltas. They are gates, context, or
# next-step guidance so selecting ten symptoms cannot stack into a giant PID jump.
MAX_ANALYSIS_PID_DELTA_FRACTION = 0.11
MAX_ADVICE_PID_DELTA_PERCENT = 11
PILOT_FEEL_OUTPUTS: Dict[str, Dict[str, str]] = {
    "felt_good": {"effect": "baseline confidence", "action": "Protect the tune unless repeated log evidence confirms a problem."},
    "good_baseline_hold": {"effect": "baseline hold", "action": "Hold PID deltas at 0% unless active symptoms were also selected."},
    "motors_cool": {"effect": "thermal safe", "action": "Allows normal conservative review, but still caps D/D Max per pass."},
    "motors_warm": {"effect": "thermal caution", "action": "Avoid stacking D changes; retest and touch motors after any change."},
    "motors_hot": {"effect": "thermal block", "action": "Block positive D/D Max and inspect filtering/mechanics first."},
    "felt_loose_floaty": {"effect": "handling symptom", "action": "Bypass baseline hold and review P/FF/D support using log evidence."},
    "felt_locked_in": {"effect": "positive feel", "action": "Treat as a good sign; hold if cool unless another symptom conflicts."},
    "twitchy_too_sharp": {"effect": "over-response symptom", "action": "Favor smaller P/FF or smoother rates before adding more authority."},
    "slow_throttle_response": {"effect": "throttle feel symptom", "action": "Review throttle curve/idle/weight before PID changes."},
    "center_stick_too_sensitive": {"effect": "stick feel symptom", "action": "Review rates/expo first; do not treat this as automatic PID change."},
    "not_enough_center_authority": {"effect": "stick authority symptom", "action": "Review rates/expo and tracking evidence before small P/FF changes."},
    "roll_weak": {"effect": "axis authority symptom", "action": "Use roll evidence to decide whether P/FF support is needed."},
    "pitch_weak": {"effect": "axis authority symptom", "action": "Use pitch evidence to decide whether P/FF support is needed."},
    "yaw_weak": {"effect": "axis authority symptom", "action": "Review yaw separately; do not copy roll/pitch D logic into yaw."},
    "tracking_delayed": {"effect": "tracking symptom", "action": "Review gyro/setpoint lag and consider tiny FF/P moves only if log agrees."},
    "recovery_slow": {"effect": "recovery symptom", "action": "Review bounceback/propwash/lag before changing damping."},
    "bounceback": {"effect": "maneuver symptom", "action": "Bypass baseline hold and review D/D Max/P balance, capped per pass."},
    "propwash": {"effect": "dirty-air symptom", "action": "Review propwash band and D-term safety before any damping change."},
    "low_throttle_wobbles": {"effect": "low-throttle symptom", "action": "Check idle, filtering, mechanics, and throttle behavior before PID chasing."},
    "high_throttle_wobbles": {"effect": "throttle-linked symptom", "action": "Review Dynamic Notch/mechanics before adding D."},
    "throttle_punch_oscillation": {"effect": "throttle-linked symptom", "action": "Inspect motor/prop/frame resonance and Dynamic Notch settings."},
    "fast_forward_shake": {"effect": "vibration symptom", "action": "Review frequency bands, frame stiffness, GoPro mount, and Dynamic Notch."},
    "low_throttle_drift_or_fall": {"effect": "hover/altitude feel", "action": "Tune throttle curve/hover point/idle first, not PID."},
    "low_throttle_climb": {"effect": "hover/altitude feel", "action": "Tune throttle curve/hover point/idle first, not PID."},
    "underpowered": {"effect": "powertrain feel", "action": "Check battery, props, weight, throttle scaling, and motor output before PID."},
    "overpowered": {"effect": "powertrain feel", "action": "Review throttle expo/curve and rates before PID."},
    "motors_rough_buzzy": {"effect": "mechanical/noise block", "action": "Block positive D/D Max and inspect props/motors/frame/filtering."},
    "one_motor_different": {"effect": "mechanical risk", "action": "Inspect that motor/prop/bearing/screws before trusting PID data."},
    "video_jello": {"effect": "camera vibration", "action": "Inspect camera mount/props/frame before filter/PID changes."},
    "rx_failsafe_or_arming_warnings": {"effect": "safety block", "action": "Suppress RPM-filter recommendations and fix reliability before tuning."},
    "rpm_filter_caused_issues": {"effect": "RPM filter suppression", "action": "Use Dynamic Notch path; do not recommend RPM filter by default."},
    "dynamic_notch_enabled": {"effect": "filter status", "action": "Keep Dynamic Notch as the default filter path."},
    "dynamic_notch_disabled": {"effect": "filter status", "action": "Review enabling Dynamic Notch before aggressive PID/filter changes."},
    "rpm_filter_enabled": {"effect": "advanced filter status", "action": "Treat RPM filtering as advanced-only and verify telemetry stability."},
    "pids_changed_before_flight": {"effect": "test context", "action": "Use tune tracking before making another PID pass."},
    "filters_changed_before_flight": {"effect": "test context", "action": "Separate filter changes from PID changes when comparing logs."},
    "props_changed_before_flight": {"effect": "test context", "action": "Prop change reduces comparison certainty; retest consistently."},
    "crash_or_hit_something": {"effect": "mechanical risk", "action": "Inspect hardware before trusting the log or PID advice."},
}


def get_pilot_feel_options() -> List[Dict[str, str]]:
    """Return frontend-safe option metadata."""
    return [dict(item) for item in PILOT_FEEL_OPTIONS]


def _flatten_values(values: Any) -> List[str]:
    """Accept FastAPI Form lists, comma strings, JSON-like lists, or None."""
    if values is None:
        return []

    if isinstance(values, str):
        raw = values.strip()
        if not raw:
            return []
        # FastAPI checkboxes arrive as repeated fields, but this keeps the API
        # usable for CLI/curl callers that pass comma-separated text.
        return [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]

    if isinstance(values, Iterable):
        out: List[str] = []
        for value in values:
            out.extend(_flatten_values(value))
        return out

    return [str(values).strip()]


def build_pilot_feel_outputs(selected: Iterable[str]) -> List[Dict[str, str]]:
    """
    Code label: PILOT-FEEL OUTPUT BUILDER.

    Every selected pilot-feel checkbox creates one visible output. These are not
    direct PID adds. They explain how the selected feel changes gates, caution,
    or next-step logic.
    """
    outputs: List[Dict[str, str]] = []
    for key in selected:
        item = OPTION_BY_ID.get(key)
        if not item:
            continue
        rule = PILOT_FEEL_OUTPUTS.get(key, {
            "effect": "context",
            "action": "Record as pilot context; do not create a direct PID delta from this checkbox alone.",
        })
        outputs.append({
            "id": key,
            "label": item["label"],
            "category": item["category"],
            "effect": rule["effect"],
            "action": rule["action"],
        })
    return outputs


def normalize_pilot_feel(values: Any) -> Dict[str, Any]:
    """
    Code label: PILOT-FEEL NORMALIZER

    Converts raw checkbox values into stable flags the analyzer can use.
    Unknown values are ignored instead of crashing the upload path.
    """
    selected: List[str] = []
    unknown: List[str] = []

    for raw in _flatten_values(values):
        key = str(raw).strip().lower().replace(" ", "_").replace("-", "_")
        if not key:
            continue
        if key in OPTION_BY_ID and key not in selected:
            selected.append(key)
        elif key not in unknown:
            unknown.append(key)

    selected_labels = [OPTION_BY_ID[key]["label"] for key in selected]
    option_outputs = build_pilot_feel_outputs(selected)

    categories: Dict[str, List[str]] = defaultdict(list)
    for key in selected:
        item = OPTION_BY_ID[key]
        categories[item["category"]].append(item["label"])

    flags = {
        "has_pilot_input": bool(selected),
        "flight_felt_good": "felt_good" in selected,
        "baseline_hold_requested": "good_baseline_hold" in selected,
        "felt_locked_in": "felt_locked_in" in selected,
        "motors_cool": "motors_cool" in selected,
        "motors_warm": "motors_warm" in selected,
        "motors_hot": "motors_hot" in selected,
        "motors_rough_or_buzzy": "motors_rough_buzzy" in selected,
        "rpm_filter_issue": bool(RPM_RISK_IDS.intersection(selected)),
        "rpm_filter_enabled": "rpm_filter_enabled" in selected,
        "dynamic_notch_enabled": "dynamic_notch_enabled" in selected,
        "dynamic_notch_disabled": "dynamic_notch_disabled" in selected,
        "low_throttle_issue": bool(LOW_THROTTLE_IDS.intersection(selected)),
        "mechanical_risk": bool(MECHANICAL_RISK_IDS.intersection(selected)),
        "active_tuning_symptom": bool(ACTIVE_TUNING_SYMPTOM_IDS.intersection(selected)),
        "active_tuning_symptom_ids": sorted(ACTIVE_TUNING_SYMPTOM_IDS.intersection(selected)),
        "throttle_linked_symptom": bool(THROTTLE_LINKED_IDS.intersection(selected)),
        "changed_before_flight": bool({
            "pids_changed_before_flight",
            "filters_changed_before_flight",
            "props_changed_before_flight",
        }.intersection(selected)),
        "crash_or_hit_something": "crash_or_hit_something" in selected,
        "bounceback_felt": "bounceback" in selected,
        "propwash_felt": "propwash" in selected,
        "loose_or_floaty": "felt_loose_floaty" in selected,
        "twitchy_or_too_sharp": "twitchy_too_sharp" in selected,
        "tracking_delayed": "tracking_delayed" in selected or "recovery_slow" in selected,
    }

    summary = "No pilot feel options selected. AeroTune is using log-only analysis."
    if selected_labels:
        summary = "Pilot reported: " + "; ".join(selected_labels[:6])
        if len(selected_labels) > 6:
            summary += f"; +{len(selected_labels) - 6} more"

    warnings: List[str] = []
    if flags["motors_hot"]:
        warnings.append("Pilot reported hot motors; positive D/D Max changes are blocked for this report.")
    if flags["rpm_filter_issue"]:
        warnings.append("Pilot reported RPM-filter or arming/failsafe trouble; RPM filtering is suppressed from default advice.")
    if flags["mechanical_risk"]:
        warnings.append("Pilot reported a mechanical/noise flag; inspect props, motor bells, screws, frame, and stack mounting before PID chasing.")
    if flags["low_throttle_issue"]:
        warnings.append("Pilot reported low-throttle float/climb/fall behavior; review throttle curve, idle, hover point, and craft weight before major PID changes.")
    if flags.get("baseline_hold_requested") and not flags.get("active_tuning_symptom"):
        warnings.append("Pilot marked this flight as a good baseline; PID deltas are downgraded unless a second comparable log confirms the same problem.")
    elif flags.get("baseline_hold_requested") and flags.get("active_tuning_symptom"):
        symptom_labels = [OPTION_BY_ID[key]["label"] for key in flags.get("active_tuning_symptom_ids", []) if key in OPTION_BY_ID]
        detail = "; ".join(symptom_labels[:4])
        warnings.append(
            "Good-baseline hold was selected, but active tuning symptoms were also selected. "
            f"AeroTune will not hide PID advice from this report. Symptoms: {detail}."
        )

    return {
        "version": "AeroTune pilot feel context v1.0",
        "selected": selected,
        "selected_labels": selected_labels,
        "option_outputs": option_outputs,
        "categories": dict(categories),
        "flags": flags,
        "summary": summary,
        "warnings": warnings,
        "unknown_values_ignored": unknown,
    }


def _axis_items(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    axes = analysis.get("axes") if isinstance(analysis, dict) else None
    if isinstance(axes, dict):
        return [dict({"axis": key}, **(value or {})) for key, value in axes.items()]
    if isinstance(axes, list):
        return [item for item in axes if isinstance(item, dict)]
    return []


def _band_for_frequency(freq: Optional[float], bands: Dict[str, Sequence[float]]) -> Optional[str]:
    if freq is None:
        return None
    try:
        f = float(freq)
    except (TypeError, ValueError):
        return None
    for name, bounds in bands.items():
        if not isinstance(bounds, Sequence) or len(bounds) < 2:
            continue
        lo, hi = float(bounds[0]), float(bounds[1])
        if lo <= f < hi:
            return str(name)
    return None


def _signal_float(axis: Dict[str, Any], key: str, default: float = 0.0) -> float:
    signal = axis.get("signal") or {}
    try:
        return float(signal.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _analysis_has_serious_issue(analysis: Dict[str, Any]) -> bool:
    serious_issues = {
        "high_frequency_noise",
        "mid_frequency_vibration",
        "high_throttle_oscillation",
        "low_frequency_oscillation",
        "mechanical_noise_suspected",
        "propwash",
        "bounceback",
    }
    for axis in _axis_items(analysis):
        if axis.get("issue") in serious_issues and float(axis.get("confidence", 0.0) or 0.0) >= 0.62:
            return True
    return False


def _build_frequency_evidence(analysis: Dict[str, Any]) -> Dict[str, Any]:
    axes = _axis_items(analysis)
    bands = analysis.get("frequency_bands_hz") or {}
    band_hits: Counter[str] = Counter()
    axis_reads: List[Dict[str, Any]] = []

    for axis in axes:
        signal = axis.get("signal") or {}
        freq = signal.get("dominant_freq_hz")
        band = _band_for_frequency(freq, bands)
        if band:
            band_hits[band] += 1
        axis_reads.append({
            "axis": axis.get("axis") or axis.get("axis_name"),
            "issue": axis.get("issue"),
            "dominant_freq_hz": freq,
            "band": band,
            "high_ratio": signal.get("high_ratio"),
            "ultra_ratio": signal.get("ultra_ratio"),
            "throttle_error_ratio": signal.get("throttle_error_ratio"),
            "high_throttle_error_ratio": signal.get("high_throttle_error_ratio"),
        })

    repeated = [name for name, count in band_hits.items() if count >= 2]
    return {
        "axis_reads": axis_reads,
        "band_counts": dict(band_hits),
        "repeated_bands": repeated,
        "repeated_frequency_evidence": bool(repeated),
    }


def build_filtering_advisor(
    analysis: Optional[Dict[str, Any]],
    pilot_context: Dict[str, Any],
    drone_size: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Code label: FILTERING + D-TERM SAFETY ENGINE

    Uses both log evidence and pilot feel to decide whether AeroTune should:
    - protect the tune,
    - recommend Dynamic Notch as the default filter path,
    - suppress RPM-filter advice,
    - block positive D/D Max changes,
    - or suggest a mechanical/throttle setup check before PID tuning.
    """
    analysis = analysis if isinstance(analysis, dict) else {}
    flags = pilot_context.get("flags", {}) if isinstance(pilot_context, dict) else {}
    size = str(drone_size or analysis.get("drone_size") or analysis.get("drone_size_profile") or "unknown")

    frequency = _build_frequency_evidence(analysis)
    axes = _axis_items(analysis)
    high_noise_axes = [
        axis for axis in axes
        if axis.get("issue") in {"high_frequency_noise", "mid_frequency_vibration", "high_throttle_oscillation"}
        or (_signal_float(axis, "high_ratio") + _signal_float(axis, "ultra_ratio")) >= 0.55
    ]
    throttle_linked_axes = [
        axis for axis in axes
        if _signal_float(axis, "throttle_error_ratio", 1.0) >= 1.22
        or _signal_float(axis, "high_throttle_error_ratio", 1.0) >= 1.28
    ]

    rpm_suppressed = bool(flags.get("rpm_filter_issue")) or bool(flags.get("dynamic_notch_disabled") and flags.get("rpm_filter_enabled"))
    d_blocked = bool(flags.get("motors_hot") or flags.get("motors_rough_or_buzzy") or flags.get("rpm_filter_issue"))
    large_quad = size in {"6", "7", "8", "9", "10"}

    # Code label: GOOD BASELINE HOLD GATE
    # A hard freestyle move can look like overshoot/bounceback in a single log.
    # If the pilot explicitly reports a good/cool flight or marks the log as a
    # baseline, AeroTune must not tell them there is a "high chance" fix from
    # one graph. But this gate is NOT allowed to hide a real symptom selected by
    # the pilot. Example: "Felt good + motors cool" can hold PID deltas; "Felt
    # loose/floaty" or "bounceback" overrides that hold and lets the advisor
    # produce a targeted plan.
    serious_log_issue = _analysis_has_serious_issue(analysis)
    active_tuning_symptom = bool(flags.get("active_tuning_symptom"))
    safe_baseline_reported = bool(
        flags.get("baseline_hold_requested")
        or (flags.get("flight_felt_good") and flags.get("motors_cool"))
        or (flags.get("felt_locked_in") and flags.get("motors_cool"))
    )
    unsafe_baseline_override = bool(
        flags.get("motors_hot")
        or flags.get("motors_rough_or_buzzy")
        or flags.get("mechanical_risk")
        or flags.get("crash_or_hit_something")
    )
    baseline_conflict_with_symptoms = bool(safe_baseline_reported and active_tuning_symptom)
    protect_good_tune = bool(
        safe_baseline_reported
        and not unsafe_baseline_override
        and not baseline_conflict_with_symptoms
    )
    log_pilot_disagreement = bool(protect_good_tune and serious_log_issue)

    dynamic_notch_state = "enabled" if flags.get("dynamic_notch_enabled") else "disabled" if flags.get("dynamic_notch_disabled") else "unknown"

    warnings: List[str] = []
    next_actions: List[str] = []
    plan: List[Dict[str, str]] = []

    plan.append({
        "title": "Default filtering path",
        "why": "Dynamic Notch is gyro-based and does not require stable motor RPM telemetry.",
        "action": "Use Dynamic Notch as AeroTune's first-line filtering recommendation. Treat RPM filtering as advanced-only.",
    })

    if baseline_conflict_with_symptoms:
        symptom_labels = [
            OPTION_BY_ID[key]["label"]
            for key in flags.get("active_tuning_symptom_ids", [])
            if key in OPTION_BY_ID
        ]
        warnings.append(
            "Good-baseline hold was bypassed because pilot-selected symptoms need attention: "
            + ("; ".join(symptom_labels[:5]) if symptom_labels else "active tuning symptom selected")
            + "."
        )
        plan.append({
            "title": "Baseline hold bypassed",
            "why": "A baseline checkbox cannot override pilot-reported symptoms like loose/floaty feel, bounceback, propwash, weak tracking, or twitchiness.",
            "action": "Use the selected symptoms plus the log evidence to build a targeted plan instead of forcing all PID deltas to zero.",
        })

    if log_pilot_disagreement:
        warnings.append(
            "Log metrics and pilot feel disagree: the log shows correction evidence, but the pilot marked the flight as good/cool. AeroTune is holding PID changes until the issue repeats in another similar log."
        )

    if rpm_suppressed:
        warnings.append("RPM filtering suppressed: pilot reported RPM-filter, arming, or failsafe reliability trouble.")
        plan.append({
            "title": "RPM filter policy",
            "why": "RPM filtering depends on bidirectional DShot, ESC firmware behavior, motor pole count, and clean RPM telemetry.",
            "action": "Do not recommend enabling RPM filter from this report. Use Dynamic Notch plus mechanical inspection first.",
        })
    else:
        plan.append({
            "title": "RPM filter policy",
            "why": "RPM filtering can be strong, but it is not the safe default for unknown builds.",
            "action": "Only discuss RPM filtering if the pilot explicitly uses advanced mode and confirms bidirectional DShot/RPM telemetry is stable.",
        })

    if dynamic_notch_state == "disabled":
        warnings.append("Dynamic Notch was reported disabled; repeated narrow-band noise should be handled there before broad PID/filter changes.")
        next_actions.append("Enable or review Dynamic Notch before chasing aggressive PID changes if repeated narrow-band vibration remains.")
    elif dynamic_notch_state == "enabled":
        next_actions.append("Keep Dynamic Notch enabled. Only tighten filtering if repeated noise appears across another comparable log.")
    else:
        next_actions.append("Confirm whether Dynamic Notch is enabled in Betaflight before changing filter strategy.")

    if frequency["repeated_frequency_evidence"] or high_noise_axes:
        plan.append({
            "title": "Frequency-band review",
            "why": "Repeated narrow-band energy is more likely to be real motor/prop/frame vibration than one random stick event.",
            "action": "Review Dynamic Notch and mechanical sources first. Avoid broad filtering changes unless the same band repeats in another log.",
        })
        next_actions.append("Save this frequency read and compare it against the next log before making another filtering change.")

    if throttle_linked_axes or flags.get("throttle_linked_symptom"):
        plan.append({
            "title": "Throttle-linked vibration",
            "why": "Noise that rises with throttle usually points to motor/prop/frame vibration, dirty-air load, or high-throttle oscillation.",
            "action": "Inspect props/motors/frame and use Dynamic Notch guidance before adding D or reducing filters.",
        })

    if flags.get("low_throttle_issue"):
        plan.append({
            "title": "Low-throttle feel",
            "why": "Slow climb/fall near hover is often throttle curve, idle, weight, rates/expo feel, or hover-point setup before it is a PID problem.",
            "action": "Adjust throttle feel/hover-point behavior separately. Do not use D-term changes to fix low-throttle altitude control.",
        })
        next_actions.append("For low-throttle climb/fall, tune throttle curve/hover point first, then retest with the same hover/ramp section.")

    if flags.get("mechanical_risk"):
        warnings.append("Mechanical/noise flag selected; do a hardware inspection before trusting PID deltas.")
        next_actions.append("Check props, motor bells, motor screws, arm screws, frame cracks, GoPro mount, and FC stack mounting.")

    if d_blocked:
        warnings.append("Positive D/D Max increases are blocked by pilot feel safety gates for this report.")
        plan.append({
            "title": "D-term safety gate",
            "why": "Hot motors, buzzy motors, or RPM-filter trouble means more D can amplify noise and heat faster.",
            "action": "Do not increase D or D Max until motors are cool and the log is cleaner.",
        })
    elif large_quad:
        plan.append({
            "title": "Large-prop D-term cap",
            "why": "6–7 inch props have more inertia and can heat motors faster when D is pushed aggressively.",
            "action": "Keep D/D Max changes small. Retest and check motor temperature after every pass.",
        })

    if protect_good_tune:
        plan.insert(0, {
            "title": "Good-flight / baseline hold",
            "why": "Pilot reported a good/cool flight or explicitly marked this log as a baseline. Single-log overshoot/bounceback metrics can be false positives after hard Acro moves.",
            "action": "Hold P/I/D/D Max/FF from this report. Save this log as the baseline and only tune if the same symptom repeats in another comparable flight.",
        })
        next_actions.insert(0, "Save this as a good baseline log. Do not apply PID changes from this one good/cool flight.")

    if not next_actions:
        next_actions.append("Make one small change at a time, fly the same 60–90 second test route, then compare before/after logs.")

    summary = "Dynamic Notch is the default filtering path. RPM filtering is advanced-only."
    if protect_good_tune:
        summary = "Good-flight/baseline hold active: PID deltas are suppressed from this one log. Hold the tune unless the same issue repeats. Dynamic Notch remains the default filter path."
    elif d_blocked:
        summary = "D-term safety gate active: do not raise D/D Max from this report. Use Dynamic Notch/mechanical checks first."
    elif frequency["repeated_frequency_evidence"] or high_noise_axes:
        summary = "Repeated/noisy frequency behavior detected. Review Dynamic Notch and mechanical sources before PID aggression."

    return {
        "version": "AeroTune filtering advisor v1.0",
        "summary": summary,
        "default_filter_path": "dynamic_notch",
        "dynamic_notch_status_from_pilot": dynamic_notch_state,
        "rpm_filter_policy": "suppressed" if rpm_suppressed else "advanced_only",
        "rpm_filter_suppressed": bool(rpm_suppressed),
        "positive_d_allowed": not d_blocked,
        "positive_d_blocked": bool(d_blocked),
        "large_prop_conservative_d_cap": bool(large_quad),
        "good_flight_protection": bool(protect_good_tune),
        "baseline_hold_requested": bool(flags.get("baseline_hold_requested")),
        "baseline_hold_bypassed_by_symptoms": bool(baseline_conflict_with_symptoms),
        "active_tuning_symptom": bool(active_tuning_symptom),
        "active_tuning_symptom_ids": list(flags.get("active_tuning_symptom_ids", [])),
        "log_pilot_disagreement": bool(log_pilot_disagreement),
        "throttle_linked_noise": bool(throttle_linked_axes or flags.get("throttle_linked_symptom")),
        "high_noise_axis_count": len(high_noise_axes),
        "frequency_evidence": frequency,
        "warnings": warnings,
        "next_actions": next_actions,
        "plan": plan,
    }


def _append_unique(target: List[str], values: Iterable[str]) -> None:
    for value in values:
        text = str(value).strip()
        if text and text not in target:
            target.append(text)


def _safe_float_for_cap(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number == number else default


def _cap_analysis_delta_map(deltas: Dict[str, Any], cap: float = MAX_ANALYSIS_PID_DELTA_FRACTION) -> Dict[str, float]:
    clean: Dict[str, float] = {}
    for key, value in deltas.items():
        clean[key] = round(float(max(-cap, min(cap, _safe_float_for_cap(value)))), 4)
    return clean


def _cap_analysis_pid_deltas(analysis: Dict[str, Any]) -> None:
    """Clamp analyzer fraction deltas so future context cannot stack huge moves."""
    axes = analysis.get("axes")
    iterable = axes.values() if isinstance(axes, dict) else axes if isinstance(axes, list) else []
    for axis in iterable:
        if isinstance(axis, dict) and isinstance(axis.get("pid_delta_pct"), dict):
            before = dict(axis["pid_delta_pct"])
            after = _cap_analysis_delta_map(before)
            if after != before:
                axis["pre_cap_pid_delta_pct"] = before
                axis["pid_delta_pct"] = after
                axis["pid_delta_cap_note"] = f"AeroTune capped PID deltas to ±{int(MAX_ANALYSIS_PID_DELTA_FRACTION * 100)}% for one safe pass."
    recs = analysis.get("recommendations")
    if isinstance(recs, list):
        for rec in recs:
            if isinstance(rec, dict) and isinstance(rec.get("pid_delta_pct"), dict):
                before = dict(rec["pid_delta_pct"])
                after = _cap_analysis_delta_map(before)
                if after != before:
                    rec["pre_cap_pid_delta_pct"] = before
                    rec["pid_delta_pct"] = after


def _cap_advice_pid_deltas(advice: Dict[str, Any]) -> None:
    """Clamp Betaflight percent deltas after pilot-feel gates run."""
    axes = advice.get("axes")
    if not isinstance(axes, dict):
        return
    for axis in axes.values():
        if not isinstance(axis, dict) or not isinstance(axis.get("deltas"), dict):
            continue
        deltas = axis["deltas"]
        capped: Dict[str, int] = {}
        for key, value in deltas.items():
            capped[key] = int(round(max(-MAX_ADVICE_PID_DELTA_PERCENT, min(MAX_ADVICE_PID_DELTA_PERCENT, _safe_float_for_cap(value)))))
        if capped != deltas:
            axis["pre_cap_deltas"] = dict(deltas)
            axis["deltas"] = capped
            notes = axis.setdefault("notes", [])
            if isinstance(notes, list):
                _append_unique(notes, [f"AeroTune one-pass safety cap limited PID deltas to ±{MAX_ADVICE_PID_DELTA_PERCENT}%."])


def _block_positive_deltas_in_analysis(analysis: Dict[str, Any], reason: str) -> None:
    """Set positive D deltas to zero when pilot safety gates say D is unsafe."""
    axes = analysis.get("axes")
    if isinstance(axes, dict):
        iterable = axes.values()
    elif isinstance(axes, list):
        iterable = axes
    else:
        iterable = []

    for axis in iterable:
        if not isinstance(axis, dict):
            continue
        deltas = axis.get("pid_delta_pct")
        if isinstance(deltas, dict) and any(float(deltas.get(key, 0.0) or 0.0) > 0 for key in ("d", "d_max")):
            for key in ("d", "d_max"):
                if float(deltas.get(key, 0.0) or 0.0) > 0:
                    deltas[key] = 0.0
            axis["pilot_feel_d_gate"] = reason
            moves = axis.setdefault("tuning_moves", [])
            if isinstance(moves, list):
                _append_unique(moves, [reason])

    recs = analysis.get("recommendations")
    if isinstance(recs, list):
        for rec in recs:
            if not isinstance(rec, dict):
                continue
            deltas = rec.get("pid_delta_pct")
            if isinstance(deltas, dict) and any(float(deltas.get(key, 0.0) or 0.0) > 0 for key in ("d", "d_max")):
                for key in ("d", "d_max"):
                    if float(deltas.get(key, 0.0) or 0.0) > 0:
                        deltas[key] = 0.0
                moves = rec.setdefault("tuning_moves", [])
                if isinstance(moves, list):
                    _append_unique(moves, [reason])


def _hold_pid_deltas_in_analysis(analysis: Dict[str, Any], reason: str) -> None:
    """Zero PID deltas when the pilot explicitly marks a good/cool baseline."""
    zero = {"p": 0.0, "i": 0.0, "d": 0.0, "d_max": 0.0, "ff": 0.0}

    axes = analysis.get("axes")
    if isinstance(axes, dict):
        iterable = axes.values()
    elif isinstance(axes, list):
        iterable = axes
    else:
        iterable = []

    for axis in iterable:
        if not isinstance(axis, dict):
            continue
        original = axis.get("pid_delta_pct")
        if isinstance(original, dict):
            axis["baseline_hold_original_pid_delta_pct"] = dict(original)
            for key in list(original.keys()):
                original[key] = 0.0
        else:
            axis["pid_delta_pct"] = dict(zero)
        axis["baseline_hold_active"] = True
        moves = axis.setdefault("tuning_moves", [])
        if isinstance(moves, list):
            _append_unique(moves, [reason])

    recs = analysis.get("recommendations")
    if isinstance(recs, list):
        for rec in recs:
            if not isinstance(rec, dict):
                continue
            original = rec.get("pid_delta_pct")
            if isinstance(original, dict):
                rec["baseline_hold_original_pid_delta_pct"] = dict(original)
                for key in list(original.keys()):
                    original[key] = 0.0
            else:
                rec["pid_delta_pct"] = dict(zero)
            rec["baseline_hold_active"] = True
            moves = rec.setdefault("tuning_moves", [])
            if isinstance(moves, list):
                _append_unique(moves, [reason])


def _hold_tuning_advice_deltas(advice: Dict[str, Any], reason: str) -> None:
    """Hold Betaflight-style tuning advice after a good/cool baseline selection."""
    zero = {
        "p_percent": 0,
        "i_percent": 0,
        "d_percent": 0,
        "dmax_percent": 0,
        "ff_percent": 0,
    }

    axes = advice.get("axes")
    if isinstance(axes, dict):
        for axis_name, axis in axes.items():
            if not isinstance(axis, dict):
                continue
            deltas = axis.get("deltas")
            if isinstance(deltas, dict):
                axis["baseline_hold_original_deltas"] = dict(deltas)
                deltas.update(zero)
            else:
                axis["deltas"] = dict(zero)
            if axis_name in {"roll", "pitch"}:
                axis["action"] = "hold_baseline"
                axis["severity"] = "baseline_protected"
                axis["reason"] = reason
            notes = axis.setdefault("notes", [])
            if isinstance(notes, list):
                _append_unique(notes, [reason])

    advice["baseline_hold_active"] = True
    advice["confidence"] = "baseline_hold"
    advice["summary"] = (
        "Pilot baseline hold active: the flight was reported good/cool, so AeroTune is not recommending PID changes from this one log. "
        "Save it as the baseline and only tune if the same issue repeats in another comparable flight."
    )
    advice["betaflight_steps"] = [
        "Hold current Betaflight P/I/D/D Max/FF values for this report.",
        "Save this log as the good baseline.",
        "Fly one more similar 30-90 second validation log if you want confirmation.",
        "Only revisit P/D/FF changes if the same bounceback, propwash, heat, or vibration symptom repeats.",
    ]


def apply_pilot_feel_to_analysis(
    analysis: Any,
    pilot_feel: Any,
    drone_size: Optional[str] = None,
) -> Any:
    """
    Code label: ANALYSIS CONTEXT MERGER

    Attaches pilot feel + filtering advisor + real plan to the analyzer output.
    This function intentionally mutates only the response dictionary, not the raw dataframe.
    """
    if not isinstance(analysis, dict):
        return analysis

    context = normalize_pilot_feel(pilot_feel)
    advisor = build_filtering_advisor(analysis, context, drone_size=drone_size)

    analysis["pilot_feel_context"] = context
    analysis["filtering_advisor"] = advisor
    analysis["real_world_tuning_plan"] = advisor.get("plan", [])

    warnings = analysis.setdefault("warnings", [])
    if isinstance(warnings, list):
        _append_unique(warnings, context.get("warnings", []))
        _append_unique(warnings, advisor.get("warnings", []))

    actions = analysis.setdefault("global_actions", [])
    if isinstance(actions, list):
        _append_unique(actions, advisor.get("next_actions", []))

    if advisor.get("positive_d_blocked"):
        reason = "Pilot feel safety gate: positive D/D Max increases are blocked until motors are cool and filtering/noise are stable."
        _block_positive_deltas_in_analysis(analysis, reason)

    if advisor.get("good_flight_protection"):
        reason = "Pilot baseline hold: good/cool flight reported, so PID deltas are held until repeated evidence confirms the same issue."
        _hold_pid_deltas_in_analysis(analysis, reason)
        original = str(analysis.get("summary") or "")
        prefix = "Pilot feel gate: good flight + cool motors / baseline hold. Hold the tune unless the same issue repeats."
        analysis["summary"] = prefix if not original else f"{prefix} {original}"
        analysis["pilot_feel_adjusted"] = True
        analysis["recommendation_strength"] = "hold_baseline_no_pid_change"
    elif context.get("flags", {}).get("has_pilot_input"):
        analysis["pilot_feel_adjusted"] = True
        analysis["recommendation_strength"] = "pilot_context_applied"

    _cap_analysis_pid_deltas(analysis)
    analysis["pid_delta_guardrail"] = {
        "max_single_pass_delta_fraction": MAX_ANALYSIS_PID_DELTA_FRACTION,
        "max_single_pass_delta_percent": int(MAX_ANALYSIS_PID_DELTA_FRACTION * 100),
        "rule": "Pilot feel options are context/gates. They do not stack into uncapped PID jumps.",
    }

    return analysis


def apply_pilot_feel_to_tuning_advice(
    advice: Any,
    pilot_feel: Any,
    existing_analysis: Optional[Dict[str, Any]] = None,
    drone_size: Optional[str] = None,
) -> Any:
    """
    Code label: PID-ADVICE SAFETY GATE

    The tuning advisor speaks in Betaflight PID deltas. This gate prevents the
    advisor from suggesting more D/D Max when the pilot selected hot/buzzy motors
    or RPM-filter reliability problems.
    """
    if not isinstance(advice, dict):
        return advice

    context = normalize_pilot_feel(pilot_feel)
    advisor = build_filtering_advisor(existing_analysis or {}, context, drone_size=drone_size)

    advice["pilot_feel_context"] = context
    advice["filtering_gate"] = {
        "default_filter_path": advisor.get("default_filter_path"),
        "rpm_filter_policy": advisor.get("rpm_filter_policy"),
        "positive_d_allowed": advisor.get("positive_d_allowed"),
        "good_flight_protection": advisor.get("good_flight_protection"),
        "baseline_hold_requested": advisor.get("baseline_hold_requested"),
        "baseline_hold_bypassed_by_symptoms": advisor.get("baseline_hold_bypassed_by_symptoms"),
        "active_tuning_symptom": advisor.get("active_tuning_symptom"),
        "active_tuning_symptom_ids": advisor.get("active_tuning_symptom_ids"),
        "log_pilot_disagreement": advisor.get("log_pilot_disagreement"),
        "summary": advisor.get("summary"),
    }

    safety = advice.setdefault("safety", [])
    test_plan = advice.setdefault("test_plan", [])
    if isinstance(safety, list):
        _append_unique(safety, advisor.get("warnings", []))
        _append_unique(safety, ["Dynamic Notch is AeroTune's default filtering path; RPM filtering is advanced-only."])
    if isinstance(test_plan, list):
        _append_unique(test_plan, advisor.get("next_actions", []))

    if advisor.get("good_flight_protection"):
        reason = "Pilot baseline hold: good/cool flight reported, so PID deltas are held until repeated evidence confirms the same issue."
        _hold_tuning_advice_deltas(advice, reason)
    elif advisor.get("positive_d_blocked"):
        reason = "Pilot feel safety gate blocked positive D/D Max because motors/filtering were not reported safe."
        axes = advice.get("axes")
        if isinstance(axes, dict):
            for axis in axes.values():
                if not isinstance(axis, dict):
                    continue
                deltas = axis.get("deltas")
                if isinstance(deltas, dict):
                    if float(deltas.get("d_percent", 0) or 0) > 0:
                        deltas["d_percent"] = 0
                    if float(deltas.get("dmax_percent", 0) or 0) > 0:
                        deltas["dmax_percent"] = 0
                notes = axis.setdefault("notes", [])
                if isinstance(notes, list):
                    _append_unique(notes, [reason])
        advice["summary"] = f"{advisor.get('summary')} {advice.get('summary', '')}".strip()

    _cap_advice_pid_deltas(advice)
    advice["pid_delta_guardrail"] = {
        "max_single_pass_delta_percent": MAX_ADVICE_PID_DELTA_PERCENT,
        "rule": "Final Betaflight percent advice is capped after pilot-feel gates so symptoms cannot stack into unsafe PID jumps.",
    }

    return advice
