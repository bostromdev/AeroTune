"""
AeroTune V1.9 Tune-Change Option Engine.

Code label: TUNE CHANGE CONTEXT LAYER
-------------------------------------
This module owns the structured checklist shown in the tune-change tracking UI.
It keeps the backend from depending only on free-text notes like
"raised D a little".

Why this code lives here:
- app/main.py should only collect form fields and route files.
- app/tune_tracking.py should decide whether the before/after result helped.
- This file turns checkbox IDs into a machine-readable record of what changed,
  what comparison conditions changed, and which safety warnings should be shown.

Important rule:
Structured tune-change options are context. They do not directly stack PID
percent changes. AeroTune still compares logs and uses hard delta caps elsewhere.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple


# Code label: TUNE CHANGE CHECKBOX CATALOG
# Keep these IDs stable. Saved tune-change reports can reference them later.
TUNE_CHANGE_OPTIONS: Tuple[Dict[str, str], ...] = (
    # PID changes
    {"id": "raised_roll_p", "label": "Raised roll P", "category": "PID changes"},
    {"id": "lowered_roll_p", "label": "Lowered roll P", "category": "PID changes"},
    {"id": "raised_pitch_p", "label": "Raised pitch P", "category": "PID changes"},
    {"id": "lowered_pitch_p", "label": "Lowered pitch P", "category": "PID changes"},
    {"id": "raised_roll_i", "label": "Raised roll I", "category": "PID changes"},
    {"id": "lowered_roll_i", "label": "Lowered roll I", "category": "PID changes"},
    {"id": "raised_pitch_i", "label": "Raised pitch I", "category": "PID changes"},
    {"id": "lowered_pitch_i", "label": "Lowered pitch I", "category": "PID changes"},
    {"id": "raised_roll_d", "label": "Raised roll D", "category": "PID changes"},
    {"id": "lowered_roll_d", "label": "Lowered roll D", "category": "PID changes"},
    {"id": "raised_pitch_d", "label": "Raised pitch D", "category": "PID changes"},
    {"id": "lowered_pitch_d", "label": "Lowered pitch D", "category": "PID changes"},
    {"id": "raised_roll_dmax", "label": "Raised roll D Max", "category": "PID changes"},
    {"id": "lowered_roll_dmax", "label": "Lowered roll D Max", "category": "PID changes"},
    {"id": "raised_pitch_dmax", "label": "Raised pitch D Max", "category": "PID changes"},
    {"id": "lowered_pitch_dmax", "label": "Lowered pitch D Max", "category": "PID changes"},
    {"id": "raised_roll_ff", "label": "Raised roll Feedforward", "category": "PID changes"},
    {"id": "lowered_roll_ff", "label": "Lowered roll Feedforward", "category": "PID changes"},
    {"id": "raised_pitch_ff", "label": "Raised pitch Feedforward", "category": "PID changes"},
    {"id": "lowered_pitch_ff", "label": "Lowered pitch Feedforward", "category": "PID changes"},
    {"id": "changed_yaw_p", "label": "Changed yaw P", "category": "PID changes"},
    {"id": "changed_yaw_i", "label": "Changed yaw I", "category": "PID changes"},
    {"id": "changed_yaw_ff", "label": "Changed yaw Feedforward", "category": "PID changes"},

    # Filtering changes
    {"id": "dynamic_notch_enabled", "label": "Dynamic Notch enabled", "category": "Filtering changes"},
    {"id": "dynamic_notch_disabled", "label": "Dynamic Notch disabled", "category": "Filtering changes"},
    {"id": "dynamic_notch_changed", "label": "Dynamic Notch changed", "category": "Filtering changes"},
    {"id": "gyro_lowpass_changed", "label": "Gyro lowpass changed", "category": "Filtering changes"},
    {"id": "dterm_lowpass_changed", "label": "D-term lowpass changed", "category": "Filtering changes"},
    {"id": "notch_filter_added", "label": "Notch filter added", "category": "Filtering changes"},
    {"id": "notch_filter_removed", "label": "Notch filter removed", "category": "Filtering changes"},
    {"id": "rpm_filter_enabled", "label": "RPM filter enabled", "category": "Filtering changes"},
    {"id": "rpm_filter_disabled", "label": "RPM filter disabled", "category": "Filtering changes"},
    {"id": "rpm_filter_caused_issue", "label": "RPM filter caused issue", "category": "Filtering changes"},
    {"id": "filter_sliders_changed", "label": "Filter sliders changed", "category": "Filtering changes"},

    # Rates / feel changes
    {"id": "raised_rates", "label": "Raised rates", "category": "Rates / feel changes"},
    {"id": "lowered_rates", "label": "Lowered rates", "category": "Rates / feel changes"},
    {"id": "changed_expo", "label": "Changed expo", "category": "Rates / feel changes"},
    {"id": "changed_throttle_curve", "label": "Changed throttle curve", "category": "Rates / feel changes"},
    {"id": "changed_throttle_midpoint", "label": "Changed throttle midpoint", "category": "Rates / feel changes"},
    {"id": "changed_throttle_expo", "label": "Changed throttle expo", "category": "Rates / feel changes"},
    {"id": "changed_motor_idle_dynamic_idle", "label": "Changed motor idle / dynamic idle", "category": "Rates / feel changes"},

    # Hardware / setup changes
    {"id": "changed_props", "label": "Changed props", "category": "Hardware / setup changes"},
    {"id": "same_props", "label": "Same props", "category": "Hardware / setup changes"},
    {"id": "changed_battery", "label": "Changed battery", "category": "Hardware / setup changes"},
    {"id": "same_battery_type", "label": "Same battery type", "category": "Hardware / setup changes"},
    {"id": "added_gopro", "label": "Added GoPro/action camera", "category": "Hardware / setup changes"},
    {"id": "removed_gopro", "label": "Removed GoPro/action camera", "category": "Hardware / setup changes"},
    {"id": "tightened_frame_hardware", "label": "Tightened frame/hardware", "category": "Hardware / setup changes"},
    {"id": "replaced_motor", "label": "Replaced motor", "category": "Hardware / setup changes"},
    {"id": "replaced_esc", "label": "Replaced ESC", "category": "Hardware / setup changes"},
    {"id": "changed_motor_timing", "label": "Changed motor timing", "category": "Hardware / setup changes"},
    {"id": "changed_pwm_frequency", "label": "Changed PWM frequency", "category": "Hardware / setup changes"},
    {"id": "changed_esc_firmware_settings", "label": "Changed ESC firmware/settings", "category": "Hardware / setup changes"},

    # Test condition changes
    {"id": "same_test_route", "label": "Same test route", "category": "Test condition changes"},
    {"id": "different_test_route", "label": "Different test route", "category": "Test condition changes"},
    {"id": "more_aggressive_flight", "label": "More aggressive flight", "category": "Test condition changes"},
    {"id": "less_aggressive_flight", "label": "Less aggressive flight", "category": "Test condition changes"},
    {"id": "wind_changed", "label": "Wind changed", "category": "Test condition changes"},
    {"id": "crash_or_impact_happened", "label": "Crash or impact happened", "category": "Test condition changes"},
    {"id": "motors_cool_after_test", "label": "Motors were cool after test", "category": "Test condition changes"},
    {"id": "motors_warm_after_test", "label": "Motors were warm after test", "category": "Test condition changes"},
    {"id": "motors_hot_after_test", "label": "Motors were hot after test", "category": "Test condition changes"},
)

OPTION_BY_ID: Dict[str, Dict[str, str]] = {item["id"]: item for item in TUNE_CHANGE_OPTIONS}

D_RAISED_IDS = {"raised_roll_d", "raised_pitch_d", "raised_roll_dmax", "raised_pitch_dmax"}
D_LOWERED_IDS = {"lowered_roll_d", "lowered_pitch_d", "lowered_roll_dmax", "lowered_pitch_dmax"}
P_RAISED_IDS = {"raised_roll_p", "raised_pitch_p"}
P_LOWERED_IDS = {"lowered_roll_p", "lowered_pitch_p"}
FF_RAISED_IDS = {"raised_roll_ff", "raised_pitch_ff", "changed_yaw_ff"}
FF_LOWERED_IDS = {"lowered_roll_ff", "lowered_pitch_ff"}
FILTER_CHANGED_IDS = {
    "dynamic_notch_enabled", "dynamic_notch_disabled", "dynamic_notch_changed",
    "gyro_lowpass_changed", "dterm_lowpass_changed", "notch_filter_added",
    "notch_filter_removed", "rpm_filter_enabled", "rpm_filter_disabled",
    "rpm_filter_caused_issue", "filter_sliders_changed",
}
COMPARISON_CONFIDENCE_REDUCER_IDS = {
    "changed_props", "changed_battery", "added_gopro", "removed_gopro",
    "different_test_route", "more_aggressive_flight", "less_aggressive_flight",
    "wind_changed", "crash_or_impact_happened", "replaced_motor", "replaced_esc",
    "changed_motor_timing", "changed_pwm_frequency", "changed_esc_firmware_settings",
}
COMPARISON_CONFIDENCE_BOOST_IDS = {"same_props", "same_battery_type", "same_test_route"}
MOTOR_HEAT_IDS = {"motors_hot_after_test"}
MOTOR_SAFE_IDS = {"motors_cool_after_test"}


def get_tune_change_options() -> List[Dict[str, str]]:
    """Return frontend-safe option metadata."""
    return [dict(item) for item in TUNE_CHANGE_OPTIONS]


def _flatten_values(values: Any) -> List[str]:
    """Accept FastAPI Form lists, comma strings, JSON-like lists, or None."""
    if values is None:
        return []
    if isinstance(values, str):
        raw = values.strip()
        if not raw:
            return []
        return [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]
    if isinstance(values, Iterable):
        out: List[str] = []
        for value in values:
            out.extend(_flatten_values(value))
        return out
    return [str(values).strip()]


def _direction(selected: set[str], up_ids: set[str], down_ids: set[str]) -> str | None:
    up = bool(selected.intersection(up_ids))
    down = bool(selected.intersection(down_ids))
    if up and down:
        return "mixed"
    if up:
        return "up"
    if down:
        return "down"
    return None


def normalize_tune_change_options(values: Any) -> Dict[str, Any]:
    """
    Code label: TUNE CHANGE OPTION NORMALIZER.

    Turns checkbox IDs into flags and warnings. This function never creates PID
    deltas; it only records what changed so the comparison can be interpreted
    without guessing from a text note alone.
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

    selected_set = set(selected)
    selected_labels = [OPTION_BY_ID[key]["label"] for key in selected]

    categories: Dict[str, List[str]] = defaultdict(list)
    for key in selected:
        item = OPTION_BY_ID[key]
        categories[item["category"]].append(item["label"])

    safety_flags: List[str] = []
    confidence_notes: List[str] = []
    interpretation_notes: List[str] = []

    if selected_set.intersection(D_RAISED_IDS):
        safety_flags.append("D/D Max was raised. Confirm motors stayed cool and high-frequency noise did not get worse before stacking more D.")
    if selected_set.intersection(D_LOWERED_IDS):
        safety_flags.append("D/D Max was lowered. Confirm propwash, bounceback, and stop recovery did not get worse.")
    if selected_set.intersection(P_RAISED_IDS):
        safety_flags.append("P was raised. Watch for bounceback, twitchiness, or high-throttle oscillation.")
    if selected_set.intersection(FILTER_CHANGED_IDS):
        safety_flags.append("Filters changed. Compare noise bands carefully before trusting PID-only conclusions.")
    if "rpm_filter_enabled" in selected_set or "rpm_filter_caused_issue" in selected_set:
        safety_flags.append("RPM filtering was enabled or caused an issue. AeroTune should keep Dynamic Notch as the safer default path unless RPM telemetry is proven stable.")
    if selected_set.intersection(MOTOR_HEAT_IDS):
        safety_flags.append("Motors were hot after the test. Do not keep raising D/D Max or reducing filtering until heat is solved.")
    if "crash_or_impact_happened" in selected_set:
        safety_flags.append("Crash/impact happened during the comparison window. Inspect hardware before trusting the after log.")

    if selected_set.intersection(COMPARISON_CONFIDENCE_REDUCER_IDS):
        confidence_notes.append("Comparison confidence reduced because hardware, route, wind, camera weight, ESC/motor setup, or aggression changed between flights.")
    if selected_set.intersection(COMPARISON_CONFIDENCE_BOOST_IDS):
        confidence_notes.append("Comparison confidence improved because props, battery type, or test route were reported consistent.")

    if not selected:
        interpretation_notes.append("No structured tune-change options selected; AeroTune is relying on free-text notes and log comparison only.")
    else:
        interpretation_notes.append("Structured tune-change options were recorded as context, not stacked PID commands.")

    flags = {
        "has_structured_options": bool(selected),
        "d_direction": _direction(selected_set, D_RAISED_IDS, D_LOWERED_IDS),
        "p_direction": _direction(selected_set, P_RAISED_IDS, P_LOWERED_IDS),
        "ff_direction": _direction(selected_set, FF_RAISED_IDS, FF_LOWERED_IDS),
        "filters_changed": bool(selected_set.intersection(FILTER_CHANGED_IDS)),
        "rpm_filter_issue": "rpm_filter_caused_issue" in selected_set,
        "rpm_filter_enabled": "rpm_filter_enabled" in selected_set,
        "motors_hot_after_test": bool(selected_set.intersection(MOTOR_HEAT_IDS)),
        "motors_cool_after_test": bool(selected_set.intersection(MOTOR_SAFE_IDS)),
        "comparison_confidence_reduced": bool(selected_set.intersection(COMPARISON_CONFIDENCE_REDUCER_IDS)),
        "comparison_confidence_boosted": bool(selected_set.intersection(COMPARISON_CONFIDENCE_BOOST_IDS)),
        "same_route_reported": "same_test_route" in selected_set,
        "same_props_reported": "same_props" in selected_set,
        "same_battery_reported": "same_battery_type" in selected_set,
    }

    summary = "No structured tune-change options selected."
    if selected_labels:
        summary = "Structured changes: " + "; ".join(selected_labels[:8])
        if len(selected_labels) > 8:
            summary += f"; +{len(selected_labels) - 8} more"

    return {
        "version": "AeroTune tune-change options v1.0",
        "selected": selected,
        "selected_labels": selected_labels,
        "categories": dict(categories),
        "flags": flags,
        "summary": summary,
        "safety_flags": safety_flags,
        "confidence_notes": confidence_notes,
        "interpretation_notes": interpretation_notes,
        "unknown_values_ignored": unknown,
    }
