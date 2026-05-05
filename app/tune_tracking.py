# Copyright © 2026 Christopher Bostrom. All Rights Reserved.
# Source-available for personal evaluation only. See LICENSE and NOTICE.
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from app.tune_change_options import normalize_tune_change_options
except Exception:
    from .tune_change_options import normalize_tune_change_options


class TuneTrackingError(ValueError):
    """Raised when a tune-change report cannot be created safely."""


UP_WORDS = {
    "up",
    "raise",
    "raised",
    "increase",
    "increased",
    "higher",
    "add",
    "added",
    "plus",
    "+",
    "bump",
    "bumped",
}

DOWN_WORDS = {
    "down",
    "lower",
    "lowered",
    "decrease",
    "decreased",
    "reduce",
    "reduced",
    "less",
    "minus",
    "-",
    "drop",
    "dropped",
}

TERM_ALIASES = {
    "p": ["p", "p gain", "p-gain", "p term", "p-term"],
    "i": ["i", "i gain", "i-gain", "i term", "i-term"],
    "d": ["d", "d gain", "d-gain", "d term", "d-term", "dterm"],
    "ff": ["ff", "feedforward", "feed forward", "feed-forward"],
}

FILTER_WORDS = [
    "filter",
    "filters",
    "gyro filter",
    "dterm filter",
    "d-term filter",
    "rpm filter",
    "dynamic notch",
    "notch",
    "lowpass",
    "low pass",
]

RATE_WORDS = [
    "rate",
    "rates",
    "expo",
    "rc rate",
    "super rate",
    "actual rates",
    "betaflight rates",
]

AXES = ("roll", "pitch", "yaw")


POSITIVE_TUNE_CHANGE_IDS = {
    "after_flight_flew_better",
    "after_flight_locked_in",
    "keep_current_tune",
    "motors_cool_after_test",
    "motors_warm_after_test",
    "motors_holdable_after_test",
}

NEGATIVE_TUNE_CHANGE_IDS = {
    "after_flight_flew_worse",
    "motors_hot_after_test",
    "crash_or_impact_happened",
}

CONSISTENCY_TUNE_CHANGE_IDS = {
    "same_props",
    "same_battery_type",
    "same_test_route",
}

ROUTE_MISMATCH_IDS = {
    "different_test_route",
    "more_aggressive_flight",
    "less_aggressive_flight",
    "wind_changed",
}


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(number):
        return default
    return number


def _round(value: Any, digits: int = 3) -> Optional[float]:
    number = _safe_float(value)
    if number is None:
        return None
    return round(number, digits)


def _normalize_notes(notes: str | None) -> str:
    return re.sub(r"\s+", " ", str(notes or "").strip())


def _window(text: str, start: int, end: int, radius: int = 34) -> str:
    lo = max(0, start - radius)
    hi = min(len(text), end + radius)
    return text[lo:hi]


def _direction_near_match(text: str, start: int, end: int) -> Optional[str]:
    """Find the closest direction word/sign around a PID term mention."""
    # First inspect the comma/semicolon-separated phrase that contains the term.
    # This handles notes like: "raised roll D 2%, lowered pitch P 1%".
    left_delims = [text.rfind(",", 0, start), text.rfind(";", 0, start), text.rfind("\n", 0, start)]
    right_delims = [pos for pos in [text.find(",", end), text.find(";", end), text.find("\n", end)] if pos != -1]
    phrase_start = max(left_delims) + 1
    phrase_end = min(right_delims) if right_delims else len(text)
    phrase = text[phrase_start:phrase_end]

    if re.search(r"\b(unchanged|same|no change|left alone|leave alone|kept)\b", phrase):
        return None

    if re.search(r"\+\s*\d+(\.\d+)?\s*%", phrase):
        return "up"
    if re.search(r"-\s*\d+(\.\d+)?\s*%", phrase):
        return "down"

    phrase_tokens = set(re.findall(r"[a-zA-Z+\-]+", phrase.lower()))
    phrase_up = bool(phrase_tokens & UP_WORDS)
    phrase_down = bool(phrase_tokens & DOWN_WORDS)
    if phrase_up and not phrase_down:
        return "up"
    if phrase_down and not phrase_up:
        return "down"

    post = text[end:min(len(text), end + 24)]
    if re.search(r"\+\s*\d+(\.\d+)?\s*%", post):
        return "up"
    if re.search(r"-\s*\d+(\.\d+)?\s*%", post):
        return "down"

    pre = text[max(0, start - 42):start]
    candidates: List[Tuple[int, str]] = []

    for word in UP_WORDS:
        for match in re.finditer(re.escape(word), pre):
            candidates.append((len(pre) - match.end(), "up"))
        for match in re.finditer(re.escape(word), post):
            candidates.append((match.start(), "up"))

    for word in DOWN_WORDS:
        for match in re.finditer(re.escape(word), pre):
            candidates.append((len(pre) - match.end(), "down"))
        for match in re.finditer(re.escape(word), post):
            candidates.append((match.start(), "down"))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    best_distance = candidates[0][0]
    nearest = [direction for distance, direction in candidates if distance == best_distance]
    unique = sorted(set(nearest))
    if len(unique) == 1:
        return unique[0]
    return "mixed"


def _term_pattern(alias: str) -> re.Pattern[str]:
    escaped = re.escape(alias).replace(r"\ ", r"\s+")
    if len(alias) == 1:
        return re.compile(rf"(?<![a-zA-Z]){escaped}(?![a-zA-Z])", flags=re.IGNORECASE)
    return re.compile(rf"(?<![a-zA-Z]){escaped}(?![a-zA-Z])", flags=re.IGNORECASE)


def parse_tune_change_notes(notes: str | None) -> Dict[str, Any]:
    """Turn human tune-change notes into a small machine-readable report."""
    raw = _normalize_notes(notes)
    text = raw.lower()

    terms: Dict[str, Dict[str, Any]] = {}
    mentioned_terms: List[str] = []

    for term, aliases in TERM_ALIASES.items():
        contexts: List[str] = []
        directions: List[str] = []

        for alias in aliases:
            for match in _term_pattern(alias).finditer(text):
                ctx = _window(text, match.start(), match.end())
                contexts.append(ctx)
                direction = _direction_near_match(text, match.start(), match.end())
                if direction:
                    directions.append(direction)

        if contexts:
            mentioned_terms.append(term.upper())
            direction = None
            unique_directions = sorted(set(directions))
            if len(unique_directions) == 1:
                direction = unique_directions[0]
            elif len(unique_directions) > 1:
                direction = "mixed"

            terms[term] = {
                "mentioned": True,
                "direction": direction,
                "contexts": contexts[:5],
            }
        else:
            terms[term] = {"mentioned": False, "direction": None, "contexts": []}

    filter_contexts = [word for word in FILTER_WORDS if word in text]
    rate_contexts = [word for word in RATE_WORDS if word in text]

    safety_flags: List[str] = []
    if terms["d"]["direction"] == "up":
        safety_flags.append("D was raised. Confirm motor temperature and noise before stacking more changes.")
    if terms["d"]["direction"] == "down":
        safety_flags.append("D was lowered. Confirm propwash/bounceback did not get worse.")
    if any(word in text for word in ["less filtering", "lower filter", "reduced filter", "raise cutoff", "higher cutoff"]):
        safety_flags.append("Filtering may have been reduced. Watch high-frequency noise and motor heat.")
    if terms["p"]["direction"] == "up":
        safety_flags.append("P was raised. Watch for bounceback, oscillation, or over-correction.")
    if not raw:
        safety_flags.append("No tune-change notes were entered. Comparison can still run, but the cause is less clear.")

    return {
        "raw_notes": raw,
        "has_notes": bool(raw),
        "mentioned_pid_terms": mentioned_terms,
        "pid_terms": terms,
        "filter_changes_detected": bool(filter_contexts),
        "filter_keywords": filter_contexts,
        "rate_changes_detected": bool(rate_contexts),
        "rate_keywords": rate_contexts,
        "safety_flags": safety_flags,
    }


def _metric_improvement(comparison: Dict[str, Any], metric_name: str) -> Optional[float]:
    values: List[float] = []

    axes = comparison.get("axes", {})
    if not isinstance(axes, dict):
        return None

    for axis_payload in axes.values():
        metric_changes = axis_payload.get("metric_changes", {}) if isinstance(axis_payload, dict) else {}
        metric = metric_changes.get(metric_name, {}) if isinstance(metric_changes, dict) else {}
        value = _safe_float(metric.get("improvement_pct"))
        if value is not None:
            values.append(value)

    if not values:
        return None
    return float(np.mean(values))


def _average_metric_group(comparison: Dict[str, Any], metric_names: Iterable[str]) -> Optional[float]:
    values = [_metric_improvement(comparison, name) for name in metric_names]
    clean_values = [value for value in values if value is not None]
    if not clean_values:
        return None
    return float(np.mean(clean_values))


def _analysis_is_clean(analysis: Dict[str, Any]) -> bool:
    if not isinstance(analysis, dict):
        return False

    if isinstance(analysis.get("clean_baseline"), bool):
        return bool(analysis["clean_baseline"])

    axes = analysis.get("axes", {})
    if not isinstance(axes, dict) or not axes:
        return False

    return all((payload or {}).get("issue") == "clean" for payload in axes.values())


def _dirty_axes(analysis: Dict[str, Any]) -> List[str]:
    axes = analysis.get("axes", {}) if isinstance(analysis, dict) else {}
    if not isinstance(axes, dict):
        return []

    dirty = []
    for axis, payload in axes.items():
        if isinstance(payload, dict) and payload.get("issue") != "clean":
            dirty.append(f"{axis.upper()}: {payload.get('issue_label') or payload.get('issue')}")
    return dirty


def _coerce_option_ids(raw_options: Any) -> List[str]:
    """Return raw checkbox IDs even when the normalizer ignores newer UI-only IDs."""
    if raw_options is None:
        return []
    if isinstance(raw_options, str):
        return [raw_options] if raw_options else []
    if isinstance(raw_options, (list, tuple, set)):
        return [str(item) for item in raw_options if item is not None and str(item)]
    return []


def _note_has_any(notes: str | None, needles: Iterable[str]) -> bool:
    text = _normalize_notes(notes).lower()
    return any(needle in text for needle in needles)


def _build_pilot_context(notes: str | None, option_ids: Iterable[str], option_report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a human-readability layer for tune tracking.

    The Blackbox comparison stays in charge of the data, but a good-feeling,
    cool-motor after flight should change the wording from "bad/retest" to
    "usable baseline, validate once more" when the math itself is low-confidence.
    """
    ids = {str(item) for item in option_ids if item}
    flags = option_report.get("flags", {}) if isinstance(option_report, dict) else {}

    positive_from_notes = _note_has_any(
        notes,
        [
            "flew perfect",
            "flies perfect",
            "flew good",
            "flies good",
            "felt good",
            "feels good",
            "locked in",
            "locked-in",
            "hold current tune",
            "keep current tune",
            "motors cool",
            "motors stayed cool",
            "motors warm",
            "holdable",
            "not hot",
        ],
    )
    negative_from_notes = _note_has_any(
        notes,
        [
            "flew worse",
            "feels worse",
            "hot motors",
            "motors hot",
            "too hot",
            "desync",
            "failsafe",
            "crash",
        ],
    )

    motor_temp_ok = bool(ids & {"motors_cool_after_test", "motors_warm_after_test", "motors_holdable_after_test"}) or _note_has_any(
        notes,
        ["motors cool", "motors stayed cool", "motors warm", "holdable", "not hot"],
    )
    motor_temp_hot = "motors_hot_after_test" in ids or _note_has_any(notes, ["hot motors", "motors hot", "too hot"])
    positive_feel = bool(ids & POSITIVE_TUNE_CHANGE_IDS) or positive_from_notes
    negative_feel = bool(ids & NEGATIVE_TUNE_CHANGE_IDS) or negative_from_notes
    consistency_reported = bool(ids & CONSISTENCY_TUNE_CHANGE_IDS) or bool(flags.get("comparison_confidence_boosted"))
    route_mismatch_possible = bool(ids & ROUTE_MISMATCH_IDS) or bool(flags.get("comparison_confidence_reduced"))

    notes_out: List[str] = []
    if positive_feel and not negative_feel:
        notes_out.append("Pilot feel supports keeping the current tune.")
    if motor_temp_ok and not motor_temp_hot:
        notes_out.append("Motor temperature was reported acceptable/holdable.")
    if route_mismatch_possible:
        notes_out.append("Comparison may be limited by route, wind, hardware, camera weight, or aggression differences.")
    elif consistency_reported:
        notes_out.append("Props, battery type, and/or test route were reported consistent.")
    if "low_throttle_hover_narrow" in ids:
        notes_out.append("Low-throttle hover feel should be handled with throttle mid/expo, not PID/filter stacking.")
    if motor_temp_hot:
        notes_out.append("Motor heat was reported; do not keep raising D/D Max without reducing heat first.")

    if not notes_out:
        notes_out.append("No strong pilot-feel outcome was recorded; treat the data conservatively.")

    return {
        "selected_option_ids": sorted(ids),
        "positive_flight_feel": bool(positive_feel and not negative_feel),
        "negative_flight_feel": bool(negative_feel),
        "motor_temp_ok": bool(motor_temp_ok and not motor_temp_hot),
        "motor_temp_hot": bool(motor_temp_hot),
        "consistency_reported": bool(consistency_reported),
        "route_mismatch_possible": bool(route_mismatch_possible),
        "summary": " ".join(notes_out),
        "notes": notes_out,
    }


def _propwash_like_text(value: str) -> str:
    text = str(value or "")
    text = text.replace("propwash-band", "propwash-like recovery-band")
    text = re.sub(r"\bPropwash\b(?!-like)", "Propwash-like recovery", text)
    text = re.sub(r"\bpropwash\b(?!-like)", "propwash-like recovery", text)
    return text


def _axis_verdicts(comparison: Dict[str, Any]) -> Dict[str, Any]:
    axes = comparison.get("axes", {})
    result: Dict[str, Any] = {}

    if not isinstance(axes, dict):
        return result

    for axis, payload in axes.items():
        if not isinstance(payload, dict):
            continue
        result[axis] = {
            "verdict": payload.get("verdict"),
            "score_pct": _round(payload.get("score_pct"), 2),
            "confidence": _round(payload.get("confidence"), 3),
            "read": payload.get("read"),
        }

    return result


def _decision(
    comparison: Dict[str, Any],
    changes: Dict[str, Any],
    after_analysis: Dict[str, Any],
    noise_improvement: Optional[float],
    tracking_improvement: Optional[float],
    propwash_improvement: Optional[float],
    comparison_confidence: float,
    pilot_context: Dict[str, Any],
) -> Tuple[str, str, List[str]]:
    verdict = str(comparison.get("overall_verdict") or "mostly unchanged")
    score = _safe_float(comparison.get("overall_score_pct"), 0.0) or 0.0
    after_clean = _analysis_is_clean(after_analysis)
    warnings: List[str] = list(changes.get("safety_flags", []))

    d_direction = changes.get("pid_terms", {}).get("d", {}).get("direction")
    p_direction = changes.get("pid_terms", {}).get("p", {}).get("direction")

    if d_direction == "up" and noise_improvement is not None and noise_improvement < -4.0:
        warnings.append("D was raised and high-frequency noise got worse. Do not keep adding D until noise is cleaner; reduce D/D Max slightly or add filtering after one controlled retest.")
    if d_direction == "down" and propwash_improvement is not None and propwash_improvement < -4.0:
        warnings.append("D was lowered and propwash-like recovery-band energy got worse. The D reduction may have removed too much damping.")
    if p_direction == "up" and tracking_improvement is not None and tracking_improvement < -4.0:
        warnings.append("P was raised but tracking did not improve. Avoid stacking more P until the log is cleaner.")

    if verdict == "improved" and after_clean:
        return (
            "keep_and_confirm",
            "The tune change helped and the after log looks clean enough to begin careful style tuning if flight feel agrees.",
            warnings,
        )

    if verdict == "improved":
        return (
            "keep_but_continue_baseline_cleanup",
            "The change helped, but the after log is not clean enough for aggressive style tuning yet. Keep the useful change small and keep cleaning the baseline.",
            warnings,
        )

    if verdict == "worse":
        return (
            "reduce_or_revert",
            "The after log got worse. Revert or halve the last change, then repeat the same test route before changing anything else.",
            warnings,
        )

    if abs(score) < 8.0:
        if pilot_context.get("positive_flight_feel") and not pilot_context.get("motor_temp_hot"):
            warnings.append(
                "Data did not prove a clean improvement, but pilot feel and motor temperature support keeping the current tune for one controlled validation pass."
            )
            return (
                "keep_current_tune_validate",
                "Mixed but usable baseline. Pilot feel supports keeping the current tune, but the logs do not prove a clean improvement because confidence is low or the two flights may not be similar enough. Treat propwash-like recovery numbers as signatures to validate, not confirmed flight-feel problems.",
                warnings,
            )
        return (
            "inconclusive_retest",
            "The logs are mostly unchanged. The change may be too small, or the two flights may not be similar enough. Retest with one clear change and the same route.",
            warnings,
        )

    return (
        "review_manually",
        "The result is not clear enough for an automatic keep/revert decision. Review the axis cards and retest.",
        warnings,
    )


def build_tune_change_tracking(
    tune_changes: str | None,
    before_analysis: Dict[str, Any],
    after_analysis: Dict[str, Any],
    comparison: Dict[str, Any],
    tune_change_options: Any = None,
    pair_selection: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Record a tune change and decide if the after log improved or worsened."""
    if not isinstance(comparison, dict) or not comparison:
        raise TuneTrackingError("Comparison result is required for tune-change tracking.")

    change_report = parse_tune_change_notes(tune_changes)
    raw_option_ids = _coerce_option_ids(tune_change_options)
    option_report = normalize_tune_change_options(tune_change_options)
    pilot_context = _build_pilot_context(tune_changes, raw_option_ids, option_report)

    # Code label: TUNE CHANGE CONTEXT MERGER
    # Free text and checkbox options are combined for warnings/confidence, but
    # they never become additive PID deltas. The before/after logs still decide
    # whether the change helped.
    combined_safety_flags = list(change_report.get("safety_flags", []))
    for flag in option_report.get("safety_flags", []):
        if flag not in combined_safety_flags:
            combined_safety_flags.append(flag)
    change_report["safety_flags"] = combined_safety_flags
    change_report["structured_options"] = option_report

    if option_report.get("flags", {}).get("d_direction") and not change_report.get("pid_terms", {}).get("d", {}).get("direction"):
        change_report["pid_terms"]["d"]["direction"] = option_report["flags"]["d_direction"]
        change_report["pid_terms"]["d"]["mentioned"] = True
    if option_report.get("flags", {}).get("p_direction") and not change_report.get("pid_terms", {}).get("p", {}).get("direction"):
        change_report["pid_terms"]["p"]["direction"] = option_report["flags"]["p_direction"]
        change_report["pid_terms"]["p"]["mentioned"] = True
    if option_report.get("flags", {}).get("ff_direction") and not change_report.get("pid_terms", {}).get("ff", {}).get("direction"):
        change_report["pid_terms"]["ff"]["direction"] = option_report["flags"]["ff_direction"]
        change_report["pid_terms"]["ff"]["mentioned"] = True

    if option_report.get("flags", {}).get("filters_changed"):
        change_report["filter_changes_detected"] = True
    if option_report.get("flags", {}).get("has_structured_options") and not change_report.get("has_notes"):
        change_report["has_notes"] = True

    overall_score = _safe_float(comparison.get("overall_score_pct"), 0.0) or 0.0
    overall_verdict = str(comparison.get("overall_verdict") or "mostly unchanged")
    comparison_confidence = _safe_float(comparison.get("confidence"), 0.35) or 0.35

    noise_improvement = _average_metric_group(comparison, ["high_noise_ratio"])
    tracking_improvement = _average_metric_group(
        comparison,
        ["tracking_error_ratio", "abs_error_p95", "lag_abs_ms"],
    )
    propwash_improvement = _average_metric_group(comparison, ["propwash_ratio"])

    after_clean = _analysis_is_clean(after_analysis)
    before_clean = _analysis_is_clean(before_analysis)
    remaining_dirty_axes = _dirty_axes(after_analysis)

    decision, decision_summary, warnings = _decision(
        comparison=comparison,
        changes=change_report,
        after_analysis=after_analysis,
        noise_improvement=noise_improvement,
        tracking_improvement=tracking_improvement,
        propwash_improvement=propwash_improvement,
        comparison_confidence=comparison_confidence,
        pilot_context=pilot_context,
    )

    confidence = comparison_confidence
    if not change_report["has_notes"]:
        confidence -= 0.12
    if overall_verdict == "mostly unchanged":
        confidence -= 0.08
    if option_report.get("flags", {}).get("comparison_confidence_reduced"):
        confidence -= 0.10
    if option_report.get("flags", {}).get("comparison_confidence_boosted"):
        confidence += 0.04
    if isinstance(pair_selection, dict) and pair_selection.get("same_flight_selected"):
        confidence -= 0.20
    confidence = float(np.clip(confidence, 0.20, 0.95))

    if after_clean:
        style_guidance = (
            "After log looks clean enough for style tuning. You can now test Locked-In or Cinematic changes carefully, "
            "one small move at a time."
        )
    elif pilot_context.get("positive_flight_feel") and not pilot_context.get("motor_temp_hot"):
        style_guidance = (
            "Current tune may be a usable baseline because pilot feel was positive and motors were acceptable. "
            "Do not stack Locked-In or Cinematic tuning yet; repeat one clean same-route validation log first."
        )
    else:
        style_guidance = (
            "Do not move into Locked-In or Cinematic style tuning yet. Clean or validate the remaining baseline issues first."
        )

    return {
        "version": "V1.5",
        "type": "tune_change_tracking",
        "overall_verdict": overall_verdict,
        "overall_score_pct": round(overall_score, 2),
        "decision": decision,
        "changed_helped": True if overall_verdict == "improved" else False if overall_verdict == "worse" else None,
        "confidence": round(confidence, 3),
        "summary": decision_summary,
        "change_notes": change_report,
        "change_options": option_report,
        "pilot_context": pilot_context,
        "pair_selection": pair_selection or {},
        "metric_groups": {
            "noise_improvement_pct": _round(noise_improvement, 2),
            "tracking_improvement_pct": _round(tracking_improvement, 2),
            "propwash_improvement_pct": _round(propwash_improvement, 2),
            "propwash_label": "propwash-like recovery",
        },
        "baseline_status": {
            "before_clean_baseline": before_clean,
            "after_clean_baseline": after_clean,
            "remaining_after_issues": remaining_dirty_axes,
            "style_tuning_allowed": after_clean,
            "style_guidance": style_guidance,
        },
        "axis_verdicts": _axis_verdicts(comparison),
        "warnings": sorted(set(_propwash_like_text(item) for item in (warnings + option_report.get("confidence_notes", []) + option_report.get("interpretation_notes", []) + ((pair_selection or {}).get("warnings", []) if isinstance(pair_selection, dict) else [])))),
        "next_step": _next_step(decision, after_clean, remaining_dirty_axes),
        "interpretation": {
            "goal": "Track what changed, compare before/after logs, and decide whether to keep, reduce, revert, or retest.",
            "not_autotune": "AeroTune does not automatically rewrite PID values. The pilot reviews the result and makes controlled changes.",
            "clean_baseline_rule": "A clean baseline should come before Locked-In or Cinematic style tuning.",
        },
    }


def _next_step(decision: str, after_clean: bool, dirty_axes: List[str]) -> str:
    if decision == "keep_current_tune_validate":
        return "Keep the current tune for now. Do not stack Locked-In/Cinematic changes yet; repeat one same-route validation log with similar props, battery, GoPro weight, and flight length."
    if decision == "keep_and_confirm":
        return "Keep the change, fly one more similar 60–90 second log, then begin small style tuning only if the quad still feels safe."
    if decision == "keep_but_continue_baseline_cleanup":
        weakest = dirty_axes[0] if dirty_axes else "the weakest axis"
        return f"Keep the useful part of the change, but continue baseline cleanup first. Focus next test on {weakest}."
    if decision == "reduce_or_revert":
        return "Revert or cut the last change in half. Do not stack new PID changes until the same test route improves again."
    if decision == "inconclusive_retest":
        return "Repeat the same route with one clearly documented PID/filter/rate change. Use similar battery, props, wind, and flight length."
    if after_clean:
        return "After log is clean. Use style modes carefully and change one thing at a time."
    return "Continue baseline cleanup before style tuning."
