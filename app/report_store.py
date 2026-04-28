# Copyright © 2026 Christopher Bostrom. All Rights Reserved.
# Source-available for personal evaluation only. See LICENSE and NOTICE.
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


REPORT_ROOT = Path("reports") / "tune_changes"
REPORT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{5,120}$")


class ReportStoreError(RuntimeError):
    """Raised when a tune-change report cannot be saved or loaded safely."""


def _safe_text(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _slugify(value: Any, fallback: str = "tune_change") -> str:
    text = _safe_text(value, fallback).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = fallback
    return text[:42]


def _clip(value: Any, limit: int = 180) -> str:
    text = _safe_text(value, "")
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _round(value: Any, digits: int = 2) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return round(number, digits)


def _json_safe(value: Any) -> Any:
    """
    Convert nested values into JSON-safe primitives.

    This keeps saved reports stable even if a future analysis object includes
    numpy/pandas scalar values.
    """
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _validate_report_id(report_id: str) -> str:
    value = _safe_text(report_id)
    if not REPORT_ID_PATTERN.match(value):
        raise ReportStoreError("Invalid report ID.")
    if ".." in value or "/" in value or "\\" in value:
        raise ReportStoreError("Invalid report ID.")
    return value


def _metric_line(label: str, value: Any) -> str:
    rounded = _round(value)
    if rounded is None:
        return f"- **{label}:** not available"
    return f"- **{label}:** {rounded}%"


def _axis_lines(axis_verdicts: Dict[str, Any]) -> str:
    if not axis_verdicts:
        return "- No axis verdicts returned."

    lines = []
    for axis in ("roll", "pitch", "yaw"):
        payload = axis_verdicts.get(axis)
        if not isinstance(payload, dict):
            continue
        score = _round(payload.get("score_pct"))
        score_text = "not available" if score is None else f"{score}%"
        lines.append(
            f"- **{axis.upper()}:** {payload.get('verdict', 'unknown')} "
            f"({score_text}) — {_clip(payload.get('read', ''))}"
        )

    return "\n".join(lines) if lines else "- No axis verdicts returned."


def _warning_lines(warnings: Iterable[Any]) -> str:
    items = [_clip(item, 240) for item in warnings if _safe_text(item)]
    if not items:
        return "- No warnings returned."
    return "\n".join(f"- {item}" for item in items)


def build_tune_change_markdown(result: Dict[str, Any], report_info: Dict[str, Any]) -> str:
    tracking = result.get("tune_tracking", {}) if isinstance(result.get("tune_tracking"), dict) else {}
    comparison = result.get("comparison", {}) if isinstance(result.get("comparison"), dict) else {}
    baseline = tracking.get("baseline_status", {}) if isinstance(tracking.get("baseline_status"), dict) else {}
    metrics = tracking.get("metric_groups", {}) if isinstance(tracking.get("metric_groups"), dict) else {}
    before = result.get("before", {}) if isinstance(result.get("before"), dict) else {}
    after = result.get("after", {}) if isinstance(result.get("after"), dict) else {}

    changed_helped = tracking.get("changed_helped")
    if changed_helped is True:
        helped_text = "Yes"
    elif changed_helped is False:
        helped_text = "No"
    else:
        helped_text = "Unclear"

    lines = [
        "# AeroTune Tune-Change Report",
        "",
        f"**Report ID:** `{report_info['report_id']}`",
        f"**Created:** {report_info['created_at']}",
        f"**Drone size:** {result.get('drone_size', 'unknown')}",
        f"**Tuning goal:** {result.get('tuning_goal', 'unknown')}",
        "",
        "## Files",
        "",
        f"- **Before:** {before.get('source_filename', 'unknown')}",
        f"- **After:** {after.get('source_filename', 'unknown')}",
        "",
        "## Tune Change Notes",
        "",
        _safe_text(result.get("tune_changes"), "No tune-change notes provided."),
        "",
        "## Verdict",
        "",
        f"- **Overall verdict:** {tracking.get('overall_verdict', comparison.get('overall_verdict', 'unknown'))}",
        f"- **Decision:** {tracking.get('decision', 'review')}",
        f"- **Changed helped:** {helped_text}",
        f"- **Score:** {tracking.get('overall_score_pct', comparison.get('overall_score_pct', 'not available'))}%",
        f"- **Confidence:** {tracking.get('confidence', comparison.get('confidence', 'not available'))}",
        "",
        "## Summary",
        "",
        _safe_text(tracking.get("summary"), comparison.get("summary", "No summary returned.")),
        "",
        "## Metric Groups",
        "",
        _metric_line("Noise improvement", metrics.get("noise_improvement_pct")),
        _metric_line("Tracking improvement", metrics.get("tracking_improvement_pct")),
        _metric_line("Propwash improvement", metrics.get("propwash_improvement_pct")),
        "",
        "## Axis Verdicts",
        "",
        _axis_lines(tracking.get("axis_verdicts", {})),
        "",
        "## Safe Baseline / Style Tuning",
        "",
        f"- **Before clean baseline:** {baseline.get('before_clean_baseline', 'unknown')}",
        f"- **After clean baseline:** {baseline.get('after_clean_baseline', 'unknown')}",
        f"- **Style tuning allowed:** {baseline.get('style_tuning_allowed', 'unknown')}",
        f"- **Guidance:** {_safe_text(baseline.get('style_guidance'), 'No style guidance returned.')}",
        "",
        "## Next Step",
        "",
        _safe_text(tracking.get("next_step"), comparison.get("next_step", "Retest with a similar flight log.")),
        "",
        "## Warnings",
        "",
        _warning_lines(tracking.get("warnings", []) or comparison.get("warnings", [])),
        "",
        "---",
        "",
        "This report was generated by AeroTune V1.5.1. AeroTune is source-available for personal evaluation only. See LICENSE and NOTICE.",
        "",
    ]

    return "\n".join(lines)


def save_tune_change_report(result: Dict[str, Any]) -> Dict[str, Any]:
    """Save a V1.5 tune-change result as JSON and Markdown."""
    if not isinstance(result, dict):
        raise ReportStoreError("Tune-change result must be a dictionary.")

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    created_at = now.isoformat(timespec="seconds")
    stamp = now.strftime("%Y%m%d_%H%M%S")
    slug = _slugify(result.get("tune_changes"), "tune_change")
    short_id = uuid.uuid4().hex[:8]
    report_id = f"{stamp}_{slug}_{short_id}"

    json_path = REPORT_ROOT / f"{report_id}.json"
    markdown_path = REPORT_ROOT / f"{report_id}.md"

    report_info = {
        "report_id": report_id,
        "created_at": created_at,
        "json_filename": json_path.name,
        "markdown_filename": markdown_path.name,
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "download_json_url": f"/reports/tune-changes/{report_id}.json",
        "download_markdown_url": f"/reports/tune-changes/{report_id}.md",
    }

    saved_result = dict(result)
    saved_result["report"] = report_info
    safe_result = _json_safe(saved_result)

    try:
        json_path.write_text(json.dumps(safe_result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        markdown_path.write_text(build_tune_change_markdown(safe_result, report_info), encoding="utf-8")
    except OSError as exc:
        raise ReportStoreError(f"Could not save tune-change report: {exc}") from exc

    return report_info


def get_tune_change_report_path(report_id: str, extension: str) -> Path:
    """Return a saved report path after validating report ID and extension."""
    safe_id = _validate_report_id(report_id)
    ext = extension.lower().lstrip(".")
    if ext not in {"json", "md"}:
        raise ReportStoreError("Unsupported report type.")

    path = REPORT_ROOT / f"{safe_id}.{ext}"
    try:
        resolved_root = REPORT_ROOT.resolve()
        resolved_path = path.resolve()
    except OSError as exc:
        raise ReportStoreError(f"Could not resolve report path: {exc}") from exc

    if resolved_root not in resolved_path.parents and resolved_path != resolved_root:
        raise ReportStoreError("Invalid report path.")

    if not resolved_path.exists() or not resolved_path.is_file():
        raise ReportStoreError("Tune-change report not found.")

    return resolved_path
