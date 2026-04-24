from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


REQUIRED_KEYWORDS = {
    "time": ["time", "timestamp"],
    "gyro": ["gyro", "gyr"],
    "throttle": ["throttle", "thr"],
}


def _find_column(df: pd.DataFrame, keywords: list[str]) -> str | None:
    for col in df.columns:
        clean = str(col).lower().strip()
        if any(key in clean for key in keywords):
            return str(col)
    return None


def _find_columns(df: pd.DataFrame, keywords: list[str]) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        clean = str(col).lower().strip()
        if any(key in clean for key in keywords):
            cols.append(str(col))
    return cols


def _duration_seconds(time_values: pd.Series) -> float | None:
    values = pd.to_numeric(time_values, errors="coerce").dropna()
    if len(values) < 3:
        return None

    duration = float(values.max() - values.min())
    if not np.isfinite(duration) or duration <= 0:
        return None

    # Betaflight exports commonly use microseconds. Some cleaned CSVs use seconds.
    if duration > 10_000:
        return duration / 1_000_000.0
    return duration


def validate_log(df: pd.DataFrame) -> dict[str, Any]:
    """
    Validate FPV Blackbox CSV quality before PID analysis.

    The validator is intentionally conservative. It does not tune the craft;
    it only decides whether the input data is trustworthy enough to analyze.
    """

    issues: list[str] = []
    warnings: list[str] = []
    score = 100
    duration_s: float | None = None

    if df is None or df.empty:
        return {
            "valid": False,
            "score": 0,
            "grade": "F",
            "duration_s": None,
            "issues": ["CSV is empty or unreadable."],
            "warnings": [],
            "summary": "Invalid log: no usable data found.",
        }

    working = df.copy()
    working.columns = [str(c).strip().lower() for c in working.columns]

    time_col = _find_column(working, REQUIRED_KEYWORDS["time"])
    gyro_cols = _find_columns(working, REQUIRED_KEYWORDS["gyro"])
    throttle_col = _find_column(working, REQUIRED_KEYWORDS["throttle"])

    if time_col is None:
        issues.append("Missing time/timestamp column.")
        score -= 25

    if not gyro_cols:
        issues.append("Missing gyro data columns.")
        score -= 35

    if throttle_col is None:
        warnings.append("Missing throttle column; flight variation confidence is reduced.")
        score -= 15

    row_count = len(working)
    if row_count < 1000:
        issues.append("Log is too short for reliable PID analysis.")
        score -= 25
    elif row_count < 5000:
        warnings.append("Log is usable but short. Longer logs improve accuracy.")
        score -= 10

    if time_col is not None:
        duration_s = _duration_seconds(working[time_col])
        if duration_s is None:
            warnings.append("Time column exists but could not be reliably parsed.")
            score -= 10
        elif duration_s < 10:
            issues.append("Flight duration is under 10 seconds.")
            score -= 25
        elif duration_s < 30:
            warnings.append("Flight duration is under 30 seconds; 30–120 seconds is preferred.")
            score -= 10

    if throttle_col is not None:
        throttle = pd.to_numeric(working[throttle_col], errors="coerce").dropna()
        if len(throttle) > 10:
            throttle_variance = float(np.std(throttle))
            if throttle_variance < 5:
                warnings.append("Throttle variation is very low; log may be hover-only.")
                score -= 20
            elif throttle_variance < 15:
                warnings.append("Throttle variation is limited; include punch-outs and turns.")
                score -= 10
        else:
            warnings.append("Throttle data could not be reliably parsed.")
            score -= 10

    if gyro_cols:
        gyro_penalty = 0
        usable_gyro_cols = gyro_cols[:3]
        for col in usable_gyro_cols:
            values = pd.to_numeric(working[col], errors="coerce")
            numeric = values.dropna()
            if numeric.empty:
                gyro_penalty += 10
                continue
            if float(values.isna().mean()) > 0.2:
                gyro_penalty += 10
            if float(np.std(numeric)) == 0:
                gyro_penalty += 15

        if gyro_penalty:
            warnings.append("Gyro signal quality appears weak or incomplete.")
            score -= gyro_penalty

    score = max(0, min(100, int(score)))

    if score >= 90:
        grade = "A"
    elif score >= 80:
        grade = "B"
    elif score >= 70:
        grade = "C"
    elif score >= 60:
        grade = "D"
    else:
        grade = "F"

    hard_failure = any(
        "missing gyro" in issue.lower()
        or "empty" in issue.lower()
        or "unreadable" in issue.lower()
        for issue in issues
    )
    valid = score >= 70 and not hard_failure

    if score >= 85:
        summary = "Excellent log quality. Suitable for PID analysis."
    elif score >= 70:
        summary = "Usable log. Analysis can run, but better flight data may improve accuracy."
    else:
        summary = "Poor log quality. Retake the flight before trusting PID recommendations."

    return {
        "valid": valid,
        "score": score,
        "grade": grade,
        "duration_s": round(duration_s, 3) if duration_s is not None else None,
        "issues": issues,
        "warnings": warnings,
        "summary": summary,
    }
