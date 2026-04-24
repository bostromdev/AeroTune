from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _duration_seconds(time_values: pd.Series) -> float | None:
    values = pd.to_numeric(time_values, errors="coerce").dropna()
    if len(values) < 3:
        return None
    duration = float(values.max() - values.min())
    if not np.isfinite(duration) or duration <= 0:
        return None
    return duration


def validate_log(df: pd.DataFrame) -> dict[str, Any]:
    issues: list[str] = []
    warnings: list[str] = []
    score = 100

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

    required = ["time", "gyro_x", "gyro_y", "gyro_z"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        issues.append("Missing required optimized columns: " + ", ".join(missing))
        score -= 40

    row_count = len(df)
    if row_count < 1000:
        issues.append("Log is too short for reliable PID analysis.")
        score -= 25
    elif row_count < 5000:
        warnings.append("Log is usable but short. Longer logs improve accuracy.")
        score -= 10

    duration_s = _duration_seconds(df["time"]) if "time" in df.columns else None
    if duration_s is None:
        warnings.append("Time column exists but could not be reliably parsed.")
        score -= 10
    elif duration_s < 10:
        issues.append("Flight duration is under 10 seconds.")
        score -= 25
    elif duration_s < 30:
        warnings.append("Flight duration is under 30 seconds; 30–120 seconds is preferred.")
        score -= 10

    if "throttle" not in df.columns:
        warnings.append("Missing throttle column; propwash/high-throttle confidence is reduced.")
        score -= 10
    else:
        throttle = pd.to_numeric(df["throttle"], errors="coerce").dropna()
        if len(throttle) > 10:
            variation = float(np.std(throttle))
            if variation < 0.03:
                warnings.append("Throttle variation is very low; include punch-outs, turns, dives, and normal flying.")
                score -= 20
            elif variation < 0.08:
                warnings.append("Throttle variation is limited; include more flight variation for better diagnosis.")
                score -= 10

    for col in ["gyro_x", "gyro_y", "gyro_z"]:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            numeric = values.dropna()
            if numeric.empty:
                score -= 15
                warnings.append(f"{col} has no usable numeric data.")
            elif float(np.std(numeric)) < 1e-9:
                score -= 10
                warnings.append(f"{col} appears flat or inactive.")
            elif float(values.isna().mean()) > 0.05:
                score -= 5
                warnings.append(f"{col} contains missing values.")

    score = max(0, min(100, int(score)))
    grade = "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 70 else "D" if score >= 60 else "F"
    hard_failure = bool(missing) or any("empty" in issue.lower() for issue in issues)
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
