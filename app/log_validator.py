from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def validate_log(df: pd.DataFrame) -> Dict[str, Any]:
    warnings: List[str] = []

    if df is None or df.empty:
        return {"valid": False, "warnings": ["CSV is empty."], "rows": 0}

    rows = int(len(df))
    columns = list(df.columns)

    required = ["time", "gyro_x", "gyro_y", "gyro_z"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        return {
            "valid": False,
            "warnings": [f"Missing required columns: {', '.join(missing)}"],
            "rows": rows,
            "columns": columns,
        }

    time = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=float)

    finite = np.isfinite(time)
    if finite.mean() < 0.95:
        warnings.append("Some timestamps are invalid.")

    if len(time) >= 2:
        dt = np.diff(time)
        good_dt = dt[np.isfinite(dt) & (dt > 0)]

        if len(good_dt):
            sample_rate_hz = float(1.0 / np.median(good_dt))
            duration_s = float(time[-1] - time[0])
        else:
            sample_rate_hz = None
            duration_s = None
            warnings.append("Timestamps are not increasing.")
    else:
        sample_rate_hz = None
        duration_s = None
        warnings.append("Not enough rows to estimate sample rate.")

    if rows < 128:
        warnings.append("Log is very short. Use a longer log for safer tuning advice.")

    return {
        "valid": len(warnings) == 0 or all("short" not in w.lower() for w in warnings),
        "warnings": warnings,
        "rows": rows,
        "columns": columns,
        "sample_rate_hz": None if sample_rate_hz is None else round(sample_rate_hz, 2),
        "duration_s": None if duration_s is None else round(duration_s, 3),
    }
