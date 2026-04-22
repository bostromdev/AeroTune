from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.analyzer import detect_oscillation
from app.parser import parse_log

app = FastAPI(title="AeroTune")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

LAST_FILE_PATH = None  # type: Optional[Path]

app.mount("/static", StaticFiles(directory="static"), name="static")

ALLOWED_DRONE_SIZES = {"1", "2", "3", "4", "5", "7"}
MAX_UPLOAD_SIZE_BYTES = 25 * 1024 * 1024


def error_response(message, status_code=400, **extra):
    payload = {"error": message}
    if extra:
        payload.update(extra)
    return JSONResponse(payload, status_code=status_code)


def safe_filename(filename):
    raw = (filename or "upload.csv").strip()
    name = Path(raw).name
    keep = []
    for ch in name:
        if ch.isalnum() or ch in {"-", "_", ".", " "}:
            keep.append(ch)
        else:
            keep.append("_")
    cleaned = "".join(keep).strip(" .") or "upload.csv"
    return cleaned


def save_upload(file):
    filename = safe_filename(file.filename)
    path = UPLOAD_DIR / filename

    suffix = 1
    stem = path.stem
    ext = path.suffix or ".csv"

    while path.exists():
        path = UPLOAD_DIR / f"{stem}_{suffix}{ext}"
        suffix += 1

    bytes_written = 0
    with open(path, "wb") as buffer:
        while True:
            chunk = file.file.read(1024 * 1024)
            if not chunk:
                break
            bytes_written += len(chunk)
            if bytes_written > MAX_UPLOAD_SIZE_BYTES:
                buffer.close()
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                raise ValueError("File too large. Max upload size is 25 MB.")
            buffer.write(chunk)

    return path


def build_plot_payload(df):
    if "time" not in df.columns or "gyro_x" not in df.columns:
        raise ValueError("Plot data requires at least time and gyro_x columns.")

    time = df["time"].to_numpy(dtype=float)
    gyro = df["gyro_x"].to_numpy(dtype=float)

    if len(time) < 8:
        raise ValueError("Not enough samples to generate a plot.")

    roll_candidates = ["setpoint_roll", "rc_command_roll", "rc_roll", "roll_setpoint"]
    setpoint_col = next((c for c in roll_candidates if c in df.columns), None)

    if setpoint_col is not None:
        setpoint = df[setpoint_col].to_numpy(dtype=float)
    else:
        setpoint = np.zeros_like(gyro)

    dt = float(np.mean(np.diff(time)))
    if not np.isfinite(dt) or dt <= 0:
        dt = 0.001

    alpha = 0.35
    smoothed = np.empty_like(gyro, dtype=float)
    smoothed[0] = gyro[0]
    for i in range(1, len(smoothed)):
        smoothed[i] = smoothed[i - 1] + alpha * (gyro[i] - smoothed[i - 1])

    centered = gyro - np.mean(gyro)
    window = np.hanning(len(centered)) if len(centered) >= 8 else np.ones_like(centered)
    fft = np.fft.rfft(centered * window)
    freqs = np.fft.rfftfreq(len(centered), d=dt)
    mags = np.abs(fft)

    valid = freqs > 0
    freqs_valid = freqs[valid]
    mags_valid = mags[valid]

    peak_frequency_hz = None
    if len(freqs_valid) > 0:
        analysis_mask = (freqs_valid >= 5.0) & (freqs_valid <= min(500.0, float(freqs_valid[-1])))
        if np.any(analysis_mask):
            local_freqs = freqs_valid[analysis_mask]
            local_mags = mags_valid[analysis_mask]
            peak_frequency_hz = float(local_freqs[int(np.argmax(local_mags))])

    return {
        "time": time.tolist(),
        "gyro": gyro.tolist(),
        "setpoint": setpoint.tolist(),
        "simulated": smoothed.tolist(),
        "freqs": freqs_valid.tolist(),
        "magnitude": mags_valid.tolist(),
        "peak_frequency_hz": round(peak_frequency_hz, 1) if peak_frequency_hz is not None else None,
    }


def build_ui_analysis(raw_analysis):
    """
    Converts analyzer.py output into a UI-friendly shape while keeping the raw
    analysis untouched.
    """
    axes = raw_analysis.get("axes", {}) or {}

    ui_axes = {}
    for axis_name in ("roll", "pitch", "yaw"):
        axis = axes.get(axis_name, {}) or {}
        signal = axis.get("signal", {}) or {}
        tracking = axis.get("tracking", {}) or {}
        pid_delta = axis.get("pid_delta_pct", {}) or {}

        ui_axes[axis_name] = {
            "axis_name": axis_name.upper(),
            "issue": axis.get("issue", "unknown"),
            "status": axis.get("status", "unknown"),
            "valid_for_pid": bool(axis.get("valid_for_pid", False)),
            "confidence": axis.get("confidence"),
            "recommendation_text": axis.get("recommendation_text", ""),
            "pid_delta_pct": {
                "p": pid_delta.get("p", 0.0),
                "d": pid_delta.get("d", 0.0),
                "ff": pid_delta.get("ff", 0.0),
            },
            "signal": {
                "dominant_freq_hz": signal.get("dominant_freq_hz"),
                "spectral_centroid_hz": signal.get("spectral_centroid_hz"),
                "low_band_energy": signal.get("low_band_energy"),
                "mid_band_energy": signal.get("mid_band_energy"),
                "high_band_energy": signal.get("high_band_energy"),
                "total_band_energy": signal.get("total_band_energy"),
                "rms": signal.get("rms"),
                "peak_to_peak": signal.get("peak_to_peak"),
            },
            "tracking": {
                "usable": tracking.get("usable"),
                "corr": tracking.get("corr"),
                "lag_ms": tracking.get("lag_ms"),
                "tracking_error_rms": tracking.get("tracking_error_rms"),
            },
            "warnings": axis.get("warnings", []) or [],
            "actions": axis.get("actions", []) or [],
        }

    return {
        "status": raw_analysis.get("status", "unknown"),
        "valid_for_pid": bool(raw_analysis.get("valid_for_pid", False)),
        "summary": raw_analysis.get("summary", ""),
        "warnings": raw_analysis.get("warnings", []) or [],
        "global_actions": raw_analysis.get("global_actions", []) or [],
        "sample_rate_hz": raw_analysis.get("sample_rate_hz"),
        "duration_s": raw_analysis.get("duration_s"),
        "drone_size": raw_analysis.get("drone_size"),
        "axes": ui_axes,
        "recommendations": raw_analysis.get("recommendations", []) or [],
    }


@app.get("/", response_class=HTMLResponse)
def home():
    try:
        with open("static/index.html", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse(
            """
            <h1>Error loading UI</h1>
            <p>static/index.html was not found.</p>
            """,
            status_code=500,
        )
    except Exception as e:
        return HTMLResponse(
            f"""
            <h1>Error loading UI</h1>
            <p>{e}</p>
            """,
            status_code=500,
        )


@app.post("/upload-log")
async def upload_log(file=File(...), drone_size=Form("7")):
    global LAST_FILE_PATH

    try:
        if not file.filename:
            return error_response("No file name provided.", 400)

        if not file.filename.lower().endswith(".csv"):
            return error_response("Only CSV files are allowed.", 400)

        if drone_size not in ALLOWED_DRONE_SIZES:
            return error_response("Invalid drone_size. Allowed values: 1, 2, 3, 4, 5, 7.", 400)

        path = save_upload(file)
        LAST_FILE_PATH = path

        df = parse_log(str(path))
        if df is None:
            return error_response(
                "Could not parse CSV. Make sure the file includes usable time and gyro data.",
                400,
            )

        if df.empty:
            return error_response("Parsed file is empty after preprocessing.", 400)

        analysis_raw = detect_oscillation(df, drone_size)
        analysis_ui = build_ui_analysis(analysis_raw)

        return {
            "filename": path.name,
            "rows": int(len(df)),
            "columns": list(df.columns),
            "analysis": analysis_ui,
            "analysis_raw": analysis_raw,
        }

    except ValueError as e:
        return error_response(str(e), 400)
    except Exception as e:
        return error_response(str(e), 500)
    finally:
        try:
            file.file.close()
        except Exception:
            pass


@app.get("/plot")
def plot():
    global LAST_FILE_PATH

    try:
        if LAST_FILE_PATH is None or not LAST_FILE_PATH.exists():
            return error_response("No file uploaded yet.", 400)

        df = parse_log(str(LAST_FILE_PATH))
        if df is None or df.empty:
            return error_response("Uploaded file could not be parsed for plotting.", 400)

        return build_plot_payload(df)

    except ValueError as e:
        return error_response(str(e), 400)
    except Exception as e:
        return error_response(str(e), 500)