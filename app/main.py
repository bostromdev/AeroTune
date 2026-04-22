from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.analyzer import detect_oscillation
from app.parser import parse_log

app = FastAPI(title="AeroTune")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

LAST_FILE_PATH: Optional[Path] = None

ALLOWED_DRONE_SIZES = {"5", "7"}
ALLOWED_TUNING_GOALS = {
    "efficiency",
    "efficiency_snappy",
    "efficiency_floaty",
    "snappy",
    "floaty",
}
MAX_UPLOAD_SIZE_BYTES = 25 * 1024 * 1024

app.mount("/static", StaticFiles(directory="static"), name="static")


def error_response(message: str, status_code: int = 400, **extra):
    payload = {"error": message}
    if extra:
        payload.update(extra)
    return JSONResponse(payload, status_code=status_code)


def safe_filename(filename: str) -> str:
    raw = (filename or "upload.csv").strip()
    name = Path(raw).name
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", ".", " "} else "_" for ch in name).strip(" .")
    return cleaned or "upload.csv"


def save_upload(file: UploadFile) -> Path:
    filename = safe_filename(file.filename or "upload.csv")
    path = UPLOAD_DIR / filename

    stem = path.stem
    ext = path.suffix or ".csv"
    suffix = 1

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
        raise ValueError("Plot data requires time and gyro_x.")

    time = df["time"].to_numpy(dtype=float)
    gyro = df["gyro_x"].to_numpy(dtype=float)
    setpoint = df["setpoint_roll"].to_numpy(dtype=float) if "setpoint_roll" in df.columns else np.zeros_like(gyro)

    if len(time) < 8:
        raise ValueError("Not enough samples to plot.")

    dt = float(np.mean(np.diff(time)))
    if not np.isfinite(dt) or dt <= 0:
        dt = 0.001

    alpha = 0.35
    smoothed = np.empty_like(gyro)
    smoothed[0] = gyro[0]
    for i in range(1, len(gyro)):
        smoothed[i] = smoothed[i - 1] + alpha * (gyro[i] - smoothed[i - 1])

    centered = gyro - np.mean(gyro)
    window = np.hanning(len(centered))
    fft_vals = np.fft.rfft(centered * window)
    freqs = np.fft.rfftfreq(len(centered), d=dt)
    mags = np.abs(fft_vals)

    valid = freqs > 0
    freqs = freqs[valid]
    mags = mags[valid]

    peak_frequency_hz = None
    if len(freqs) > 0:
        idx = int(np.argmax(mags))
        peak_frequency_hz = float(freqs[idx])

    return {
        "time": time.tolist(),
        "gyro": gyro.tolist(),
        "setpoint": setpoint.tolist(),
        "simulated": smoothed.tolist(),
        "freqs": freqs.tolist(),
        "magnitude": mags.tolist(),
        "peak_frequency_hz": round(peak_frequency_hz, 2) if peak_frequency_hz is not None else None,
    }


@app.get("/", response_class=HTMLResponse)
def home():
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except Exception as exc:
        return HTMLResponse(f"<h1>Error loading UI</h1><p>{exc}</p>", status_code=500)


@app.post("/upload-log")
async def upload_log(
    file: UploadFile = File(...),
    drone_size: str = Form("7"),
    tuning_goal: str = Form("efficiency"),
):
    global LAST_FILE_PATH

    try:
        if not file.filename:
            return error_response("No file selected.", 400)

        if not file.filename.lower().endswith(".csv"):
            return error_response("Only CSV files are allowed.", 400)

        if drone_size not in ALLOWED_DRONE_SIZES:
            return error_response("Invalid drone size. Use 5 or 7.", 400)

        if tuning_goal not in ALLOWED_TUNING_GOALS:
            return error_response(
                "Invalid tuning goal. Use efficiency, efficiency_snappy, efficiency_floaty, snappy, or floaty.",
                400,
            )

        saved_path = save_upload(file)
        LAST_FILE_PATH = saved_path

        df = parse_log(str(saved_path))
        if df is None or df.empty:
            return error_response(
                "Could not parse CSV. Make sure the file includes usable time and gyro data.",
                400,
            )

        analysis = detect_oscillation(
            df,
            drone_size=drone_size,
            tuning_goal=tuning_goal,
        )

        return {
            "filename": saved_path.name,
            "rows": int(len(df)),
            "columns": list(df.columns),
            "analysis": analysis,
        }

    except ValueError as exc:
        return error_response(str(exc), 400)
    except Exception as exc:
        return error_response(str(exc), 500)
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
            return error_response("No uploaded file available yet.", 400)

        df = parse_log(str(LAST_FILE_PATH))
        if df is None or df.empty:
            return error_response("Could not parse the uploaded file for plotting.", 400)

        return build_plot_payload(df)

    except ValueError as exc:
        return error_response(str(exc), 400)
    except Exception as exc:
        return error_response(str(exc), 500)