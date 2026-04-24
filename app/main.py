from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.analyzer import COMMON_PROBLEM_LIBRARY, detect_oscillation, normalize_goal
from app.log_validator import validate_log
from app.parser import optimize_csv_file, parse_log

app = FastAPI(title="AeroTune")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

LAST_FILE_PATH: Optional[Path] = None
LAST_OPTIMIZED_CSV: Optional[str] = None

ALLOWED_DRONE_SIZES = {"5", "7"}
MAX_UPLOAD_SIZE_BYTES = 250 * 1024 * 1024

app.mount("/static", StaticFiles(directory="static"), name="static")


def error_response(message: str, status_code: int = 400, **extra):
    payload = {"error": message}
    payload.update(extra)
    return JSONResponse(payload, status_code=status_code)


def safe_filename(filename: str | None) -> str:
    raw = (filename or "upload.csv").strip()
    name = Path(raw).name
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", ".", " "} else "_" for ch in name).strip(" .")
    return cleaned or "upload.csv"


def save_upload(file: UploadFile) -> Path:
    filename = safe_filename(file.filename)
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
                raise ValueError("File too large. Max upload size is 250 MB.")
            buffer.write(chunk)

    return path


def optimized_csv_response(csv_text: str, filename: str = "aerotune_optimized.csv") -> StreamingResponse:
    return StreamingResponse(
        iter([csv_text]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def build_plot_payload(df):
    if "time" not in df.columns or "gyro_x" not in df.columns:
        raise ValueError("Plot data requires time and gyro_x.")

    time = df["time"].to_numpy(dtype=float)
    gyro = df["gyro_x"].to_numpy(dtype=float)
    setpoint = df["setpoint_roll"].to_numpy(dtype=float) if "setpoint_roll" in df.columns else np.zeros_like(gyro)

    if len(time) < 8:
        raise ValueError("Not enough samples to plot.")

    max_points = 3000
    step_plot = max(1, int(np.ceil(len(time) / max_points)))
    plot_time = time[::step_plot]
    plot_gyro = gyro[::step_plot]
    plot_setpoint = setpoint[::step_plot]

    dt = float(np.mean(np.diff(time)))
    if not np.isfinite(dt) or dt <= 0:
        dt = 0.001

    centered = gyro - np.mean(gyro)
    step_fft = max(1, int(np.ceil(len(centered) / 8192)))
    fft_signal = centered[::step_fft]
    effective_dt = dt * step_fft
    window = np.hanning(len(fft_signal))
    fft_vals = np.fft.rfft(fft_signal * window)
    freqs = np.fft.rfftfreq(len(fft_signal), d=effective_dt)
    mags = np.abs(fft_vals)
    valid = freqs > 0
    freqs = freqs[valid]
    mags = mags[valid]

    peak_frequency_hz = None
    if len(freqs) > 0:
        peak_frequency_hz = float(freqs[int(np.argmax(mags))])

    return {
        "time": plot_time.tolist(),
        "gyro": plot_gyro.tolist(),
        "setpoint": plot_setpoint.tolist(),
        "freqs": freqs[:2000].tolist(),
        "magnitude": mags[:2000].tolist(),
        "peak_frequency_hz": round(peak_frequency_hz, 2) if peak_frequency_hz is not None else None,
    }


@app.get("/", response_class=HTMLResponse)
def home():
    try:
        return Path("static/index.html").read_text(encoding="utf-8")
    except Exception as exc:
        return HTMLResponse(f"<h1>Error loading UI</h1><p>{exc}</p>", status_code=500)


@app.post("/upload-log")
async def upload_log(
    file: UploadFile = File(...),
    drone_size: str = Form("7"),
    tuning_goal: str = Form("efficient"),
):
    global LAST_FILE_PATH, LAST_OPTIMIZED_CSV

    try:
        if not file.filename:
            return error_response("No file selected.", 400)
        if not file.filename.lower().endswith(".csv"):
            return error_response("Only CSV files are allowed.", 400)
        if drone_size not in ALLOWED_DRONE_SIZES:
            return error_response("Invalid drone size. Use 5 or 7.", 400)

        goal = normalize_goal(tuning_goal)
        saved_path = save_upload(file)
        LAST_FILE_PATH = saved_path

        df = parse_log(saved_path)
        if df is None or df.empty:
            return error_response("Could not parse CSV. Make sure the file includes usable time and gyro data.", 400)

        LAST_OPTIMIZED_CSV = df.to_csv(index=False)
        validation = validate_log(df)
        analysis = detect_oscillation(df, drone_size=drone_size, tuning_goal=goal)

        return {
            "filename": saved_path.name,
            "rows": int(len(df)),
            "columns": list(df.columns),
            "optimized_available": True,
            "optimized_columns": list(df.columns),
            "validation": validation,
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


@app.post("/optimize-log")
async def optimize_log(file: UploadFile = File(...)):
    global LAST_OPTIMIZED_CSV

    try:
        if not file.filename:
            return error_response("No file selected.", 400)
        if not file.filename.lower().endswith(".csv"):
            return error_response("Only CSV files are allowed.", 400)

        saved_path = save_upload(file)
        df = optimize_csv_file(saved_path)
        LAST_OPTIMIZED_CSV = df.to_csv(index=False)
        download_name = f"{Path(safe_filename(file.filename)).stem}_aerotune_ready.csv"
        return optimized_csv_response(LAST_OPTIMIZED_CSV, download_name)

    except ValueError as exc:
        return error_response(str(exc), 400)
    except Exception as exc:
        return error_response(str(exc), 500)
    finally:
        try:
            file.file.close()
        except Exception:
            pass


@app.get("/download-optimized")
def download_optimized():
    if not LAST_OPTIMIZED_CSV:
        return error_response("No optimized CSV available yet. Upload and analyze a log first.", 400)
    return optimized_csv_response(LAST_OPTIMIZED_CSV, "aerotune_optimized.csv")


@app.get("/common-problems")
def common_problems():
    return {"problems": COMMON_PROBLEM_LIBRARY}


@app.get("/plot")
def plot():
    try:
        if LAST_FILE_PATH is None or not LAST_FILE_PATH.exists():
            return error_response("No uploaded file available yet.", 400)
        df = parse_log(LAST_FILE_PATH)
        if df is None or df.empty:
            return error_response("Could not parse the uploaded file for plotting.", 400)
        return build_plot_payload(df)
    except ValueError as exc:
        return error_response(str(exc), 400)
    except Exception as exc:
        return error_response(str(exc), 500)


@app.get("/health")
def health():
    return {"status": "ok"}
