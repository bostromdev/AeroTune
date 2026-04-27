from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.analyzer import (
    PUBLIC_DRONE_SIZE_OPTIONS,
    detect_oscillation,
    normalize_drone_size,
    normalize_goal,
)
from app.log_validator import validate_log
from app.parser import optimize_csv_file_with_report, parse_log_with_report


app = FastAPI(title="AeroTune")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

LAST_FILE_PATH: Optional[Path] = None
LAST_OPTIMIZED_CSV: Optional[str] = None
LAST_OPTIMIZED_NAME: str = "aerotune_optimized.csv"

# AeroTune can analyze many sizes. Analyzer bands currently fall back safely for unsupported values.
ALLOWED_DRONE_SIZES = set(PUBLIC_DRONE_SIZE_OPTIONS)

# Local-first tool: large Blackbox CSV files are normal.
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


def csv_extension_error(filename: str | None):
    suffix = Path(filename or "").suffix.lower()
    parser_report = {
        "ok": False,
        "filename": safe_filename(filename),
        "file_type": suffix or "unknown",
        "stage": "file_type",
        "message": "AeroTune V1.1/V1.2 only accepts Betaflight CSV exports. Raw .bbl/.bfl support is planned for V1.3.",
        "error_code": "not_csv",
        "header_row": None,
        "metadata_rows_skipped": 0,
        "repeated_header_rows_removed": 0,
        "raw_columns_count": 0,
        "raw_columns_preview": [],
        "detected_columns": {
            "time": None,
            "gyro": {"roll": None, "pitch": None, "yaw": None},
            "setpoint": {"roll": None, "pitch": None, "yaw": None},
            "throttle": None,
        },
        "missing_required": ["time", "gyro_x", "gyro_y", "gyro_z"],
        "missing_optional": ["setpoint_roll", "setpoint_pitch", "setpoint_yaw", "throttle"],
        "sample_rate_hz": None,
        "duration_seconds": None,
        "usable_rows": 0,
        "total_rows_read": 0,
        "warnings": [],
        "suggestions": [
            "Export the log as CSV from Betaflight Blackbox Explorer.",
            "Use .bbl/.bfl files after the V1.3 converter is added.",
        ],
    }
    return error_response(parser_report["message"], 400, parser_report=parser_report)


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
    step = max(1, int(np.ceil(len(centered) / 8192)))
    centered = centered[::step]
    effective_dt = dt * step

    window = np.hanning(len(centered))
    fft_vals = np.fft.rfft(centered * window)
    freqs = np.fft.rfftfreq(len(centered), d=effective_dt)
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


def optimized_csv_response(csv_text: str, filename: str = "aerotune_optimized.csv") -> StreamingResponse:
    return StreamingResponse(
        iter([csv_text]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/", response_class=HTMLResponse)
def home():
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except Exception as exc:
        return HTMLResponse(f"<h1>Error loading UI</h1><pre>{exc}</pre>", status_code=500)


@app.post("/upload-log")
async def upload_log(
    file: UploadFile = File(...),
    drone_size: str = Form("7"),
    tuning_goal: str = Form("efficient"),
):
    global LAST_FILE_PATH, LAST_OPTIMIZED_CSV, LAST_OPTIMIZED_NAME

    try:
        if not file.filename:
            return error_response("No file selected.", 400)

        if not file.filename.lower().endswith(".csv"):
            return csv_extension_error(file.filename)

        size_key = normalize_drone_size(drone_size)
        if size_key is None or size_key not in ALLOWED_DRONE_SIZES:
            return error_response(
                "Invalid drone size. Use 3, 3.5, 4, 5, or 7.",
                400,
                allowed_drone_sizes=sorted(ALLOWED_DRONE_SIZES, key=float),
            )

        goal = normalize_goal(tuning_goal)

        saved_path = save_upload(file)
        LAST_FILE_PATH = saved_path

        parsed = parse_log_with_report(saved_path)
        if parsed.df is None or parsed.df.empty:
            return error_response(
                parsed.report.get("message", "Could not parse CSV."),
                400,
                parser_report=parsed.report,
            )

        df = parsed.df
        LAST_OPTIMIZED_CSV = df.to_csv(index=False)
        LAST_OPTIMIZED_NAME = f"{saved_path.stem}_aerotune_ready.csv"

        validation = validate_log(df)
        analysis = detect_oscillation(df, drone_size=size_key, tuning_goal=goal)

        return {
            "filename": saved_path.name,
            "rows": int(len(df)),
            "columns": list(df.columns),
            "optimized_available": True,
            "optimized_columns": list(df.columns),
            "parser_report": parsed.report,
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
    global LAST_OPTIMIZED_CSV, LAST_OPTIMIZED_NAME

    try:
        if not file.filename:
            return error_response("No file selected.", 400)

        if not file.filename.lower().endswith(".csv"):
            return csv_extension_error(file.filename)

        saved_path = save_upload(file)
        parsed = optimize_csv_file_with_report(saved_path)

        if parsed.df is None or parsed.df.empty:
            return error_response(
                parsed.report.get("message", "Could not optimize CSV."),
                400,
                parser_report=parsed.report,
            )

        df = parsed.df
        LAST_OPTIMIZED_CSV = df.to_csv(index=False)
        LAST_OPTIMIZED_NAME = f"{Path(safe_filename(file.filename)).stem}_aerotune_ready.csv"

        return {
            "ok": True,
            "filename": saved_path.name,
            "download_filename": LAST_OPTIMIZED_NAME,
            "rows": int(len(df)),
            "columns": list(df.columns),
            "optimized_csv": LAST_OPTIMIZED_CSV,
            "parser_report": parsed.report,
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


@app.get("/download-optimized")
def download_optimized():
    if not LAST_OPTIMIZED_CSV:
        return error_response("No optimized CSV available yet. Upload and analyze a log first.", 400)

    return optimized_csv_response(LAST_OPTIMIZED_CSV, LAST_OPTIMIZED_NAME)


@app.get("/plot")
def plot():
    try:
        if LAST_FILE_PATH is None or not LAST_FILE_PATH.exists():
            return error_response("No uploaded file available yet.", 400)

        parsed = parse_log_with_report(LAST_FILE_PATH)
        if parsed.df is None or parsed.df.empty:
            return error_response(
                parsed.report.get("message", "Could not parse the uploaded file for plotting."),
                400,
                parser_report=parsed.report,
            )

        return build_plot_payload(parsed.df)

    except ValueError as exc:
        return error_response(str(exc), 400)

    except Exception as exc:
        return error_response(str(exc), 500)


@app.get("/health")
def health():
    return {"status": "ok"}
