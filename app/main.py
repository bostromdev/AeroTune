# Copyright © 2026 Christopher Bostrom. All Rights Reserved.
# Source-available for personal evaluation only. See LICENSE and NOTICE.

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.analyzer import (
    PUBLIC_DRONE_SIZE_OPTIONS,
    detect_oscillation,
    normalize_drone_size,
    normalize_goal,
)
from app.comparison import ComparisonError, build_multilog_comparison
from app.converter import ConverterError, SUPPORTED_UPLOAD_EXTENSIONS, prepare_analysis_file
from app.log_validator import validate_log
from app.parser import optimize_csv_file_with_report, parse_log_with_report


app = FastAPI(title="AeroTune")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OPTIMIZED_DIR = UPLOAD_DIR / "_optimized"
OPTIMIZED_DIR.mkdir(parents=True, exist_ok=True)

LAST_FILE_PATH: Optional[Path] = None
LAST_SOURCE_FILE_PATH: Optional[Path] = None
LAST_OPTIMIZED_PATH: Optional[Path] = None
LAST_OPTIMIZED_CSV: Optional[str] = None
LAST_OPTIMIZED_NAME: str = "aerotune_optimized.csv"

# AeroTune can analyze many sizes. Analyzer bands currently fall back safely for unsupported values.
ALLOWED_DRONE_SIZES = set(PUBLIC_DRONE_SIZE_OPTIONS)

# Local-first tool: large Blackbox CSV/raw files are normal.
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


def unsupported_upload_error(filename: str | None):
    suffix = Path(filename or "").suffix.lower()
    converter_report = {
        "ok": False,
        "converted": False,
        "source_filename": safe_filename(filename),
        "source_file_type": suffix or "unknown",
        "analysis_path": None,
        "analysis_filename": None,
        "tool": "blackbox_decode",
        "tool_path": None,
        "command": [],
        "duration_seconds": None,
        "stdout": "",
        "stderr": "",
        "returncode": None,
        "generated_csv_files": [],
        "message": "Unsupported file type. Upload a Betaflight CSV, .bbl, .bfl, or .txt Blackbox log.",
        "error_code": "unsupported_file_type",
        "warnings": [],
        "suggestions": [
            "Use a Betaflight CSV export for direct analysis.",
            "Use .bbl, .bfl, or .txt only after blackbox_decode is installed locally.",
        ],
    }
    return error_response(converter_report["message"], 400, converter_report=converter_report)


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


def _unique_output_path(filename: str) -> Path:
    OPTIMIZED_DIR.mkdir(parents=True, exist_ok=True)
    path = OPTIMIZED_DIR / safe_filename(filename)
    stem = path.stem
    ext = path.suffix or ".csv"
    suffix = 1

    while path.exists():
        path = OPTIMIZED_DIR / f"{stem}_{suffix}{ext}"
        suffix += 1

    return path


def save_optimized_dataframe(df, filename: str) -> Path:
    path = _unique_output_path(filename)
    df.to_csv(path, index=False)
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
    global LAST_FILE_PATH, LAST_SOURCE_FILE_PATH, LAST_OPTIMIZED_PATH, LAST_OPTIMIZED_CSV, LAST_OPTIMIZED_NAME

    try:
        if not file.filename:
            return error_response("No file selected.", 400)

        if Path(file.filename).suffix.lower() not in SUPPORTED_UPLOAD_EXTENSIONS:
            return unsupported_upload_error(file.filename)

        size_key = normalize_drone_size(drone_size)
        if size_key is None or size_key not in ALLOWED_DRONE_SIZES:
            return error_response(
                "Invalid drone size. Use 3, 3.5, 4, 5, or 7.",
                400,
                allowed_drone_sizes=sorted(ALLOWED_DRONE_SIZES, key=float),
            )

        goal = normalize_goal(tuning_goal)

        saved_path = save_upload(file)
        LAST_SOURCE_FILE_PATH = saved_path

        try:
            analysis_path, converter_report = prepare_analysis_file(saved_path)
        except ConverterError as exc:
            return error_response(str(exc), 400, converter_report=exc.report)

        LAST_FILE_PATH = analysis_path

        parsed = parse_log_with_report(analysis_path)
        if parsed.df is None or parsed.df.empty:
            return error_response(
                parsed.report.get("message", "Could not parse CSV."),
                400,
                converter_report=converter_report,
                parser_report=parsed.report,
            )

        df = parsed.df
        LAST_OPTIMIZED_NAME = f"{saved_path.stem}_aerotune_ready.csv"
        LAST_OPTIMIZED_PATH = save_optimized_dataframe(df, LAST_OPTIMIZED_NAME)
        LAST_OPTIMIZED_CSV = None

        validation = validate_log(df)
        analysis = detect_oscillation(df, drone_size=size_key, tuning_goal=goal)

        return {
            "filename": saved_path.name,
            "source_filename": saved_path.name,
            "source_file_type": saved_path.suffix.lower(),
            "analysis_filename": analysis_path.name,
            "converted": bool(converter_report.get("converted")),
            "download_available": True,
            "download_filename": LAST_OPTIMIZED_PATH.name if LAST_OPTIMIZED_PATH else LAST_OPTIMIZED_NAME,
            "download_url": "/download-optimized",
            "rows": int(len(df)),
            "columns": list(df.columns),
            "optimized_available": True,
            "optimized_columns": list(df.columns),
            "converter_report": converter_report,
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
    global LAST_OPTIMIZED_PATH, LAST_OPTIMIZED_CSV, LAST_OPTIMIZED_NAME

    try:
        if not file.filename:
            return error_response("No file selected.", 400)

        if Path(file.filename).suffix.lower() not in SUPPORTED_UPLOAD_EXTENSIONS:
            return unsupported_upload_error(file.filename)

        saved_path = save_upload(file)

        try:
            analysis_path, converter_report = prepare_analysis_file(saved_path)
        except ConverterError as exc:
            return error_response(str(exc), 400, converter_report=exc.report)

        parsed = optimize_csv_file_with_report(analysis_path)

        if parsed.df is None or parsed.df.empty:
            return error_response(
                parsed.report.get("message", "Could not optimize CSV."),
                400,
                converter_report=converter_report,
                parser_report=parsed.report,
            )

        df = parsed.df
        LAST_OPTIMIZED_NAME = f"{Path(safe_filename(file.filename)).stem}_aerotune_ready.csv"
        LAST_OPTIMIZED_PATH = save_optimized_dataframe(df, LAST_OPTIMIZED_NAME)
        LAST_OPTIMIZED_CSV = None

        return {
            "ok": True,
            "filename": saved_path.name,
            "source_filename": saved_path.name,
            "source_file_type": saved_path.suffix.lower(),
            "analysis_filename": analysis_path.name,
            "converted": bool(converter_report.get("converted")),
            "download_filename": LAST_OPTIMIZED_PATH.name,
            "download_url": "/download-optimized",
            "rows": int(len(df)),
            "columns": list(df.columns),
            "converter_report": converter_report,
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


@app.post("/compare-logs")
async def compare_logs(
    before_file: UploadFile = File(...),
    after_file: UploadFile = File(...),
    drone_size: str = Form("7"),
    tuning_goal: str = Form("efficient"),
):
    """
    V1.4 before/after comparison endpoint.

    Accepts two supported AeroTune inputs:
    - before_file: baseline log before a PID/filter/rate change
    - after_file: follow-up log after the tune change

    Each file can be a direct CSV export or a raw .bbl/.bfl/.txt Blackbox log
    when blackbox_decode is installed locally.
    """
    try:
        if not before_file.filename:
            return error_response("Before log is missing.", 400)
        if not after_file.filename:
            return error_response("After log is missing.", 400)

        if Path(before_file.filename).suffix.lower() not in SUPPORTED_UPLOAD_EXTENSIONS:
            return unsupported_upload_error(before_file.filename)
        if Path(after_file.filename).suffix.lower() not in SUPPORTED_UPLOAD_EXTENSIONS:
            return unsupported_upload_error(after_file.filename)

        size_key = normalize_drone_size(drone_size)
        if size_key is None or size_key not in ALLOWED_DRONE_SIZES:
            return error_response(
                "Invalid drone size. Use 3, 3.5, 4, 5, or 7.",
                400,
                allowed_drone_sizes=sorted(ALLOWED_DRONE_SIZES, key=float),
            )

        goal = normalize_goal(tuning_goal)

        before_saved_path = save_upload(before_file)
        after_saved_path = save_upload(after_file)

        try:
            before_analysis_path, before_converter_report = prepare_analysis_file(before_saved_path)
        except ConverterError as exc:
            return error_response(
                f"Before log conversion failed: {exc}",
                400,
                before={"converter_report": exc.report},
            )

        try:
            after_analysis_path, after_converter_report = prepare_analysis_file(after_saved_path)
        except ConverterError as exc:
            return error_response(
                f"After log conversion failed: {exc}",
                400,
                before={"converter_report": before_converter_report},
                after={"converter_report": exc.report},
            )

        before_parsed = parse_log_with_report(before_analysis_path)
        if before_parsed.df is None or before_parsed.df.empty:
            return error_response(
                before_parsed.report.get("message", "Could not parse before log."),
                400,
                before={
                    "source_filename": before_saved_path.name,
                    "analysis_filename": before_analysis_path.name,
                    "converter_report": before_converter_report,
                    "parser_report": before_parsed.report,
                },
                after={
                    "source_filename": after_saved_path.name,
                    "analysis_filename": after_analysis_path.name,
                    "converter_report": after_converter_report,
                },
            )

        after_parsed = parse_log_with_report(after_analysis_path)
        if after_parsed.df is None or after_parsed.df.empty:
            return error_response(
                after_parsed.report.get("message", "Could not parse after log."),
                400,
                before={
                    "source_filename": before_saved_path.name,
                    "analysis_filename": before_analysis_path.name,
                    "converter_report": before_converter_report,
                    "parser_report": before_parsed.report,
                },
                after={
                    "source_filename": after_saved_path.name,
                    "analysis_filename": after_analysis_path.name,
                    "converter_report": after_converter_report,
                    "parser_report": after_parsed.report,
                },
            )

        before_df = before_parsed.df
        after_df = after_parsed.df

        before_validation = validate_log(before_df)
        after_validation = validate_log(after_df)
        before_analysis = detect_oscillation(before_df, drone_size=size_key, tuning_goal=goal)
        after_analysis = detect_oscillation(after_df, drone_size=size_key, tuning_goal=goal)

        try:
            comparison = build_multilog_comparison(
                before_df=before_df,
                after_df=after_df,
                before_analysis=before_analysis,
                after_analysis=after_analysis,
                drone_size=size_key,
                tuning_goal=goal,
            )
        except ComparisonError as exc:
            return error_response(str(exc), 400)

        return {
            "ok": True,
            "version": "V1.4",
            "message": "Before/after comparison complete.",
            "drone_size": size_key,
            "tuning_goal": goal,
            "before": {
                "source_filename": before_saved_path.name,
                "source_file_type": before_saved_path.suffix.lower(),
                "analysis_filename": before_analysis_path.name,
                "converted": bool(before_converter_report.get("converted")),
                "rows": int(len(before_df)),
                "columns": list(before_df.columns),
                "converter_report": before_converter_report,
                "parser_report": before_parsed.report,
                "validation": before_validation,
                "analysis": before_analysis,
            },
            "after": {
                "source_filename": after_saved_path.name,
                "source_file_type": after_saved_path.suffix.lower(),
                "analysis_filename": after_analysis_path.name,
                "converted": bool(after_converter_report.get("converted")),
                "rows": int(len(after_df)),
                "columns": list(after_df.columns),
                "converter_report": after_converter_report,
                "parser_report": after_parsed.report,
                "validation": after_validation,
                "analysis": after_analysis,
            },
            "comparison": comparison,
        }

    except ValueError as exc:
        return error_response(str(exc), 400)

    except Exception as exc:
        return error_response(str(exc), 500)

    finally:
        for upload in (before_file, after_file):
            try:
                upload.file.close()
            except Exception:
                pass


@app.get("/download-optimized")
def download_optimized():
    if LAST_OPTIMIZED_PATH is not None and LAST_OPTIMIZED_PATH.exists():
        return FileResponse(
            LAST_OPTIMIZED_PATH,
            media_type="text/csv",
            filename=LAST_OPTIMIZED_PATH.name,
        )

    if LAST_OPTIMIZED_CSV:
        return optimized_csv_response(LAST_OPTIMIZED_CSV, LAST_OPTIMIZED_NAME)

    return error_response("No optimized CSV available yet. Upload and analyze a log first.", 400)


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
