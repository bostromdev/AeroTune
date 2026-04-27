from __future__ import annotations

import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CSV_EXTENSIONS = {".csv"}
RAW_BLACKBOX_EXTENSIONS = {".bbl", ".bfl", ".txt"}
SUPPORTED_UPLOAD_EXTENSIONS = CSV_EXTENSIONS | RAW_BLACKBOX_EXTENSIONS

CONVERSION_ROOT = Path("uploads") / "_converted"
DEFAULT_CONVERTER_TIMEOUT_SECONDS = 120
MAX_CAPTURED_OUTPUT_CHARS = 8000


class ConverterError(Exception):
    """Raised when raw Blackbox conversion fails with a user-safe report."""

    def __init__(self, message: str, report: Dict[str, Any]):
        super().__init__(message)
        self.report = report


def _clip_text(value: str | bytes | None, limit: int = MAX_CAPTURED_OUTPUT_CHARS) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    value = str(value)
    if len(value) <= limit:
        return value
    return value[:limit] + "\n...[truncated]"


def _base_report(source_path: str | Path) -> Dict[str, Any]:
    path = Path(source_path)
    return {
        "ok": False,
        "converted": False,
        "source_filename": path.name,
        "source_file_type": path.suffix.lower() or "unknown",
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
        "message": "Raw Blackbox conversion has not started.",
        "error_code": None,
        "warnings": [],
        "suggestions": [],
    }


def _is_executable(path: Path) -> bool:
    return path.exists() and path.is_file() and os.access(path, os.X_OK)


def find_blackbox_decode() -> Optional[Path]:
    """
    Locate blackbox_decode without bundling it into AeroTune.

    Search order:
    1. BLACKBOX_DECODE_PATH environment variable
    2. local tools/blackbox-tools/obj/blackbox_decode
    3. PATH lookup
    """
    env_path = os.environ.get("BLACKBOX_DECODE_PATH", "").strip()
    if env_path:
        candidate = Path(env_path).expanduser()
        if _is_executable(candidate):
            return candidate.resolve()

    local_candidate = Path.cwd() / "tools" / "blackbox-tools" / "obj" / "blackbox_decode"
    if _is_executable(local_candidate):
        return local_candidate.resolve()

    path_candidate = shutil.which("blackbox_decode")
    if path_candidate:
        candidate = Path(path_candidate)
        if _is_executable(candidate):
            return candidate.resolve()

    return None


def csv_passthrough_report(csv_path: str | Path) -> Dict[str, Any]:
    path = Path(csv_path)
    report = _base_report(path)
    report.update(
        {
            "ok": True,
            "converted": False,
            "analysis_path": str(path),
            "analysis_filename": path.name,
            "message": "CSV input detected. Raw Blackbox conversion was not needed.",
        }
    )
    return report


def unsupported_file_report(source_path: str | Path) -> Dict[str, Any]:
    path = Path(source_path)
    report = _base_report(path)
    report.update(
        {
            "message": "Unsupported file type. Upload a Betaflight CSV, .bbl, .bfl, or .txt Blackbox log.",
            "error_code": "unsupported_file_type",
            "suggestions": [
                "Use a Betaflight CSV export for direct analysis.",
                "Use .bbl, .bfl, or .txt only after blackbox_decode is installed locally.",
            ],
        }
    )
    return report


def _safe_conversion_dir(source_path: Path) -> Path:
    CONVERSION_ROOT.mkdir(parents=True, exist_ok=True)
    stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in source_path.stem) or "blackbox"
    return CONVERSION_ROOT / f"{stem}_{uuid.uuid4().hex[:10]}"


def _generated_csv_files(directory: Path) -> List[Path]:
    return sorted(
        [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".csv"],
        key=lambda p: p.stat().st_size,
        reverse=True,
    )


def convert_raw_blackbox_to_csv(
    source_path: str | Path,
    timeout_seconds: int = DEFAULT_CONVERTER_TIMEOUT_SECONDS,
) -> Tuple[Path, Dict[str, Any]]:
    """Convert .bbl/.bfl/.txt Blackbox logs into a CSV file using blackbox_decode."""
    source = Path(source_path)
    report = _base_report(source)

    if source.suffix.lower() not in RAW_BLACKBOX_EXTENSIONS:
        report = unsupported_file_report(source)
        raise ConverterError(report["message"], report)

    if not source.exists() or not source.is_file():
        report.update(
            {
                "message": "Uploaded Blackbox file could not be found on disk.",
                "error_code": "source_missing",
                "suggestions": ["Try uploading the log again."],
            }
        )
        raise ConverterError(report["message"], report)

    tool_path = find_blackbox_decode()
    if tool_path is None:
        report.update(
            {
                "message": "Raw Blackbox conversion requires blackbox_decode, but AeroTune could not find it.",
                "error_code": "converter_missing",
                "suggestions": [
                    "Build it with: make -C tools/blackbox-tools obj/blackbox_decode",
                    "Or set BLACKBOX_DECODE_PATH to the full path of blackbox_decode.",
                    "CSV exports still work without blackbox_decode.",
                ],
            }
        )
        raise ConverterError(report["message"], report)

    work_dir = _safe_conversion_dir(source)
    work_dir.mkdir(parents=True, exist_ok=True)
    working_input = work_dir / source.name
    shutil.copy2(source, working_input)

    command = [str(tool_path), working_input.name]
    report["tool_path"] = str(tool_path)
    report["command"] = [str(tool_path.name), working_input.name]

    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.monotonic() - started
        report.update(
            {
                "duration_seconds": round(duration, 3),
                "stdout": _clip_text(exc.stdout),
                "stderr": _clip_text(exc.stderr),
                "message": f"blackbox_decode timed out after {timeout_seconds} seconds.",
                "error_code": "converter_timeout",
                "suggestions": [
                    "Try a shorter Blackbox log first.",
                    "Export CSV from Betaflight Blackbox Explorer if this raw log is very large.",
                ],
            }
        )
        raise ConverterError(report["message"], report) from exc

    duration = time.monotonic() - started
    report.update(
        {
            "duration_seconds": round(duration, 3),
            "stdout": _clip_text(completed.stdout),
            "stderr": _clip_text(completed.stderr),
            "returncode": int(completed.returncode),
        }
    )

    if completed.returncode != 0:
        report.update(
            {
                "message": "blackbox_decode failed to convert this raw log.",
                "error_code": "converter_failed",
                "suggestions": [
                    "Confirm the file is a real Betaflight Blackbox log.",
                    "Try opening the same file in Betaflight Blackbox Explorer.",
                    "If Blackbox Explorer can read it, export CSV and upload that CSV.",
                ],
            }
        )
        raise ConverterError(report["message"], report)

    csv_files = _generated_csv_files(work_dir)
    report["generated_csv_files"] = [p.name for p in csv_files]

    if not csv_files:
        report.update(
            {
                "message": "blackbox_decode finished, but no CSV file was generated.",
                "error_code": "converter_no_csv_output",
                "suggestions": [
                    "Try blackbox_decode manually on this file and check its output.",
                    "Export CSV from Betaflight Blackbox Explorer as a fallback.",
                ],
            }
        )
        raise ConverterError(report["message"], report)

    selected_csv = csv_files[0]
    if len(csv_files) > 1:
        report["warnings"].append(
            f"blackbox_decode generated {len(csv_files)} CSV files. AeroTune selected the largest one: {selected_csv.name}."
        )

    gpx_files = [p.name for p in work_dir.iterdir() if p.is_file() and p.suffix.lower() == ".gpx"]
    if gpx_files:
        report["warnings"].append("GPS GPX output was generated but is not used by AeroTune yet.")

    report.update(
        {
            "ok": True,
            "converted": True,
            "analysis_path": str(selected_csv),
            "analysis_filename": selected_csv.name,
            "message": "Raw Blackbox log converted to CSV successfully.",
        }
    )
    return selected_csv, report


def prepare_analysis_file(source_path: str | Path) -> Tuple[Path, Dict[str, Any]]:
    """Return a parseable CSV path and a converter report for CSV or raw Blackbox uploads."""
    source = Path(source_path)
    ext = source.suffix.lower()

    if ext in CSV_EXTENSIONS:
        return source, csv_passthrough_report(source)

    if ext in RAW_BLACKBOX_EXTENSIONS:
        return convert_raw_blackbox_to_csv(source)

    report = unsupported_file_report(source)
    raise ConverterError(report["message"], report)
