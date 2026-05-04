# Copyright © 2026 Christopher Bostrom. All Rights Reserved.
# Source-available for personal evaluation only. See LICENSE and NOTICE.

from __future__ import annotations

import os
import re
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
        "conversion_id": None,
        "flight_count": 0,
        "selected_flight_index": None,
        "selected_flight_label": None,
        "available_flights": [],
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


CONVERSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{4,120}$")


def _natural_sort_key(path: Path) -> Tuple[Any, ...]:
    """
    Sort Blackbox Decode CSV outputs in flight order.

    blackbox_decode commonly emits one CSV per flight with numbered suffixes.
    The last numbered CSV is treated as the newest/default flight, matching the
    way Blackbox Explorer lists multi-flight logs as Flight 1/N ... Flight N/N.
    """
    name = path.name.lower()
    parts = re.split(r"(\d+)", name)
    key: List[Any] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part)
    return tuple(key)


def _generated_csv_files(directory: Path) -> List[Path]:
    """Return generated CSV files in oldest -> newest flight order."""
    return sorted(
        [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".csv"],
        key=_natural_sort_key,
    )


def _flight_payload(csv_files: List[Path], selected_flight_index: int) -> List[Dict[str, Any]]:
    total = len(csv_files)
    flights: List[Dict[str, Any]] = []

    for idx, path in enumerate(csv_files, start=1):
        flights.append(
            {
                "flight_index": idx,
                "flight_count": total,
                "label": f"Flight {idx}/{total}" + (" — latest" if idx == total else ""),
                "filename": path.name,
                "size_bytes": int(path.stat().st_size),
                "is_latest": idx == total,
                "is_selected": idx == selected_flight_index,
            }
        )

    return flights


def _safe_conversion_path(conversion_id: str) -> Path:
    raw = str(conversion_id or "").strip()
    if not CONVERSION_ID_PATTERN.match(raw) or ".." in raw or "/" in raw or "\\" in raw:
        raise ConverterError(
            "Invalid raw Blackbox conversion ID.",
            {
                "ok": False,
                "converted": True,
                "message": "Invalid raw Blackbox conversion ID.",
                "error_code": "invalid_conversion_id",
                "warnings": [],
                "suggestions": ["Upload the raw Blackbox log again."],
            },
        )

    root = CONVERSION_ROOT.resolve()
    path = (CONVERSION_ROOT / raw).resolve()

    if root not in path.parents and path != root:
        raise ConverterError(
            "Invalid raw Blackbox conversion location.",
            {
                "ok": False,
                "converted": True,
                "message": "Invalid raw Blackbox conversion location.",
                "error_code": "invalid_conversion_path",
                "warnings": [],
                "suggestions": ["Upload the raw Blackbox log again."],
            },
        )

    if not path.exists() or not path.is_dir():
        raise ConverterError(
            "Converted raw Blackbox flights are no longer available.",
            {
                "ok": False,
                "converted": True,
                "message": "Converted raw Blackbox flights are no longer available.",
                "error_code": "conversion_not_found",
                "warnings": [],
                "suggestions": ["Upload the raw Blackbox log again."],
            },
        )

    return path


def get_converted_flight_path(conversion_id: str, flight_index: int) -> Tuple[Path, Dict[str, Any]]:
    """
    Return the selected decoded CSV for a raw multi-flight conversion.

    Flight indexes are 1-based: Flight 1/N is oldest, Flight N/N is latest.
    """
    work_dir = _safe_conversion_path(conversion_id)
    csv_files = _generated_csv_files(work_dir)

    if not csv_files:
        raise ConverterError(
            "No converted CSV flights were found for this raw Blackbox log.",
            {
                "ok": False,
                "converted": True,
                "conversion_id": conversion_id,
                "message": "No converted CSV flights were found for this raw Blackbox log.",
                "error_code": "converted_flights_missing",
                "warnings": [],
                "suggestions": ["Upload the raw Blackbox log again."],
            },
        )

    try:
        selected_index = int(flight_index)
    except (TypeError, ValueError):
        selected_index = len(csv_files)

    if selected_index < 1 or selected_index > len(csv_files):
        selected_index = len(csv_files)

    selected_csv = csv_files[selected_index - 1]
    report = _base_report(selected_csv)
    report.update(
        {
            "ok": True,
            "converted": True,
            "conversion_id": work_dir.name,
            "flight_count": len(csv_files),
            "selected_flight_index": selected_index,
            "selected_flight_label": f"Flight {selected_index}/{len(csv_files)}" + (" — latest" if selected_index == len(csv_files) else ""),
            "available_flights": _flight_payload(csv_files, selected_index),
            "generated_csv_files": [p.name for p in csv_files],
            "analysis_path": str(selected_csv),
            "analysis_filename": selected_csv.name,
            "message": f"Selected raw Blackbox Flight {selected_index}/{len(csv_files)} for analysis.",
        }
    )
    return selected_csv, report


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

    # V1.7 behavior: Blackbox files may contain multiple flights.
    # Keep all decoded CSVs and select the newest/default flight as Flight N/N.
    selected_flight_index = len(csv_files)
    selected_csv = csv_files[selected_flight_index - 1]
    report["conversion_id"] = work_dir.name
    report["flight_count"] = len(csv_files)
    report["selected_flight_index"] = selected_flight_index
    report["selected_flight_label"] = f"Flight {selected_flight_index}/{len(csv_files)}" + (" — latest" if len(csv_files) > 1 else "")
    report["available_flights"] = _flight_payload(csv_files, selected_flight_index)

    if len(csv_files) > 1:
        report["warnings"].append(
            f"blackbox_decode generated {len(csv_files)} flight CSVs. AeroTune selected Flight {selected_flight_index}/{len(csv_files)} as the latest/default flight."
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
            "message": f"Raw Blackbox log converted successfully. Selected Flight {selected_flight_index}/{len(csv_files)} for analysis.",
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
