# AeroTune Release Notes

## AeroTune v1.5.1 — Saved Tune-Change Reports + Updated Guides

### Major Updates

- Added saved tune-change reports as JSON and Markdown.
- Added report storage under `reports/tune_changes/`.
- Added documentation for the V1.5 tune-change tracking workflow.
- Updated the install/run guide for local-first use, raw-log conversion, saved reports, and privacy notes.
- Updated the PID tuning guide around clean baseline first, style tuning second, one change at a time, and keep/reduce/revert/retest decisions.
- Added a final converter/CSV explanation page to the install guide.
- Updated README project structure so it matches the current app layout.

### Safety Notes

AeroTune still gives conservative tuning recommendations, not guaranteed final PID values. Pilots should make small changes, test one change at a time, check motor temperature after D-term changes, and fix mechanical vibration before tuning around it.

---

## AeroTune v1.5 — Tune-Change Tracking

### Major Updates

- Added V1.5 tune-change tracking.
- Added workflow for uploading before and after logs.
- Added tune-change notes so the pilot can record exactly what was changed.
- Added keep / reduce / revert / retest style decision logic.
- Added grouping around noise improvement, tracking improvement, and propwash improvement.
- Added safe-baseline gate before style tuning.

### Goal

V1.5 is meant to answer:

```text
I changed something.
Did the log actually improve?
Should I keep it, reduce it, revert it, or retest?
```

---

## AeroTune v1.4 — Multi-Log Comparison

### Major Updates

- Added before/after log comparison.
- Added comparison metrics for tracking error, high-frequency noise, propwash-band energy, and lag.
- Added overall improvement score.
- Added comparison confidence.
- Added strongest improvement axis and weakest axis output.
- Added next-step recommendation.

### Best Practice

Comparison works best when both logs come from similar flights:

```text
Same drone
Same props
Same battery type
Same tune goal
Same approximate flight length
Same route or maneuver pattern
```

---

## AeroTune v1.3 — Raw Blackbox Converter

### Major Updates

- Added raw `.bbl`, `.bfl`, and `.txt` upload support through local `blackbox_decode` conversion.
- Added converter report output.
- Kept CSV as the actual analysis format after conversion.
- Kept `blackbox_decode` external instead of bundling it into AeroTune.

### Supported Upload Types

```text
.csv
.bbl
.bfl
.txt
```

CSV files are analyzed directly. Raw Blackbox logs are converted to CSV first, then analyzed.

---

## AeroTune v1.1 / v1.2 — Parser Diagnostic Update

### Major Updates

- Parser returns `parser_report` instead of failing silently.
- UI shows detected time column.
- UI shows detected gyro columns.
- UI shows detected setpoint columns.
- UI shows detected throttle column.
- UI shows missing required and optional columns.
- UI shows sample rate, duration, usable rows, and cleanup information.
- Optimizer returns parser diagnostics.
- Optimizer downloads standardized CSV after returning JSON report.

---

## AeroTune v1.02 — Size Profiles + Betaflight PID Value Calculator

### Major Updates

- Added drone size profiles for 3", 3.5", 4", 5", and 7" builds.
- Added backend size handling so the selected drone size affects analyzer behavior.
- Added conservative PID percentage-change recommendations.

---

## AeroTune v1.0 — Initial Public Release

### Major Features

- Smart PID analysis engine.
- Converts Blackbox CSV logs into tuning advice.
- Uses gyro vs. setpoint tracking and residual error analysis.
- Detects clean tune, propwash, bounceback, high-frequency noise, low-frequency wobble, mid-frequency vibration, high-throttle oscillation, weak hold, poor tracking, and slow stick response.
- Built-in CSV optimizer.
- Clean dark telemetry-style interface.
- Tuning modes for Efficient, Locked-In, and Floaty goals.

### Safety

- Conservative PID changes.
- Yaw D normally kept at 0.
- Avoids unsafe D increases during noisy-log conditions.
