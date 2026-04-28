# ⚡ AeroTune

**AeroTune** is a local-first FPV drone tuning assistant that turns Betaflight Blackbox logs into clear, pilot-readable PID recommendations.

Instead of overwhelming pilots with raw graphs and confusing numbers, AeroTune translates flight-log behavior into simple tuning decisions:

```text
What is wrong?
What PID term should change?
Why should it change?
What should I test next?
```

---

## Current Status

AeroTune is an early engineering project, not a finished commercial PID tuner.

The goal is to test whether Betaflight Blackbox logs can be translated into useful, feel-based tuning recommendations using known FPV PID tuning principles.

AeroTune supports **Betaflight CSV exports** directly. V1.3 adds raw `.bbl`, `.bfl`, and `.txt` upload support by using a locally installed `blackbox_decode` converter. When a raw Blackbox file is uploaded, AeroTune automatically converts it into a CSV first, then sends that converted CSV through the same parser/analyzer pipeline.

V1.4 adds before/after multi-log comparison so pilots can check whether a tune change improved or worsened the flight data.

V1.5 adds tune-change tracking so pilots can record what changed, compare the after log against the before log, and decide whether to keep, reduce, revert, or retest the change.

V1.5.1 adds saved tune-change reports with JSON and Markdown downloads.

Some newer Betaflight firmware versions may export CSV files with different column names or structure. AeroTune V1.1 / V1.2 focuses on making the parser stronger and showing clear parser diagnostics instead of failing silently.

---

## Engineering Goal

AeroTune is being built as a practical engineering project to prove capability in:

- Data analysis
- Control-system thinking
- FPV flight tuning logic
- Parser design for real-world exported logs
- Before/after validation using real flight data

The project is not intended to replace pilot judgment or official Betaflight documentation. It is meant to turn existing PID theory into a repeatable analysis workflow, then improve that workflow using real before/after flight logs.

The long-term goal is to compare recommendations against real flight results and make the analyzer more accurate over time.

---

## Safe Tuning Workflow: Clean Baseline First, Style Second

AeroTune is intentionally engineering-focused, not a magic auto-tune button.

The safest workflow is:

```text
1. Get a clean baseline log first.
2. Use Efficient / Smooth while fixing problems.
3. Remove noise, propwash, bounceback, weak hold, and poor tracking.
4. Confirm the improvement with another similar log.
5. Only after the log is clean, choose Locked-In or Cinematic to shape the feel.
```

This matters because a drone can look different from one flight to the next because of voltage sag, wind, battery weight, prop condition, motor temperature, dirty air, or mechanical vibration. AeroTune should not chase a style preference while the baseline tune is still unsafe or noisy.

### Tune Goal Meanings

**Efficient / Smooth** is the baseline cleanup mode. Start here. Use it to get the quad safe, clean, and predictable.

**Locked-In / Responsive** is a style mode. Use it only after AeroTune detects a clean baseline. It may suggest tiny P/FF changes for sharper stick connection.

**Floaty / Cinematic** is a style mode. Use it only after AeroTune detects a clean baseline. It may suggest tiny P/FF reductions for smoother camera movement.

If a user selects Locked-In or Cinematic before the log is clean, AeroTune will hold the analysis in Efficient / Smooth cleanup mode. The app will explain that style tuning is blocked until the baseline log is clean.

```text
Dirty / noisy log -> baseline cleanup only
Clean log -> style tuning allowed
```

The goal is controlled engineering validation:

```text
Fly -> log -> analyze -> make one small change -> fly again -> compare before/after
```

---

## Notice

AeroTune is an independent source-available FPV Blackbox analysis project created by **Christopher Bostrom / bostromdev**.

This project is not affiliated with AeroTune7, aerobot2.com, or any similarly named paid tuning tool. AeroTune is being built as a free-to-run local evaluation project for FPV pilots, developers, and researchers. Collaborators and testers who provide useful CSV logs may be credited in the project as the validation dataset grows.

---

## Why AeroTune Runs Locally

AeroTune is designed as a **local-first tool**.

FPV Blackbox CSV logs can be large, especially when recording longer flights or high-rate gyro data. Running AeroTune locally avoids common web-hosting issues such as:

- Upload limits
- Request timeouts
- Slow processing
- Failed large-file uploads
- Hosted demo file-size restrictions

Local use also keeps flight logs on the pilot's own machine instead of forcing uploads to a server.

---

## Dashboard

![AeroTune Dashboard](assets/screenshots/dashboard.png)

## Axis Recommendations

![Axis Recommendations](assets/screenshots/axis-recommendations.png)

## CSV Optimizer

![CSV Optimizer](assets/screenshots/csv-optimizer.png)

---

## Recommended Blackbox Test Flight

For the best AeroTune results, do **not** upload a random hover-only log.

AeroTune works best when each CSV comes from a repeatable Blackbox test flight that gives the analyzer enough useful movement to compare:

- Gyro behavior
- Setpoint tracking
- Throttle response
- Propwash recovery
- Vibration
- General tune feel

Recommended flight:

- Record about **60-90 seconds** of clean Blackbox data.
- 90 seconds is ideal when possible.
- Use the same basic flight style every time you test a tune change.
- Include smooth normal flying, not just hovering.
- Add a few controlled throttle punches.
- Add medium turns, quick stops, and direction changes.
- Add some dirty-air recovery / propwash moments if safe.
- Keep the flight controlled and repeatable.
- Avoid crashes, bumps, heavy wind, or damaged props during test logs.

When comparing two tune changes, repeat the same basic flight pattern for both CSV files.

Best practice:

```text
Same drone
Same props
Same battery type
Same tune goal
Same Blackbox settings
Same approximate flight length
Same style of test flight
```

Example test log:

```text
90-second Blackbox log:
- smooth cruise
- small roll/pitch/yaw inputs
- a few throttle punches
- a few turns
- one or two propwash recovery moments
```

---

## How to Export Your Blackbox CSV

AeroTune analyzes **CSV files directly**. It can also accept raw `.BBL`, `.BFL`, or `.TXT` Blackbox logs when Betaflight's `blackbox_decode` tool is installed locally.

The easiest beginner workflow is still CSV export:

1. Open **Betaflight Blackbox Explorer**.
2. Click **Open log file/video**.
3. Select your Blackbox log file from your flight controller, SD card, or computer.
4. Pick the correct flight/log inside the file if the file contains multiple logs.
5. Use **Export CSV**.
6. Save the exported `.csv` file somewhere easy to find.
7. Upload that `.csv` into AeroTune.

Raw-log workflow:

1. Build or install `blackbox_decode` locally.
2. Upload the raw `.bbl`, `.bfl`, or `.txt` file directly.
3. AeroTune converts it to CSV internally, then runs the normal parser/analyzer.

Recommended naming style:

```text
7inch_6s_90s_smooth_punches_before.csv
7inch_6s_90s_smooth_punches_after.csv
```

Keep the original Blackbox log too. The CSV is for AeroTune, but the raw log is still useful if you need to reopen it in Blackbox Explorer later.

---

## Features

- Upload FPV Blackbox CSV logs
- Upload raw `.bbl`, `.bfl`, and `.txt` logs when `blackbox_decode` is installed locally
- Analyze roll, pitch, and yaw independently
- Detect common tuning problems
- Get simple PID direction changes instead of fake final PID numbers
- Built-in CSV optimizer for messy or oversized logs
- Parser diagnostic report for supported CSV files
- Pilot-focused recommendations with confidence reasons
- Clean local web UI
- Drone size profiles for 3", 3.5", 4", 5", and 7" builds
- Local-first analysis for privacy and large CSV support
- Conservative PID percentage-change recommendations
- Safe workflow gate: clean baseline first, style tuning second
- V1.4 before/after log comparison for tune validation
- V1.5 tune-change tracking for validating specific tune changes
- V1.5.1 saved tune-change reports with JSON/Markdown downloads

---

## V1.1 / V1.2 Parser Diagnostic Update

AeroTune V1.1 / V1.2 improves the parser so users can understand exactly what happened when a CSV loads or fails.

Added:

- Parser returns `parser_report` instead of failing silently
- UI shows detected time column
- UI shows detected gyro columns
- UI shows detected setpoint columns
- UI shows detected throttle column
- UI shows missing required columns
- UI shows missing optional columns
- UI shows sample rate
- UI shows duration
- UI shows usable rows
- UI shows repeated header / metadata cleanup
- Backend `/upload-log` returns `parser_report`
- Backend `/optimize-log` returns `parser_report`
- Optimizer now downloads CSV after returning JSON report
- Raw `.bbl`, `.bfl`, and `.txt` files now route through the V1.3 converter when `blackbox_decode` is available

---

## V1.3 Raw Blackbox Log Converter

AeroTune V1.3 adds raw Blackbox upload support through a local converter.

Supported upload types:

```text
.csv
.bbl
.bfl
.txt
```

CSV files are analyzed directly. Raw `.bbl`, `.bfl`, and `.txt` files are **not analyzed directly as binary/raw logs**. When one of those files is uploaded, AeroTune uses Betaflight's `blackbox_decode` tool to create a CSV export first.

In plain language:

```text
Upload .CSV -> AeroTune parses/analyzes it directly
Upload .BBL -> AeroTune converts it to CSV -> then analyzes the CSV
Upload .BFL -> AeroTune converts it to CSV -> then analyzes the CSV
Upload .TXT -> AeroTune converts it to CSV -> then analyzes the CSV
```

The user does not have to manually export CSV first when `blackbox_decode` is installed and detected. AeroTune does **not** bundle `blackbox_decode` inside this repository.

The converter must be installed locally or pointed to with an environment variable:

```bash
export BLACKBOX_DECODE_PATH="$PWD/tools/blackbox-tools/obj/blackbox_decode"
```

AeroTune automatically searches for `blackbox_decode` in this order:

```text
1. BLACKBOX_DECODE_PATH
2. tools/blackbox-tools/obj/blackbox_decode
3. system PATH
```

Why the converter is separate:

- `blackbox_decode` is part of Betaflight / blackbox-tools.
- AeroTune uses `blackbox_decode` only as an optional local converter for raw `.bbl`, `.bfl`, and `.txt` logs.
- AeroTune does not bundle or redistribute the `blackbox_decode` binary.
- CSV exports remain the safest fallback for all users.

---

## V1.4 Multi-Log Comparison

AeroTune V1.4 adds a before/after comparison workflow.

Upload:

```text
Before tune log
After tune log
```

AeroTune analyzes both logs using the same parser, converter, drone-size profile, and tuning goal.

It then compares the after log against the before log and answers:

```text
Did it improve?
Did it get worse?
Which axis changed most?
What metric changed most?
What should I test next?
```

V1.4 compares lower-is-better metrics:

- Tracking error ratio
- 95th-percentile absolute error
- Propwash-band energy
- High-frequency noise ratio
- Absolute tracking lag

Important: before/after comparison only works well when both logs come from similar flights.

---

## V1.5 Tune-Change Tracking

AeroTune V1.5 adds tune-change tracking. The goal is to answer a more useful question than "what should I change?":

```text
I changed something.
Did the log actually improve?
Should I keep it, reduce it, revert it, or retest?
```

Workflow:

```text
1. Fly a clean, repeatable before log.
2. Make one clear PID/filter/rate change.
3. Fly the same test route again.
4. Upload the before log.
5. Upload the after log.
6. Write what changed.
7. AeroTune compares the logs and tracks whether the change helped or hurt.
```

V1.5 checks:

- Overall before/after score
- Noise improvement
- Tracking improvement
- Propwash improvement
- Axis-specific verdicts
- Whether the after log is clean enough for style tuning
- Whether the change should be kept, reduced, reverted, or retested

Important: V1.5 is **not automatic PID rewriting**. It does not change values for the pilot. It records the change, compares the evidence, and gives a conservative decision.

---

## V1.5.1 Saved Tune-Change Reports

AeroTune V1.5.1 saves every tune-change tracking result as both JSON and Markdown.

When the V1.5 form is submitted, AeroTune now:

- Saves the full tune-change result as JSON
- Saves a short human-readable Markdown summary
- Creates a unique report ID
- Stores reports in `reports/tune_changes/`
- Returns report download links to the UI
- Adds download buttons for JSON and Markdown

Saved report files include:

- Before filename
- After filename
- Tune-change notes
- Overall verdict
- Improvement score
- Confidence
- Noise improvement
- Tracking improvement
- Propwash improvement
- Axis verdicts
- Safe-baseline / style-tuning status
- Next step

Generated reports are local working files and are ignored by Git by default so personal flight data is not accidentally committed.

---

## What AeroTune Detects

AeroTune currently identifies:

- Clean tune
- Propwash / bounceback
- High-frequency noise
- Low-frequency wobble / bounce
- Mid-frequency vibration
- High-throttle oscillation
- Weak hold / drift
- Poor tracking
- Slow stick response

Example output:

```text
ROLL: Propwash detected
Change: D up slightly, P down slightly
Why: D helps damp dirty-air recovery, but motor heat must be checked.
```

---

## Converter / CSV Optimizer

The built-in converter/optimizer has two jobs:

1. If the upload is raw `.bbl`, `.bfl`, or `.txt`, AeroTune uses `blackbox_decode` to convert it into CSV.
2. Once the file is CSV, AeroTune cleans it into a standard AeroTune-ready format:

```text
time, gyro_x, gyro_y, gyro_z, setpoint_roll, setpoint_pitch, setpoint_yaw, throttle
```

This helps keep analysis consistent across logs with different column names or large exported files.

CSV exports still work without any extra tool. Raw `.bbl`, `.bfl`, and `.txt` Blackbox logs require `blackbox_decode` to be installed locally or available through `BLACKBOX_DECODE_PATH`.

---

## Why AeroTune Exists

FPV tuning is hard because raw data does not always explain flight feel.

AeroTune bridges the gap between:

```text
Blackbox data -> pilot intuition -> safe tuning decision
```

It is designed for pilots building or tuning drones with mixed parts, DIY frames, different prop sizes, noisy motors, imperfect filtering, or custom freestyle/cinematic goals.

---

## Tuning Modes

### Efficient / Smooth

Conservative tuning for smooth flight, lower heat risk, and stable behavior.

### Locked-In / Responsive

Sharper response and tighter stick feel.

### Floaty / Cinematic

Softer movement for smoother cinematic flying.

---

## Local Setup

Clone the repo:

```bash
git clone https://github.com/bostromdev/AeroTune.git
cd AeroTune
```

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install python-multipart
```

Run locally:

```bash
python3 -m uvicorn main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

---

## Tech Stack

- Python
- FastAPI
- Pandas
- NumPy
- HTML
- CSS
- JavaScript
- FPV Blackbox CSV exports

---

## Project Structure

```text
AeroTune/
├── app/
│   ├── analyzer.py
│   ├── comparison.py
│   ├── converter.py
│   ├── log_validator.py
│   ├── main.py
│   ├── parser.py
│   ├── report_store.py
│   └── tune_tracking.py
├── assets/
│   └── screenshots/
├── docs/
│   ├── AeroTune_Install_Run_Guide.md
│   ├── AeroTune_Install_Run_Guide.pdf
│   ├── AeroTune_PID_Tuning_Guide.md
│   ├── AeroTune_PID_Tuning_Guide.pdf
│   └── aerotune-ui.png
├── reports/
│   └── tune_changes/
│       └── .gitkeep
├── sample_logs/
├── static/
│   ├── index.html
│   └── favicon.svg
├── tools/
├── uploads/
├── LICENSE
├── NOTICE
├── Procfile
├── README.md
├── RELEASE_NOTES.md
├── main.py
├── requirements.txt
└── sources.md
```

---

## References

AeroTune tuning logic is based on established FPV PID tuning principles:

- P controls tracking and sharpness
- I improves attitude hold
- D adds damping for propwash and bounceback but can increase heat/noise
- Feedforward improves stick response
- Yaw D is normally kept at 0

See:

```text
sources.md
```

---

## License

AeroTune is **source-available, not open-source**.

Copyright © 2026 Christopher Bostrom. All Rights Reserved.

You may view the source code, run an unmodified copy of AeroTune locally for personal non-commercial evaluation, and use it to analyze your own Betaflight Blackbox logs.

You may **not** copy, redistribute, modify, repackage, sell, host, commercialize, fork for public release, publish derivative works, or reuse AeroTune's analyzer logic, parser logic, comparison logic, UI structure, documentation, branding, or project materials without written permission from Christopher Bostrom.

Permission is required before using AeroTune's logic or implementation in another project. For collaboration, licensing, or permission requests, contact **Christopher Bostrom / bostromdev**.

See `LICENSE` and `NOTICE` for the full terms.

---

## Disclaimer

AeroTune gives conservative tuning recommendations, not guaranteed final PID values.

Always:

- Make small changes
- Test one change at a time
- Check motor temperature after D-term changes
- Fix mechanical vibration before tuning around it
- Confirm flight behavior with safe test flights

---

## Roadmap

### V1 — Betaflight CSV export support

Status: mostly complete.

### V1.1 — Stronger parser for unusual Betaflight CSV exports

Status: in progress.

### V1.2 — Auto-detect firmware/export column names

Status: partially implemented.

### V1.3 — Native `.bbl` / `.bfl` / `.txt` upload support

Status: implemented as a local-converter workflow.

### V1.4 — Multi-log comparison

Status: implemented.

### V1.5 — Tune-change tracking

Status: implemented.

Future improvements:

- Saved before / after tune records
- Flight-feel prediction
- Community log examples
- Better support for newer Betaflight CSV formats
- More flexible column detection
- Real before/after log validation
- Public test-log dataset

---

## Author

Built by **Christopher Bostrom**

GitHub: [bostromdev](https://github.com/bostromdev)

---

## Summary

AeroTune is a local-first Betaflight Blackbox analyzer for feel-based PID tuning.

It is an engineering project focused on turning real flight-log data into clear, conservative tuning suggestions, then validating those suggestions through before/after testing.
