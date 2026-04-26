# ⚡ AeroTune

**AeroTune** is a feel-based FPV drone tuning assistant that turns Betaflight Blackbox CSV logs into clear, pilot-readable PID recommendations.

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

The goal is to test whether Betaflight Blackbox CSV logs can be translated into useful, feel-based tuning recommendations using known FPV PID tuning principles.

AeroTune currently supports Betaflight CSV exports. Native `.bbl` upload support is not added yet because Betaflight Blackbox Explorer can already export logs as CSV.

Some newer Betaflight firmware versions may export CSV files with different column names or structure. If a log does not load, the parser may need to be updated for that format.

## Engineering Goal

AeroTune is being built as a practical engineering project to prove capability in data analysis, control-system thinking, and FPV flight tuning logic.

The project is not intended to replace pilot judgment or official Betaflight documentation. It is meant to turn existing PID theory into a repeatable analysis workflow, then improve that workflow using real before/after flight logs.

The long-term goal is to compare recommendations against real flight results and make the analyzer more accurate over time.


## Notice

AeroTune is an independent open-source FPV Blackbox analysis project created by Christopher Bostrom / bostromdev. Will list collaborators that have sent csv files and I will prove it myself with my drone. Looking forward to helping as many as I can for FREE!

This project is not affiliated with AeroTune7, aerobot2.com, or any similarly named paid tuning tool.
AeroTune is an independent FPV Blackbox analysis project created by Christopher Bostrom / bostromdev. This project is not affiliated with similarly named commercial tools or services.
## Project Origin

AeroTune was created by Christopher Bostrom / bostromdev as an independent FPV Blackbox log analysis tool for Betaflight tuning support.

The project focuses on translating Blackbox CSV data into practical tuning guidance for drone pilots, including noise, tracking, propwash, filter, and PID adjustment feedback.

AeroTune, AeroTune FPV, and the related project materials in this repository are not affiliated with any similarly named commercial tools or third-party services. 
## Why AeroTune Runs Locally

AeroTune is designed as a **local-first tool**.

FPV Blackbox CSV logs can be large, especially when recording longer flights or high-rate gyro data. Running AeroTune locally avoids common web-hosting issues such as upload limits, request timeouts, slow processing, and failed large-file uploads.

Local use also keeps flight logs on the pilot’s own machine instead of forcing uploads to a server.

For this stage of the project, local-first development keeps AeroTune practical, fast, and easier to maintain as a solo-built tool. The focus is on improving the analyzer logic, CSV optimizer, and tuning recommendations before building a more complex hosted interface.

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

AeroTune works best when each CSV comes from a repeatable Blackbox test flight that gives the analyzer enough useful movement to compare gyro behavior, setpoint tracking, throttle response, propwash recovery, vibration, and general tune feel.

Recommended flight:

- Record about **60–90 seconds** of clean Blackbox data.
- 90 seconds is ideal when possible.
- Use the same basic flight style every time you test a tune change.
- Include smooth normal flying, not just hovering.
- Add a few controlled throttle punches.
- Add medium turns, quick stops, and direction changes.
- Add some dirty-air recovery / propwash moments if safe.
- Keep the flight controlled and repeatable.
- Avoid crashes, bumps, heavy wind, or damaged props during test logs.

When comparing two tune changes, repeat the same basic flight pattern for both CSV files.

This is important. If one CSV is only smooth cruising and the next CSV has hard throttle punches, AeroTune may be comparing different flying conditions instead of the actual tune change.

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

This gives AeroTune a better chance to identify real issues like propwash, bounceback, weak hold, poor tracking, high-throttle oscillation, or excess vibration.

---

## How to Export Your Blackbox CSV

AeroTune analyzes **CSV files**, not raw `.BBL`, `.BFL`, or `.TXT` Blackbox logs directly.

The easiest beginner workflow is:

1. Open **Betaflight Blackbox Explorer**.
2. Click **Open log file/video**.
3. Select your Blackbox log file from your flight controller, SD card, or computer.
4. Pick the correct flight/log inside the file if the file contains multiple logs.
5. Use **Export CSV**.
6. Save the exported `.csv` file somewhere easy to find.
7. Upload that `.csv` into AeroTune.

Recommended naming style:

```text
7inch_6s_90s_smooth_punches_before.csv
7inch_6s_90s_smooth_punches_after.csv
```

Keep the original Blackbox log too. The CSV is for AeroTune, but the raw log is still useful if you need to reopen it in Blackbox Explorer later.

Best CSV export checklist:

- Export the same flight you want AeroTune to analyze.
- Prefer a 60–90 second test flight.
- Avoid exporting crash-only logs unless you are diagnosing a crash.
- Name the file clearly so you know the drone size, battery, test type, and whether it was before or after a tune change.
- If the exported CSV is too large for a hosted demo, run AeroTune locally.

Advanced users can also use command-line Blackbox tools to decode supported logs into CSV, but beginners should start with Betaflight Blackbox Explorer.

---

## Features

- Upload FPV Blackbox CSV logs
- Analyze roll, pitch, and yaw independently
- Detect common tuning problems
- Get simple PID direction changes instead of fake final PID numbers
- Built-in CSV optimizer for messy or oversized logs
- Pilot-focused recommendations with confidence reasons
- Clean local web UI
- Drone size profiles for 3", 3.5", 4", 5", and 7" builds
- Local-first analysis for privacy and large CSV support
- Conservative PID percentage-change recommendations
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

## CSV Optimizer

The built-in optimizer converts compatible logs into a standard AeroTune-ready format:

```text
time, gyro_x, gyro_y, gyro_z, setpoint_roll, setpoint_pitch, setpoint_yaw, throttle
```

This helps keep analysis consistent across logs with different column names or large exported files.
Note: AeroTune currently expects CSV exports, not raw `.bbl` files. To use a Betaflight Blackbox log, open it in Betaflight Blackbox Explorer and export the log as CSV first.
---

## Why AeroTune Exists

FPV tuning is hard because raw data does not always explain flight feel.

AeroTune bridges the gap between:

```text
Blackbox data → pilot intuition → safe tuning decision
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
│   ├── parser.py
│   ├── main.py
│   └── log_validator.py
├── static/
│   └── index.html
├── assets/
│   └── screenshots/
├── sources.md
├── RELEASE_NOTES.md
├── requirements.txt
└── README.md
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

AeroTune is released under the Apache License 2.0.

You are free to use, modify, fork, study, and build from this project. If you publish or redistribute work based on AeroTune, please keep the license notice and give credit where reasonable:

```text
Based on AeroTune by Christopher Bostrom
```

The goal is to encourage FPV pilots, developers, and researchers to use and improve the project while keeping clear attribution to the original creator.

---

## Disclaimer

AeroTune gives conservative tuning recommendations, not guaranteed final PID values.

Always:

- Make small changes
- Test one change at a time
- Check motor temperature after D-term changes
- Fix mechanical vibration before tuning around it

---

## Roadmap

Planned future improvements:

- Before / after tune records
- Flight-feel prediction
- Community log examples
* Native `.bbl` support
* Better support for newer Betaflight CSV formats
* More flexible column detection
* Real before/after log validation
* Public test-log dataset
---

## Author

Built by **Christopher Bostrom**

GitHub: [bostromdev](https://github.com/bostromdev)

---

## Summary

## Summary

AeroTune is a local-first Betaflight Blackbox CSV analyzer for feel-based PID tuning.

It is an engineering project focused on turning real flight-log data into clear, conservative tuning suggestions, then validating those suggestions through before/after testing.
