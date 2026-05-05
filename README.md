# AeroTune

**FPV Blackbox analysis and PID tuning assistant for Betaflight logs.**

AeroTune reads Betaflight Blackbox flight-log data and turns gyro/setpoint behavior into conservative tuning guidance. It is built to help pilots understand symptoms like overshoot, bounceback, propwash, tracking error, and noisy D-term behavior without pretending to generate a perfect automatic tune.

## Current Best Workflow

AeroTune supports two practical workflows:

1. **CSV direct upload** - works without extra tools.
2. **Raw Blackbox upload** - works locally after `blackbox_decode` is installed or included at `tools/blackbox-tools/obj/blackbox_decode`.

For beginners, CSV is still the safest fallback because Betaflight Blackbox Explorer lets you manually choose the exact flight before exporting.

For users who have raw conversion set up, `.BBL`, `.BFL`, and `.TXT` logs can be uploaded directly. AeroTune decodes the raw log locally and analyzes the selected decoded flight.

## V1.7 Raw Blackbox Multi-Flight Selector

AeroTune V1.7 adds a raw Blackbox multi-flight selector.

When a raw `.BBL`, `.BFL`, or `.TXT` file contains more than one decoded flight, AeroTune creates one selectable profile per decoded flight:

```text
Flight 1/N = oldest detected flight
Flight N/N = newest/latest detected flight
```

The newest/latest flight is selected by default. For example, if a raw Blackbox file contains seven flights, AeroTune treats `Flight 7/7` as the newest/default flight unless the pilot chooses another one.

This means you do **not** need to manually export CSV when raw conversion is set up correctly. CSV is still useful if raw conversion is missing, fails, or if you prefer manually choosing the flight in Betaflight Blackbox Explorer first.

## Raw Blackbox Requirements

Raw upload support depends on Betaflight `blackbox_decode`. AeroTune does not replace that decoder; it calls a local copy and analyzes the decoded CSV output.

AeroTune searches for `blackbox_decode` in this order:

```text
1. BLACKBOX_DECODE_PATH
2. tools/blackbox-tools/obj/blackbox_decode
3. system PATH
```

If you download AeroTune with the local `tools/blackbox-tools/obj/blackbox_decode` binary present, raw `.BBL` support can work immediately from the project folder. If the binary is missing, follow the setup instructions below or export CSV from Betaflight Blackbox Explorer.

## V1.8 Pilot Flight Feel + Filtering Advisor

AeroTune now lets the pilot select what the flight actually felt like during upload. These checkbox notes are combined with the Blackbox log to build a safer real-world plan.

Examples of pilot-feel inputs:

- Felt good / motors stayed cool
- Loose, twitchy, delayed, propwash, bounceback, low-throttle wobble
- Motors hot, buzzy, one motor sounded different, video jello
- Dynamic Notch enabled/disabled
- RPM filter caused issues or arming/failsafe warnings
- PIDs, filters, props, or crash changed before the flight

Filtering policy:

```text
Dynamic Notch = default AeroTune recommendation path
RPM filtering = advanced-only, suppressed when the pilot reports RPM-filter, arming, or failsafe trouble
Hot/buzzy motors = block positive D/D Max increases
Good flight + cool motors + no active tuning symptom = protect the tune and save as baseline
Active symptoms like loose/floaty, bounceback, propwash, twitchy feel, or weak tracking = bypass baseline hold and build a targeted plan
```

### Why this was added

A Blackbox log can show frequency energy and tracking error, but it cannot always tell whether the quad felt good, motors were hot, or RPM filtering caused reliability issues. The pilot-feel layer prevents blind tuning advice. It helps AeroTune say “hold this good baseline” instead of chasing a perfect graph.

## Main Features

- Single-log Blackbox analysis
- PID Tuning Advice card
- Pilot Flight Feel upload options
- Filtering Advisor with Dynamic Notch default logic
- RPM-filter suppression when reliability issues are reported
- Size-aware D-term safety gates
- Betaflight PID value calculator
- Roll / pitch / yaw evidence display
- Conservative percentage-based PID recommendations
- Tune-change tracking for before/after validation
- Saved tune-change reports
- Local-first workflow for large logs and privacy

## What AeroTune Looks For

AeroTune focuses on real tuning symptoms:

- Overshoot
- Bounceback
- Propwash recovery behavior
- Tracking error
- High-frequency noise
- Mid-frequency vibration
- Weak hold / drift tendency
- Stick-response behavior

## What AeroTune Does Not Do

AeroTune does not auto-tune your quad, flash Betaflight, or guarantee perfect PID values. It gives conservative evidence-based direction. The pilot still makes the final tuning decision.

## Run Locally - macOS / Linux

```bash
git clone https://github.com/bostromdev/AeroTune.git
cd AeroTune
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

## Run Locally - Windows PowerShell

```powershell
git clone https://github.com/bostromdev/AeroTune.git
cd AeroTune
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -3 -m pip install --upgrade pip
py -3 -m pip install -r requirements.txt
py -3 -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Windows: `uvicorn` Command Not Found

If Windows says `uvicorn` is not recognized even though pip says it is installed, run it through Python:

```powershell
py -3 -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

This avoids PATH problems.

## Optional Raw Converter Setup

CSV export is recommended for beginners. Raw `.BBL`, `.BFL`, and `.TXT` upload requires `blackbox_decode`.

AeroTune searches for the converter in this order:

```text
1. BLACKBOX_DECODE_PATH
2. tools/blackbox-tools/obj/blackbox_decode
3. system PATH
```

If the converter is not found, use Betaflight Blackbox Explorer and export CSV.

## Tuning Safety

- Make one change at a time.
- Check motor temperature after D or D Max changes.
- Do not tune around loose props, loose camera mounts, frame vibration, or damaged motors.
- Keep yaw D at normal Betaflight-safe defaults unless you know exactly why you are changing it.
- Use a safe test area with enough room.

## Creator

Built by **BostromDev**.

YouTube: https://www.youtube.com/@BostromDev


## Pilot Feel: Good Baseline Hold

If a flight felt good and motors stayed cool, select **Felt good / no major issue**, **Motors stayed cool**, and optionally **Use this as a good baseline / hold current tune** during upload. AeroTune will keep PID deltas at 0% for that report and ask for another comparable log before recommending a correction. This prevents one hard Acro maneuver from being mistaken for a tune that needs aggressive D/D Max changes.

If you also select an active symptom such as **Felt loose / floaty**, **Bounceback**, **Propwash**, **Felt twitchy / too sharp**, **Roll/pitch tracking feels delayed**, **Roll feels weak**, **Pitch feels weak**, or **Oscillation during throttle punches**, AeroTune bypasses the baseline hold and uses those symptoms with the log to build a targeted plan. The baseline hold is only for genuinely good/cool flights with no PID-relevant complaint.

## V1.9 Tune-Change Tracking: Raw Flight Pair Selector

For raw `.BBL`, `.BFL`, or `.TXT` logs that contain multiple flights, AeroTune can now decode the file once and compare two selected flights from the same raw log.

Default behavior:

```text
2 flights total  -> before = Flight 1/2, after = Flight 2/2
6 flights total  -> before = Flight 5/6, after = Flight 6/6
```

The pilot can override both selectors before running the report. The UI labels the first selector as **Before recommendations / baseline flight** and the second selector as **After change flight**.

Tune-change tracking now also includes structured **Tune Change Details** checkboxes for PID changes, filtering changes, rates/throttle feel, hardware/setup changes, and test conditions. These options are context only; they do not stack into giant PID recommendations. AeroTune caps final PID percentage advice to ±11% per test pass.

