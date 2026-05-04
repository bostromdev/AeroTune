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

## Main Features

- Single-log Blackbox analysis
- PID Tuning Advice card
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
