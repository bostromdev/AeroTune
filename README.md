# AeroTune

**FPV Blackbox analysis and PID tuning assistant for Betaflight logs.**

AeroTune reads Betaflight Blackbox flight-log data and turns gyro/setpoint behavior into conservative tuning guidance. It is built to help pilots understand symptoms like overshoot, bounceback, propwash, tracking error, and noisy D-term behavior without pretending to generate a perfect automatic tune.

## Current Best Workflow

AeroTune is **CSV-first**.

For the most reliable beginner workflow:

1. Open the raw `.BBL` log in Betaflight Blackbox Explorer.
2. Select the correct flight inside that log.
3. If there are multiple flights, the newest flight is usually the last one shown, such as `3/3` or `7/7`.
4. Export that selected flight as CSV.
5. Upload the CSV into AeroTune.

This matters because a raw `.BBL` can contain more than one flight. If the wrong internal flight is decoded or exported, AeroTune can analyze the wrong flight. Choosing the correct/latest flight in Blackbox Explorer before CSV export makes the analysis more predictable.

## Raw Blackbox Support

AeroTune can accept raw `.BBL`, `.BFL`, and `.TXT` logs **locally** when Betaflight `blackbox_decode` is installed and available.

Raw-log support works like this:

```text
Raw .BBL / .BFL / .TXT
        -> blackbox_decode converts it locally
        -> AeroTune reads the decoded CSV
        -> AeroTune runs the tuning advisor
```

AeroTune does **not** bundle `blackbox_decode`. It calls your local copy when available. If raw upload fails, export CSV from Betaflight Blackbox Explorer instead.

## Planned Raw Multi-Flight Selector

A future AeroTune upgrade can make raw `.BBL` handling easier by detecting multiple flights inside one raw log and showing a flight selector:

```text
Flight 1/7
Flight 2/7
Flight 3/7
...
Flight 7/7  <- newest / default selection
```

The tuning advisor would then create an analysis profile for each decoded flight and default to the highest-numbered flight. For example, if a `.BBL` contains seven flights, AeroTune should treat `7/7` as the most recent flight unless the pilot selects a different one.

Until that selector is implemented, CSV export from Blackbox Explorer remains the safest workflow because the pilot manually chooses the correct flight before analysis.

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
