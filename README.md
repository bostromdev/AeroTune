# AeroTune

AeroTune is an FPV drone tuning and flight log analysis tool that processes blackbox-style CSV logs to detect oscillation patterns, evaluate gyro-to-setpoint tracking, and generate data-driven PID recommendations.

## Features

- FFT-based vibration analysis across roll, pitch, and yaw
- PID tuning suggestions for P, D, and feedforward
- Gyro vs setpoint tracking evaluation
- Frequency spectrum and time-domain graphs
- Betaflight CLI preview
- Sample log generator for repeatable testing
- Included sample logs for validation

## Project Structure

```text
aerotune/
├── app/
│   ├── __init__.py
│   ├── analyzer.py
│   ├── parser.py
│   └── main.py
├── static/
│   └── index.html
├── tools/
│   └── generate_sample_logs.py
├── sample_logs/
├── uploads/
├── requirements.txt
├── README.md
└── .gitignore
```

## Run locally

```bash
pip3 install -r requirements.txt
python3 -m uvicorn app.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

## Generate sample logs

```bash
python3 tools/generate_sample_logs.py
```

Generated logs will appear in `sample_logs/`.

## Notes

- Renaming the top-level repo folder does not require code changes.
- Imports are based on the `app` package, not the repo folder name.
- Upload a CSV log, choose drone size, and review the summary, axis analysis, graphs, and CLI preview.
