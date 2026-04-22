# ⚡ AeroTune


<p align="center">
  <b>Control-System Intelligence for PID Tuning</b><br>
  Analyze • Diagnose • Optimize
</p>

# ⚡ AeroTune

AeroTune is a signal-based PID log analyzer for drones and other control systems.

It reads real or realistic CSV logs, checks whether the data is trustworthy, and gives **small, conservative PID percentage changes** instead of pretending it knows the perfect final tune.

## What AeroTune focuses on

AeroTune is built around three simple pilot goals:

- **Efficiency**  
  Keep the tune clean, cool, and stable. Avoid aggressive moves.

- **Snappiness**  
  Make response feel sharper and more direct without getting reckless.

- **Less Floaty**  
  Make the quad feel more locked-in and less soft during movement.

## What it does

- Validates log quality before making PID suggestions
- Uses FFT-based frequency analysis to detect vibration bands
- Looks at setpoint vs gyro tracking
- Separates **diagnosis** from **safe-to-adjust PID advice**
- Outputs **P / D / FF percentage deltas**
- Supports sample log generation for testing

## What it does not do

- It does **not** guess magical final PID values
- It does **not** tune from corrupted or fake timing
- It does **not** force changes when confidence is weak

## Repo structure

```text
AeroTune/
├── app/
│   ├── analyzer.py
│   ├── main.py
│   └── parser.py
├── sample_logs/
├── static/
│   └── index.html
├── tools/
│   └── generate_sample_logs.py
├── uploads/
├── main.py
├── requirements.txt
└── README.md
