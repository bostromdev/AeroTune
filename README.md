# AeroTune
AeroTune is an advanced FPV drone tuning and flight analysis tool designed to extract meaningful insights from blackbox logs and translate them into actionable PID tuning improvements.

It combines signal processing techniques with control system analysis to identify oscillations, classify vibration sources, and evaluate setpoint tracking performance across roll, pitch, and yaw axes.

Using FFT-based frequency analysis and real-time data processing, AeroTune detects low, mid, and high-frequency issues such as frame resonance, prop/motor imbalance, and D-term noise, while also analyzing gyro vs setpoint response to assess tuning quality.

Core features include:
- Automated oscillation detection and frequency classification
- Setpoint vs gyro tracking analysis (error, delay, overshoot)
- Data-driven PID and feedforward tuning suggestions
- Frequency spectrum visualization (FFT)
- Betaflight CLI output generation
- Multi-axis scoring and overall tuning rating
- Support for multiple drone sizes and profiles

Built with FastAPI, NumPy, and a modular backend architecture, AeroTune serves both as a practical FPV tuning assistant and a demonstration of applied control theory, signal analysis, and backend engineering.
