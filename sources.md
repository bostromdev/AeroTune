# 📚 AeroTune Sources & Tuning References

AeroTune is built on established Betaflight PID tuning principles and 
real-world FPV tuning practices.  
This file documents the sources and logic used in the analyzer.

---

## 🔧 Primary Source

### Betaflight PID Tuning Guide
https://betaflight.com/docs/wiki/guides/current/PID-Tuning-Guide

Core principles used:

- **P (Proportional)**
  - Increases responsiveness and tracking
  - Too high → oscillation, bounceback

- **I (Integral)**
  - Improves attitude hold
  - Helps resist drift from wind, CG imbalance, throttle changes

- **D (Derivative)**
  - Provides damping (reduces bounceback and propwash)
  - Too high → amplifies noise, causes hot motors

- **Feedforward (FF)**
  - Improves stick response without increasing oscillation risk
  - Used for snappier feel rather than stability

- **Yaw Axis**
  - D is typically 0
  - Tune with P, I, and FF instead

---

## 🧠 AeroTune Logic Mapping

The analyzer converts log data into tuning recommendations based on:

### Clean Tune Detection
- High gyro ↔ setpoint correlation
- Low tracking error
- No dominant oscillation band

Output:
> “Clean tune — no PID change needed”

---

### Propwash / Bounceback
Detected when:
- Mid-frequency disturbance
- High error during throttle transitions

Fix:
- Increase D slightly (roll/pitch only)
- Lower P if overshooting

---

### Low Frequency Oscillation (Bounce / Wobble)
Detected when:
- Strong low-frequency energy
- Overshoot behavior

Fix:
- Lower P first
- Only add D if bounce remains

---

### Mid / High Frequency Noise
Detected when:
- High-frequency energy dominates

Fix:
- Reduce D
- Check mechanical sources (props, motors, frame)

---

### Weak Hold / Drift
Detected when:
- Tracking error persists at low input

Fix:
- Increase I

---

### Poor Tracking / Soft Feel
Detected when:
- Low correlation between setpoint and gyro

Fix:
- Increase P or FF

---

### Slow Stick Response
Detected when:
- Response lag present

Fix:
- Increase FF first
- Then P if needed

---

## 🎯 Feel-Based Tuning

AeroTune adjusts recommendations based on user-selected tuning goals:

### Efficient / Smooth
- Conservative changes
- Lower D and P bias
- Prioritizes cool motors and smoothness

### Locked-In
- Slightly higher P and FF
- Tighter tracking
- Higher responsiveness

### Floaty / Cinematic
- Lower P and FF
- Softer, smoother movement
- Reduced sharpness

---

## ⚠️ Safety Notes

- Always make small changes (1–3%)
- Check motor temperature after raising D
- Mechanical issues should be fixed before tuning PID
- Logs should be 30–120 seconds for best results

---

## 🧩 Future Expansion

Planned improvements:
- PSD-based analysis (80–200Hz propwash band)
- RMS tracking error scoring
- Before/after comparison system
- AI-assisted conversational tuning

---

Built by Christopher Bostrom  
AeroTune Project
