# 🚀 AeroTune v1.0 — Initial Public Release

### FPV PID Tuning Assistant + Log Analyzer

AeroTune is a feel-based FPV tuning tool that translates Blackbox logs into clear, pilot-readable tuning decisions.

---

## ✨ Major Features

### 🧠 Smart PID Analysis Engine
- Converts Blackbox CSV logs into real tuning advice
- Based on Betaflight tuning principles (logic only, not UI/branding)
- Uses gyro vs setpoint tracking + residual error analysis

---

## 🎯 Issue Detection

- Clean tune → no PID change needed
- Propwash / bounceback → D up (roll/pitch), P adjusted
- High-frequency noise → P down, D down
- Low-frequency oscillation → P down first, D unchanged
- Mid-frequency vibration → P/D reduction + mechanical check
- High-throttle oscillation → safer P/D tuning
- Weak hold / drift → I up
- Poor tracking → P / FF up
- Slow response → FF first, then P

---

## 🔍 Classification Improvements

- Clean detection requires BOTH:
  - High correlation
  - Low residual error
- Problems always override clean state
- Each issue gives a unique fix
- Removed conflicting recommendations

---

## ⚡ Performance

- Fixed lag analysis freezing
- Handles large Blackbox logs (30k+ rows)
- Faster correlation logic
- Increased CSV upload limit (local)

---

## 📊 CSV Optimizer

Outputs standardized logs:

time, gyro_x, gyro_y, gyro_z, setpoint_roll, setpoint_pitch, setpoint_yaw, throttle

- Built into UI
- Cleans messy logs automatically
- Enables consistent analysis

---

## 🎨 UI

- Custom AeroTune branding
- Drone logo added
- Removed “Betaflight-style” text
- Clean dark telemetry-style interface

---

## 🧾 Output Style

Each axis shows:
- Issue
- PID changes (simple %)
- Recommendation
- What to try
- Confidence

---

## 🧠 Tuning Modes

- Efficient → smooth, safe
- Locked-In → sharp response
- Floaty → cinematic feel

---

## ⚠️ Safety

- Yaw D always = 0
- Prevents unsafe D increases during noise
- Conservative PID changes

---

## 🧪 Testing

Validated with:
- Clean logs
- Propwash logs
- High-frequency noise logs
- Large datasets

---

## 🔮 Future Plans

- Log trimming
- Before/after comparison
- PSD analysis
- Feel prediction system
- Mobile UI improvements

---

## 👨‍💻 Author

Christopher Bostrom  
AeroTune

---

## 💬 Summary

AeroTune converts:

raw data → pilot decisions


---

# AeroTune v1.02 — Size Profiles + Betaflight PID Value Calculator

## Major Updates

### Size-Specific Drone Profiles
AeroTune now supports backend size profiles for:

- 3" cinewhoop / compact freestyle
- 3.5" cinewhoop / compact freestyle
- 4" sub-250 / compact long-range
- 5" freestyle / racing
- 7" long-range / heavy freestyle

Drone size now affects analyzer behavior instead of being only a UI option.

### Backend Size Handling
The backend now normalizes drone size input and only accepts sizes that have calibrated analyzer profiles:

```text
3, 3.5, 4, 5, 7

