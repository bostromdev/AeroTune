# ⚡ AeroTune  
### Betaflight Log Analyzer with Real PID Guidance

![Status](https://img.shields.io/badge/status-active-brightgreen)
![Backend](https://img.shields.io/badge/backend-FastAPI-blue)
![Frontend](https://img.shields.io/badge/frontend-HTML%20%2B%20Chart.js-orange)
![Focus](https://img.shields.io/badge/focus-FPV%20PID%20Tuning-purple)

---

## 🧭 Overview

**AeroTune** is a data-driven FPV tuning tool that analyzes Betaflight Blackbox logs and returns **conservative, real-world PID adjustments** based on actual flight behavior.

It is designed to:
- Detect oscillations and noise patterns  
- Evaluate tracking performance (gyro vs setpoint)  
- Recommend safe percentage changes for **P, I, D, and Feed Forward**  
- Align with real Betaflight tuning principles  

---

## 🚀 Features

- 📊 **CSV Blackbox Log Analysis**
- 🧠 **Automatic Issue Detection**
- 🎯 **3 Pilot Goals**
  - Efficient (safe + smooth)
  - Locked-In (sharp + responsive)
  - Floaty (cinematic + soft)
- 📈 **Frequency + Tracking Analysis**
- 🛡️ **Log Validation + Quality Score**
- 🔧 **Real PID % Deltas (not fake numbers)**

---

## 🧠 How It Works

```text
Flight → Blackbox Log → CSV → AeroTune → PID Recommendations
```

AeroTune analyzes:
- Gyro frequency content (FFT)
- Tracking correlation
- Response lag
- Error ratio

Then applies Betaflight-based tuning logic:
- **P** → tracking strength  
- **I** → hold / drift correction  
- **D** → damping / propwash control  
- **FF** → stick responsiveness  

---

## 📘 Documentation

- 📊 How to Get a CSV Log  
- 🧠 PID Tuning Guide (PDF)

---

## 🖥️ Setup

### 1. Clone Repo
```
git clone https://github.com/YOUR_USERNAME/AeroTune.git
cd AeroTune
```

### 2. Install Dependencies
```
pip install -r requirements.txt
```

### 3. Run Server
```
uvicorn main:app --reload
```

### 4. Open UI
```
http://127.0.0.1:8000
```

---

## 📂 Project Structure

```
AeroTune/
├── app/
│   ├── analyzer.py
│   ├── parser.py
│   ├── log_validator.py
│
├── static/
│   └── index.html
│
├── docs/
│   ├── blackbox-csv-guide.md
│   ├── AeroTune_PID_Tuning_Guide.pdf
│
├── main.py
├── requirements.txt
├── README.md
```

---

## ⚠️ Important Notes

- This tool provides **conservative tuning guidance**, not aggressive presets  
- Always test changes incrementally  
- Monitor **motor temperatures when increasing D**  
- Log quality directly affects accuracy  

---

## 🎯 Goal Philosophy

- **Efficient** → safer, cooler, smoother  
- **Locked-In** → tighter tracking, sharper feel  
- **Floaty** → softer, cinematic movement  

---

## 🛠️ Roadmap

- [ ] Live log preview graphs  
- [ ] Auto log scoring UI display  
- [ ] PID profile export  
- [ ] Mobile app version  

---

## 📜 License

Apache 2.0 — free to use, modify, and build on with attribution.

---

## 👤 Author

Built by Christopher Bostrom  
GitHub: https://github.com/bostromdev  

---

## 🚀 Final Note

AeroTune is built to bridge the gap between **raw Blackbox data** and **real tuning decisions**.

> Not guesses. Not presets.  
> Just data → analysis → results.
