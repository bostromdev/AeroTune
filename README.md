# ⚡ AeroTune  
### Betaflight Blackbox Analyzer • Data-Driven PID Tuning

![Status](https://img.shields.io/badge/status-live-brightgreen)
![Backend](https://img.shields.io/badge/backend-FastAPI-blue)
![Frontend](https://img.shields.io/badge/frontend-HTML%20%2B%20Chart.js-orange)
![Focus](https://img.shields.io/badge/focus-FPV%20Engineering-purple)
![License](https://img.shields.io/badge/license-Apache%202.0-lightgrey)

---

## 🌐 Live Demo

👉 https://aerotune.onrender.com  

⚠️ First load may take ~30 seconds (free hosting cold start)

---

## 🧭 What AeroTune Is

**AeroTune** converts raw Betaflight Blackbox logs into **real, conservative PID tuning guidance** using signal analysis — not presets.

This tool is built for:
- FPV pilots
- Drone engineers
- People who want *understandable tuning*, not guesswork

---

## 🔬 Core Idea

```text
Flight Data → Signal Analysis → Behavior Detection → PID Adjustment
```

AeroTune removes “trial and error” and replaces it with:
- Measurable data
- Repeatable logic
- Interpretable outputs

---

## 🚀 Features

### 📊 Log Analysis
- CSV Blackbox ingestion
- Gyro + Setpoint comparison
- Real-time metrics extraction

### 🧠 Smart Detection
- Oscillation classification
- Frequency band analysis (low / mid / high)
- Tracking error + lag detection

### 🎯 Tuning Modes
| Mode | Purpose |
|------|--------|
| Efficient | Cooler, safer, smooth |
| Locked-In | Sharp, responsive, aggressive |
| Floaty | Soft, cinematic feel |

### 🔧 Output
- % adjustments for **P / I / D / FF**
- Human-readable explanations
- Actionable next steps

---

## 🧠 Engineering Approach

AeroTune is based on real Betaflight principles:

| Term | Role |
|------|-----|
| P | Tracking strength |
| I | Stability / hold |
| D | Damping / propwash |
| FF | Stick responsiveness |

Internally it uses:
- FFT (frequency analysis)
- Cross-correlation (tracking)
- Lag estimation
- Error ratio modeling

---

## 📦 Sample Data Strategy

Large logs are **not stored in GitHub**.

Instead:
- Small demo logs are included
- Full logs are generated locally

```bash
python3 tools/generate_realistic_logs.py
```

Generated logs:
- 30–120 seconds
- Realistic throttle + motor behavior
- Multi-axis signal fidelity

---

## 🖥️ Local Setup

```bash
git clone https://github.com/bostromdev/AeroTune.git
cd AeroTune
python3 -m pip install -r requirements.txt
uvicorn main:app --reload
```

Open:
```
http://127.0.0.1:8000
```

---

## 🌐 Deployment (Render)

- Build: `pip install -r requirements.txt`
- Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`

Free tier note:
- App sleeps after inactivity
- First request may take ~30s

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
├── tools/
│   └── generate_realistic_logs.py
│
├── sample_logs/
├── uploads/
├── main.py
├── requirements.txt
├── README.md
```

---

## ⚠️ Important Notes

- Recommendations are **conservative by design**
- Always test changes incrementally
- Check motor temps after increasing D
- Log quality determines output quality

---

## 🤝 Collaboration

Open to:
- FPV pilots testing logs
- Signal processing improvements
- UI/UX upgrades
- Advanced tuning logic

---

## 💰 Future Direction

AeroTune may evolve into:

- Hosted “Pro” platform  
- Advanced tuning engine  
- AI-assisted analysis  
- Mobile (iOS) version  
- FPV ecosystem integration  

---

## 🛡️ License

Apache 2.0

- Free to use  
- Free to modify  
- Attribution required  
- Open collaboration encouraged  

---

## 👤 Author

Christopher Bostrom  
https://github.com/bostromdev  

---

## 🚀 Final Thought

> Stop guessing your tune.  
> Start understanding it.

**AeroTune = Data → Insight → Control**
