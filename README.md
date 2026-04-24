# ⚡ AeroTune  
### Betaflight Log Analyzer with Real PID Guidance

![Status](https://img.shields.io/badge/status-live-brightgreen)
![Backend](https://img.shields.io/badge/backend-FastAPI-blue)
![Frontend](https://img.shields.io/badge/frontend-HTML%20%2B%20Chart.js-orange)
![Focus](https://img.shields.io/badge/focus-FPV%20PID%20Tuning-purple)

---

## 🌐 Live Demo

https://aerotune.onrender.com  

⚠️ First load may take ~30 seconds due to free hosting (Render cold start)

---

## 🧭 Overview

**AeroTune** is a data-driven FPV tuning tool that analyzes Betaflight Blackbox logs and returns **conservative, real-world PID adjustments** based on actual flight behavior.

---

## 🚀 Features

- CSV Blackbox Log Analysis  
- Automatic Issue Detection  
- 3 Pilot Goals (Efficient / Locked-In / Floaty)  
- Frequency + Tracking Analysis  
- Log Validation + Quality Scoring  
- Real PID % Deltas  

---

## 🧠 How It Works

Flight → Blackbox Log → CSV → AeroTune → PID Recommendations

## 📸 UI Preview

![AeroTune UI](docs/aerotune-ui.png)
---

## 🖥️ Setup

git clone https://github.com/bostromdev/AeroTune.git  
cd AeroTune  
pip install -r requirements.txt  
uvicorn main:app --reload  

---

## 📂 Structure

app/ → backend  
static/ → frontend  
docs/ → guides  
sample_logs/ → demo logs  

---

## ⚠️ Notes

- Use 30–120s logs for best results  
- Increase D carefully (check motor temps)  

---

## 🛡️ License

Apache 2.0 — free to use, modify, and collaborate.

---

## 👤 Author

Christopher Bostrom  
https://github.com/bostromdev  

---

## 🚀 Final

Data → Analysis → Results
