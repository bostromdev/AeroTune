# ⚡ AeroTune

<<<<<<< HEAD
<p align="center">
  <b>Control-System Intelligence for PID Tuning</b><br>
  Analyze • Diagnose • Optimize
</p>

---

## 🚀 Overview

**AeroTune is a signal-driven PID tuning system designed to analyze real behavior and generate intelligent tuning adjustments.**

It processes time-series data from PID-controlled systems (drones, robotics, motion control) and converts it into:

- actionable tuning decisions  
- system stability insights  
- performance optimization strategies  

---

## 🧠 What Makes It Different

Unlike traditional tuning methods, AeroTune:

- ❌ Does NOT rely on guesswork  
- ❌ Does NOT blindly increase/decrease gains  
- ✅ Uses real signal analysis (FFT + tracking response)  
- ✅ Validates data before making decisions  
- ✅ Adapts tuning based on system behavior  

---

## ⚙️ Core Features

### 📊 Frequency Analysis
- Detects dominant vibration bands  
- Identifies harmonic structures  
- Classifies system instability sources  

### 🎯 Adaptive PID Logic
- Independently adjusts:
  - Proportional (P)
  - Derivative (D)
  - Feedforward (FF)
- Reacts to:
  - noise
  - oscillation
  - lag
  - overshoot  

### 📈 Tracking Analysis
- Measures:
  - delay
  - overshoot
  - response accuracy  
- Detects invalid or misaligned data  
- Prevents false tuning recommendations  

### ⚠️ Fault Detection
- Ignores corrupted or unrealistic signals  
- Flags unreliable tracking  
- Protects tuning logic from bad input  

### 📋 CLI Output
- Generates ready-to-use tuning commands  
- Compatible with real PID-based systems  
=======
**AeroTune is a control-system analysis tool that uses real signal processing to optimize PID behavior.**

It processes log data from PID-controlled systems (drones, robotics, motion systems) to detect oscillations, evaluate response quality, and generate data-driven tuning recommendations.

---

## 🚀 What It Does

AeroTune analyzes system behavior using:

- 📊 Frequency analysis (FFT) to detect vibration bands  
- 🎯 PID tuning logic (P, D, Feedforward adjustments)  
- 📈 Response tracking (delay, overshoot, control accuracy)  
- ⚠️ Fault detection (invalid tracking, noisy signals, instability)  

---

## 🧠 Why It Matters

Tuning PID systems is often:
- trial and error  
- inconsistent  
- hard to interpret  

AeroTune turns raw data into **clear engineering decisions**, reducing guesswork and improving system efficiency.

---

## 🛠️ Features

- Multi-axis signal analysis (roll, pitch, yaw / generic axes)
- Dynamic PID adjustment recommendations
- Frequency band classification (low / mid / high)
- Tracking validation (ignores bad or misaligned data)
- CLI output for direct system tuning
- Sample log generator for testing scenarios
>>>>>>> c212d5df8bf03fe24be62e0ce4d67f91ac49f066

---

## 📁 Project Structure

<<<<<<< HEAD
## 📜 License

This project is licensed under the Apache 2.0 License.

You are free to use, modify, and distribute this software, but must 
provide proper attribution to the original author.
=======
aerotune/
├── app/
├── static/
├── tools/
├── sample_logs/
├── uploads/
├── requirements.txt
└── README.md

---

## ⚡ Run Locally

pip3 install -r requirements.txt  
python3 -m uvicorn app.main:app --reload  

Open:  
http://127.0.0.1:8000

---

## 🧪 Generate Sample Logs

python3 tools/generate_sample_logs.py

---

## 🧠 Engineering Focus

This project demonstrates:

- Signal processing (FFT, spectral analysis)  
- Control systems (PID behavior & response tuning)  
- Data validation and fault detection  
- Backend system design (FastAPI)  
- Real-time analysis pipelines  

---

## 🚧 Development Notes

- Graphs are functional but not yet fully optimized  
- Visualization and performance improvements are planned  
- PID modeling and tuning logic will continue to evolve  

---

## 🛩️ Real-World Testing (Upcoming)

AeroTune will be validated with real system data:

- Live flight logs (drones)  
- Real PID-controlled system behavior  
- Hardware-based tuning validation  

Future updates will improve accuracy using real-world data.

---

## 🧠 Philosophy

AeroTune does not aim to maximize raw output.

It aims to:

**maximize usable performance by minimizing wasted energy in instability and vibration**

---

## ⚠️ Status

Active development — core logic is functional, with ongoing improvements.


