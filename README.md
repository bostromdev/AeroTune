# ⚡ AeroTune

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

---

## 📁 Project Structure

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

