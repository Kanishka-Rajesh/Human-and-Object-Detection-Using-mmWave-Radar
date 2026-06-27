<div align="center">

# 📡 Human and Object Detection Using mmWave Radar

### Real-Time Human and Object Detection using TI IWR6843ISK mmWave Radar Sensor, Signal Processing, Machine Learning, Flask, and Interactive Dashboard.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-Backend-black?style=for-the-badge&logo=flask)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-green?style=for-the-badge)
![Signal Processing](https://img.shields.io/badge/Signal-Processing-red?style=for-the-badge)
![Radar](https://img.shields.io/badge/mmWave-Radar-orange?style=for-the-badge)
![Plotly](https://img.shields.io/badge/Plotly-Dashboard-blueviolet?style=for-the-badge)

</div>

---

# 📖 Project Overview

This project implements a **real-time Human and Object Detection System** using the **TI IWR6843ISK mmWave Radar Sensor**.

The radar continuously captures reflected signals from nearby objects. These signals are processed using signal processing algorithms and machine learning techniques to identify whether the detected target is a human or another object.

The processed data is served through a Flask backend and visualized using an interactive web dashboard.

---

# ✨ Features

- 📡 Real-Time mmWave Radar Data Acquisition
- 🧠 Signal Processing Pipeline
- 🤖 Human & Object Classification
- 📈 Interactive Live Dashboard
- 📊 Real-Time Data Visualization
- 🌐 Flask REST API
- ⚡ Low-Latency Processing
- 💻 Cross-Platform Python Backend

---

# 🛠 Technology Stack

| Category | Technologies |
|-----------|--------------|
| Programming Language | Python |
| Backend | Flask |
| Dashboard | HTML, CSS, JavaScript |
| Visualization | Plotly.js |
| Hardware | TI IWR6843ISK mmWave Radar |
| Communication | Serial Communication (UART) |
| Libraries | NumPy, SciPy, PySerial |
| Version Control | Git, GitHub |

---

# 🏗 System Architecture

```
TI IWR6843ISK Radar
          │
          ▼
   Serial Communication
          │
          ▼
      Python Backend
          │
          ▼
 Signal Processing Pipeline
          │
          ▼
 Machine Learning Classifier
          │
          ▼
       Flask REST API
          │
          ▼
 Interactive Dashboard
```

---

# 📂 Project Structure

```
Human-and-Object-Detection-Using-mmWave-Radar
│
├── backend
│   └── app.py
│
├── frontend
│   └── radar_dashboard.html
│
├── README.md
└── requirements.txt
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/Kanishka-Rajesh/Human-and-Object-Detection-Using-mmWave-Radar.git
```

Navigate to the project

```bash
cd Human-and-Object-Detection-Using-mmWave-Radar
```

Install dependencies

```bash
pip install flask flask-cors numpy scipy pyserial
```

Run backend

```bash
python backend/app.py
```

Open the dashboard

```
frontend/radar_dashboard.html
```

---

# 📸 Dashboard Preview

> Screenshots will be added after deployment.

- Dashboard Home
- Live Radar Plot
- Detection Results
- Object Classification
- Performance Metrics

---

# ⚠ Hardware Requirement

This project requires the **TI IWR6843ISK mmWave Radar Sensor** for live data acquisition.

Without the hardware, the backend cannot receive real-time radar data.

The frontend dashboard can still be viewed independently for interface demonstration purposes.

---

# 🎯 Applications

- Smart Surveillance
- Human Presence Detection
- Industrial Automation
- Smart Buildings
- Security Systems
- Contactless Monitoring
- Robotics
- Healthcare Monitoring

---

# 📚 Learning Outcomes

This project provided hands-on experience in:

- Signal Processing
- Embedded Systems
- mmWave Radar Technology
- Machine Learning
- Flask Backend Development
- Real-Time Data Processing
- Dashboard Development
- Serial Communication

---

# 🚀 Future Improvements

- Deep Learning Models
- Multi-Object Tracking
- Gesture Recognition
- Cloud Integration
- Edge AI Deployment
- Mobile Dashboard
- Docker Support

---

# 👨‍💻 Author

**Kanishka Rajesh**

🎓 B.Tech Information Technology

SSN College of Engineering

GitHub:

https://github.com/Kanishka-Rajesh

---

# ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.
