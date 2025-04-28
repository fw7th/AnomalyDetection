# 🚀 Real-Time Video Processing and Security Detection System

## Overview  
This project is an **AI-powered smart zone monitoring system** designed for **security applications**.  
Instead of complex action recognition, it detects **unauthorized presence** in predefined restricted areas.

### Why This?  
- ⚡ **Lightweight and fast** compared to full action recognition.  
- 🛠️ **Easier deployment** for real-world setups.  
- 🎥 **Works seamlessly with static surveillance cameras.**

---

## Features  
- ✅ **Real-time video processing** with support for files, RTSP streams, and camera inputs.  
- ✅ **Multi-threaded architecture** for optimized performance.  
- ✅ **Intrusion detection** in marked zones.  
- ✅ **Configurable detection areas** for flexible security setups.  
- ✅ **Real-time alerts and notifications.**  
- 🔄 **(Planned)** API for easy integration.  
- 🔄 **(Planned)** Edge device deployment for low-power efficiency.

<p align="center">
  <img src="AnomalyDetection/media/demo.gif" alt="Demo" width="600">
</p>

---

## Setup & Installation  
1. Clone the repository:
   ```bash
   git clone https://github.com/fw7th/AnomalyDetection.git
   cd AnomalyDetection
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables with notifications:
   - Create a ```.env``` file with your email credentials
   - Configure Twilio credentials in ```AnomalyDetection/config/keys/twilio.txt.```


---

## Usage
Basic usage:
   ```bash
   python main.py --source /path/to/video.mp4 --enable-saving --output output.mp4
   ```
Run the script on the provided sample video
   ```bash
   python detect.py --source AnomalyDetection/examples/sample.mp4
   ```
Camera input:
   ```bash
   python main.py --source 0
   ```
RTSP stream:
   ```bash
   python main.py --source rtsp://your-stream-url
   ```
With email notifications:
   ```bash
   python main.py --source 0 --email you@example.com
   ```
With SMS notifications:
   ```bash
   python main.py --source 0 --phone +1234567890
   ```
With a path for saved video:
   ```bash
   python main.py --source 0 --enable-saving --output /your/save/path.mp4
   ```
With selected detection accuracy:
   ```bash
   python main.py --source 0 --accuracy 2
   ```

---

## How It Works
1. 🗺️ Define restricted zones on the surveillance feed.
2. 🎯 Detect intrusions inside the marked areas.
3. 📢 Trigger alerts and log unauthorized access.

---

## Configuration
- Zone setup: ```scripts/draw_zones.py```
- System parameters: ```config/settings.py```

---

## Architecture
This system uses a modular pipeline:
1. **Frame Acquisition** — Capture frames from source.
2. **Preprocessing** — Prepare frames for detection.
3. **Detection** — AI-driven analysis.
4. **Display/Output** — Visualization and output saving.

---

## Future Plans
- 📈 Optimize model for real-time processing.

- ☁️ Deploy a cloud-based API version.

- 🎯 Improve detection accuracy and lower false positives.
  
- 🧠 Integrate facial recognition for authorized faces.

---

## Contributing
Pull requests, suggestions, and issue reports are welcome!

---

## Connect
🔗 **LinkedIn: (https://www.linkedin.com/in/7th-david/)** 

🐦 **Twitter/X: [@fw7th](https://twitter.com/fw7th)** 

---

### **License**  
_This project is open-source under [MIT License](LICENSE)._  
