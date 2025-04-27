# **Real-Time Video Processing and Security Detection System 🚀 **

## **Overview**  
This project is an **AI-powered smart zone monitoring system** designed for **security applications**. Instead of complex action recognition, it detects **unauthorized presence** in predefined restricted areas.  

### **Why This?**  
🔹 **Lighter & faster** than full action recognition.  
🔹 **Easier deployment** for real-world security setups.  
🔹 **Works with static surveillance cameras.**  

---

## **Features** (Planned & Implemented ✅)  
✔️ **Real-time video processing** with support for files, RTSP streams, and camera inputs
✔️ **Multi-threaded architecture for optimized performance**
✔️ **Intrusion detection** in marked zones.  
✔️ **Configurable detection areas** for different security setups.  
✔️  Real-time alerts & notifications.  
🔄 **(Planned)** API for easy integration.  
🔄 **(Planned)** Edge device deployment for efficiency.  

---

## **Setup & Installation**  
1. Clone the repo:  
   ```bash
   git clone https://github.com/fw7th/AnomalyDetection.git
   cd AnomalyDetection
   ```
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables for notifications:
- Create a `.env` file with your email credentials
- Configure Twilio credentials in `AnomalyDetection/config/keys/twilio.txt`

## **Usage**
Basic usage:
```bash
python main.py --source /path/to/video.mp4 --enable-saving --output output.mp4
```
For camera input:
```bash
python main.py --source 0
```
For RTSP stream:
```bash
python main.py --source rtsp://your-stream-url
```
With notifications:
```bash
python main.py --source 0 --phone +1234567890 --email you@example.com
```
With selected detection accuracy:
```bash
python main.py --source 0 --phone +1234567890 --email you@example.com --accuracy 2 or "mid"
```

4. Run the detection module:  
   ```bash
   python main.py
   ```
---

## **How It Works**  
1. **Define restricted zones** on a surveillance feed.  
2. **AI detects intrusions** in marked areas.  
3. **Alerts & logs unauthorized access**
---

## Configuration
- Configure detection zone in `scripts/draw_zones.py`
- Adjust processing parameters in `config/settings.py`
  
---

## Architecture
This system uses a pipeline architecture with four main components:
1. Frame Acquisition - Reading frames from video sources
2. Preprocessing - Preparing frames for analysis
3. Detection - AI-powered object detection
4. Display/Output - Visualization and saving results

---

## **Future Plans & Improvements**  
📌 **Optimize model for real-time detection.**  
📌 **Deploy as a cloud-based API.**  
📌 **Improve accuracy & reduce false positives.** 
📌 **Facial Recognition system to negate alarm system for specific faces.**

---

## **Contributing**  
_Feel free to open issues, suggest improvements, or contribute!_  

---

## **Connect with Me**  
📌 **LinkedIn: (https://www.linkedin.com/in/7th-david/)**  
📌 **Twitter/X: [@fw7th](https://twitter.com/fw7th)** 
📌 **Discord: (Jaeger#7454)**

---

### **License**  
_This project is open-source under [MIT License](LICENSE)._  
