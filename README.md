# Intruder Detection System  
### **(InsightFace + RTSP CCTV + Flask Dashboard + Telegram Alerts)**

A fully local, real-time **Intruder Detection System** built with:

- **InsightFace** (state-of-the-art 512-D face embeddings)  
- **RTSP CCTV camera feed**  
- **Flask dashboard** for monitoring and user management  
- **Telegram bot alerts** for intruder events  
- **Threaded OpenCV streaming**  
- **Offline-first, privacy-focused design**

This system is ideal for home surveillance, office entry monitoring, and real-time security alerting.

---

# Features

### CCTV RTSP Input  
Reads video from any RTSP-enabled CCTV camera using OpenCV.

### Accurate Face Recognition (InsightFace)  
- 512-D embeddings  
- Cosine similarity matching  
- Robust even in lower lighting / distance

### Smart Intruder Detection  
- Unknown face must persist **≥ 2 seconds**  
- Reduces false alarms  
- Intruder snapshot saved locally  
- Telegram alert with cooldown

### Full Flask Dashboard  
- View live feed with bounding boxes  
- Add users (via **local webcam only**)  
- View/Delete users  
- Refresh recognizer  
- View intruder images

### Secure & Offline  
All face processing is done locally.  
No cloud model usage.

### Thread-Safe Architecture  
- Shared camera thread  
- Locks for processed frames  
- Dashboard + detection engine run in parallel

### Raspberry Pi Ready  
Runs on **Raspberry Pi 4/5** with RTSP or PiCamera.

---

# System Architecture

![alt text](image.png)


# ⚙️ Workflow Overview

### **1. RTSP Camera Feed**
- The system continuously reads video from RTSP using OpenCV.
- `CameraStream` runs in a background thread.
- Shared frames are protected by locks.

### **2. User Enrollment (Webcam)**
- The RTSP feed is **NOT** used for adding users.
- Laptop webcam captures 4–5 clear images.
- Stored in `data/users/<username>/`.

### **3. Training the Recognizer (InsightFace)**
- Extract embeddings using **buffalo_l** model.
- Compute mean embedding per user.
- Store embeddings in RAM:
    - global_user_embeddings
    - global_label_map


### **4. Recognition Pipeline**
For each frame from RTSP:
1. Detect faces  
2. Get 512-D embedding  
3. Compare with stored user embeddings  
4. Draw bounding box + score  
5. Send processed frame to dashboard  

### **5. Intruder Detection Logic**
- If identity = "Unknown":
- Start timer  
- If persists ≥ 2 seconds → intruder  
- Save snapshot  
- Send Telegram alert (cooldown protected)

### **6. Dashboard**
- Serves annotated live feed  
- Add user  
- Refresh recognizer  
- View/delete users  
- See intruder photos  

---

# 📁 config.json Format

```json
{
  "camera_index": "rtsp://user:pass@192.168.0.100:554/stream2",
  "device_camera_index": 0,
  "match_threshold": 0.45,
  "unknown_duration_threshold": 2.0,
  "save_intruder_path": "data/intruders",
  "send_telegram": true,
  "telegram_token": "",
  "telegram_chat_id": ""
}
```

---

# Installation

- pip install opencv-python insightface flask numpy requests onnxruntime

---

# Running the System

1. Place RTSP URL in `config.json`
2. Run `python3 main.py`
3. Access dashboard at `http://localhost:5000`

---

## 🎯 Threshold Tuning

| Scenario               | Suggested Threshold |
|------------------------|---------------------|
| Clear webcam close-up  | **0.35 – 0.45**     |
| Mid-distance CCTV      | **0.45 – 0.60**     |
| Low-light CCTV         | **0.55 – 0.70**     |

---

## 💾 Where Embeddings Are Stored

InsightFace embeddings are stored **in memory only**, not on disk:

- `global_user_embeddings`  
- `global_label_map`

Embeddings are regenerated:

- ✔️ At program startup  
- ✔️ When clicking **Refresh Recognizer** in the dashboard  

No embeddings are saved to disk → **privacy-first & secure**.

---

## Telegram Alerts

- Intruder snapshot stored at:  
data/intruders/

- Telegram alert is sent with snapshot  
- Cooldown prevents multiple alerts  
- Telegram is the **only** external network call (optional)

---

## Raspberry Pi Support

### Works on:
- **Raspberry Pi 4 (4GB)**  
- **Raspberry Pi 5 (8GB recommended)**  
