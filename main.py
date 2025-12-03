import time
import json
import os
import threading

from camera_stream import CameraStream
from face_recog import train_recognizer, recognize_and_process, get_processed_frame
from utils import save_intruder_image, send_telegram_alert
from dashboard.app import app, set_annotated_frame_provider


CONFIG_PATH = "config.json"

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return {}

cfg = load_config()

CAM_INDEX = cfg.get("camera_index", "rtsp://Yash_Padam:Madap1973@192.168.0.100:554/stream2")
MATCH_THRESHOLD = cfg.get("match_threshold", 0.45)
UNKNOWN_DURATION = cfg.get("unknown_duration_threshold", 2.0)  # seconds
SEND_TELEGRAM = cfg.get("send_telegram", False)
TELEGRAM_TOKEN = cfg.get("telegram_token", "")
TELEGRAM_CHAT_ID = cfg.get("telegram_chat_id", "")
INTRUDER_SAVE_PATH = cfg.get("save_intruder_path", "data/intruders")
ALERT_COOLDOWN = 30


#global state
global_unknown_start = None
global_last_alert_time = 0
intruder_alerted = False

global_user_embeddings = None
global_label_map = {}
recognizer_lock = threading.Lock()


#retrain recognizer function
def retrain_recognizer():
    """Retrain InsightFace embeddings"""
    global global_user_embeddings, global_label_map

    print("[INFO] Retraining recognizer...")
    user_embeddings, label_map = train_recognizer()

    with recognizer_lock:
        global_user_embeddings = user_embeddings
        global_label_map = label_map if label_map else {}

    if user_embeddings is None:
        print("[WARN] No users — all faces will be Unknown.")
        return False

    print(f"[INFO] Recognizer updated with {len(label_map)} users.")
    return True


#background detection loop
def detection_loop():
    global global_unknown_start, global_last_alert_time, intruder_alerted

    print("[INFO] Starting Camera Stream...")
    cam = CameraStream(src=CAM_INDEX)

    print("[INFO] Loading recognizer...")
    retrain_recognizer()

    print("[INFO] Detection thread started.")

    frame_skip_counter = 0
    PROCESS_EVERY_N_FRAMES = 10  # Process every 10th frame (~100ms at 30fps)

    while True:
        frame = cam.read()
        if frame is None:
            time.sleep(0.05)
            continue

        #skip frames to reduce CPU load
        frame_skip_counter += 1
        if frame_skip_counter < PROCESS_EVERY_N_FRAMES:
            time.sleep(0.01)
            continue
        frame_skip_counter = 0

        #get current recognizer safely
        with recognizer_lock:
            user_embeddings = global_user_embeddings
            label_map = global_label_map.copy()
            results = recognize_and_process(
                frame,
                user_embeddings,
                label_map,
                threshold=MATCH_THRESHOLD
            )

        #intruder duration logic
        #if no recognizer (empty database), treat all detected faces as unknown
        if user_embeddings is None and len(results) > 0:
            all_unknown = True
        elif len(results) > 0:
            all_unknown = all(r["name"] == "Unknown" for r in results)
        else:
            all_unknown = False

        if all_unknown:
            if global_unknown_start is None:
                global_unknown_start = time.time()
                intruder_alerted = False  # Reset for new session

            elapsed = time.time() - global_unknown_start

            #only capture if duration exceeded and not yet alerted for this session
            if elapsed >= UNKNOWN_DURATION and not intruder_alerted:
                print("[ALERT] Intruder detected > threshold duration")

                #save intruder snapshot (processed frame ideally)
                processed = get_processed_frame()
                if processed is not None:
                    intr_path = save_intruder_image(processed, base_path=INTRUDER_SAVE_PATH)
                else:
                    intr_path = save_intruder_image(frame, base_path=INTRUDER_SAVE_PATH)
                
                #mark as alerted so we don't spam photos
                intruder_alerted = True

                #telegram alert (cooldown protected)
                now = time.time()
                if SEND_TELEGRAM and (now - global_last_alert_time >= ALERT_COOLDOWN):
                    send_telegram_alert(intr_path, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
                    global_last_alert_time = now

        else:
            #reset everything when intruder leaves or is recognized
            global_unknown_start = None
            intruder_alerted = False

        time.sleep(0.01)


#provide processed frames to dashboard
def get_annotated_frame():
    """
    Dashboard will consume this.
    Comes directly from recognize_and_process() → processed_frame.
    """
    frame = get_processed_frame()
    return frame


#main
if __name__ == "__main__":
    #start background detection
    t = threading.Thread(target=detection_loop, daemon=True)
    t.start()

    #inject processed frame provider into Flask app
    set_annotated_frame_provider(get_annotated_frame)
    
    #inject retrain function into Flask app
    from dashboard.app import set_retrain_function
    set_retrain_function(retrain_recognizer)

    print("[INFO] Starting Dashboard Server...")
    app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False)
