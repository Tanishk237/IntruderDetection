import sys
import os
import json
import time
import cv2
from flask import Flask, render_template, Response, request, jsonify, send_from_directory

#add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#local imports
from utils import save_user_image, get_all_users, delete_user_data, get_user_images
from camera_stream import CameraStream

app = Flask(__name__)

#configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return {}

cfg = load_config()

CAM_INDEX = cfg.get("camera_index", 0)
MATCH_THRESHOLD = cfg.get("match_threshold", 70)
FRAME_W = cfg.get("frame_width", 640)
FRAME_H = cfg.get("frame_height", 480)

#frame provider for annotated (bounding box) frames
annotated_frame_provider = None

def set_annotated_frame_provider(func):
    """
    Inject the function that returns processed frame
    from the face recognition engine.
    """
    global annotated_frame_provider
    annotated_frame_provider = func


#retrain function (injected from main.py)
retrain_function = None

def set_retrain_function(func):
    """
    Inject the retrain_recognizer function from main.py
    """
    global retrain_function
    retrain_function = func


#stream generator for dashboard live feed

# Separate camera instance for device camera (used for adding users)
DEVICE_CAM_INDEX = cfg.get("device_camera_index", 0)
_cam_instance = None

def get_raw_camera():
    """
    Returns device camera (webcam) for user registration.
    This is separate from the RTSP stream used in main detection.
    """
    global _cam_instance
    if _cam_instance is None:
        print(f"[DEBUG] Dashboard initializing device camera at index {DEVICE_CAM_INDEX}")
        _cam_instance = CameraStream(src=DEVICE_CAM_INDEX, width=FRAME_W, height=FRAME_H)
    return _cam_instance


def generate_frames():
    """
    Sends frames to the dashboard browser.
    Prioritizes annotated frames (processed with bounding boxes).
    Falls back to raw camera only if annotated frame is None.
    """
    cam = get_raw_camera()

    while True:
        frame = None

        #prefer annotated frame (from face recognition thread)
        if annotated_frame_provider:
            frame = annotated_frame_provider()

        #if no processed frame yet → fallback to raw camera
        if frame is None:
            frame = cam.read()

        if frame is None:
            time.sleep(0.05)
            continue

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


def generate_device_camera_frames():
    """
    Streams ONLY from device camera (for Add User page).
    Does NOT use annotated frames from RTSP detection.
    """
    cam = get_raw_camera()

    while True:
        frame = cam.read()

        if frame is None:
            time.sleep(0.05)
            continue

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


#routes
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/view_users')
def view_users():
    users = get_all_users()
    return render_template("view_users.html", users=users)

@app.route('/refresh_recognizer', methods=['POST'])
def refresh_recognizer():
    """Endpoint to retrain the recognizer without restarting the server"""
    if retrain_function is None:
        return jsonify({"status": "error", "message": "Retrain function not available"}), 500
    
    try:
        success = retrain_function()
        if success:
            return jsonify({"status": "success", "message": "Recognizer refreshed successfully!"})
        else:
            return jsonify({"status": "success", "message": "Recognizer refreshed (no users in database)"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/monitor')
def monitor():
    return render_template("monitor.html")

@app.route('/user/<username>')
def user_details(username):
    images = get_user_images(username)
    return render_template("user_details.html", username=username, images=images)

@app.route('/delete_user/<username>', methods=['POST'])
def delete_user(username):
    if delete_user_data(username):
        return jsonify({"status": "success", "message": f"User {username} deleted"})
    return jsonify({"status": "error", "message": "User not found"}), 404

@app.route('/user_img/<username>/<filename>')
def get_image(username, filename):
    data_dir = os.path.join(BASE_DIR, 'data', 'users', username)
    return send_from_directory(data_dir, filename)

@app.route('/video_feed')
def video_feed():
    """
    This will now show the LIVE FEED WITH BOUNDING BOXES,
    because annotated_frame_provider gives processed frames.
    """
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/device_camera_feed')
def device_camera_feed():
    """
    Streams ONLY device camera (webcam) for Add User page.
    Does NOT show RTSP stream or annotated frames.
    """
    return Response(generate_device_camera_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/add_user')
def add_user():
    return render_template("add_user.html")

@app.route('/capture_image', methods=['POST'])
def capture_image():
    username = request.form.get('username')
    img_num = request.form.get('img_num')

    if not username or not img_num:
        return jsonify({"status": "error", "message": "Missing data"}), 400

    cam = get_raw_camera()
    frame_to_save = cam.read()

    if frame_to_save is None:
        return jsonify({"status": "error", "message": "No frame available"}), 500

    try:
        save_user_image(frame_to_save, username, img_num)
        return jsonify({"status": "success", "message": f"Image {img_num} saved"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


#launch
if __name__ == "__main__":
    app.run(debug=True, threaded=True, use_reloader=False)
