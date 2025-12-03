import os
import cv2
import time
import requests
import shutil


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "users")
INTRUDER_DIR = os.path.join(BASE_DIR, "data", "intruders")

def create_user_folder(username):
    folder_path = os.path.join(DATA_DIR, username)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

def save_user_image(frame, username, img_num):
    user_folder = os.path.join(DATA_DIR, username)
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)
    path = os.path.join(user_folder, f"img_{img_num}.jpg")
    cv2.imwrite(path, frame)
    print(f"Saved {path}")

def get_all_users():
    if not os.path.exists(DATA_DIR):
        return []
    return [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]

def delete_user_data(username):
    user_folder = os.path.join(DATA_DIR, username)
    if os.path.exists(user_folder):
        shutil.rmtree(user_folder)
        return True
    return False

def get_user_images(username):
    user_folder = os.path.join(DATA_DIR, username)
    if not os.path.exists(user_folder):
        return []
    return [f for f in os.listdir(user_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

def ensure_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def save_intruder_image(frame, base_path="data/intruders"):
    ensure_folder(base_path)
    ts = time.strftime("%Y%m%d_%H%M%S")
    filename = f"intruder_{ts}.jpg"
    path = os.path.join(base_path, filename)
    cv2.imwrite(path, frame)
    print(f"[INTRUDER] Saved {path}")
    return path

def send_telegram_alert(image_path, bot_token, chat_id, caption="Intruder Detected!"):
    if not bot_token or not chat_id:
        print("[telegram] token or chat_id not provided. Skipping telegram.")
        return False

    url_base = f"https://api.telegram.org/bot{bot_token}"
    send_photo_url = url_base + "/sendPhoto"

    with open(image_path, "rb") as f:
        files = {"photo": f}
        data = {"chat_id": chat_id, "caption": caption}
        try:
            r = requests.post(send_photo_url, data=data, files=files, timeout=10)
            if r.status_code == 200:
                print("[telegram] Alert sent.")
                return True
            else:
                print(f"[telegram] Failed to send: {r.status_code} {r.text}")
                return False
        except Exception as e:
            print(f"[telegram] Exception while sending: {e}")
            return False
