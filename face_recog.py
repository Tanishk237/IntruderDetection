
import cv2
import os
import numpy as np
import threading
import insightface
from insightface.app import FaceAnalysis

# ------------------------------------------------------------
# GLOBAL SHARED PROCESSED FRAME
# ------------------------------------------------------------
processed_frame = None
processed_lock = threading.Lock()

def set_processed_frame(frame):
    global processed_frame
    with processed_lock:
        processed_frame = frame.copy()

def get_processed_frame():
    global processed_frame
    with processed_lock:
        if processed_frame is None:
            return None
        return processed_frame.copy()


#init insightface model

print("[INFO] Initializing InsightFace model...")
app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(320, 320))  #reduced from 640 for better CPU performance
print("[INFO] InsightFace loaded successfully.")

#training data load — extract embeddings for users
def load_training_data(base_path="data/users"):
    embeddings = []
    labels = []
    label_to_name = {}

    if not os.path.exists(base_path):
        print(f"[WARN] Missing user folder: {base_path}")
        return embeddings, labels, label_to_name

    current_label = 0

    for username in os.listdir(base_path):
        user_path = os.path.join(base_path, username)
        if not os.path.isdir(user_path):
            continue

        print(f"[INFO] Loading images for user: {username}")
        label_to_name[current_label] = username

        for img_name in os.listdir(user_path):
            img_path = os.path.join(user_path, img_name)

            img = cv2.imread(img_path)
            if img is None:
                continue

            faces = app.get(img)
            if len(faces) == 0:
                continue

            # Take the first detected face
            emb = faces[0].embedding
            embeddings.append(emb)
            labels.append(current_label)

        current_label += 1

    print(f"[INFO] Loaded {len(embeddings)} embeddings for training.")
    return np.array(embeddings), np.array(labels), label_to_name


#train recognizer(stores user embeddings per class)

def train_recognizer():
    embeddings, labels, label_to_name = load_training_data()

    if len(embeddings) == 0:
        print("[ERROR] No valid faces found for training.")
        return None, None

    print("[INFO] Training embedding-based recognizer...")

    #store mean embedding per user
    user_embeddings = {}

    for lbl in np.unique(labels):
        user_embs = embeddings[labels == lbl]
        user_embeddings[lbl] = np.mean(user_embs, axis=0)

    print("[INFO] Training complete.")
    return user_embeddings, label_to_name

#cosine similarity

def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)

#recognize and process frame
def recognize_and_process(frame, user_embeddings, label_to_name, threshold=0.45):
    faces = app.get(frame)
    processed = frame.copy()
    results = []

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        emb = face.embedding

        identity = "Unknown"
        best_score = -1
        best_label = None

        if user_embeddings is not None:
            for lbl, saved_emb in user_embeddings.items():
                score = cosine_similarity(emb, saved_emb)
                if score > best_score:
                    best_score = score
                    best_label = lbl

            if best_score > threshold:
                identity = label_to_name.get(best_label, "Unknown")

        #draw bounding box
        color = (0, 255, 0) if identity != "Unknown" else (0, 0, 255)
        cv2.rectangle(processed, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            processed,
            f"{identity} ({best_score:.2f})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        results.append({
            "name": identity,
            "score": float(best_score),
            "box": (x1, y1, x2-x1, y2-y1)
        })

    set_processed_frame(processed)
    return results
