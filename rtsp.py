import cv2

rtsp_url = "rtsp://Yash_Padam:Madap1973@192.168.0.100:554/stream2"

cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Cannot open RTSP stream. Check URL / camera / credentials.")
    exit()

print("RTSP Stream Connected. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Retrying...")
        continue

    cv2.imshow("RTSP Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
