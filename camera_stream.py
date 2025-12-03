import cv2
import threading
import time

class CameraStream:
    _instances = {} 
    _lock = threading.Lock()

    def __new__(cls, src=0, width=None, height=None):
        #create unique key for this camera source
        src_key = str(src)
        
        if src_key not in cls._instances:
            with cls._lock:
                if src_key not in cls._instances:
                    instance = super(CameraStream, cls).__new__(cls)
                    instance.src = src
                    instance.cap = cv2.VideoCapture(src)

                    #resolution setting 
                    if width and height:
                        instance.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                        instance.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    
                    instance.frame = None
                    instance.stopped = False
                    instance.read_lock = threading.Lock()
                    
                    #waiting for first frame
                    if instance.cap.isOpened():
                        print(f"[DEBUG] Camera {src} opened successfully.")
                        ret, instance.frame = instance.cap.read()
                        if not ret:
                            print("[ERROR] failed to read from camera source")
                        else:
                            print("[DEBUG] first frame read successfully.")
                    else:
                        print(f"[ERROR] failed to open camera {src}")
                    
                    #starting the thread
                    instance.t = threading.Thread(target=instance.update, daemon=True)
                    instance.t.start()
                    
                    cls._instances[src_key] = instance
        
        return cls._instances[src_key]

    def __init__(self, *args, **kwargs):
        #prevent re-initialization if __init__ is called multiple times
        pass

    def update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                time.sleep(0.1)
                continue
                
            ret, frame = self.cap.read()
            if ret:
                with self.read_lock:
                    self.frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        with self.read_lock:
            if self.frame is not None:
                return self.frame.copy()
            return None

    def stop(self):
        self.stopped = True
        if self.t.is_alive():
            self.t.join()
        if self.cap.isOpened():
            self.cap.release()
