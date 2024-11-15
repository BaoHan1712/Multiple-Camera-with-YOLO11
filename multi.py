import cv2
from ultralytics import YOLO
import threading
import time
from queue import Queue


class CameraThread:
    def __init__(self, camera_id, width=640, height=640):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.frame_queue = Queue(maxsize=2)
        self.stopped = False
        self.fps = 0
        self.prev_time = time.time()
        
    def start(self):
        thread = threading.Thread(target=self.update, args=())
        thread.daemon = True
        thread.start()
        return self
    
    def update(self):
        # Kiểm tra xem camera_id có phải là string (đường dẫn video) hay không
        if isinstance(self.camera_id, str):
            cam = cv2.VideoCapture(self.camera_id)
        else:
            cam = cv2.VideoCapture(self.camera_id)
            
        # Kiểm tra xem camera có được mở thành công không
        if not cam.isOpened():
            print(f"Không thể mở camera/video {self.camera_id}")
            self.stopped = True
            return
            
        while not self.stopped:
            ret, frame = cam.read()
            if not ret:
                if isinstance(self.camera_id, str):  # Nếu là video file, reset lại
                    cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
                
            frame = cv2.resize(frame, (self.width, self.height))
            
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
            
        cam.release()
    
    def read(self):
        return self.frame_queue.get() if not self.frame_queue.empty() else None
    
    def stop(self):
        self.stopped = True
    
    def calculate_fps(self):
        current_time = time.time()
        time_diff = current_time - self.prev_time
        self.fps = 1 / time_diff if time_diff > 0 else 0
        self.prev_time = current_time
        return self.fps

def process_frame(frame, model):
    if frame is not None:
        results = model.predict(frame, imgsz=320, conf=0.4)
        # Xử lý kết quả detection ở đây
        return results
    return None

# Khởi tạo model YOLO
model = YOLO("model/yolo11n.onnx")

# CÓ thể thêm nhiều camera vào đây  
camera_threads = [
    CameraThread(0),  # Camera 0
    # CameraThread("data/cars2.mp4"),    
    CameraThread("data/cars2.mp4")
]


# Khởi động các camera threads
for thread in camera_threads:
    thread.start()

try:
    while True:
        frames = []
        fps_list = []
        
        # Đọc frame từ tất cả camera
        for thread in camera_threads:
            frame = thread.read()
            if frame is not None:
                fps = thread.calculate_fps()
                frames.append((frame, fps))
        
        # Xử lý các frame
        for i, (frame, fps) in enumerate(frames):
            results = process_frame(frame, model)
            
            # Vẽ kết quả detection lên frame
            for r in results:
                annotated_frame = r.plot()
                # Thêm text FPS vào frame
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                          1, (0, 255, 0), 2)
                cv2.imshow(f"Camera {i}", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    for thread in camera_threads:
        thread.stop()
    cv2.destroyAllWindows()


