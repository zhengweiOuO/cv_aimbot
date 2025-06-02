import cv2
import numpy as np
import time

class GameCapture:
    def __init__(self, camera_id=0, fps_limit=180):
        """
        初始化捕捉模組
        camera_id: 0 通常是第一個攝影機裝置 (可能是 OBS 虛擬攝影機)
        fps_limit: 限制捕捉的最大幀率
        """
        self.camera_id = camera_id
        self.fps_limit = fps_limit
        self.cap = None
        self.last_frame_time = 0
        self.frame_interval = 1.0 / fps_limit
    
    def start(self, width = 1920, height = 1080):
        """啟動捕捉"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise Exception(f"無法連接到攝影機裝置 {self.camera_id}")
        
         # 設定解析度
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        return self
    
    def read(self):
        """讀取一幀畫面，考慮 FPS 限制"""
        current_time = time.time()
        if current_time - self.last_frame_time < self.frame_interval:
            return None
        
        ret, frame = self.cap.read()
        if ret:
            self.last_frame_time = current_time
            return frame
        return None
    
    def release(self):
        """釋放資源"""
        if self.cap is not None:
            self.cap.release()