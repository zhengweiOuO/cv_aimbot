import cv2
import time
import numpy as np
from src.capture import GameCapture
from src.detector import PoseDetector
from src.visualizer import PoseVisualizer

def main():
    # 初始化各模組
    capture = GameCapture(camera_id=0).start()  # 0 是 OBS 虛擬攝影機或其他攝影機裝置
    detector = PoseDetector()
    visualizer = PoseVisualizer()
    
    # FPS 計算
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0
    
    try:
        while True:
            # 讀取畫面
            frame = capture.read()
            if frame is None:
                continue
            
            # 姿勢檢測
            results = detector.detect(frame)
            
            # 繪製關鍵點
            frame = detector.draw_landmarks(frame, results)
            
            # 計算 FPS
            fps_counter += 1
            current_time = time.time()
            if current_time - fps_start_time >= 1.0:
                current_fps = fps_counter / (current_time - fps_start_time)
                fps_counter = 0
                fps_start_time = current_time
            
            # 顯示結果
            if not visualizer.show(frame, current_fps):
                break
    
    except KeyboardInterrupt:
        print("程式被使用者中斷")
    finally:
        # 釋放資源
        capture.release()
        visualizer.close()

if __name__ == "__main__":
    main()