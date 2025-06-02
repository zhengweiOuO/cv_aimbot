import cv2
import time
import numpy as np
import pyautogui
import keyboard
from src.capture import GameCapture
from src.visualizer import PoseVisualizer
from src.move import move_mouse_to_head
from ultralytics import YOLO
from pynput import mouse
from pynput.mouse import Controller

def main():
    # 初始化各模組
    capture = GameCapture(camera_id=0).start()  # 0 是 OBS 虛擬攝影機或其他攝影機裝置
    detector = YOLO('C:/Users/zhengwei/aimbot/yolov12-main/CSGO-YOLOv8/train/runs/detect/valorant-yolo12n/weights/best.pt')
    visualizer = PoseVisualizer()
    controller = Controller()
    
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
            results = detector(frame, conf=0.3)
            frame = results[0].plot()
            
            # 滑鼠移動
        if keyboard.is_pressed('x'):
            move_mouse_to_head(results)
            
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
