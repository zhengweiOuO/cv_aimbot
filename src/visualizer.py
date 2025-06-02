import cv2
import numpy as np

class PoseVisualizer:
    def __init__(self, window_name="Game Character Pose Detection"):
        """
        初始化視覺化模組
        """
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1920, 1080)

    def show(self, frame, fps=None):
        """
        顯示處理後的畫面
        """
        # 添加 FPS 顯示
        if fps is not None:
            cv2.putText(
                frame, 
                f"FPS: {fps:.1f}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )
        
        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(1)
        
        # 按 ESC 退出
        return key != 27
    
    def close(self):
        """
        關閉所有視窗
        """
        cv2.destroyAllWindows()