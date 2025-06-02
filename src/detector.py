import mediapipe as mp
import numpy as np
import cv2

class PoseDetector:
    def __init__(self, model_complexity=1, min_detection_confidence=0.5):
        """
        初始化 MediaPipe 姿勢檢測器
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )
    
    # def detect(self, frame):
    #     """
    #     檢測姿勢關鍵點
    #     """
    #     # 轉換為 RGB (MediaPipe 需要)
    #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
    #     # 進行檢測 (設定 writeable=False 可提高效能)
    #     frame_rgb.flags.writeable = False
    #     results = self.pose.process(frame_rgb)
    #     frame_rgb.flags.writeable = True
        
    #     return results
    def detect(self, frame):
        """
        檢測並提取頭部關鍵點
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.pose.process(frame_rgb)
        frame_rgb.flags.writeable = True

        head_landmarks = None

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            # 取得頭部關鍵點
            head_landmarks = {
                "nose": (landmarks[0].x, landmarks[0].y, landmarks[0].z),
                "left_eye": (landmarks[2].x, landmarks[2].y, landmarks[2].z),
                "right_eye": (landmarks[5].x, landmarks[5].y, landmarks[5].z),
                "left_ear": (landmarks[7].x, landmarks[7].y, landmarks[7].z),
                "right_ear": (landmarks[8].x, landmarks[8].y, landmarks[8].z),
            }

        return head_landmarks
    
    # def draw_landmarks(self, frame, results):
    #     """
    #     繪製檢測到的姿勢關鍵點
    #     """
    #     if results.pose_landmarks:
    #         self.mp_drawing.draw_landmarks(
    #             frame,
    #             results.pose_landmarks,
    #             self.mp_pose.POSE_CONNECTIONS,
    #             landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
    #         )
        
    #     return frame
    def draw_landmarks(self, frame, head_landmarks):
        """
        只繪製頭部關鍵點
        """
        if head_landmarks:
            for key, (x, y, _) in head_landmarks.items():
                h, w, _ = frame.shape
                cx, cy = int(x * w), int(y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)  # 綠色圓點標記
                cv2.putText(frame, key, (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 255, 0), 1, cv2.LINE_AA)

        return frame
