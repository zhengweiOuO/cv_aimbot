import pyautogui
import keyboard
# from pynput import mouse
from pynput.mouse import Controller

def move_mouse_to_head(results, screen_width=2560, screen_height=1440):
    if not results or not results[0].boxes:
        return

    boxes = results[0].boxes
    names = results[0].names
    mouse = Controller()

    for i, cls_id in enumerate(boxes.cls):
        class_name = names[int(cls_id)]
        if class_name == "enemy" :
            xyxy = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = xyxy
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # 將滑鼠移動到此位置（根據螢幕解析度）
            pyautogui.moveTo(center_x, center_y)
            print(f"Moved mouse to head at ({center_x}, {center_y})")
            break  # 只鎖定一個頭就好