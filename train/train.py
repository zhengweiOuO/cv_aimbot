# train_csgo_yolo.py

from datasets import load_dataset
from ultralytics import YOLO
from PIL import Image
import os

def prepare_yolo_format():
    print("下載並轉換 瓦羅蘭 資料集中...")
    dataset = load_dataset("keremberke/valorant-object-detection", name="full")

    base_path = os.path.join(os.path.dirname(__file__), "datasets", "valorant")
    image_dir = os.path.join(base_path, "images")
    label_dir = os.path.join(base_path, "labels")
    splits = ["train", "validation"]

    for split in splits:
        os.makedirs(os.path.join(image_dir, split), exist_ok=True)
        os.makedirs(os.path.join(label_dir, split), exist_ok=True)

        for i, example in enumerate(dataset[split]):
            img: Image.Image = example['image']
            w, h = img.size
            labels = example['objects']

            img_path = os.path.join(image_dir, split, f"{i:06}.jpg")
            img.save(img_path)

            label_path = os.path.join(label_dir, split, f"{i:06}.txt")
            with open(label_path, "w") as f:
                for box, name in zip(labels['bbox'], labels['category']):
                    '''
                    example = dataset['train'][0]
                    print(example['objects']['category'])
                    print(type(example['objects']['category'][0]))
                    '''
                    x_min, y_min, box_w, box_h = box
                    x_center = (x_min + box_w / 2) / w
                    y_center = (y_min + box_h / 2) / h
                    box_w /= w
                    box_h /= h
                    class_id = name
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")

    print("轉換完成！")

def create_yaml():
    print("建立 valorant.yaml...")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets", "valorant", "images"))
    yaml_content = f"""\
train: {os.path.join(base_dir, "train")}
val: {os.path.join(base_dir, "validation")}

names:
  0: dropped spike
  1: enemy
  2: planted spike
  3: teammate
"""
    with open("valorant.yaml", "w") as f:
        f.write(yaml_content)
    print("valorant.yaml 建立完成。")

def train_yolov12():
    print("開始訓練 YOLOv12 模型...")
    model = YOLO("yolov12n.pt")
    model.train(
        data="valorant.yaml",
        epochs=100,
        imgsz=512,
        batch=4,
        workers=2,
        amp=False,
        name="valorant-yolo12n6"
    )
    print("訓練完成！")

if __name__ == "__main__":
    prepare_yolo_format()
    create_yaml()
    train_yolov12()
