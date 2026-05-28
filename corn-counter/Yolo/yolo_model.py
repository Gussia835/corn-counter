import os
import torch
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

TRAIN_DATASET = os.path.join(PROJECT_ROOT, 'dataset_COCO_split', 'images', 'train')
YAML_GENERATED = os.path.join(BASE_DIR, 'corn-dataset-generated.yaml')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = YOLO('yolov8s-seg.pt')

print(DEVICE)

"""Генерация YAML с абсолютными путями"""

def generate_yaml():
    """Создаёт YAML-файл с правильными абсолютными путями"""
    dataset_path = os.path.join(PROJECT_ROOT, 'dataset_COCO_split').replace('\\', '/')

    yaml_content = f"""path: {dataset_path}

train: images/train
test: images/test
val: images/val

names:
  0: kernel
"""

    with open(YAML_GENERATED, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print(f'YAML сгенерирован: {YAML_GENERATED}')
    print(f'path: {dataset_path}')
    return YAML_GENERATED

"""Проверка что все работает"""

import cv2
import matplotlib.pyplot as plt

file_path = os.path.join(TRAIN_DATASET, os.listdir(TRAIN_DATASET)[0])
img = cv2.imread(file_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

result = model.predict(
    source=file_path,
    conf=0.25,
    device=DEVICE,
    save=False
)

result[0].plot()
plt.figure(figsize=(10, 8))
plt.imshow(img)
plt.title(f'Найдено {len(result[0].masks)}')
plt.show()

"""Тренируем модель YOLO"""

# Генерируем YAML с правильными путями
yaml_path = generate_yaml()

result = model.train(
    data=yaml_path,
    project=os.path.join(BASE_DIR, 'Yolo'),
    name='corn_yolov8s',
    patience=10,
    epochs=40,
    imgsz=640,
    batch=8,
    device=DEVICE,
    save=True,
    verbose=True,
    amp=True,
    workers=2,
    cache=True
)

# После обучения предсказание
result = model.predict(
    source=file_path,
    conf=0.25,
    device=DEVICE,
    save=False
)

print(f'После обучения: Найдено {len(result[0].masks)}')