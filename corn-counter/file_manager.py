"""Подготовка данных"""

import os
import random
import shutil
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILEPATH_XML = os.path.join(BASE_DIR, 'annotations.xml')

SRC_IMAGES = os.path.join(BASE_DIR, 'dataset_COCO/images')
SRC_YOLO = os.path.join(BASE_DIR, 'dataset_COCO/labels')
DST_SPLIT_IMAGES = os.path.join(BASE_DIR, 'dataset_COCO_split/images')
DST_SPLIT_YOLO = os.path.join(BASE_DIR, 'dataset_COCO_split/labels')
DENSITY_PATH = os.path.join(BASE_DIR, 'dataset_COCO_split/density_maps')


class FileManagerHelper:
  """Класс помощника по работе с файлами и датасетами"""

  def __init__(self,
               src_img=SRC_IMAGES,
               src_label=SRC_YOLO,
               dst_img=DST_SPLIT_IMAGES,
               dst_label=DST_SPLIT_YOLO,
               xml_path=FILEPATH_XML,
               density_path=DENSITY_PATH):
    self.src_img = src_img
    self.src_label = src_label
    self.dst_img = dst_img
    self.dst_label = dst_label
    self.filepath = xml_path
    self.density_path = density_path

    self.splits = {}
    for fold in ['train', 'test', 'val']:
      self.splits[fold] = {
          'img': os.path.join(dst_img, fold),
          'label': os.path.join(dst_label, fold),
          'density': os.path.join(density_path, fold)
      }

    self._create_dirs()

  def _create_dirs(self):
    for fold in self.splits.values():
      os.makedirs(fold['img'], exist_ok=True)
      os.makedirs(fold['label'], exist_ok=True)
      os.makedirs(fold['density'], exist_ok=True)

  def _get_paths_by_type(self, data_type):
    '''Возвращает словарь {split: path} для указанного типа данных'''
    return {split: paths[data_type] for split, paths in self.splits.items()}

  def split_dataset(self,
                    train_ratio=0.8,
                    test_ratio=0.1):
      '''Разделение по заданному соотношению на train/val/test'''
      files = [f for f in os.listdir(self.src_img)]

      random.seed(42)
      random.shuffle(files)

      n_total = len(files)
      n_train = int(train_ratio * n_total)
      n_test = int(test_ratio * n_total)
      n_val = n_total - n_train - n_test

      train_files = files[:n_train]
      val_files = files[n_train:n_train+n_val]
      test_files = files[n_train+n_val:]

      for file_lst, split in [(train_files, 'train'),
                              (val_files, 'val'),
                              (test_files, 'test')]:
          paths = self.splits[split]
          for file in file_lst:
              shutil.copy2(
                  os.path.join(self.src_img, file),
                  os.path.join(paths['img'], file)
              )

              label_name = file.replace('.jpg', '.txt')
              shutil.copy2(
                  os.path.join(self.src_label, label_name),
                  os.path.join(paths['label'], label_name)
              )

  def xml_parsing(self):
      root = ET.parse(self.filepath).getroot()
      images = []

      for image in root.findall('.//image'):
          if image is None:
              continue

          width = int(image.get('width'))
          height = int(image.get('height'))
          name = image.get('name')

          polygons = []
          for polygon in image.findall('polygon'):
              if polygon.get('label') != 'kernel':
                  continue

              points = []
              point_arr = polygon.get('points').split(';')
              for coords in point_arr:
                  x, y = map(float, coords.split(','))
                  points.append([x, y])
              polygons.append(points)

          images.append({
              'name': name,
              'width': width,
              'height': height,
              'polygons': polygons
          })
      return images

  def conv_to_yolo(self, data_images):
      '''Конвертация в yolo-формат для Yolo-модели.'''
      os.makedirs(self.src_label, exist_ok=True)

      for img_data in data_images:
          name = img_data['name']
          lines = []
          for polygon in img_data['polygons']:
              line = ['0']
              for nx, ny in polygon:
                  x_norm = nx / img_data['width']
                  y_norm = ny / img_data['height']

                  line.append(f'{x_norm:.6f}')
                  line.append(f'{y_norm:.6f}')
              lines.append(' '.join(line))

          yolo_path = os.path.join(self.src_label, name.replace('.jpg', '.txt'))
          with open(yolo_path, 'w') as f:
              f.write('\n'.join(lines))

  def visualize_annotations(self,
                            split=None,
                            dst_visualized=None,
                            show=True,
                            num_samples=7):
    '''Обводим зернышки заданной по разметке.'''
    if split:
      paths = self.splits[split]
      img_dir = paths['img']
      label_dir = paths['label']
    else:
      img_dir = self.src_img
      label_dir = self.src_label

    files = [f for f in os.listdir(img_dir)]

    if len(files) == 0:
      print('Нечего визуализировать!')
      return
    sample_files = random.sample(files, min(num_samples, len(files)))

    n_cols = 3
    n_rows = (len(sample_files) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    if n_rows * n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, file in enumerate(sample_files):
      img_path = os.path.join(img_dir, file)
      img = cv2.imread(img_path)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      height, width, _ = img.shape

      label_path = os.path.join(label_dir, file.replace('.jpg', '.txt'))

      with open(label_path, 'r') as f:
        lines = f.readlines()
        kernel_count = 0

        for line in lines:
            coords = line.strip().split()[1:]

            points = []
            for j in range(0, len(coords), 2):
                points.append( (int(float(coords[j])*width), int(float(coords[j+1])*height)) )

            points = np.array(points).reshape((-1, 1, 2))

            cv2.polylines(img, [points],
                        isClosed=True,
                        color=(0, 255, 0),
                        thickness=3)
            kernel_count += 1

        cv2.putText(img, f'{kernel_count} зерен',
                  org=(10, 30),
                  fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                  fontScale=1,
                  color=(255, 0, 0),
                  thickness=2)

      if dst_visualized:
          os.makedirs(dst_visualized, exist_ok=True)
          cv2.imwrite(os.path.join(dst_visualized, file),
                      cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

      axes[i].imshow(img)
      axes[i].set_title(f"{file}\n{kernel_count}", fontsize=9)

    plt.tight_layout()

    for j in range(len(sample_files), len(axes)):
      axes[j].axis('off')

    if show:
      plt.show()
    else:
      plt.close()

  def generate_density_map(self, split='train', sigma_orig=15):
    '''Генерация карты плотности с адаптивным сигма под размер 640'''
    paths = self.splits.get(split)

    img_dir = paths['img']
    label_dir = paths['label']
    density_dir = paths['density']

    files = [f for f in os.listdir(img_dir)]

    for file in files:
      img = cv2.imread(os.path.join(img_dir, file))
      if img is None:
        continue

      h_orig, w_orig = img.shape[:2]
      density_map = np.zeros((h_orig, w_orig), dtype=np.float32)

      label_path = os.path.join(label_dir, file.replace('.jpg', '.txt'))
      centers = []

      with open(label_path, 'r') as f:
        for line in f:
          coords = line.strip().split()[1:]
          xs = [float(coords[i]) * w_orig for i in range(0, len(coords), 2)]
          ys = [float(coords[i]) * h_orig for i in range(1, len(coords), 2)]
          centers.append([np.mean(xs), np.mean(ys)])

      for cx, cy in centers:
          cx_int = int(cx)
          cy_int = int(cy)
          if 0 <= cx_int < w_orig and 0 <= cy_int < h_orig:
              density_map[cy_int, cx_int] = 1.0

      max_side = 640
      scale = max_side / max(h_orig, w_orig)

      current_sigma = max(1.0, sigma_orig / scale)

      density_map = gaussian_filter(density_map, sigma=current_sigma)

      if density_map.sum() > 0:
        density_map = density_map * (len(centers) / density_map.sum())

      np.save(os.path.join(density_dir, file.replace('.jpg', '.npy')), density_map)


# Парсинг annotations.xml
file_manager = FileManagerHelper()

try:
  images_data = file_manager.xml_parsing()
  print('Успешный парсинг')
except Exception as e:
  print(f'Ошибка при парсинге annotations.xml: {e}')


# Конвертация в YOLO-формат 
try:
  file_manager.conv_to_yolo(images_data)
  print('Успешная конвертация в yolo')

except Exception as e:
  print(f'Ошибка при конвертации в yolo')


# Разбиение на train/test/val
try:
  file_manager.split_dataset()
  print('Успешное разбиение dataset')

except Exception as e:
  print(f'Ошибка разбиения dataset: {e}')


# Визуализация по YOLO формату (просто для проверки правильности размтеки)
try:
  file_manager.visualize_annotations()

except Exception as e:
  print(f'Ошибка визуализации разметки: {e}')

# Генерация карт плотностей
file_manager.generate_density_map(split='train')
file_manager.generate_density_map(split='test')
file_manager.generate_density_map(split='val')