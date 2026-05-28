import os, sys, cv2
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR) 

# Пример для путей:
IMAGES_TEST = os.path.join(PROJECT_ROOT, 'dataset_COCO_split', 'images', 'test')
DENSITY_TEST = os.path.join(PROJECT_ROOT, 'dataset_COCO_split', 'density_maps', 'test')
LABELS_TEST = os.path.join(PROJECT_ROOT, 'dataset_COCO_split', 'labels', 'test')
YOLO_MODEL_PATH = os.path.join(PROJECT_ROOT, 'Yolo', 'corn_yolov8s', 'train', 'weights', 'best.pt')
CSRNET_MODEL_PATH = os.path.join(PROJECT_ROOT, 'CSRNet', 'csrnet_final.pt')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""Базовый класс предок для предикторов"""

from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm

class BasePredictor(ABC):
  @abstractmethod
  def predict(self, img_path):
      pass

  @property
  @abstractmethod
  def name(self):
      pass

"""Класс предиктор для yolo"""

from ultralytics import YOLO

class YoloPredictor(BasePredictor):
  def __init__(self, model_path=YOLO_MODEL_PATH, conf=0.4, device=DEVICE):
      self.model = YOLO(model_path)
      self.conf_thresh = conf
      self.device = device

  @property
  def name(self):
      return 'yolov8-seg'

  def predict(self, img_path):
      results = self.model.predict(
          source=img_path,
          conf=self.conf_thresh,
          device=self.device,
          verbose=False
      )
      masks = results[0].masks

      return len(masks.xy) if masks is not None and masks.xy is not None else 0

"""Класс предиктор для CSRNet"""

sys.path.append(os.path.join(BASE_DIR, 'CSRNet'))
from csrnet import CSRNet
from torchvision import transforms

class CSRNetPredictor(BasePredictor):
    def __init__(self, model_path=CSRNET_MODEL_PATH, device=DEVICE):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.transformations = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        self.model = CSRNet().to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()

    @property
    def name(self):
        return 'CSRNet'

    def predict(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            return 0
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h_orig, w_orig = img.shape[:2]
        max_side = 640
        scale = max_side / max(h_orig, w_orig)
        h_new = max(32, int(round(h_orig * scale / 32)) * 32)
        w_new = max(32, int(round(w_orig * scale / 32)) * 32)

        img_resized = cv2.resize(img, (w_new, h_new))
        img_tensor = self.transformations(img_resized).unsqueeze(0).to(self.device)

        with torch.no_grad():
            y_pred = self.model(img_tensor)
            y_pred = torch.clamp(y_pred, min=0.0)
            count = y_pred.sum().item()

        return round(count)

"""Базовый класс для тестеров"""

class BaseSource(ABC):
  def __init__(self, file_dir=IMAGES_TEST, label_dir=LABELS_TEST):
      self.file_dir = file_dir
      self.label_dir = label_dir

      self.files = [f for f in os.listdir(file_dir) if f.endswith('.jpg')]

  @abstractmethod
  def __iter__(self):
      pass

"""Класс с файлами для yolo модели"""

class YoloSource(BaseSource):
  def __init__(self, file_dir=IMAGES_TEST, label_dir=LABELS_TEST):
      super().__init__(file_dir, label_dir)

  def __iter__(self):
      for f in self.files:
          file_path = os.path.join(self.file_dir, f)
          label_path = os.path.join(self.label_dir, f.replace('.jpg', '.txt'))
          try:
              with open(label_path, 'r') as file:
                  true_count = sum(1 for line in file if line.strip())
          except FileNotFoundError:
              continue
          yield file_path, true_count

"""Класс с файлами для CSRNet модели"""

class CSRNetSource(BaseSource):
  def __init__(self, file_dir=IMAGES_TEST, label_dir=DENSITY_TEST):
      super().__init__(file_dir, label_dir)

  def __iter__(self):
      for f in self.files:
          file_path = os.path.join(self.file_dir, f)
          label_path = os.path.join(self.label_dir, f.replace('.jpg', '.npy'))
          try:
              density = np.load(label_path)
              true_count = int(round(density.sum()))
          except FileNotFoundError:
              continue
          yield file_path, true_count

"""Класс тестировщик моделей"""

import pandas as pd

class Benchmark:
  def evaluate_models(self, predictors, test_sources):
      results = {}
      for predictor, src in zip(predictors, test_sources):
          mae, rmse = [], []

          for img_file, target in tqdm(src, desc=predictor.name):
              pred = predictor.predict(img_file)
              err = abs(pred - target)
              mae.append(err)
              rmse.append(err**2)

          results[predictor.name] = {
              'samples': len(mae),
              'MAE': np.mean(mae),
              'RMSE': np.sqrt(np.mean(rmse))
          }
      return results

models = [
    YoloPredictor(model_path=YOLO_MODEL_PATH, conf=0.4, device=DEVICE),
    CSRNetPredictor(model_path=CSRNET_MODEL_PATH, device=DEVICE)
]

sources = [
    YoloSource(file_dir=IMAGES_TEST, label_dir=LABELS_TEST),
    CSRNetSource(file_dir=IMAGES_TEST, label_dir=DENSITY_TEST)
]

torch.cuda.empty_cache()
bench = Benchmark()
results = bench.evaluate_models(models, sources)
print('\n')
print(pd.DataFrame(results))

"""Hybrid-модель Простое среднее арифметическое двух моделей"""

class HybridAvg(BasePredictor):
  def __init__(self, yolo, csr):
      self.yolo, self.csr = yolo, csr

  @property
  def name(self):
      return 'Hybrid-Avg'

  def predict(self, img_path):
      y = self.yolo.predict(img_path)
      c = self.csr.predict(img_path)
      return round((y + c) / 2)

"""Больший вес у модели с лучшей исторической точностью"""

class HybridWeighted(BasePredictor):
  def __init__(self, yolo, csr, w_yolo=0.3, w_csr=0.7):
      self.yolo, self.csr = yolo, csr
      self.w_yolo, self.w_csr = w_yolo, w_csr

  @property
  def name(self):
      return f'Hybrid-W({self.w_yolo:.1f}/{self.w_csr:.1f})'

  def predict(self, img_path):
      y = self.yolo.predict(img_path)
      c = self.csr.predict(img_path)
      return round(self.w_yolo * y + self.w_csr * c)

"""если CSRNet насчитал значительно больше то доверяем ему"""

class HybridSwitch(BasePredictor):
  def __init__(self, yolo, csr, ratio_thresh=1.2):
      self.yolo, self.csr = yolo, csr
      self.thresh = ratio_thresh

  @property
  def name(self):
      return f'Hybrid-Switch(r>{self.thresh:.1f})'

  def predict(self, img_path):
      y = self.yolo.predict(img_path)
      c = self.csr.predict(img_path)
      return c if c > y * self.thresh else y

"""Ограничение расхождения: если модели сильно расходятся берем CSRNet, иначе YOLO"""

class HybridClip(BasePredictor):
  def __init__(self, yolo, csr, margin=0.2):
      self.yolo, self.csr = yolo, csr
      self.margin = margin

  @property
  def name(self):
      return f'Hybrid-Clip(m={self.margin:.1f})'

  def predict(self, img_path):
      y = self.yolo.predict(img_path)
      c = self.csr.predict(img_path)

      if y == 0:
          return c
      return c if abs(c - y) > y * self.margin else y

"""В областях с перекрытиям использовать csrnet а без перекрытия yolo"""

from shapely.geometry import Polygon

class HybridPartialModels(BasePredictor):
    def __init__(self, yolo, csrnet, IoU_thresold=0.3, conf_thresold=0.4, padding=20, device=DEVICE):
        self.yolo = yolo
        self.csrnet = csrnet
        self.IoU_thresold = IoU_thresold
        self.conf_thresold = conf_thresold
        self.device = device
        self.padding = padding

    @property
    def name(self):
        return 'Partial Hybrid'

    def _iou(self, pnt1, pnt2):
        # считаем IoU через shapely, с фолбэком на bbox-версию
        try:
            poly1, poly2 = Polygon(pnt1), Polygon(pnt2)
            if not poly1.is_valid:
                poly1 = poly1.buffer(0)
            if not poly2.is_valid:
                poly2 = poly2.buffer(0)
            inter = poly1.intersection(poly2).area
            union = poly1.union(poly2).area
            return inter / union if union else 0.0
        except:
            x1_1, y1_1 = pnt1.min(axis=0)
            x2_1, y2_1 = pnt1.max(axis=0)
            x1_2, y1_2 = pnt2.min(axis=0)
            x2_2, y2_2 = pnt2.max(axis=0)
            iw = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
            ih = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
            inter_area = iw * ih
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union_area = area1 + area2 - inter_area
            return inter_area / union_area if union_area else 0.0

    def _get_overlap(self, masks):
        # группируем маски, которые пересекаются
        groups = []
        processed = [False] * len(masks)
        for i in range(len(masks)):
            if processed[i]:
                continue
            group = [i]
            processed[i] = True
            for j in range(i+1, len(masks)):
                if not processed[j] and self._iou(masks[i], masks[j]) > self.IoU_thresold:
                    group.append(j)
                    processed[j] = True
            if len(group) > 1:
                groups.append(group)
        return groups

    def _pred_csrnet(self, crop):
      h_orig, w_orig = crop.shape[:2]
      max_side = 640
      scale = max_side / max(h_orig, w_orig)

      h_new = max(32, int(round(h_orig * scale / 32)) * 32)
      w_new = max(32, int(round(w_orig * scale / 32)) * 32)

      crop_resized = cv2.resize(crop, (w_new, h_new))
      tensor = self.csrnet.transformations(crop_resized).unsqueeze(0).to(self.device)

      with torch.no_grad():
          density_map = self.csrnet.model(tensor)
          density_map = torch.clamp(density_map, min=0.0)
          raw_sum = density_map.sum().item()

      # если кроп был маленьким, модель могла переоценить
      correction = min(1.0, (h_orig * w_orig) / (640 * 640))
      return max(0, round(raw_sum * correction))

    def predict(self, img_path):
        result = self.yolo.model.predict(
            img_path, conf=self.conf_thresold, verbose=False, device=self.device
        )

        # если YOLO ничего не нашёл - сразу идём в CSRNet
        if len(result[0].boxes) == 0:
            return self.csrnet.predict(img_path)

        masks = result[0].masks.xy
        groups = self._get_overlap(masks)

        # Если перекрытий нет - просто считаем маски
        if not groups:
            return len(masks)

        # Исключаем маски из групп перекрытий
        overlap_groups = {idx for g in groups for idx in g}
        total = len(masks) - len(overlap_groups)

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # Для каждой группы - кропим область и считаем через CSRNet
        for g in groups:
            pts = np.vstack([masks[i] for i in g])
            x1, y1 = pts.min(axis=0) - self.padding
            x2, y2 = pts.max(axis=0) + self.padding
            crop = img[max(int(y1), 0):min(int(y2), h), max(int(x1), 0):min(int(x2), w)]
            if crop.size > 0:
              total += self._pred_csrnet(crop)
        return round(total)

"""Источник для Hybrid моделей"""

class UnifiedSource(BaseSource):
  def __init__(self, file_dir=IMAGES_TEST, label_dir_txt=LABELS_TEST):
      self.file_dir = file_dir
      self.label_dir = label_dir_txt
      self.files = sorted([f for f in os.listdir(file_dir) if f.endswith('.jpg')])

  def __iter__(self):
      for f in self.files:
          img_path = os.path.join(self.file_dir, f)
          txt_path = os.path.join(self.label_dir, f.replace('.jpg', '.txt'))
          try:
              with open(txt_path, 'r') as file:
                  true_count = sum(1 for line in file if line.strip())
          except FileNotFoundError:
              continue
          yield img_path, true_count

"""Тестирование Hybrid-моделей"""

hybrid_models = [
    HybridAvg(models[0], models[1]),
    HybridWeighted(models[0], models[1], w_yolo=0.3, w_csr=0.7),
    HybridSwitch(models[0], models[1], ratio_thresh=1.2),
    HybridClip(models[0], models[1], margin=0.2),
    HybridPartialModels(yolo=models[0], csrnet=models[1])
]

all_predictors = [models[0], models[1]] + hybrid_models
gt_source = UnifiedSource(file_dir=IMAGES_TEST, label_dir_txt=LABELS_TEST)
all_sources = [gt_source] * len(all_predictors)

torch.cuda.empty_cache()
bench = Benchmark()
results = bench.evaluate_models(all_predictors, all_sources)

df = pd.DataFrame(results).T
df_sorted = df.sort_values('MAE')
print(df_sorted)
