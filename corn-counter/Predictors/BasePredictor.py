"""Базовый класс предок для предикторов"""

from abc import ABC, abstractmethod
import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR) 

YOLO_MODEL_PATH = os.path.join(PROJECT_ROOT, 'Yolo', 'corn_yolov8s', 'train', 'weights', 'best.pt')
CSRNET_MODEL_PATH = os.path.join(PROJECT_ROOT, 'CSRNet', 'csrnet_final.pt')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BasePredictor(ABC):
    @abstractmethod
    def predict(self, img_path):
        pass

    @property
    @abstractmethod
    def name(self):
        pass
