import os
from abc import ABC, abstractmethod

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR) 

IMAGES_TEST = os.path.join(PROJECT_ROOT, 'dataset_COCO_split', 'images', 'test')
DENSITY_TEST = os.path.join(PROJECT_ROOT, 'dataset_COCO_split', 'density_maps', 'test')
LABELS_TEST = os.path.join(PROJECT_ROOT, 'dataset_COCO_split', 'labels', 'test')

"""Базовый класс для тестеров"""


class BaseSource(ABC):
    def __init__(self, file_dir=IMAGES_TEST, label_dir=LABELS_TEST):
        self.file_dir = file_dir
        self.label_dir = label_dir

        self.files = [f for f in os.listdir(file_dir) if f.endswith('.jpg')]

    @abstractmethod
    def __iter__(self):
        pass
