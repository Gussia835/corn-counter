"""Класс предиктор для yolo"""

from ultralytics import YOLO
from .BasePredictor import BasePredictor, YOLO_MODEL_PATH, DEVICE


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