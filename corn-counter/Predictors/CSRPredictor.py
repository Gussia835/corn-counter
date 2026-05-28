import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

sys.path.append(os.path.join(PROJECT_ROOT, 'CSRNet'))
from csrnet import CSRNet

import torch
from torchvision import transforms
from .BasePredictor import BasePredictor, CSRNET_MODEL_PATH, DEVICE
import cv2

"""Класс предиктор для CSRNet"""


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