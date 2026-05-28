"""В областях с перекрытиям использовать csrnet а без перекрытия yolo"""

from shapely.geometry import Polygon
from .BasePredictor import BasePredictor, DEVICE
import torch
import cv2
import numpy as np


class HybridPartialModels(BasePredictor):
    def __init__(self, yolo, csrnet,
                 IoU_thresold=0.3,
                 conf_thresold=0.4,
                 padding=20,
                 device=DEVICE):

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
            img_path,
            conf=self.conf_thresold,
            verbose=False,
            device=self.device
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