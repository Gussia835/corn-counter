from .BasePredictor import BasePredictor

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