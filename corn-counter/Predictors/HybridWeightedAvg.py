from .BasePredictor import BasePredictor

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
