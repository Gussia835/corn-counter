from .BasePredictor import BasePredictor

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
