from .BasePredictor import BasePredictor

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
