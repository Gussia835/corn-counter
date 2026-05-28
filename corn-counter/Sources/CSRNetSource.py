from .BaseSource import BaseSource, IMAGES_TEST, DENSITY_TEST
import numpy as np
import os

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
