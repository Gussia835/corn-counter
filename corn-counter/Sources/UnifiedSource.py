from .BaseSource import BaseSource, IMAGES_TEST, LABELS_TEST
import os

"""Источник для Hybrid моделей"""


class UnifiedSource(BaseSource):
    def __init__(self, file_dir=IMAGES_TEST, label_dir_txt=LABELS_TEST):
        self.file_dir = file_dir
        self.label_dir = label_dir_txt
        self.files = sorted([f for f in os.listdir(file_dir) if f.endswith('.jpg')])

    def __iter__(self):
        for f in self.files:
            img_path = os.path.join(self.file_dir, f)
            txt_path = os.path.join(self.label_dir, f.replace('.jpg', '.txt'))

            try:
                with open(txt_path, 'r') as file:
                    true_count = sum(1 for line in file if line.strip())

            except FileNotFoundError:
                continue

            yield img_path, true_count
