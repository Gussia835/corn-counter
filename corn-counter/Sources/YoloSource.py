from .BaseSource import BaseSource, IMAGES_TEST, LABELS_TEST
import os

"""Класс с файлами для yolo модели"""


class YoloSource(BaseSource):
    def __init__(self, file_dir=IMAGES_TEST, label_dir=LABELS_TEST):
        super().__init__(file_dir, label_dir)

    def __iter__(self):
        for f in self.files:
            file_path = os.path.join(self.file_dir, f)
            label_path = os.path.join(self.label_dir, f.replace('.jpg', '.txt'))

            try:
                with open(label_path, 'r') as file:
                    true_count = sum(1 for line in file if line.strip())

            except FileNotFoundError:
                continue

            yield file_path, true_count