import numpy as np
import xml.etree.ElementTree as ET
import cv2
import os
from typing import Optional, List
import random
import shutil


def xml_parsing(file_path='corn-counter/annotations.xml'):
    '''
        Парсит xml-файла.
        Возвращает список изображений с полигонами.

        file_path: Путь до annotations.xml
    '''

    tree = ET.parse(file_path)
    root = tree.getroot()
    images = []

    for image in root.findall('.//image'):
        width = int(image.get('width'))
        height = int(image.get('height'))
        name = image.get('name')
        if image is None:
            print(f'{image} не найдено!')
            continue

        polygons = []
        for polygon in image.findall('polygon'):
            label = polygon.get('label')
            if label != 'kernel':
                print(f'{label} - не зернышко!')
                continue

            points = []
            points_str = polygon.get('points')
            for point in points_str.split(';'):
                x, y = map(float, point.split(','))
                points.append([x, y])
            polygons.append(points)

        images.append({
            'name': name,
            'width': width,
            'height': height,
            'polygons': polygons
        })
    return images


def visualize_images(data_images: Optional[List],
                     image_dir='corn-counter/dataset/images',
                     output_dir='corn-counter/dataset/visualized_images'):
    ''' Обводит зернышки на картинках.
        Cоздает картинки по пути output_dirs.

        data_im: Данные картинок в форме словаря: 
                {'name': name,
                'width': width,
                'height': height,
                'polygons': polygons} или массива таких словарей.
        image_dirs: Путь до картинок которые визуализируются.
        output_dirs: Путь куда будут загружены визуализированные картинки.
    '''
    os.makedirs(output_dir, exist_ok=True)

    for img_data in data_images:
        name = img_data['name']
        img_path = os.path.join(image_dir, name)
        img = cv2.imread(img_path)
        if img is None:
            print(f'{img} не найдено!')
            continue

        for pol in img_data['polygons']:
            pts = np.array([pol], dtype=np.int32)
            cv2.polylines(img,
                          pts,
                          color=(0, 255, 0),
                          isClosed=True,
                          thickness=2)
        output_path = os.path.join(output_dir, f'vis_{name}')
        cv2.imwrite(output_path, img)


def conv_in_yolo(data_images: Optional[List],
                 output_dir='corn-counter/dataset/label'):
    '''Конвертация в yolo-формат для Yolo-модели.

        Обводит зернышки на картинках.
        Cоздает картинки по пути output_dirs.

        data_im: Данные картинок в форме словаря:
                {'name': name,
                'width': width,
                'height': height,
                'polygons': polygons} или массива таких словарей.
        image_dirs: Путь до txt файла который получается.
    '''
    os.makedirs(output_dir, exist_ok=True)

    for img_data in data_images:
        name = img_data['name']
        width = img_data['width']
        height = img_data['height']

        labels = []

        for polygon in img_data['polygons']:
            norm_points = []
            for x, y in polygon:
                norm_points.extend([x / width, y / height])

            labels.append(f'0 {' '.join(map(str, norm_points))}\n')

        if labels:
            label_path = os.path.join(output_dir, name.replace('.jpg', '.txt'))
            with open(label_path, 'w') as f:
                f.writelines(labels)
            print(f'Создан файл: {label_path}')


def split_dataset(images_dir='corn-counter/dataset/images',
                  labels_dir='corn-counter/dataset/label',
                  output_dir='corn-counter/dataset_split',
                  train_ratio=0.8,
                  val_ratio=0.1):
    '''Разделение по соотношению 80-10-10 на train/val/test'''
    os.makedirs(output_dir, exist_ok=True)

    structure_folders = ['images/train', 'images/val', 'images/test',
                         'labels/train', 'labels/val', 'labels/test']
    for dir_folder in structure_folders:
        os.makedirs(os.path.join(output_dir, dir_folder), exist_ok=True)

    images_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    random.seed(42)
    random.shuffle(images_files)

    n_total = len(images_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_files = images_files[:n_train] 
    val_files = images_files[n_train:n_val]
    test_files = images_files[n_train+n_val:]

    for file_list, split in [(train_files, 'train'),
                             (val_files, 'val'),
                             (test_files, 'test')]:
        for img_file in file_list:
            file_path_src = os.path.join(images_dir,
                                         f'{img_file}')
            file_path_dst = os.path.join(output_dir,
                                         'images',
                                         split,
                                         img_file)
            shutil.copy(file_path_src, file_path_dst)

            label_file = img_file.replace('.jpg', '.txt')
            label_file_path_src = os.path.join(labels_dir,
                                               label_file)
            if os.path.exists(label_file_path_src):
                label_file_path_dst = os.path.join(output_dir,
                                                   'labels',
                                                   split,
                                                   label_file)
                shutil.copy(label_file_path_src, label_file_path_dst)


if __name__ == '__main__':
    data = xml_parsing('corn-counter/annotations.xml')
    visualize_images(data)
    conv_in_yolo(data)
    split_dataset()
