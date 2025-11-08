import numpy as np
import xml.etree.ElementTree as ET
import cv2
import os
from typing import Optional, List


def xml_parsing(file_path='annotations.xml'):
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
                     image_dir='dataset/images',
                     output_dir='dataset/visualized_images'):
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
                 output_dir='dataset/label'):
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


if __name__ == '__main__':
    data = xml_parsing('annotations.xml')
    visualize_images(data)
    conv_in_yolo(data)
