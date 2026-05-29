
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

SRC_IMAGES = os.path.join(PROJECT_ROOT, 'dataset_COCO_split', 'images')
SRC_DENSITY = os.path.join(PROJECT_ROOT, 'dataset_COCO_split', 'density_maps')
DENSITY_TEST = os.path.join(PROJECT_ROOT, 'dataset_COCO_split', 'density_maps', 'test')
SRC_LABELS = os.path.join(PROJECT_ROOT, 'dataset_COCO_split', 'labels')

CSRNET_MODEL_PATH = os.path.join(BASE_DIR, 'csrnet_final.pt')

"""Описание модели"""

import torch
import torch.nn as nn
from torchvision import models

class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:10])
        self.backend = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64,
                      kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,  32,  kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1,
                      kernel_size=1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.backend.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return x

"""класс датасета"""

import torch.utils.data as data
import os
import cv2
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import random

class ClassDataset(data.Dataset):
    def __init__(self, img_dir, density_dir, augment=False):
        self.img_dir = img_dir
        self.density_dir = density_dir
        self.augment = augment
        self.files = [f for f in os.listdir(img_dir)]
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = os.path.join(self.img_dir, fname)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        density_path = os.path.join(self.density_dir, fname.replace('.jpg', '.npy'))
        density = np.load(density_path)

        h_orig, w_orig = img.shape[:2]
        max_side = 640
        scale = max_side / max(h_orig, w_orig)
        h_new = max(32, int(round(h_orig * scale / 32)) * 32)
        w_new = max(32, int(round(w_orig * scale / 32)) * 32)

        img = cv2.resize(img, (w_new, h_new))
        output_h = h_new // 4
        output_w = w_new // 4

        original_area = density.shape[0] * density.shape[1]
        new_area = output_h * output_w
        area_ratio = original_area / new_area

        density = cv2.resize(density, (output_w, output_h), interpolation=cv2.INTER_CUBIC)
        density = density * area_ratio

        if self.augment and random.random() > 0.5:
            img = cv2.flip(img, 1)
            density = cv2.flip(density, 1)

        img = self.img_transform(img)
        density = torch.from_numpy(density.copy()).float().unsqueeze(0)
        return img, density

"""Унифицированный датасет"""

class TxtCountDataset(data.Dataset):
    def __init__(self, img_dir, label_dir_txt, augment=False):
        self.img_dir = img_dir
        self.label_dir = label_dir_txt
        self.augment = False
        self.files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = os.path.join(self.img_dir, fname)
        txt_path = os.path.join(self.label_dir, fname.replace('.jpg', '.txt'))

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h_orig, w_orig = img.shape[:2]
        max_side = 640
        scale = max_side / max(h_orig, w_orig)
        h_new = max(32, int(round(h_orig * scale / 32)) * 32)
        w_new = max(32, int(round(w_orig * scale / 32)) * 32)

        img = cv2.resize(img, (w_new, h_new))

        with open(txt_path, 'r') as f:
            true_count = sum(1 for line in f if line.strip())

        img = self.img_transform(img)
        return img, float(true_count)

"""для batch

"""

import torch.nn.functional as F

def pad_collate(batch):
    imgs = [item[0] for item in batch]
    densities = [item[1] for item in batch]

    max_h = max(img.shape[1] for img in imgs)
    max_w = max(img.shape[2] for img in imgs)
    max_dh = max(d.shape[1] for d in densities)
    max_dw = max(d.shape[2] for d in densities)

    padded_imgs = []
    padded_densities = []

    for img, density in batch:
        pad_w = max_w - img.shape[2]
        pad_h = max_h - img.shape[1]
        padded_img = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
        padded_imgs.append(padded_img)

        pad_dw = max_dw - density.shape[2]
        pad_dh = max_dh - density.shape[1]
        padded_density = F.pad(density, (0, pad_dw, 0, pad_dh), mode='constant', value=0)
        padded_densities.append(padded_density)

    return torch.stack(padded_imgs, dim=0), torch.stack(padded_densities, dim=0)


def combined_loss(pred, target, alpha=0.1):
    mse = nn.MSELoss()(pred, target)
    pred_count = pred.sum(dim=[1, 2, 3])
    target_count = target.sum(dim=[1, 2, 3])
    count_loss = nn.L1Loss()(pred_count, target_count)
    return mse + alpha * count_loss

"""Задаю начальные параметры для обучения"""

BATCH_SIZE = 4
LR = 1e-4
EPOCHS = 200
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""Подготовка датасета"""

d_train = ClassDataset(img_dir=os.path.join(SRC_IMAGES, 'train'), density_dir=os.path.join(SRC_DENSITY, 'train'))
d_val_txt_dataset = TxtCountDataset(img_dir=os.path.join(SRC_IMAGES, 'val'), label_dir_txt=os.path.join(SRC_LABELS, 'val'))

train_data = DataLoader(d_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=pad_collate)

val_loader_txt = DataLoader(d_val_txt_dataset, batch_size=1, shuffle=False, num_workers=2)

print('Данные подготовлены')

model = CSRNet().to(DEVICE)
optim = torch.optim.Adam(model.parameters(), lr=LR)
loss_func = combined_loss

"""Обучение"""

import matplotlib.pyplot as plt

if __name__ == '__main__':
    best_val_mae = float('inf')
    patience = 30
    patience_counter = 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=10)
    
    train_losses = []
    val_maes = []

    for epoch in range(EPOCHS):
        train_loss = 0
        model.train()

        for x, y in train_data:
            x_train, y_train = x.to(DEVICE), y.to(DEVICE)
            y_pred = model(x_train)
            y_pred = torch.clamp(y_pred, min=0.0)
            err = loss_func(y_pred, y_train)
            optim.zero_grad()
            err.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            train_loss += err.item()

        train_loss /= len(train_data)
        train_losses.append(train_loss) 

        model.eval()
        val_mae_sum = 0.0
        scale_ratios = []

        with torch.no_grad():
            for img, true_count_scalar in val_loader_txt:
                img = img.to(DEVICE)
                true_count = true_count_scalar.item()
                y_pred_map = model(img)
                pred_count = torch.clamp(y_pred_map, min=0.0).sum().item()

                if pred_count > 1e-6 and true_count > 0:
                    scale_ratios.append(true_count / pred_count)
                val_mae_sum += abs(pred_count - true_count)

        val_mae = val_mae_sum / len(val_loader_txt)
        val_maes.append(val_mae)
        
        avg_ratio = np.mean(scale_ratios) if scale_ratios else 1.0

        print(f'Epoch {epoch+1}/{EPOCHS}: Train Loss: {train_loss:.6f}, Val MAE: {val_mae:.2f} | Scale Ratio: {avg_ratio:.2f}')

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            os.makedirs(os.path.dirname(CSRNET_MODEL_PATH), exist_ok=True)
            torch.save(model.state_dict(), CSRNET_MODEL_PATH)
            print(f'Saved best model with MAE: {best_val_mae:.2f}')
        else:
            patience_counter += 1

        scheduler.step(val_mae)

        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break 

    # ПОСТРОЕНИЕ ГРАФИКОВ
    epochs_range = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, 'b-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, val_maes, 'r-', label='Validation MAE')
    plt.title('Validation MAE (Counting Error)')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    
    plot_path = CSRNET_MODEL_PATH.replace('.pt', '_plot.png')
    plt.savefig(plot_path)
    print(f'Training plot saved to: {plot_path}')
    
    plt.show()

import os
density_files = os.listdir(DENSITY_TEST)
print(f"Всего density maps: {len(density_files)}")

sums = []
for f in density_files[:5]:
    path = os.path.join(DENSITY_TEST, f)
    density = np.load(path)
    sums.append(density.sum())
    print(f"{f}: sum = {density.sum():.2f}")

print(f'Средняя сумма: {np.mean(sums):.2f}')



model_check = CSRNet().to('cpu')
model_check.load_state_dict(torch.load(CSRNET_MODEL_PATH, map_location='cpu'))

first_weight = model_check.frontend[0].weight
print(f"Min: {first_weight.min():.4f}")
print(f"Max: {first_weight.max():.4f}")
print(f"Mean: {first_weight.mean():.4f}")
print(f"Std: {first_weight.std():.4f}")