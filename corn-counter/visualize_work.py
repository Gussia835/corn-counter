import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO
from torchvision import transforms

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CSRNet'))
from csrnet import CSRNet

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_WEIGHTS = os.path.join(BASE_DIR, 'Yolo', 'corn_yolov8s', 'train', 'weights', 'best.pt')
CSRNET_WEIGHTS = os.path.join(BASE_DIR, 'CSRNet', 'csrnet_final.pt')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_models():
    print("Loading models...")
    yolo = YOLO(YOLO_WEIGHTS) if os.path.exists(YOLO_WEIGHTS) else None
    if not yolo:
        print("YOLO weights not found. YOLO prediction will be skipped.")

    csrnet = None
    if os.path.exists(CSRNET_WEIGHTS):
        csrnet = CSRNet().to(DEVICE)
        csrnet.load_state_dict(torch.load(CSRNET_WEIGHTS, map_location=DEVICE, weights_only=True))
        csrnet.eval()
    else:
        print("CSRNet weights not found. CSRNet prediction will be skipped.")
    return yolo, csrnet

def run_visualization(img_path, yolo, csrnet):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: cannot load image {img_path}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    gt_path = img_path.replace('.jpg', '.txt')
    gt_count = 0
    if os.path.exists(gt_path):
        with open(gt_path, 'r') as f:
            gt_count = sum(1 for line in f if line.strip())

    yolo_count = 0
    yolo_img = img_rgb.copy()
    if yolo:
        res = yolo(img_path, conf=0.4, device=DEVICE, verbose=False)[0]
        if res.masks:
            yolo_count = len(res.masks.xy)
            yolo_img = res.plot()

    csr_count = 0.0
    csr_heatmap = None
    if csrnet:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        max_side = 640
        scale = max_side / max(h, w)
        h_new = max(32, int(round(h * scale / 32)) * 32)
        w_new = max(32, int(round(w * scale / 32)) * 32)
        img_resized = cv2.resize(img_rgb, (w_new, h_new))
        tensor = transform(img_resized).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            density = csrnet(tensor).clamp(min=0.0)
            csr_count = density.sum().item()
            csr_heatmap = density.squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(img_rgb)
    axes[0].set_title(f"Original\nGround Truth: {gt_count}")
    axes[0].axis('off')

    axes[1].imshow(yolo_img)
    axes[1].set_title(f"YOLOv8-seg\nPredicted: {yolo_count}")
    axes[1].axis('off')

    if csr_heatmap is not None:
        im = axes[2].imshow(csr_heatmap, cmap='hot')
        plt.colorbar(im, ax=axes[2])
    axes[2].set_title(f"CSRNet Density\nPredicted: {csr_count:.1f}")
    axes[2].axis('off')

    methods = ['Ground Truth', 'YOLO', 'CSRNet']
    counts = [gt_count, yolo_count, csr_count]
    axes[3].bar(methods, counts, color=['#2c3e50', '#2980b9', '#c0392b'])
    axes[3].set_title("Count Comparison")
    axes[3].set_ylabel("Kernel Count")
    axes[3].grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    out_path = img_path.replace('.jpg', '_result.png')
    plt.savefig(out_path, dpi=150)
    print(f"Result saved to: {out_path}")
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Visualize model predictions')
    parser.add_argument('--img', type=str, required=True, help='Path to input image')
    args = parser.parse_args()

    yolo_model, csr_model = load_models()
    run_visualization(args.img, yolo_model, csr_model)