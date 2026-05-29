# 🌽 Corn Counter

Automated counting of threshed corn kernels on flat surfaces using **YOLOv8-seg**, **CSRNet**, and hybrid ensemble methods.

## 🎯 Challenges
- High density of objects
- Partial occlusions (overlapping kernels)
- Presence of small debris (dust, dirt)

## 📁 Project Structure
CORN-COUNTER/
├── corn-counter/
│   ├── CSRNet/
│   ├── dataset_COCO/
│   ├── dataset_COCO_split/
│   ├── examples/
│   ├── Predictors/
│   ├── Sources/
│   ├── Yolo/
│   ├── annotations.xml
│   ├── file_manager.py
│   ├── run.py
│   ├── tester.py
│   ├── visualize_work.py
│   └── yolov8s-seg.pt
├── results/
│   ├── YOLO_graphics.png
│   ├── курсовая.docx
│   └── курсовая.pdf
├── venv/
├── .gitattributes
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt

## ▶️ Quick Start

Option A. Step-by-step: 
1. **Prepare environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate    # Windows
   pip install -r requirements.txt
```
2. **Run Pipeline**
You can run steps step by step or all at once.

   ```bash
   cd corn-counter

   # 1. Prepare data (split dataset, generate density maps & labels)
   python run.py --prepare

   # 2. Train models
   python run.py --train-yolo # Start train yolo
   python run.py --train-csrnet # Start train csrnet

   # 3. Evaluate models (calculate MAE/RMSE for all hybrids)
   python run.py --evaluate

   # 4. Visualize predictions for a single image
   python run.py --visualize {path_to_image} # Example counting for 1 image 
```

Option B. All-in-one-command
   ```bash
   cd corn-counter
   python run.py --prepare --train-csrnet --train-yolo --evaluate
```

## 📊 Results & Outputs

    Visualizations: After running --visualize, images with masks, density maps, and count comparisons are saved in the examples/ folder.
    Training Plots: 
        YOLO graphs are located in Yolo/corn_yolov8s/train/results.png | results/YOLO_graphics.png.
        CSRNet training loss/MAE plot is saved as results/csrnet_plot.png.
    Benchmark Metrics: The --evaluate command prints a table with MAE and RMSE for YOLO, CSRNet, and 5 hybrid strategies directly to the console.

**In folder "corn-counter/examples/" you can see some examples of model's predictions**

**In folder "results/" you can see graphics and results of comparing models**




