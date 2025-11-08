# corn-counter
Corn kernel detection and segmentation with **YOLOv8-seg**

Automated counting of threshed corn kernels on flat surfaces from images.  
Challenges:
- High density of objects
- Partial occlusions (overlapping kernels)
- Presence of small debris (dust, dirt)

## ▶️ Quick Start

1. **Prepare environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate    # Windows
   pip install -r requirements.txt

2. **Process data**
   ```bash
   python .\corn-counter\parser.py
   python python .\corn-counter\model.py
