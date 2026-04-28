# 🎯 YOLO Computer Vision Learning Hub

A comprehensive learning repository for mastering **YOLO (You Only Look Once)** and computer vision fundamentals with hands-on examples and step-by-step guides.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Learning Modules](#learning-modules)
- [Requirements](#requirements)
- [Resources](#resources)

---

## 🚀 Overview

This repository contains a collection of **practical scripts and tutorials** for learning computer vision using **YOLOv8** (the latest YOLO version). Whether you're a beginner exploring object detection or diving into advanced segmentation tasks, you'll find step-by-step examples to guide your learning journey.

### Key Capabilities
- 🔍 **Object Detection** - Identify and locate objects in images and videos
- 🏷️ **Image Classification** - Categorize images into predefined classes
- 🎭 **Instance Segmentation** - Pixel-level object identification and segmentation
- ⚡ **Nano-models** - Lightweight models optimized for fast inference

---

## ✨ Features

✅ **Step-by-Step Tutorials** - Learn at your own pace with detailed scripts  
✅ **Multiple YOLO Tasks** - Detection, classification, and segmentation examples  
✅ **Pre-trained Models** - Ready-to-use YOLOv8 weights (download on first run)  
✅ **Real-World Examples** - Practical applications with images and videos  
✅ **Performance Optimized** - From Nano to X-Large model variants  
✅ **Beginner Friendly** - Clear code comments and explanations  

---

## 📁 Project Structure

```
CV_Practice/
├── README.md                                    # This file
├── .gitignore                                   # Git ignore rules
└── YOLO/
    ├── YOLO_Detection_Step_by_Step.py          # Object detection tutorial
    ├── YOLO_Detection_Complete_Guide.py        # Advanced detection guide
    ├── YOLO_Classification_Step_by_Step.py     # Image classification tutorial
    ├── YOLO_Segmentation_Step_by_Step.py       # Instance segmentation tutorial
    ├── YOLO_Nano_Classification_Quick_Start.py # Lightweight model quickstart
    ├── test.py                                  # Testing and experimentation
    └── runs/                                    # Output predictions and results
        └── detect/
            ├── predict/                         # Detection results
            └── predict-2/                       # Additional predictions
```

---

## 💻 Installation

### Prerequisites
- **Python 3.8+** installed on your system
- **pip** package manager
- Sufficient storage (~4GB for all models)

### Step 1: Clone the Repository

```bash
git clone https://github.com/FahimShah651/CV_Practice.git
cd CV_Practice
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv yolo-env
yolo-env\Scripts\activate

# macOS/Linux
python3 -m venv yolo-env
source yolo-env/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install ultralytics opencv-python matplotlib numpy
```

Or install all requirements at once:

```bash
pip install -r requirements.txt  # If available
```

---

## 🎓 Getting Started

### Run Your First Object Detection

```bash
cd YOLO
python YOLO_Detection_Step_by_Step.py
```

### Quick Test with Nano Model

```bash
python YOLO_Nano_Classification_Quick_Start.py
```

### Explore Available Models

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| Nano (n) | 3.2M | ⚡⚡⚡ | Good | Edge devices, mobile |
| Small (s) | 11.2M | ⚡⚡ | Better | Balanced performance |
| Medium (m) | 25.9M | ⚡ | Very Good | Standard applications |
| Large (l) | 52.9M | 🐢 | Excellent | High accuracy needed |
| Extra Large (x) | 130.5M | 🐢🐢 | Best | Maximum accuracy |

---

## 📚 Learning Modules

### 1. **Object Detection** 🔍
Learn to identify and locate multiple objects in images.
- **File**: `YOLO_Detection_Step_by_Step.py`
- **Concepts**: Bounding boxes, confidence scores, NMS
- **Output**: Annotated images with detected objects

### 2. **Classification** 🏷️
Classify entire images into categories.
- **File**: `YOLO_Classification_Step_by_Step.py`
- **Concepts**: Class probabilities, top-k predictions
- **Use Cases**: Image categorization, filtering

### 3. **Segmentation** 🎭
Segment objects at the pixel level.
- **File**: `YOLO_Segmentation_Step_by_Step.py`
- **Concepts**: Masks, instance segmentation, semantic boundaries
- **Applications**: Medical imaging, autonomous driving

### 4. **Nano Quick Start** ⚡
Get up and running fast with lightweight models.
- **File**: `YOLO_Nano_Classification_Quick_Start.py`
- **Perfect for**: Beginners, resource-constrained environments

---

## 📦 Requirements

```
ultralytics>=8.0.0          # YOLO library
opencv-python>=4.5.0        # Computer vision tasks
numpy>=1.19.0               # Numerical computing
matplotlib>=3.3.0           # Visualization
torch>=2.0.0                # Deep learning framework
torchvision>=0.15.0         # Vision utilities
```

---

## 🎯 Next Steps

1. ✅ **Install dependencies** - Follow the installation guide
2. 📖 **Read through scripts** - Understand the code structure
3. 🧪 **Run examples** - Execute the step-by-step tutorials
4. 🔧 **Modify parameters** - Experiment with confidence thresholds, IoU values
5. 📸 **Test on your data** - Use your own images and videos
6. 🚀 **Deploy models** - Learn to integrate into applications

---

## 📝 Tips & Best Practices

### For Beginners
- Start with `YOLO_Nano_Classification_Quick_Start.py` - it's the easiest
- Read all comments in the scripts carefully
- Experiment with one parameter at a time

### For Performance
- Use smaller models (Nano, Small) for real-time applications
- Increase confidence threshold to reduce false positives
- Use GPU acceleration when available

### For Custom Data
- Annotate your images using tools like [Roboflow](https://roboflow.com) or [Labelimg](https://github.com/heartexlabs/labelImg)
- Train custom models with your labeled data
- Evaluate performance on test sets

---

## 🔗 Resources

- 📖 [Official YOLO Documentation](https://docs.ultralytics.com/)
- 🎥 [Ultralytics YouTube Channel](https://www.youtube.com/c/Ultralytics)
- 📚 [Computer Vision Fundamentals](https://en.wikipedia.org/wiki/Computer_vision)
- 🤖 [Deep Learning Basics](https://www.deeplearningbook.org/)

---

## 📊 Model Performance Comparison

All models tested on standard benchmarks (COCO dataset):

```
YOLOv8 Models Performance:
├── Nano   - mAP50: 37.3%  | Speed: 6ms
├── Small  - mAP50: 44.9%  | Speed: 16ms
├── Medium - mAP50: 50.2%  | Speed: 34ms
├── Large  - mAP50: 52.9%  | Speed: 53ms
└── XLarge - mAP50: 54.4%  | Speed: 115ms
```

---

## 🤝 Contributing

Have improvements or spotted an issue? Feel free to:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🎓 About

**Created for**: Learning and practicing computer vision with modern deep learning techniques.

**Target Audience**: 
- Students learning CV fundamentals
- Developers building vision applications
- AI enthusiasts exploring YOLO models

---

## ⭐ Show Your Support

If you found this helpful, please consider starring this repository! ⭐

---

**Happy Learning! 🚀**

For questions or discussions, open an issue on GitHub or reach out to the community.
