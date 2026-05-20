<div align="center">

<img src="https://img.shields.io/badge/Computer%20Vision-6%20Month%20Roadmap-1D6A9E?style=for-the-badge&logo=opencv&logoColor=white"/>

# 🎯 Computer Vision Mastery — 6-Month Roadmap

### Zero to Advanced · Daily Activity Plan · 130 Working Days

### Author Information

- **Author:** Fahim Ur Rehman Shah
- **Supervisor:** Dr. Yasir Jan

### Academic Background

- **Degree:** B.S. Electrical Engineering*Namal University, Mianwali*
- **Graduate Study:** M.S. Electrical Engineering
  *Ghulam Ishaq Khan Institute (GIKI)*

### Professional Training

- **PEC GET Training:**
  *Sir Syed CASE Institute of Technology*

[![Phase 1](https://img.shields.io/badge/Phase%201-Foundations-1D6A9E?style=flat-square)](./01_April)
[![Phase 2](https://img.shields.io/badge/Phase%202-Classical%20ML-0F6E56?style=flat-square)](./02_May)
[![Phase 3](https://img.shields.io/badge/Phase%203-Deep%20Learning-533AB7?style=flat-square)](./03_June)
[![Phase 4](https://img.shields.io/badge/Phase%204-Advanced%20DL-BA7517?style=flat-square)](./04_July)
[![Phase 5](https://img.shields.io/badge/Phase%205-Specialization-993C1D?style=flat-square)](./05_Agust)
[![Phase 6](https://img.shields.io/badge/Phase%206-Capstone-3B6D11?style=flat-square)](./06_September)

</div>

---

## 📌 About This Repository

This repository is a **self-taught, structured Computer Vision curriculum** built over 6 months, following a daily Mon–Fri schedule. Every folder maps to a specific week and day. Every concept is implemented from scratch **before** using a library shortcut. The goal is not just to run models — it's to **understand what's happening inside them**.

> **Philosophy:** If you can't implement it from scratch, you don't understand it yet. Libraries come after understanding.

---

## 🗓️ Schedule

| Block           | Time               | Focus                                                |
| --------------- | ------------------ | ---------------------------------------------------- |
| 🌅 AM Session   | 9:00 AM – 1:00 PM | Deep theory + new concepts + scratch implementations |
| ☕ Break        | 1:00 PM – 2:00 PM | Rest                                                 |
| 🖥️ PM Session | 2:00 PM – 6:00 PM | Projects, coding practice, experimentation & review  |
| 📅 Days         | Monday – Friday   | 8 effective hours/day · 130 days total              |

---

## 📝 Today's Attendance

- 2026-05-14: Remote work (work from home)

---

## 📊 Roadmap Overview

```
Month 1 ──► Python + Math + Classical Image Processing (Weeks 1–4)
Month 2 ──► Classical Machine Learning for Vision (Weeks 5–8)
Month 3 ──► Deep Learning + CNNs + Detection (Weeks 9–12)
Month 3.5 ► Vision Transformers + Attention (Weeks 13–14)
Month 4 ──► GANs + 3D Vision + Self-supervised + Deployment (Weeks 15–18)
Month 5 ──► Specialization: Face, Pose, OCR, Autonomous (Weeks 19–22)
Month 6 ──► Paper Implementation + Capstone + Portfolio (Weeks 23–26)
```

---

## 🗂️ Repository Structure

```
CV_Practice/
│
├── README.md                    ← You are here
│
├── 01_April/                    ← Phase 1: Python & Math Foundations
│   ├── Week_01/                 ← Python & NumPy for Images
│   │   ├── 01_file_of_the_day_1/
│   │   ├── 02_file_of_the_day_2/
│   │   ├── 03_file_of_the_day_3/
│   │   ├── 04_file_of_the_day_4/
│   │   └── 05_file_of_the_day_5/
│   ├── Week_02/                 ← Image Processing Fundamentals
│   ├── Week_03/                 ← Geometric Transforms & Contours
│   └── Week_04/                 ← Feature Detection & Description
│
├── 02_May/                      ← Phase 2: Classical ML for Vision
│   ├── Week_01/                 ← ML Fundamentals
│   ├── Week_02/                 ← Bag of Visual Words & Clustering
│   │   ├── 04_CV Evaluation Metrics.ipynb
│   ├── Week_03/                 ← Video Analysis & Motion
│   └── Week_04/                 ← Phase 2 Integration
│
├── 03_June/                     ← Phase 3: Deep Learning & CNNs
│   ├── Week_01/                 ← Neural Networks Foundations
│   ├── Week_02/                 ← Classic CNN Architectures
│   ├── Week_03/                 ← Object Detection
│   ├── Week_04/                 ← Segmentation
│   ├── Week_05/                 ← Vision Transformers (Week 13)
│   └── Week_06/                 ← ViT Capstone (Week 14)
│
├── 04_July/                     ← Phase 4: Advanced Deep Learning
│   ├── Week_01/                 ← Generative Models (GANs/VAEs)
│   ├── Week_02/                 ← 3D Computer Vision
│   ├── Week_03/                 ← Self-supervised Learning
│   └── Week_04/                 ← Model Optimization & Deployment
│
├── 05_Agust/                    ← Phase 5: Specialization Tracks
│   ├── Week_01/                 ← Face Recognition & Biometrics
│   ├── Week_02/                 ← Pose Estimation & Body Tracking
│   ├── Week_03/                 ← OCR & Document AI
│   └── Week_04/                 ← Autonomous Vehicle & Scene Understanding
│
├── 06_September/                ← Phase 6: Capstone Projects & Portfolio
│   ├── Week_01/                 ← Research Paper Implementation
│   ├── Week_02/                 ← Major Capstone Project (Part 1)
│   ├── Week_03/                 ← Major Capstone Project (Part 2)
│   └── Week_04/                 ← Interview Preparation & Launch
│
├── YOLO/                        ← YOLO experiments & practice
│
└── requirements.txt
```

---

## ⚙️ Environment Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended for Phase 3+)
- Anaconda or Miniconda

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/cv-roadmap.git
cd cv-roadmap

# Create virtual environment
conda create -n cv-roadmap python=3.10
conda activate cv-roadmap

# Install all dependencies
pip install -r requirements.txt

# Verify OpenCV
python -c "import cv2; print('OpenCV:', cv2.__version__)"

# Verify PyTorch + CUDA
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

### Core Requirements (`requirements.txt`)

```text
# Core
numpy>=1.24
opencv-python>=4.8
matplotlib>=3.7
scikit-learn>=1.3
scikit-image>=0.21
Pillow>=10.0
scipy>=1.11

# Deep Learning
torch>=2.1
torchvision>=0.16
torchaudio>=2.1
timm>=0.9

# Object Detection
ultralytics>=8.0        # YOLOv8

# Experiment Tracking
wandb>=0.16

# Deployment & APIs
fastapi>=0.104
uvicorn>=0.24
gradio>=4.0
streamlit>=1.28

# Data & Augmentation
albumentations>=1.3
roboflow>=1.1

# 3D Vision
open3d>=0.17

# Generative
diffusers>=0.24
transformers>=4.35
accelerate>=0.24

# OCR
easyocr>=1.7
pytesseract>=0.3

# Pose & Face
mediapipe>=0.10

# Visualization
plotly>=5.17
seaborn>=0.13
```

---

## 📚 Full Curriculum — Phase by Phase

---

## 🔵 Phase 1 — [Python &amp; Math Foundations for CV](./01_April)

> **Weeks 1–4 · April**
> Build rock-solid foundations in Python, NumPy, linear algebra, and classical image processing.

<details>
<summary><b>📅 Week 1 — [Python & NumPy for Images](./01_April/Week_01)</b></summary>

| Day | Topic                        | Key Concepts                                           | Deliverable                              |
| --- | ---------------------------- | ------------------------------------------------------ | ---------------------------------------- |
| Mon | Environment Setup + NumPy    | Anaconda, OpenCV install, NumPy arrays, broadcasting   | Working environment + NumPy exercises    |
| Tue | Image Representation         | Images as arrays, RGB/BGR/Grayscale, OpenCV read/write | Image manipulation script                |
| Wed | Linear Algebra for CV        | Matrices, determinants, eigenvalues, 2D transforms     | Rotation/scaling from scratch            |
| Thu | Probability & Statistics     | Gaussian, Bayes, histograms, Otsu thresholding         | Histogram equalization demo              |
| Fri | Calculus Intuition + Project | Gradients, gradient descent, chain rule                | **Project: Image Stats Dashboard** |

**Daily Attendance Tracking:**

| Day | Topic                        | File                                                                                                                                                                                         | Remote | Campus |
| --- | ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | ------ |
| Mon | Environment Setup + NumPy    | [01_Python environment setup + NumPy basics.ipynb](https://github.com/FahimShah651/CV_Practice/blob/main/01_April/Week_01/01_Python%20environment%20setup%20+%20NumPy%20basics.ipynb)           |        | Campus |
| Tue | Image Representation         | [02_Image representation with NumPy.ipynb](https://github.com/FahimShah651/CV_Practice/blob/main/01_April/Week_01/02_Image%20representation%20with%20NumPy.ipynb)                               |        | Campus |
| Wed | Linear Algebra for CV        | [03_Linear algebra for CV.ipynb](https://github.com/FahimShah651/CV_Practice/blob/main/01_April/Week_01/03_Linear%20algebra%20for%20CV.ipynb)                                                   |        | Campus |
| Thu | Probability & Statistics     | [04_Probability &amp; statistics for CV.ipynb](https://github.com/FahimShah651/CV_Practice/blob/main/01_April/Week_01/04_Probability%20%26%20statistics%20for%20CV.ipynb)                       |        | Campus |
| Fri | Calculus Intuition + Project | [05_Calculus intuition for CV + Week 1 project.ipynb](https://github.com/FahimShah651/CV_Practice/blob/main/01_April/Week_01/05_Calculus%20intuition%20for%20CV%20+%20Week%201%20project.ipynb) | -      | Campus |

**Week 1 Project:** Image Stats Dashboard — input any image, output histogram, mean, std, min/max, channel breakdown with Matplotlib visualization.

</details>

<details>
<summary><b>📅 Week 2 — [Image Processing Fundamentals](./01_April/Week_02)</b></summary>

| Day | Topic                   | Key Concepts                                           | Deliverable                                     |
| --- | ----------------------- | ------------------------------------------------------ | ----------------------------------------------- |
| Mon | Color Spaces            | RGB, BGR, HSV, LAB, YCrCb — conversions and use cases | Color-range object detector                     |
| Tue | Filtering & Convolution | Kernel math, box/Gaussian/median/bilateral filter      | Noise removal comparisons                       |
| Wed | Edge Detection          | Sobel, Prewitt, Laplacian, Canny pipeline              | Canny from scratch                              |
| Thu | Morphological Ops       | Erosion, dilation, opening, closing, top-hat           | Shape isolation tasks                           |
| Fri | Project                 | Combine all ops into a reusable class                  | **Project: Image Preprocessing Pipeline** |

**Daily Attendance Tracking:**

| Day | Topic                   | File                                                                                                                         | Remote | Campus |
| --- | ----------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ------ | ------ |
| Mon | Color Spaces            | [01_Color Spaces &amp; Channel Processing.ipynb](./01_April/Week_02/01_Color%20Spaces%20%26%20Channel%20Processing.ipynb)       |        | Campus |
| Tue | Filtering & Convolution | [02_Image Filtering &amp; Convolution Basics.ipynb](./01_April/Week_02/02_Image%20Filtering%20%26%20Convolution%20Basics.ipynb) |        | Campus |
| Wed | Edge Detection          | [03_Edge Detection.ipynb](./01_April/Week_02/03_Edge%20Detection.ipynb)                                                         |        | Campus |
| Thu | Morphological Ops       | [04_Morphological Operations.ipynb](./01_April/Week_02/04_Morphological%20Operations.ipynb)                                     |        | Campus |
| Fri | Project                 | [05_Image Processing Fundamentals Project.ipynb](./01_April/Week_02/05_Image%20Processing%20Fundamentals%20Project.ipynb)       |        | Campus |

**Week 2 Project:** Modular image preprocessing class — parameterized, handles denoise → edge → color space conversion for any input image type.

</details>

<details>
<summary><b>📅 Week 3 — [Geometric Transforms & Contours](./01_April/Week_03)</b></summary>

| Day | Topic                     | Key Concepts                                       | Deliverable                               |
| --- | ------------------------- | -------------------------------------------------- | ----------------------------------------- |
| Mon | Geometric Transformations | Affine, perspective, homography, warpPerspective   | Document scanner mini-demo                |
| Tue | Contours & Shape Analysis | findContours, moments, bounding boxes, convex hull | Shape classifier                          |
| Wed | Hough Transforms          | Line transform (ρ,θ), circle transform           | Lane line + coin detector                 |
| Thu | Segmentation              | Adaptive thresholding, Watershed, GrabCut          | Touching object segmentation              |
| Fri | Project                   | Full pipeline                                      | **Project: Smart Document Scanner** |

**Daily Attendance Tracking:**

| Day | Topic                     | Remote | Campus |
| --- | ------------------------- | ------ | ------ |
| Mon | Geometric Transformations | - [ ]  | - [ ]  |
| Tue | Contours & Shape Analysis | - [ ]  | - [ ]  |
| Wed | Hough Transforms          | - [ ]  | - [ ]  |
| Thu | Segmentation              | - [ ]  | - [ ]  |
| Fri | Project                   | - [ ]  | - [ ]  |

**Week 3 Project:** Smart Document Scanner — phone photo → edge detection → perspective warp → flattened OCR-ready output.

</details>

<details>
<summary><b>📅 Week 4 — [Feature Detection & Description](./01_April/Week_04)</b></summary>

| Day | Topic             | Key Concepts                                          | Deliverable                           |
| --- | ----------------- | ----------------------------------------------------- | ------------------------------------- |
| Mon | Harris & FAST     | Corner response, scale space, Harris from scratch     | Keypoint visualizer                   |
| Tue | SIFT / SURF / ORB | Scale space, DoG, orientation, BRIEF descriptor       | Descriptor comparison                 |
| Wed | Feature Matching  | BF matcher, FLANN, Lowe's ratio test, RANSAC          | Image stitching prototype             |
| Thu | Optical Flow      | Lucas-Kanade, Farneback dense flow, motion estimation | Real-time feature tracker             |
| Fri | Capstone          | Full stitching pipeline                               | **Capstone: Panorama Stitcher** |

**Daily Attendance Tracking:**

| Day | Topic             | Remote | Campus |
| --- | ----------------- | ------ | ------ |
| Mon | Harris & FAST     | - [ ]  | - [ ]  |
| Tue | SIFT / SURF / ORB | - [ ]  | - [ ]  |
| Wed | Feature Matching  | - [ ]  | - [ ]  |
| Thu | Optical Flow      | - [ ]  | - [ ]  |
| Fri | Capstone          | - [ ]  | - [ ]  |

**Phase 1 Capstone:** Panorama Stitcher — SIFT → ratio test matching → RANSAC homography → multi-image warp + blend. CLI interface. GitHub repo.

</details>

---

## 🟢 Phase 2 — [Classical Machine Learning for Vision](./02_May)

> **Weeks 5–8 · May**
> Classical ML applied to images — HOG, SVM, BoVW, tracking, and full video analysis pipelines.

<details>
<summary><b>📅 Week 5 — [ML Fundamentals](./02_May/Week_01)</b></summary>

| Day | Topic                    | Key Concepts                                        | Deliverable                                        |
| --- | ------------------------ | --------------------------------------------------- | -------------------------------------------------- |
| Mon | ML Core + scikit-learn   | Train/val/test, bias-variance, KNN, SVM             | scikit-learn image classifier                      |
| Tue | HOG & LBP Features       | HOG cell/block/normalization, LBP texture patterns  | SVM on CIFAR-10 with HOG                           |
| Wed | PCA & Eigenfaces         | Covariance matrix, eigenvectors, explained variance | Eigenfaces face recognizer                         |
| Thu | Sliding Window Detection | Image pyramid, Viola-Jones, Haar cascades           | Pedestrian detector (HOG+SVM)                      |
| Fri | Project                  | Full face system                                    | **Project: Real-time Face Detection System** |

**Daily Attendance Tracking:**

| Day | Topic                    | File                                                                                                                                                         | Remote | Campus |
| --- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------ | ------ |
| Mon | ML Core + scikit-learn   | [01_ML Core + scikit-learn.ipynb](https://github.com/FahimShah651/CV_Practice/blob/main/02_May/Week_01/01_ML%20Core%20+%20scikit-learn.ipynb)                   | Remote |        |
| Tue | HOG & LBP Features       | [02_HOG &amp; LBP Image Features.ipynb](https://github.com/FahimShah651/CV_Practice/blob/main/02_May/Week_01/02_HOG%20%26%20LBP%20Image%20Features.ipynb)       | Remote |        |
| Wed | PCA & Eigenfaces         | [03_PCA &amp; Eigenfaces.ipynb](https://github.com/FahimShah651/CV_Practice/blob/main/02_May/Week_01/03_PCA%20%26%20Eigenfaces.ipynb)                           | Remote |        |
| Thu | Sliding Window Detection | [04_Sliding Window &amp; Viola-Jones.ipynb](https://github.com/FahimShah651/CV_Practice/blob/main/02_May/Week_01/04_Sliding%20Window%20%26%20Viola-Jones.ipynb) |        | Campus |
| Fri | Project                  | [05_Face Detection System.ipynb](https://github.com/FahimShah651/CV_Practice/blob/main/02_May/Week_01/05_Face%20Detection%20System.ipynb)                       |        | Campus |

</details>

<details>
<summary><b>📅 Week 6 — [Bag of Visual Words & Clustering](./02_May/Week_02)</b></summary>

| Day      | Topic                 | Key Concepts                                           | Deliverable                                |
| -------- | --------------------- | ------------------------------------------------------ | ------------------------------------------ |
| Mon–Tue | k-Means + BoVW        | k-Means, visual vocabulary, histogram encoding         | BoVW + SVM on Caltech-101                  |
| Wed      | Image Retrieval       | TF-IDF visual words, inverted index, cosine similarity | CBIR query system                          |
| Thu      | CV Evaluation Metrics | Precision, Recall, mAP, IoU, ROC/AUC, confusion matrix | Evaluation report                          |
| Fri      | Project               | Flask web app                                          | **Project: CBIR System with Web UI** |

**Daily Attendance Tracking:**

| Day | Topic                 | File                                                                                                                                                        | Remote | Campus |
| --- | --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | ------ |
| Mon | k-Means + BoVW        | [01_k-Means + Visual Vocabulary.ipynb](https://github.com/FahimShah651/CV_Practice/blob/main/02_May/Week_02/01_k-Means%20+%20Visual%20Vocabulary.ipynb) |        | Campus |
| Tue | BoVW from Scratch     | [02_BoVW from Scratch.ipynb](https://github.com/FahimShah651/CV_Practice/blob/main/02_May/Week_02/02_BoVW%20from%20Scratch.ipynb)                       | Remote |        |
| Wed | Image Retrieval       | [03_Image Retrieval System.ipynb](https://github.com/FahimShah651/CV_Practice/blob/main/02_May/Week_02/03_Image%20Retrieval%20System.ipynb)             | Remote |        |
| Thu | CV Evaluation Metrics | [04_CV Evaluation Metrics.ipynb](https://github.com/FahimShah651/CV_Practice/blob/main/02_May/Week_02/04_CV%20Evaluation%20Metrics.ipynb)               | Remote |        |
| Fri | Project               | [05_PROJECT_CBIR Flask Web App.py](https://github.com/FahimShah651/CV_Practice/blob/main/02_May/Week_02/05_PROJECT_CBIR%20Flask%20Web%20App.py)         | Remote |        |

</details>

<details>
<summary><b>📅 Week 7 — [Video Analysis & Motion](./02_May/Week_03)</b></summary>

| Day | Topic                  | Key Concepts                                       | Deliverable                                     |
| --- | ---------------------- | -------------------------------------------------- | ----------------------------------------------- |
| Mon | Background Subtraction | MOG2, KNN subtractor, frame differencing           | Moving object detector                          |
| Tue | Object Tracking        | Kalman filter, SORT, MeanShift, CamShift, CSRT/KCF | Multi-object tracker                            |
| Wed | Dense Optical Flow     | Farneback flow, activity from flow magnitude       | Action pattern recognizer                       |
| Thu | Camera Calibration     | Pinhole model, intrinsics/extrinsics, distortion   | Calibrated camera undistortion                  |
| Fri | Project                | Full system                                        | **Project: Motion-Based Security System** |

**Daily Attendance Tracking:**

| Day | Topic                  | Remote | Campus |
| --- | ---------------------- | ------ | ------ |
| Mon | Background Subtraction |        | Campus |
| Tue | Object Tracking        |        | Campus |
| Wed | Dense Optical Flow     |        | Campus |
| Thu | Camera Calibration     | [ ]    | [ ]    |
| Fri | Project                | [ ]    | [ ]    |

</details>

<details>
<summary><b>📅 Week 8 — [Phase 2 Integration](./02_May/Week_04)</b></summary>

| Days     | Topic                               | Deliverable                                                                |
| -------- | ----------------------------------- | -------------------------------------------------------------------------- |
| Mon–Wed | Vehicle Counting + Speed Estimation | Background subtraction + contour tracking + Kalman filter on highway video |
| Thu–Fri | Portfolio & GitHub Polish           | All Phase 1+2 projects documented with READMEs + demo GIFs                 |

**Daily Attendance Tracking:**

| Day | Topic            | Remote | Campus |
| --- | ---------------- | ------ | ------ |
| Mon | Vehicle Counting | [ ]    | [ ]    |
| Tue | Vehicle Counting | [ ]    | [ ]    |
| Wed | Vehicle Counting | [ ]    | [ ]    |
| Thu | Portfolio Polish | [ ]    | [ ]    |
| Fri | Portfolio Polish | [ ]    | [ ]    |

**Phase 2 Capstone:** Vehicle counter + speed estimator on highway dashcam footage.

</details>

---

## 🟣 Phase 3 — [Deep Learning &amp; CNNs for Vision](./03_June)

> **Weeks 9–14 · June**
> From neural network math to Vision Transformers. Build everything before using pretrained weights.

<details>
<summary><b>📅 Week 9 — [Neural Networks Foundations](./03_June/Week_01)</b></summary>

| Day | Topic                 | Key Concepts                                               | Deliverable                              |
| --- | --------------------- | ---------------------------------------------------------- | ---------------------------------------- |
| Mon | MLP Theory            | Perceptron, activations, forward pass, backprop derivation | MLP from scratch in NumPy                |
| Tue | PyTorch Fundamentals  | Tensors, autograd, nn.Module, DataLoader                   | MLP on MNIST                             |
| Wed | Optimizers & Training | SGD, Adam, LR schedules, BatchNorm, Dropout                | Optimizer comparison study               |
| Thu | CNN Theory            | Conv layer math, receptive field, feature maps, pooling    | CNN from scratch in PyTorch              |
| Fri | Project               | Custom dataset                                             | **Project: Custom CNN Classifier** |

**Daily Attendance Tracking:**

| Day | Topic                 | Remote | Campus |
| --- | --------------------- | ------ | ------ |
| Mon | MLP Theory            |     | Campus    |
| Tue | PyTorch Fundamentals  |    | Campus  |
| Wed | Optimizers & Training |    | Campus   |
| Thu | CNN Theory            |   | Campus  |
| Fri | Project               |    | Campus   |

</details>

<details>
<summary><b>📅 Week 10 — [Classic CNN Architectures](./03_June/Week_02)</b></summary>

| Day | Topic                      | Key Concepts                                           | Deliverable                                            |
| --- | -------------------------- | ------------------------------------------------------ | ------------------------------------------------------ |
| Mon | LeNet → AlexNet → VGGNet | Architecture evolution, depth vs width                 | VGGNet on CIFAR-100                                    |
| Tue | ResNet                     | Vanishing gradients, skip connections, ResNet-18       | ResNet-18 from scratch                                 |
| Wed | Inception & DenseNet       | Parallel convolutions, dense connections, EfficientNet | Architecture comparison                                |
| Thu | Transfer Learning          | Feature extraction, fine-tuning, layer freezing        | ResNet-50 → custom 5-class                            |
| Fri | Project                    | Medical domain                                         | **Project: Plant Disease Classifier + Grad-CAM** |

**Daily Attendance Tracking:**

| Day | Topic                      | Remote | Campus |
| --- | -------------------------- | ------ | ------ |
| Mon | LeNet → AlexNet → VGGNet | [ ]    | [ ]    |
| Tue | ResNet                     | [ ]    | [ ]    |
| Wed | Inception & DenseNet       | [ ]    | [ ]    |
| Thu | Transfer Learning          | [ ]    | [ ]    |
| Fri | Project                    | [ ]    | [ ]    |

</details>

<details>
<summary><b>📅 Week 11 — [Object Detection with Deep Learning](./03_June/Week_03)</b></summary>

| Day | Topic                  | Key Concepts                                              | Deliverable                                                  |
| --- | ---------------------- | --------------------------------------------------------- | ------------------------------------------------------------ |
| Mon | Detection Fundamentals | Anchor boxes, IoU, NMS, regression + classification heads | NMS from scratch                                             |
| Tue | R-CNN Family           | R-CNN → Fast → Faster R-CNN, RPN, ROI Pooling           | Faster R-CNN on COCO                                         |
| Wed | YOLO                   | YOLOv1→v8 evolution, grid prediction, multi-scale        | YOLOv8 fine-tuning                                           |
| Thu | Custom Training        | LabelImg, YOLO format, Albumentations augmentation        | 200-image custom detector                                    |
| Fri | Project                | Real-time system                                          | **Project: Real-time Object Detector (webcam + ONNX)** |

**Daily Attendance Tracking:**

| Day | Topic                  | Remote | Campus |
| --- | ---------------------- | ------ | ------ |
| Mon | Detection Fundamentals | [ ]    | [ ]    |
| Tue | R-CNN Family           | [ ]    | [ ]    |
| Wed | YOLO                   | [ ]    | [ ]    |
| Thu | Custom Training        | [ ]    | [ ]    |
| Fri | Project                | [ ]    | [ ]    |

</details>

<details>
<summary><b>📅 Week 12 — [Semantic & Instance Segmentation](./03_June/Week_04)</b></summary>

| Day | Topic          | Key Concepts                                                 | Deliverable                                                      |
| --- | -------------- | ------------------------------------------------------------ | ---------------------------------------------------------------- |
| Mon | FCN & U-Net    | Pixel-wise classification, encoder-decoder, skip connections | U-Net on Oxford Pets                                             |
| Tue | DeepLab        | Atrous convolution, ASPP, DeepLabv3+                         | DeepLabv3 on Cityscapes                                          |
| Wed | Mask R-CNN     | ROI Align, mask branch, instance vs semantic                 | Mask R-CNN via detectron2                                        |
| Thu | SAM + Panoptic | Segment Anything Model, zero-shot segmentation               | SAM vs fine-tuned U-Net                                          |
| Fri | Project        | Medical domain                                               | **Project: Medical Image Segmentation (U-Net + Dice/IoU)** |

**Daily Attendance Tracking:**

| Day | Topic          | Remote | Campus |
| --- | -------------- | ------ | ------ |
| Mon | FCN & U-Net    | [ ]    | [ ]    |
| Tue | DeepLab        | [ ]    | [ ]    |
| Wed | Mask R-CNN     | [ ]    | [ ]    |
| Thu | SAM + Panoptic | [ ]    | [ ]    |
| Fri | Project        | [ ]    | [ ]    |

</details>

<details>
<summary><b>📅 Weeks 13–14 — [Vision Transformers & Attention](./03_June)</b></summary>

| Days          | Topic                   | Key Concepts                                               | Deliverable                                                      |
| ------------- | ----------------------- | ---------------------------------------------------------- | ---------------------------------------------------------------- |
| Wk13 Mon–Wed | Attention & Transformer | Self-attention math, multi-head, positional encoding       | Transformer block from scratch                                   |
| Wk13 Thu–Fri | ViT & CLIP              | Patch embedding, CLS token, contrastive pretraining        | ViT fine-tune + CLIP zero-shot                                   |
| Wk14 Mon–Wed | Swin + DETR             | Hierarchical ViT, end-to-end detection, bipartite matching | DETR vs YOLOv8 comparison                                        |
| Wk14 Thu–Fri | Phase 3 Capstone        | Multi-task system                                          | **Capstone: Classify + Detect + Segment unified pipeline** |

**Daily Attendance Tracking:**

| Week | Day | Topic                   | Remote | Campus |
| ---- | --- | ----------------------- | ------ | ------ |
| 13   | Mon | Attention & Transformer | [ ]    | [ ]    |
| 13   | Tue | Attention & Transformer | [ ]    | [ ]    |
| 13   | Wed | Attention & Transformer | [ ]    | [ ]    |
| 13   | Thu | ViT & CLIP              | [ ]    | [ ]    |
| 13   | Fri | ViT & CLIP              | [ ]    | [ ]    |
| 14   | Mon | Swin + DETR             | [ ]    | [ ]    |
| 14   | Tue | Swin + DETR             | [ ]    | [ ]    |
| 14   | Wed | Swin + DETR             | [ ]    | [ ]    |
| 14   | Thu | Capstone                | [ ]    | [ ]    |
| 14   | Fri | Capstone                | [ ]    | [ ]    |

</details>

---

## 🟠 Phase 4 — [Advanced Deep Learning](./04_July)

> **Weeks 15–18 · July**
> GANs, 3D vision, self-supervised learning, and real production deployment.

<details>
<summary><b>📅 Week 15 — [Generative Models](./04_July/Week_01)</b></summary>

| Day | Topic              | Key Concepts                                         | Deliverable                                                 |
| --- | ------------------ | ---------------------------------------------------- | ----------------------------------------------------------- |
| Mon | GAN Theory         | Minimax loss, generator/discriminator, mode collapse | Vanilla GAN on MNIST                                        |
| Tue | DCGAN & StyleGAN   | DCGAN rules, AdaIN, style mapping                    | DCGAN on CelebA                                             |
| Wed | Pix2Pix & CycleGAN | Conditional GAN, PatchGAN, unpaired translation      | Horse↔Zebra CycleGAN                                       |
| Thu | VAEs & Diffusion   | ELBO loss, latent space, Stable Diffusion overview   | VAE + diffusers inference                                   |
| Fri | Project            | Custom dataset                                       | **Project: Image-to-Image Translation + Gradio Demo** |

**Daily Attendance Tracking:**

| Day | Topic              | Remote | Campus |
| --- | ------------------ | ------ | ------ |
| Mon | GAN Theory         | [ ]    | [ ]    |
| Tue | DCGAN & StyleGAN   | [ ]    | [ ]    |
| Wed | Pix2Pix & CycleGAN | [ ]    | [ ]    |
| Thu | VAEs & Diffusion   | [ ]    | [ ]    |
| Fri | Project            | [ ]    | [ ]    |

</details>

<details>
<summary><b>📅 Week 16 — [3D Computer Vision](./04_July/Week_02)</b></summary>

| Day | Topic           | Key Concepts                                  | Deliverable                                                  |
| --- | --------------- | --------------------------------------------- | ------------------------------------------------------------ |
| Mon | 3D Geometry     | Epipolar geometry, fundamental matrix, stereo | Disparity map from stereo pair                               |
| Tue | Monocular Depth | MiDaS, DPT, monodepth2                        | Colored depth map on video                                   |
| Wed | Point Clouds    | LiDAR, Open3D, PointNet, 3D bounding boxes    | KITTI point cloud visualization                              |
| Thu | NeRF & 3DGS     | Neural Radiance Fields, Gaussian Splatting    | Custom scene with nerfstudio                                 |
| Fri | Project         | Fusion                                        | **Project: Depth-Aware Background Blur (MiDaS + SAM)** |

**Daily Attendance Tracking:**

| Day | Topic           | Remote | Campus |
| --- | --------------- | ------ | ------ |
| Mon | 3D Geometry     | [ ]    | [ ]    |
| Tue | Monocular Depth | [ ]    | [ ]    |
| Wed | Point Clouds    | [ ]    | [ ]    |
| Thu | NeRF & 3DGS     | [ ]    | [ ]    |
| Fri | Project         | [ ]    | [ ]    |

</details>

<details>
<summary><b>📅 Week 17 — [Self-supervised & Contrastive Learning](./04_July/Week_03)</b></summary>

| Day      | Topic                   | Key Concepts                                          | Deliverable                                               |
| -------- | ----------------------- | ----------------------------------------------------- | --------------------------------------------------------- |
| Mon–Tue | SimCLR, MoCo, DINO      | Contrastive loss, momentum encoder, self-distillation | SimCLR on STL-10                                          |
| Wed–Thu | MAE & Foundation Models | Masked autoencoders, DINOv2, vision-language models   | DINOv2 few-shot classifier                                |
| Fri      | Project                 | Few-shot                                              | **Project: 5-Shot Classifier with DINOv2 Features** |

**Daily Attendance Tracking:**

| Day | Topic                   | Remote | Campus |
| --- | ----------------------- | ------ | ------ |
| Mon | SimCLR, MoCo, DINO      | [ ]    | [ ]    |
| Tue | SimCLR, MoCo, DINO      | [ ]    | [ ]    |
| Wed | MAE & Foundation Models | [ ]    | [ ]    |
| Thu | MAE & Foundation Models | [ ]    | [ ]    |
| Fri | Project                 | [ ]    | [ ]    |

</details>

<details>
<summary><b>📅 Week 18 — [Model Optimization & Deployment](./04_July/Week_04)</b></summary>

| Day | Topic                  | Key Concepts                                         | Deliverable                                            |
| --- | ---------------------- | ---------------------------------------------------- | ------------------------------------------------------ |
| Mon | Pruning & Quantization | INT8 quant, weight pruning, knowledge distillation   | INT8 YOLOv8 benchmark                                  |
| Tue | ONNX & TensorRT        | Model graph, operator compatibility, engine building | ResNet-50 ONNX export                                  |
| Wed | Edge Deployment        | TFLite, PyTorch Mobile, NCNN for ARM                 | Edge inference benchmarks                              |
| Thu | MLOps                  | W&B, DVC, CI/CD for models                           | Full experiment tracking setup                         |
| Fri | Capstone               | Full microservice                                    | **Capstone: Dockerized FastAPI CV Microservice** |

**Daily Attendance Tracking:**

| Day | Topic                  | Remote | Campus |
| --- | ---------------------- | ------ | ------ |
| Mon | Pruning & Quantization | [ ]    | [ ]    |
| Tue | ONNX & TensorRT        | [ ]    | [ ]    |
| Wed | Edge Deployment        | [ ]    | [ ]    |
| Thu | MLOps                  | [ ]    | [ ]    |
| Fri | Capstone               | [ ]    | [ ]    |

</details>

---

## 🔴 Phase 5 — [Specialization Tracks](./05_Agust)

> **Weeks 19–22 · August**
> Deep mastery in four applied domains — face recognition, pose estimation, OCR, and autonomous vehicles.

<details>
<summary><b>📅 Week 19 — [Face Recognition & Biometrics](./05_Agust/Week_01)</b></summary>

| Days     | Topic                   | Key Concepts                                          | Deliverable                                                      |
| -------- | ----------------------- | ----------------------------------------------------- | ---------------------------------------------------------------- |
| Mon–Tue | Advanced Face Detection | MTCNN, RetinaFace, MediaPipe Mesh, 68-point alignment | Robust face alignment pipeline                                   |
| Wed–Thu | ArcFace & FaceNet       | Triplet loss, ArcFace, CosFace, embedding space       | 1:N identification system                                        |
| Fri      | Project                 | Real-time system                                      | **Project: Face Recognition Attendance System + Liveness** |

**Daily Attendance Tracking:**

| Day | Topic                   | Remote | Campus |
| --- | ----------------------- | ------ | ------ |
| Mon | Advanced Face Detection | [ ]    | [ ]    |
| Tue | Advanced Face Detection | [ ]    | [ ]    |
| Wed | ArcFace & FaceNet       | [ ]    | [ ]    |
| Thu | ArcFace & FaceNet       | [ ]    | [ ]    |
| Fri | Project                 | [ ]    | [ ]    |

</details>

<details>
<summary><b>📅 Week 20 — [Pose Estimation & Body Tracking](./05_Agust/Week_02)</b></summary>

| Days     | Topic                 | Key Concepts                                            | Deliverable                                                     |
| -------- | --------------------- | ------------------------------------------------------- | --------------------------------------------------------------- |
| Mon–Tue | Human Pose Estimation | OpenPose, HRNet, MediaPipe Pose, PAF, heatmaps          | Joint angle extractor                                           |
| Wed–Fri | Action Recognition    | GCN for skeletons, LSTM temporal modeling, rep counting | **Project: Real-time Exercise Counter (push-up / squat)** |

**Daily Attendance Tracking:**

| Day | Topic                 | Remote | Campus |
| --- | --------------------- | ------ | ------ |
| Mon | Human Pose Estimation | [ ]    | [ ]    |
| Tue | Human Pose Estimation | [ ]    | [ ]    |
| Wed | Action Recognition    | [ ]    | [ ]    |
| Thu | Action Recognition    | [ ]    | [ ]    |
| Fri | Action Recognition    | [ ]    | [ ]    |

</details>

<details>
<summary><b>📅 Week 21 — [OCR & Document AI](./05_Agust/Week_03)</b></summary>

| Days     | Topic            | Key Concepts                                         | Deliverable                                                          |
| -------- | ---------------- | ---------------------------------------------------- | -------------------------------------------------------------------- |
| Mon–Tue | Text Detection   | EAST, CRAFT, DBNet, differentiable binarization      | CRAFT on ICDAR dataset                                               |
| Wed      | Text Recognition | CRNN (CNN+BiLSTM+CTC), Tesseract, EasyOCR, PaddleOCR | English + Urdu OCR comparison                                        |
| Thu–Fri | Project          | Full pipeline                                        | **Project: Document Digitization Pipeline → Structured JSON** |

**Daily Attendance Tracking:**

| Day | Topic            | Remote | Campus |
| --- | ---------------- | ------ | ------ |
| Mon | Text Detection   | [ ]    | [ ]    |
| Tue | Text Detection   | [ ]    | [ ]    |
| Wed | Text Recognition | [ ]    | [ ]    |
| Thu | Project          | [ ]    | [ ]    |
| Fri | Project          | [ ]    | [ ]    |

</details>

<details>
<summary><b>📅 Week 22 — [Autonomous Vehicle & Scene Understanding](./05_Agust/Week_04)</b></summary>

| Days     | Topic               | Key Concepts                                                | Deliverable                                                                    |
| -------- | ------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Mon–Tue | Lane Detection      | Hough + polynomial fit, LaneATT, UFLD, curvature            | Lane detection on dashcam                                                      |
| Wed–Fri | Multi-sensor Fusion | LiDAR+camera fusion, traffic sign recognition, scene graphs | **Project: Dashcam Analysis System (lanes + detection + signs + speed)** |

**Daily Attendance Tracking:**

| Day | Topic               | Remote | Campus |
| --- | ------------------- | ------ | ------ |
| Mon | Lane Detection      | [ ]    | [ ]    |
| Tue | Lane Detection      | [ ]    | [ ]    |
| Wed | Multi-sensor Fusion | [ ]    | [ ]    |
| Thu | Multi-sensor Fusion | [ ]    | [ ]    |
| Fri | Multi-sensor Fusion | [ ]    | [ ]    |

</details>

---

## 🟩 Phase 6 — [Capstone Projects &amp; Portfolio](./06_September)

> **Weeks 23–26 · September**
> Paper implementation, a major end-to-end project, full deployment, and job-ready portfolio.

<details>
<summary><b>📅 Week 23 — [Research Paper Implementation](./06_September/Week_01)</b></summary>

Choose one recent CV paper (2023/2024) from the list below and fully implement it:

| Paper                                     | Venue        | Complexity |
| ----------------------------------------- | ------------ | ---------- |
| RT-DETR (Real-Time Detection Transformer) | CVPR 2023    | ⭐⭐⭐     |
| YOLO-World (Open-Vocabulary Detection)    | CVPR 2024    | ⭐⭐⭐     |
| Grounded SAM (Detection + Segmentation)   | arXiv 2023   | ⭐⭐⭐⭐   |
| Depth Anything v2 (Monocular Depth)       | NeurIPS 2024 | ⭐⭐⭐     |
| DINOv2 (Self-supervised Foundation)       | TMLR 2024    | ⭐⭐⭐⭐   |

**Daily Attendance Tracking:**

- [ ] Monday — Remote / Campus
- [ ] Tuesday — Remote / Campus
- [ ] Wednesday — Remote / Campus
- [ ] Thursday — Remote / Campus
- [ ] Friday — Remote / Campus

**Deliverable:** Full implementation with ablation study, results table, and technical write-up.

</details>

<details>
<summary><b>📅 Weeks 24–25 — [Major Capstone Project](./06_September/Week_02)</b></summary>

Choose one domain application:

| Domain           | Project Idea                       | Core Tech Stack                     |
| ---------------- | ---------------------------------- | ----------------------------------- |
| 🏭 Manufacturing | Defect Detection on PCB/Fabric     | YOLOv8 + SAM + anomaly detection    |
| 🏥 Healthcare    | X-Ray Pathology Detector           | U-Net + classification + Grad-CAM   |
| 🌾 Agriculture   | Crop Disease + Yield Estimator     | EfficientNet + detection + tracking |
| 🏪 Retail        | Automated Checkout (no cashier)    | Detection + OCR + tracking          |
| 🏫 Campus        | Smart Attendance + Intrusion Alert | Face recognition + pose + tracking  |

**Week 24:** Design → data → baseline → optimize → ONNX export
**Week 25:** Full-stack deploy → Dockerize → HuggingFace Spaces / EC2 → demo video + portfolio

**Daily Attendance Tracking:**

| Week | Day | Focus       | Remote | Campus |
| ---- | --- | ----------- | ------ | ------ |
| 24   | Mon | Design      | [ ]    | [ ]    |
| 24   | Tue | Data        | [ ]    | [ ]    |
| 24   | Wed | Baseline    | [ ]    | [ ]    |
| 24   | Thu | Optimize    | [ ]    | [ ]    |
| 24   | Fri | ONNX Export | [ ]    | [ ]    |
| 25   | Mon | Deployment  | [ ]    | [ ]    |
| 25   | Tue | Dockerize   | [ ]    | [ ]    |
| 25   | Wed | Deployment  | [ ]    | [ ]    |
| 25   | Thu | Demo        | [ ]    | [ ]    |
| 25   | Fri | Portfolio   | [ ]    | [ ]    |

</details>

<details>
<summary><b>📅 Week 26 — [Interview Preparation & Launch](./06_September/Week_04)</b></summary>

| Day      | Focus                                                                         |
| -------- | ----------------------------------------------------------------------------- |
| Mon–Tue | Theory Q&A flashcards · Mock interviews · Project elevator pitches          |
| Wed      | LeetCode array/matrix problems · Implement CV algorithms under time pressure |
| Thu      | Kaggle CV competition submission · Open source PR (Ultralytics / timm)       |
| Fri      | Portfolio review · LinkedIn update · Job applications                       |

**Daily Attendance Tracking:**

| Day | Focus                | Remote | Campus |
| --- | -------------------- | ------ | ------ |
| Mon | Theory Q&A           | [ ]    | [ ]    |
| Tue | Mock Interviews      | [ ]    | [ ]    |
| Wed | LeetCode Practice    | [ ]    | [ ]    |
| Thu | Kaggle & Open Source | [ ]    | [ ]    |
| Fri | Portfolio Review     | [ ]    | [ ]    |

</details>

---

## 📝 Daily Log Convention

Each day folder should contain:

```
Day-XX_Topic-Name/
├── notes.md              ← Theory notes + key equations
├── implementation.py     ← Scratch implementation
├── experiments.ipynb     ← Experiments + visualizations
├── results/              ← Output images / plots
└── README.md             ← Summary: what I learned + what was hard
```

---

## 🏆 Projects Built (End of 6 Months)

| #  | Project                             | Phase   | Tech                                      |
| -- | ----------------------------------- | ------- | ----------------------------------------- |
| 1  | Image Stats Dashboard               | Phase 1 | NumPy, Matplotlib                         |
| 2  | Image Preprocessing Pipeline        | Phase 1 | OpenCV                                    |
| 3  | Smart Document Scanner              | Phase 1 | OpenCV, perspective warp                  |
| 4  | Panorama Stitcher                   | Phase 1 | SIFT, RANSAC, homography                  |
| 5  | Real-time Face Detector             | Phase 2 | HOG, SVM, Haar, dlib                      |
| 6  | CBIR System with Web UI             | Phase 2 | BoVW, Flask                               |
| 7  | Motion Security System              | Phase 2 | MOG2, Kalman, tracking                    |
| 8  | Vehicle Counter + Speed Estimator   | Phase 2 | Background subtraction, centroid tracking |
| 9  | Custom CNN Classifier               | Phase 3 | PyTorch                                   |
| 10 | Plant Disease Classifier + Grad-CAM | Phase 3 | EfficientNet, transfer learning           |
| 11 | Real-time Object Detector           | Phase 3 | YOLOv8, ONNX                              |
| 12 | Medical Image Segmentation          | Phase 3 | U-Net, Dice/IoU                           |
| 13 | Multi-task CV System                | Phase 3 | ViT + YOLO + U-Net                        |
| 14 | Image-to-Image Translation          | Phase 4 | Pix2Pix, Gradio                           |
| 15 | Depth-Aware Background Blur         | Phase 4 | MiDaS, SAM                                |
| 16 | Few-Shot Classifier                 | Phase 4 | DINOv2, kNN                               |
| 17 | Dockerized CV Microservice          | Phase 4 | FastAPI, Docker, ONNX                     |
| 18 | Face Recognition Attendance         | Phase 5 | ArcFace, liveness                         |
| 19 | Exercise Rep Counter                | Phase 5 | MediaPipe Pose, LSTM                      |
| 20 | Document Digitization Pipeline      | Phase 5 | CRAFT, EasyOCR, PaddleOCR                 |
| 21 | Dashcam Analysis System             | Phase 5 | Lane detection, YOLO, OCR                 |
| 22 | Research Paper Implementation       | Phase 6 | From paper                                |
| 23 | Major Capstone                      | Phase 6 | Full stack                                |

---

## 📖 Core Learning Resources

### Books

| Book                                                                                                     | Use                         |
| -------------------------------------------------------------------------------------------------------- | --------------------------- |
| [Computer Vision: Algorithms and Applications](http://szeliski.org/Book/) — Szeliski                       | Phase 1–2 theory reference |
| [Deep Learning](https://www.deeplearningbook.org/) — Goodfellow et al.                                     | Phase 3–4 math foundation  |
| [Understanding Deep Learning](https://udlbook.github.io/udlbook/) — Prince                                 | Modern DL with visuals      |
| [Hands-On ML with scikit-learn &amp; TensorFlow](https://github.com/ageron/handson-ml3) — Aurélien Géron | Phase 2–3 practical guide  |

### Courses

| Course                                                                                   | Platform | Phase      |
| ---------------------------------------------------------------------------------------- | -------- | ---------- |
| [CS231n: CNNs for Visual Recognition](http://cs231n.stanford.edu/)                          | Stanford | Phase 1–3 |
| [Practical Deep Learning for Coders](https://course.fast.ai/)                               | fast.ai  | Phase 3–4 |
| [DeepLearning.AI CV Specialization](https://www.coursera.org/specializations/deep-learning) | Coursera | Phase 3    |
| [Full Stack Deep Learning](https://fullstackdeeplearning.com/)                              | FSDL     | Phase 4–6 |

### Paper Reading — Recommended Order

```
Phase 3: AlexNet → VGGNet → ResNet → Inception → MobileNet
         → Faster R-CNN → YOLO → SSD → U-Net → DeepLab
         → Attention Is All You Need → ViT → CLIP → Swin → DETR

Phase 4: GAN → DCGAN → Pix2Pix → CycleGAN → StyleGAN
         → SimCLR → MoCo → DINO → MAE → DINOv2

Phase 5: ArcFace → FaceNet → OpenPose → HRNet → CRNN → LaneATT

Phase 6: RT-DETR → YOLO-World → Grounded SAM → Depth Anything
```

---

## 🔑 Key Concepts Quick Reference

<details>
<summary><b>Core Math for CV</b></summary>

- **Convolution:** `(f * g)(t) = ∫ f(τ)g(t-τ)dτ` — For images: slide kernel, multiply, sum
- **Gradient:** `∇I = [∂I/∂x, ∂I/∂y]` — points in direction of steepest intensity increase
- **Homography:** 3×3 matrix H mapping points between planes — 4 point correspondences to solve
- **IoU:** `IoU = Area(A ∩ B) / Area(A ∪ B)` — used for both detection evaluation and NMS
- **Softmax:** `σ(z)ᵢ = eᶻⁱ / Σⱼ eᶻʲ` — converts logits to probability distribution
- **Cross-entropy loss:** `L = -Σ y·log(ŷ)` — classification training objective
- **Dice loss:** `1 - (2|A∩B|)/(|A|+|B|)` — segmentation training objective

</details>

<details>
<summary><b>Architecture Design Rules (from papers)</b></summary>

- **VGGNet:** Only 3×3 convolutions. Depth over width.
- **ResNet:** Skip connections when depth > 20 layers. Bottleneck blocks for 50+ layers.
- **YOLO anchor design:** Cluster ground-truth boxes with k-Means on training data.
- **U-Net:** Skip connections at every encoder level. Input size divisible by 2^(depth).
- **ViT patch size:** 16×16 for ImageNet-scale, 4×4 for CIFAR-scale.
- **Batch norm:** Before activation in CNNs. Layer norm before attention in Transformers.

</details>

<details>
<summary><b>Debugging Checklist</b></summary>

- [ ] Loss not decreasing → check learning rate (try 1e-3 first), check data loading, verify labels
- [ ] Loss NaN → gradient explosion — add gradient clipping, reduce LR, check for log(0)
- [ ] Overfitting → add dropout, weight decay (1e-4), data augmentation, reduce model size
- [ ] Poor validation → shuffle train set, check data leakage, normalize inputs correctly
- [ ] CUDA out of memory → reduce batch size, use gradient checkpointing, fp16 training
- [ ] Slow training → check DataLoader num_workers (set to 4–8), pin_memory=True

</details>

---

## 🧪 How to Run Projects

Each project has its own README, but the general pattern is:

```bash
# Navigate to a project
cd 01_April/Week_04/Day_05

# Install specific requirements (if any)
pip install -r requirements.txt

# Run the main script
python main.py --input images/ --output output/

# Run the notebook for theory + experiments
jupyter notebook experiments.ipynb
```

For deep learning projects (Phase 3+):

```bash
# Train a model
python train.py --config configs/default.yaml --wandb

# Evaluate
python eval.py --checkpoint checkpoints/best.pth --data data/test

# Run inference
python inference.py --image path/to/image.jpg --visualize
```

---

## 📈 Progress Tracking

| Phase                                | Weeks  | Status         |
| ------------------------------------ | ------ | -------------- |
| [Phase 1 — Foundations](./01_April)    | 1–4   | 🔲 In Progress |
| [Phase 2 — Classical ML](./02_May)     | 5–8   | 🔲 In Progress |
| [Phase 3 — Deep Learning](./03_June)   | 9–14  | 🔲 Not Started |
| [Phase 4 — Advanced DL](./04_July)     | 15–18 | 🔲 Not Started |
| [Phase 5 — Specialization](./05_Agust) | 19–22 | 🔲 Not Started |
| [Phase 6 — Capstone](./06_September)   | 23–26 | 🔲 Not Started |

> Update this table as you progress. Change 🔲 to ✅ when a phase is complete.

---

## 💡 Tips for Getting the Most Out of This

1. **Implement before you import.** Write every algorithm from scratch at least once before using the library version. You'll debug faster forever after.
2. **Commit every single day.** Your GitHub contribution graph is a portfolio asset. Even a small experiment counts.
3. **Break things on purpose.** When learning a new model, deliberately break it — remove a skip connection, change the loss function — and observe what happens. This builds intuition faster than reading.
4. **Use Weights & Biases from Phase 3 onward.** Log everything: loss curves, sample predictions, hyperparameters. Future-you will thank present-you.
5. **Your EE background is an asset.** Your signal processing knowledge maps directly to convolution, Fourier transforms in images, and filter design. Lean into this — it separates you from pure CS graduates.
6. **Read papers, not just tutorials.** Pick 1 paper per week in Phase 3+. You don't need to understand every equation — understand the motivation, the proposed change, and the result.
7. **Kaggle is free GPU + feedback.** Use it. A public notebook with good methodology gets noticed.

---

## 🤝 Contributing / Feedback

If you're following along or have suggestions:

- **Open an issue** for errors or improvements
- **Star the repo** if this helped you
- **Fork and adapt** for your own learning journey

---

## 📄 License

This repository is for educational purposes. Code is MIT licensed. Notes and write-ups are CC BY 4.0.

---

<div align="center">

**Built with 💪 and 8 hours/day for 6 months**

*"The best way to learn computer vision is to make the computer see — one line of code at a time."*

[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github)](https://github.com/FahimShah651)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/fahim-ur-rehman-shah/)

</div>
