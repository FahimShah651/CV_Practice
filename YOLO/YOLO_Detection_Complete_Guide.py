"""
================================================================================
                    YOLOV8 OBJECT DETECTION - COMPLETE GUIDE
                        Learning & Full Implementation
================================================================================

This comprehensive guide covers all aspects of YOLOv8 object detection:
- Introduction to Object Detection
- Available Models and their Performance
- Training Models
- Validation and Metrics
- Inference/Prediction
- Model Export
- Practical Examples and Best Practices

Reference: https://docs.ultralytics.com/tasks/detect/
================================================================================
"""

# ============================================================================
# SECTION 1: INTRODUCTION TO OBJECT DETECTION
# ============================================================================

"""
WHAT IS OBJECT DETECTION?
==========================

Object Detection is a computer vision task that:
1. Identifies the LOCATION of objects in an image or video
2. Classifies WHAT those objects are
3. Provides CONFIDENCE SCORES for each detection

Key Components:
- Bounding Boxes: Rectangle coordinates (x, y, width, height) around objects
- Class Labels: What the object is (e.g., "person", "car", "dog")
- Confidence Scores: How confident the model is (0-1 probability)

WHY USE YOLOV8?
================
✓ State-of-the-art accuracy and speed
✓ Pretrained models on COCO dataset (80 classes)
✓ Easy-to-use Python API
✓ Export to multiple formats (ONNX, TensorRT, CoreML, etc.)
✓ Real-time inference capabilities
✓ Support for custom datasets and fine-tuning
✓ Lightweight models for edge deployment

USE CASES:
- Autonomous vehicles and traffic monitoring
- Security and surveillance systems
- Retail and inventory management
- Medical imaging analysis
- Wildlife monitoring
- Robot vision applications
"""

# ============================================================================
# SECTION 2: AVAILABLE YOLOV8 MODELS
# ============================================================================

"""
YOLOV8 PRETRAINED MODELS
=========================

The YOLOv8 family includes 5 model sizes, each with different trade-offs:

Model      | Size | mAP50-95 | mAP50 | Speed (ms) | Params (M) | FLOPs (B)
-----------|------|----------|-------|-----------|------------|----------
YOLOv8n    | 640  | 37.3     | 52.7  | 0.6       | 3.2        | 8.7
YOLOv8s    | 640  | 44.6     | 60.7  | 1.0       | 11.2       | 28.6
YOLOv8m    | 640  | 50.2     | 66.9  | 1.5       | 25.9       | 78.9
YOLOv8l    | 640  | 52.9     | 69.2  | 2.0       | 43.7       | 165.2
YOLOv8x    | 640  | 54.8     | 71.3  | 2.8       | 68.2       | 257.8

METRICS EXPLANATION:
- mAP50-95: Mean Average Precision (higher is better) - overall accuracy
- mAP50: Precision at IoU (Intersection over Union) = 0.50
- Speed: Inference time in milliseconds per image
- Params: Number of model parameters (millions)
- FLOPs: Floating Point Operations (billions)

HOW TO CHOOSE A MODEL:
- YOLOv8n: Fastest, smallest, edge devices, real-time on CPU
- YOLOv8s: Good balance of speed and accuracy
- YOLOv8m: Standard choice for most applications
- YOLOv8l: High accuracy, requires better hardware
- YOLOv8x: Best accuracy, requires GPU, slowest inference

COCO DATASET:
- 80 object classes (person, car, dog, cat, etc.)
- 330K images
- 1.5M object instances
- Publicly available benchmark dataset
"""

# ============================================================================
# SECTION 3: INSTALLATION AND SETUP
# ============================================================================

"""
INSTALLATION STEPS:
====================

1. Create a virtual environment (recommended):
   python -m venv yolo-env
   
2. Activate the environment:
   # Windows
   yolo-env\Scripts\activate
   
   # Mac/Linux
   source yolo-env/bin/activate

3. Install Ultralytics YOLO:
   pip install ultralytics

4. Install additional dependencies:
   pip install opencv-python
   pip install torch torchvision torchaudio
   pip install matplotlib
   pip install pillow

5. Verify installation:
   python -c "from ultralytics import YOLO; print('YOLO installed successfully')"
"""

# ============================================================================
# SECTION 4: BASIC USAGE - LOADING MODELS
# ============================================================================

from ultralytics import YOLO
from pathlib import Path

# Method 1: Load a pretrained model
# Automatically downloads if not already present
model = YOLO("yolov8n.pt")  # nano model - fastest
model = YOLO("yolov8s.pt")  # small model
model = YOLO("yolov8m.pt")  # medium model (recommended)
model = YOLO("yolov8l.pt")  # large model
model = YOLO("yolov8x.pt")  # extra large model

# Method 2: Build a new model from YAML configuration
# Useful for custom architectures or starting fresh
model = YOLO("yolov8n.yaml")  # build new model from scratch

# Method 3: Load and transfer weights
# Build from YAML and load pretrained weights
model = YOLO("yolov8n.yaml").load("yolov8n.pt")

# Method 4: Load a custom trained model
# After training, load your best weights
# Only load if the file exists (after you've trained a model)
from pathlib import Path
custom_model_path = "runs/detect/train/weights/best.pt"
if Path(custom_model_path).exists():
    custom_model = YOLO(custom_model_path)
else:
    print(f"Note: {custom_model_path} does not exist yet.")
    print("Train a model first using: model.train(data='dataset.yaml', epochs=100)")
    # Don't try to load non-existent model


# ============================================================================
# SECTION 5: TRAINING - DETAILED GUIDE
# ============================================================================

def train_yolo_model():
    """
    TRAINING YOUR YOLOV8 MODEL
    ==========================
    
    Training involves fitting the model to your dataset.
    The model learns to detect objects specific to your domain.
    """
    
    # Load a pretrained model (recommended for transfer learning)
    model = YOLO("yolov8n.pt")
    
    # TRAINING PARAMETERS:
    results = model.train(
        data="coco8.yaml",              # Dataset configuration file
        epochs=100,                      # Number of training cycles
        imgsz=640,                       # Training image size
        batch=16,                        # Batch size per GPU
        patience=20,                     # Early stopping patience
        save=True,                       # Save model checkpoints
        device=0,                        # GPU device ID (0 for first GPU, CPU if not available)
        
        # Optional parameters for fine-tuning:
        lr0=0.01,                       # Initial learning rate
        momentum=0.937,                 # Optimizer momentum
        weight_decay=0.0005,            # Weight decay (L2 regularization)
        warmup_epochs=3,                # Warmup epochs
        warmup_momentum=0.8,            # Warmup momentum
        
        # Data augmentation:
        hsv_h=0.015,                   # Image HSV-Hue augmentation
        hsv_s=0.7,                     # Image HSV-Saturation augmentation
        hsv_v=0.4,                     # Image HSV-Value augmentation
        degrees=10.0,                  # Rotation augmentation
        translate=0.1,                 # Translation augmentation
        scale=0.5,                     # Scaling augmentation
        flipud=0.0,                    # Flip up-down probability
        fliplr=0.5,                    # Flip left-right probability
        
        # Other useful options:
        name="my_detection_model",     # Experiment name
        save_period=10,                # Save checkpoint every N epochs
        workers=8,                     # Dataloader workers
        project="runs/detect",         # Project directory
        exist_ok=False,                # Overwrite existing project
        pretrained=True,               # Start from pretrained weights
        optimizer="SGD",               # Optimizer (SGD, Adam, AdamW, RMSProp, etc.)
        seed=0,                        # Random seed for reproducibility
    )
    
    print("Training completed!")
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")
    
    return results


# ============================================================================
# SECTION 6: DATASET FORMAT & PREPARATION
# ============================================================================

"""
YOLOV8 DATASET FORMAT
======================

Your custom dataset must follow this structure:

dataset/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── val/
│   │   ├── img1.jpg
│   │   └── ...
│   └── test/
│       └── ...
└── labels/
    ├── train/
    │   ├── img1.txt
    │   ├── img2.txt
    │   └── ...
    ├── val/
    │   ├── img1.txt
    │   └── ...
    └── test/
        └── ...

LABEL FILE FORMAT (.txt files):
==============================
One row per object in the image:
<class_id> <x_center> <y_center> <width> <height>

Example (img1.txt):
    0 0.5 0.5 0.3 0.4    # Class 0 at center, 30% width, 40% height
    1 0.2 0.3 0.1 0.2    # Class 1 at normalized coordinates

All coordinates are NORMALIZED (0-1) relative to image dimensions.

DATASET CONFIGURATION FILE (dataset.yaml):
==========================================
path: /path/to/dataset
train: images/train
val: images/val
test: images/test

nc: 2  # Number of classes
names: ['person', 'car']  # Class names

CONVERTING FROM OTHER FORMATS:
==============================
Use Ultralytics JSON2YOLO tool:
https://github.com/ultralytics/JSON2YOLO

Or use CVAT, Roboflow, or other annotation tools that export to YOLO format.
"""


# ============================================================================
# SECTION 7: VALIDATION - EVALUATING MODEL PERFORMANCE
# ============================================================================

def validate_model():
    """
    VALIDATING YOUR MODEL
    =====================
    
    Validation measures how well your model performs on unseen data.
    Returns metrics like mAP (mean Average Precision), precision, recall, etc.
    """
    
    # Load a trained model
    model = YOLO("yolov8n.pt")
    
    # Validate the model
    metrics = model.val(
        data="coco8.yaml",              # Dataset configuration
        imgsz=640,                      # Validation image size
        batch=32,                       # Batch size
        conf=0.25,                      # Confidence threshold (0-1)
        iou=0.6,                        # IoU threshold for NMS
        device=0,                       # GPU device ID
        workers=8,                      # Dataloader workers
        save=True,                      # Save validation results
        project="runs/detect",
        name="val_results",
    )
    
    # ACCESS VALIDATION METRICS:
    print("=== VALIDATION METRICS ===")
    print(f"mAP50-95: {metrics.box.map}")       # Mean Average Precision (0-100%)
    print(f"mAP50: {metrics.box.map50}")         # mAP at IoU=0.50
    print(f"mAP75: {metrics.box.map75}")         # mAP at IoU=0.75
    print(f"Precision: {metrics.box.mp}")        # Precision (0-1)
    print(f"Recall: {metrics.box.mr}")           # Recall (0-1)
    
    # Per-class metrics
    if metrics.box.maps:
        print(f"\nPer-class mAP: {metrics.box.maps}")
    
    # Per-image metrics
    if metrics.box.image_metrics:
        image_metrics = metrics.box.image_metrics
        print(f"\nPer-image metrics available")
        print(f"Keys: {image_metrics.keys()}")
    
    return metrics


"""
UNDERSTANDING VALIDATION METRICS:
==================================

mAP (mean Average Precision):
- Measure of detection accuracy
- Calculated at different IoU thresholds
- mAP50: Average precision at IoU=0.50 (more lenient)
- mAP75: Average precision at IoU=0.75 (stricter)
- mAP50-95: Average precision across IoU thresholds (0.50 to 0.95)
- Range: 0-100% (higher is better)

Precision:
- Of all detected objects, how many were correct?
- Formula: TP / (TP + FP)
- Range: 0-1 (higher is better)

Recall:
- Of all actual objects, how many did we detect?
- Formula: TP / (TP + FN)
- Range: 0-1 (higher is better)

TP (True Positive): Correctly detected object
FP (False Positive): Detected but not there
FN (False Negative): Missed detection
IoU (Intersection over Union): Overlap between predicted and ground truth boxes
"""


# ============================================================================
# SECTION 8: PREDICTION / INFERENCE
# ============================================================================

def predict_on_image():
    """
    RUNNING PREDICTIONS ON IMAGES
    ==============================
    
    Inference is the process of running your trained model on new images.
    """
    
    # Load a model
    model = YOLO("yolov8n.pt")
    
    # Predict on a single image
    results = model.predict(
        source="https://ultralytics.com/images/bus.jpg",  # Image path or URL
        conf=0.25,                      # Confidence threshold
        iou=0.45,                       # IoU threshold for NMS
        imgsz=640,                      # Inference image size
        device=0,                       # GPU device (0 for GPU, 'cpu' for CPU)
        save=True,                      # Save predictions
        save_txt=True,                  # Save predictions as text files
        save_conf=True,                 # Save confidence scores
        line_width=2,                   # Bounding box line width
        project="runs/detect",
        name="predict_results",
    )
    
    # ACCESSING PREDICTION RESULTS:
    for result in results:
        # Bounding boxes
        boxes = result.boxes
        
        # Different box formats:
        print("Bounding box formats:")
        print(f"xyxy (top-left, bottom-right): {boxes.xyxy}")     # [[x1, y1, x2, y2], ...]
        print(f"xyxyn (normalized): {boxes.xyxyn}")               # Normalized to 0-1
        print(f"xywh (center, width, height): {boxes.xywh}")      # [[x_c, y_c, w, h], ...]
        print(f"xywhn (normalized): {boxes.xywhn}")               # Normalized
        
        # Class predictions and confidence
        class_ids = boxes.cls                                     # Class IDs
        confidences = boxes.conf                                  # Confidence scores
        
        # Class names
        class_names = [result.names[int(cls_id)] for cls_id in class_ids]
        
        # Iterate through detections
        print("\nDetections:")
        for i, (box, cls_id, conf, name) in enumerate(
            zip(boxes.xyxy, class_ids, confidences, class_names)
        ):
            x1, y1, x2, y2 = box
            print(f"Detection {i}: {name} ({conf:.2%}) at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
        
        # Save annotated image
        annotated_image = result.plot()  # Returns annotated image as numpy array
        
    return results


def predict_on_video():
    """
    RUNNING PREDICTIONS ON VIDEO
    =============================
    
    Process video frames sequentially for real-time detection.
    """
    
    model = YOLO("yolov8n.pt")
    
    # Predict on video
    results = model.predict(
        source="video.mp4",             # Video path or webcam (0)
        conf=0.25,
        iou=0.45,
        imgsz=640,
        device=0,
        save=True,                      # Save output video
        project="runs/detect",
        name="video_results",
        line_width=2,
        classes=None,                   # Detect all classes or specify specific ones
    )
    
    # Process frame-by-frame
    for result in results:
        # Get detections for this frame
        detections = result.boxes
        frame_annotations = result.plot()
        # Use frame_annotations for further processing or display
    
    return results


def predict_on_webcam():
    """
    REAL-TIME DETECTION FROM WEBCAM
    ================================
    """
    
    model = YOLO("yolov8n.pt")
    
    # 0 = default webcam
    results = model.predict(
        source=0,                       # Webcam
        conf=0.25,
        iou=0.45,
        imgsz=640,
        device=0,
        line_width=2,
        classes=None,
        verbose=False,
    )
    
    return results


def batch_predict():
    """
    BATCH PREDICTION ON MULTIPLE IMAGES
    ====================================
    
    Efficiently process multiple images.
    """
    
    model = YOLO("yolov8n.pt")
    
    # Predict on multiple images
    results = model.predict(
        source="path/to/image/folder",  # Folder with images
        conf=0.25,
        iou=0.45,
        imgsz=640,
        device=0,
        save=True,
        batch=32,                       # Batch size for efficiency
        project="runs/detect",
        name="batch_results",
    )
    
    # Process results
    for result in results:
        print(f"Image: {result.path}")
        print(f"Detections: {len(result.boxes)}")
        print(f"Classes: {result.boxes.cls.tolist()}")
    
    return results


"""
PREDICTION OUTPUT STRUCTURE:
=============================

result.boxes contains:
- xyxy: Bounding boxes in [x1, y1, x2, y2] format (pixel coordinates)
- xyxyn: Normalized bounding boxes (0-1)
- xywh: [x_center, y_center, width, height] format
- xywhn: Normalized xywh format
- conf: Confidence scores for each detection
- cls: Class IDs for each detection

result.names:
- Dictionary mapping class IDs to class names
- Example: {0: 'person', 1: 'bicycle', 2: 'car', ...}

result.plot():
- Returns annotated image as numpy array
- Includes bounding boxes and labels
"""


# ============================================================================
# SECTION 9: MODEL EXPORT
# ============================================================================

def export_model():
    """
    EXPORTING MODELS TO DIFFERENT FORMATS
    ======================================
    
    Export your trained model to various formats for deployment
    on different platforms and devices.
    """
    
    # Load a model
    model = YOLO("yolov8n.pt")
    
    # Export to different formats:
    
    # PyTorch (default)
    export_path = model.export(format="pt")
    print(f"PyTorch: {export_path}")
    
    # TorchScript (optimized for inference)
    export_path = model.export(
        format="torchscript",
        imgsz=640,
        half=False,
        dynamic=False,
        optimize=False,
        nms=False,
        batch=1,
        device=0,
    )
    print(f"TorchScript: {export_path}")
    
    # ONNX (Open Neural Network Exchange - cross-platform)
    export_path = model.export(
        format="onnx",
        imgsz=640,
        half=False,                    # FP32
        dynamic=True,                  # Variable input dimensions
        simplify=True,                 # Simplify model graph
        opset=13,                      # ONNX opset version
        nms=False,
        batch=1,
        device=0,
    )
    print(f"ONNX: {export_path}")
    
    # TensorRT (NVIDIA GPU inference - fastest)
    export_path = model.export(
        format="engine",
        imgsz=640,
        half=True,                     # FP16 precision (faster)
        dynamic=False,
        simplify=False,
        workspace=4,                   # Workspace size in GB
        int8=False,                    # Quantization option
        nms=False,
        batch=1,
        device=0,
    )
    print(f"TensorRT: {export_path}")
    
    # CoreML (Apple devices)
    export_path = model.export(
        format="coreml",
        imgsz=640,
        half=False,
        int8=False,
        nms=False,
        batch=1,
        device=0,
    )
    print(f"CoreML: {export_path}")
    
    # TensorFlow SavedModel
    export_path = model.export(
        format="saved_model",
        imgsz=640,
        keras=False,
        int8=False,
        nms=False,
        batch=1,
        device=0,
    )
    print(f"TensorFlow SavedModel: {export_path}")
    
    # TensorFlow Lite (mobile devices)
    export_path = model.export(
        format="tflite",
        imgsz=640,
        half=False,
        int8=False,
        nms=False,
        batch=1,
        device=0,
    )
    print(f"TensorFlow Lite: {export_path}")
    
    # OpenVINO (Intel optimization)
    export_path = model.export(
        format="openvino",
        imgsz=640,
        half=False,
        int8=False,
        nms=False,
        batch=1,
        device=0,
    )
    print(f"OpenVINO: {export_path}")


"""
EXPORT FORMAT COMPARISON:
==========================

Format          | Platform          | Speed | Accuracy | Use Case
----------------|-------------------|-------|----------|------------------
PyTorch (.pt)   | Python/Colab      | Fast  | 100%     | Training/Development
ONNX            | Cross-platform    | Fast  | 100%     | Interoperability
TensorRT        | NVIDIA GPU        | ⚡⚡⚡ | 100%     | Production GPU inference
TensorFlow      | Google/Cloud      | Fast  | 100%     | TensorFlow ecosystem
TFLite          | Mobile/Edge       | ⚡    | ~95%     | Mobile devices
CoreML          | Apple devices     | ⚡    | ~95%     | iOS/macOS
OpenVINO        | Intel CPU         | ⚡    | 100%     | CPU inference
NCNN            | Android           | ⚡    | ~95%     | Android/embedded
"""


# ============================================================================
# SECTION 10: ADVANCED USAGE - FILTERING & CUSTOMIZATION
# ============================================================================

def advanced_prediction_with_filters():
    """
    ADVANCED PREDICTION WITH CLASS FILTERING
    =========================================
    
    Filter predictions to detect only specific classes of interest.
    """
    
    model = YOLO("yolov8n.pt")
    
    # Get model class names
    class_names = model.names  # {0: 'person', 1: 'bicycle', ...}
    
    # Find class IDs for specific classes
    target_classes = ['person', 'car', 'truck']
    target_class_ids = [
        class_id for class_id, name in class_names.items()
        if name in target_classes
    ]
    
    # Predict with class filtering
    results = model.predict(
        source="image.jpg",
        conf=0.25,
        iou=0.45,
        classes=target_class_ids,      # Only detect these classes
        imgsz=640,
        device=0,
    )
    
    for result in results:
        for box, class_id, confidence in zip(
            result.boxes.xyxy,
            result.boxes.cls,
            result.boxes.conf
        ):
            class_name = class_names[int(class_id)]
            print(f"{class_name}: {confidence:.2%}")
    
    return results


def track_objects_in_video():
    """
    OBJECT TRACKING IN VIDEO
    ========================
    
    Track the same object across video frames using unique IDs.
    """
    
    model = YOLO("yolov8n.pt")
    
    # Enable tracking (requires tracker configuration)
    results = model.track(
        source="video.mp4",             # Video source
        conf=0.25,
        iou=0.5,
        imgsz=640,
        device=0,
        tracker="botsort.yaml",         # Tracker type
        persist=True,                   # Persist tracks
    )
    
    # Access tracking results
    for result in results:
        for box in result.boxes:
            # Get track ID
            track_id = box.id
            class_id = box.cls
            confidence = box.conf
            
            print(f"Track ID: {track_id}, Class: {class_id}, Conf: {confidence}")
    
    return results


# ============================================================================
# SECTION 11: PRACTICAL COMPLETE EXAMPLE
# ============================================================================

def complete_workflow_example():
    """
    COMPLETE WORKFLOW EXAMPLE
    ==========================
    
    A practical example showing the entire pipeline:
    Load → Predict → Process → Visualize
    """
    
    from PIL import Image
    import cv2
    import numpy as np
    
    # 1. Load model
    print("Loading model...")
    model = YOLO("yolov8n.pt")
    
    # 2. Run inference
    print("Running inference...")
    results = model.predict(
        source="https://ultralytics.com/images/bus.jpg",
        conf=0.25,
        imgsz=640,
        device=0,
        save=False,
    )
    
    # 3. Process results
    print("\nProcessing results...")
    for result in results:
        # Get image
        image_array = result.orig_img
        
        # Get detections
        boxes = result.boxes
        class_names = result.names
        
        # Filter high-confidence detections
        high_conf_indices = boxes.conf > 0.5
        
        print(f"Total detections: {len(boxes)}")
        print(f"High-confidence detections (>50%): {high_conf_indices.sum()}")
        
        # Print each detection
        for i, (box, cls_id, conf) in enumerate(
            zip(boxes.xyxy, boxes.cls, boxes.conf)
        ):
            if conf > 0.5:
                x1, y1, x2, y2 = map(int, box)
                class_name = class_names[int(cls_id)]
                print(f"  {i}: {class_name} ({conf:.1%}) at [{x1},{y1},{x2},{y2}]")
        
        # 4. Save annotated image
        annotated_image = result.plot()
        image_path = "detection_result.jpg"
        
        # Save using PIL
        Image.fromarray(annotated_image).save(image_path)
        print(f"\nAnnotated image saved: {image_path}")
        
        return result


# ============================================================================
# SECTION 12: PERFORMANCE OPTIMIZATION TIPS
# ============================================================================

"""
PERFORMANCE OPTIMIZATION TIPS
==============================

1. MODEL SELECTION:
   - Use YOLOv8n for real-time CPU inference
   - Use YOLOv8m for balanced speed/accuracy
   - Use YOLOv8x for maximum accuracy (GPU required)

2. INPUT SIZE:
   - Smaller images (320) = faster but less accurate
   - Larger images (1280) = slower but more accurate
   - Trade-off depends on use case

3. BATCH PROCESSING:
   - Process multiple images at once for efficiency
   - Batch size depends on GPU memory
   - Larger batches = better GPU utilization

4. GPU ACCELERATION:
   - Always use GPU (device=0) when available
   - CPU is 10-100x slower than GPU
   - Check GPU availability: torch.cuda.is_available()

5. QUANTIZATION:
   - Export with half=True for FP16 (faster, ~95% accuracy)
   - Export with int8=True for INT8 (fastest, ~90% accuracy)
   - Use for deployment on edge devices

6. CONFIDENCE THRESHOLD:
   - Lower conf threshold = more detections (higher FP)
   - Higher conf threshold = fewer but confident detections
   - Typical: 0.25-0.5

7. NMS (Non-Maximum Suppression):
   - Removes overlapping boxes
   - Higher iou threshold = more overlapping boxes allowed
   - Typical: 0.45-0.6

8. MULTI-THREADING:
   - Set workers > 1 for faster data loading
   - Example: workers=8 for parallel loading

9. HALF PRECISION:
   - Use device float16 for faster inference
   - Almost no accuracy loss
   - Requires GPU support

10. MODEL FUSION:
    - model.fuse() reduces model size and speeds up inference
    - Run after training or loading
"""


# ============================================================================
# SECTION 13: TROUBLESHOOTING & BEST PRACTICES
# ============================================================================

"""
COMMON ISSUES & SOLUTIONS
==========================

1. Out of Memory (OOM) Error:
   - Reduce batch size
   - Reduce image size (imgsz)
   - Use smaller model (yolov8n instead of yolov8x)
   - Use gradient accumulation for training

2. Low Accuracy:
   - Increase training epochs
   - Use larger model (yolov8m, yolov8l)
   - Improve dataset quality and quantity
   - Increase image size (imgsz)
   - Use data augmentation

3. Slow Inference:
   - Use GPU instead of CPU
   - Use smaller model (yolov8n)
   - Reduce image size
   - Use quantization (FP16, INT8)
   - Export to optimized format (TensorRT)

4. Training Divergence:
   - Reduce learning rate (lr0)
   - Increase warmup_epochs
   - Reduce weight_decay
   - Check dataset format

5. Class Imbalance:
   - Use class weights
   - Increase training epochs
   - Apply data augmentation
   - Collect more minority class samples

BEST PRACTICES
==============

✓ ALWAYS use transfer learning (start from pretrained weights)
✓ Split dataset into train/val/test (70/15/15 typical)
✓ Validate on separate test set not seen during training
✓ Monitor training metrics (loss, mAP) to catch issues early
✓ Use early stopping (patience parameter) to prevent overfitting
✓ Document hyperparameters and results for reproducibility
✓ Use appropriate image size for your objects
✓ Collect diverse, representative data
✓ Clean and verify dataset labels
✓ Regular backups of trained models
"""


# ============================================================================
# SECTION 14: QUICK REFERENCE CODE SNIPPETS
# ============================================================================

# Quick Load & Predict
quick_predict = """
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model.predict('image.jpg')
for result in results:
    print(result.boxes)
"""

# Quick Train
quick_train = """
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='dataset.yaml', epochs=100, imgsz=640)
"""

# Quick Validate
quick_validate = """
from ultralytics import YOLO
model = YOLO('runs/detect/train/weights/best.pt')
metrics = model.val()
print(f"mAP: {metrics.box.map}")
"""

# Quick Export
quick_export = """
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='onnx')
model.export(format='tflite')
"""


# ============================================================================
# SECTION 15: COMPLETE PRODUCTION TEMPLATE
# ============================================================================

class YOLODetectionPipeline:
    """
    PRODUCTION-READY YOLO DETECTION PIPELINE
    ==========================================
    
    A complete, reusable class for object detection in production.
    """
    
    def __init__(self, model_path="yolov8m.pt", device=0, conf_threshold=0.25):
        """Initialize the detection pipeline."""
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.device = device
        self.conf_threshold = conf_threshold
        self.class_names = self.model.names
        
    def detect(self, source):
        """
        Run detection on image/video/webcam.
        
        Args:
            source: Image path, video path, or webcam index
            
        Returns:
            List of detection results
        """
        results = self.model.predict(
            source=source,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )
        return results
    
    def get_detections_dict(self, result):
        """
        Convert detection results to dictionary format.
        
        Args:
            result: Single prediction result
            
        Returns:
            Dictionary with detection information
        """
        detections = []
        for box, cls_id, conf in zip(
            result.boxes.xyxy,
            result.boxes.cls,
            result.boxes.conf
        ):
            detection = {
                'class_id': int(cls_id),
                'class_name': self.class_names[int(cls_id)],
                'confidence': float(conf),
                'bbox': {
                    'x1': float(box[0]),
                    'y1': float(box[1]),
                    'x2': float(box[2]),
                    'y2': float(box[3]),
                    'width': float(box[2] - box[0]),
                    'height': float(box[3] - box[1]),
                },
            }
            detections.append(detection)
        return detections
    
    def filter_detections(self, detections, class_names=None, min_conf=None):
        """
        Filter detections by class or confidence.
        
        Args:
            detections: List of detection dictionaries
            class_names: List of class names to keep (None = all)
            min_conf: Minimum confidence threshold (None = use default)
            
        Returns:
            Filtered detections list
        """
        filtered = []
        min_confidence = min_conf or self.conf_threshold
        
        for det in detections:
            # Check confidence
            if det['confidence'] < min_confidence:
                continue
            
            # Check class name
            if class_names and det['class_name'] not in class_names:
                continue
            
            filtered.append(det)
        
        return filtered
    
    def process_image(self, image_path, save_result=True):
        """
        Complete image processing pipeline.
        
        Args:
            image_path: Path to image
            save_result: Whether to save annotated image
            
        Returns:
            Dictionary with image path and detections
        """
        # Run detection
        results = self.detect(image_path)
        
        # Extract detections
        detections_list = []
        for result in results:
            detections = self.get_detections_dict(result)
            detections_list.append({
                'image_path': image_path,
                'detections': detections,
                'num_detections': len(detections),
            })
            
            # Save annotated image
            if save_result:
                annotated_img = result.plot()
                save_path = image_path.replace('.jpg', '_annotated.jpg')
                from PIL import Image
                Image.fromarray(annotated_img).save(save_path)
        
        return detections_list


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    print("=" * 80)
    print("YOLOV8 OBJECT DETECTION - COMPLETE GUIDE")
    print("=" * 80)
    
    # Example 1: Simple prediction
    print("\n1. Simple Prediction Example:")
    print("-" * 80)
    try:
        model = YOLO("yolov8n.pt")
        print(f"Model loaded successfully")
        print(f"Model classes: {model.names}")
    except Exception as e:
        print(f"Note: {e}")
    
    # Example 2: Using the production pipeline
    print("\n2. Production Pipeline Example:")
    print("-" * 80)
    try:
        pipeline = YOLODetectionPipeline(model_path="yolov8n.pt")
        print(f"Pipeline initialized with {len(pipeline.class_names)} classes")
        print(f"Available classes: {list(pipeline.class_names.values())[:5]}...")
    except Exception as e:
        print(f"Note: {e}")
    
    print("\n" + "=" * 80)
    print("For full training/validation/export examples, uncomment the functions above")
    print("=" * 80)

"""
================================================================================
SUMMARY & NEXT STEPS
================================================================================

WHAT YOU'VE LEARNED:

1. ✓ Object Detection Basics
2. ✓ Available YOLOv8 Models and Their Performance
3. ✓ Installation and Setup
4. ✓ Loading Pretrained Models
5. ✓ Training on Custom Datasets
6. ✓ Validation and Metrics
7. ✓ Running Inference on Images/Videos
8. ✓ Exporting to Different Formats
9. ✓ Advanced Filtering and Tracking
10. ✓ Production-Ready Implementation

NEXT STEPS:

1. Start with a pretrained model for quick experimentation
2. Prepare your custom dataset in YOLO format
3. Fine-tune on your specific use case
4. Validate and iterate on hyperparameters
5. Export to your target platform (ONNX, TensorRT, etc.)
6. Deploy using the production pipeline template

RESOURCES:

- Official Docs: https://docs.ultralytics.com/tasks/detect/
- GitHub: https://github.com/ultralytics/ultralytics
- Blog: https://www.ultralytics.com/blog
- Discord Community: https://discord.com/invite/ultralytics

HELPFUL COMMANDS:

# Training from command line
yolo detect train data=coco128.yaml model=yolov8n.yaml epochs=100

# Validation
yolo detect val model=yolov8n.pt data=coco128.yaml

# Prediction
yolo detect predict model=yolov8n.pt source=image.jpg

# Export
yolo detect export model=yolov8n.pt format=onnx

================================================================================
"""
