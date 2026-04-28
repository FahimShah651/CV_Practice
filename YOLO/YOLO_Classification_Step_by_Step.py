"""
================================================================================
        YOLOV8 CLASSIFICATION - STEP BY STEP LINE-BY-LINE EXPLANATION
================================================================================

WHAT IS CLASSIFICATION?
=======================
Image Classification is a computer vision task that:
1. Takes an image as input
2. Analyzes the entire image
3. Assigns it to ONE class category
4. Outputs confidence scores for each possible class

KEY CHARACTERISTICS:
- Single-label: One main class per image (unlike detection with multiple objects)
- Whole-image: Analyzes entire image, not individual objects
- Simple output: Class name + confidence score
- Fast: Typically faster than detection or segmentation

CLASSIFICATION vs DETECTION vs SEGMENTATION:
==============================================
Classification:
- Question: "What is in this image?" (one main category)
- Output: Single class label + confidence
- Speed: Fastest

Detection:
- Question: "Where are objects and what are they?" (multiple objects)
- Output: Multiple bounding boxes + classes + confidence
- Speed: Slower than classification

Segmentation:
- Question: "Which pixels belong to which object?"
- Output: Pixel-level masks + classes + confidence
- Speed: Slowest

USE CASES:
- Medical imaging: Disease classification (cancer, etc.)
- Document scanning: Document type classification
- Animal/Plant recognition: Species identification
- Product recognition: Product category classification
- Scene understanding: Indoor/outdoor, sunny/rainy
- Quality control: Defective/non-defective
- Content moderation: Appropriate/inappropriate images
- Satellite imagery: Land use classification (forest, urban, water, etc.)

================================================================================
                    SECTION 1: BASIC CLASSIFICATION SETUP
================================================================================
"""

# ============================================================================
# STEP 1: Import Required Libraries
# ============================================================================

# This line: from ultralytics import YOLO
# - Imports YOLO class from ultralytics library
# - YOLO is main class for loading and using YOLOv8 models
# - Works for detection, segmentation, AND classification
from ultralytics import YOLO

# This line: import cv2
# - Imports OpenCV library for image/video processing
# - Used for:
#   * Loading images: cv2.imread()
#   * Saving images: cv2.imwrite()
#   * Image display: cv2.imshow()
#   * Image resizing: cv2.resize()
import cv2

# This line: import numpy as np
# - Imports NumPy for numerical operations
# - Classification results are NumPy arrays
# - Used to process probabilities and create visualizations
import numpy as np

# This line: from PIL import Image
# - Imports PIL for image manipulation
# - Alternative to cv2 for saving images
# - More intuitive syntax for some operations
from PIL import Image

# This line: import matplotlib.pyplot as plt
# - Imports matplotlib for plotting and visualization
# - Used to:
#   * Display images with predictions
#   * Plot confidence bars
#   * Create multi-image visualizations
import matplotlib.pyplot as plt

# This line: import json
# - Imports JSON module for working with JSON data
# - Classification results often saved/shared as JSON
# - Useful for storing predictions in structured format
import json


# ============================================================================
# STEP 2: Loading a Classification Model
# ============================================================================

def load_classification_model():
    """
    LOADING YOLOV8 CLASSIFICATION MODELS
    ====================================
    
    YOLOv8 provides classification models trained on ImageNet dataset.
    ImageNet has 1000 classes (animals, objects, scenes, etc.)
    
    Model sizes available:
    - yolov8n-cls.pt (Nano) - Smallest, fastest
    - yolov8s-cls.pt (Small) - Good balance
    - yolov8m-cls.pt (Medium) - Recommended
    - yolov8l-cls.pt (Large) - High accuracy
    - yolov8x-cls.pt (Extra-Large) - Best accuracy
    
    Note: "-cls" suffix indicates classification model (not detection)
    """
    
    # LINE EXPLANATION 1:
    # model = YOLO("yolov8n-cls.pt")
    #
    # What it does:
    # - Creates YOLO object and loads nano classification model
    # - "-cls" suffix indicates this is classification model
    # - ".pt" is PyTorch format (weights file)
    # - Trained on ImageNet (1000 object classes)
    #
    # Process:
    # 1. YOLO class constructor called
    # 2. Checks if "yolov8n-cls.pt" exists locally
    # 3. If not found, automatically downloads
    # 4. Loads model weights into memory
    # 5. Model ready for classification predictions
    #
    # Return value: model object with .predict(), .train(), etc. methods
    model = YOLO("yolov8n-cls.pt")
    
    # LINE EXPLANATION 2:
    # print("Classification model loaded successfully!")
    #
    # What it does:
    # - Prints confirmation message
    # - Verifies model loaded without errors
    # - Useful for debugging
    print("Classification model loaded successfully!")
    
    # LINE EXPLANATION 3:
    # return model
    #
    # What it does:
    # - Returns loaded model to calling function
    # - Allows other functions to use model for predictions
    return model


# ============================================================================
# STEP 3: Classifying a Single Image
# ============================================================================

def classify_single_image():
    """
    RUNNING CLASSIFICATION ON A SINGLE IMAGE
    ========================================
    
    The classification process:
    1. Load model
    2. Load image
    3. Run inference
    4. Extract prediction results
    5. Display classification
    """
    
    # LINE EXPLANATION 1:
    # model = YOLO("yolov8m-cls.pt")
    #
    # What it does:
    # - Loads medium classification model
    # - Medium size balances speed and accuracy
    # - Good for most classification tasks
    model = YOLO("yolov8m-cls.pt")
    
    # LINE EXPLANATION 2:
    # image_path = "https://ultralytics.com/images/bus.jpg"
    #
    # What it stores:
    # - Path to image to classify
    # - Can be:
    #   * Local file path: "photos/image.jpg"
    #   * URL: "https://example.com/image.jpg"
    #   * Folder: "images/" (classifies all images)
    image_path = "https://ultralytics.com/images/bus.jpg"
    
    # LINE EXPLANATION 3:
    # results = model.predict(source=image_path, conf=0.0, imgsz=640)
    #
    # What it does:
    # - Runs classification inference on image
    # - Returns list of Result objects (one per input image)
    #
    # Parameter breakdown:
    # - source=image_path: Path to image to classify
    # - conf=0.0: Confidence threshold (typically 0 for classification)
    #   * For classification, threshold often not used
    #   * All classes ranked by confidence
    #   * Can be 0 (return all predictions)
    # - imgsz=640: Image size for inference
    #   * Classification models process at 640x640
    #   * Larger = better accuracy but slower
    #   * Must be multiple of 32
    #
    # Return value: List[Results]
    # Each Result contains:
    # - Predicted class ID
    # - Confidence score (0-1)
    # - All class probabilities
    # - Original image
    results = model.predict(source=image_path, conf=0.0, imgsz=640)
    
    # LINE EXPLANATION 4:
    # for result in results:
    #
    # What it does:
    # - Loops through each result
    # - For single image: runs once
    # - For batch of images: runs multiple times
    # - Each iteration processes one image
    for result in results:
        
        # LINE EXPLANATION 5:
        # print(f"Image: {result.path}")
        #
        # What it does:
        # - Prints path to classified image
        # - result.path contains full file path
        # - Useful for batch processing to track progress
        print(f"Image: {result.path}")
        
        # LINE EXPLANATION 6:
        # probs = result.probs
        #
        # What it contains:
        # - Classification probabilities object
        # - result.probs.data: Tensor with confidence for all classes
        # - result.probs.top1: Index of highest confidence class
        # - result.probs.top5: Indices of top 5 classes
        # - Contains all model's prediction data
        probs = result.probs
        
        # LINE EXPLANATION 7:
        # top1_class_id = probs.top1
        #
        # What it contains:
        # - Index of class with highest confidence
        # - Integer from 0 to num_classes-1
        # - For ImageNet: 0 to 999
        # - Example: 207 might be "golden_retriever"
        top1_class_id = probs.top1
        
        # LINE EXPLANATION 8:
        # top1_confidence = probs.top1conf
        #
        # What it contains:
        # - Confidence score for predicted class
        # - Range: 0 to 1 (probability)
        # - 0 = not confident, 1 = very confident
        # - Example: 0.95 = 95% confidence
        top1_confidence = probs.top1conf
        
        # LINE EXPLANATION 9:
        # class_names = result.names
        #
        # What it contains:
        # - Dictionary mapping class IDs to class names
        # - For ImageNet: {0: 'tench', 1: 'goldfish', ..., 999: 'black swan'}
        # - Used to convert class ID to human-readable name
        # - Over 1000 classes in ImageNet
        class_names = result.names
        
        # LINE EXPLANATION 10:
        # predicted_class_name = class_names[int(top1_class_id)]
        #
        # What it does:
        # - Looks up class name from class ID
        # - int(top1_class_id): Converts tensor to Python int
        # - class_names[int(...)]: Retrieves name from dictionary
        # - Returns human-readable class name
        # - Example: class_id=207 becomes "golden_retriever"
        predicted_class_name = class_names[int(top1_class_id)]
        
        # LINE EXPLANATION 11:
        # print(f"Predicted class: {predicted_class_name}")
        #
        # What it does:
        # - Prints predicted class name to console
        # - Example output: "Predicted class: school_bus"
        print(f"Predicted class: {predicted_class_name}")
        
        # LINE EXPLANATION 12:
        # print(f"Confidence: {top1_confidence:.1%}")
        #
        # What it does:
        # - Prints confidence score as percentage
        # - {top1_confidence:.1%}: Formats as percentage
        #   * .1% = 1 decimal place + percent symbol
        #   * 0.95 becomes "95.0%"
        # - Example output: "Confidence: 95.0%"
        print(f"Confidence: {top1_confidence:.1%}")


# ============================================================================
# STEP 4: Accessing Detailed Classification Results
# ============================================================================

def access_classification_results():
    """
    DETAILED BREAKDOWN OF CLASSIFICATION OUTPUT
    ===========================================
    Understanding all the data returned by classification
    """
    
    # LINE EXPLANATION 1:
    # model = YOLO("yolov8m-cls.pt")
    # Load classification model
    model = YOLO("yolov8m-cls.pt")
    
    # LINE EXPLANATION 2:
    # results = model.predict("https://ultralytics.com/images/bus.jpg")
    # Run classification
    results = model.predict(r"C:\Users\User\Desktop\YOLO-Project\images.jpg")
    
    # LINE EXPLANATION 3:
    # for result in results:
    # Loop through results
    for result in results:
        
        # ===== ACCESSING PROBABILITY DATA =====
        
        # LINE EXPLANATION 4:
        # probs = result.probs
        #
        # What it contains:
        # - Complete probability distribution
        # - Confidence score for every possible class
        # - 1000 values for ImageNet (one per class)
        # - Sum of all probabilities = 1.0 (100%)
        probs = result.probs
        
        # LINE EXPLANATION 5:
        # all_confidences = probs.data
        #
        # What it contains:
        # - Tensor with confidence for all 1000 classes
        # - Format: [conf_class0, conf_class1, ..., conf_class999]
        # - Sum = 1.0 (normalized probability distribution)
        # - Most values are very small (near 0)
        # - Only top few have significant values
        #
        # Example:
        # [0.0001, 0.0002, ..., 0.95, ..., 0.0003, ...]
        # Only one class has high confidence (0.95)
        all_confidences = probs.data
        
        # LINE EXPLANATION 6:
        # top1_class_id = probs.top1
        #
        # What it contains:
        # - Index of highest confidence class
        # - Single integer value (0-999 for ImageNet)
        # - Predicted class for this image
        # - Example: 207 (golden_retriever)
        top1_class_id = probs.top1
        
        # LINE EXPLANATION 7:
        # top1_confidence = probs.top1conf
        #
        # What it contains:
        # - Confidence score of top predicted class
        # - Single float value (0-1)
        # - Model's confidence in its prediction
        # - Example: 0.95 = 95% confidence
        top1_confidence = probs.top1conf
        
        # LINE EXPLANATION 8:
        # top5_class_ids = probs.top5
        #
        # What it contains:
        # - Indices of top 5 classes by confidence
        # - List of 5 class IDs
        # - Ranked from highest to lowest confidence
        # - Useful for showing alternative predictions
        #
        # Example:
        # top5 = [207, 209, 208, 210, 206]
        # - 1st choice: class 207 (highest confidence)
        # - 2nd choice: class 209
        # - 3rd choice: class 208
        # - etc.
        top5_class_ids = probs.top5
        
        # LINE EXPLANATION 9:
        # top5_confidences = probs.top5conf
        #
        # What it contains:
        # - Confidence scores for top 5 classes
        # - List of 5 confidence values
        # - Matched with top5_class_ids
        # - Sum NOT necessarily 1.0 (just top 5)
        #
        # Example:
        # top5conf = [0.95, 0.03, 0.01, 0.005, 0.003]
        # - 1st: 95% confidence
        # - 2nd: 3% confidence
        # - etc.
        top5_confidences = probs.top5conf
        
        # ===== ACCESSING CLASS INFORMATION =====
        
        # LINE EXPLANATION 10:
        # class_names = result.names
        #
        # What it contains:
        # - Dictionary mapping class IDs to names
        # - ImageNet: {0: 'tench', 1: 'goldfish', ..., 999: 'black swan'}
        # - Over 1000 entries total
        # - Used to convert IDs to human-readable names
        class_names = result.names
        
        # ===== DISPLAYING RESULTS =====
        
        # LINE EXPLANATION 11:
        # top1_name = class_names[int(top1_class_id)]
        #
        # Get top prediction class name
        top1_name = class_names[int(top1_class_id)]
        
        # LINE EXPLANATION 12:
        # print(f"Top prediction: {top1_name} ({top1_confidence:.1%})")
        #
        # What it does:
        # - Prints main prediction with confidence
        # - Example: "Top prediction: school_bus (95.0%)"
        print(f"Top prediction: {top1_name} ({top1_confidence:.1%})")
        
        # LINE EXPLANATION 13:
        # print("\\nTop 5 predictions:")
        #
        # What it does:
        # - Prints header for alternative predictions
        # - \\n = newline character (empty line)
        print("\nTop 5 predictions:")
        
        # LINE EXPLANATION 14:
        # for i, (class_id, confidence) in enumerate(zip(top5_class_ids, top5_confidences)):
        #
        # What it does:
        # - Loops through top 5 predictions
        # - zip(): Combines two lists into pairs
        # - enumerate(): Adds index (1-5) to each pair
        # - i: Rank (0, 1, 2, 3, 4)
        # - class_id: Class ID for this rank
        # - confidence: Confidence score for this rank
        for i, (class_id, confidence) in enumerate(zip(top5_class_ids, top5_confidences)):
            
            # LINE EXPLANATION 15:
            # class_name = class_names[int(class_id)]
            #
            # Get class name for current rank
            class_name = class_names[int(class_id)]
            
            # LINE EXPLANATION 16:
            # print(f"  {i+1}. {class_name}: {confidence:.1%}")
            #
            # What it does:
            # - Print ranked prediction with confidence
            # - {i+1}: Rank 1-5 (i starts at 0)
            # - Example output:
            #   "  1. school_bus: 95.0%"
            #   "  2. minibus: 3.0%"
            #   "  3. pickup: 1.0%"
            print(f"  {i+1}. {class_name}: {confidence:.1%}")


# ============================================================================
# STEP 5: Classifying Batch of Images
# ============================================================================

def classify_batch_images():
    """
    CLASSIFYING MULTIPLE IMAGES AT ONCE
    ===================================
    
    Efficiently classify multiple images in batch
    """
    
    # LINE EXPLANATION 1:
    # model = YOLO("yolov8m-cls.pt")
    # Load classification model
    model = YOLO("yolov8m-cls.pt")
    
    # LINE EXPLANATION 2:
    # image_folder = "path/to/images/"
    #
    # What it stores:
    # - Path to folder containing multiple images
    # - Model automatically finds all images in folder
    # - Processes all images in batch
    image_folder = "path/to/images/"
    
    # LINE EXPLANATION 3:
    # results = model.predict(
    #
    # Start classification with parameters
    results = model.predict(
        
        # LINE EXPLANATION 4:
        # source=image_folder,
        #
        # What it does:
        # - Specifies folder of images to classify
        # - Model automatically detects input is folder
        # - Finds all supported image formats (.jpg, .png, etc.)
        source=image_folder,
        
        # LINE EXPLANATION 5:
        # conf=0.0,
        #
        # What it does:
        # - Confidence threshold (typically 0 for classification)
        # - For classification, all predictions returned
        # - Threshold not typically used
        conf=0.0,
        
        # LINE EXPLANATION 6:
        # imgsz=640,
        #
        # Image size for inference
        imgsz=640,
        
        # LINE EXPLANATION 7:
        # batch=32,
        #
        # What it does:
        # - Number of images processed simultaneously
        # - batch=32: Process 32 images at a time
        # - Larger batch = faster processing but more memory
        # - Depends on available GPU/CPU memory
        # - Typical: 8-64
        batch=32,
    )
    
    # LINE EXPLANATION 8:
    # for result in results:
    #
    # Loop through each classified image
    for result in results:
        
        # LINE EXPLANATION 9:
        # probs = result.probs
        # Get probability data
        probs = result.probs
        
        # LINE EXPLANATION 10:
        # top1_name = result.names[int(probs.top1)]
        #
        # Get predicted class name
        top1_name = result.names[int(probs.top1)]
        
        # LINE EXPLANATION 11:
        # top1_conf = probs.top1conf
        #
        # Get confidence score
        top1_conf = probs.top1conf
        
        # LINE EXPLANATION 12:
        # print(f"{result.path}: {top1_name} ({top1_conf:.1%})")
        #
        # What it does:
        # - Print image filename and classification
        # - result.path: Full path to image file
        # - Example: "img1.jpg: golden_retriever (95.0%)"
        print(f"{result.path}: {top1_name} ({top1_conf:.1%})")


# ============================================================================
# STEP 6: Visualizing Classification Results
# ============================================================================

def visualize_classification():
    """
    DISPLAYING CLASSIFICATION RESULTS VISUALLY
    ==========================================
    
    Creating visual representations of predictions
    """
    
    # LINE EXPLANATION 1:
    # model = YOLO("yolov8m-cls.pt")
    # Load model
    model = YOLO("yolov8m-cls.pt")
    
    # LINE EXPLANATION 2:
    # results = model.predict("https://ultralytics.com/images/bus.jpg")
    # Run classification
    results = model.predict(r"C:\Users\User\Desktop\YOLO-Project\images.jpg")
    
    # LINE EXPLANATION 3:
    # for result in results:
    # Loop through results
    for result in results:
        
        # LINE EXPLANATION 4:
        # annotated_image = result.plot()
        #
        # What it does:
        # - Creates annotated image with classification result
        # - Built-in visualization function
        # - Returns NumPy array with prediction drawn
        # - Includes class label and confidence
        #
        # Features:
        # - Class name displayed on image
        # - Confidence percentage
        # - Professional formatting
        annotated_image = result.plot()
        
        # LINE EXPLANATION 5:
        # Image.fromarray(annotated_image).save("classification_result.jpg")
        #
        # What it does:
        # - Saves annotated image to disk
        # - Image.fromarray(): Convert NumPy array to PIL Image
        # - .save(): Write to file
        Image.fromarray(annotated_image).save("classification_result.jpg")
        print("Classification visualization saved!")
        
        # ===== MANUAL VISUALIZATION WITH MATPLOTLIB =====
        
        # LINE EXPLANATION 6:
        # probs = result.probs
        # Get probability data
        probs = result.probs
        
        # LINE EXPLANATION 7:
        # original_image = result.orig_img
        #
        # What it contains:
        # - Original input image
        # - NumPy array (height, width, 3)
        # - BGR format (OpenCV standard)
        original_image = result.orig_img
        
        # LINE EXPLANATION 8:
        # top5_ids = probs.top5
        # Get top 5 class indices
        top5_ids = probs.top5
        
        # LINE EXPLANATION 9:
        # top5_confs = probs.top5conf
        # Get top 5 confidence scores
        top5_confs = probs.top5conf
        
        # LINE EXPLANATION 10:
        # class_names = result.names
        # Get class name dictionary
        class_names = result.names
        
        # LINE EXPLANATION 11:
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        #
        # What it does:
        # - Creates matplotlib figure with 2 subplots
        # - 1, 2 = 1 row, 2 columns
        # - figsize=(12, 5) = 12 inches wide, 5 inches tall
        # - ax1: Left subplot (image)
        # - ax2: Right subplot (confidence bar chart)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # LINE EXPLANATION 12:
        # ax1.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        #
        # What it does:
        # - Display original image in left subplot
        # - cv2.cvtColor(..., cv2.COLOR_BGR2RGB): Convert BGR to RGB
        #   * OpenCV uses BGR color order
        #   * Matplotlib uses RGB
        #   * Without conversion, colors would be wrong
        # - ax1.imshow(): Display image
        ax1.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        
        # LINE EXPLANATION 13:
        # ax1.set_title(f"Image: {result.path}")
        #
        # What it does:
        # - Set title for image subplot
        # - Shows file path
        ax1.set_title(f"Image: {result.path}")
        
        # LINE EXPLANATION 14:
        # ax1.axis('off')
        #
        # What it does:
        # - Hide axes (remove x/y axis labels and ticks)
        # - Makes display cleaner
        ax1.axis('off')
        
        # LINE EXPLANATION 15:
        # top5_names = [class_names[int(id)] for id in top5_ids]
        #
        # What it does:
        # - Convert top 5 class IDs to names
        # - List comprehension: loop through top5_ids and convert each
        # - Result: list of 5 class names
        # - Example: ['school_bus', 'minibus', 'pickup', ...]
        top5_names = [class_names[int(id)] for id in top5_ids]
        
        # LINE EXPLANATION 16:
        # ax2.barh(range(5), top5_confs)
        #
        # What it does:
        # - Creates horizontal bar chart
        # - range(5): Y-axis positions 0-4
        # - top5_confs: Height of each bar (confidence values)
        # - Each bar represents one prediction
        ax2.barh(range(5), top5_confs)
        
        # LINE EXPLANATION 17:
        # ax2.set_yticks(range(5))
        #
        # What it does:
        # - Set Y-axis tick positions
        # - Positions: 0, 1, 2, 3, 4
        ax2.set_yticks(range(5))
        
        # LINE EXPLANATION 18:
        # ax2.set_yticklabels(top5_names)
        #
        # What it does:
        # - Label Y-axis with class names
        # - Each tick labeled with corresponding class name
        ax2.set_yticklabels(top5_names)
        
        # LINE EXPLANATION 19:
        # ax2.set_xlabel('Confidence')
        #
        # What it does:
        # - Label X-axis as "Confidence"
        ax2.set_xlabel('Confidence')
        
        # LINE EXPLANATION 20:
        # ax2.set_title('Top 5 Predictions')
        #
        # What it does:
        # - Set title for confidence chart
        ax2.set_title('Top 5 Predictions')
        
        # LINE EXPLANATION 21:
        # plt.tight_layout()
        #
        # What it does:
        # - Auto-adjust subplot spacing
        # - Prevents labels from overlapping
        plt.tight_layout()
        
        # LINE EXPLANATION 22:
        # plt.savefig('classification_visualization.png')
        #
        # What it does:
        # - Save figure to disk as PNG
        # - Creates professional visualization
        plt.savefig('classification_visualization.png')
        
        # LINE EXPLANATION 23:
        # plt.show()
        #
        # What it does:
        # - Display figure in window
        # - Blocks execution until window closed
        plt.show()


# ============================================================================
# STEP 7: Classifying Images from Webcam
# ============================================================================

def classify_from_webcam():
    """
    REAL-TIME CLASSIFICATION FROM WEBCAM
    ====================================
    
    Classify images captured from webcam in real-time
    """
    
    # LINE EXPLANATION 1:
    # model = YOLO("yolov8n-cls.pt")
    #
    # Load nano model (smallest for speed)
    # Using nano for minimal latency
    model = YOLO("yolov8n-cls.pt")
    
    # LINE EXPLANATION 2:
    # cap = cv2.VideoCapture(0)
    #
    # What it does:
    # - Opens webcam stream
    # - cv2.VideoCapture(0): Access default camera (ID 0)
    # - Returns VideoCapture object (stream handle)
    # - 0 = primary camera on computer
    # - Other values (1, 2, ...) for other cameras
    cap = cv2.VideoCapture(0)
    
    # LINE EXPLANATION 3:
    # while True:
    #
    # What it does:
    # - Infinite loop for continuous webcam capture
    # - Runs until user breaks it (presses 'q')
    # - Each iteration processes one frame
    while True:
        
        # LINE EXPLANATION 4:
        # ret, frame = cap.read()
        #
        # What it does:
        # - Captures one frame from webcam
        # - cap.read(): Returns (success, frame) tuple
        # - ret: Boolean (True if read successful, False otherwise)
        # - frame: NumPy array containing frame image
        # - If camera disconnects: ret=False
        ret, frame = cap.read()
        
        # LINE EXPLANATION 5:
        # if not ret:
        #
        # What it does:
        # - Check if frame capture was successful
        # - not ret: True if read failed
        # - Handles camera disconnection or error
        if not ret:
            
            # LINE EXPLANATION 6:
            # break
            #
            # Exit loop if camera disconnected
            break
        
        # LINE EXPLANATION 7:
        # results = model.predict(source=frame, conf=0.0, verbose=False)
        #
        # What it does:
        # - Classify current frame
        # - source=frame: Direct frame array (not file path)
        # - conf=0.0: No confidence threshold
        # - verbose=False: Suppress console output (quiet mode)
        # - Returns classification result for frame
        results = model.predict(source=frame, conf=0.0, verbose=False)
        
        # LINE EXPLANATION 8:
        # for result in results:
        #
        # Loop through results (typically 1)
        for result in results:
            
            # LINE EXPLANATION 9:
            # probs = result.probs
            # Get probability data
            probs = result.probs
            
            # LINE EXPLANATION 10:
            # top1_name = result.names[int(probs.top1)]
            # Get predicted class name
            top1_name = result.names[int(probs.top1)]
            
            # LINE EXPLANATION 11:
            # top1_conf = probs.top1conf
            # Get confidence score
            top1_conf = probs.top1conf
            
            # LINE EXPLANATION 12:
            # label = f"{top1_name} ({top1_conf:.1%})"
            #
            # What it does:
            # - Create label text for display
            # - Format: "class_name (confidence%)"
            # - Example: "school_bus (95.0%)"
            label = f"{top1_name} ({top1_conf:.1%})"
            
            # LINE EXPLANATION 13:
            # cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            #
            # What it does:
            # - Draw classification label on frame
            # - cv2.putText(): Text drawing function
            # - frame: Image to draw on
            # - label: Text to display
            # - (10, 30): Position (10 pixels from left, 30 from top)
            # - cv2.FONT_HERSHEY_SIMPLEX: Font type
            # - 1: Font scale (normal size)
            # - (0, 255, 0): Color in BGR (green)
            # - 2: Text thickness
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # LINE EXPLANATION 14:
        # cv2.imshow("YOLOv8 Classification", frame)
        #
        # What it does:
        # - Display frame in window
        # - cv2.imshow(): Create/update display window
        # - "YOLOv8 Classification": Window title
        # - frame: Image to display
        cv2.imshow("YOLOv8 Classification", frame)
        
        # LINE EXPLANATION 15:
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #
        # What it does:
        # - Check if user pressed 'q' key
        # - cv2.waitKey(1): Wait 1ms for key press
        #   * 1ms = responsive (checks every frame)
        #   * Returns -1 if no key pressed
        # - & 0xFF: Bitwise AND (handles 64-bit systems)
        # - ord('q'): ASCII code for 'q' character (113)
        # - Allows user to exit by pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            
            # LINE EXPLANATION 16:
            # break
            #
            # Exit main loop
            break
    
    # LINE EXPLANATION 17:
    # cap.release()
    #
    # What it does:
    # - Close/release webcam stream
    # - Frees up camera resource
    # - Allows other apps to use camera
    # - Should be called when done
    cap.release()
    
    # LINE EXPLANATION 18:
    # cv2.destroyAllWindows()
    #
    # What it does:
    # - Close all OpenCV windows
    # - Cleanup after program ends
    # - Prevents windows from staying open
    cv2.destroyAllWindows()


# ============================================================================
# STEP 8: Saving Classification Results as JSON
# ============================================================================

def save_results_as_json():
    """
    EXPORTING CLASSIFICATION RESULTS
    ===============================
    
    Save predictions to JSON format for data storage/sharing
    """
    
    # LINE EXPLANATION 1:
    # model = YOLO("yolov8m-cls.pt")
    # Load model
    model = YOLO("yolov8m-cls.pt")
    
    # LINE EXPLANATION 2:
    # results = model.predict("https://ultralytics.com/images/bus.jpg")
    # Run classification
    results = model.predict(r"C:\Users\User\Desktop\YOLO-Project\images.jpg")
    
    # LINE EXPLANATION 3:
    # predictions_list = []
    #
    # What it does:
    # - Initialize empty list for storing results
    # - Will append result dictionaries to this list
    # - Final list will contain all predictions
    predictions_list = []
    
    # LINE EXPLANATION 4:
    # for result in results:
    #
    # Loop through results
    for result in results:
        
        # LINE EXPLANATION 5:
        # probs = result.probs
        # Get probability data
        probs = result.probs
        
        # LINE EXPLANATION 6:
        # top5_ids = probs.top5.tolist()
        #
        # What it does:
        # - Convert tensor to Python list
        # - .tolist(): Convert PyTorch tensor to list
        # - Needed for JSON serialization (JSON doesn't understand tensors)
        # - Example: tensor([207, 209]) becomes [207, 209]
        top5_ids = probs.top5.tolist()
        
        # LINE EXPLANATION 7:
        # top5_confs = probs.top5conf.tolist()
        #
        # Convert confidence tensor to list
        top5_confs = probs.top5conf.tolist()
        
        # LINE EXPLANATION 8:
        # class_names = result.names
        # Get class name dictionary
        class_names = result.names
        
        # LINE EXPLANATION 9:
        # top5_names = [class_names[id] for id in top5_ids]
        #
        # What it does:
        # - Convert class IDs to names
        # - List comprehension: loop and convert each ID
        # - Result: list of 5 class names
        top5_names = [class_names[id] for id in top5_ids]
        
        # LINE EXPLANATION 10:
        # prediction_dict = {
        #
        # Create dictionary with all prediction info
        # This dictionary will be stored as JSON
        prediction_dict = {
            
            # LINE EXPLANATION 11:
            # "image_path": result.path,
            #
            # Store image file path
            "image_path": result.path,
            
            # LINE EXPLANATION 12:
            # "top1_class": class_names[int(probs.top1)],
            #
            # Store predicted class name
            "top1_class": class_names[int(probs.top1)],
            
            # LINE EXPLANATION 13:
            # "top1_confidence": float(probs.top1conf),
            #
            # What it does:
            # - Convert confidence to Python float
            # - float(...): Ensures JSON compatibility
            # - Stores confidence as decimal number
            "top1_confidence": float(probs.top1conf),
            
            # LINE EXPLANATION 14:
            # "top5_classes": top5_names,
            #
            # Store top 5 class names as list
            "top5_classes": top5_names,
            
            # LINE EXPLANATION 15:
            # "top5_confidences": top5_confs,
            #
            # Store top 5 confidence scores as list
            "top5_confidences": top5_confs,
        }
        
        # LINE EXPLANATION 16:
        # predictions_list.append(prediction_dict)
        #
        # What it does:
        # - Add this prediction dictionary to list
        # - Builds up complete list of all predictions
        predictions_list.append(prediction_dict)
    
    # LINE EXPLANATION 17:
    # with open("classifications.json", "w") as f:
    #
    # What it does:
    # - Opens file for writing
    # - "classifications.json": File to create
    # - "w": Write mode (creates new file or overwrites)
    # - with statement: Automatically closes file when done
    with open("classifications.json", "w") as f:
        
        # LINE EXPLANATION 18:
        # json.dump(predictions_list, f, indent=2)
        #
        # What it does:
        # - Write Python list to JSON file
        # - json.dump(): Serialize Python object to JSON
        # - predictions_list: Data to save
        # - f: File object
        # - indent=2: Pretty-print with 2-space indentation
        #   * indent=None: Compact (single line)
        #   * indent=2: Readable (multi-line with indentation)
        json.dump(predictions_list, f, indent=2)
    
    # LINE EXPLANATION 19:
    # print("Predictions saved to 'classifications.json'")
    #
    # Confirmation message
    print("Predictions saved to 'classifications.json'")


# ============================================================================
# STEP 9: Complete Classification Example
# ============================================================================

def complete_classification_example():
    """
    COMPLETE WORKFLOW EXAMPLE
    ==========================
    
    Full example showing entire classification pipeline
    """
    
    # STEP 1: Load model
    print("Step 1: Loading model...")
    model = YOLO("yolov8m-cls.pt")
    
    # STEP 2: Run classification
    print("Step 2: Running classification...")
    results = model.predict(
        source=r"C:\Users\User\Desktop\YOLO-Project\images.jpg",
        conf=0.0,
    )
    
    # STEP 3: Process results
    print("Step 3: Processing results...")
    for result in results:
        
        # Get data
        probs = result.probs
        class_names = result.names
        
        # Get predictions
        top1_name = class_names[int(probs.top1)]
        top1_conf = probs.top1conf
        top5_names = [class_names[int(id)] for id in probs.top5]
        top5_confs = probs.top5conf.tolist()
        
        # Print summary
        print(f"\nTop prediction: {top1_name} ({top1_conf:.1%})")
        print("\nTop 5 predictions:")
        for i, (name, conf) in enumerate(zip(top5_names, top5_confs)):
            print(f"  {i+1}. {name}: {conf:.1%}")
        
        # Visualize
        annotated_image = result.plot()
        Image.fromarray(annotated_image).save("complete_classification.jpg")
        print("\nClassification results saved to 'complete_classification.jpg'")


# ============================================================================
# QUICK REFERENCE
# ============================================================================

"""
QUICK REFERENCE - KEY COMMANDS
================================

# Load model
model = YOLO("yolov8m-cls.pt")

# Run classification
results = model.predict(source="image.jpg", conf=0.0)

# Access probabilities
probs = result.probs           # Probability object
top1_id = probs.top1           # Top class ID
top1_conf = probs.top1conf     # Top class confidence
top5_ids = probs.top5          # Top 5 class IDs
top5_confs = probs.top5conf    # Top 5 confidences
all_confs = probs.data         # All class confidences

# Get class names
class_names = result.names     # Dictionary: {0: 'tench', 1: 'goldfish', ...}
name = class_names[int(top1_id)]

# Visualize
annotated = result.plot()      # Annotated with prediction
Image.fromarray(annotated).save("output.jpg")

# Batch classification
results = model.predict(source="folder/", batch=32)

# Webcam classification
results = model.predict(source=0, verbose=False)  # 0 = webcam
"""

if __name__ == "__main__":
    # Run examples (uncomment to test)
    # load_classification_model()
    # classify_single_image()
    # access_classification_results()
    # classify_batch_images()
    # visualize_classification()
    # save_results_as_json()
    complete_classification_example()
