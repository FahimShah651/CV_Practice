"""
================================================================================
        YOLOV8 DETECTION - STEP BY STEP LINE-BY-LINE EXPLANATION
================================================================================

WHAT IS DETECTION?
==================
Object Detection is a computer vision task that:
1. Locates objects in images/videos (where are they?)
2. Identifies what they are (what are they?)
3. Provides confidence scores (how sure?)

KEY COMPONENTS:
- Bounding Boxes: Rectangle coordinates around objects [x1, y1, x2, y2]
- Class Labels: What the object is ("person", "car", "dog")
- Confidence Scores: Model's confidence (0-1 probability)

DETECTION vs SEGMENTATION:
- Detection: Rectangular boxes around objects
- Segmentation: Pixel-level masks showing object boundaries
- Both provide object location, but different precision levels

USE CASES:
- Autonomous vehicles and traffic monitoring
- Security and surveillance systems
- Retail and inventory management
- Medical imaging analysis
- Wildlife monitoring
- Robot vision applications

================================================================================
                    SECTION 1: BASIC DETECTION SETUP
================================================================================
"""

# ============================================================================
# STEP 1: Import Required Libraries
# ============================================================================

# This line: from ultralytics import YOLO
# - Imports the YOLO class from the ultralytics library
# - YOLO is the main class for loading and using YOLOv8 models
# - We'll use it to load detection models and run predictions
from ultralytics import YOLO

# This line: import cv2
# - Imports OpenCV library for image/video processing
# - Used to:
#   * Load and save images
#   * Draw bounding boxes on images
#   * Process video frames
# - Functions: cv2.imread(), cv2.rectangle(), cv2.imshow(), etc.
import cv2

# This line: import numpy as np
# - Imports NumPy for numerical operations
# - Used to:
#   * Work with arrays and coordinates
#   * Filter detections
#   * Perform mathematical operations on box coordinates
# - Detection boxes are NumPy arrays
import numpy as np

# This line: from PIL import Image
# - Imports PIL (Python Imaging Library) for image manipulation
# - Useful for:
#   * Saving detection results
#   * Converting between image formats
# - Alternative to cv2 for certain image operations
from PIL import Image

# This line: import matplotlib.pyplot as plt
# - Imports matplotlib for plotting and visualization
# - Used to:
#   * Display images with detections
#   * Create subplots
#   * Plot confidence score graphs
# - Functions: plt.imshow(), plt.show(), plt.subplot(), etc.
import matplotlib.pyplot as plt


# ============================================================================
# STEP 2: Loading a Detection Model
# ============================================================================

def load_detection_model():
    """
    LOADING YOLOV8 DETECTION MODELS
    ===============================
    
    YOLOv8 provides 5 model sizes for object detection:
    - yolov8n.pt (Nano) - Fastest, smallest, CPU-friendly
    - yolov8s.pt (Small) - Good balance of speed/accuracy
    - yolov8m.pt (Medium) - Recommended for most use cases
    - yolov8l.pt (Large) - High accuracy, requires GPU
    - yolov8x.pt (Extra-Large) - Best accuracy, slowest
    
    Performance Trade-offs:
    Size   | Speed | Accuracy | Memory | Use Case
    -------|-------|----------|--------|------------------
    nano   | ⚡⚡⚡ | Low      | 3MB    | Edge devices, CPU
    small  | ⚡⚡  | Medium   | 11MB   | Real-time, balanced
    medium | ⚡   | High     | 26MB   | General purpose
    large  | Slow  | Very High| 44MB   | Max accuracy, GPU
    xlarge | Slow  | Max      | 68MB   | Production, GPU
    """
    
    # LINE EXPLANATION 1:
    # model = YOLO("yolov8n.pt")
    #
    # What it does:
    # - Creates YOLO object and loads the nano detection model
    # - ".pt" extension means PyTorch format (weights file)
    # - "n" in "yolov8n" means nano (smallest model)
    #
    # Process flow:
    # 1. YOLO class constructor is called
    # 2. Checks if "yolov8n.pt" exists in local cache (~/.yolov8/weights/)
    # 3. If not found, automatically downloads from Ultralytics servers
    # 4. Downloads ~6MB from internet (first time only)
    # 5. Loads model weights into GPU/CPU memory
    # 6. Model ready for inference/predictions
    #
    # Return value: model object with methods like .predict(), .train(), etc.
    model = YOLO("yolov8n.pt")
    
    # LINE EXPLANATION 2:
    # print("Model loaded successfully!")
    #
    # What it does:
    # - Prints confirmation message to console
    # - Useful to verify model loaded without errors
    # - If download fails, exception is raised before this line
    print("Model loaded successfully!")
    
    # LINE EXPLANATION 3:
    # return model
    #
    # What it does:
    # - Returns the loaded model object to calling function
    # - Returned model can be used to:
    #   * Run predictions: model.predict()
    #   * Train on data: model.train()
    #   * Export to formats: model.export()
    # - Needed so other functions can use the model
    return model


# ============================================================================
# STEP 3: Running Detection on a Single Image
# ============================================================================

def detect_objects_in_image():
    """
    RUNNING OBJECT DETECTION ON A SINGLE IMAGE
    ===========================================
    
    The detection process:
    1. Load model
    2. Load image
    3. Run inference
    4. Extract results (boxes, classes, confidence)
    5. Process and visualize
    """
    
    # LINE EXPLANATION 1:
    # model = YOLO("yolov8m.pt")
    #
    # What it does:
    # - Loads medium-sized detection model
    # - Medium model balances speed and accuracy
    # - Faster than large/xlarge, more accurate than nano/small
    # - Good for general purpose object detection
    model = YOLO("yolov8m.pt")
    
    # LINE EXPLANATION 2:
    # image_path = r"C:\Users\User\Desktop\YOLO-Project\images.jpg"
    #
    # What it stores:
    # - URL to image to detect objects in
    # - Can be:
    #   * Local file path: "photos/image.jpg"
    #   * URL: "https://example.com/image.jpg"
    #   * Webcam: 0 (for default camera)
    #   * Video file: "video.mp4"
    # - Ultralytics library automatically detects input type
    image_path = r"C:\Users\User\Desktop\YOLO-Project\images.jpg"
    
    # LINE EXPLANATION 3:
    # results = model.predict(source=image_path, conf=0.25, imgsz=640)
    #
    # What it does:
    # - Runs object detection inference on the image
    # - Returns list of Result objects (one per input image)
    #
    # Parameter breakdown:
    # - source=image_path: Path/URL to image to analyze
    # - conf=0.25: Confidence threshold (0-1)
    #   * Only detections with score > 0.25 returned
    #   * Lower conf = more detections (includes weak ones)
    #   * Higher conf = fewer detections (only confident ones)
    #   * Range: 0=all detections, 1=perfect confidence only
    #   * Typical: 0.25-0.5
    # - imgsz=640: Image size for inference
    #   * Model processes image at 640x640 pixels
    #   * Larger size = better accuracy but slower inference
    #   * Must be multiple of 32: 320, 640, 1280, etc.
    #   * Trade-off: accuracy vs speed
    #
    # Return value: List[Results]
    # - Each Result contains:
    #   * Bounding boxes (xyxy, xywh formats)
    #   * Class IDs and names
    #   * Confidence scores
    #   * Original image
    results = model.predict(source=image_path, conf=0.25, imgsz=640)
    
    # LINE EXPLANATION 4:
    # for result in results:
    #
    # What it does:
    # - Iterates through each result in results list
    # - Usually one result per input image
    # - Since we provided one image, loop runs once
    # - Each result contains all detections for that image
    #
    # Why a loop?
    # - Code is generic, works for single image or batch
    # - Batch processing runs loop multiple times
    # - Makes code flexible for different use cases
    for result in results:
        
        # LINE EXPLANATION 5:
        # print(f"Image: {result.path}")
        #
        # What it does:
        # - Prints the file path of processed image
        # - result.path contains full path to input image
        # - Useful for batch processing to track progress
        # - Example output: "Image: /path/to/image.jpg"
        print(f"Image: {result.path}")
        
        # LINE EXPLANATION 6:
        # boxes = result.boxes
        #
        # What it contains:
        # - Detection boxes object containing:
        #   * Bounding box coordinates (multiple formats)
        #   * Class IDs for each detection
        #   * Confidence scores for each detection
        # - Type: ultralytics.yolo.results.Boxes
        # - Contains all detection metadata for image
        boxes = result.boxes
        
        # LINE EXPLANATION 7:
        # print(f"Number of objects detected: {len(boxes)}")
        #
        # What it does:
        # - Prints count of detected objects
        # - len(boxes) returns number of detections
        # - len() is Python built-in function for object count
        # - Example output: "Number of objects detected: 5"
        print(f"Number of objects detected: {len(boxes)}")
        
        # LINE EXPLANATION 8:
        # class_names = result.names
        #
        # What it contains:
        # - Dictionary mapping class IDs to class names
        # - result.names example:
        #   {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorbike', ...}
        # - YOLOv8 trained on COCO dataset (80 classes)
        # - Used to convert class ID (integer) to readable name (string)
        #
        # Example usage:
        # class_id = 0
        # class_name = class_names[class_id]  # Returns 'person'
        class_names = result.names


# ============================================================================
# STEP 4: Accessing Detection Results
# ============================================================================

def access_detection_results():
    """
    DETAILED BREAKDOWN OF DETECTION OUTPUT
    ======================================
    
    Understanding all the data in detection results
    """
    
    # LINE EXPLANATION 1:
    # model = YOLO("yolov8m.pt")
    # Load detection model
    model = YOLO("yolov8m.pt")
    
    # LINE EXPLANATION 2:
    # results = model.predict(r"C:\Users\User\Desktop\YOLO-Project\images.jpg", conf=0.25)
    # Run detection inference
    results = model.predict(r"C:\Users\User\Desktop\YOLO-Project\images.jpg", conf=0.25)
    
    # LINE EXPLANATION 3:
    # for result in results:
    # Loop through results
    for result in results:
        
        # ===== ACCESSING BOUNDING BOXES =====
        
        # LINE EXPLANATION 4:
        # boxes = result.boxes
        #
        # What it contains:
        # - Collection of all detected bounding boxes
        # - Each box represents one detected object
        # - Contains coordinates in multiple formats
        boxes = result.boxes
        
        # LINE EXPLANATION 5:
        # xyxy = boxes.xyxy
        #
        # What it contains:
        # - Bounding boxes in [x1, y1, x2, y2] format
        # - x1, y1: Top-left corner (row 0, column 0 is image top-left)
        # - x2, y2: Bottom-right corner
        # - Unit: Pixel coordinates (0 to image dimension)
        # - Type: Tensor of shape (num_detections, 4)
        #
        # Example:
        # xyxy = [[100, 50, 300, 200], [150, 100, 400, 350]]
        # - Box 1: (100,50) to (300,200) on image
        # - Box 2: (150,100) to (400,350) on image
        #
        # Usage:
        # - Drawing rectangles: cv2.rectangle(img, (x1,y1), (x2,y2), color)
        # - Cropping: img[y1:y2, x1:x2]
        xyxy = boxes.xyxy
        
        # LINE EXPLANATION 6:
        # xywh = boxes.xywh
        #
        # What it contains:
        # - Bounding boxes in center [x_center, y_center, width, height]
        # - x_center, y_center: Center point of box
        # - width, height: Box dimensions in pixels
        # - Type: Tensor of shape (num_detections, 4)
        #
        # Example:
        # xywh = [[200, 125, 200, 150]]
        # - Center at (200, 125)
        # - Width 200 pixels, height 150 pixels
        #
        # Relationship to xyxy:
        # x1 = x_center - width/2
        # y1 = y_center - height/2
        # x2 = x_center + width/2
        # y2 = y_center + height/2
        xywh = boxes.xywh
        
        # LINE EXPLANATION 7:
        # xyxyn = boxes.xyxyn
        #
        # What it contains:
        # - Normalized bounding boxes [x1_norm, y1_norm, x2_norm, y2_norm]
        # - Same as xyxy but values normalized to 0-1 range
        # - Normalized = pixel_value / image_dimension
        # - Independent of image size (always 0-1)
        #
        # Example (for 640x480 image):
        # xyxyn = [[0.156, 0.104, 0.469, 0.417]]
        # - This equals xyxy = [[100, 50, 300, 200]]
        # - Because: 100/640=0.156, 50/480=0.104, etc.
        #
        # Usage:
        # - Storing in databases (size-independent)
        # - Transferring between different sized images
        # - Comparing detections across images
        xyxyn = boxes.xyxyn
        
        # ===== ACCESSING CLASS INFORMATION =====
        
        # LINE EXPLANATION 8:
        # cls_ids = boxes.cls
        #
        # What it contains:
        # - Class ID for each detected object
        # - Integer value (0-79 for COCO dataset)
        # - Type: Tensor of shape (num_detections,)
        #
        # Example:
        # cls_ids = [0, 5, 2, 0, 5]
        # - Detection 1: class 0 (person)
        # - Detection 2: class 5 (bus)
        # - Detection 3: class 2 (car)
        # - etc.
        #
        # To get class name:
        # class_name = result.names[int(cls_id)]
        cls_ids = boxes.cls
        
        # ===== ACCESSING CONFIDENCE SCORES =====
        
        # LINE EXPLANATION 9:
        # conf = boxes.conf
        #
        # What it contains:
        # - Confidence score for each detection
        # - Value range: 0 to 1 (probability)
        # - 0 = definitely not object, 1 = definitely object
        # - Type: Tensor of shape (num_detections,)
        #
        # Example:
        # conf = [0.95, 0.87, 0.92, 0.65]
        # - Detection 1: 95% confident
        # - Detection 2: 87% confident
        # - Detection 3: 92% confident
        # - Detection 4: 65% confident
        #
        # Note:
        # - Already filtered by conf threshold from predict()
        # - All values > threshold specified in predict()
        conf = boxes.conf
        
        # ===== ITERATING THROUGH DETECTIONS =====
        
        # LINE EXPLANATION 10:
        # for i, (box, cls_id, conf_score) in enumerate(zip(xyxy, cls_ids, conf)):
        #
        # What it does:
        # - Loops through each detection
        # - zip(): Combines 3 lists into tuples of 3
        # - enumerate(): Adds index (0, 1, 2, ...) to each iteration
        # - i: Detection index (0, 1, 2, ...)
        # - box: [x1, y1, x2, y2] for current detection
        # - cls_id: Class ID for current detection
        # - conf_score: Confidence for current detection
        #
        # Example iteration 1:
        # i = 0, box = [100, 50, 300, 200], cls_id = 0, conf_score = 0.95
        # Example iteration 2:
        # i = 1, box = [150, 100, 400, 350], cls_id = 5, conf_score = 0.87
        for i, (box, cls_id, conf_score) in enumerate(zip(xyxy, cls_ids, conf)):
            
            # LINE EXPLANATION 11:
            # x1, y1, x2, y2 = map(int, box)
            #
            # What it does:
            # - Unpacks box coordinates from tensor
            # - map(int, box): Converts all 4 values to integers
            # - box might be tensor/float, needs int for image coordinates
            # - x1, y1: Top-left corner (pixels)
            # - x2, y2: Bottom-right corner (pixels)
            #
            # Why convert to int?
            # - Image pixel coordinates must be integers
            # - OpenCV drawing functions require int
            # - Floating point (123.45) can't specify pixel location
            x1, y1, x2, y2 = map(int, box)
            
            # LINE EXPLANATION 12:
            # class_name = result.names[int(cls_id)]
            #
            # What it does:
            # - Looks up class name from class ID
            # - int(cls_id): Converts tensor value to Python int
            # - result.names: Dictionary {0: 'person', 1: 'bicycle', ...}
            # - Returns human-readable class name
            #
            # Example:
            # cls_id = 5 (tensor)
            # int(cls_id) = 5 (Python int)
            # result.names[5] = 'bus'
            # class_name = 'bus'
            class_name = result.names[int(cls_id)]
            
            # LINE EXPLANATION 13:
            # print(f"Detection {i}: {class_name} ({conf_score:.1%}) - Box: [{x1},{y1},{x2},{y2}]")
            #
            # What it does:
            # - Prints information about detected object
            # - f"..." is formatted string (f-string)
            # - {i}: Detection index
            # - {class_name}: Object type
            # - {conf_score:.1%}: Confidence formatted as percentage
            #   * .1% means: 1 decimal place + percent symbol
            #   * 0.95 becomes "95.0%"
            # - Box coordinates: Pixel locations
            #
            # Example output:
            # "Detection 0: bus (95.0%) - Box: [100,50,300,200]"
            # "Detection 1: person (87.0%) - Box: [150,100,400,350]"
            print(f"Detection {i}: {class_name} ({conf_score:.1%}) - Box: [{x1},{y1},{x2},{y2}]")


# ============================================================================
# STEP 5: Visualizing Detection Results
# ============================================================================

def visualize_detections():
    """
    DRAWING BOUNDING BOXES AND LABELS
    ==================================
    
    Creating visual representations of detection results
    """
    
    # LINE EXPLANATION 1:
    # model = YOLO("yolov8m.pt")
    # Load model
    model = YOLO("yolov8m.pt")
    
    # LINE EXPLANATION 2:
    # results = model.predict(r"C:\Users\User\Desktop\YOLO-Project\images.jpg", conf=0.25)
    # Run detection
    results = model.predict(r"C:\Users\User\Desktop\YOLO-Project\images.jpg", conf=0.25)
    
    # LINE EXPLANATION 3:
    # for result in results:
    # Loop through results
    for result in results:
        
        # ===== METHOD 1: Using built-in plot() function =====
        
        # LINE EXPLANATION 4:
        # annotated_image = result.plot()
        #
        # What it does:
        # - Automatically draws all boxes and labels on image
        # - Built-in visualization function from Ultralytics
        # - Returns: NumPy array with boxes drawn
        # - Much easier than manual drawing
        #
        # Features automatically included:
        # - Colored bounding boxes for each object
        # - Class label above each box
        # - Confidence score percentage
        # - Color-coded by class for distinction
        # - Professional-looking annotations
        #
        # Equivalent to manually drawing boxes, but one-liner
        annotated_image = result.plot()
        
        # LINE EXPLANATION 5:
        # Image.fromarray(annotated_image).save("detection_result.jpg")
        #
        # What it does:
        # - Saves annotated image to disk
        # - Image.fromarray(): Converts NumPy array to PIL Image
        # - .save(): Writes image to file
        # - Creates file "detection_result.jpg" in current directory
        #
        # Process:
        # 1. Convert NumPy array (annotated_image) to PIL Image
        # 2. Save PIL Image to JPEG file
        # 3. File created at working directory
        Image.fromarray(annotated_image).save("detection_result.jpg")
        print("Detection visualization saved to 'detection_result.jpg'")
        
        # ===== METHOD 2: Manual drawing with OpenCV =====
        
        # LINE EXPLANATION 6:
        # original_image = result.orig_img
        #
        # What it contains:
        # - Original input image before any processing
        # - NumPy array of shape (height, width, 3) - BGR format
        # - Used as canvas for drawing boxes
        # - Unmodified, allows custom drawing
        original_image = result.orig_img
        
        # LINE EXPLANATION 7:
        # boxes = result.boxes
        # Get detection boxes
        boxes = result.boxes
        
        # LINE EXPLANATION 8:
        # canvas = original_image.copy()
        #
        # What it does:
        # - Creates copy of original image
        # - .copy(): NumPy function creating independent copy
        # - Prevents modifying original image
        # - Allows multiple visualizations from same result
        canvas = original_image.copy()
        
        # LINE EXPLANATION 9:
        # for box, cls_id, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
        #
        # Loop through each detection
        for box, cls_id, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
            
            # LINE EXPLANATION 10:
            # x1, y1, x2, y2 = map(int, box)
            #
            # Extract box coordinates as integers
            x1, y1, x2, y2 = map(int, box)
            
            # LINE EXPLANATION 11:
            # cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #
            # What it does:
            # - Draws rectangle on image
            # - cv2.rectangle(): OpenCV function to draw boxes
            # - canvas: Image to draw on (modified in-place)
            # - (x1, y1): Top-left corner coordinates
            # - (x2, y2): Bottom-right corner coordinates
            # - (0, 255, 0): Color in BGR format
            #   * B=0 (no blue), G=255 (full green), R=0 (no red)
            #   * Result: Green color
            # - 2: Box line thickness in pixels
            #
            # Result: Green rectangle drawn around detected object
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # LINE EXPLANATION 12:
            # class_name = result.names[int(cls_id)]
            #
            # Get class name for label
            class_name = result.names[int(cls_id)]
            
            # LINE EXPLANATION 13:
            # label = f"{class_name} ({conf:.2%})"
            #
            # What it does:
            # - Creates label string with class name and confidence
            # - {class_name}: Object type (e.g., "bus")
            # - {conf:.2%}: Confidence as percentage (2 decimals)
            # - Example label: "bus (95.23%)"
            #
            # Why create label string?
            # - Will be drawn on image next to box
            # - Informs viewer what was detected and confidence
            label = f"{class_name} ({conf:.2%})"
            
            # LINE EXPLANATION 14:
            # cv2.putText(canvas, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            #
            # What it does:
            # - Writes text (label) on image
            # - cv2.putText(): OpenCV text drawing function
            # - canvas: Image to draw on
            # - label: Text to draw (class name + confidence)
            # - (x1, y1-10): Position of text
            #   * x1: Same x as box (align with left edge)
            #   * y1-10: 10 pixels above top of box (above label)
            # - cv2.FONT_HERSHEY_SIMPLEX: Font type (standard sans-serif)
            # - 0.9: Font scale (0.9x normal size)
            # - (0, 255, 0): Text color (green, matches box)
            # - 2: Text line thickness in pixels
            #
            # Result: Green text label above each box
            cv2.putText(canvas, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # LINE EXPLANATION 15:
        # Image.fromarray(canvas).save("detection_manual.jpg")
        #
        # Save manually annotated image
        Image.fromarray(canvas).save("detection_manual.jpg")
        print("Manual detection visualization saved to 'detection_manual.jpg'")


# ============================================================================
# STEP 6: Filtering Detections by Confidence
# ============================================================================

def filter_detections_by_confidence():
    """
    POST-PROCESSING DETECTION RESULTS
    =================================
    
    Filtering detections to keep only high-confidence ones
    """
    
    # LINE EXPLANATION 1:
    # model = YOLO("yolov8m.pt")
    # Load model
    model = YOLO("yolov8m.pt")
    
    # LINE EXPLANATION 2:
    # results = model.predict(r"C:\Users\User\Desktop\YOLO-Project\images.jpg", conf=0.25)
    #
    # Run detection with low confidence threshold
    # This returns all detections > 0.25 confidence
    results = model.predict(r"C:\Users\User\Desktop\YOLO-Project\images.jpg", conf=0.25)
    
    # LINE EXPLANATION 3:
    # for result in results:
    # Loop through results
    for result in results:
        
        # LINE EXPLANATION 4:
        # boxes = result.boxes
        # Get detection boxes
        boxes = result.boxes
        
        # LINE EXPLANATION 5:
        # confidence_threshold = 0.5
        #
        # What it does:
        # - Define high confidence threshold
        # - Only keep detections > 50% confidence
        # - Reduces false positives
        # - More strict than model's default 0.25
        confidence_threshold = 0.5
        
        # LINE EXPLANATION 6:
        # high_conf_mask = boxes.conf > confidence_threshold
        #
        # What it does:
        # - Creates boolean mask (True/False for each detection)
        # - boxes.conf > 0.5: Compares each confidence score
        # - Result: Boolean array [True, False, True, True, False, ...]
        # - True: Confidence > 0.5 (keep this detection)
        # - False: Confidence <= 0.5 (reject this detection)
        #
        # Example:
        # boxes.conf = [0.95, 0.45, 0.92, 0.85, 0.30]
        # high_conf_mask = [True, False, True, True, False]
        high_conf_mask = boxes.conf > confidence_threshold
        
        # LINE EXPLANATION 7:
        # high_conf_detections = boxes[high_conf_mask]
        #
        # What it does:
        # - Uses boolean mask to filter boxes
        # - Keeps only detections where mask is True
        # - Discards low-confidence detections
        #
        # This is called "boolean indexing" in NumPy
        # Example:
        # boxes = [A, B, C, D, E]
        # mask = [True, False, True, True, False]
        # result = [A, C, D] (only True entries)
        high_conf_detections = boxes[high_conf_mask]
        
        # LINE EXPLANATION 8:
        # print(f"Total detections: {len(boxes)}")
        #
        # Print count of all detections
        print(f"Total detections: {len(boxes)}")
        
        # LINE EXPLANATION 9:
        # print(f"High-confidence detections (>{confidence_threshold:.0%}): {len(high_conf_detections)}")
        #
        # What it does:
        # - Prints count of filtered detections
        # - Shows how many were kept after filtering
        # - {confidence_threshold:.0%}: Formats threshold as percentage
        #   * .0% means: 0 decimal places + percent symbol
        #   * 0.5 becomes "50%"
        #
        # Example output:
        # "High-confidence detections (>50%): 3"
        # (3 out of maybe 5 detections were high confidence)
        print(f"High-confidence detections (>{confidence_threshold:.0%}): {len(high_conf_detections)}")


# ============================================================================
# STEP 7: Detecting Objects in Video
# ============================================================================

def detect_in_video():
    """
    RUNNING DETECTION ON VIDEO
    ==========================
    
    Process video frames to detect objects across frames
    """
    
    # LINE EXPLANATION 1:
    # model = YOLO("yolov8m.pt")
    # Load detection model
    model = YOLO("yolov8m.pt")
    
    # LINE EXPLANATION 2:
    # video_path = "video.mp4"
    #
    # What it stores:
    # - Path to video file to process
    # - Can be local file or path
    # - Model automatically detects this is video (from .mp4 extension)
    video_path = "video.mp4"
    
    # LINE EXPLANATION 3:
    # results = model.predict(source=video_path, conf=0.25, imgsz=640)
    #
    # What it does:
    # - Runs detection on all video frames
    # - model.predict automatically detects input is video
    # - For video: processes frame-by-frame automatically
    # - Returns generator (memory efficient, not full list)
    #
    # Why generator?
    # - Videos have many frames (thousands or millions)
    # - Loading all at once would use massive memory
    # - Generator yields one result at a time
    results = model.predict(source=video_path, conf=0.25, imgsz=640)
    
    # LINE EXPLANATION 4:
    # frame_count = 0
    #
    # What it does:
    # - Initialize frame counter to track progress
    # - Used to count which frame we're processing
    frame_count = 0
    
    # LINE EXPLANATION 5:
    # for result in results:
    #
    # What it does:
    # - Loops through each frame's detection result
    # - Results generated one at a time from video
    # - Similar to image loop, but for video frames
    for result in results:
        
        # LINE EXPLANATION 6:
        # frame_count += 1
        #
        # What it does:
        # - Increment frame counter by 1
        # - += means: frame_count = frame_count + 1
        # - Tracks which frame we're on
        frame_count += 1
        
        # LINE EXPLANATION 7:
        # if frame_count % 10 == 0:
        #
        # What it does:
        # - Check if current frame is multiple of 10
        # - % is modulo operator (remainder after division)
        # - frame_count % 10 == 0 is True for frames: 10, 20, 30, ...
        # - False for frames: 1, 2, 3, ..., 9, 11, 12, ...
        #
        # Why check every 10th frame?
        # - Video typically has 30 FPS (frames per second)
        # - Processing every frame: 30 per second = slow
        # - Every 10th frame: 3 per second = reasonable speed
        # - Still provides good temporal coverage
        if frame_count % 10 == 0:
            
            # LINE EXPLANATION 8:
            # boxes = result.boxes
            # Get detections for current frame
            boxes = result.boxes
            
            # LINE EXPLANATION 9:
            # print(f"Frame {frame_count}: {len(boxes)} objects detected")
            #
            # Print progress and detection count
            print(f"Frame {frame_count}: {len(boxes)} objects detected")
            
            # LINE EXPLANATION 10:
            # annotated_frame = result.plot()
            #
            # Draw boxes on frame
            annotated_frame = result.plot()
            
            # LINE EXPLANATION 11:
            # output_path = f"frame_{frame_count:05d}.jpg"
            #
            # What it does:
            # - Create filename for saving frame
            # - f"..." is f-string for formatting
            # - {frame_count:05d}: Frame number padded with zeros
            #   * :05d means: integer, 5 digits, pad with zeros
            #   * frame_count=10 becomes "00010"
            #   * frame_count=100 becomes "00100"
            #
            # Why pad with zeros?
            # - Makes files sort correctly alphabetically
            # - Without padding: frame_1, frame_10, frame_2 (wrong order)
            # - With padding: frame_00001, frame_00010, frame_00002 (correct)
            output_path = f"frame_{frame_count:05d}.jpg"
            
            # LINE EXPLANATION 12:
            # Image.fromarray(annotated_frame).save(output_path)
            #
            # Save annotated frame to disk
            Image.fromarray(annotated_frame).save(output_path)


# ============================================================================
# STEP 8: Real-time Webcam Detection
# ============================================================================

def detect_from_webcam():
    """
    REAL-TIME DETECTION FROM WEBCAM
    ===============================
    
    Detect objects from live webcam stream
    """
    
    # LINE EXPLANATION 1:
    # model = YOLO("yolov8n.pt")
    #
    # Load nano model (fastest for real-time)
    # Using smallest model for minimum latency
    model = YOLO("yolov8n.pt")
    
    # LINE EXPLANATION 2:
    # results = model.predict(source=0, conf=0.25, device=0)
    #
    # What it does:
    # - Runs real-time detection from webcam
    # - source=0: Default webcam (ID 0)
    #   * 0 = primary camera on computer
    # - conf=0.25: Confidence threshold
    # - device=0: GPU device ID (0 for first GPU)
    #
    # Returns: Generator of detection results
    results = model.predict(source=0, conf=0.25, device=0)
    
    # LINE EXPLANATION 3:
    # for result in results:
    #
    # Loop through webcam frames
    for result in results:
        
        # LINE EXPLANATION 4:
        # annotated_frame = result.plot()
        #
        # Draw detections on frame
        annotated_frame = result.plot()
        
        # LINE EXPLANATION 5:
        # cv2.imshow("YOLOv8 Detection", annotated_frame)
        #
        # What it does:
        # - Display frame in window on screen
        # - cv2.imshow(): OpenCV function to display image
        # - "YOLOv8 Detection": Window title
        # - annotated_frame: Image to display (with boxes drawn)
        #
        # Creates window that updates every frame
        cv2.imshow("YOLOv8 Detection", annotated_frame)
        
        # LINE EXPLANATION 6:
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #
        # What it does:
        # - Check if user pressed 'q' key
        # - cv2.waitKey(1): Wait 1ms for key press
        #   * 1ms = responsive (checks every frame)
        #   * Returns -1 if no key pressed
        #   * Returns ASCII code of pressed key
        # - & 0xFF: Bitwise AND operation (handles 64-bit systems)
        # - == ord('q'): Check if key is 'q'
        #   * ord('q') = ASCII code 113
        #
        # Allows user to exit by pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            
            # LINE EXPLANATION 7:
            # break
            #
            # Exit the loop (stop webcam stream)
            break
    
    # LINE EXPLANATION 8:
    # cv2.destroyAllWindows()
    #
    # What it does:
    # - Close all OpenCV windows
    # - Cleanup after webcam loop ends
    # - Prevents windows from staying open
    cv2.destroyAllWindows()


# ============================================================================
# STEP 9: Complete Detection Example
# ============================================================================

def complete_detection_example():
    """
    COMPLETE WORKFLOW EXAMPLE
    ==========================
    
    Full example showing entire detection pipeline:
    Load → Detect → Process → Visualize
    """
    
    # STEP 1: Load model
    print("Step 1: Loading model...")
    model = YOLO("yolov8m.pt")
    
    # STEP 2: Run detection
    print("Step 2: Running detection...")
    results = model.predict(
        source=r"C:\Users\User\Desktop\YOLO-Project\images.jpg",
        conf=0.25,
        imgsz=640,
    )
    
    # STEP 3: Process results
    print("Step 3: Processing results...")
    for result in results:
        
        # Get data
        boxes = result.boxes
        class_names = result.names
        
        # Print summary
        print(f"Objects detected: {len(boxes)}")
        
        # Analyze each detection
        for i, (box, cls_id, conf) in enumerate(
            zip(boxes.xyxy, boxes.cls, boxes.conf)
        ):
            class_name = class_names[int(cls_id)]
            x1, y1, x2, y2 = map(int, box)
            print(f"  {i}: {class_name} ({conf:.1%}) - [{x1},{y1},{x2},{y2}]")
        
        # Visualize
        annotated_image = result.plot()
        Image.fromarray(annotated_image).save("complete_detection.jpg")
        print("\nDetection results saved to 'complete_detection.jpg'")


# ============================================================================
# QUICK REFERENCE
# ============================================================================

"""
QUICK REFERENCE - KEY COMMANDS
================================

# Load model
model = YOLO("yolov8m.pt")

# Run detection
results = model.predict(source="image.jpg", conf=0.25)

# Access boxes
boxes = result.boxes           # All boxes
boxes.xyxy                     # [x1, y1, x2, y2] format
boxes.xywh                     # [x_center, y_center, w, h] format
boxes.xyxyn                    # Normalized coordinates
boxes.cls                      # Class IDs
boxes.conf                     # Confidence scores

# Get class names
class_names = result.names     # Dictionary: {0: 'person', 1: 'car', ...}
name = class_names[int(cls_id)]

# Visualize
annotated = result.plot()      # Draw all boxes
Image.fromarray(annotated).save("output.jpg")

# Filter by confidence
high_conf = boxes[boxes.conf > 0.5]

# Draw manually
cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
"""

if __name__ == "__main__":
    # Run examples (uncomment to test)
    # load_detection_model()
    # detect_objects_in_image()
    # access_detection_results()
    # visualize_detections()
    # filter_detections_by_confidence()
    complete_detection_example()
