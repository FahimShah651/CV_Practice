"""
================================================================================
        YOLOV8 SEGMENTATION - STEP BY STEP LINE-BY-LINE EXPLANATION
================================================================================

WHAT IS SEGMENTATION?
=====================
Segmentation is a computer vision task that goes beyond object detection.
Instead of just drawing a box around objects, it provides pixel-level masks
that show exactly which pixels belong to each object.

KEY DIFFERENCES FROM DETECTION:
- Detection: Rectangular bounding boxes around objects
- Segmentation: Pixel-perfect masks for each object (Instance Segmentation)
- Semantic Segmentation: Each pixel is labeled with a class
- Panoptic Segmentation: Combines semantic and instance segmentation

USE CASES:
- Medical imaging (tumor/organ segmentation)
- Autonomous vehicles (road segmentation, object boundaries)
- Satellite imagery analysis
- Image editing and composition
- Robot manipulation (grasping)
- Quality control in manufacturing

================================================================================
                    SECTION 1: BASIC SEGMENTATION SETUP
================================================================================
"""

# ============================================================================
# STEP 1: Import Required Libraries
# ============================================================================

# This line: from ultralytics import YOLO
# - Imports the YOLO class from the ultralytics library
# - YOLO is the main class for loading and using YOLOv8 models
# - We'll use it to load segmentation models and run inference
from ultralytics import YOLO

# This line: import cv2
# - Imports OpenCV library for image/video processing
# - Used to display images, draw masks, and manipulate video frames
# - Functions: cv2.imread(), cv2.imshow(), cv2.polylines(), etc.
import cv2

# This line: import numpy as np
# - Imports NumPy for numerical operations and array handling
# - Segmentation masks are numpy arrays, we need NumPy to process them
# - Functions: np.array(), np.where(), np.multiply(), etc.
import numpy as np

# This line: from PIL import Image
# - Imports PIL (Python Imaging Library) for image handling
# - Useful for saving and loading images in different formats
# - Alternative to cv2 for certain image operations
from PIL import Image

# This line: import matplotlib.pyplot as plt
# - Imports matplotlib for plotting and visualization
# - Used to display images with masks overlaid
# - Functions: plt.imshow(), plt.show(), plt.subplot(), etc.
import matplotlib.pyplot as plt

# This line: from pathlib import Path
# - Imports Path for cross-platform file path handling
# - More reliable than string paths (works on Windows, Mac, Linux)
# - Functions: Path.exists(), Path.mkdir(), etc.
from pathlib import Path


# ============================================================================
# STEP 2: Loading a Segmentation Model
# ============================================================================

def load_segmentation_model():
    """
    LOADING YOLOV8 SEGMENTATION MODELS
    ===================================
    
    YOLOv8 provides 5 model sizes for segmentation:
    - yolov8n-seg.pt (Nano) - Fastest, smallest
    - yolov8s-seg.pt (Small) - Good balance
    - yolov8m-seg.pt (Medium) - Recommended
    - yolov8l-seg.pt (Large) - High accuracy
    - yolov8x-seg.pt (Extra-Large) - Best accuracy, slowest
    """
    
    # LINE EXPLANATION 1:
    # model = YOLO("yolov8n-seg.pt")
    # 
    # What it does:
    # - Creates a YOLO object and loads the nano segmentation model
    # - The "-seg" suffix indicates this is a segmentation model
    # - ".pt" is PyTorch format (weights file)
    # 
    # Process:
    # 1. YOLO class constructor is called
    # 2. Checks if "yolov8n-seg.pt" exists locally
    # 3. If not found, automatically downloads from Ultralytics servers
    # 4. Loads model weights into memory
    # 5. Model is ready for inference
    #
    # Return value: model object that can run predictions
    model = YOLO("yolov8n-seg.pt")
    
    # LINE EXPLANATION 2:
    # print("Model loaded successfully!")
    # 
    # What it does:
    # - Prints a confirmation message to console
    # - Useful to verify the model loaded without errors
    print("Model loaded successfully!")
    
    # LINE EXPLANATION 3:
    # return model
    #
    # What it does:
    # - Returns the loaded model object to the caller
    # - The returned object can be used to run inference
    return model


# ============================================================================
# STEP 3: Segmenting a Single Image
# ============================================================================

def segment_image():
    """
    RUNNING SEGMENTATION ON A SINGLE IMAGE
    =======================================
    
    The segmentation process:
    1. Load model
    2. Run prediction on image
    3. Extract masks
    4. Visualize results
    """
    
    # LINE EXPLANATION 1:
    # model = YOLO("yolov8m-seg.pt")
    #
    # What it does:
    # - Loads the medium segmentation model
    # - The medium model balances speed and accuracy
    # - More accurate than nano, but slower
    model = YOLO("yolov8m-seg.pt")
    
    # LINE EXPLANATION 2:
    # image_path = "path/to/image.jpg"
    #
    # What it does:
    # - Stores the path to the image file
    # - Can be a local path or URL
    # - Example: "photos/cat.jpg" or "https://example.com/image.jpg"
    image_path = r"C:\Users\User\Desktop\YOLO-Project\images.jpg"
    
    # LINE EXPLANATION 3:
    # results = model.predict(source=image_path, conf=0.25, imgsz=640)
    #
    # What it does:
    # - Runs segmentation inference on the image
    # - Returns a list of Result objects (one per image)
    #
    # Parameter breakdown:
    # - source=image_path: Path/URL to image to segment
    # - conf=0.25: Confidence threshold (0-1)
    #   * Only detections with confidence > 0.25 are returned
    #   * Lower = more detections (more false positives possible)
    #   * Higher = fewer detections (might miss objects)
    # - imgsz=640: Image size for inference
    #   * Model processes image at 640x640 pixels
    #   * Larger = better accuracy but slower
    #   * Must be multiple of 32 (e.g., 320, 640, 1280)
    #
    # Return value: List of Result objects containing:
    # - Segmentation masks
    # - Bounding boxes
    # - Confidence scores
    # - Class labels
    results = model.predict(source=image_path, conf=0.25, imgsz=640)
    
    # LINE EXPLANATION 4:
    # for result in results:
    #
    # What it does:
    # - Iterates through each result in the results list
    # - Usually one result per input image
    # - We process one image, so loop runs once
    for result in results:
        
        # LINE EXPLANATION 5:
        # print(f"Image: {result.path}")
        #
        # What it does:
        # - Prints the file path of the processed image
        # - result.path contains the full path to the input image
        # - Useful for batch processing to track which image is processed
        print(f"Image: {result.path}")
        
        # LINE EXPLANATION 6:
        # masks = result.masks
        #
        # What it does:
        # - Extracts the segmentation masks from the result
        # - result.masks is a Masks object containing:
        #   * Pixel-level binary masks for each detected object
        #   * One mask per detected instance (not per class)
        #   * Shape: (num_detections, height, width)
        #
        # Example: If 3 cars detected, masks.shape = (3, 640, 640)
        # Each mask is a binary image (0 = background, 1 = object)
        masks = result.masks
        
        # LINE EXPLANATION 7:
        # boxes = result.boxes
        #
        # What it does:
        # - Extracts bounding boxes from the result
        # - result.boxes contains:
        #   * Box coordinates in various formats (xyxy, xywh, etc.)
        #   * Confidence scores for each detection
        #   * Class IDs for each detection
        #
        # These boxes are the bounding regions around segmented objects
        boxes = result.boxes
        
        # LINE EXPLANATION 8:
        # class_names = result.names
        #
        # What it does:
        # - Gets the mapping of class IDs to class names
        # - result.names is a dictionary: {0: 'person', 1: 'bicycle', ...}
        # - YOLOv8 is trained on COCO dataset (80 classes)
        # - Used to label detections with human-readable names
        class_names = result.names
        
        # LINE EXPLANATION 9:
        # print(f"Number of objects detected: {len(boxes)}")
        #
        # What it does:
        # - Prints the count of detected objects
        # - len(boxes) returns number of detections
        # - Example output: "Number of objects detected: 5"
        print(f"Number of objects detected: {len(boxes)}")
        
        # LINE EXPLANATION 10:
        # if masks is not None:
        #
        # What it does:
        # - Checks if segmentation masks were detected
        # - masks is None if no objects detected
        # - Prevents errors when trying to access mask data
        if masks is not None:
            
            # LINE EXPLANATION 11:
            # print(f"Mask shape: {masks.data.shape}")
            #
            # What it does:
            # - Prints dimensions of the mask array
            # - masks.data is a tensor containing all masks
            # - Shape format: (num_objects, height, width)
            # - Example: (5, 640, 640) = 5 objects in 640x640 image
            print(f"Mask shape: {masks.data.shape}")


# ============================================================================
# STEP 4: Detailed Analysis of Segmentation Results
# ============================================================================

def analyze_segmentation_results():
    """
    DETAILED BREAKDOWN OF SEGMENTATION OUTPUTS
    ===========================================
    
    Understanding all the data returned by model.predict()
    """
    
    # LINE EXPLANATION 1:
    # model = YOLO("yolov8m-seg.pt")
    # Loads medium segmentation model
    model = YOLO("yolov8m-seg.pt")
    
    # LINE EXPLANATION 2:
    # results = model.predict(r"C:\Users\User\Desktop\YOLO-Project\images.jpg")
    # Runs inference on the image
    results = model.predict(r"C:\Users\User\Desktop\YOLO-Project\images.jpg")
    
    # LINE EXPLANATION 3:
    # for result in results:
    # Loops through each result (typically one per image)
    for result in results:
        
        # ===== ACCESSING MASKS =====
        
        # LINE EXPLANATION 4:
        # masks = result.masks
        #
        # What it contains:
        # - masks.data: Tensor of shape (num_detections, height, width)
        # - masks.xy: List of contours in (x, y) format
        #   * Each contour defines the boundary of an object
        #   * Can be used for precise object boundary
        #
        # Example: For a detected dog, mask shows exactly which pixels
        # are part of the dog (1) and which are background (0)
        masks = result.masks
        
        # LINE EXPLANATION 5:
        # mask_array = masks.data.cpu().numpy()
        #
        # What it does:
        # - Converts mask tensor to NumPy array
        # - .cpu() moves tensor from GPU to CPU memory
        # - .numpy() converts PyTorch tensor to NumPy array
        # - Result: Array of shape (num_detections, 640, 640)
        #
        # Why convert?
        # - NumPy arrays are easier to manipulate
        # - Can use NumPy functions for processing
        # - Can save to disk using standard image formats
        mask_array = masks.data.cpu().numpy()
        
        # LINE EXPLANATION 6:
        # boxes = result.boxes
        #
        # Contains all bounding box and detection information:
        # - boxes.xyxy: Top-left and bottom-right corners [[x1,y1,x2,y2],...]
        # - boxes.cls: Class IDs for each detection [0, 1, 1, 2, ...]
        # - boxes.conf: Confidence scores [0.95, 0.87, 0.92, ...]
        # - boxes.xywh: Center coordinates and dimensions
        boxes = result.boxes
        
        # ===== ITERATING THROUGH DETECTIONS =====
        
        # LINE EXPLANATION 7:
        # for i, (mask, box, cls_id, conf) in enumerate(
        #
        # What it does:
        # - Loops through each detected object
        # - Unpacks mask, box info, class, and confidence in each iteration
        # - enumerate() provides index (0, 1, 2, ...) for each object
        # - i: Index of current detection
        # - mask: Binary mask for current object
        # - box: Bounding box coordinates for current object
        # - cls_id: Class ID for current object
        # - conf: Confidence score for current object
        #
        # zip() combines 4 separate lists into tuples of 4
        for i, (mask, box, cls_id, conf) in enumerate(
            zip(mask_array, boxes.xyxy, boxes.cls, boxes.conf)
        ):
            
            # LINE EXPLANATION 8:
            # class_name = result.names[int(cls_id)]
            #
            # What it does:
            # - Looks up the class name from class ID
            # - int(cls_id) converts tensor to Python int
            # - result.names is dictionary: {0: 'person', 1: 'bicycle', ...}
            # - Returns human-readable name, e.g., 'person'
            class_name = result.names[int(cls_id)]
            
            # LINE EXPLANATION 9:
            # x1, y1, x2, y2 = map(int, box)
            #
            # What it does:
            # - Unpacks bounding box coordinates
            # - box = [x1, y1, x2, y2] (top-left and bottom-right corners)
            # - map(int, box) converts all coordinates to integers
            # - x1, y1: Top-left corner pixel coordinates
            # - x2, y2: Bottom-right corner pixel coordinates
            # - Pixel coordinates range from 0 to image dimensions
            x1, y1, x2, y2 = map(int, box)
            
            # LINE EXPLANATION 10:
            # mask_pixels = np.where(mask > 0.5)
            #
            # What it does:
            # - Finds all pixels where mask value > 0.5 (considered object)
            # - Masks are probabilistic: values 0-1 (not binary)
            # - 0.5 threshold: pixels with >50% confidence are object
            # - np.where returns tuple: (y_coordinates, x_coordinates)
            # - Used to extract exact pixels belonging to object
            #
            # Example output:
            # (array([100, 101, 102, ...]), array([200, 201, 202, ...]))
            # Means pixels at (100,200), (101,201), (102,202), ... are object
            mask_pixels = np.where(mask > 0.5)
            
            # LINE EXPLANATION 11:
            # print(f"Object {i}: {class_name} ({conf:.1%}) - Box: [{x1},{y1},{x2},{y2}]")
            #
            # What it does:
            # - Prints information about detected object
            # - {i}: Object index (0, 1, 2, ...)
            # - {class_name}: Type of object (e.g., "bus", "person")
            # - {conf:.1%}: Confidence score formatted as percentage
            #   * .1% means 1 decimal place and % symbol
            #   * Example: 0.95 becomes "95.0%"
            # - Box coordinates: Pixel locations of bounding box
            #
            # Example output:
            # "Object 0: bus (92.5%) - Box: [100,50,400,350]"
            print(f"Object {i}: {class_name} ({conf:.1%}) - Box: [{x1},{y1},{x2},{y2}]")
            
            # LINE EXPLANATION 12:
            # num_mask_pixels = np.sum(mask > 0.5)
            #
            # What it does:
            # - Counts number of pixels belonging to this object
            # - mask > 0.5: Creates boolean array (True/False for each pixel)
            # - np.sum: Adds up all True values (counts pixels)
            # - Result: Integer count of object pixels
            #
            # Useful for:
            # - Calculating object size/area
            # - Filtering out small noise detections
            # - Estimating object prominence in image
            num_mask_pixels = np.sum(mask > 0.5)
            
            # LINE EXPLANATION 13:
            # print(f"  Mask pixels: {num_mask_pixels}")
            #
            # What it does:
            # - Prints number of pixels belonging to detected object
            # - Example: "  Mask pixels: 15234" (15,234 pixels are this object)
            print(f"  Mask pixels: {num_mask_pixels}")


# ============================================================================
# STEP 5: Visualizing Segmentation Results
# ============================================================================

def visualize_segmentation():
    """
    DRAWING MASKS AND VISUALIZING RESULTS
    =====================================
    
    How to create visual representations of segmentation results
    """
    
    # LINE EXPLANATION 1:
    # model = YOLO("yolov8m-seg.pt")
    # Loads the segmentation model
    model = YOLO("yolov8m-seg.pt")
    
    # LINE EXPLANATION 2:
    # results = model.predict(r"C:\Users\User\Desktop\YOLO-Project\images.jpg")
    # Runs inference
    results = model.predict(r"C:\Users\User\Desktop\YOLO-Project\images.jpg")
    
    # LINE EXPLANATION 3:
    # for result in results:
    # Loops through results
    for result in results:
        
        # ===== METHOD 1: Using built-in plot() function =====
        
        # LINE EXPLANATION 4:
        # annotated_image = result.plot()
        #
        # What it does:
        # - Automatically draws all masks and boxes on the image
        # - Built-in visualization function from Ultralytics
        # - Returns: NumPy array with masks and labels drawn
        # - Much simpler than manual drawing
        #
        # Features automatically included:
        # - Colored masks for each object
        # - Bounding boxes around objects
        # - Class labels with confidence scores
        # - Color-coded by class for distinction
        annotated_image = result.plot()
        
        # LINE EXPLANATION 5:
        # image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        #
        # What it does:
        # - Converts image color space from BGR to RGB
        # - OpenCV uses BGR (Blue, Green, Red) by default
        # - Matplotlib expects RGB (Red, Green, Blue)
        # - cv2.cvtColor: Image color conversion function
        # - Without this, colors would be wrong (blue becomes red, etc.)
        image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        # LINE EXPLANATION 6:
        # Image.fromarray(image_rgb).save("segmentation_result.jpg")
        #
        # What it does:
        # - Saves the annotated image to disk
        # - Image.fromarray(): Converts NumPy array to PIL Image
        # - .save(): Saves image to specified file
        # - Creates file "segmentation_result.jpg"
        #
        # The process:
        # 1. Convert NumPy array to PIL Image format
        # 2. Save to JPEG file
        # 3. File stored in current working directory
        Image.fromarray(image_rgb).save("segmentation_result.jpg")
        print("Segmentation visualization saved!")
        
        # ===== METHOD 2: Manual mask drawing with OpenCV =====
        
        # LINE EXPLANATION 7:
        # original_image = result.orig_img
        #
        # What it does:
        # - Gets the original input image
        # - result.orig_img is the raw image before any processing
        # - NumPy array of shape (height, width, 3) - RGB format
        # - Used as canvas for drawing masks
        original_image = result.orig_img
        
        # LINE EXPLANATION 8:
        # masks = result.masks
        #
        # Get segmentation masks
        masks = result.masks
        
        # LINE EXPLANATION 9:
        # mask_array = masks.data.cpu().numpy()
        # Convert to NumPy array for processing
        mask_array = masks.data.cpu().numpy()
        
        # LINE EXPLANATION 10:
        # mask_image = original_image.copy()
        #
        # What it does:
        # - Creates a copy of original image
        # - .copy(): NumPy function that creates independent copy
        # - Why needed? So we don't modify the original image
        # - Allows us to overlay masks on the copy
        mask_image = original_image.copy()
        
        # LINE EXPLANATION 11:
        # for i, mask in enumerate(mask_array):
        #
        # What it does:
        # - Loops through each segmentation mask
        # - i: Index of current mask (0, 1, 2, ...)
        # - mask: Binary mask array for current object
        for i, mask in enumerate(mask_array):
            
            # LINE EXPLANATION 12:
            # mask_binary = (mask > 0.5).astype(np.uint8) * 255
            #
            # What it does:
            # - Converts probabilistic mask to binary image (0 or 255)
            # - mask > 0.5: Boolean array (True where object, False elsewhere)
            # - .astype(np.uint8): Convert bool to unsigned int (0 or 1)
            # - * 255: Convert 1 to 255 (white in image)
            # - Result: Image where object=255 (white), background=0 (black)
            #
            # Why 255? Standard image value range is 0-255
            # 0 = black (background)
            # 255 = white (object)
            mask_binary = (mask > 0.5).astype(np.uint8) * 255
            
            # LINE EXPLANATION 13:
            # random_color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            #
            # What it does:
            # - Generates random color for this object's mask
            # - np.random.randint(0, 255): Random int from 0 to 254
            # - Three values: BGR (Blue, Green, Red) color tuple
            # - Each object gets unique color for visual distinction
            #
            # Example random_color: (128, 200, 50) - greenish color
            random_color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            
            # LINE EXPLANATION 14:
            # mask_image[mask_binary == 255] = random_color
            #
            # What it does:
            # - Applies color to pixels where mask=255 (object pixels)
            # - mask_image[mask_binary == 255]: Selects pixels where mask is object
            # - = random_color: Sets those pixels to random color
            # - Result: Object pixels become colored, background unchanged
            #
            # This is called "masking" - only modify pixels matching condition
            mask_image[mask_binary == 255] = random_color
        
        # LINE EXPLANATION 15:
        # blended_image = cv2.addWeighted(original_image, 0.7, mask_image, 0.3, 0)
        #
        # What it does:
        # - Blends original image with colored masks
        # - cv2.addWeighted(src1, alpha, src2, beta, gamma)
        #   * src1: First image (original)
        #   * alpha: Weight for src1 (0.7 = 70%)
        #   * src2: Second image (colored masks)
        #   * beta: Weight for src2 (0.3 = 30%)
        #   * gamma: Brightness adjustment (0 = no change)
        # - Formula: result = src1*0.7 + src2*0.3
        # - Result: See-through masks overlaid on original
        #
        # Why blend?
        # - Masks alone hide original image details
        # - Blending shows both original and segmentation
        # - 70% original + 30% masks = transparent effect
        blended_image = cv2.addWeighted(original_image, 0.7, mask_image, 0.3, 0)
        
        # LINE EXPLANATION 16:
        # Image.fromarray(blended_image).save("blended_segmentation.jpg")
        #
        # Save the blended visualization to disk
        Image.fromarray(blended_image).save("blended_segmentation.jpg")
        print("Blended segmentation saved!")


# ============================================================================
# STEP 6: Segmenting Video Frames
# ============================================================================

def segment_video():
    """
    RUNNING SEGMENTATION ON VIDEO
    ==============================
    
    Processing video frame-by-frame to get segmentation masks
    """
    
    # LINE EXPLANATION 1:
    # model = YOLO("yolov8m-seg.pt")
    # Load segmentation model
    model = YOLO("yolov8m-seg.pt")
    
    # LINE EXPLANATION 2:
    # video_path = "video.mp4"
    #
    # What it stores:
    # - Path to video file to process
    # - Can be local file or file path
    video_path = "video.mp4"
    
    # LINE EXPLANATION 3:
    # results = model.predict(source=video_path, conf=0.25, imgsz=640)
    #
    # What it does:
    # - Runs segmentation on all video frames
    # - model.predict automatically detects input type (image/video)
    # - For video: processes frame-by-frame automatically
    # - Returns generator of results (one per frame)
    #
    # Process:
    # 1. Opens video file
    # 2. Reads frames sequentially
    # 3. Runs segmentation on each frame
    # 4. Yields result after each frame
    results = model.predict(source=video_path, conf=0.25, imgsz=640)
    
    # LINE EXPLANATION 4:
    # frame_count = 0
    #
    # What it does:
    # - Initializes frame counter
    # - Used to track which frame we're processing
    # - Useful for debugging and progress tracking
    frame_count = 0
    
    # LINE EXPLANATION 5:
    # for result in results:
    #
    # What it does:
    # - Loops through each frame's segmentation result
    # - Results are generated one at a time (memory efficient)
    # - Similar to image loop, but for video frames
    for result in results:
        
        # LINE EXPLANATION 6:
        # frame_count += 1
        #
        # What it does:
        # - Increments frame counter by 1
        # - += means: frame_count = frame_count + 1
        # - Tracks which frame we're on (1, 2, 3, ...)
        frame_count += 1
        
        # LINE EXPLANATION 7:
        # masks = result.masks
        #
        # Get segmentation masks for current frame
        masks = result.masks
        
        # LINE EXPLANATION 8:
        # if masks is not None and frame_count % 10 == 0:
        #
        # What it does:
        # - Checks two conditions:
        #   1. masks is not None: Objects detected in frame
        #   2. frame_count % 10 == 0: Process every 10th frame
        # - % operator is modulo (remainder after division)
        # - frame_count % 10 == 0: Every 10th frame (10, 20, 30, ...)
        #
        # Why every 10th frame?
        # - Video has 30 FPS (30 frames per second)
        # - Processing every frame is slow
        # - Every 10th frame = 3 times per second = manageable
        if masks is not None and frame_count % 10 == 0:
            
            # LINE EXPLANATION 9:
            # annotated_frame = result.plot()
            #
            # Draw masks on current frame
            annotated_frame = result.plot()
            
            # LINE EXPLANATION 10:
            # output_path = f"frame_{frame_count}.jpg"
            #
            # What it does:
            # - Creates filename for saving frame
            # - f"..." is f-string (formatted string)
            # - {frame_count} inserts current frame number
            # - Example: "frame_10.jpg", "frame_20.jpg", etc.
            output_path = f"frame_{frame_count}.jpg"
            
            # LINE EXPLANATION 11:
            # Image.fromarray(annotated_frame).save(output_path)
            #
            # Save annotated frame to disk
            Image.fromarray(annotated_frame).save(output_path)
            
            # LINE EXPLANATION 12:
            # print(f"Processed frame {frame_count}")
            #
            # Print progress message
            print(f"Processed frame {frame_count}")


# ============================================================================
# STEP 7: Extracting Object Contours from Masks
# ============================================================================

def extract_object_contours():
    """
    GETTING PRECISE OBJECT BOUNDARIES FROM MASKS
    =============================================
    
    Using contours for pixel-perfect object boundaries
    """
    
    # LINE EXPLANATION 1:
    # model = YOLO("yolov8m-seg.pt")
    # Load model
    model = YOLO("yolov8m-seg.pt")
    
    # LINE EXPLANATION 2:
    # results = model.predict(r"C:\Users\User\Desktop\YOLO-Project\images.jpg")
    # Run inference
    results = model.predict(r"C:\Users\User\Desktop\YOLO-Project\images.jpg")
    
    # LINE EXPLANATION 3:
    # for result in results:
    # Loop through results
    for result in results:
        
        # LINE EXPLANATION 4:
        # masks = result.masks
        #
        # Get segmentation masks
        masks = result.masks
        
        # LINE EXPLANATION 5:
        # if masks.xy is not None:
        #
        # What it does:
        # - Checks if contours are available
        # - masks.xy contains contours (x, y coordinates of boundaries)
        # - Not all models provide contours, so check first
        if masks.xy is not None:
            
            # LINE EXPLANATION 6:
            # contours = masks.xy
            #
            # What it contains:
            # - List of contours (object boundaries)
            # - Each contour is array of (x, y) coordinates
            # - Contour traces the outline of each detected object
            # - More precise than bounding box
            #
            # Example:
            # contours = [
            #   [[100, 50], [101, 49], [102, 50], ...],  # First object outline
            #   [[200, 100], [201, 99], ...],             # Second object outline
            # ]
            contours = masks.xy
            
            # LINE EXPLANATION 7:
            # original_image = result.orig_img
            #
            # Get original image to draw on
            original_image = result.orig_img
            
            # LINE EXPLANATION 8:
            # contour_image = original_image.copy()
            #
            # Create copy for drawing
            contour_image = original_image.copy()
            
            # LINE EXPLANATION 9:
            # for i, contour in enumerate(contours):
            #
            # What it does:
            # - Loops through each object's contour
            # - i: Index of contour (0, 1, 2, ...)
            # - contour: Array of (x, y) points forming object boundary
            for i, contour in enumerate(contours):
                
                # LINE EXPLANATION 10:
                # contour_int = np.array(contour, dtype=np.int32)
                #
                # What it does:
                # - Converts contour to NumPy array of integers
                # - Contour coordinates might be floats
                # - Image drawing requires integer pixel coordinates
                # - np.int32: 32-bit integer type used by OpenCV
                contour_int = np.array(contour, dtype=np.int32)
                
                # LINE EXPLANATION 11:
                # cv2.polylines(contour_image, [contour_int], True, (0, 255, 0), 2)
                #
                # What it does:
                # - Draws contour line on image
                # - cv2.polylines: Function to draw polygon lines
                # - contour_image: Image to draw on (modified in-place)
                # - [contour_int]: List of contours to draw
                # - True: Close polygon (connect last point to first)
                # - (0, 255, 0): Color in BGR format (Green)
                # - 2: Line thickness in pixels
                #
                # Result: Green outline around each object
                cv2.polylines(contour_image, [contour_int], True, (0, 255, 0), 2)
            
            # LINE EXPLANATION 12:
            # Image.fromarray(contour_image).save("contour_result.jpg")
            #
            # Save image with contours drawn
            Image.fromarray(contour_image).save("contour_result.jpg")
            print("Contour visualization saved!")


# ============================================================================
# STEP 8: Complete Segmentation Example with All Features
# ============================================================================

def complete_segmentation_example():
    """
    COMPLETE WORKFLOW EXAMPLE
    ==========================
    
    Full example combining all segmentation features
    """
    
    # STEP 1: Load model
    print("Step 1: Loading model...")
    model = YOLO("yolov8m-seg.pt")
    
    # STEP 2: Run inference
    print("Step 2: Running segmentation...")
    results = model.predict(
        source=r"C:\Users\User\Desktop\YOLO-Project\images.jpg",
        conf=0.25,
        imgsz=640,
    )
    
    # STEP 3: Process results
    print("Step 3: Processing results...")
    for result in results:
        
        # Get masks and boxes
        masks = result.masks
        boxes = result.boxes
        class_names = result.names
        
        print(f"Objects detected: {len(boxes)}")
        
        # Analyze each detection
        for i, (box, cls_id, conf) in enumerate(
            zip(boxes.xyxy, boxes.cls, boxes.conf)
        ):
            class_name = class_names[int(cls_id)]
            print(f"  {i}: {class_name} ({conf:.1%})")
        
        # Visualize
        annotated_image = result.plot()
        Image.fromarray(annotated_image).save("complete_segmentation.jpg")
        print("\nSegmentation results saved to 'complete_segmentation.jpg'")


# ============================================================================
# QUICK REFERENCE
# ============================================================================

"""
QUICK REFERENCE - KEY COMMANDS
================================

# Load model
model = YOLO("yolov8m-seg.pt")

# Run segmentation
results = model.predict(source="image.jpg", conf=0.25)

# Access masks
masks = result.masks           # Segmentation masks
mask_array = masks.data.cpu().numpy()  # Convert to NumPy

# Access boxes
boxes = result.boxes           # Bounding boxes
boxes.xyxy                     # Coordinates format
boxes.cls                      # Class IDs
boxes.conf                     # Confidence scores

# Visualize
annotated = result.plot()      # Annotated image
Image.fromarray(annotated).save("output.jpg")

# Get contours
contours = masks.xy            # Object boundaries
"""

if __name__ == "__main__":
    # Run examples (uncomment to test)
    # load_segmentation_model()
    # segment_image()
    # analyze_segmentation_results()
    # visualize_segmentation()
    # extract_object_contours()
    complete_segmentation_example()
