"""
YOLOV8 NANO CLASSIFICATION - QUICK START EXAMPLE
================================================
Using the smallest and fastest YOLOv8 classification model
"""

from ultralytics import YOLO
from PIL import Image

# ============================================================================
# SIMPLE WORKING EXAMPLE
# ============================================================================

def quick_classification():
    """
    Quick example using YOLOv8 Nano Classification model
    This is the fastest model, suitable for real-time applications
    """
    
    # Step 1: Load the nano classification model
    # yolov8n-cls.pt = nano model (smallest, fastest)
    # First time: automatically downloads ~5MB
    print("Loading YOLOv8 Nano Classification model...")
    model = YOLO("yolov8n-cls.pt")
    print("✓ Model loaded successfully!")
    
    # Step 2: Run classification on an image
    # Using an example image from Ultralytics
    print("\nRunning classification...")
    results = model.predict(
        source="https://ultralytics.com/images/bus.jpg",
        conf=0.0,  # Include all predictions
        imgsz=640,  # Input image size
    )
    
    # Step 3: Extract and display results
    print("\nClassification Results:")
    print("=" * 50)
    
    for result in results:
        # Get prediction probabilities
        probs = result.probs
        class_names = result.names
        
        # Get top prediction
        top1_class_id = int(probs.top1)
        top1_class_name = class_names[top1_class_id]
        top1_confidence = float(probs.top1conf)
        
        print(f"Predicted Class: {top1_class_name}")
        print(f"Confidence: {top1_confidence:.1%}")
        
        # Display top 5 predictions
        print("\nTop 5 Predictions:")
        print("-" * 50)
        
        top5_ids = probs.top5.tolist()
        top5_confs = probs.top5conf.tolist()
        
        for rank, (class_id, confidence) in enumerate(zip(top5_ids, top5_confs), 1):
            class_name = class_names[class_id]
            print(f"{rank}. {class_name:30s} {confidence:.1%}")
        
        # Save annotated result
        annotated_image = result.plot()
        Image.fromarray(annotated_image).save("classification_result.jpg")
        print("\n✓ Visualization saved: classification_result.jpg")


# ============================================================================
# CLASSIFY LOCAL IMAGE
# ============================================================================

def classify_local_image(image_path):
    """
    Classify a local image file
    
    Args:
        image_path: Path to your image file
        Example: "C:\\Users\\User\\Desktop\\photo.jpg"
    """
    
    print(f"Classifying: {image_path}")
    
    model = YOLO("yolov8n-cls.pt")
    results = model.predict(source=image_path, conf=0.0)
    
    for result in results:
        probs = result.probs
        class_names = result.names
        
        top_class = class_names[int(probs.top1)]
        confidence = float(probs.top1conf)
        
        print(f"Result: {top_class} ({confidence:.1%})")
        
        return top_class, confidence


# ============================================================================
# BATCH CLASSIFY MULTIPLE IMAGES
# ============================================================================

def classify_folder(folder_path):
    """
    Classify all images in a folder
    
    Args:
        folder_path: Path to folder containing images
        Example: "C:\\Users\\User\\Desktop\\photos"
    """
    
    print(f"Classifying images in: {folder_path}")
    
    model = YOLO("yolov8n-cls.pt")
    results = model.predict(source=folder_path, conf=0.0, batch=32)
    
    print("\nResults:")
    print("-" * 60)
    
    for result in results:
        probs = result.probs
        class_names = result.names
        
        top_class = class_names[int(probs.top1)]
        confidence = float(probs.top1conf)
        
        filename = result.path.split("\\")[-1]
        print(f"{filename:30s} → {top_class:20s} ({confidence:.1%})")
    
    print("-" * 60)
    print(f"✓ Classified {len(list(results))} images")


# ============================================================================
# REAL-TIME WEBCAM CLASSIFICATION
# ============================================================================

def webcam_classification():
    """
    Real-time classification from webcam
    Press 'q' to exit
    """
    
    import cv2
    
    print("Starting webcam classification...")
    print("Press 'q' to exit")
    
    model = YOLO("yolov8n-cls.pt")
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Classify current frame
        results = model.predict(source=frame, conf=0.0, verbose=False)
        
        for result in results:
            probs = result.probs
            class_names = result.names
            
            top_class = class_names[int(probs.top1)]
            confidence = float(probs.top1conf)
            
            # Draw on frame
            label = f"{top_class} ({confidence:.0%})"
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)
        
        cv2.imshow("YOLOv8 Nano Classification", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✓ Webcam classification closed")


# ============================================================================
# RUN EXAMPLES
# ============================================================================

if __name__ == "__main__":
    
    # Example 1: Quick classification
    print("\n" + "="*60)
    print("EXAMPLE 1: Quick Classification")
    print("="*60)
    quick_classification()
    
    # Example 2: Classify local image
    # Uncomment and change path to your image
    # print("\n" + "="*60)
    # print("EXAMPLE 2: Local Image Classification")
    # print("="*60)
    # classify_local_image("C:\\path\\to\\your\\image.jpg")
    
    # Example 3: Classify folder
    # Uncomment and change path to your folder
    # print("\n" + "="*60)
    # print("EXAMPLE 3: Batch Classification")
    # print("="*60)
    # classify_folder("C:\\path\\to\\your\\folder")
    
    # Example 4: Webcam classification
    # Uncomment to run
    # print("\n" + "="*60)
    # print("EXAMPLE 4: Webcam Classification")
    # print("="*60)
    # webcam_classification()
