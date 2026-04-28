from ultralytics import YOLO
import cv2

# Load the pretrained YOLOv8 nano model (downloads ~6MB automatically)
model = YOLO('yolov8n.pt')

# Run detection on the official sample image
results = model.predict(
    source='https://ultralytics.com/images/bus.jpg',
    save=True,        # saves output image with boxes drawn
    conf=0.5          # only show detections with 50%+ confidence
)

# Print results to the VS Code terminal
print("\n📦 Detection Results:")
print("-" * 40)
for result in results:
    for box in result.boxes:
        cls_id = int(box.cls)
        confidence = float(box.conf)
        label = model.names[cls_id]
        coords = box.xyxy[0].tolist()
        print(f"  ✔ {label:<15} confidence: {confidence:.0%}   bbox: {[round(c) for c in coords]}")

print("-" * 40)
print(f"\n✅ Result image saved to: runs/detect/predict/")