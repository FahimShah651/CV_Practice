# Quick Reference: Processing Your Own Videos

## 🎬 How to Process Your Own Video Files

This guide shows you how to use the Background Subtraction notebook to process any video you have.

## ✅ Before You Start

1. Your video file should be in a common format: `.mp4`, `.avi`, `.mov`, `.mkv`, `.flv`, or `.wmv`
2. Keep note of the video's file path
3. Have the notebook open and all previous cells executed

## 📝 Basic Usage

### Option 1: Simple Processing (Recommended)

```python
results = process_custom_video('path/to/your/video.mp4', 
                              background_subtractor='mog2')
```

This will:
- ✓ Process the video with MOG2 algorithm
- ✓ Save individual frames to `output_videos/frames/`
- ✓ Save processed video to `output_videos/processed_videos/`
- ✓ Display statistics about detections

### Option 2: Using KNN Instead

```python
results = process_custom_video('path/to/your/video.mp4', 
                              background_subtractor='knn')
```

KNN works better for:
- Videos with dynamic/changing backgrounds
- Scenes with reflections
- Complex lighting conditions

### Option 3: Save More Frames

```python
results = process_custom_video('path/to/your/video.mp4', 
                              background_subtractor='mog2',
                              frame_interval=1)  # Save every frame
```

Frame intervals:
- `1` = Save every frame (uses more disk space)
- `5` = Save every 5th frame (default)
- `10` = Save every 10th frame (faster processing)
- `30` = Save every 30th frame (for very long videos)

## 📂 File Path Examples

### Windows Paths

**Absolute path:**
```python
results = process_custom_video(
    'C:\\Users\\YourName\\Videos\\surveillance.mp4',
    background_subtractor='mog2'
)
```

**Relative path (from current directory):**
```python
results = process_custom_video(
    'videos\\traffic.mp4',
    background_subtractor='knn'
)
```

**From Downloads folder:**
```python
results = process_custom_video(
    'C:\\Users\\YourName\\Downloads\\video.mp4',
    background_subtractor='mog2'
)
```

## 🔍 Finding Your Video Path

### Method 1: Using File Explorer
1. Right-click on your video file
2. Select "Properties" or "Copy path"
3. Paste the path into the code

### Method 2: Using the Processor Tool
```python
# List available videos
processor.list_videos()

# Get info about a specific video
info = processor.get_video_info('C:\\Users\\YourName\\Videos\\traffic.mp4')
print(info)
```

## 📊 Understanding the Results

After processing, you'll get a `results` dictionary:

```python
results = process_custom_video('video.mp4')

# Access information about processing:
print(results['total_frames'])           # Total frames in video
print(results['fps'])                    # Frames per second
print(results['resolution'])             # (width, height)
print(results['total_detections'])       # Total objects detected
print(results['avg_objects_per_frame'])  # Average objects per frame
print(results['frames_directory'])       # Where frames are saved
print(results['output_video_path'])      # Where processed video is saved
```

## 🎯 Complete Examples

### Example 1: Process Traffic Monitoring Video

```python
# Process surveillance footage
results = process_custom_video(
    'C:\\Users\\YourName\\Videos\\traffic_cam.mp4',
    background_subtractor='mog2',
    save_frames=True,
    frame_interval=5
)

# View results
print(f"Total vehicles detected: {results['total_detections']}")
print(f"Average per frame: {results['avg_objects_per_frame']:.1f}")
print(f"Processed video saved at: {results['output_video_path']}")
```

### Example 2: Compare Two Methods on Same Video

```python
# Process with MOG2
results_mog2 = process_custom_video(
    'video.mp4',
    background_subtractor='mog2'
)

# Process with KNN
results_knn = process_custom_video(
    'video.mp4',
    background_subtractor='knn'
)

# Compare
print(f"MOG2 avg objects/frame: {results_mog2['avg_objects_per_frame']:.2f}")
print(f"KNN avg objects/frame: {results_knn['avg_objects_per_frame']:.2f}")
```

### Example 3: Create Comparison Video

```python
# Generate comparison of all three methods
create_comparison_video(
    'video.mp4',
    'comparison_output.mp4',
    methods=['mog2', 'knn', 'framediff']
)
```

## 📹 Output Files Explained

After processing `traffic.mp4` with MOG2, you'll get:

```
output_videos/
├── frames/
│   └── traffic_mog2/
│       ├── frame_0000.jpg    ← Frame at start
│       ├── frame_0005.jpg    ← Frame 5 (0.2 sec at 30fps)
│       ├── frame_0010.jpg    ← Frame 10 (0.3 sec)
│       └── ... (every 5th frame)
│
└── processed_videos/
    └── traffic_mog2_processed.mp4  ← Full processed video
```

### What Each File Contains

**Frame files (JPG):**
- Left half: Original video + detection boxes
- Right half: Foreground mask (what algorithm detected)

**Processed video (MP4):**
- Complete video with detections
- All frames included
- Playable in any video player

## ⚙️ Advanced Options

### Changing Detection Sensitivity

```python
# For sensitive detection (many false positives):
results = process_and_save_video(
    'video.mp4',
    background_subtractor='mog2',
    min_area=50  # Smaller threshold
)

# For less sensitive detection (may miss small objects):
results = process_and_save_video(
    'video.mp4',
    background_subtractor='mog2',
    min_area=500  # Larger threshold
)
```

### Customize the Detector

```python
# Create custom detector with specific settings
detector = SurveillanceObjectDetector(
    background_subtractor='mog2',
    min_area=100
)

# Use it on your frames
cap = cv2.VideoCapture('video.mp4')
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    detections, mask = detector.detect_objects(frame)
    annotated = detector.draw_detections(frame, detections)
    
    # ... do something with annotated frame

cap.release()
```

## 🐛 Troubleshooting

### "File not found" Error
```
ERROR: Video file not found: C:\path\to\video.mp4
```
**Solution:** Check the file path is correct. Use absolute paths if unsure.

### "Cannot open video" Error
```
ERROR: Cannot open video: video.mp4
```
**Solution:** Video format might not be supported. Try converting to MP4 first.

### Takes Too Long to Process
**Solution:** Use `frame_interval=10` or higher to skip frames:
```python
results = process_custom_video('video.mp4', frame_interval=10)
```

### Disk Space Issues
**Solution:** Reduce number of saved frames:
```python
# Save only every 20th frame
results = process_custom_video('video.mp4', frame_interval=20)

# Or disable frame saving
results = process_and_save_video(
    'video.mp4',
    save_frames=False  # Only save processed video
)
```

## 💡 Tips & Best Practices

1. **For Real-time Scenarios:** Use MOG2 with `detectShadows=True`
2. **For Outdoor Videos:** Use KNN to handle changing lighting
3. **For Large Videos:** Process with `frame_interval=20` or higher
4. **For Accuracy:** Compare results with both MOG2 and KNN
5. **For Production:** Test parameters on sample frames first

## 📞 Getting Video Info Before Processing

```python
# Check video details before processing
info = processor.get_video_info('video.mp4')

print(f"Filename: {info['filename']}")
print(f"Size: {info['size_mb']:.2f} MB")
print(f"Resolution: {info['width']}x{info['height']}")
print(f"FPS: {info['fps']:.2f}")
print(f"Duration: {info['duration_sec']:.2f} seconds")
print(f"Total frames: {info['total_frames']}")

# Estimate processing time
estimated_time = info['total_frames'] / 30  # Rough estimate
print(f"Estimated processing time: {estimated_time:.1f} seconds")
```

## ✨ Next Steps

After processing your video:

1. **Analyze the outputs:**
   - View frames to see what was detected
   - Play the video to see results in motion

2. **Adjust parameters if needed:**
   - If too many false positives: increase `min_area`
   - If missing objects: decrease `min_area`
   - If too noisy: switch to KNN

3. **Combine with tracking:**
   - Use `CentroidTracker` for object counting
   - Track objects across frames
   - Count crossing events

4. **Integrate with your application:**
   - Extract detection coordinates
   - Feed into downstream processing
   - Create alerts or reports

---

**Happy video processing! 🎥**

For more details, see:
- Main notebook: Background Subtraction.ipynb
- Output README: output_videos/README.md
