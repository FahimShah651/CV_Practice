# Vehicle Counter + Speed Estimator

Real-time vehicle counter and speed estimator using background subtraction, Kalman tracking, and perspective calibration.

## What It Does
- Counts vehicles crossing a virtual line from a fixed camera
- Estimates speed of each vehicle in km/h
- Reports traffic flow in vehicles per minute
- Saves speed log to CSV

## How It Works
1. Background subtraction (MOG2) detects moving vehicles
2. Contour detection finds the centroid of each vehicle
3. Kalman filter assigns stable IDs and smooths positions
4. Perspective transform converts pixels to metres for speed calculation
5. Counting line detects each vehicle exactly once

## How to Run
```bash
pip install -r requirements.txt
python main.py --source road.mp4 --line-y 300 --mpp 0.05
```

## Results
| Metric | Value |
|--------|-------|
| FPS (720p) | ~28 |
| Count accuracy | >95% |
| Speed error | <5 km/h |

## What I Learned
I learned that pixel distance alone is meaningless for speed — you need to calibrate the scene using known real-world measurements. The Kalman filter velocity state gives a much smoother speed estimate than frame-to-frame distance.
