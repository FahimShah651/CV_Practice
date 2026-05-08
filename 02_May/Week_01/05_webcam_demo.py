import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from skimage import exposure
import dlib

# Load classifiers and detectors
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

try:
    hog_detector = dlib.get_frontal_face_detector()
    dlib_available = True
except:
    dlib_available = False
    hog_detector = None

# ... include your detect_faces_haar and run_webcam_demo functions here ...

# Then run:
run_webcam_demo(face_cascade, eye_cascade, hog_detector, None, method='combined')