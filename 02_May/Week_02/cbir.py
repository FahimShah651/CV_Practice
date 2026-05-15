"""
Content-Based Image Retrieval (CBIR) System
Implements image feature extraction and similarity matching
"""

import cv2
import numpy as np
from sklearn.preprocessing import normalize
from pathlib import Path
import pickle
import os


class CBIRSystem:
    """Content-Based Image Retrieval using ORB features and histograms"""
    
    def __init__(self, index_path='image_index.pkl'):
        self.orb = cv2.ORB_create(nfeatures=500)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.index_path = index_path
        self.image_features = {}  # Maps image path to features
        self.image_descriptors = {}  # Maps image path to descriptors
        
    def extract_features(self, image_path):
        """
        Extract multiple features from an image:
        - ORB keypoints and descriptors
        - Color histogram
        - Edge histogram
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return None
                
            # Convert to grayscale for ORB
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # ORB features
            kp, des = self.orb.detectAndCompute(gray, None)
            
            # Color histogram (HSV)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hist_h = cv2.calcHist([hsv], [0], None, [50], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [50], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [50], [0, 256])
            hist_combined = np.concatenate([hist_h, hist_s, hist_v]).flatten()
            hist_combined = normalize(hist_combined.reshape(1, -1), norm='l2')[0]
            
            # Edge histogram using Sobel
            edges = cv2.Canny(gray, 100, 200)
            edge_hist = np.histogram(edges, bins=10, range=(0, 256))[0]
            edge_hist = normalize(edge_hist.reshape(1, -1), norm='l2')[0]
            
            features = {
                'descriptors': des if des is not None else np.array([]),
                'histogram': hist_combined,
                'edges': edge_hist,
                'image_shape': img.shape
            }
            
            return features
            
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None
    
    def build_index(self, dataset_dir):
        """Build image index from dataset directory"""
        dataset_path = Path(dataset_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        image_files = [f for f in dataset_path.rglob('*') 
                      if f.suffix.lower() in image_extensions]
        
        print(f"Found {len(image_files)} images. Extracting features...")
        
        for idx, img_path in enumerate(image_files):
            if idx % 50 == 0:
                print(f"Processing image {idx}/{len(image_files)}")
            
            features = self.extract_features(img_path)
            if features is not None:
                rel_path = str(img_path.relative_to(dataset_path))
                self.image_features[rel_path] = features
                self.image_descriptors[rel_path] = features['descriptors']
        
        print(f"Indexed {len(self.image_features)} images")
        self.save_index()
    
    def save_index(self):
        """Save index to disk"""
        with open(self.index_path, 'wb') as f:
            pickle.dump({
                'features': self.image_features,
                'descriptors': self.image_descriptors
            }, f)
        print(f"Index saved to {self.index_path}")
    
    def load_index(self):
        """Load index from disk"""
        if os.path.exists(self.index_path):
            with open(self.index_path, 'rb') as f:
                data = pickle.load(f)
                self.image_features = data['features']
                self.image_descriptors = data['descriptors']
            print(f"Loaded index with {len(self.image_features)} images")
            return True
        return False
    
    def histogram_similarity(self, hist1, hist2, method='chi_square'):
        """Compare histograms using different methods"""
        if method == 'chi_square':
            # Chi-square distance
            distance = 0
            for h1, h2 in zip(hist1, hist2):
                if h1 + h2 > 0:
                    distance += (h1 - h2) ** 2 / (h1 + h2)
            return distance
        elif method == 'euclidean':
            return np.linalg.norm(hist1 - hist2)
        elif method == 'cosine':
            return 1 - np.dot(hist1, hist2) / (np.linalg.norm(hist1) * np.linalg.norm(hist2) + 1e-6)
    
    def descriptor_similarity(self, des1, des2):
        """Compare descriptors using feature matching"""
        if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
            return 0
        
        try:
            matches = self.bf_matcher.knnMatch(des1, des2, k=2)
            
            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            # Return normalized match count
            total_possible = max(len(des1), len(des2))
            return len(good_matches) / (total_possible + 1e-6)
        except:
            return 0
    
    def search(self, query_image_path, dataset_dir, top_k=10):
        """
        Search for similar images
        
        Args:
            query_image_path: Path to query image
            dataset_dir: Path to dataset directory
            top_k: Number of results to return
        
        Returns:
            List of (image_path, similarity_score) tuples
        """
        # Extract query features
        query_features = self.extract_features(query_image_path)
        if query_features is None:
            return []
        
        # Compute similarity scores
        scores = {}
        dataset_path = Path(dataset_dir)
        
        for img_rel_path, features in self.image_features.items():
            # Skip query image itself
            full_path = dataset_path / img_rel_path
            if str(full_path) == str(query_image_path):
                continue
            
            # Histogram similarity (weighted 60%)
            hist_score = -self.histogram_similarity(
                query_features['histogram'], 
                features['histogram']
            )  # Negative because smaller distance is better
            
            # Descriptor similarity (weighted 30%)
            desc_score = self.descriptor_similarity(
                query_features['descriptors'],
                features['descriptors']
            )
            
            # Edge similarity (weighted 10%)
            edge_score = -self.histogram_similarity(
                query_features['edges'],
                features['edges']
            )
            
            # Combined score
            combined_score = (0.6 * hist_score + 0.3 * desc_score + 0.1 * edge_score)
            scores[img_rel_path] = combined_score
        
        # Sort and return top-k results
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        
        for img_rel_path, score in sorted_results[:top_k]:
            full_path = dataset_path / img_rel_path
            # Normalize score to 0-100 range
            normalized_score = max(0, min(100, (score + 10) * 5))  # Rough normalization
            results.append((str(full_path), normalized_score))
        
        return results
