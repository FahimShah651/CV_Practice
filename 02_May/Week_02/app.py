"""
CBIR Flask Web Application
Content-Based Image Retrieval web interface
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
from pathlib import Path
from cbir import CBIRSystem
import cv2
import numpy as np
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
DATASET_FOLDER = 'static/dataset'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'gif'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16 MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DATASET_FOLDER'] = DATASET_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Initialize CBIR system
cbir_system = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def ensure_directories():
    """Ensure required directories exist"""
    Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(DATASET_FOLDER).mkdir(parents=True, exist_ok=True)

def initialize_cbir():
    """Initialize CBIR system"""
    global cbir_system
    ensure_directories()
    cbir_system = CBIRSystem(index_path='image_index.pkl')
    
    # Try to load existing index
    if not cbir_system.load_index():
        # Build index from dataset
        dataset_count = len(list(Path(DATASET_FOLDER).rglob('*')))
        if dataset_count > 0:
            print("Building CBIR index from dataset...")
            cbir_system.build_index(DATASET_FOLDER)
        else:
            print("Warning: Dataset folder is empty. No index built.")

@app.before_request
def startup():
    """Initialize on first request"""
    global cbir_system
    if cbir_system is None:
        initialize_cbir()

@app.route('/')
def index():
    """Main page"""
    dataset_count = len(list(Path(DATASET_FOLDER).rglob('*')))
    indexed_count = len(cbir_system.image_features) if cbir_system else 0
    
    return render_template('index.html', 
                         dataset_count=dataset_count,
                         indexed_count=indexed_count)

@app.route('/api/search_dataset', methods=['POST'])
def search_dataset():
    """API endpoint for searching using a dataset image"""
    try:
        data = request.get_json()
        query_path = data.get('query_path')
        top_k = data.get('top_k', 10)
        
        if not query_path:
            return jsonify({'error': 'Query path not provided'}), 400
        
        # Construct full path
        filepath = os.path.join(DATASET_FOLDER, query_path.replace('/', os.sep))
        
        if not os.path.exists(filepath):
            return jsonify({'error': f'Query image not found: {filepath}'}), 400
        
        if cbir_system is None or len(cbir_system.image_features) == 0:
            return jsonify({'error': 'CBIR system not initialized. Dataset may be empty.'}), 500
        
        # Perform search
        results = cbir_system.search(filepath, DATASET_FOLDER, top_k=top_k)
        
        # Format results
        formatted_results = []
        for img_path, similarity in results:
            rel_path = os.path.relpath(img_path, DATASET_FOLDER)
            formatted_results.append({
                'path': f"/dataset/{rel_path.replace(os.sep, '/')}",
                'similarity': round(similarity, 2),
                'filename': os.path.basename(img_path)
            })
        
        return jsonify({
            'success': True,
            'query_image': f"/dataset/{query_path}",
            'results': formatted_results,
            'total_results': len(formatted_results)
        })
        
    except Exception as e:
        print(f"Search dataset error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/search', methods=['POST'])
def search():
    """API endpoint for image search"""
    try:
        # Check if this is a dataset image or uploaded file
        query_path = request.form.get('query_path', None)
        
        if query_path:
            # Direct dataset search
            filepath = os.path.join(DATASET_FOLDER, query_path)
            if not os.path.exists(filepath):
                return jsonify({'error': 'Query image not found in dataset'}), 400
            filename = os.path.basename(filepath)
        else:
            # File upload search
            if 'file' not in request.files:
                return jsonify({'error': 'No file provided'}), 400
            
            file = request.files['file']
            
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if not allowed_file(file.filename):
                return jsonify({'error': 'File type not allowed'}), 400
            
            # Save uploaded file
            filename = secure_filename(f"{datetime.now().timestamp()}_{file.filename}")
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
        
        if cbir_system is None or len(cbir_system.image_features) == 0:
            return jsonify({'error': 'CBIR system not initialized. Dataset may be empty.'}), 500
        
        # Get number of results
        top_k = request.form.get('top_k', 10, type=int)
        top_k = min(max(top_k, 1), 50)  # Between 1 and 50
        
        # Perform search
        results = cbir_system.search(filepath, DATASET_FOLDER, top_k=top_k)
        
        # Format results
        formatted_results = []
        for img_path, similarity in results:
            rel_path = os.path.relpath(img_path, DATASET_FOLDER)
            formatted_results.append({
                'path': f"/dataset/{rel_path.replace(os.sep, '/')}",
                'similarity': round(similarity, 2),
                'filename': os.path.basename(img_path)
            })
        
        query_image_path = filepath if query_path else f"/uploads/{filename}"
        
        return jsonify({
            'success': True,
            'query_image': query_image_path,
            'results': formatted_results,
            'total_results': len(formatted_results)
        })
        
    except Exception as e:
        print(f"Search error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def stats():
    """Get system statistics"""
    try:
        dataset_count = len(list(Path(DATASET_FOLDER).rglob('*')))
        indexed_count = len(cbir_system.image_features) if cbir_system else 0
        
        return jsonify({
            'dataset_images': dataset_count,
            'indexed_images': indexed_count,
            'system_ready': indexed_count > 0
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/rebuild_index', methods=['POST'])
def rebuild_index():
    """Rebuild the CBIR index"""
    try:
        global cbir_system
        if cbir_system is None:
            initialize_cbir()
        
        cbir_system.image_features.clear()
        cbir_system.image_descriptors.clear()
        
        dataset_count = len(list(Path(DATASET_FOLDER).rglob('*')))
        if dataset_count == 0:
            return jsonify({'error': 'No images in dataset folder'}), 400
        
        cbir_system.build_index(DATASET_FOLDER)
        
        return jsonify({
            'success': True,
            'indexed_images': len(cbir_system.image_features)
        })
    except Exception as e:
        print(f"Rebuild error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/dataset/<path:filename>')
def serve_dataset(filename):
    """Serve dataset images"""
    return send_from_directory(DATASET_FOLDER, filename)

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    """Serve uploaded images"""
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/api/sample_images')
def sample_images():
    """Get sample images from dataset for browsing"""
    try:
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        images = [f for f in Path(DATASET_FOLDER).rglob('*') 
                 if f.suffix.lower() in image_extensions]
        
        # Return up to 20 random samples
        np.random.seed(42)
        sample_indices = np.random.choice(len(images), min(20, len(images)), replace=False)
        samples = [images[i] for i in sample_indices]
        
        sample_data = []
        for img_path in samples:
            rel_path = os.path.relpath(img_path, DATASET_FOLDER)
            sample_data.append({
                'path': f"/dataset/{rel_path}",
                'filename': img_path.name
            })
        
        return jsonify({'samples': sample_data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Server error'}), 500

if __name__ == '__main__':
    ensure_directories()
    print("Starting CBIR Flask Web App...")
    print(f"Dataset folder: {os.path.abspath(DATASET_FOLDER)}")
    print(f"Upload folder: {os.path.abspath(UPLOAD_FOLDER)}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
