/**
 * CBIR Flask Web App - Frontend JavaScript
 */

// Initialize upload area
document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    
    // Click to upload
    uploadArea.addEventListener('click', () => fileInput.click());
    
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.backgroundColor = '#e8f1ff';
        uploadArea.style.borderColor = '#2d5aa0';
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.backgroundColor = '#f0f7ff';
        uploadArea.style.borderColor = '#4a90e2';
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.backgroundColor = '#f0f7ff';
        uploadArea.style.borderColor = '#4a90e2';
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });
    
    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });
    
    // Load sample images
    loadSampleImages();
    
    // Load stats
    updateStats();
});

/**
 * Handle file upload and search
 */
function handleFileUpload(file) {
    // Validate file type
    const allowedTypes = ['image/jpeg', 'image/png', 'image/bmp', 'image/gif', 'image/tiff'];
    if (!allowedTypes.includes(file.type)) {
        showError('Invalid file type. Please upload an image file.');
        return;
    }
    
    // Validate file size (16MB)
    if (file.size > 16 * 1024 * 1024) {
        showError('File too large. Maximum file size is 16MB.');
        return;
    }
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('query-preview').src = e.target.result;
        document.getElementById('query-preview-container').classList.remove('hidden');
        
        // Perform search
        performSearch(file);
    };
    reader.readAsDataURL(file);
}

/**
 * Perform image search
 */
function performSearch(file) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('top_k', document.getElementById('top-k').value);
    
    // Show loading
    document.getElementById('loading-spinner').classList.remove('hidden');
    document.getElementById('results-section').classList.remove('hidden');
    
    // Scroll to results
    setTimeout(() => {
        document.getElementById('results-section').scrollIntoView({ behavior: 'smooth' });
    }, 100);
    
    // Send request
    fetch('/api/search', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('loading-spinner').classList.add('hidden');
        
        if (data.error) {
            showError('Search failed: ' + data.error);
            document.getElementById('results-section').classList.add('hidden');
        } else {
            displayResults(data);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('loading-spinner').classList.add('hidden');
        showError('An error occurred during search: ' + error.message);
        document.getElementById('results-section').classList.add('hidden');
    });
}

/**
 * Display search results
 */
function displayResults(data) {
    const resultsContainer = document.getElementById('results-container');
    resultsContainer.innerHTML = '';
    
    const resultCount = document.getElementById('result-count');
    resultCount.textContent = `Found ${data.total_results} similar images`;
    
    if (data.results.length === 0) {
        resultsContainer.innerHTML = '<p style="text-align: center; grid-column: 1/-1;">No similar images found in the dataset.</p>';
        return;
    }
    
    data.results.forEach((result, index) => {
        const card = document.createElement('div');
        card.className = 'result-card';
        card.innerHTML = `
            <img src="${result.path}" alt="Result ${index + 1}" class="card-image" onerror="this.src='data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 width=%22200%22 height=%22200%22%3E%3Crect fill=%22%23f0f0f0%22 width=%22200%22 height=%22200%22/%3E%3Ctext x=%2250%25%22 y=%2250%25%22 text-anchor=%22middle%22 dy=%22.3em%22 font-family=%22Arial%22%3EImage Not Found%3C/text%3E%3C/svg%3E'">
            <div class="card-content">
                <div class="card-title" title="${result.filename}">${result.filename}</div>
                <div class="card-similarity">
                    <span class="similarity-label">Similarity:</span>
                    <span class="similarity-score">${result.similarity}%</span>
                </div>
                <button class="btn-view" onclick="viewFullImage('${result.path}')">View Full Image</button>
            </div>
        `;
        resultsContainer.appendChild(card);
    });
}

/**
 * Load and display sample images
 */
function loadSampleImages() {
    fetch('/api/sample_images')
    .then(response => response.json())
    .then(data => {
        const samplesContainer = document.getElementById('samples-container');
        samplesContainer.innerHTML = '';
        
        if (data.samples.length === 0) {
            samplesContainer.innerHTML = '<p style="text-align: center; grid-column: 1/-1;">No sample images available. Please add images to the dataset folder.</p>';
            return;
        }
        
        data.samples.forEach((sample, index) => {
            const card = document.createElement('div');
            card.className = 'sample-card';
            card.innerHTML = `
                <img src="${sample.path}" alt="Sample ${index + 1}" class="card-image" onerror="this.src='data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 width=%22200%22 height=%22200%22%3E%3Crect fill=%22%23f0f0f0%22 width=%22200%22 height=%22200%22/%3E%3Ctext x=%2250%25%22 y=%2250%25%22 text-anchor=%22middle%22 dy=%22.3em%22 font-family=%22Arial%22%3EImage Not Found%3C/text%3E%3C/svg%3E'">
                <div class="card-content">
                    <div class="card-title" title="${sample.filename}">${sample.filename}</div>
                    <button class="btn-view" onclick="useAsQuery('${sample.path}')">Use as Query</button>
                </div>
            `;
            samplesContainer.appendChild(card);
        });
    })
    .catch(error => {
        console.error('Error loading samples:', error);
        document.getElementById('samples-container').innerHTML = '<p style="text-align: center; grid-column: 1/-1;">Error loading sample images.</p>';
    });
}

/**
 * Use a sample image as query - directly search from dataset
 */
function useAsQuery(imagePath) {
    // Extract dataset path from image URL (e.g., "/dataset/natural/img_0061.jpg" -> "natural/img_0061.jpg")
    const datasetPath = imagePath.replace(/^\/dataset\//, '');
    
    // Show preview
    document.getElementById('query-preview').src = imagePath;
    document.getElementById('query-preview-container').classList.remove('hidden');
    
    // Show loading
    document.getElementById('loading-spinner').classList.remove('hidden');
    document.getElementById('results-section').classList.remove('hidden');
    
    // Scroll to results
    setTimeout(() => {
        document.getElementById('results-section').scrollIntoView({ behavior: 'smooth' });
    }, 100);
    
    // Send request to dedicated search_dataset endpoint
    fetch('/api/search_dataset', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            query_path: datasetPath,
            top_k: parseInt(document.getElementById('top-k').value)
        })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('loading-spinner').classList.add('hidden');
        
        if (data.error) {
            showError('Search failed: ' + data.error);
            document.getElementById('results-section').classList.add('hidden');
        } else {
            displayResults(data);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('loading-spinner').classList.add('hidden');
        showError('An error occurred during search: ' + error.message);
        document.getElementById('results-section').classList.add('hidden');
    });
}

/**
 * View full image in modal
 */
function viewFullImage(imagePath) {
    const modal = document.createElement('div');
    modal.className = 'modal';
    modal.style.display = 'block';
    modal.innerHTML = `
        <div class="modal-content" style="text-align: center;">
            <span class="close" onclick="this.parentElement.parentElement.remove()">&times;</span>
            <img src="${imagePath}" style="max-width: 90%; max-height: 80vh; border-radius: 8px; margin-top: 20px;" onerror="this.src='data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 width=%22400%22 height=%22300%22%3E%3Crect fill=%22%23f0f0f0%22 width=%22400%22 height=%22300%22/%3E%3Ctext x=%2250%25%22 y=%2250%25%22 text-anchor=%22middle%22 dy=%22.3em%22 font-family=%22Arial%22%3EImage Not Found%3C/text%3E%3C/svg%3E'">
        </div>
    `;
    document.body.appendChild(modal);
    modal.onclick = (e) => {
        if (e.target === modal) modal.remove();
    };
}

/**
 * Clear query and search again
 */
function clearQuery() {
    document.getElementById('file-input').value = '';
    document.getElementById('query-preview-container').classList.add('hidden');
    document.getElementById('results-section').classList.add('hidden');
    document.getElementById('results-container').innerHTML = '';
    document.getElementById('upload-area').style.backgroundColor = '#f0f7ff';
    document.getElementById('upload-area').style.borderColor = '#4a90e2';
}

/**
 * Rebuild the image index
 */
function rebuildIndex() {
    if (!confirm('This will rebuild the image index. This may take a few minutes. Continue?')) {
        return;
    }
    
    showInfo('Rebuilding index...');
    
    fetch('/api/rebuild_index', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            showError('Error: ' + data.error);
        } else {
            closeInfoModal();
            showInfo(`Index rebuilt successfully! Indexed ${data.indexed_images} images.`);
            updateStats();
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showError('Error rebuilding index: ' + error.message);
    });
}

/**
 * Update statistics
 */
function updateStats() {
    fetch('/api/stats')
    .then(response => response.json())
    .then(data => {
        document.getElementById('dataset-count').textContent = data.dataset_images;
        document.getElementById('indexed-count').textContent = data.indexed_images;
        
        if (!data.system_ready && data.indexed_images === 0) {
            showWarning('Dataset is empty. Please add images to the static/dataset folder and rebuild the index.');
        }
    })
    .catch(error => console.error('Error loading stats:', error));
}

/**
 * Show error modal
 */
function showError(message) {
    document.getElementById('error-message').textContent = message;
    document.getElementById('error-modal').classList.remove('hidden');
}

/**
 * Close error modal
 */
function closeErrorModal() {
    document.getElementById('error-modal').classList.add('hidden');
}

/**
 * Show info modal
 */
function showInfo(message) {
    document.getElementById('info-message').textContent = message;
    document.getElementById('info-modal').classList.remove('hidden');
}

/**
 * Close info modal
 */
function closeInfoModal() {
    document.getElementById('info-modal').classList.add('hidden');
}

/**
 * Show warning (info modal)
 */
function showWarning(message) {
    document.getElementById('info-message').textContent = message;
    document.getElementById('info-modal').classList.remove('hidden');
}

/**
 * Auto-refresh stats periodically
 */
setInterval(updateStats, 30000);  // Every 30 seconds
