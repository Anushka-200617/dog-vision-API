// DOM Elements
const uploadBox = document.getElementById('uploadBox');
const fileInput = document.getElementById('fileInput');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const changeImageBtn = document.getElementById('changeImageBtn');
const loading = document.getElementById('loading');
const resultsSection = document.getElementById('resultsSection');
const topPrediction = document.getElementById('topPrediction');
const otherPredictions = document.getElementById('otherPredictions');
const analyzeAnotherBtn = document.getElementById('analyzeAnotherBtn');
const errorMessage = document.getElementById('errorMessage');
const errorText = document.getElementById('errorText');

let selectedFile = null;

// Event Listeners
fileInput.addEventListener('change', handleFileSelect);
changeImageBtn.addEventListener('change', handleFileSelect);
analyzeAnotherBtn.addEventListener('click', resetApp);

// Drag and Drop
uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.classList.add('dragover');
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.classList.remove('dragover');
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

uploadBox.addEventListener('click', () => {
    fileInput.click();
});

// Handle File Selection
function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

// Handle File
function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showError('Please select an image file (JPG, PNG, or JPEG)');
        return;
    }
    
    // Validate file size (10MB)
    if (file.size > 10 * 1024 * 1024) {
        showError('File size must be less than 10MB');
        return;
    }
    
    selectedFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadBox.classList.add('hidden');
        previewSection.classList.remove('hidden');
        
        // Auto-analyze after preview
        setTimeout(() => {
            analyzeImage();
        }, 500);
    };
    reader.readAsDataURL(file);
}

// Analyze Image
async function analyzeImage() {
    if (!selectedFile) return;
    
    // Hide preview, show loading
    previewSection.classList.add('hidden');
    loading.classList.remove('hidden');
    errorMessage.classList.add('hidden');
    resultsSection.classList.add('hidden');
    
    // Create FormData
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    try {
        // Call API
        const response = await fetch('/predict?top_k=5', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Prediction failed');
        }
        
        const data = await response.json();
        
        // Show results
        displayResults(data);
        
    } catch (error) {
        console.error('Error:', error);
        showError('Failed to analyze image. Please try again.');
        previewSection.classList.remove('hidden');
    } finally {
        loading.classList.add('hidden');
    }
}

// Display Results
function displayResults(data) {
    const predictions = data.predictions;
    
    // Top prediction
    const top = predictions[0];
    topPrediction.innerHTML = `
        <h3>🏆 Top Prediction</h3>
        <div class="confidence">${(top.confidence * 100).toFixed(1)}%</div>
        <h2 style="margin-top: 10px; font-size: 36px;">${top.breed}</h2>
    `;
    
    // Other predictions
    otherPredictions.innerHTML = '<h4 style="margin-bottom: 15px; color: #666;">Other Possibilities:</h4>';
    predictions.slice(1).forEach((pred, index) => {
        const item = document.createElement('div');
        item.className = 'prediction-item';
        item.style.animationDelay = `${index * 0.1}s`;
        
        item.innerHTML = `
            <div>
                <div class="prediction-breed">${pred.breed}</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: 0%;" data-width="${pred.confidence * 100}%"></div>
                </div>
            </div>
            <div class="prediction-confidence">${pred.confidence_percentage}</div>
        `;
        
        otherPredictions.appendChild(item);
    });
    
    // Animate confidence bars
    setTimeout(() => {
        document.querySelectorAll('.confidence-fill').forEach(bar => {
            bar.style.width = bar.getAttribute('data-width');
        });
    }, 100);
    
    // Show results
    resultsSection.classList.remove('hidden');
}

// Show Error
function showError(message) {
    errorText.textContent = message;
    errorMessage.classList.remove('hidden');
    
    setTimeout(() => {
        errorMessage.classList.add('hidden');
    }, 5000);
}

// Reset App
function resetApp() {
    selectedFile = null;
    fileInput.value = '';
    uploadBox.classList.remove('hidden');
    previewSection.classList.add('hidden');
    resultsSection.classList.add('hidden');
    errorMessage.classList.add('hidden');
}
