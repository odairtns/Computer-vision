/**
 * Equipment Verification System Frontend JavaScript
 */

// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// Global state
let currentImage = null;
let currentResults = null;

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const imageInput = document.getElementById('imageInput');
const checklistSelect = document.getElementById('checklistSelect');
const confidenceSlider = document.getElementById('confidenceSlider');
const confidenceValue = document.getElementById('confidenceValue');
const resultsSection = document.getElementById('resultsSection');
const loadingSection = document.getElementById('loadingSection');
const errorSection = document.getElementById('errorSection');
const errorMessage = document.getElementById('errorMessage');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    updateConfidenceDisplay();
    loadAvailableChecklists();
});

/**
 * Initialize event listeners
 */
function initializeEventListeners() {
    // File input change
    imageInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    
    // Click to upload
    uploadArea.addEventListener('click', () => imageInput.click());
    
    // Confidence slider
    confidenceSlider.addEventListener('input', updateConfidenceDisplay);
    
    // Checklist change
    checklistSelect.addEventListener('change', handleChecklistChange);
}

/**
 * Update confidence value display
 */
function updateConfidenceDisplay() {
    confidenceValue.textContent = confidenceSlider.value;
}

/**
 * Handle file selection
 */
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        processImage(file);
    }
}

/**
 * Handle drag over
 */
function handleDragOver(event) {
    event.preventDefault();
    uploadArea.classList.add('dragover');
}

/**
 * Handle drag leave
 */
function handleDragLeave(event) {
    event.preventDefault();
    uploadArea.classList.remove('dragover');
}

/**
 * Handle file drop
 */
function handleDrop(event) {
    event.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        processImage(files[0]);
    }
}

/**
 * Process uploaded image
 */
async function processImage(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showError('Please select a valid image file.');
        return;
    }
    
    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        showError('Image file is too large. Please select an image smaller than 10MB.');
        return;
    }
    
    currentImage = file;
    showLoading();
    hideError();
    
    try {
        // Create form data
        const formData = new FormData();
        formData.append('file', file);
        formData.append('checklist', checklistSelect.value);
        formData.append('confidence_threshold', confidenceSlider.value);
        
        // Make API request
        const response = await fetch(`${API_BASE_URL}/detect`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to process image');
        }
        
        const results = await response.json();
        currentResults = results;
        displayResults(results);
        
    } catch (error) {
        console.error('Error processing image:', error);
        showError(error.message || 'Failed to process image. Please try again.');
    } finally {
        hideLoading();
    }
}

/**
 * Display detection results
 */
function displayResults(results) {
    // Update verification status
    updateVerificationStatus(results.verification);
    
    // Display annotated image
    displayAnnotatedImage(results.annotated_image_path);
    
    // Display detections
    displayDetections(results.detections);
    
    // Display verification details
    displayVerificationDetails(results.verification);
    
    // Show results section
    showResults();
}

/**
 * Update verification status indicator
 */
function updateVerificationStatus(verification) {
    const statusIndicator = document.getElementById('statusIndicator');
    const statusText = document.getElementById('statusText');
    
    statusIndicator.className = 'status-indicator';
    statusIndicator.classList.add(verification.status.toLowerCase());
    
    statusText.textContent = verification.status === 'PASS' ? 
        'All Required Equipment Present' : 
        'Missing Required Equipment';
}

/**
 * Display annotated image
 */
function displayAnnotatedImage(imagePath) {
    const img = document.getElementById('annotatedImage');
    img.src = `${API_BASE_URL}/detect/annotated/${imagePath}`;
    img.alt = 'Annotated detection results';
}

/**
 * Display detected objects
 */
function displayDetections(detections) {
    const detectionsList = document.getElementById('detectionsList');
    detectionsList.innerHTML = '';
    
    if (detections.length === 0) {
        detectionsList.innerHTML = '<p>No objects detected.</p>';
        return;
    }
    
    detections.forEach((detection, index) => {
        const detectionItem = document.createElement('div');
        detectionItem.className = 'detection-item';
        
        const confidence = (detection.bounding_box.confidence * 100).toFixed(1);
        
        detectionItem.innerHTML = `
            <h4>${detection.class_name}</h4>
            <p>Confidence: ${confidence}%</p>
            <p>Position: (${detection.bounding_box.x1.toFixed(0)}, ${detection.bounding_box.y1.toFixed(0)}) - (${detection.bounding_box.x2.toFixed(0)}, ${detection.bounding_box.y2.toFixed(0)})</p>
        `;
        
        detectionsList.appendChild(detectionItem);
    });
}

/**
 * Display verification details
 */
function displayVerificationDetails(verification) {
    const verificationDetails = document.getElementById('verificationDetails');
    verificationDetails.innerHTML = '';
    
    // Present equipment
    if (verification.present_equipment.length > 0) {
        verification.present_equipment.forEach(equipment => {
            const item = document.createElement('div');
            item.className = 'verification-item present';
            item.innerHTML = `
                <span class="equipment-name">${equipment}</span>
                <span class="equipment-status present">✓ Present</span>
            `;
            verificationDetails.appendChild(item);
        });
    }
    
    // Missing equipment
    if (verification.missing_equipment.length > 0) {
        verification.missing_equipment.forEach(equipment => {
            const item = document.createElement('div');
            item.className = 'verification-item missing';
            item.innerHTML = `
                <span class="equipment-name">${equipment}</span>
                <span class="equipment-status missing">✗ Missing</span>
            `;
            verificationDetails.appendChild(item);
        });
    }
    
    // Overall confidence
    const confidenceItem = document.createElement('div');
    confidenceItem.className = 'verification-item';
    confidenceItem.innerHTML = `
        <span class="equipment-name">Overall Confidence</span>
        <span class="equipment-status">${(verification.confidence * 100).toFixed(1)}%</span>
    `;
    verificationDetails.appendChild(confidenceItem);
}

/**
 * Load available checklists
 */
async function loadAvailableChecklists() {
    try {
        const response = await fetch(`${API_BASE_URL}/checklists`);
        if (response.ok) {
            const checklists = await response.json();
            updateChecklistOptions(checklists);
        }
    } catch (error) {
        console.error('Failed to load checklists:', error);
    }
}

/**
 * Update checklist options
 */
function updateChecklistOptions(checklists) {
    const select = document.getElementById('checklistSelect');
    select.innerHTML = '';
    
    Object.entries(checklists).forEach(([key, checklist]) => {
        const option = document.createElement('option');
        option.value = key;
        option.textContent = checklist.name;
        select.appendChild(option);
    });
}

/**
 * Handle checklist change
 */
function handleChecklistChange() {
    // If we have results, reprocess with new checklist
    if (currentImage && currentResults) {
        processImage(currentImage);
    }
}

/**
 * Show loading state
 */
function showLoading() {
    loadingSection.style.display = 'block';
    resultsSection.style.display = 'none';
    errorSection.style.display = 'none';
}

/**
 * Hide loading state
 */
function hideLoading() {
    loadingSection.style.display = 'none';
}

/**
 * Show results
 */
function showResults() {
    resultsSection.style.display = 'block';
    errorSection.style.display = 'none';
}

/**
 * Show error
 */
function showError(message) {
    errorMessage.textContent = message;
    errorSection.style.display = 'block';
    resultsSection.style.display = 'none';
    loadingSection.style.display = 'none';
}

/**
 * Hide error
 */
function hideError() {
    errorSection.style.display = 'none';
}

/**
 * Download results as JSON
 */
function downloadResults() {
    if (!currentResults) return;
    
    const dataStr = JSON.stringify(currentResults, null, 2);
    const dataBlob = new Blob([dataStr], {type: 'application/json'});
    const url = URL.createObjectURL(dataBlob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = 'detection_results.json';
    link.click();
    
    URL.revokeObjectURL(url);
}

/**
 * Copy results to clipboard
 */
async function copyResults() {
    if (!currentResults) return;
    
    try {
        await navigator.clipboard.writeText(JSON.stringify(currentResults, null, 2));
        alert('Results copied to clipboard!');
    } catch (error) {
        console.error('Failed to copy to clipboard:', error);
        alert('Failed to copy results to clipboard.');
    }
}

// Utility functions for external use
window.EquipmentVerification = {
    processImage,
    downloadResults,
    copyResults,
    showError,
    hideError
};


