// Global variables
let trainingStatusInterval;
let isTraining = false;

// DOM elements
const uploadBtn = document.getElementById('uploadBtn');
const trainingFile = document.getElementById('trainingFile');
const startTrainingBtn = document.getElementById('startTrainingBtn');
const generateBtn = document.getElementById('generateBtn');
const clearOutputBtn = document.getElementById('clearOutputBtn');
const uploadStatus = document.getElementById('uploadStatus');
const trainingLogs = document.getElementById('trainingLogs');
const progressBar = document.getElementById('progressBar');
const progressText = document.getElementById('progressText');
const statusIndicator = document.getElementById('statusIndicator');
const currentStep = document.getElementById('currentStep');
const totalSteps = document.getElementById('totalSteps');
const currentLoss = document.getElementById('currentLoss');
const generationOutput = document.getElementById('generationOutput');
const modelsList = document.getElementById('modelsList');

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
    // Initialize
    loadModels();
    checkTrainingStatus();
    
    // File upload
    uploadBtn.addEventListener('click', uploadFile);
    
    // Training
    startTrainingBtn.addEventListener('click', startTraining);
    
    // Text generation
    generateBtn.addEventListener('click', generateText);
    clearOutputBtn.addEventListener('click', clearOutput);
    
    // Tab switching
    document.getElementById('models-tab').addEventListener('click', loadModels);
});

// File upload function
function uploadFile() {
    const file = trainingFile.files[0];
    
    if (!file) {
        showAlert('Please select a file first.', 'warning');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    uploadBtn.disabled = true;
    uploadBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Uploading...';
    
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showAlert(data.message, 'success');
        } else {
            showAlert(data.error, 'danger');
        }
    })
    .catch(error => {
        showAlert('Upload failed: ' + error.message, 'danger');
    })
    .finally(() => {
        uploadBtn.disabled = false;
        uploadBtn.innerHTML = '<i class="fas fa-cloud-upload-alt me-2"></i>Upload File';
    });
}

// Start training function
function startTraining() {
    if (isTraining) {
        showAlert('Training is already in progress.', 'warning');
        return;
    }
    
    const trainingConfig = {
        model_type: document.getElementById('modelType').value,
        vocab_size: parseInt(document.getElementById('vocabSize').value),
        n_embd: parseInt(document.getElementById('nEmbd').value),
        n_layer: parseInt(document.getElementById('nLayer').value),
        n_head: parseInt(document.getElementById('nLayer').value), // Assuming n_head = n_layer for simplicity
        batch_size: parseInt(document.getElementById('batchSize').value),
        learning_rate: 3e-4,
        max_steps: parseInt(document.getElementById('maxSteps').value)
    };
    
    startTrainingBtn.disabled = true;
    startTrainingBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Starting...';
    
    fetch('/start_training', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(trainingConfig)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showAlert(data.message, 'success');
            isTraining = true;
            updateStatusIndicator('training');
            startTrainingStatusPolling();
        } else {
            showAlert(data.error, 'danger');
        }
    })
    .catch(error => {
        showAlert('Failed to start training: ' + error.message, 'danger');
    })
    .finally(() => {
        startTrainingBtn.disabled = false;
        startTrainingBtn.innerHTML = '<i class="fas fa-play me-2"></i>Start Training';
    });
}

// Check training status
function checkTrainingStatus() {
    fetch('/training_status')
    .then(response => response.json())
    .then(data => {
        updateTrainingUI(data);
        
        if (data.is_training && !trainingStatusInterval) {
            startTrainingStatusPolling();
        } else if (!data.is_training && trainingStatusInterval) {
            stopTrainingStatusPolling();
        }
    })
    .catch(error => {
        console.error('Error checking training status:', error);
    });
}

// Start polling training status
function startTrainingStatusPolling() {
    if (trainingStatusInterval) {
        clearInterval(trainingStatusInterval);
    }
    
    trainingStatusInterval = setInterval(() => {
        fetch('/training_status')
        .then(response => response.json())
        .then(data => {
            updateTrainingUI(data);
            
            if (!data.is_training) {
                stopTrainingStatusPolling();
                if (data.progress >= 100) {
                    updateStatusIndicator('complete');
                    showAlert('Training completed successfully!', 'success');
                } else {
                    updateStatusIndicator('error');
                    showAlert('Training stopped unexpectedly.', 'warning');
                }
            }
        })
        .catch(error => {
            console.error('Error polling training status:', error);
        });
    }, 2000); // Poll every 2 seconds
}

// Stop polling training status
function stopTrainingStatusPolling() {
    if (trainingStatusInterval) {
        clearInterval(trainingStatusInterval);
        trainingStatusInterval = null;
    }
    isTraining = false;
}

// Update training UI
function updateTrainingUI(data) {
    // Update progress
    const progress = Math.round(data.progress);
    progressBar.style.width = progress + '%';
    progressText.textContent = progress + '%';
    
    // Update step info
    currentStep.textContent = data.current_step;
    totalSteps.textContent = data.total_steps;
    currentLoss.textContent = data.loss.toFixed(4);
    
    // Update logs
    if (data.logs && data.logs.length > 0) {
        const logsHtml = data.logs.slice(-50).map(log => `<div>${escapeHtml(log)}</div>`).join('');
        trainingLogs.innerHTML = logsHtml;
        trainingLogs.scrollTop = trainingLogs.scrollHeight;
    }
    
    // Update status indicator
    if (data.is_training) {
        updateStatusIndicator('training');
        isTraining = true;
    } else {
        isTraining = false;
    }
}

// Update status indicator
function updateStatusIndicator(status) {
    statusIndicator.className = 'status-indicator';
    switch (status) {
        case 'training':
            statusIndicator.classList.add('status-training');
            break;
        case 'complete':
            statusIndicator.classList.add('status-complete');
            break;
        case 'error':
            statusIndicator.classList.add('status-error');
            break;
        default:
            statusIndicator.classList.add('status-idle');
    }
}

// Generate text function
function generateText() {
    const prompt = document.getElementById('promptInput').value.trim();
    
    if (!prompt) {
        showAlert('Please enter a prompt.', 'warning');
        return;
    }
    
    const generationConfig = {
        prompt: prompt,
        max_length: parseInt(document.getElementById('maxLength').value),
        temperature: parseFloat(document.getElementById('temperature').value),
        top_k: parseInt(document.getElementById('topK').value),
        top_p: parseFloat(document.getElementById('topP').value)
    };
    
    generateBtn.disabled = true;
    generateBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Generating...';
    
    fetch('/generate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(generationConfig)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            displayGeneratedText(data.prompt, data.generated_text);
        } else {
            showAlert(data.error, 'danger');
        }
    })
    .catch(error => {
        showAlert('Generation failed: ' + error.message, 'danger');
    })
    .finally(() => {
        generateBtn.disabled = false;
        generateBtn.innerHTML = '<i class="fas fa-magic me-2"></i>Generate Text';
    });
}

// Display generated text
function displayGeneratedText(prompt, generatedText) {
    const outputHtml = `
        <div class="generation-output">
            <h6><i class="fas fa-user me-2"></i>Prompt:</h6>
            <p class="mb-2">${escapeHtml(prompt)}</p>
            <h6><i class="fas fa-robot me-2"></i>Generated:</h6>
            <p class="mb-0">${escapeHtml(generatedText)}</p>
            <small class="text-muted">Generated at ${new Date().toLocaleTimeString()}</small>
        </div>
    `;
    
    if (generationOutput.innerHTML.includes('Generated text will appear here')) {
        generationOutput.innerHTML = outputHtml;
    } else {
        generationOutput.innerHTML += outputHtml;
    }
    
    // Scroll to bottom
    generationOutput.scrollTop = generationOutput.scrollHeight;
}

// Clear output
function clearOutput() {
    generationOutput.innerHTML = '<p class="text-muted">Generated text will appear here...</p>';
}

// Load models
function loadModels() {
    fetch('/models')
    .then(response => response.json())
    .then(data => {
        if (data.models && data.models.length > 0) {
            const modelsHtml = data.models.map(model => `
                <div class="card mb-2">
                    <div class="card-body">
                        <h6 class="card-title">${escapeHtml(model.name)}</h6>
                        <p class="card-text">
                            <small class="text-muted">
                                Size: ${formatFileSize(model.size)} | 
                                Modified: ${model.modified}
                            </small>
                        </p>
                    </div>
                </div>
            `).join('');
            modelsList.innerHTML = modelsHtml;
        } else {
            modelsList.innerHTML = '<p class="text-muted">No trained models found. Train a model first.</p>';
        }
    })
    .catch(error => {
        modelsList.innerHTML = '<p class="text-danger">Error loading models: ' + error.message + '</p>';
    });
}

// Utility functions
function showAlert(message, type) {
    const alertHtml = `
        <div class="alert alert-${type} alert-dismissible fade show" role="alert">
            ${escapeHtml(message)}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    uploadStatus.innerHTML = alertHtml;
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        const alert = uploadStatus.querySelector('.alert');
        if (alert) {
            alert.remove();
        }
    }, 5000);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Initialize on page load
window.addEventListener('load', function() {
    // Check if training is in progress on page load
    checkTrainingStatus();
});