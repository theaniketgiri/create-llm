// Initialize Socket.IO connection
const socket = io();

// Global variables
let isTraining = false;
let currentConfig = {};

// DOM elements
const elements = {
    statusIndicator: document.getElementById('status-indicator'),
    tokenizerStatus: document.getElementById('tokenizer-status'),
    datasetStatus: document.getElementById('dataset-status'),
    modelStatus: document.getElementById('model-status'),
    trainingStatus: document.getElementById('training-status'),
    startTrainingBtn: document.getElementById('start-training'),
    stopTrainingBtn: document.getElementById('stop-training'),
    trainingLogs: document.getElementById('training-logs'),
    clearLogsBtn: document.getElementById('clear-logs'),
    chatInput: document.getElementById('chat-input'),
    sendMessageBtn: document.getElementById('send-message'),
    chatMessages: document.getElementById('chat-messages'),
    modelSelect: document.getElementById('model-select'),
    temperature: document.getElementById('temperature'),
    tempValue: document.getElementById('temp-value'),
    maxLength: document.getElementById('max-length'),
    fileUpload: document.getElementById('file-upload'),
    uploadProgress: document.getElementById('upload-progress'),
    uploadBar: document.getElementById('upload-bar'),
    uploadStatus: document.getElementById('upload-status'),
    dataPreview: document.getElementById('data-preview')
};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    loadProjectStatus();
    setupEventListeners();
    setupSocketListeners();
    loadConfiguration();
});

// Setup event listeners
function setupEventListeners() {
    // Training controls
    elements.startTrainingBtn.addEventListener('click', startTraining);
    elements.stopTrainingBtn.addEventListener('click', stopTraining);
    elements.clearLogsBtn.addEventListener('click', clearLogs);
    
    // Chat interface
    elements.sendMessageBtn.addEventListener('click', sendMessage);
    elements.chatInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') sendMessage();
    });
    
    // Configuration
    document.getElementById('save-config').addEventListener('click', saveConfiguration);
    
    // File upload
    elements.fileUpload.addEventListener('change', handleFileUpload);
    
    // Temperature slider
    elements.temperature.addEventListener('input', function() {
        elements.tempValue.textContent = this.value;
    });
}

// Setup Socket.IO listeners
function setupSocketListeners() {
    socket.on('training_log', function(data) {
        appendTrainingLog(data.data, data.type);
        updateTrainingProgress(data.data);
    });
    
    socket.on('training_complete', function(data) {
        isTraining = false;
        updateTrainingStatus('Completed');
        elements.startTrainingBtn.disabled = false;
        elements.stopTrainingBtn.disabled = true;
        loadProjectStatus();
    });
}

// Load project status
async function loadProjectStatus() {
    try {
        const response = await fetch('/api/status');
        const status = await response.json();
        
        elements.tokenizerStatus.textContent = status.hasTokenizer ? 'Trained' : 'Not trained';
        elements.datasetStatus.textContent = status.hasData ? 'Ready' : 'Not prepared';
        elements.modelStatus.textContent = status.hasModel ? 'Available' : 'Not trained';
        
        if (status.config) {
            currentConfig = status.config;
            loadConfigurationUI();
        }
    } catch (error) {
        console.error('Error loading project status:', error);
    }
}

// Start training
async function startTraining() {
    try {
        const config = getConfigurationFromUI();
        
        const response = await fetch('/api/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ config })
        });
        
        if (response.ok) {
            isTraining = true;
            updateTrainingStatus('Training...');
            elements.startTrainingBtn.disabled = true;
            elements.stopTrainingBtn.disabled = false;
            clearLogs();
            appendTrainingLog('Starting training...', 'info');
        }
    } catch (error) {
        console.error('Error starting training:', error);
        appendTrainingLog('Error starting training: ' + error.message, 'error');
    }
}

// Stop training
function stopTraining() {
    // This would need to be implemented on the server side
    appendTrainingLog('Stopping training...', 'info');
    isTraining = false;
    updateTrainingStatus('Stopped');
    elements.startTrainingBtn.disabled = false;
    elements.stopTrainingBtn.disabled = true;
}

// Send chat message
async function sendMessage() {
    const message = elements.chatInput.value.trim();
    if (!message) return;
    
    // Add user message to chat
    addChatMessage('user', message);
    elements.chatInput.value = '';
    
    try {
        const response = await fetch('/api/test', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                prompt: message,
                modelPath: elements.modelSelect.value,
                temperature: parseFloat(elements.temperature.value),
                maxLength: parseInt(elements.maxLength.value)
            })
        });
        
        const data = await response.json();
        addChatMessage('assistant', data.response || 'No response from model');
    } catch (error) {
        addChatMessage('error', 'Error: ' + error.message);
    }
}

// Add chat message
function addChatMessage(role, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `flex ${role === 'user' ? 'justify-end' : 'justify-start'} mb-3`;
    
    const bubble = document.createElement('div');
    bubble.className = `max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
        role === 'user' 
            ? 'bg-blue-500 text-white' 
            : role === 'error'
            ? 'bg-red-500 text-white'
            : 'bg-gray-200 text-gray-800'
    }`;
    bubble.textContent = content;
    
    messageDiv.appendChild(bubble);
    elements.chatMessages.appendChild(messageDiv);
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

// Handle file upload
async function handleFileUpload(event) {
    const files = event.target.files;
    if (!files.length) return;
    
    elements.uploadProgress.classList.remove('hidden');
    elements.uploadStatus.textContent = 'Uploading...';
    
    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const formData = new FormData();
        formData.append('dataset', file);
        
        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                const data = await response.json();
                elements.uploadStatus.textContent = `Uploaded: ${file.name}`;
                elements.uploadBar.style.width = `${((i + 1) / files.length) * 100}%`;
                
                // Update data preview
                await updateDataPreview();
            }
        } catch (error) {
            elements.uploadStatus.textContent = `Error uploading ${file.name}: ${error.message}`;
        }
    }
    
    setTimeout(() => {
        elements.uploadProgress.classList.add('hidden');
    }, 2000);
}

// Update data preview
async function updateDataPreview() {
    try {
        const response = await fetch('/api/logs');
        const data = await response.json();
        
        if (data.logs && data.logs !== 'No logs available') {
            elements.dataPreview.innerHTML = `<pre class="text-sm">${data.logs.substring(0, 500)}...</pre>`;
        } else {
            elements.dataPreview.innerHTML = '<div class="text-gray-500">No data available</div>';
        }
    } catch (error) {
        elements.dataPreview.innerHTML = '<div class="text-red-500">Error loading data preview</div>';
    }
}

// Tab switching
function showTab(tabName) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.add('hidden');
    });
    
    // Remove active class from all tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active', 'border-blue-500', 'text-blue-600');
        btn.classList.add('border-transparent', 'text-gray-500');
    });
    
    // Show selected tab
    document.getElementById(tabName + '-tab').classList.remove('hidden');
    
    // Add active class to selected tab button
    event.target.classList.add('active', 'border-blue-500', 'text-blue-600');
    event.target.classList.remove('border-transparent', 'text-gray-500');
}

// Configuration functions
function loadConfiguration() {
    // Load default configuration
    currentConfig = {
        learning_rate: '3e-4',
        batch_size: 8,
        max_steps: 100000,
        n_embd: 768,
        n_layer: 12,
        n_head: 12
    };
    loadConfigurationUI();
}

function loadConfigurationUI() {
    document.getElementById('learning-rate').value = currentConfig.learning_rate || '3e-4';
    document.getElementById('batch-size').value = currentConfig.batch_size || 8;
    document.getElementById('max-steps').value = currentConfig.max_steps || 100000;
    document.getElementById('n-embd').value = currentConfig.n_embd || 768;
    document.getElementById('n-layer').value = currentConfig.n_layer || 12;
    document.getElementById('n-head').value = currentConfig.n_head || 12;
}

function getConfigurationFromUI() {
    return {
        learning_rate: document.getElementById('learning-rate').value,
        batch_size: parseInt(document.getElementById('batch-size').value),
        max_steps: parseInt(document.getElementById('max-steps').value),
        n_embd: parseInt(document.getElementById('n-embd').value),
        n_layer: parseInt(document.getElementById('n-layer').value),
        n_head: parseInt(document.getElementById('n-head').value)
    };
}

async function saveConfiguration() {
    try {
        const config = getConfigurationFromUI();
        currentConfig = config;
        
        const response = await fetch('/api/config', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(config)
        });
        
        if (response.ok) {
            showNotification('Configuration saved successfully!', 'success');
        }
    } catch (error) {
        showNotification('Error saving configuration: ' + error.message, 'error');
    }
}

// Utility functions
function appendTrainingLog(message, type = 'info') {
    const logDiv = document.createElement('div');
    logDiv.className = `mb-1 ${type === 'error' ? 'text-red-400' : type === 'warning' ? 'text-yellow-400' : 'text-green-400'}`;
    logDiv.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    
    elements.trainingLogs.appendChild(logDiv);
    elements.trainingLogs.scrollTop = elements.trainingLogs.scrollHeight;
}

function clearLogs() {
    elements.trainingLogs.innerHTML = '<div class="text-gray-500">Training logs will appear here...</div>';
}

function updateTrainingStatus(status) {
    elements.trainingStatus.textContent = status;
}

function updateTrainingProgress(logData) {
    // Parse training progress from log data
    const epochMatch = logData.match(/epoch[:\s]+(\d+)/i);
    const lossMatch = logData.match(/loss[:\s]+([\d.]+)/i);
    
    if (epochMatch) {
        document.getElementById('current-epoch').textContent = epochMatch[1];
        // Update progress bar (simplified)
        const progress = Math.min((parseInt(epochMatch[1]) / 100) * 100, 100);
        document.getElementById('epoch-progress').style.width = progress + '%';
    }
    
    if (lossMatch) {
        document.getElementById('current-loss').textContent = lossMatch[1];
        // Update loss progress (simplified)
        const loss = parseFloat(lossMatch[1]);
        const progress = Math.max(0, Math.min((1 - loss / 10) * 100, 100));
        document.getElementById('loss-progress').style.width = progress + '%';
    }
}

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `fixed top-4 right-4 px-6 py-3 rounded-lg text-white z-50 ${
        type === 'success' ? 'bg-green-500' : type === 'error' ? 'bg-red-500' : 'bg-blue-500'
    }`;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.remove();
    }, 3000);
}

// Global function for tab switching (called from HTML)
window.showTab = showTab;
