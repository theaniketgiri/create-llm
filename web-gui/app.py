from flask import Flask, render_template, request, jsonify, send_file
import os
import subprocess
import threading
import json
import time
from werkzeug.utils import secure_filename
import yaml

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables to track training status
training_status = {
    'is_training': False,
    'progress': 0,
    'logs': [],
    'current_step': 0,
    'total_steps': 0,
    'loss': 0.0
}

ALLOWED_EXTENSIONS = {'txt', 'json', 'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Copy file to data directory
        project_data_dir = '../demo-llm-project/data'
        os.makedirs(project_data_dir, exist_ok=True)
        
        import shutil
        shutil.copy(filepath, os.path.join(project_data_dir, 'raw.txt'))
        
        return jsonify({
            'success': True,
            'message': f'File {filename} uploaded successfully',
            'filename': filename
        })
    
    return jsonify({'error': 'Invalid file type. Please upload .txt, .json, or .csv files'}), 400

@app.route('/start_training', methods=['POST'])
def start_training():
    global training_status
    
    if training_status['is_training']:
        return jsonify({'error': 'Training is already in progress'}), 400
    
    # Get training parameters from request
    data = request.get_json()
    model_type = data.get('model_type', 'gpt')
    vocab_size = data.get('vocab_size', 50000)
    n_embd = data.get('n_embd', 768)
    n_layer = data.get('n_layer', 12)
    n_head = data.get('n_head', 12)
    batch_size = data.get('batch_size', 8)
    learning_rate = data.get('learning_rate', 3e-4)
    max_steps = data.get('max_steps', 10000)
    
    # Update config file
    config_path = '../demo-llm-project/training/config.yaml'
    config = {
        'model': {
            'model_type': model_type,
            'vocab_size': vocab_size,
            'n_embd': n_embd,
            'n_layer': n_layer,
            'n_head': n_head
        },
        'training': {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'max_steps': max_steps,
            'save_steps': 1000,
            'eval_steps': 500
        },
        'data': {
            'max_length': 1024,
            'train_data_path': 'data/processed/train.txt',
            'val_data_path': 'data/processed/validation.txt'
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Reset training status
    training_status = {
        'is_training': True,
        'progress': 0,
        'logs': [],
        'current_step': 0,
        'total_steps': max_steps,
        'loss': 0.0
    }
    
    # Start training in a separate thread
    training_thread = threading.Thread(target=run_training_pipeline)
    training_thread.daemon = True
    training_thread.start()
    
    return jsonify({'success': True, 'message': 'Training started successfully'})

def run_training_pipeline():
    global training_status
    
    try:
        project_dir = '../demo-llm-project'
        
        # Step 1: Train tokenizer
        training_status['logs'].append('Training tokenizer...')
        result = subprocess.run([
            'python', 'tokenizer/train_tokenizer.py',
            '--input', 'data/raw.txt',
            '--vocab_size', str(training_status.get('vocab_size', 50000))
        ], cwd=project_dir, capture_output=True, text=True)
        
        if result.returncode != 0:
            training_status['logs'].append(f'Tokenizer training failed: {result.stderr}')
            training_status['is_training'] = False
            return
        
        training_status['logs'].append('Tokenizer training completed')
        training_status['progress'] = 25
        
        # Step 2: Prepare dataset
        training_status['logs'].append('Preparing dataset...')
        result = subprocess.run([
            'python', 'data/prepare_dataset.py'
        ], cwd=project_dir, capture_output=True, text=True)
        
        if result.returncode != 0:
            training_status['logs'].append(f'Dataset preparation failed: {result.stderr}')
            training_status['is_training'] = False
            return
        
        training_status['logs'].append('Dataset preparation completed')
        training_status['progress'] = 50
        
        # Step 3: Start model training
        training_status['logs'].append('Starting model training...')
        process = subprocess.Popen([
            'python', 'training/train.py',
            '--config', 'training/config.yaml'
        ], cwd=project_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Monitor training progress
        for line in iter(process.stdout.readline, ''):
            if line:
                training_status['logs'].append(line.strip())
                
                # Parse training progress from logs
                if 'Step' in line and 'Loss' in line:
                    try:
                        parts = line.split()
                        step_idx = parts.index('Step') + 1
                        loss_idx = parts.index('Loss') + 1
                        
                        current_step = int(parts[step_idx].rstrip(':'))
                        loss = float(parts[loss_idx])
                        
                        training_status['current_step'] = current_step
                        training_status['loss'] = loss
                        training_status['progress'] = 50 + (current_step / training_status['total_steps']) * 50
                    except (ValueError, IndexError):
                        pass
        
        process.wait()
        
        if process.returncode == 0:
            training_status['logs'].append('Training completed successfully!')
            training_status['progress'] = 100
        else:
            training_status['logs'].append('Training failed')
        
    except Exception as e:
        training_status['logs'].append(f'Error: {str(e)}')
    finally:
        training_status['is_training'] = False

@app.route('/training_status')
def get_training_status():
    return jsonify(training_status)

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.get_json()
    prompt = data.get('prompt', '')
    max_length = data.get('max_length', 50)
    temperature = data.get('temperature', 0.7)
    top_k = data.get('top_k', 40)
    top_p = data.get('top_p', 0.9)
    
    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400
    
    try:
        project_dir = '../demo-llm-project'
        
        # Find the latest checkpoint
        checkpoints_dir = os.path.join(project_dir, 'checkpoints')
        if not os.path.exists(checkpoints_dir):
            return jsonify({'error': 'No trained model found. Please train a model first.'}), 400
        
        checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pt')]
        if not checkpoint_files:
            return jsonify({'error': 'No model checkpoints found. Please train a model first.'}), 400
        
        # Use the latest checkpoint
        latest_checkpoint = max(checkpoint_files, key=lambda x: os.path.getctime(os.path.join(checkpoints_dir, x)))
        model_path = os.path.join('checkpoints', latest_checkpoint)
        
        # Run text generation
        result = subprocess.run([
            'python', 'eval/generate.py',
            '--model', model_path,
            '--prompt', prompt,
            '--max_length', str(max_length),
            '--temperature', str(temperature),
            '--top_k', str(top_k),
            '--top_p', str(top_p)
        ], cwd=project_dir, capture_output=True, text=True)
        
        if result.returncode != 0:
            return jsonify({'error': f'Generation failed: {result.stderr}'}), 500
        
        # Extract generated text from output
        output_lines = result.stdout.strip().split('\n')
        generated_text = ''
        for line in output_lines:
            if line.startswith('Generated:'):
                generated_text = line.replace('Generated:', '').strip()
                break
        
        if not generated_text:
            generated_text = result.stdout.strip()
        
        return jsonify({
            'success': True,
            'generated_text': generated_text,
            'prompt': prompt
        })
        
    except Exception as e:
        return jsonify({'error': f'Generation error: {str(e)}'}), 500

@app.route('/models')
def list_models():
    project_dir = '../demo-llm-project'
    checkpoints_dir = os.path.join(project_dir, 'checkpoints')
    
    if not os.path.exists(checkpoints_dir):
        return jsonify({'models': []})
    
    checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pt')]
    models = []
    
    for checkpoint in checkpoint_files:
        filepath = os.path.join(checkpoints_dir, checkpoint)
        stat = os.stat(filepath)
        models.append({
            'name': checkpoint,
            'size': stat.st_size,
            'modified': time.ctime(stat.st_mtime)
        })
    
    return jsonify({'models': models})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)