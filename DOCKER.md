# 🐳 Docker Guide for Create-LLM

This guide shows you how to use create-llm with Docker, eliminating the need for local Python/Node.js installations.

## Quick Start

### 🚀 One-Command Setup (Recommended)

```bash
# Download and run the setup script
curl -fsSL https://raw.githubusercontent.com/theaniketgiri/create-llm/main/setup-docker.sh | bash
```

This interactive script will:
- ✅ Check Docker installation
- ✅ Build or pull the Docker image  
- ✅ Create a sample project (optional)
- ✅ Generate helper scripts
- ✅ Show you next steps

### Manual Setup

#### 1. Build the Docker Image

```bash
# Clone the repository
git clone https://github.com/theaniketgiri/create-llm.git
cd create-llm

# Build the all-in-one image
docker build -t create-llm .
```

### 2. Create a New Project

```bash
# Create a new directory for your project
mkdir my-llm-project
cd my-llm-project

# Create a new LLM project using Docker
docker run -it -v $(pwd):/workspace create-llm scaffold my-awesome-llm --template tiny
```

### 3. Train Your Model

```bash
# Navigate to your project
cd my-awesome-llm

# Start training (with GPU support if available)
docker run --gpus all -v $(pwd):/workspace create-llm train

# Or without GPU
docker run -v $(pwd):/workspace create-llm train
```

### 4. Chat with Your Model

```bash
# Start interactive chat interface
docker run -p 7860:7860 -v $(pwd):/workspace create-llm chat

# Open http://localhost:7860 in your browser
```

## Docker Images

We provide three specialized Docker images:

### 1. All-in-One Image (Recommended)
```bash
# Build
docker build -t create-llm .

# Usage: Scaffolding + Training + Chat
docker run -v $(pwd):/workspace create-llm scaffold my-project
docker run --gpus all -v $(pwd):/workspace create-llm train
```

### 2. CLI-Only Image (Lightweight)
```bash
# Build
docker build -f Dockerfile.cli -t create-llm:cli .

# Usage: Project scaffolding only
docker run -v $(pwd):/workspace create-llm:cli my-project --template nano
```

### 3. Training-Only Image (GPU Optimized)
```bash
# Build
docker build -f Dockerfile.training -t create-llm:training .

# Usage: Training existing projects
docker run --gpus all -v $(pwd):/workspace create-llm:training python training/train.py
```

## Docker Compose (Recommended)

For easier management, use Docker Compose:

### Basic Usage

```bash
# Start CLI service
docker-compose --profile cli run --rm create-llm scaffold my-project

# Start training service with GPU
docker-compose --profile training run --rm training

# Start chat interface
docker-compose --profile chat up chat
```

### Development Environment

```bash
# Start interactive development shell
docker-compose --profile dev run --rm dev

# Start Jupyter notebook for experimentation
docker-compose --profile jupyter up jupyter
# Access at http://localhost:8888
```

## Common Workflows

### Complete Project Lifecycle

```bash
# 1. Create project directory
mkdir llm-experiments && cd llm-experiments

# 2. Create new project
docker run -it -v $(pwd):/workspace create-llm scaffold sentiment-analyzer --template small

# 3. Navigate to project
cd sentiment-analyzer

# 4. Add your training data
# Place text files in data/raw/

# 5. Start training with monitoring
docker run --gpus all -p 6006:6006 -v $(pwd):/workspace create-llm train --tensorboard

# 6. Monitor training (open another terminal)
# TensorBoard will be available at http://localhost:6006

# 7. Test your model
docker run -p 7860:7860 -v $(pwd):/workspace create-llm chat

# 8. Evaluate performance
docker run -v $(pwd):/workspace create-llm evaluate

# 9. Deploy to HuggingFace
docker run -it -v $(pwd):/workspace create-llm deploy --platform huggingface
```

### GPU Training with Resource Limits

```bash
# Limit GPU memory usage
docker run --gpus all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -v $(pwd):/workspace \
  create-llm train --max-gpu-memory 8GB

# Use multiple GPUs
docker run --gpus all \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -v $(pwd):/workspace \
  create-llm train --distributed
```

### Batch Processing Multiple Projects

```bash
# Train multiple models in sequence
projects=("nano-model" "tiny-model" "small-model")
templates=("nano" "tiny" "small")

for i in "${!projects[@]}"; do
  project="${projects[$i]}"
  template="${templates[$i]}"
  
  echo "Creating $project with $template template..."
  docker run -v $(pwd):/workspace create-llm scaffold $project --template $template
  
  echo "Training $project..."
  docker run --gpus all -v $(pwd)/$project:/workspace create-llm train
done
```

## Volume Mounts

### Essential Mounts

```bash
# Project files
-v $(pwd):/workspace

# Cache for faster subsequent runs
-v llm-cache:/root/.cache

# Persistent model checkpoints
-v llm-models:/workspace/models/checkpoints
```

### Example with All Mounts

```bash
docker run --gpus all \
  -v $(pwd):/workspace \
  -v llm-cache:/root/.cache \
  -v llm-models:/workspace/models/checkpoints \
  -v llm-datasets:/workspace/data \
  create-llm train
```

## Environment Variables

### Training Configuration

```bash
# Python settings
-e PYTHONUNBUFFERED=1
-e TOKENIZERS_PARALLELISM=false

# GPU settings
-e CUDA_VISIBLE_DEVICES=0,1
-e TORCH_CUDA_ARCH_LIST="8.0;8.6"

# Training parameters
-e LLM_BATCH_SIZE=16
-e LLM_LEARNING_RATE=3e-4
-e LLM_MAX_STEPS=10000
```

### Example with Environment Variables

```bash
docker run --gpus all \
  -e PYTHONUNBUFFERED=1 \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e LLM_BATCH_SIZE=8 \
  -v $(pwd):/workspace \
  create-llm train
```

## Port Mappings

```bash
# Gradio chat interface
-p 7860:7860

# TensorBoard
-p 6006:6006

# Jupyter notebook
-p 8888:8888

# Custom web interfaces
-p 8080:8080
```

## Troubleshooting

### Common Issues

1. **Permission Issues**
   ```bash
   # Fix file permissions
   docker run -v $(pwd):/workspace create-llm shell
   chown -R $(id -u):$(id -g) /workspace
   ```

2. **GPU Not Detected**
   ```bash
   # Check NVIDIA Docker runtime
   docker run --gpus all nvidia/cuda:12.1-base nvidia-smi
   
   # Verify in create-llm
   docker run --gpus all -v $(pwd):/workspace create-llm python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **Out of Memory**
   ```bash
   # Reduce batch size
   docker run --gpus all -v $(pwd):/workspace create-llm train --batch-size 4
   
   # Enable gradient checkpointing
   docker run --gpus all -v $(pwd):/workspace create-llm train --gradient-checkpointing
   ```

4. **Slow Training**
   ```bash
   # Use optimized Docker image
   docker build -f Dockerfile.training -t create-llm:fast .
   docker run --gpus all -v $(pwd):/workspace create-llm:fast python training/train.py
   ```

### Debugging

```bash
# Start interactive shell
docker run -it -v $(pwd):/workspace create-llm shell

# Check system resources
docker run --gpus all -v $(pwd):/workspace create-llm python -c "
import torch
import psutil
print(f'GPUs: {torch.cuda.device_count()}')
print(f'RAM: {psutil.virtual_memory().total / 1e9:.1f} GB')
print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# View container logs
docker logs create-llm-training
```

## Performance Optimization

### Docker Settings

```bash
# Increase shared memory for data loading
--shm-size=16g

# Use host networking for better performance
--network host

# Optimize Docker daemon settings
# In /etc/docker/daemon.json:
{
  "default-runtime": "nvidia",
  "storage-driver": "overlay2",
  "storage-opts": ["overlay2.override_kernel_check=true"]
}
```

### Resource Limits

```bash
# Limit CPU and memory usage
docker run --cpus="4.0" --memory="16g" \
  -v $(pwd):/workspace create-llm train

# Set GPU memory fraction
docker run --gpus all \
  -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
  -v $(pwd):/workspace create-llm train
```

## Integration Examples

### CI/CD Pipeline

```yaml
# .github/workflows/train-model.yml
name: Train LLM Model

on:
  push:
    paths: ['data/**']

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Build Docker image
      run: docker build -t create-llm .
    
    - name: Train model
      run: |
        docker run -v ${{ github.workspace }}:/workspace \
          create-llm train --max-steps 1000
    
    - name: Upload artifacts
      uses: actions/upload-artifact@v4
      with:
        name: model-checkpoints
        path: models/checkpoints/
```

### Docker Swarm Deployment

```yaml
# docker-stack.yml
version: '3.8'
services:
  training:
    image: create-llm:latest
    deploy:
      replicas: 2
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - training-data:/workspace/data
      - model-checkpoints:/workspace/models
    command: ["train", "--distributed"]
```

This Docker setup provides a complete containerized solution for create-llm, allowing users to:

- ✅ Run without local Python/Node.js installations
- ✅ Use GPU acceleration seamlessly  
- ✅ Manage dependencies automatically
- ✅ Scale training across multiple containers
- ✅ Deploy in production environments
- ✅ Integrate with CI/CD pipelines

Choose the Docker configuration that best fits your use case!