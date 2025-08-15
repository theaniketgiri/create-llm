#!/usr/bin/env python3
"""
Setup script for the LLM project.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("✗ Python 3.8 or higher is required")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def create_directories():
    """Create necessary directories."""
    directories = [
        'data/raw',
        'data/processed',
        'tokenizer',
        'checkpoints',
        'logs',
        'scripts'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def install_dependencies():
    """Install Python dependencies."""
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        return False
    return True

def download_sample_data():
    """Download sample data for testing."""
    print("\nDownloading sample data...")
    
    # Create sample text file
    sample_text = '''The quick brown fox jumps over the lazy dog. This is a sample text for training the language model.

Language models are a type of artificial intelligence that can understand and generate human language. They are trained on large amounts of text data and can perform various natural language processing tasks.

Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models that enable computers to perform tasks without explicit instructions.

Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data.

Natural language processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and human language.

Transformers are a type of neural network architecture that has revolutionized natural language processing tasks.

Attention mechanisms allow neural networks to focus on different parts of the input when processing information.

Tokenization is the process of breaking down text into smaller units called tokens for processing by language models.

Training a language model involves feeding it large amounts of text data and adjusting its parameters to minimize prediction errors.

Evaluation metrics like perplexity and accuracy help measure how well a language model performs on various tasks.'''
    
    with open('data/raw/sample.txt', 'w', encoding='utf-8') as f:
        f.write(sample_text)
    
    print("✓ Created sample data file: data/raw/sample.txt")

def setup_tokenizer():
    """Setup tokenizer with sample data."""
    if os.path.exists('data/raw/sample.txt'):
        print("\nSetting up tokenizer...")
        if run_command(
            "python tokenizer/train_tokenizer.py --input data/raw/sample.txt --output tokenizer/ --vocab-size 1000",
            "Training tokenizer on sample data"
        ):
            print("✓ Tokenizer setup completed")
        else:
            print("⚠ Tokenizer setup failed, you can train it manually later")
    else:
        print("⚠ Sample data not found, skipping tokenizer setup")

def main():
    """Main setup function."""
    print("🚀 Setting up LLM project...")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("\n✗ Setup failed during dependency installation")
        print("Please install dependencies manually: pip install -r requirements.txt")
        sys.exit(1)
    
    # Download sample data
    download_sample_data()
    
    # Setup tokenizer
    setup_tokenizer()
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Add your training data to data/raw/")
    print("2. Train tokenizer: python tokenizer/train_tokenizer.py --input data/raw/your_data.txt --output tokenizer/")
    print("3. Prepare dataset: python data/prepare_dataset.py --input data/raw/ --output data/processed/")
    print("4. Start training: python training/train.py --config training/config.yaml")
    print("\nFor more information, see the README.md file.")

if __name__ == "__main__":
    main()
