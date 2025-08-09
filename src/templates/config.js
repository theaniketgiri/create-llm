const fs = require('fs-extra');
const path = require('path');

async function createConfigFiles(projectPath, options) {
  // Create .gitignore
  await createGitignore(projectPath);
  
  // Create setup script
  await createSetupScript(projectPath, options);
  
  // Create synthetic data script
  if (options.includeSyntheticData) {
    await createSyntheticDataScript(projectPath, options);
  }
  
  // Create utility scripts
  await createUtilityScripts(projectPath, options);
}

async function createGitignore(projectPath) {
  const gitignoreContent = `# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/
env.bak/
venv.bak/

# PyTorch
*.pth
*.pt

# Checkpoints and logs
checkpoints/
logs/
*.ckpt

# Data
data/raw/
data/processed/
*.txt
*.json
*.csv

# Tokenizer
tokenizer/tokenizer.json
tokenizer/vocab.json
tokenizer/special_tokens.json

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Environment variables
.env
.env.local

# Temporary files
*.tmp
*.temp
`;
  
  await fs.writeFile(path.join(projectPath, '.gitignore'), gitignoreContent);
}

async function createSetupScript(projectPath, options) {
  const setupContent = `#!/usr/bin/env python3
"""
Setup script for the LLM project.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\\n{description}...")
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
    print("\\nDownloading sample data...")
    
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
        print("\\nSetting up tokenizer...")
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
        print("\\n✗ Setup failed during dependency installation")
        print("Please install dependencies manually: pip install -r requirements.txt")
        sys.exit(1)
    
    # Download sample data
    download_sample_data()
    
    # Setup tokenizer
    setup_tokenizer()
    
    print("\\n🎉 Setup completed successfully!")
    print("\\nNext steps:")
    print("1. Add your training data to data/raw/")
    print("2. Train tokenizer: python tokenizer/train_tokenizer.py --input data/raw/your_data.txt --output tokenizer/")
    print("3. Prepare dataset: python data/prepare_dataset.py --input data/raw/ --output data/processed/")
    print("4. Start training: python training/train.py --config training/config.yaml")
    print("\\nFor more information, see the README.md file.")

if __name__ == "__main__":
    main()
`;
  
  await fs.writeFile(path.join(projectPath, 'setup.py'), setupContent);
}

async function createSyntheticDataScript(projectPath, options) {
  const syntheticContent = `#!/usr/bin/env python3
"""
Synthetic data generation for LLM training.
Powered by SynthexAI - https://synthex.theaniketgiri.me

Usage:
    python scripts/generate_synthetic_data.py --type code --size 10000
    python scripts/generate_synthetic_data.py --type medical --size 5000
"""

import argparse
import os
import json
import random
import re
from pathlib import Path
from typing import List, Dict, Optional

class SyntheticDataGenerator:
    """Generate synthetic data for LLM training."""
    
    def __init__(self):
        self.templates = {
            'code': self._get_code_templates(),
            'medical': self._get_medical_templates(),
            'news': self._get_news_templates(),
            'fiction': self._get_fiction_templates(),
            'technical': self._get_technical_templates()
        }
    
    def _get_code_templates(self) -> List[str]:
        """Get code generation templates."""
        return [
            "def {function_name}({params}):\n    \"\"\"{docstring}\"\"\"\n    {implementation}\n    return {return_value}",
            "class {class_name}:\n    def __init__(self, {params}):\n        {init_implementation}\n    \n    def {method_name}(self, {method_params}):\n        {method_implementation}",
            "import {module}\n\n{code_block}",
            "if {condition}:\n    {true_block}\nelse:\n    {false_block}",
            "for {variable} in {iterable}:\n    {loop_body}",
            "try:\n    {try_block}\nexcept {exception} as {error_var}:\n    {except_block}",
            "def {function_name}({params}):\n    {implementation}\n    if {condition}:\n        return {return_value1}\n    return {return_value2}"
        ]
    
    def _get_medical_templates(self) -> List[str]:
        """Get medical text templates."""
        return [
            "The patient presented with {symptoms}. Upon examination, {findings}. The diagnosis was {diagnosis}.",
            "Treatment for {condition} typically involves {treatment}. {explanation}.",
            "The {procedure} was performed successfully. {outcome}.",
            "Common side effects of {medication} include {side_effects}. {management}.",
            "The {test} results showed {results}. {interpretation}.",
            "Risk factors for {disease} include {risk_factors}. {prevention}.",
            "The {organ} is responsible for {function}. {anatomy}."
        ]
    
    def _get_news_templates(self) -> List[str]:
        """Get news article templates."""
        return [
            "{location} - {event} occurred today, {description}. {impact}.",
            "Officials announced {announcement} in response to {situation}. {details}.",
            "The {industry} sector reported {news}. {analysis}.",
            "Research shows that {finding}. {implications}.",
            "Experts believe that {prediction}. {reasoning}.",
            "The government approved {policy}. {effects}.",
            "Local residents expressed {reaction} to {event}. {quotes}."
        ]
    
    def _get_fiction_templates(self) -> List[str]:
        """Get fiction writing templates."""
        return [
            "The {character} walked through the {setting}, {description}. {action}.",
            "In the distance, {observation}. {character} felt {emotion}.",
            "The {object} glowed with {quality}. {character} reached out and {action}.",
            "Memories of {past_event} flooded {character}'s mind. {reflection}.",
            "The {weather} created an atmosphere of {mood}. {character} {action}.",
            "Through the {obstacle}, {character} could see {vision}. {reaction}.",
            "The {sound} echoed through the {location}. {character} {response}."
        ]
    
    def _get_technical_templates(self) -> List[str]:
        """Get technical documentation templates."""
        return [
            "The {system} architecture consists of {components}. {explanation}.",
            "To configure {feature}, follow these steps: {steps}. {notes}.",
            "The {algorithm} operates by {process}. {complexity}.",
            "When {condition}, the system will {behavior}. {implications}.",
            "The {protocol} ensures {guarantee}. {implementation}.",
            "Performance metrics show {results}. {analysis}.",
            "The {interface} provides {functionality}. {usage}."
        ]
    
    def _get_random_words(self, category: str, count: int) -> List[str]:
        """Get random words for a category."""
        word_lists = {
            'function_name': ['process_data', 'calculate_result', 'validate_input', 'transform_data', 'generate_report'],
            'class_name': ['DataProcessor', 'UserManager', 'FileHandler', 'NetworkClient', 'DatabaseConnection'],
            'params': ['data', 'config', 'options', 'parameters', 'settings'],
            'docstring': ['Process the input data', 'Calculate the final result', 'Validate user input', 'Transform data format'],
            'implementation': ['result = data * 2', 'return data.upper()', 'data.sort()', 'data.reverse()'],
            'return_value': ['result', 'data', 'True', 'False', 'None'],
            'symptoms': ['fever', 'headache', 'nausea', 'fatigue', 'pain'],
            'diagnosis': ['common cold', 'migraine', 'food poisoning', 'stress', 'injury'],
            'treatment': ['rest', 'medication', 'therapy', 'surgery', 'lifestyle changes'],
            'location': ['New York', 'London', 'Tokyo', 'Paris', 'Sydney'],
            'event': ['accident', 'announcement', 'discovery', 'meeting', 'celebration'],
            'character': ['hero', 'villain', 'witness', 'detective', 'teacher'],
            'setting': ['forest', 'city', 'castle', 'beach', 'mountain'],
            'emotion': ['fear', 'joy', 'sadness', 'anger', 'surprise'],
            'system': ['database', 'network', 'application', 'server', 'client'],
            'feature': ['authentication', 'logging', 'caching', 'monitoring', 'backup']
        }
        
        words = word_lists.get(category, ['example', 'sample', 'test', 'demo', 'placeholder'])
        return random.choices(words, k=count)
    
    def _fill_template(self, template: str) -> str:
        """Fill a template with random values."""
        # Find all placeholders
        placeholders = re.findall(r'\{(\w+)\}', template)
        
        # Replace each placeholder
        result = template
        for placeholder in placeholders:
            words = self._get_random_words(placeholder, 1)
            result = result.replace(f'{{{placeholder}}}', words[0])
        
        return result
    
    def generate_data(self, data_type: str, size: int) -> List[str]:
        """
        Generate synthetic data.
        
        Args:
            data_type: Type of data to generate
            size: Number of samples to generate
            
        Returns:
            List of generated texts
        """
        if data_type not in self.templates:
            raise ValueError(f"Unknown data type: {data_type}")
        
        templates = self.templates[data_type]
        generated_texts = []
        
        for _ in range(size):
            # Select random template
            template = random.choice(templates)
            
            # Fill template
            text = self._fill_template(template)
            
            # Add some variation
            if random.random() < 0.3:
                text += f" {random.choice(['Additionally,', 'Furthermore,', 'Moreover,', 'However,'])} {self._fill_template(random.choice(templates))}"
            
            generated_texts.append(text)
        
        return generated_texts
    
    def save_data(self, texts: List[str], output_path: str):
        """Save generated data to file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\\n\\n')

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data for LLM training")
    parser.add_argument("--type", "-t", choices=["code", "medical", "news", "fiction", "technical"], 
                       required=True, help="Type of data to generate")
    parser.add_argument("--size", "-s", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--output", "-o", help="Output file path")
    
    args = parser.parse_args()
    
    # Set default output path
    if not args.output:
        args.output = f"data/raw/synthetic_{args.type}_{args.size}.txt"
    
    try:
        # Create generator
        generator = SyntheticDataGenerator()
        
        # Generate data
        print(f"Generating {args.size} {args.type} samples...")
        texts = generator.generate_data(args.type, args.size)
        
        # Save data
        generator.save_data(texts, args.output)
        
        print(f"Generated {len(texts)} samples")
        print(f"Data saved to: {args.output}")
        
        # Show sample
        print("\\nSample generated text:")
        print(texts[0])
        
        return 0
    except Exception as e:
        print(f"Error generating data: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
`;
  
  const scriptsPath = path.join(projectPath, 'scripts');
  await fs.ensureDir(scriptsPath);
  await fs.writeFile(path.join(scriptsPath, 'generate_synthetic_data.py'), syntheticContent);
}

async function createUtilityScripts(projectPath, options) {
  const scriptsPath = path.join(projectPath, 'scripts');
  
  // Create model info script
  const modelInfoContent = `#!/usr/bin/env python3
"""
Display model information and statistics.
"""

import argparse
import torch
from model import TransformerLM, ModelConfig

def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size_mb(model):
    """Get model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb

def main():
    parser = argparse.ArgumentParser(description="Display model information")
    parser.add_argument("--config", "-c", help="Path to model config file")
    parser.add_argument("--checkpoint", help="Path to model checkpoint")
    
    args = parser.parse_args()
    
    if args.checkpoint:
        # Load from checkpoint
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        config = checkpoint['config']
        model_config = ModelConfig(
            vocab_size=config.get('vocab_size', 50257),
            n_positions=config.get('max_length', 1024),
            n_embd=config.get('n_embd', 768),
            n_layer=config.get('n_layer', 12),
            n_head=config.get('n_head', 12),
            model_type="${options.template}"
        )
    else:
        # Use default config
        model_config = ModelConfig()
    
    # Create model
    model = TransformerLM(model_config)
    
    # Calculate statistics
    num_params = count_parameters(model)
    model_size = get_model_size_mb(model)
    
    print("Model Information:")
    print(f"Architecture: {model_config.model_type.upper()}")
    print(f"Vocabulary size: {model_config.vocab_size:,}")
    print(f"Max sequence length: {model_config.n_positions:,}")
    print(f"Embedding dimension: {model_config.n_embd:,}")
    print(f"Number of layers: {model_config.n_layer}")
    print(f"Number of attention heads: {model_config.n_head}")
    print(f"Total parameters: {num_params:,}")
    print(f"Model size: {model_size:.2f} MB")
    
    # Memory requirements
    batch_size = 1
    seq_len = model_config.n_positions
    memory_per_sample = (batch_size * seq_len * model_config.n_embd * 4) / 1024**2  # 4 bytes per float32
    print(f"Memory per sample (batch_size=1): {memory_per_sample:.2f} MB")

if __name__ == "__main__":
    main()
`;
  
  await fs.writeFile(path.join(scriptsPath, 'model_info.py'), modelInfoContent);
  
  // Create data analysis script
  const dataAnalysisContent = `#!/usr/bin/env python3
"""
Analyze training data statistics.
"""

import argparse
import os
import re
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

def analyze_text_file(file_path: str):
    """Analyze a single text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Basic statistics
    lines = text.split('\\n')
    paragraphs = text.split('\\n\\n')
    sentences = re.split(r'[.!?]+', text)
    words = re.findall(r'\\b\\w+\\b', text.lower())
    characters = len(text)
    
    # Word frequency
    word_freq = Counter(words)
    most_common = word_freq.most_common(10)
    
    # Average lengths
    avg_sentence_length = sum(len(s.split()) for s in sentences if s.strip()) / len([s for s in sentences if s.strip()])
    avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
    
    return {
        'file': file_path,
        'characters': characters,
        'lines': len(lines),
        'paragraphs': len(paragraphs),
        'sentences': len([s for s in sentences if s.strip()]),
        'words': len(words),
        'unique_words': len(word_freq),
        'avg_sentence_length': avg_sentence_length,
        'avg_word_length': avg_word_length,
        'most_common_words': most_common
    }

def main():
    parser = argparse.ArgumentParser(description="Analyze training data")
    parser.add_argument("--input", "-i", required=True, help="Input file or directory")
    parser.add_argument("--output", "-o", help="Output directory for plots")
    
    args = parser.parse_args()
    
    if os.path.isfile(args.input):
        # Analyze single file
        results = [analyze_text_file(args.input)]
    else:
        # Analyze directory
        results = []
        for file_path in Path(args.input).rglob("*.txt"):
            results.append(analyze_text_file(str(file_path)))
    
    # Print summary
    print("Data Analysis Summary:")
    print("=" * 50)
    
    total_chars = sum(r['characters'] for r in results)
    total_words = sum(r['words'] for r in results)
    total_sentences = sum(r['sentences'] for r in results)
    
    print(f"Total files analyzed: {len(results)}")
    print(f"Total characters: {total_chars:,}")
    print(f"Total words: {total_words:,}")
    print(f"Total sentences: {total_sentences:,}")
    print(f"Average words per sentence: {total_words / total_sentences:.2f}")
    
    # Show per-file statistics
    for result in results:
        print(f"\\nFile: {result['file']}")
        print(f"  Characters: {result['characters']:,}")
        print(f"  Words: {result['words']:,}")
        print(f"  Sentences: {result['sentences']:,}")
        print(f"  Unique words: {result['unique_words']:,}")
        print(f"  Avg sentence length: {result['avg_sentence_length']:.2f}")
        print(f"  Avg word length: {result['avg_word_length']:.2f}")
    
    # Generate plots if output directory specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        
        # Word frequency plot
        all_words = []
        for result in results:
            all_words.extend([word for word, _ in result['most_common_words']])
        
        word_freq = Counter(all_words)
        top_words = word_freq.most_common(20)
        
        plt.figure(figsize=(12, 6))
        words, counts = zip(*top_words)
        plt.bar(range(len(words)), counts)
        plt.xticks(range(len(words)), words, rotation=45)
        plt.title('Most Common Words')
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output, 'word_frequency.png'))
        plt.close()
        
        print(f"\\nPlots saved to: {args.output}")

if __name__ == "__main__":
    main()
`;
  
  await fs.writeFile(path.join(scriptsPath, 'analyze_data.py'), dataAnalysisContent);
}

module.exports = { createConfigFiles }; 