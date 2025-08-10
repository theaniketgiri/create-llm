# Create LLM 🚀

> **Create a custom Large Language Model from scratch - like create-react-app for language models**

[![npm version](https://badge.fury.io/js/create-llm.svg)](https://badge.fury.io/js/create-llm)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Create LLM is a framework that allows any developer or researcher to easily create, train, and evaluate a custom Large Language Model from scratch — without writing all the complex boilerplate code themselves.

## 🎯 Mission

**Democratize LLM creation** — make it accessible to students, indie hackers, startups, and researchers. Just like React made UI development easy, we want to make LLM creation easy.

## ✨ Features

- **🚀 One-command setup** - Get a complete LLM project in under 5 minutes
- **🏗️ Multiple architectures** - GPT, Mistral, RWKV, Mixtral support
- **🔤 Custom tokenizers** - BPE, WordPiece, Unigram training
- **📊 Built-in datasets** - WikiText, C4, OpenWebText support
- **⚡ Production-ready** - Mixed precision, distributed training, checkpointing
- **📈 Comprehensive evaluation** - Perplexity, accuracy, BLEU, ROUGE metrics
- **🎨 Synthetic data generation** - Generate training data programmatically
- **📝 Interactive generation** - Chat with your trained model

## 🚀 Quick Start

### Install

```bash
npm install -g create-llm
```

### Create a new LLM project

```bash
npx create-llm my-awesome-llm
```

### Follow the interactive prompts

```
🚀 Create LLM - Build Your Custom Language Model

? What is your project named? my-awesome-llm
? Which model architecture would you like to use? GPT-2 Style Transformer
? Which tokenizer would you like to use? BPE (Byte Pair Encoding)
? Which dataset would you like to use for training? WikiText-103
? Would you like to use TypeScript? No
? Would you like to include synthetic data generation capabilities? Yes

📁 Your project structure:
  my-awesome-llm/
  ├── model/               # Transformer architecture
  ├── tokenizer/           # Tokenizer scripts
  ├── data/                # Dataset preprocessing
  ├── training/            # Training pipeline
  ├── eval/                # Evaluation scripts
  ├── checkpoints/         # Model checkpoints
  ├── logs/                # Training logs
  └── README.md            # Project documentation
```

### Start training

```bash
cd my-awesome-llm

# Setup environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Train tokenizer
python tokenizer/train_tokenizer.py --input data/raw.txt

# Prepare dataset
python data/prepare_dataset.py

# Start training
python training/train.py --config training/config.yaml
```

## 📁 Project Structure

```
my-llm/
├── model/                 # Transformer architecture
│   ├── __init__.py
│   ├── config.py         # Model configuration
│   ├── transformer.py    # Main transformer model
│   ├── attention.py      # Multi-head attention
│   └── feed_forward.py   # Feed-forward networks
├── tokenizer/            # Tokenizer training
│   ├── __init__.py
│   ├── train_tokenizer.py
│   ├── train_multiple.py
│   └── tokenizer.py      # Custom tokenizer class
├── data/                 # Dataset processing
│   ├── __init__.py
│   ├── dataset.py        # PyTorch datasets
│   ├── prepare_dataset.py
│   ├── dataloader.py
│   └── download_dataset.py
├── training/             # Training pipeline
│   ├── __init__.py
│   ├── train.py          # Main training script
│   ├── config.py         # Training configuration
│   ├── trainer.py        # Trainer class
│   └── config.yaml       # Default config
├── eval/                 # Evaluation
│   ├── __init__.py
│   ├── run_eval.py       # Evaluation script
│   ├── evaluator.py      # Evaluator class
│   ├── metrics.py        # Evaluation metrics
│   └── generate.py       # Text generation
├── scripts/              # Utility scripts
│   ├── generate_synthetic_data.py
│   ├── model_info.py
│   └── analyze_data.py
├── checkpoints/          # Model checkpoints
├── logs/                 # Training logs
├── requirements.txt      # Python dependencies
├── setup.py             # Setup script
└── README.md            # Project documentation
```

## 🏗️ Supported Architectures

### GPT-2 Style Transformer
- Standard decoder-only transformer
- Causal attention masking
- Configurable layers, heads, dimensions

### Mistral (7B style)
- Sliding window attention
- Optimized for long sequences
- Reduced memory usage

### RWKV (RNN-style)
- Linear attention mechanism
- Memory efficient
- Fast inference

### Mixtral (MoE)
- Mixture of Experts
- Sparse activation
- High capacity, efficient training

## 🔤 Tokenizer Options

### BPE (Byte Pair Encoding)
- Most common choice
- Good for most languages
- Efficient vocabulary

### WordPiece
- Similar to BPE
- Used by BERT models
- Good for multilingual

### Unigram
- Probabilistic tokenization
- Language agnostic
- Good for unknown languages

## 📊 Dataset Support

### Built-in Datasets
- **WikiText-103** - Wikipedia articles (recommended for beginners)
- **C4** - Common Crawl (large, diverse)
- **OpenWebText** - Web content (very large)

### Custom Data
- Add your own text files
- Automatic preprocessing
- Support for multiple formats

### Synthetic Data Generation
```bash
# Generate synthetic training data
python scripts/generate_synthetic_data.py --type code --size 10000
python scripts/generate_synthetic_data.py --type medical --size 5000
```

## ⚙️ Configuration

Edit `training/config.yaml` to customize your training:

```yaml
# Model architecture
n_embd: 768
n_layer: 12
n_head: 12

# Training hyperparameters
batch_size: 8
learning_rate: 3e-4
max_steps: 100000

# Data
max_length: 1024
train_data_path: "data/processed/train.txt"
val_data_path: "data/processed/validation.txt"

# Optimization
optimizer: "adamw"
scheduler: "cosine"
fp16: true

# Checkpointing
save_steps: 1000
eval_steps: 500
```

## 🚀 Training

### Complete Training Workflow

Follow this complete workflow to ensure your model is properly trained:

```bash
# 1. Download a substantial dataset (if not using your own data)
python data/download_dataset.py --dataset wikitext

# 2. Train your tokenizer on the raw data
python tokenizer/train_tokenizer.py --input data/raw.txt --vocab_size 50000

# 3. Preprocess the dataset
python data/prepare_dataset.py

# 4. Verify your data is properly processed
head -n 20 data/processed/train.txt

# 5. Start training with proper configuration
python training/train.py --config training/config.yaml
```

> **Important:** The default `train.py` script should use the `Trainer` class from `trainer.py`. If your model isn't learning, check that your training script is actually training the model and not just initializing it.

### Basic Training
```bash
python training/train.py --config training/config.yaml
```

### Resume Training
```bash
python training/train.py --config training/config.yaml --resume checkpoints/latest.pt
```

### Distributed Training
```bash
# Multi-GPU training
torchrun --nproc_per_node=4 training/train.py --config training/config.yaml
```

### Monitor Training
```bash
# TensorBoard
tensorboard --logdir logs/

# View logs
tail -f logs/training.log

# Check training loss trend
grep "loss" logs/training.log | tail -n 50
```

### Verifying Training Progress

To confirm your model is actually training:

1. Check that loss is decreasing over time in logs
2. Verify model checkpoints are being saved regularly
3. Test intermediate checkpoints with the generation script
4. Monitor training metrics in TensorBoard

## 📈 Evaluation

### Evaluate Model
```bash
python eval/run_eval.py --model checkpoints/best.pt --data data/processed/test.txt
```

### Generate Text
```bash
# Single generation
python eval/generate.py --model checkpoints/best.pt --prompt "Hello, world!"

# Interactive mode
python eval/generate.py --model checkpoints/best.pt --interactive
```

### Compute Metrics
- **Perplexity** - Language modeling quality
- **Accuracy** - Next token prediction
- **BLEU/ROUGE** - Text generation quality
- **Diversity** - Vocabulary usage
- **Fluency** - Text coherence

## 🛠️ Advanced Usage

### Custom Model Architecture
```python
from model import ModelConfig, TransformerLM

# Custom configuration
config = ModelConfig(
    vocab_size=50000,
    n_embd=1024,
    n_layer=24,
    n_head=16,
    model_type="gpt"
)

model = TransformerLM(config)
```

### Custom Training Loop
```python
from training.trainer import Trainer
from data import create_dataloaders

# Create dataloaders
train_dataloader, val_dataloader = create_dataloaders(
    tokenizer=tokenizer,
    train_path="data/train.txt",
    val_path="data/val.txt",
    batch_size=16
)

# Create trainer
trainer = Trainer(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    config=config,
    device=device
)

# Train
trainer.train()
```

### Custom Tokenizer
```python
from tokenizer import CustomTokenizer

# Load tokenizer
tokenizer = CustomTokenizer("tokenizer/tokenizer.json")

# Encode text
tokens = tokenizer.encode("Hello, world!")
text = tokenizer.decode(tokens)
```

## 📊 Performance

### Model Sizes
| Architecture | Parameters | Memory (MB) | Training Time* |
|--------------|------------|-------------|----------------|
| GPT-2 Small  | 124M       | 500         | 2-4 hours      |
| GPT-2 Medium | 355M       | 1.4GB       | 6-12 hours     |
| GPT-2 Large  | 774M       | 3.1GB       | 12-24 hours    |

*Training time on single V100 GPU with WikiText-103

### Hardware Requirements
- **Minimum**: 8GB GPU RAM
- **Recommended**: 16GB+ GPU RAM
- **Multi-GPU**: 2-8 GPUs for faster training

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/theaniketgiri/create-llm.git
cd create-llm
npm install
npm link
```

### Running Tests
```bash
npm test
```

## 📚 Documentation

- [Getting Started Guide](docs/getting-started.md)
- [Model Architectures](docs/architectures.md)
- [Training Guide](docs/training.md)
- [Evaluation Guide](docs/evaluation.md)
- [API Reference](docs/api.md)

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/theaniketgiri/create-llm/issues)
- **Discussions**: [GitHub Discussions](https://github.com/theaniketgiri/create-llm/discussions)
- **Discord**: [Join our community](https://discord.gg/your-invite)

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Hugging Face](https://huggingface.co/) - Tokenizers and datasets
- [Transformers](https://github.com/huggingface/transformers) - Model architectures
- [OpenAI](https://openai.com/) - GPT architecture inspiration

## 🔍 Troubleshooting & Debugging

### Common Issues

#### Model Not Generating Meaningful Text

**Symptoms:**
- Model generates very short or nonsensical responses
- Output doesn't extend beyond the prompt
- Text generation stops after a few tokens

**Possible Causes & Solutions:**

1. **Insufficient Training:**
   - The initial model (`initial_model.pt`) is only initialized and not trained
   - Solution: Run the full training process with `python training/train.py`
   - Ensure training completes several epochs (check logs/training.log)

2. **Limited Training Data:**
   - Check your data files in `data/raw.txt` or `data/processed/train.txt`
   - Solution: Download a larger dataset using `python data/download_dataset.py`
   - For WikiText-103: `python data/download_dataset.py --dataset wikitext`
   - For OpenWebText: Follow instructions in download_dataset.py

3. **Incorrect Training Configuration:**
   - Check `training/config.yaml` for proper hyperparameters
   - Ensure `train.py` is using the `Trainer` class from `trainer.py`
   - Verify that the model is being saved after actual training

4. **Tokenizer Issues:**
   - Ensure tokenizer is properly trained: `python tokenizer/train_tokenizer.py`
   - Check if the same tokenizer is used for both training and generation

#### Common Error Messages

**1. "ModuleNotFoundError: No module named 'X'"**
- **Cause**: Missing Python dependency
- **Solution**: `pip install -r requirements.txt` or `pip install X`

**2. "CUDA out of memory"**
- **Cause**: Model or batch size too large for your GPU
- **Solution**: Reduce `batch_size` or `n_embd` in config.yaml, or enable gradient accumulation

**3. "ValueError: Tokenizer file not found"**
- **Cause**: Tokenizer not trained or path incorrect
- **Solution**: Run `python tokenizer/train_tokenizer.py` first

**4. "FileNotFoundError: data/processed/train.txt"**
- **Cause**: Dataset not preprocessed
- **Solution**: Run `python data/prepare_dataset.py`

**5. "RuntimeError: Expected tensor for argument #1 'indices' to have scalar type Long"**
- **Cause**: Data type mismatch, often in tokenized data
- **Solution**: Check data preprocessing and ensure tokenizer outputs correct types

**6. "ImportError: cannot import name 'X' from 'Y'"**
- **Cause**: Incorrect project structure or missing implementation
- **Solution**: Check file structure matches the expected project layout

### Debugging Steps

1. **Verify Data Processing:**
   ```bash
   # Check raw data size
   wc -l data/raw.txt
   
   # Check processed data
   head -n 20 data/processed/train.txt
   ```

2. **Monitor Training Progress:**
   ```bash
   # Check training logs
   tail -f logs/training.log
   
   # Monitor with TensorBoard
   tensorboard --logdir logs/
   ```

3. **Inspect Model Checkpoints:**
   ```bash
   # List available checkpoints
   ls -la checkpoints/
   
   # Check model info
   python scripts/model_info.py --model checkpoints/latest.pt
   ```

4. **Test Generation with Different Parameters:**
   ```bash
   # Try different temperature values
   python eval/generate.py --model checkpoints/latest.pt --prompt "Hello" --temperature 0.7
   
   # Try different sampling methods
   python eval/generate.py --model checkpoints/latest.pt --prompt "Hello" --top_p 0.9 --top_k 40
   ```

### Environment and Python Path Issues

#### Setting the Python Path

If you encounter import errors when running scripts, you may need to set the Python path to include your project directory:

**Linux/Mac:**
```bash
export PYTHONPATH=/path/to/your/project:$PYTHONPATH
python eval/generate.py --model checkpoints/latest.pt --prompt "Hello"
```

**Windows (PowerShell):**
```powershell
$env:PYTHONPATH="C:\path\to\your\project"; python eval/generate.py --model checkpoints/latest.pt --prompt "Hello"
```

**Windows (CMD):**
```cmd
set PYTHONPATH=C:\path\to\your\project
python eval/generate.py --model checkpoints/latest.pt --prompt "Hello"
```

#### Virtual Environment Setup

It's recommended to use a virtual environment to avoid dependency conflicts:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### GPU Setup

To verify your GPU is properly configured for PyTorch:

```python
# Run this in Python to check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

## 🗺️ Roadmap

- [ ] JAX/Flax backend support
- [ ] More model architectures (LLaMA, PaLM)
- [ ] Web UI for training monitoring
- [ ] Model serving and deployment
- [ ] Fine-tuning support
- [ ] Multi-modal models
- [ ] Cloud training integration

---

**Made with ❤️ by the Create LLM community**

[Star us on GitHub](https://github.com/theaniketgiri/create-llm) | [Follow us on Twitter](https://twitter.com/theaniketgiri)