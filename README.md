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
```

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

[Star us on GitHub](https://github.com/theaniketgiri/create-llm) | [Follow us on Twitter](https://twitter.com/create_llm) 