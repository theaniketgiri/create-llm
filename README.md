# create-llm

```
                          _                  _  _              
  ___  _ __   ___   __ _ | |_   ___         | || | _ __ ___   
 / __|| '__| / _ \ / _` || __| / _ \  _____ | || || '_ ` _ \  
| (__ | |   |  __/| (_| || |_ |  __/ |_____|| || || | | | | | 
 \___||_|    \___| \__,_| \__| \___|        |_||_||_| |_| |_| 
                                                               
```

**CLI tool for scaffolding LLM Creation and training **

Create production-ready LLM training projects in seconds. Similar to create-next-app but for training custom language models.

[![npm version](https://img.shields.io/npm/v/@theanikrtgiri/create-llm.svg)](https://www.npmjs.com/package/@theanikrtgiri/create-llm)
[![npm downloads](https://img.shields.io/npm/dm/@theanikrtgiri/create-llm.svg)](https://www.npmjs.com/package/@theanikrtgiri/create-llm)
[![GitHub stars](https://img.shields.io/github/stars/theaniketgiri/create-llm.svg)](https://github.com/theaniketgiri/create-llm/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[npm Package](https://www.npmjs.com/package/@theanikrtgiri/create-llm) • [Documentation](#documentation) • [Report Bug](https://github.com/theaniketgiri/create-llm/issues) • [Request Feature](https://github.com/theaniketgiri/create-llm/issues)

```bash
npx @theanikrtgiri/create-llm my-awesome-llm
cd my-awesome-llm
pip install -r requirements.txt
python training/train.py
```

---

## Why create-llm?

Training a language model from scratch requires:
- Model architecture (GPT, BERT, T5...)
- Data preprocessing pipeline
- Tokenizer training
- Training loop with callbacks
- Checkpoint management
- Evaluation metrics
- Text generation
- Deployment tools

**create-llm provides all of this in one command.**

---

## Features

### Right-Sized Templates
Choose from 4 templates optimized for different use cases:
- **NANO** (1M params) - Learn in 2 minutes on any laptop
- **TINY** (6M params) - Prototype in 15 minutes on CPU
- **SMALL** (100M params) - Production models in hours
- **BASE** (1B params) - Research-grade in days

### Complete Toolkit
Everything you need out of the box:
- PyTorch training infrastructure
- Data preprocessing pipeline
- Tokenizer training (BPE, WordPiece, Unigram)
- Checkpoint management with auto-save
- TensorBoard integration
- Live training dashboard
- Interactive chat interface
- Model comparison tools
- Deployment scripts

### Smart Defaults
Intelligent configuration that:
- Auto-detects vocab size from tokenizer
- Automatically handles sequence length mismatches
- Warns about model/data size mismatches
- Detects overfitting during training
- Suggests optimal hyperparameters
- Handles cross-platform paths
- Provides detailed diagnostic messages for errors

### Plugin System
Optional integrations:
- **WandB** - Experiment tracking
- **HuggingFace** - Model sharing

---

## Quick Start

### One-Command Setup

```bash
# Using npx (recommended - no installation needed)
npx @theanikrtgiri/create-llm my-llm

# Or install globally
npm install -g @theanikrtgiri/create-llm
create-llm my-llm
```

### Interactive Setup

```bash
npx @theanikrtgiri/create-llm
```

You'll be prompted for:
- Project name
- Template (NANO, TINY, SMALL, BASE)
- Tokenizer type (BPE, WordPiece, Unigram)
- Optional plugins (WandB, HuggingFace)

### Quick Mode

```bash
# Specify everything upfront
npx create-llm my-llm --template tiny --tokenizer bpe --skip-install
```

---

## 🐳 Docker Support

**Run create-llm without installing Node.js or Python locally!**

### Quick Docker Setup

```bash
# Build the Docker image
git clone https://github.com/theaniketgiri/create-llm.git
cd create-llm
docker build -t create-llm .

# Create a new project
mkdir my-projects && cd my-projects
docker run -it -v $(pwd):/workspace create-llm scaffold my-llm --template tiny

# Train your model
cd my-llm
docker run --gpus all -v $(pwd):/workspace create-llm train

# Start chat interface
docker run -p 7860:7860 -v $(pwd):/workspace create-llm chat
```

### Docker Compose (Recommended)

```bash
# Clone and build
git clone https://github.com/theaniketgiri/create-llm.git
cd create-llm
make build

# Create project with interactive prompts
make compose-cli

# Train with GPU support
make compose-train

# Start chat interface at http://localhost:7860
make compose-chat

# Development environment
make dev
```

### Benefits

✅ **No Local Dependencies** - Skip Node.js and Python installation  
✅ **GPU Support** - Automatic NVIDIA GPU detection  
✅ **Consistent Environment** - Same setup across all machines  
✅ **Easy Scaling** - Run multiple training jobs in parallel  
✅ **Production Ready** - Deploy anywhere Docker runs  

📖 **Full Docker Guide**: See [DOCKER.md](DOCKER.md) for complete documentation

---

## Templates

### NANO
**For learning and quick experiments**

```
Parameters: ~1M
Hardware:   Any CPU (2GB RAM)
Time:       1-2 minutes
Data:       100+ examples
Use:        Learning, testing, demos
```

When to use:
- First time training an LLM
- Quick experiments and testing
- Educational purposes
- Understanding the pipeline
- Limited data (100-1000 examples)

### TINY
**For prototyping and small projects**

```
Parameters: ~6M
Hardware:   CPU or basic GPU (4GB RAM)
Time:       5-15 minutes
Data:       1,000+ examples
Use:        Prototypes, small projects
```

When to use:
- Small-scale projects
- Limited data (1K-10K examples)
- Prototyping before scaling
- Personal experiments
- CPU-only environments

### SMALL
**For production applications**

```
Parameters: ~100M
Hardware:   RTX 3060+ (12GB VRAM)
Time:       1-3 hours
Data:       10,000+ examples
Use:        Production, real apps
```

When to use:
- Production applications
- Domain-specific models
- Real-world deployments
- Good data availability
- GPU available

### BASE
**For research and high-quality models**

```
Parameters: ~1B
Hardware:   A100 or multi-GPU
Time:       1-3 days
Data:       100,000+ examples
Use:        Research, high-quality
```

When to use:
- Research projects
- High-quality requirements
- Large datasets available
- Multi-GPU setup
- Competitive performance needed

---

## Complete Workflow

### 1. Create Your Project

```bash
npx @theanikrtgiri/create-llm my-llm --template tiny --tokenizer bpe
cd my-llm
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add Your Data

Place your text files in `data/raw/`:

```bash
# Example: Download Shakespeare
curl https://www.gutenberg.org/files/100/100-0.txt > data/raw/shakespeare.txt

# Or add your own files
cp /path/to/your/data.txt data/raw/
```

**Tip:** Start with at least 1MB of text for meaningful results

### 4. Train Tokenizer

```bash
python tokenizer/train.py --data data/raw/
```

This creates a vocabulary from your data.

### 5. Prepare Dataset

```bash
python data/prepare.py
```

This tokenizes and prepares your data for training.

### 6. Start Training

```bash
# Basic training
python training/train.py

# With live dashboard
python training/train.py --dashboard
# Then open http://localhost:5000

# Resume from checkpoint
python training/train.py --resume checkpoints/checkpoint-1000.pt
```

### 7. Evaluate Your Model

```bash
python evaluation/evaluate.py --checkpoint checkpoints/checkpoint-best.pt
```

### 8. Generate Text

```bash
python evaluation/generate.py \
  --checkpoint checkpoints/checkpoint-best.pt \
  --prompt "Once upon a time" \
  --temperature 0.8
```

### 9. Interactive Chat

```bash
python chat.py --checkpoint checkpoints/checkpoint-best.pt
```

### 10. Deploy

```bash
# To Hugging Face
python deploy.py --to huggingface --repo-id username/my-model

# To Replicate
python deploy.py --to replicate --model-name my-model
```

---

## Project Structure

```
my-llm/
├── data/
│   ├── raw/              # Your training data goes here
│   ├── processed/        # Tokenized data (auto-generated)
│   ├── dataset.py        # PyTorch dataset classes
│   └── prepare.py        # Data preprocessing script
│
├── models/
│   ├── architectures/    # Model implementations
│   │   ├── gpt.py       # GPT architecture
│   │   ├── nano.py      # 1M parameter model
│   │   ├── tiny.py      # 6M parameter model
│   │   ├── small.py     # 100M parameter model
│   │   └── base.py      # 1B parameter model
│   ├── __init__.py
│   └── config.py        # Configuration loader
│
├── tokenizer/
│   ├── train.py         # Tokenizer training script
│   └── tokenizer.json   # Trained tokenizer (auto-generated)
│
├── training/
│   ├── train.py         # Main training script
│   ├── trainer.py       # Trainer class
│   ├── callbacks/       # Training callbacks
│   └── dashboard/       # Live training dashboard
│
├── evaluation/
│   ├── evaluate.py      # Model evaluation
│   └── generate.py      # Text generation
│
├── plugins/             # Optional integrations
├── checkpoints/         # Saved models (auto-generated)
├── logs/               # Training logs (auto-generated)
│
├── llm.config.js       # Main configuration file
├── requirements.txt    # Python dependencies
├── chat.py            # Interactive chat interface
├── deploy.py          # Deployment script
└── README.md          # Project documentation
```

---

## Configuration

Everything is controlled via `llm.config.js`:

```javascript
module.exports = {
  // Model architecture
  model: {
    type: 'gpt',
    size: 'tiny',
    vocab_size: 10000,      // Auto-detected from tokenizer
    max_length: 512,
    layers: 4,
    heads: 4,
    dim: 256,
    dropout: 0.2,
  },

  // Training settings
  training: {
    batch_size: 16,
    learning_rate: 0.0006,
    warmup_steps: 500,
    max_steps: 10000,
    eval_interval: 500,
    save_interval: 2000,
  },

  // Plugins
  plugins: [
    // 'wandb',
    // 'huggingface',
  ],
};
```

---

## CLI Reference

### Commands

```bash
npx @theanikrtgiri/create-llm [project-name] [options]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--template <name>` | Template to use (nano, tiny, small, base, custom) | Interactive |
| `--tokenizer <type>` | Tokenizer type (bpe, wordpiece, unigram) | Interactive |
| `--skip-install` | Skip npm/pip installation | false |
| `-y, --yes` | Skip all prompts, use defaults | false |
| `-h, --help` | Show help | - |
| `-v, --version` | Show version | - |

### Examples

```bash
# Interactive mode (recommended for first time)
npx @theanikrtgiri/create-llm

# Quick start with defaults
npx @theanikrtgiri/create-llm my-project

# Specify everything
npx @theanikrtgiri/create-llm my-project --template nano --tokenizer bpe --skip-install

# Skip prompts
npx @theanikrtgiri/create-llm my-project -y
```

---

## Best Practices

### Data Preparation

**Minimum Data Requirements:**
- NANO: 100+ examples (good for learning)
- TINY: 1,000+ examples (minimum for decent results)
- SMALL: 10,000+ examples (recommended)
- BASE: 100,000+ examples (for quality)

**Data Quality:**
- Use clean, well-formatted text
- Remove HTML, markdown, or special formatting
- Ensure consistent encoding (UTF-8)
- Remove duplicates
- Balance different content types

### Training Tips

**Avoid Overfitting:**
- Watch for perplexity < 1.5 (warning sign)
- Use validation split (10% recommended)
- Increase dropout if overfitting
- Add more data if possible
- Use smaller model for small datasets

**Optimize Training:**
- Start with NANO to test pipeline
- Use mixed precision on GPU (`mixed_precision: true`)
- Increase `gradient_accumulation_steps` if OOM
- Monitor GPU usage with dashboard
- Save checkpoints frequently

---

## Troubleshooting

### Common Issues

**"Vocab size mismatch detected"**
- This is normal. The tool auto-detects and fixes it.
- The model will use the actual tokenizer vocab size.

**"Position embedding index error" or sequences too long**
- Automatically handled. Sequences exceeding max_length are truncated.
- The model logs warnings when truncation occurs.
- Check your data preprocessing if you see frequent truncation warnings.
- Consider increasing `max_length` in config if you need longer sequences.

**"Model may be too large for dataset"**
- Warning: Risk of overfitting
- Solutions: Add more data, use smaller template, increase dropout

**"CUDA out of memory"**
- Reduce `batch_size` in llm.config.js
- Enable `mixed_precision: true`
- Increase `gradient_accumulation_steps`
- Use smaller model template

**"Training loss not decreasing"**
- Check learning rate (try 1e-4 to 1e-3)
- Verify data is loading correctly
- Check for data preprocessing issues
- Try longer warmup period

### Getting Help

- [Full Documentation](https://github.com/theaniketgiri/create-llm/docs)
- [Report Issues](https://github.com/theaniketgiri/create-llm/issues)
- [Email Support](mailto:theaniketgiri@gmail.com)

---

## Requirements

### Option 1: Local Installation

**For CLI Tool**
- Node.js 18.0.0 or higher
- npm 8.0.0 or higher

**For Training**
- Python 3.8 or higher
- PyTorch 2.0.0 or higher
- 4GB RAM minimum (NANO/TINY)
- 12GB VRAM recommended (SMALL)
- 40GB+ VRAM for BASE

### Option 2: Docker (Recommended)

**No local dependencies needed!**
- Docker 20.10+ or Docker Desktop
- NVIDIA Docker (for GPU support)
- 8GB RAM minimum
- NVIDIA GPU (optional, for faster training)

### Operating Systems
- Windows 10/11
- macOS 10.15+
- Linux (Ubuntu 20.04+)

---

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for development setup and guidelines.

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas We Need Help

| Area | Description | Difficulty |
|------|-------------|------------|
| Bug Fixes | Fix issues and improve stability | Easy |
| Documentation | Improve guides and examples | Easy |
| New Templates | Add BERT, T5, custom architectures | Medium |
| Plugins | Integrate new services | Medium |
| Testing | Increase test coverage | Medium |
| i18n | Internationalization support | Hard |

---

## License

MIT © [Aniket Giri](https://github.com/theaniketgiri)

See [LICENSE](LICENSE) for more information.

---

## Acknowledgments

Built with:
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Transformers](https://huggingface.co/transformers/) - Model implementations
- [Tokenizers](https://github.com/huggingface/tokenizers) - Fast tokenization
- [Commander.js](https://github.com/tj/commander.js) - CLI framework
- [Inquirer.js](https://github.com/SBoudrias/Inquirer.js) - Interactive prompts

---

If you find this project useful, please consider giving it a star!

[GitHub](https://github.com/theaniketgiri/create-llm) • [npm](https://www.npmjs.com/package/@theanikrtgiri/create-llm) • [Issues](https://github.com/theaniketgiri/create-llm/issues)
