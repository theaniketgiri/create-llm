<div align="center">

# 🚀 create-llm

**The fastest way to start training your own Language Model**

Create production-ready LLM training projects in seconds. Like `create-next-app` but for training custom language models.

[![npm version](https://img.shields.io/npm/v/@theanikrtgiri/create-llm.svg)](https://www.npmjs.com/package/@theanikrtgiri/create-llm)
[![npm downloads](https://img.shields.io/npm/dm/@theanikrtgiri/create-llm.svg)](https://www.npmjs.com/package/@theanikrtgiri/create-llm)
[![GitHub stars](https://img.shields.io/github/stars/theaniketgiri/create-llm.svg)](https://github.com/theaniketgiri/create-llm/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub issues](https://img.shields.io/github/issues/theaniketgiri/create-llm.svg)](https://github.com/theaniketgiri/create-llm/issues)

[**📦 npm Package**](https://www.npmjs.com/package/@theanikrtgiri/create-llm) • [**📖 Documentation**](#documentation) • [**🐛 Report Bug**](https://github.com/theaniketgiri/create-llm/issues) • [**💡 Request Feature**](https://github.com/theaniketgiri/create-llm/issues)

</div>

```bash
npx @theanikrtgiri/create-llm my-awesome-llm
cd my-awesome-llm
pip install -r requirements.txt
python training/train.py
```

<div align="center">

**That's it! You're training an LLM.** ✨

</div>

---

## Why create-llm?

Training a language model from scratch is complex. You need:
- ✅ Model architecture (GPT, BERT, T5...)
- ✅ Data preprocessing pipeline
- ✅ Tokenizer training
- ✅ Training loop with callbacks
- ✅ Checkpoint management
- ✅ Evaluation metrics
- ✅ Text generation
- ✅ Deployment tools

**create-llm gives you all of this in one command.**

---

## Features

### 🎯 **Right-Sized Templates**
Choose from 4 templates optimized for different use cases:
- **NANO** (1M params) - Learn in 2 minutes on any laptop
- **TINY** (6M params) - Prototype in 15 minutes on CPU
- **SMALL** (100M params) - Production models in hours
- **BASE** (1B params) - Research-grade in days

### 🔧 **Complete Toolkit**
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

### 📊 **Smart Defaults**
Intelligent configuration that:
- Auto-detects vocab size from tokenizer
- Warns about model/data size mismatches
- Detects overfitting during training
- Suggests optimal hyperparameters
- Handles cross-platform paths

### 🎨 **Plugin System**
Optional integrations:
- **WandB** - Experiment tracking
- **HuggingFace** - Model sharing
- **SynthexAI** - Synthetic data generation

---

## Quick Start

### 🚀 One-Command Setup

```bash
# Using npx (recommended - no installation needed)
npx @theanikrtgiri/create-llm my-llm

# Or install globally
npm install -g @theanikrtgiri/create-llm
create-llm my-llm
```

### 🎯 Interactive Setup

```bash
npx @theanikrtgiri/create-llm
```

**You'll be prompted for:**
- 📝 Project name
- 🎯 Template (NANO, TINY, SMALL, BASE)
- 🔤 Tokenizer type (BPE, WordPiece, Unigram)
- 🔌 Optional plugins (WandB, HuggingFace, SynthexAI)

### ⚡ Quick Mode

```bash
# Specify everything upfront
npx @theanikrtgiri/create-llm my-llm --template tiny --tokenizer bpe --skip-install
```

---

## Templates

### 📦 NANO (NEW!)
**Perfect for learning and quick experiments**

```
Parameters: ~1M
Hardware:   Any CPU (2GB RAM)
Time:       1-2 minutes
Data:       100+ examples
Use:        Learning, testing, demos
```

**When to use:**
- First time training an LLM
- Quick experiments and testing
- Educational purposes
- Understanding the pipeline
- Limited data (100-1000 examples)

### 📦 TINY
**Perfect for prototyping and small projects**

```
Parameters: ~6M
Hardware:   CPU or basic GPU (4GB RAM)
Time:       5-15 minutes
Data:       1,000+ examples
Use:        Prototypes, small projects
```

**When to use:**
- Small-scale projects
- Limited data (1K-10K examples)
- Prototyping before scaling
- Personal experiments
- CPU-only environments

### 📦 SMALL
**Perfect for production applications**

```
Parameters: ~100M
Hardware:   RTX 3060+ (12GB VRAM)
Time:       1-3 hours
Data:       10,000+ examples
Use:        Production, real apps
```

**When to use:**
- Production applications
- Domain-specific models
- Real-world deployments
- Good data availability
- GPU available

### 📦 BASE
**Perfect for research and high-quality models**

```
Parameters: ~1B
Hardware:   A100 or multi-GPU
Time:       1-3 days
Data:       100,000+ examples
Use:        Research, high-quality
```

**When to use:**
- Research projects
- High-quality requirements
- Large datasets available
- Multi-GPU setup
- Competitive performance needed

---

## Complete Workflow

### 1️⃣ Create Your Project

```bash
npx @theanikrtgiri/create-llm my-llm --template tiny --tokenizer bpe
cd my-llm
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Add Your Data

Place your text files in `data/raw/`:

```bash
# Example: Download Shakespeare
curl https://www.gutenberg.org/files/100/100-0.txt > data/raw/shakespeare.txt

# Or add your own files
cp /path/to/your/data.txt data/raw/
```

<div align="center">

**💡 Pro Tip:** Start with at least 1MB of text for meaningful results

</div>

### 4️⃣ Train Tokenizer

```bash
python tokenizer/train.py --data data/raw/
```

<div align="center">

**🔤 This creates a vocabulary from your data**

</div>

### 5️⃣ Prepare Dataset

```bash
python data/prepare.py
```

<div align="center">

**📊 This tokenizes and prepares your data for training**

</div>

### 6️⃣ Start Training

```bash
# Basic training
python training/train.py

# With live dashboard (recommended!)
python training/train.py --dashboard
# Then open http://localhost:5000

# Resume from checkpoint
python training/train.py --resume checkpoints/checkpoint-1000.pt
```

<div align="center">

**📈 Watch your model learn in real-time!**

</div>

### 7️⃣ Evaluate Your Model

```bash
python evaluation/evaluate.py --checkpoint checkpoints/checkpoint-best.pt
```

### 8️⃣ Generate Text

```bash
python evaluation/generate.py \
  --checkpoint checkpoints/checkpoint-best.pt \
  --prompt "Once upon a time" \
  --temperature 0.8
```

<div align="center">

**✨ See your model's creativity in action!**

</div>

### 9️⃣ Interactive Chat

```bash
python chat.py --checkpoint checkpoints/checkpoint-best.pt
```

<div align="center">

**💬 Chat with your trained model!**

</div>

### 🔟 Deploy

```bash
# To Hugging Face
python deploy.py --to huggingface --repo-id username/my-model

# To Replicate
python deploy.py --to replicate --model-name my-model
```

<div align="center">

**🚀 Share your model with the world!**

</div>

---

## Project Structure

```
my-llm/
├── 📁 data/
│   ├── raw/              # Your training data goes here
│   ├── processed/        # Tokenized data (auto-generated)
│   ├── dataset.py        # PyTorch dataset classes
│   └── prepare.py        # Data preprocessing script
│
├── 📁 models/
│   ├── architectures/    # Model implementations
│   │   ├── gpt.py       # GPT architecture
│   │   ├── nano.py      # 1M parameter model
│   │   ├── tiny.py      # 6M parameter model
│   │   ├── small.py     # 100M parameter model
│   │   └── base.py      # 1B parameter model
│   ├── __init__.py
│   └── config.py        # Configuration loader
│
├── 📁 tokenizer/
│   ├── train.py         # Tokenizer training script
│   └── tokenizer.json   # Trained tokenizer (auto-generated)
│
├── 📁 training/
│   ├── train.py         # Main training script
│   ├── trainer.py       # Trainer class
│   ├── callbacks/       # Training callbacks
│   │   ├── base.py
│   │   ├── checkpoint.py
│   │   ├── logging.py
│   │   └── checkpoint_manager.py
│   └── dashboard/       # Live training dashboard
│       ├── dashboard_server.py
│       └── templates/
│
├── 📁 evaluation/
│   ├── evaluate.py      # Model evaluation
│   └── generate.py      # Text generation
│
├── 📁 plugins/          # Optional integrations
│   ├── wandb_plugin.py
│   ├── huggingface_plugin.py
│   └── synthex_plugin.py
│
├── 📁 checkpoints/      # Saved models (auto-generated)
├── 📁 logs/            # Training logs (auto-generated)
│
├── 📄 llm.config.js    # Main configuration file
├── 📄 requirements.txt # Python dependencies
├── 📄 chat.py         # Interactive chat interface
├── 📄 deploy.py       # Deployment script
├── 📄 compare.py      # Model comparison tool
└── 📄 README.md       # Project documentation
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
    optimizer: 'adamw',
    weight_decay: 0.01,
    gradient_clip: 1.0,
    mixed_precision: false,
    gradient_accumulation_steps: 1,
  },

  // Data settings
  data: {
    max_length: 512,
    stride: 256,
    val_split: 0.1,
    shuffle: true,
  },

  // Tokenizer settings
  tokenizer: {
    type: 'bpe',
    vocab_size: 10000,
    min_frequency: 2,
    special_tokens: ["<pad>", "<unk>", "<s>", "</s>"],
  },

  // Plugins
  plugins: [
    // 'wandb',
    // 'huggingface',
    // 'synthex',
  ],
};
```

---

## 📋 CLI Reference

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

## Advanced Features

### Live Training Dashboard

Monitor training in real-time with a web interface:

```bash
python training/train.py --dashboard
```

Then open http://localhost:5000 to see:
- Real-time loss curves
- Learning rate schedule
- Tokens per second
- GPU memory usage
- Recent checkpoints

### Model Comparison

Compare multiple trained models:

```bash
python compare.py checkpoints/model-v1/ checkpoints/model-v2/
```

Shows:
- Side-by-side metrics
- Sample generations
- Performance comparison
- Recommendation

### Checkpoint Management

Automatic checkpoint management:
- Saves best model based on validation loss
- Keeps last N checkpoints (configurable)
- Auto-saves on Ctrl+C
- Resume from any checkpoint

```bash
# Resume training
python training/train.py --resume checkpoints/checkpoint-5000.pt

# Evaluate specific checkpoint
python evaluation/evaluate.py --checkpoint checkpoints/checkpoint-best.pt
```

### Custom Plugins

Create your own plugins:

```python
# plugins/my_plugin.py
from plugins.base import BasePlugin

class MyPlugin(BasePlugin):
    def on_train_start(self, trainer):
        print("Training started!")
    
    def on_step_end(self, trainer, step, loss):
        # Log to your service
        pass
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

**Hyperparameter Tuning:**
- Learning rate: Start with 3e-4, adjust if unstable
- Batch size: As large as GPU allows
- Warmup steps: 10% of total steps
- Dropout: 0.1-0.3 depending on data size

### Deployment

**Before Deploying:**
- Evaluate on held-out test set
- Test generation quality
- Check model size
- Verify inference speed
- Test on target hardware

**Deployment Options:**
- Hugging Face Hub (easiest)
- Replicate (API endpoint)
- Docker container (custom)
- Cloud platforms (AWS, GCP, Azure)

---

## Troubleshooting

### Common Issues

**"Vocab size mismatch detected"**
- ✅ This is normal! The tool auto-detects and fixes it
- The model will use the actual tokenizer vocab size

**"Model may be too large for dataset"**
- ⚠️ Warning: Risk of overfitting
- Solutions: Add more data, use smaller template, increase dropout

**"Perplexity < 1.1 indicates severe overfitting"**
- ❌ Model memorized the data
- Solutions: Add much more data, use smaller model, increase regularization

**"CUDA out of memory"**
- Reduce `batch_size` in llm.config.js
- Enable `mixed_precision: true`
- Increase `gradient_accumulation_steps`
- Use smaller model template

**"Tokenizer not found"**
- Run `python tokenizer/train.py --data data/raw/` first
- Make sure data/raw/ contains .txt files

**"Training loss not decreasing"**
- Check learning rate (try 1e-4 to 1e-3)
- Verify data is loading correctly
- Check for data preprocessing issues
- Try longer warmup period

### Getting Help

- 📖 [Full Documentation](https://github.com/theaniketgiri/create-llm/docs)
- 💬 [Discord Community](https://discord.gg/create-llm)
- 🐛 [Report Issues](https://github.com/theaniketgiri/create-llm/issues)
- 📧 [Email Support](mailto:theaniketgiri@gmail.com)

---

## Requirements

### For CLI Tool
- Node.js 18.0.0 or higher
- npm 8.0.0 or higher

### For Training
- Python 3.8 or higher
- PyTorch 2.0.0 or higher
- 4GB RAM minimum (NANO/TINY)
- 12GB VRAM recommended (SMALL)
- 40GB+ VRAM for BASE

### Operating Systems
- ✅ Windows 10/11
- ✅ macOS 10.15+
- ✅ Linux (Ubuntu 20.04+)

---

## Development

### Setup Development Environment

```bash
git clone https://github.com/theaniketgiri/create-llm.git
cd create-llm
npm install
```

### Build

```bash
npm run build
```

### Development Mode

```bash
npm run dev
```

### Test Locally

```bash
node dist/index.js test-project --template nano
```

### Run Tests

```bash
npm test
```

### Publish

```bash
npm version patch  # or minor/major
npm publish
```

---

## 🤝 Contributing

<div align="center">

**We welcome contributions from everyone!**

[**📖 Contributing Guide**](CONTRIBUTING.md) • [**🐛 Report Bug**](https://github.com/theaniketgiri/create-llm/issues) • [**💡 Request Feature**](https://github.com/theaniketgiri/create-llm/issues)

</div>

### 🎯 Areas We Need Help

| Area | Description | Difficulty |
|------|-------------|------------|
| 🐛 **Bug Fixes** | Fix issues and improve stability | 🟢 Easy |
| 📝 **Documentation** | Improve guides and examples | 🟢 Easy |
| 🎨 **New Templates** | Add BERT, T5, custom architectures | 🟡 Medium |
| 🔌 **Plugins** | Integrate new services | 🟡 Medium |
| 🧪 **Testing** | Increase test coverage | 🟡 Medium |
| 🌍 **i18n** | Internationalization support | 🔴 Hard |

### 👥 Contributors

<div align="center">

Thanks to all contributors who have helped make this project better!

[![Contributors](https://contrib.rocks/image?repo=theaniketgiri/create-llm)](https://github.com/theaniketgiri/create-llm/graphs/contributors)

</div>

---

## Roadmap

### v1.1 (Next Release)
- [ ] More model architectures (BERT, T5)
- [ ] Distributed training support
- [ ] Model quantization tools
- [ ] Fine-tuning templates

### v1.2
- [ ] Web UI for project management
- [ ] Automatic hyperparameter tuning
- [ ] Model compression tools
- [ ] More deployment targets

### v2.0
- [ ] Multi-modal support
- [ ] Reinforcement learning from human feedback
- [ ] Advanced optimization techniques
- [ ] Cloud training integration

---

## 📄 License

MIT © [Aniket Giri](https://github.com/theaniketgiri)

See [LICENSE](LICENSE) for more information.

---

## Acknowledgments

Built with amazing open-source tools:
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Transformers](https://huggingface.co/transformers/) - Model implementations
- [Tokenizers](https://github.com/huggingface/tokenizers) - Fast tokenization
- [Commander.js](https://github.com/tj/commander.js) - CLI framework
- [Inquirer.js](https://github.com/SBoudrias/Inquirer.js) - Interactive prompts

Special thanks to the LLM community for inspiration and feedback.

---

## ⭐ Star History

<div align="center">

If you find this project useful, please consider giving it a star!

[![Star History Chart](https://api.star-history.com/svg?repos=theaniketgiri/create-llm&type=Date)](https://star-history.com/#theaniketgiri/create-llm&Date)

</div>

---

<div align="center">

**Made with ❤️ for the LLM community**

[**GitHub**](https://github.com/theaniketgiri/create-llm) • [**npm**](https://www.npmjs.com/package/@theanikrtgiri/create-llm) • [**Issues**](https://github.com/theaniketgiri/create-llm/issues) • [**Twitter**](https://twitter.com/theaniketgiri)

### 🙏 Support This Project

If create-llm helped you, consider:
- ⭐ Starring the repo
- 🐛 Reporting bugs
- 💡 Suggesting features
- 📝 Improving docs
- 🔀 Contributing code

**Together, let's make LLM training accessible to everyone!**

</div>
