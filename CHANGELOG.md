# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-24

### 🎉 Initial Release

The first production-ready release of create-llm!

### ✨ Added

#### Templates
- **NANO Template** (1M params) - Perfect for learning and quick experiments
- **TINY Template** (6M params) - For prototyping and small projects
- **SMALL Template** (100M params) - For production applications
- **BASE Template** (1B params) - For research and high-quality models

#### Core Features
- Complete PyTorch training infrastructure
- Data preprocessing pipeline
- Tokenizer training (BPE, WordPiece, Unigram)
- Checkpoint management with auto-save
- TensorBoard integration
- Live training dashboard
- Interactive chat interface
- Model comparison tools
- Deployment scripts

#### Smart Features
- Automatic vocab size detection from tokenizer
- Model/data size mismatch warnings
- Overfitting detection during training
- Cross-platform path handling
- UTF-8 encoding support for Windows

#### Plugins
- WandB integration for experiment tracking
- HuggingFace Hub integration for model sharing
- SynthexAI integration for synthetic data

#### Documentation
- Comprehensive README with examples
- Detailed project READMEs for scaffolded projects
- Contributing guidelines
- Troubleshooting guides

### 🔧 Technical Details

- Node.js 18+ required
- Python 3.8+ required
- PyTorch 2.0+ required
- Cross-platform support (Windows, macOS, Linux)

### 🐛 Bug Fixes

- Fixed data loading with 2D tensors
- Fixed vocab size mismatch (32K → auto-detect)
- Fixed Windows UTF-8 encoding issues
- Fixed deploy.py unicode escape errors
- Fixed chat.py cross-platform path handling
- Fixed model forward method to accept attention_mask

### 📝 Known Issues

- Dashboard may show garbled emojis in Windows PowerShell (functionality works)
- PyTorch FutureWarning about torch.load (will be addressed in PyTorch 2.x)

---

## [Unreleased]

### Planned Features

- More model architectures (BERT, T5)
- Distributed training support
- Model quantization tools
- Fine-tuning templates
- Web UI for project management
- Automatic hyperparameter tuning

---

## Version History

- **1.0.0** (2025-01-24) - Initial release
- **0.1.0** (2025-01-20) - Beta release (internal)

---

For more details, see the [full commit history](https://github.com/theaniketgiri/create-llm/commits/main).
