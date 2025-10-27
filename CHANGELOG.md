# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.1] - 2025-10-26

### Fixed

#### Position Embedding Bug Fix
- **Critical Fix**: Resolved `IndexError: index out of range in self` that occurred when validation sequences exceeded model's max_length
- Added automatic sequence truncation in GPT model forward method with warning logs
- Implemented max_length parameter in DataLoader with custom collate function
- Added position embedding size validation in checkpoint loading
- Enhanced error handling with detailed diagnostic messages and actionable solutions
- Fixed tensor contiguity issues by using `reshape()` instead of `view()` for loss calculation

#### Improvements
- Added model configuration logging in Evaluator class (displays max_length, vocab_size, position embedding size)
- Evaluation script now extracts and uses max_length from loaded model
- Enhanced error messages provide clear guidance on fixing sequence length issues
- Added comprehensive unit and integration tests for sequence length validation

#### Documentation
- Updated README with position embedding troubleshooting section
- Added implementation guide for applying fixes to existing projects
- Created detailed test results documentation
- Updated smart defaults section to mention automatic sequence length handling

### Testing
- Added 3 unit tests for sequence length validation
- Added 5 integration tests for evaluation with various sequence lengths
- All tests pass successfully with sequences at, exceeding, and far beyond max_length

## [1.0.0] - 2025-01-24

### Added

#### Templates
- **NANO Template** (1M params) - For learning and quick experiments
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

### Technical Details

- Node.js 18+ required
- Python 3.8+ required
- PyTorch 2.0+ required
- Cross-platform support (Windows, macOS, Linux)

### Bug Fixes

- Fixed data loading with 2D tensors
- Fixed vocab size mismatch (32K to auto-detect)
- Fixed Windows UTF-8 encoding issues
- Fixed deploy.py unicode escape errors
- Fixed chat.py cross-platform path handling
- Fixed model forward method to accept attention_mask

### Known Issues

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

- **2.0.1** (2025-10-26) - Position embedding bug fix
- **1.0.0** (2025-01-24) - Initial release
- **0.1.0** (2025-01-20) - Beta release (internal)

---

For more details, see the [full commit history](https://github.com/theaniketgiri/create-llm/commits/main).
