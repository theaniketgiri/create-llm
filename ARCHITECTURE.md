# System Architecture - create-llm

## Overview

`create-llm` is a CLI tool that scaffolds production-ready LLM training projects. It's built with TypeScript and generates Python-based training infrastructure.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     CLI Interface (Node.js)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Prompts    │  │   Config     │  │  Templates   │      │
│  │   (Inquirer) │  │  Generator   │  │   Manager    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Scaffolder Engine                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Python Code Generation (Template System)            │   │
│  │  • Model architectures (GPT, Transformer blocks)     │   │
│  │  • Training infrastructure (Trainer, callbacks)      │   │
│  │  • Data pipeline (Dataset, DataLoader)               │   │
│  │  • Evaluation & generation scripts                   │   │
│  │  • Tokenizer training                                │   │
│  │  • Plugin system (WandB, HuggingFace, SynthexAI)    │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Generated Python Project                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Models  │  │ Training │  │   Data   │  │   Utils  │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. CLI Layer (TypeScript)


**Location**: `src/`

**Components**:
- `index.ts` - Entry point, CLI argument parsing
- `prompts.ts` - Interactive user prompts (Inquirer.js)
- `config-generator.ts` - Generates llm.config.js files
- `template-manager.ts` - Manages template selection and validation
- `scaffolder.ts` - Main scaffolding engine

**Responsibilities**:
- Parse CLI arguments
- Collect user preferences interactively
- Validate inputs
- Orchestrate project generation
- Display progress and results

### 2. Template System

**Location**: `src/*-templates.ts`

**Components**:
- `python-templates.ts` - Core model architectures (GPT, config loaders)
- `python-trainer-templates.ts` - Training loop, optimizer, scheduler
- `python-dataset-templates.ts` - Data loading and preprocessing
- `python-test-templates.ts` - Unit and integration tests
- `python-callback-templates.ts` - Training callbacks (checkpointing, logging)
- `python-dashboard-templates.ts` - Live training dashboard
- `python-tokenizer-templates.ts` - Tokenizer training (BPE, WordPiece, Unigram)
- `python-plugin-templates.ts` - Plugin integrations
- `python-error-templates.ts` - Error handling utilities

**Responsibilities**:
- Generate Python code from templates
- Inject configuration values
- Handle template variations based on user choices
- Ensure code consistency and best practices

### 3. Configuration System

**Location**: `templates/*.json`

**Templates**:
- `nano.json` - 1M parameters (learning/prototyping)
- `tiny.json` - 6M parameters (small projects)
- `small.json` - 100M parameters (production)
- `base.json` - 1B parameters (research)
- `custom.json` - User-defined configuration

**Structure**:
```json
{
  "name": "template-name",
  "config": {
    "model": { /* architecture params */ },
    "training": { /* training hyperparams */ },
    "data": { /* data processing params */ },
    "tokenizer": { /* tokenizer config */ },
    "hardware": { /* hardware requirements */ }
  }
}
```

## Generated Project Architecture

### Directory Structure


```
project-name/
├── models/
│   ├── architectures/
│   │   ├── gpt.py              # GPT model implementation
│   │   ├── nano.py             # NANO template
│   │   ├── tiny.py             # TINY template
│   │   ├── small.py            # SMALL template
│   │   └── base.py             # BASE template
│   └── config.py               # Configuration loader
├── training/
│   ├── train.py                # Main training script
│   └── callbacks.py            # Training callbacks
├── data/
│   ├── dataset.py              # Dataset and DataLoader
│   ├── prepare.py              # Data preprocessing
│   └── raw/                    # Raw training data
├── tokenizer/
│   ├── train.py                # Tokenizer training
│   └── tokenizer.json          # Trained tokenizer
├── evaluation/
│   ├── evaluate.py             # Model evaluation
│   └── generate.py             # Text generation
├── utils/
│   ├── exceptions.py           # Custom exceptions
│   └── handlers.py             # Error handlers
├── plugins/                    # Optional plugins
├── tests/                      # Unit and integration tests
├── llm.config.js               # Project configuration
└── README.md                   # Project documentation
```

### Data Flow

```
Raw Text Data
    │
    ▼
Tokenizer Training ──► tokenizer.json
    │
    ▼
Data Preprocessing ──► train.pt, val.pt
    │
    ▼
Training Loop
    │
    ├──► Checkpoints (checkpoint-*.pt)
    ├──► Logs (training.log)
    └──► TensorBoard events
    │
    ▼
Evaluation ──► Metrics (loss, perplexity)
    │
    ▼
Generation/Chat ──► Generated text
```

## Key Design Patterns

### 1. Template Method Pattern
- Base templates define structure
- Specific templates override details
- Consistent interface across all templates

### 2. Factory Pattern
- `create_gpt_model()` functions
- Template-specific model creation
- Centralized configuration

### 3. Strategy Pattern
- Pluggable tokenizers (BPE, WordPiece, Unigram)
- Pluggable optimizers (Adam, AdamW, SGD)
- Pluggable callbacks

### 4. Observer Pattern
- Training callbacks observe training events
- Dashboard updates on training progress
- Logging and checkpointing

## Critical Components

### 1. GPT Model (python-templates.ts)


**Architecture**:
```
GPTModel
├── Token Embedding (vocab_size → dim)
├── Position Embedding (max_length → dim)
├── Transformer Blocks (N layers)
│   ├── Multi-Head Attention
│   ├── Feed-Forward Network
│   ├── Layer Normalization
│   └── Residual Connections
└── Language Model Head (dim → vocab_size)
```

**Key Features**:
- Automatic sequence truncation (fixes position embedding bug)
- Weight tying between embeddings and output
- Causal attention masking
- Dropout regularization

### 2. Training Pipeline (python-trainer-templates.ts)

**Flow**:
```
Initialize Model & Optimizer
    │
    ▼
For each epoch:
    │
    ├──► Training Loop
    │    ├── Forward pass
    │    ├── Loss calculation
    │    ├── Backward pass
    │    ├── Gradient clipping
    │    └── Optimizer step
    │
    ├──► Evaluation (periodic)
    │    ├── Validation loss
    │    ├── Perplexity
    │    └── Overfitting detection
    │
    └──► Checkpointing
         ├── Save best model
         ├── Save periodic checkpoints
         └── Save final model
```

### 3. Data Pipeline (python-dataset-templates.ts)

**Processing**:
```
Raw Text
    │
    ▼
Tokenization ──► Token IDs
    │
    ▼
Chunking (max_length, stride) ──► Sequences
    │
    ▼
Dataset (PyTorch) ──► Batches
    │
    ▼
DataLoader (with truncation) ──► Training
```

**Key Features**:
- Automatic sequence truncation
- Custom collate function
- Efficient batching
- Train/validation split

## Error Handling & Validation

### 1. Configuration Validation
- Template validation (model size, hyperparameters)
- Hardware requirement checks
- Data path validation
- Tokenizer compatibility

### 2. Runtime Validation
- Vocab size mismatch detection (auto-fix)
- Position embedding size validation
- Sequence length validation (auto-truncate)
- Overfitting detection

### 3. Error Recovery
- Graceful degradation
- Informative error messages
- Suggested solutions
- Automatic fixes where possible

## Plugin System

### Architecture


```
Plugin Interface
    │
    ├──► WandB Plugin
    │    ├── Experiment tracking
    │    ├── Metric logging
    │    └── Model versioning
    │
    ├──► HuggingFace Plugin
    │    ├── Model upload
    │    ├── Model card generation
    │    └── Repository management
    │
    └──► SynthexAI Plugin
         ├── Synthetic data generation
         ├── Data augmentation
         └── Quality filtering
```

**Integration Points**:
- Training callbacks
- Data preprocessing
- Model deployment
- Evaluation metrics

## Testing Strategy

### 1. Unit Tests
- Model architecture tests
- Configuration validation tests
- Data pipeline tests
- Utility function tests

### 2. Integration Tests
- End-to-end training tests
- Evaluation workflow tests
- Generation tests
- Plugin integration tests

### 3. Template Tests
- Template generation tests
- Code syntax validation
- Configuration consistency tests

## Performance Considerations

### 1. Memory Optimization
- Gradient accumulation
- Mixed precision training
- Efficient data loading
- Checkpoint management

### 2. Speed Optimization
- DataLoader workers
- GPU utilization
- Batch size tuning
- Gradient checkpointing

### 3. Scalability
- Template-based sizing
- Hardware-aware defaults
- Distributed training support (future)

## Security Considerations

### 1. Input Validation
- Path traversal prevention
- Configuration sanitization
- Template injection prevention

### 2. Dependency Management
- Pinned versions
- Security audits
- Minimal dependencies

### 3. Data Safety
- No PII in templates
- Secure checkpoint loading
- Safe file operations

## Future Architecture Enhancements

### Planned Features
1. Distributed training support
2. Model quantization
3. ONNX export
4. More model architectures (BERT, T5)
5. Advanced plugin system
6. Cloud deployment integration
7. Model registry
8. Experiment tracking dashboard

### Scalability Improvements
1. Streaming data loading
2. Incremental checkpointing
3. Dynamic batch sizing
4. Adaptive learning rates

## Technology Stack

### CLI Tool (TypeScript)
- **Node.js** - Runtime
- **TypeScript** - Type safety
- **Commander** - CLI framework
- **Inquirer** - Interactive prompts
- **Chalk** - Terminal styling
- **Ora** - Progress spinners

### Generated Projects (Python)
- **PyTorch** - Deep learning framework
- **Tokenizers** - Fast tokenization
- **TensorBoard** - Visualization
- **tqdm** - Progress bars
- **pytest** - Testing framework

## Deployment

### NPM Package
- Scoped package: `@theaniketgiri/create-llm`
- Binary: `create-llm`
- Auto-build on publish
- Semantic versioning

### Generated Projects
- Standalone Python projects
- No runtime dependencies on CLI
- Self-contained training infrastructure
- Portable across environments

## Maintenance & Updates

### Version Strategy
- Major: Breaking changes
- Minor: New features
- Patch: Bug fixes

### Update Mechanism
- Users regenerate projects for updates
- Templates evolve independently
- Backward compatibility maintained

---

**Last Updated**: 2025-10-26
**Version**: 2.0.1
**Maintainer**: Aniket Giri
