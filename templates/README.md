# LLM Training Templates

This directory contains pre-configured templates for different LLM training scenarios. Each template is optimized for specific hardware constraints and use cases.

## Available Templates

### 🔹 Tiny (10M parameters)
**Best for:** Learning, experimentation, and quick prototyping

- **Hardware:** CPU-friendly, runs on laptops with 4GB RAM
- **Training Time:** 10-30 minutes
- **Use Cases:**
  - Learning LLM training basics
  - Quick prototyping and testing
  - Educational purposes
  - Resource-constrained environments

### 🔸 Small (100M parameters)
**Best for:** Production-ready models with good performance

- **Hardware:** Single GPU (NVIDIA RTX 3060 12GB or better)
- **Training Time:** 2-6 hours
- **Use Cases:**
  - Domain-specific language models
  - Text generation tasks
  - Fine-tuning for specific applications
  - Production-ready small models

### 🔶 Base (1B parameters)
**Best for:** High-quality models and research

- **Hardware:** High-end GPU(s) (NVIDIA A100 40GB or 2x RTX 4090)
- **Training Time:** 1-3 days
- **Use Cases:**
  - High-quality language models
  - Research and experimentation
  - Large-scale text generation
  - Foundation models for fine-tuning

### 🔷 Custom
**Best for:** Experimentation and custom architectures

- **Hardware:** Varies based on configuration
- **Training Time:** Varies
- **Use Cases:**
  - Custom model architectures
  - Experimentation with hyperparameters
  - Specialized use cases
  - Research projects

## Template Structure

Each template JSON file contains:

```json
{
  "name": "template-name",
  "config": {
    "model": {
      "type": "gpt",
      "parameters": 10000000,
      "layers": 6,
      "heads": 6,
      "dim": 384,
      "vocab_size": 32000,
      "max_length": 512,
      "dropout": 0.1
    },
    "training": {
      "batch_size": 16,
      "learning_rate": 0.0006,
      "warmup_steps": 500,
      "max_steps": 10000,
      "eval_interval": 500,
      "save_interval": 2000,
      "optimizer": "adamw",
      "weight_decay": 0.01,
      "gradient_clip": 1.0,
      "mixed_precision": false,
      "gradient_accumulation_steps": 1
    },
    "data": {
      "max_length": 512,
      "stride": 256,
      "val_split": 0.1,
      "shuffle": true
    },
    "tokenizer": {
      "type": "bpe",
      "vocab_size": 32000,
      "min_frequency": 2,
      "special_tokens": ["<pad>", "<unk>", "<s>", "</s>"]
    },
    "hardware": {
      "min_ram": "4GB",
      "recommended_gpu": "None (CPU-friendly)",
      "estimated_training_time": "10-30 minutes",
      "can_run_on_cpu": true
    },
    "documentation": {
      "description": "Template description",
      "use_cases": ["Use case 1", "Use case 2"],
      "hardware_notes": "Hardware requirements and notes",
      "training_tips": ["Tip 1", "Tip 2"]
    }
  }
}
```

## Validation Rules

Templates are validated on load to ensure:

### Model Configuration
- Valid model type (gpt, bert, t5)
- Positive parameter counts
- Dimension divisible by number of heads
- Valid dropout range (0-1)

### Training Configuration
- Positive batch size and learning rates
- Valid optimizer (adamw, adam, sgd)
- Non-negative weight decay
- Positive gradient clipping

### Data Configuration
- Positive max length and stride
- Stride not greater than max length
- Valid validation split (0-1)

### Tokenizer Configuration
- Valid tokenizer type (bpe, wordpiece, unigram)
- Positive vocabulary size
- At least one special token

### Hardware Requirements
- All fields must be present and non-empty
- can_run_on_cpu must be boolean

### Documentation
- Description must be present
- At least one use case
- Hardware notes must be present
- At least one training tip

## Adding Custom Templates

To add a new template:

1. Create a new JSON file in this directory (e.g., `my-template.json`)
2. Follow the structure shown above
3. Ensure all validation rules are met
4. Update the TemplateManager to include your template name
5. Test with: `npm run build && node dist/test-template-manager.js`

## Usage

Templates are loaded automatically by the TemplateManager:

```typescript
import { TemplateManager } from './template-manager';

const manager = new TemplateManager();
const template = manager.getTemplate('tiny');
```

When creating a project:

```bash
npx create-llm my-project --template tiny
```
