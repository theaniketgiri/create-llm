# create-llm CLI Demo Commands

This document contains example commands to demonstrate the create-llm CLI tool.

## Basic Usage

### 1. Full Interactive Mode
```bash
npx create-llm
```
Prompts for:
- Project name
- Template selection
- Tokenizer type
- Plugins
- Confirmation

### 2. With Project Name
```bash
npx create-llm my-awesome-llm
```
Prompts for:
- Template selection
- Tokenizer type
- Plugins
- Confirmation

### 3. With Template
```bash
npx create-llm my-project --template tiny
```
Prompts for:
- Tokenizer type
- Plugins
- Confirmation

### 4. With Template and Tokenizer
```bash
npx create-llm my-project --template small --tokenizer bpe
```
Prompts for:
- Plugins
- Confirmation

### 5. Skip Confirmation (Non-Interactive)
```bash
npx create-llm my-project --template tiny --tokenizer bpe --yes
```
Only prompts for:
- Plugins

## Template Examples

### Tiny Template (CPU-Friendly)
```bash
npx create-llm tiny-model --template tiny --tokenizer bpe
```
- 10M parameters
- CPU-friendly
- Training time: 10-30 minutes

### Small Template (Single GPU)
```bash
npx create-llm small-model --template small --tokenizer bpe
```
- 100M parameters
- Requires RTX 3060 12GB or better
- Training time: 2-6 hours

### Base Template (Multi-GPU)
```bash
npx create-llm base-model --template base --tokenizer wordpiece
```
- 1B parameters
- Requires A100 40GB or 2x RTX 4090
- Training time: 1-3 days

### Custom Template
```bash
npx create-llm custom-model --template custom --tokenizer unigram
```
- 50M parameters
- Fully customizable
- Flexible hardware requirements

## Tokenizer Examples

### BPE (Byte Pair Encoding)
```bash
npx create-llm gpt-style --template small --tokenizer bpe
```
Used by: GPT-2, GPT-3, RoBERTa

### WordPiece
```bash
npx create-llm bert-style --template small --tokenizer wordpiece
```
Used by: BERT, DistilBERT

### Unigram
```bash
npx create-llm t5-style --template small --tokenizer unigram
```
Used by: T5, ALBERT

## Error Handling Examples

### Invalid Project Name
```bash
npx create-llm Invalid-Name --template tiny
# Error: Project name must contain only lowercase letters, numbers, and hyphens
```

### Invalid Template
```bash
npx create-llm my-project --template invalid
# Error: Invalid template: invalid
# Available templates: tiny, small, base, custom
```

### Invalid Tokenizer
```bash
npx create-llm my-project --template tiny --tokenizer invalid
# Error: Invalid tokenizer: invalid
# Available tokenizers: bpe, wordpiece, unigram
```

## Help and Version

### Display Help
```bash
npx create-llm --help
```

### Display Version
```bash
npx create-llm --version
```

## Advanced Usage

### Skip Dependency Installation
```bash
npx create-llm my-project --template tiny --tokenizer bpe --skip-install
```

### Complete Non-Interactive
```bash
npx create-llm my-project --template tiny --tokenizer bpe --yes --skip-install
```

## Real-World Scenarios

### 1. Quick Learning Project (CPU)
```bash
npx create-llm learn-llm --template tiny --tokenizer bpe
```
Perfect for:
- Learning LLM training basics
- Quick prototyping
- Testing on laptop

### 2. Production Model (Single GPU)
```bash
npx create-llm production-model --template small --tokenizer bpe
```
Perfect for:
- Domain-specific models
- Production deployments
- Single GPU training

### 3. Research Project (Multi-GPU)
```bash
npx create-llm research-model --template base --tokenizer wordpiece
```
Perfect for:
- High-quality models
- Research experiments
- Large-scale training

### 4. Custom Architecture
```bash
npx create-llm custom-arch --template custom --tokenizer unigram
```
Perfect for:
- Experimentation
- Custom architectures
- Specialized use cases

## Testing Commands

### Test Project Name Validation
```bash
node dist/test-prompts.js
```

### Test Template Manager
```bash
node dist/test-template-manager.js
```

### Test Template Validation
```bash
node dist/test-validation.js
```

## Build Commands

### Build TypeScript
```bash
npm run build
```

### Watch Mode
```bash
npm run dev
```

## Output Examples

### Successful Creation
```
🚀 Welcome to create-llm!

📋 Project Configuration:
──────────────────────────────────────────────────
  Project Name:  my-project
  Template:      TINY
  Model:         GPT (10M parameters)
  Tokenizer:     BPE
  Hardware:      None (CPU-friendly)
  Training Time: 10-30 minutes
  Plugins:       None
──────────────────────────────────────────────────

✨ Project Details:
──────────────────────────────────────────────────
  Location: ./my-project
  Template: TINY
  Model: GPT (10M parameters)
  Tokenizer: BPE
──────────────────────────────────────────────────

📝 Next steps:
  cd my-project
  pip install -r requirements.txt
  python tokenizer/train.py --data data/sample.txt
  python training/train.py

💡 Tips:
  • Start with this template if you're new to LLM training
  • Great for testing data pipelines and configurations

✅ Happy training!
```

## Notes

- All commands assume you're in the project root directory
- The CLI uses inquirer for interactive prompts
- Colors and formatting require a terminal that supports ANSI colors
- Project names must be lowercase with hyphens only
- Templates are validated against available options
- Tokenizer types are validated against supported types
