# Requirements Document

## Introduction

create-llm is a CLI tool that democratizes LLM training by providing an opinionated, batteries-included framework similar to create-next-app. The tool enables developers of all skill levels to train custom language models with minimal setup friction, following a convention-over-configuration philosophy. Users can scaffold complete LLM training projects with pre-configured architectures, data pipelines, training loops, and evaluation tools, progressing from beginner-friendly templates to advanced customization as their needs grow.

## Requirements

### Requirement 1: CLI Scaffolding Tool

**User Story:** As a developer, I want to scaffold a complete LLM training project with a single command, so that I can start training immediately without manual setup.

#### Acceptance Criteria

1. WHEN the user runs `npx create-llm <project-name>` THEN the system SHALL create a new directory with the project name containing a complete project structure
2. WHEN the user runs the CLI with `--template <template-name>` flag THEN the system SHALL scaffold the project using the specified template (tiny, small, base, or custom)
3. WHEN the user runs the CLI with `--tokenizer <type>` flag THEN the system SHALL configure the specified tokenizer type (bpe, wordpiece, or unigram)
4. WHEN scaffolding completes THEN the system SHALL display next steps and quick start instructions
5. WHEN scaffolding completes THEN the system SHALL create all necessary directories (data/, models/, tokenizer/, training/, evaluation/, checkpoints/, logs/)
6. WHEN scaffolding completes THEN the system SHALL generate a requirements.txt with all necessary dependencies
7. WHEN scaffolding completes THEN the system SHALL create a README.md with project-specific documentation

### Requirement 2: Template System

**User Story:** As a user with varying skill levels and hardware constraints, I want to choose from pre-configured templates, so that I can train models appropriate for my experience and resources.

#### Acceptance Criteria

1. WHEN the user selects the "tiny" template THEN the system SHALL configure a 10M parameter model optimized for CPU/laptop training
2. WHEN the user selects the "small" template THEN the system SHALL configure a 100M parameter model optimized for single GPU training
3. WHEN the user selects the "base" template THEN the system SHALL configure a 1B parameter model optimized for multi-GPU training
4. WHEN the user selects the "custom" template THEN the system SHALL provide a fully customizable architecture starting point
5. WHEN a template is selected THEN the system SHALL configure appropriate hyperparameters (batch size, learning rate, etc.) for that model size
6. WHEN a template is selected THEN the system SHALL include template-specific documentation about expected training time and hardware requirements

### Requirement 3: Configuration Management

**User Story:** As a developer, I want a centralized configuration file similar to next.config.js, so that I can easily customize my LLM training setup without editing multiple files.

#### Acceptance Criteria

1. WHEN the project is scaffolded THEN the system SHALL create an llm.config.js file in the project root
2. WHEN the llm.config.js file exists THEN it SHALL contain sections for model, training, data, tokenizer, and plugins configuration
3. WHEN the user modifies llm.config.js THEN the training scripts SHALL read and apply those configurations
4. WHEN invalid configuration is provided THEN the system SHALL display clear error messages with suggestions
5. WHEN the user runs training without custom config THEN the system SHALL use smart defaults from the selected template
6. WHEN the config file is updated THEN the system SHALL validate all parameters before training starts

### Requirement 4: Data Pipeline

**User Story:** As a user, I want a simple way to prepare my training data, so that I don't have to learn complex data preprocessing techniques.

#### Acceptance Criteria

1. WHEN the user places text files in data/raw/ THEN the system SHALL provide a prepare.py script to process them
2. WHEN the user runs prepare.py THEN the system SHALL tokenize the data and save it to data/processed/
3. WHEN preparing data THEN the system SHALL automatically split into train/validation sets if not already split
4. WHEN preparing data THEN the system SHALL display statistics (total tokens, vocabulary size, dataset size)
5. WHEN data preparation fails THEN the system SHALL provide clear error messages about data format issues
6. WHEN the user has multiple text files THEN the system SHALL concatenate them appropriately

### Requirement 5: Training Script

**User Story:** As a user, I want to start training with a single command, so that I can focus on results rather than implementation details.

#### Acceptance Criteria

1. WHEN the user runs `python train.py` THEN the system SHALL start training using the configuration from llm.config.js
2. WHEN training starts THEN the system SHALL display a progress bar with loss, tokens/sec, and ETA
3. WHEN training runs THEN the system SHALL automatically save checkpoints at configured intervals
4. WHEN training runs THEN the system SHALL evaluate on validation set at configured intervals
5. WHEN training completes THEN the system SHALL save the final model and display training summary
6. WHEN training is interrupted THEN the system SHALL save a checkpoint and allow resuming from that point
7. WHEN the user runs `python train.py --dashboard` THEN the system SHALL open a browser-based live training dashboard
8. WHEN GPU is available THEN the system SHALL automatically use it, otherwise fall back to CPU

### Requirement 6: Tokenizer Training

**User Story:** As a user, I want to train a custom tokenizer on my data, so that my model can efficiently represent my domain-specific vocabulary.

#### Acceptance Criteria

1. WHEN the user runs `python tokenizer/train.py` THEN the system SHALL train a tokenizer on data in data/raw/
2. WHEN tokenizer training completes THEN the system SHALL save tokenizer.json in the tokenizer/ directory
3. WHEN the user specifies tokenizer type in config THEN the system SHALL train that specific type (BPE, WordPiece, or Unigram)
4. WHEN the user specifies vocab_size in config THEN the system SHALL train a tokenizer with that vocabulary size
5. WHEN tokenizer training completes THEN the system SHALL display vocabulary statistics
6. WHEN no custom tokenizer exists THEN the system SHALL use a default pre-trained tokenizer

### Requirement 7: Model Evaluation

**User Story:** As a user, I want to evaluate my trained model's performance, so that I can understand its quality and capabilities.

#### Acceptance Criteria

1. WHEN the user runs `python evaluation/evaluate.py` THEN the system SHALL compute perplexity on the validation set
2. WHEN evaluation runs THEN the system SHALL display metrics including loss, perplexity, and tokens/sec
3. WHEN the user runs `python evaluation/generate.py` THEN the system SHALL generate sample text using the trained model
4. WHEN generating text THEN the user SHALL be able to specify prompts, temperature, and max length
5. WHEN evaluation completes THEN the system SHALL save results to logs/ directory

### Requirement 8: Interactive Chat Interface

**User Story:** As a user, I want to chat with my trained model interactively, so that I can quickly test its capabilities.

#### Acceptance Criteria

1. WHEN the user runs `python chat.py` THEN the system SHALL start an interactive terminal chat session
2. WHEN in chat mode THEN the user SHALL be able to type prompts and receive model responses
3. WHEN in chat mode THEN the system SHALL maintain conversation context
4. WHEN the user types "exit" or "quit" THEN the system SHALL gracefully exit the chat session
5. WHEN the model is generating THEN the system SHALL display a loading indicator

### Requirement 9: Live Training Dashboard

**User Story:** As a user, I want to monitor training progress in real-time with visualizations, so that I can understand how my model is learning.

#### Acceptance Criteria

1. WHEN the user runs training with `--dashboard` flag THEN the system SHALL open a web browser with a live dashboard
2. WHEN the dashboard is open THEN it SHALL display real-time loss curves (training and validation)
3. WHEN the dashboard is open THEN it SHALL display current tokens/sec and GPU utilization
4. WHEN the dashboard is open THEN it SHALL show sample text generations updated periodically
5. WHEN the dashboard is open THEN it SHALL display ETA for training completion
6. WHEN training completes THEN the dashboard SHALL display a completion message

### Requirement 10: Model Deployment

**User Story:** As a user, I want to deploy my trained model with a single command, so that I can share it or use it in production.

#### Acceptance Criteria

1. WHEN the user runs `python deploy.py --to huggingface` THEN the system SHALL upload the model to Hugging Face Hub
2. WHEN the user runs `python deploy.py --to replicate` THEN the system SHALL deploy the model to Replicate
3. WHEN deploying THEN the system SHALL prompt for necessary credentials if not configured
4. WHEN deployment completes THEN the system SHALL display the model URL
5. WHEN deployment fails THEN the system SHALL provide clear error messages and troubleshooting steps

### Requirement 11: Plugin System

**User Story:** As a user, I want to extend create-llm with plugins, so that I can integrate additional tools and services.

#### Acceptance Criteria

1. WHEN the user adds a plugin to llm.config.js THEN the system SHALL load and initialize that plugin
2. WHEN the "wandb" plugin is enabled THEN the system SHALL log training metrics to Weights & Biases
3. WHEN the "synthex" plugin is enabled THEN the system SHALL integrate SynthexAI for synthetic data generation
4. WHEN the "huggingface" plugin is enabled THEN the system SHALL enable easy model hub integration
5. WHEN a plugin fails to load THEN the system SHALL display a warning but continue with available plugins

### Requirement 12: Model Comparison Tool

**User Story:** As a user, I want to compare different model versions side-by-side, so that I can choose the best performing model.

#### Acceptance Criteria

1. WHEN the user runs `python compare.py model_v1/ model_v2/` THEN the system SHALL evaluate both models on the same validation set
2. WHEN comparison runs THEN the system SHALL display metrics for both models in a table format
3. WHEN comparison runs THEN the system SHALL generate sample outputs from both models for the same prompts
4. WHEN comparison completes THEN the system SHALL save a comparison report to logs/

### Requirement 13: Synthetic Data Generation

**User Story:** As a user, I want to generate synthetic training data, so that I can augment my dataset or bootstrap training without existing data.

#### Acceptance Criteria

1. WHEN the user runs `python data/generate.py --synthex --samples <count>` THEN the system SHALL generate synthetic training data using SynthexAI
2. WHEN synthetic data generation completes THEN the system SHALL save the data to data/raw/
3. WHEN the user specifies a topic or domain THEN the system SHALL generate domain-specific synthetic data
4. WHEN generation completes THEN the system SHALL display statistics about the generated data

### Requirement 14: Documentation and Help

**User Story:** As a user, I want comprehensive documentation and help commands, so that I can learn how to use create-llm effectively.

#### Acceptance Criteria

1. WHEN the project is scaffolded THEN the system SHALL create a README.md with quick start instructions
2. WHEN the user runs any script with `--help` flag THEN the system SHALL display usage information and available options
3. WHEN the user encounters an error THEN the system SHALL provide helpful error messages with suggestions
4. WHEN the project is created THEN the README SHALL include links to full documentation and examples
5. WHEN the project is created THEN the README SHALL include expected training times and hardware requirements for the selected template

### Requirement 15: Checkpoint Management

**User Story:** As a user, I want automatic checkpoint saving and resuming, so that I don't lose progress if training is interrupted.

#### Acceptance Criteria

1. WHEN training runs THEN the system SHALL save checkpoints at configured intervals to checkpoints/
2. WHEN a checkpoint is saved THEN it SHALL include model weights, optimizer state, and training step
3. WHEN the user runs training and a checkpoint exists THEN the system SHALL ask whether to resume from checkpoint
4. WHEN resuming from checkpoint THEN the system SHALL restore model state and continue from the saved step
5. WHEN checkpoints exceed a configured limit THEN the system SHALL automatically delete oldest checkpoints
6. WHEN training completes THEN the system SHALL save a final checkpoint marked as "final"
