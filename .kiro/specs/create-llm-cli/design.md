# Design Document

## Overview

create-llm is a Node.js-based CLI tool that scaffolds Python-based LLM training projects. The architecture follows a clear separation between the scaffolding tool (Node.js) and the generated project (Python), similar to how create-next-app works. The design emphasizes convention over configuration, progressive complexity, and an excellent developer experience through smart defaults and clear abstractions.

The system consists of three main layers:
1. **CLI Layer**: Node.js-based scaffolding tool that generates projects
2. **Project Layer**: Python-based training framework with modular components
3. **Configuration Layer**: Centralized configuration management that bridges user intent with implementation

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     CLI Tool (Node.js)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Template   │  │  Scaffolder  │  │    Config    │      │
│  │   Manager    │  │    Engine    │  │  Generator   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼ Generates
┌─────────────────────────────────────────────────────────────┐
│              Generated Project (Python)                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                  llm.config.js                        │  │
│  │         (Central Configuration Management)            │  │
│  └──────────────────────────────────────────────────────┘  │
│                            │                                 │
│         ┌──────────────────┼──────────────────┐            │
│         ▼                  ▼                  ▼             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │    Data     │  │   Training  │  │  Evaluation │        │
│  │   Pipeline  │  │   Engine    │  │   System    │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│         │                  │                  │             │
│         └──────────────────┼──────────────────┘            │
│                            ▼                                 │
│                  ┌─────────────────┐                        │
│                  │  Model Manager  │                        │
│                  │  & Checkpoints  │                        │
│                  └─────────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

### Component Breakdown

**CLI Tool (Node.js)**
- Handles project scaffolding and initialization
- Manages templates and their configurations
- Generates project structure and files
- Provides interactive prompts for user choices
- Validates user inputs and system requirements

**Generated Project (Python)**
- Self-contained training framework
- Modular architecture with clear separation of concerns
- Configuration-driven behavior
- Extensible through plugins

## Components and Interfaces

### 1. CLI Tool Components

#### Template Manager
Manages the four core templates and their configurations.

```typescript
interface Template {
  name: 'tiny' | 'small' | 'base' | 'custom';
  config: {
    model: ModelConfig;
    training: TrainingConfig;
    hardware: HardwareRequirements;
    documentation: TemplateDocumentation;
  };
}

interface ModelConfig {
  type: 'gpt' | 'bert' | 't5';
  parameters: number;
  layers: number;
  heads: number;
  dim: number;
  vocab_size: number;
  max_length: number;
}

interface TrainingConfig {
  batch_size: number;
  learning_rate: number;
  warmup_steps: number;
  max_steps: number;
  eval_interval: number;
  save_interval: number;
  optimizer: 'adamw' | 'adam' | 'sgd';
  gradient_accumulation_steps: number;
}

interface HardwareRequirements {
  min_ram: string;
  recommended_gpu: string;
  estimated_training_time: string;
  can_run_on_cpu: boolean;
}
```

**Template Specifications:**

- **Tiny Template**: 10M parameters, CPU-friendly, 10-30 min training
- **Small Template**: 100M parameters, single GPU, 2-6 hours training
- **Base Template**: 1B parameters, multi-GPU, 1-3 days training
- **Custom Template**: User-defined, full control

#### Scaffolder Engine
Generates the project structure and files.

```typescript
interface ScaffolderEngine {
  createProjectStructure(projectName: string, template: Template): void;
  generateConfigFile(template: Template, options: UserOptions): void;
  copyTemplateFiles(template: Template, destination: string): void;
  installDependencies(projectPath: string): Promise<void>;
  displayNextSteps(projectName: string, template: Template): void;
}

interface UserOptions {
  template: string;
  tokenizer?: 'bpe' | 'wordpiece' | 'unigram';
  plugins?: string[];
  skipInstall?: boolean;
}
```

**Generated Project Structure:**
```
my-llm/
├── data/
│   ├── raw/              # User places training data here
│   ├── processed/        # Tokenized and processed data
│   └── prepare.py        # Data preprocessing script
├── models/
│   ├── config.py         # Model architecture configuration
│   └── architectures/    # Pre-built model architectures
│       ├── __init__.py
│       ├── tiny.py       # 10M param architecture
│       ├── small.py      # 100M param architecture
│       ├── base.py       # 1B param architecture
│       └── gpt.py        # GPT-style transformer
├── tokenizer/
│   ├── train.py          # Tokenizer training script
│   └── tokenizer.json    # Saved tokenizer (generated)
├── training/
│   ├── train.py          # Main training script
│   ├── trainer.py        # Trainer class
│   ├── config.yaml       # Training hyperparameters
│   └── callbacks/        # Training callbacks
│       ├── __init__.py
│       ├── checkpoint.py # Checkpoint saving
│       ├── logging.py    # Training logging
│       └── dashboard.py  # Live dashboard
├── evaluation/
│   ├── evaluate.py       # Evaluation script
│   ├── generate.py       # Text generation
│   └── metrics.py        # Evaluation metrics
├── checkpoints/          # Model checkpoints (generated)
├── logs/                 # Training logs (generated)
├── llm.config.js         # Main configuration file
├── requirements.txt      # Python dependencies
├── chat.py               # Interactive chat interface
├── deploy.py             # Deployment script
├── compare.py            # Model comparison tool
└── README.md             # Project documentation
```

### 2. Configuration System

#### llm.config.js Structure
The central configuration file that controls all aspects of the project.

```javascript
module.exports = {
  // Model architecture configuration
  model: {
    type: 'gpt',              // Architecture type
    size: 'small',            // Template size
    vocab_size: 32000,        // Vocabulary size
    max_length: 512,          // Maximum sequence length
    layers: 12,               // Number of transformer layers
    heads: 8,                 // Number of attention heads
    dim: 512,                 // Model dimension
    dropout: 0.1,             // Dropout rate
  },
  
  // Training configuration
  training: {
    batch_size: 32,           // Training batch size
    learning_rate: 3e-4,      // Learning rate
    warmup_steps: 1000,       // Warmup steps
    max_steps: 100000,        // Maximum training steps
    eval_interval: 1000,      // Evaluation frequency
    save_interval: 5000,      // Checkpoint save frequency
    optimizer: 'adamw',       // Optimizer type
    weight_decay: 0.01,       // Weight decay
    gradient_clip: 1.0,       // Gradient clipping
    mixed_precision: true,    // Use mixed precision training
    gradient_accumulation: 1, // Gradient accumulation steps
  },
  
  // Data configuration
  data: {
    train_path: 'data/raw/train.txt',
    val_path: 'data/raw/val.txt',
    max_length: 512,          // Maximum sequence length
    stride: 256,              // Sliding window stride
    val_split: 0.1,           // Validation split ratio
    shuffle: true,            // Shuffle training data
  },
  
  // Tokenizer configuration
  tokenizer: {
    type: 'bpe',              // Tokenizer type
    vocab_size: 32000,        // Vocabulary size
    min_frequency: 2,         // Minimum token frequency
    special_tokens: ['<pad>', '<unk>', '<s>', '</s>'],
  },
  
  // Checkpoint configuration
  checkpoints: {
    save_total_limit: 3,      // Maximum checkpoints to keep
    save_on_interrupt: true,  // Save on keyboard interrupt
    resume_from_checkpoint: null, // Path to resume from
  },
  
  // Logging configuration
  logging: {
    log_interval: 100,        // Logging frequency
    log_dir: 'logs',          // Log directory
    tensorboard: true,        // Enable TensorBoard
    wandb: false,             // Enable Weights & Biases
  },
  
  // Plugin configuration
  plugins: [
    // 'wandb',               // Weights & Biases integration
    // 'synthex',             // SynthexAI integration
    // 'huggingface',         // Hugging Face Hub integration
  ],
  
  // Deployment configuration
  deployment: {
    huggingface: {
      repo_name: null,        // HF repo name
      private: false,         // Private repo
    },
    replicate: {
      model_name: null,       // Replicate model name
    },
  },
};
```

#### Config Loader (Python)
Python module that reads and validates the JavaScript config file.

```python
class ConfigLoader:
    def __init__(self, config_path: str = 'llm.config.js'):
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> dict:
        """Load JavaScript config file using Node.js"""
        # Execute Node.js to parse the config file
        # Return parsed configuration as Python dict
        pass
    
    def _validate_config(self) -> None:
        """Validate configuration parameters"""
        # Check required fields
        # Validate value ranges
        # Check hardware compatibility
        pass
    
    def get(self, key: str, default=None):
        """Get configuration value by key"""
        pass
```

### 3. Data Pipeline Components

#### Data Preprocessor
Handles data loading, tokenization, and preparation.

```python
class DataPreprocessor:
    def __init__(self, config: dict, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
    
    def prepare_data(self, input_path: str, output_path: str):
        """
        Prepare training data:
        1. Load raw text files
        2. Tokenize text
        3. Create training examples with sliding window
        4. Split into train/val if needed
        5. Save processed data
        """
        pass
    
    def load_raw_text(self, path: str) -> str:
        """Load and concatenate text files"""
        pass
    
    def create_examples(self, text: str) -> List[List[int]]:
        """Create training examples with sliding window"""
        pass
    
    def split_train_val(self, examples: List, val_split: float):
        """Split data into train and validation sets"""
        pass
```

#### Dataset Class
PyTorch dataset for efficient data loading.

```python
class LLMDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, max_length: int):
        self.data = self._load_data(data_path)
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            {
                'input_ids': tensor of token ids,
                'attention_mask': tensor of attention mask,
                'labels': tensor of labels (shifted input_ids)
            }
        """
        pass
```

### 4. Model Architecture Components

#### Model Factory
Creates model instances based on configuration.

```python
class ModelFactory:
    @staticmethod
    def create_model(config: dict) -> nn.Module:
        """
        Create model based on config.model.type:
        - 'gpt': GPT-style decoder-only transformer
        - 'bert': BERT-style encoder-only transformer
        - 't5': T5-style encoder-decoder transformer
        """
        model_type = config['model']['type']
        if model_type == 'gpt':
            return GPTModel(config['model'])
        elif model_type == 'bert':
            return BERTModel(config['model'])
        elif model_type == 't5':
            return T5Model(config['model'])
        else:
            raise ValueError(f"Unknown model type: {model_type}")
```

#### GPT Model Architecture
Core transformer implementation.

```python
class GPTModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(
            config['vocab_size'], 
            config['dim']
        )
        self.position_embedding = nn.Embedding(
            config['max_length'], 
            config['dim']
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) 
            for _ in range(config['layers'])
        ])
        
        # Output layer
        self.ln_f = nn.LayerNorm(config['dim'])
        self.lm_head = nn.Linear(config['dim'], config['vocab_size'])
    
    def forward(self, input_ids, attention_mask=None):
        """Forward pass through the model"""
        pass
```

### 5. Training Engine Components

#### Trainer Class
Main training loop and logic.

```python
class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Setup callbacks
        self.callbacks = self._create_callbacks()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
    
    def train(self):
        """Main training loop"""
        for epoch in range(self.config['training']['epochs']):
            self.epoch = epoch
            self._train_epoch()
            
            if self.global_step >= self.config['training']['max_steps']:
                break
    
    def _train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        for batch in self.train_loader:
            loss = self._training_step(batch)
            self._backward_step(loss)
            
            # Callbacks
            for callback in self.callbacks:
                callback.on_step_end(self.global_step, loss)
            
            self.global_step += 1
            
            # Evaluation
            if self.global_step % self.config['training']['eval_interval'] == 0:
                self._evaluate()
    
    def _training_step(self, batch) -> torch.Tensor:
        """Single training step"""
        pass
    
    def _evaluate(self):
        """Evaluate on validation set"""
        pass
```

#### Callback System
Extensible callback system for training events.

```python
class Callback:
    def on_train_begin(self): pass
    def on_train_end(self): pass
    def on_epoch_begin(self, epoch): pass
    def on_epoch_end(self, epoch): pass
    def on_step_begin(self, step): pass
    def on_step_end(self, step, loss): pass

class CheckpointCallback(Callback):
    """Saves model checkpoints"""
    def on_step_end(self, step, loss):
        if step % self.save_interval == 0:
            self.save_checkpoint(step)

class LoggingCallback(Callback):
    """Logs training metrics"""
    def on_step_end(self, step, loss):
        if step % self.log_interval == 0:
            self.log_metrics(step, loss)

class DashboardCallback(Callback):
    """Updates live training dashboard"""
    def on_step_end(self, step, loss):
        self.update_dashboard(step, loss)
```

### 6. Evaluation System Components

#### Evaluator Class
Handles model evaluation and metrics.

```python
class Evaluator:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
    
    def evaluate(self, val_loader) -> dict:
        """
        Evaluate model on validation set.
        Returns metrics: loss, perplexity, tokens/sec
        """
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in val_loader:
                loss = self._eval_step(batch)
                total_loss += loss
                total_tokens += batch['input_ids'].numel()
        
        avg_loss = total_loss / len(val_loader)
        perplexity = torch.exp(avg_loss)
        
        return {
            'loss': avg_loss.item(),
            'perplexity': perplexity.item(),
            'tokens': total_tokens
        }
    
    def generate(self, prompt: str, max_length: int = 100, 
                 temperature: float = 1.0) -> str:
        """Generate text from prompt"""
        pass
```

#### Text Generator
Handles text generation with various sampling strategies.

```python
class TextGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def generate(self, prompt: str, max_length: int = 100,
                 temperature: float = 1.0, top_k: int = 50,
                 top_p: float = 0.95) -> str:
        """
        Generate text using:
        - Temperature sampling
        - Top-k sampling
        - Top-p (nucleus) sampling
        """
        pass
```

### 7. Live Dashboard Component

#### Dashboard Server
Flask-based web server for live training visualization.

```python
class DashboardServer:
    def __init__(self, port: int = 5000):
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.port = port
        self.metrics_history = []
        
        self._setup_routes()
    
    def _setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template('dashboard.html')
        
        @self.socketio.on('connect')
        def handle_connect():
            # Send historical metrics to new client
            emit('history', self.metrics_history)
    
    def update_metrics(self, step: int, metrics: dict):
        """Update dashboard with new metrics"""
        self.metrics_history.append({
            'step': step,
            'metrics': metrics,
            'timestamp': time.time()
        })
        self.socketio.emit('update', {
            'step': step,
            'metrics': metrics
        })
    
    def start(self):
        """Start dashboard server in background thread"""
        threading.Thread(
            target=lambda: self.socketio.run(self.app, port=self.port),
            daemon=True
        ).start()
```

### 8. Plugin System

#### Plugin Manager
Loads and manages plugins.

```python
class PluginManager:
    def __init__(self, config: dict):
        self.config = config
        self.plugins = {}
        self._load_plugins()
    
    def _load_plugins(self):
        """Load plugins specified in config"""
        for plugin_name in self.config.get('plugins', []):
            try:
                plugin = self._import_plugin(plugin_name)
                self.plugins[plugin_name] = plugin
                plugin.initialize(self.config)
            except Exception as e:
                print(f"Warning: Failed to load plugin {plugin_name}: {e}")
    
    def get_plugin(self, name: str):
        """Get loaded plugin by name"""
        return self.plugins.get(name)
```

#### Plugin Interface
Base class for plugins.

```python
class Plugin:
    def initialize(self, config: dict):
        """Initialize plugin with configuration"""
        pass
    
    def on_train_begin(self, trainer):
        """Called when training begins"""
        pass
    
    def on_train_end(self, trainer):
        """Called when training ends"""
        pass
    
    def on_step_end(self, step: int, metrics: dict):
        """Called after each training step"""
        pass
```

**Built-in Plugins:**

1. **WandB Plugin**: Logs metrics to Weights & Biases
2. **SynthexAI Plugin**: Generates synthetic training data
3. **HuggingFace Plugin**: Easy model hub integration

### 9. Deployment System

#### Deployment Manager
Handles model deployment to various platforms.

```python
class DeploymentManager:
    def __init__(self, model_path: str, config: dict):
        self.model_path = model_path
        self.config = config
    
    def deploy_to_huggingface(self, repo_name: str, private: bool = False):
        """
        Deploy model to Hugging Face Hub:
        1. Load model and tokenizer
        2. Create model card
        3. Push to hub
        """
        pass
    
    def deploy_to_replicate(self, model_name: str):
        """
        Deploy model to Replicate:
        1. Create Cog configuration
        2. Build Docker image
        3. Push to Replicate
        """
        pass
```

## Data Models

### Training State
```python
@dataclass
class TrainingState:
    global_step: int
    epoch: int
    best_val_loss: float
    optimizer_state: dict
    scheduler_state: dict
    rng_state: dict
```

### Checkpoint
```python
@dataclass
class Checkpoint:
    model_state_dict: dict
    optimizer_state_dict: dict
    scheduler_state_dict: dict
    training_state: TrainingState
    config: dict
    timestamp: float
```

### Metrics
```python
@dataclass
class TrainingMetrics:
    step: int
    loss: float
    learning_rate: float
    tokens_per_sec: float
    gpu_memory_used: float
    timestamp: float

@dataclass
class EvaluationMetrics:
    loss: float
    perplexity: float
    tokens_per_sec: float
    sample_generations: List[str]
```

## Error Handling

### Error Categories

1. **Configuration Errors**
   - Invalid config values
   - Missing required fields
   - Hardware incompatibility
   - Strategy: Validate early, provide clear error messages with suggestions

2. **Data Errors**
   - Missing data files
   - Invalid data format
   - Tokenization failures
   - Strategy: Validate data before training, provide helpful error messages

3. **Training Errors**
   - Out of memory
   - NaN/Inf losses
   - Checkpoint corruption
   - Strategy: Graceful degradation, auto-save on error, recovery suggestions

4. **Deployment Errors**
   - Missing credentials
   - Network failures
   - Invalid model format
   - Strategy: Pre-flight checks, clear error messages, retry logic

### Error Handling Strategy

```python
class CreateLLMError(Exception):
    """Base exception for create-llm"""
    def __init__(self, message: str, suggestion: str = None):
        self.message = message
        self.suggestion = suggestion
        super().__init__(self.message)
    
    def __str__(self):
        error_msg = f"Error: {self.message}"
        if self.suggestion:
            error_msg += f"\n\nSuggestion: {self.suggestion}"
        return error_msg

class ConfigurationError(CreateLLMError):
    """Configuration-related errors"""
    pass

class DataError(CreateLLMError):
    """Data-related errors"""
    pass

class TrainingError(CreateLLMError):
    """Training-related errors"""
    pass
```

### Graceful Degradation

- If GPU unavailable, fall back to CPU with warning
- If dashboard fails to start, continue training without it
- If plugin fails to load, warn but continue
- If checkpoint save fails, retry with exponential backoff

## Testing Strategy

### Unit Tests

1. **CLI Tool Tests**
   - Template generation
   - Config file generation
   - Project structure creation
   - Input validation

2. **Data Pipeline Tests**
   - Tokenization correctness
   - Data preprocessing
   - Dataset loading
   - Train/val splitting

3. **Model Tests**
   - Forward pass correctness
   - Gradient flow
   - Output shape validation
   - Architecture variants

4. **Training Tests**
   - Training loop execution
   - Checkpoint saving/loading
   - Callback execution
   - Metric calculation

### Integration Tests

1. **End-to-End Scaffolding**
   - Create project with each template
   - Verify all files generated
   - Validate config correctness

2. **End-to-End Training**
   - Train tiny model on small dataset
   - Verify checkpoints saved
   - Verify metrics logged
   - Verify model can generate text

3. **Plugin Integration**
   - Test each plugin loads correctly
   - Test plugin callbacks execute
   - Test plugin error handling

### Performance Tests

1. **Training Speed**
   - Tokens/sec benchmarks for each template
   - Memory usage profiling
   - GPU utilization monitoring

2. **Scalability**
   - Multi-GPU training
   - Large dataset handling
   - Long training runs

### Test Infrastructure

```python
# pytest fixtures for common test setup
@pytest.fixture
def temp_project_dir():
    """Create temporary project directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        'model': {'type': 'gpt', 'size': 'tiny', ...},
        'training': {'batch_size': 4, ...},
        ...
    }

@pytest.fixture
def sample_data():
    """Sample training data"""
    return "This is sample training text. " * 100
```

### Continuous Integration

- Run unit tests on every commit
- Run integration tests on PR
- Run performance tests nightly
- Test on multiple Python versions (3.8, 3.9, 3.10, 3.11)
- Test on multiple platforms (Linux, macOS, Windows)

## Performance Considerations

### Memory Optimization

1. **Gradient Checkpointing**: Trade compute for memory
2. **Mixed Precision Training**: Use FP16/BF16 to reduce memory
3. **Gradient Accumulation**: Simulate larger batch sizes
4. **Model Sharding**: For very large models

### Training Speed Optimization

1. **DataLoader Workers**: Parallel data loading
2. **Compiled Models**: Use torch.compile() for PyTorch 2.0+
3. **Efficient Attention**: Flash Attention for long sequences
4. **Distributed Training**: Multi-GPU support

### Disk I/O Optimization

1. **Memory-Mapped Files**: For large datasets
2. **Compressed Checkpoints**: Reduce checkpoint size
3. **Async Checkpoint Saving**: Don't block training

## Security Considerations

1. **Config File Execution**: Safely execute JavaScript config files
2. **Credential Management**: Secure storage of API keys
3. **Model Uploads**: Validate before uploading to hubs
4. **Dependency Management**: Pin versions, check for vulnerabilities

## Extensibility

### Adding New Templates

1. Create template config in `templates/` directory
2. Add template-specific documentation
3. Register template in template manager

### Adding New Model Architectures

1. Implement model class inheriting from `nn.Module`
2. Add to model factory
3. Create template using new architecture

### Adding New Plugins

1. Implement plugin class inheriting from `Plugin`
2. Register plugin in plugin system
3. Document plugin usage

### Adding New Deployment Targets

1. Implement deployment method in `DeploymentManager`
2. Add configuration options
3. Document deployment process
