import dataclasses
from typing import Optional
import yaml

@dataclasses.dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Model architecture
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    
    # Training hyperparameters
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100000
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Data
    max_length: int = 1024
    train_data_path: str = "data/processed/train.txt"
    val_data_path: str = "data/processed/validation.txt"
    test_data_path: str = "data/processed/test.txt"
    tokenizer_path: str = "tokenizer/tokenizer.json"
    num_workers: int = 4
    streaming: bool = False
    
    # Optimization
    optimizer: str = "adamw"  # adamw, adam, sgd
    scheduler: str = "cosine"  # cosine, linear, constant
    lr_schedule: str = "warmup_cosine"  # warmup_cosine, warmup_linear
    
    # Checkpointing
    save_dir: str = "checkpoints"
    save_steps: int = 1000
    eval_steps: int = 500
    log_steps: int = 100
    
    # Logging
    log_dir: str = "logs"
    tensorboard: bool = True
    wandb: bool = False
    wandb_project: str = "llm-training"
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 5
    min_delta: float = 0.001
    
    # Mixed precision
    fp16: bool = True
    bf16: bool = False
    
    # Distributed training
    distributed: bool = False
    local_rank: int = -1
    
    def save(self, path: str):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(dataclasses.asdict(self), f, default_flow_style=False, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
