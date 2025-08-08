const fs = require('fs-extra');
const path = require('path');

async function createTrainingFiles(projectPath, options) {
  const trainingPath = path.join(projectPath, 'training');
  
  // Create __init__.py
  await fs.writeFile(path.join(trainingPath, '__init__.py'), `from .trainer import Trainer
from .config import TrainingConfig

__all__ = ['Trainer', 'TrainingConfig']
`);

  // Create main training script
  await createTrainingScript(trainingPath, options);
  
  // Create training configuration
  await createTrainingConfig(trainingPath, options);
  
  // Create trainer class
  await createTrainerClass(trainingPath, options);
  
  // Create requirements.txt
  await createRequirements(projectPath, options);
}

async function createTrainingScript(trainingPath, options) {
  const scriptContent = `#!/usr/bin/env python3
"""
Main training script for the LLM.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --resume checkpoints/latest.pt
"""

import argparse
import os
import yaml
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import logging
from pathlib import Path

from model import TransformerLM, ModelConfig
from tokenizer import CustomTokenizer
from data import create_dataloaders
from training.trainer import Trainer
from training.config import TrainingConfig

def setup_logging(log_dir: str):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )

def setup_distributed():
    """Setup distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        return rank, world_size
    return 0, 1

def main():
    parser = argparse.ArgumentParser(description="Train LLM")
    parser.add_argument("--config", "-c", required=True, help="Path to config file")
    parser.add_argument("--resume", "-r", help="Path to checkpoint to resume from")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = TrainingConfig(**config_dict)
    
    # Setup distributed training
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        rank, world_size = setup_distributed()
    else:
        rank, world_size = 0, 1
    
    # Setup logging
    if rank == 0:
        setup_logging(config.log_dir)
        logging.info(f"Starting training with config: {config}")
    
    # Load tokenizer
    tokenizer = CustomTokenizer(config.tokenizer_path)
    
    # Create model
    model_config = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(),
        n_positions=config.max_length,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
        model_type="${options.template}"
    )
    
    model = TransformerLM(model_config)
    
    # Move to GPU
    device = torch.device(f"cuda:{args.local_rank}" if args.local_rank != -1 else "cuda:0")
    model = model.to(device)
    
    # Setup distributed training
    if world_size > 1:
        model = DDP(model, device_ids=[args.local_rank])
    
    # Create dataloaders
    train_dataloader, val_dataloader = create_dataloaders(
        tokenizer=tokenizer,
        train_path=config.train_data_path,
        val_path=config.val_data_path,
        batch_size=config.batch_size,
        max_length=config.max_length,
        num_workers=config.num_workers,
        streaming=config.streaming
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config,
        device=device,
        rank=rank
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
`;
  
  await fs.writeFile(path.join(trainingPath, 'train.py'), scriptContent);
}

async function createTrainingConfig(trainingPath, options) {
  const configContent = `import dataclasses
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
`;
  
  await fs.writeFile(path.join(trainingPath, 'config.py'), configContent);
  
  // Create default config.yaml
  const defaultConfig = `# Model architecture
n_embd: 768
n_layer: 12
n_head: 12

# Training hyperparameters
batch_size: 8
learning_rate: 3e-4
weight_decay: 0.01
warmup_steps: 1000
max_steps: 100000
gradient_accumulation_steps: 1
max_grad_norm: 1.0

# Data
max_length: 1024
train_data_path: "data/processed/train.txt"
val_data_path: "data/processed/validation.txt"
test_data_path: "data/processed/test.txt"
tokenizer_path: "tokenizer/tokenizer.json"
num_workers: 4
streaming: false

# Optimization
optimizer: "adamw"
scheduler: "cosine"
lr_schedule: "warmup_cosine"

# Checkpointing
save_dir: "checkpoints"
save_steps: 1000
eval_steps: 500
log_steps: 100

# Logging
log_dir: "logs"
tensorboard: true
wandb: false
wandb_project: "llm-training"

# Early stopping
early_stopping: true
patience: 5
min_delta: 0.001

# Mixed precision
fp16: true
bf16: false

# Distributed training
distributed: false
local_rank: -1
`;
  
  await fs.writeFile(path.join(trainingPath, 'config.yaml'), defaultConfig);
}

async function createTrainerClass(trainingPath, options) {
  const trainerContent = `import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ConstantLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any
import math

from .config import TrainingConfig

class Trainer:
    """Main trainer class for LLM training."""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        config: TrainingConfig,
        device: torch.device,
        rank: int = 0
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            config: Training configuration
            device: Device to train on
            rank: Process rank for distributed training
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.device = device
        self.rank = rank
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler()
        
        # Setup mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.fp16 else None
        
        # Setup logging
        if rank == 0:
            self.writer = SummaryWriter(config.log_dir) if config.tensorboard else None
            self.logger = logging.getLogger(__name__)
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Create save directory
        os.makedirs(config.save_dir, exist_ok=True)
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer."""
        if self.config.optimizer == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999)
            )
        elif self.config.optimizer == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler."""
        if self.config.scheduler == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.max_steps,
                eta_min=0
            )
        elif self.config.scheduler == "linear":
            return LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=self.config.max_steps
            )
        elif self.config.scheduler == "constant":
            return ConstantLR(
                self.optimizer,
                factor=1.0,
                total_iters=self.config.max_steps
            )
        else:
            return None
    
    def _get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
    
    def _update_lr(self, step: int):
        """Update learning rate with warmup."""
        if step < self.config.warmup_steps:
            # Linear warmup
            lr = self.config.learning_rate * (step / self.config.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        elif self.scheduler is not None:
            self.scheduler.step()
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss for a batch."""
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        # Forward pass
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs['logits']
        
        # Shift logits and labels for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return loss
    
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        
        # Forward pass with mixed precision
        if self.config.fp16:
            with torch.cuda.amp.autocast():
                loss = self._compute_loss(batch)
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss = self._compute_loss(batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            
            # Optimizer step
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        
        return {'loss': loss.item()}
    
    def _eval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single evaluation step."""
        self.model.eval()
        
        with torch.no_grad():
            if self.config.fp16:
                with torch.cuda.amp.autocast():
                    loss = self._compute_loss(batch)
            else:
                loss = self._compute_loss(batch)
        
        return {'loss': loss.item()}
    
    def _evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                metrics = self._eval_step(batch)
                total_loss += metrics['loss']
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return {'val_loss': avg_loss}
    
    def _save_checkpoint(self, step: int, is_best: bool = False):
        """Save model checkpoint."""
        if self.rank != 0:
            return
        
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'best_val_loss': self.best_val_loss
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(self.config.save_dir, 'latest.pt')
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.config.save_dir, 'best.pt')
            torch.save(checkpoint, best_path)
        
        # Save step checkpoint
        step_path = os.path.join(self.config.save_dir, f'step_{step}.pt')
        torch.save(checkpoint, step_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.global_step = checkpoint['step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if self.rank == 0:
            self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def train(self):
        """Main training loop."""
        if self.rank == 0:
            self.logger.info("Starting training...")
        
        train_iter = iter(self.train_dataloader)
        
        for step in range(self.global_step, self.config.max_steps):
            # Get batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_dataloader)
                batch = next(train_iter)
            
            # Training step
            metrics = self._train_step(batch)
            
            # Update learning rate
            self._update_lr(step)
            
            # Logging
            if step % self.config.log_steps == 0 and self.rank == 0:
                lr = self._get_lr()
                self.logger.info(
                    f"Step {step}/{self.config.max_steps} - "
                    f"Loss: {metrics['loss']:.4f} - "
                    f"LR: {lr:.2e}"
                )
                
                if self.writer:
                    self.writer.add_scalar('train/loss', metrics['loss'], step)
                    self.writer.add_scalar('train/lr', lr, step)
            
            # Evaluation
            if step % self.config.eval_steps == 0:
                eval_metrics = self._evaluate()
                
                if self.rank == 0:
                    self.logger.info(f"Validation Loss: {eval_metrics['val_loss']:.4f}")
                    
                    if self.writer:
                        self.writer.add_scalar('val/loss', eval_metrics['val_loss'], step)
                    
                    # Check for best model
                    is_best = eval_metrics['val_loss'] < self.best_val_loss
                    if is_best:
                        self.best_val_loss = eval_metrics['val_loss']
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1
                    
                    # Early stopping
                    if self.config.early_stopping and self.patience_counter >= self.config.patience:
                        self.logger.info("Early stopping triggered!")
                        break
            
            # Save checkpoint
            if step % self.config.save_steps == 0:
                self._save_checkpoint(step, is_best)
            
            self.global_step = step + 1
        
        # Save final checkpoint
        if self.rank == 0:
            self._save_checkpoint(self.global_step)
            self.logger.info("Training completed!")
`;
  
  await fs.writeFile(path.join(trainingPath, 'trainer.py'), trainerContent);
}

async function createRequirements(projectPath, options) {
  const requirementsContent = `# Core ML libraries
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0

# Tokenizer
tokenizers>=0.13.0

# Data processing
datasets>=2.10.0
transformers>=4.20.0

# Training utilities
tensorboard>=2.10.0
wandb>=0.15.0
accelerate>=0.20.0

# Utilities
numpy>=1.21.0
pandas>=1.3.0
tqdm>=4.64.0
pyyaml>=6.0
requests>=2.28.0

# Optional: JAX support (uncomment if needed)
# jax>=0.4.0
# jaxlib>=0.4.0
# flax>=0.6.0

# Optional: Synthetic data generation
# openai>=0.27.0
# anthropic>=0.3.0
`;
  
  await fs.writeFile(path.join(projectPath, 'requirements.txt'), requirementsContent);
}

module.exports = { createTrainingFiles }; 