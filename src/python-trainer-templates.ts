/**
 * Python trainer templates
 */

export class PythonTrainerTemplates {
  /**
   * Get Trainer class
   */
  static getTrainer(): string {
    return `"""
Trainer class for LLM training
Handles the complete training loop with callbacks, checkpointing, and monitoring
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import List, Optional, Dict, Any
from tqdm import tqdm
import time


class Trainer:
    """
    Main trainer class for LLM training
    
    Features:
    - Automatic mixed precision training
    - Gradient accumulation
    - Learning rate scheduling with warmup
    - Callback system
    - Progress tracking
    - Checkpoint management
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: Dict[str, Any],
        callbacks: Optional[List[Any]] = None,
        device: Optional[str] = None
    ):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            callbacks: List of callbacks
            device: Device to train on
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.callbacks = callbacks or []
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.last_loss = 0.0
        self.best_val_loss = float('inf')
        self.tokens_processed = 0
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup mixed precision
        self.use_amp = config.get('training', {}).get('mixed_precision', False)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Gradient accumulation
        self.gradient_accumulation_steps = config.get('training', {}).get(
            'gradient_accumulation_steps', 1
        )
        
        print(f"Trainer initialized on device: {self.device}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with weight decay"""
        training_config = self.config.get('training', {})
        
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Don't apply weight decay to biases and layer norms
            if 'bias' in name or 'ln' in name or 'norm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optimizer_groups = [
            {'params': decay_params, 'weight_decay': training_config.get('weight_decay', 0.01)},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        optimizer_type = training_config.get('optimizer', 'adamw').lower()
        lr = training_config.get('learning_rate', 3e-4)
        
        if optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(optimizer_groups, lr=lr)
        elif optimizer_type == 'adam':
            optimizer = torch.optim.Adam(optimizer_groups, lr=lr)
        elif optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(optimizer_groups, lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup"""
        training_config = self.config.get('training', {})
        warmup_steps = training_config.get('warmup_steps', 0)
        max_steps = training_config.get('max_steps', 100000)
        
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return float(step) / float(max(1, warmup_steps))
            else:
                # Cosine decay
                progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
                return max(0.1, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265359))))
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train(self):
        """Main training loop"""
        training_config = self.config.get('training', {})
        max_steps = training_config.get('max_steps', 100000)
        eval_interval = training_config.get('eval_interval', 1000)
        
        # Call train begin callbacks
        for callback in self.callbacks:
            callback.on_train_begin(self)
        
        print(f"\\nStarting training for {max_steps} steps...")
        print(f"{'='*60}")
        
        self.model.train()
        start_time = time.time()
        
        try:
            while self.global_step < max_steps:
                # Train epoch
                self._train_epoch(max_steps, eval_interval)
                
                if self.global_step >= max_steps:
                    break
                
                self.epoch += 1
        
        except KeyboardInterrupt:
            print("\\n\\nTraining interrupted by user!")
        
        finally:
            # Call train end callbacks
            for callback in self.callbacks:
                callback.on_train_end(self)
            
            elapsed = time.time() - start_time
            print(f"\\n{'='*60}")
            print(f"Training completed!")
            print(f"Total time: {self._format_time(elapsed)}")
            print(f"Total steps: {self.global_step}")
            print(f"{'='*60}")
    
    def _train_epoch(self, max_steps: int, eval_interval: int):
        """Train for one epoch"""
        # Call epoch begin callbacks
        for callback in self.callbacks:
            callback.on_epoch_begin(self, self.epoch)
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.epoch}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            if self.global_step >= max_steps:
                break
            
            # Call batch begin callbacks
            for callback in self.callbacks:
                callback.on_batch_begin(self, batch)
            
            # Training step
            loss = self._training_step(batch, batch_idx)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                'step': self.global_step
            })
            
            # Call step end callbacks
            for callback in self.callbacks:
                callback.on_step_end(self, self.global_step, loss)
            
            # Evaluate
            if self.val_loader and self.global_step % eval_interval == 0 and self.global_step > 0:
                val_loss = self.evaluate()
                self.model.train()
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
        
        # Call epoch end callbacks
        for callback in self.callbacks:
            callback.on_epoch_end(self, self.epoch)
    
    def _training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> float:
        """Single training step with gradient accumulation"""
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass with mixed precision
        if self.use_amp:
            with autocast():
                outputs = self.model(**batch)
                loss = outputs['loss']
        else:
            outputs = self.model(**batch)
            loss = outputs['loss']
        
        # Scale loss for gradient accumulation
        loss = loss / self.gradient_accumulation_steps
        
        # Backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights
        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
            
            grad_clip = self.config.get('training', {}).get('gradient_clip', 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            
            # Optimizer step
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            # Scheduler step
            self.scheduler.step()
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Increment step
            self.global_step += 1
        
        # Track tokens
        self.tokens_processed += batch['input_ids'].numel()
        
        # Store last loss
        self.last_loss = loss.item() * self.gradient_accumulation_steps
        
        return self.last_loss
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on validation set"""
        if not self.val_loader:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        print("\\nEvaluating...")
        for batch in tqdm(self.val_loader, desc="Validation", leave=False):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            outputs = self.model(**batch)
            total_loss += outputs['loss'].item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss))
        
        print(f"Validation Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
        
        # Check for overfitting
        if perplexity < 1.1:
            print("\\n⚠️  WARNING: Perplexity < 1.1 indicates severe overfitting!")
            print("   The model has memorized the training data.")
            print("   Suggestions:")
            print("   - Add more training data")
            print("   - Increase dropout (try 0.3)")
            print("   - Reduce model size")
            print("   - Add weight decay\\n")
        
        return avg_loss
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


if __name__ == '__main__':
    print("Trainer class loaded successfully")
`;
  }

  /**
   * Get training __init__.py
   */
  static getTrainingInit(): string {
    return `"""
Training package
"""

from .trainer import Trainer

__all__ = [
    'Trainer',
]
`;
  }
}
