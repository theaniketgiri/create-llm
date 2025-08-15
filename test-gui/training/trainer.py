import torch
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
