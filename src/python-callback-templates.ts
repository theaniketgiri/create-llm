/**
 * Python training callback templates
 */

export class PythonCallbackTemplates {
  /**
   * Get base Callback class
   */
  static getBaseCallback(): string {
    return `"""
Base callback class for training lifecycle hooks
"""

from typing import Any, Dict, Optional


class Callback:
    """
    Base class for training callbacks
    
    Callbacks allow you to hook into the training loop at various points:
    - on_train_begin: Called at the start of training
    - on_train_end: Called at the end of training
    - on_epoch_begin: Called at the start of each epoch
    - on_epoch_end: Called at the end of each epoch
    - on_batch_begin: Called before processing each batch
    - on_batch_end: Called after processing each batch
    - on_step_end: Called after each training step
    """
    
    def on_train_begin(self, trainer: Any):
        """Called when training begins"""
        pass
    
    def on_train_end(self, trainer: Any):
        """Called when training ends"""
        pass
    
    def on_epoch_begin(self, trainer: Any, epoch: int):
        """Called at the beginning of each epoch"""
        pass
    
    def on_epoch_end(self, trainer: Any, epoch: int):
        """Called at the end of each epoch"""
        pass
    
    def on_batch_begin(self, trainer: Any, batch: Dict[str, Any]):
        """Called before processing each batch"""
        pass
    
    def on_batch_end(self, trainer: Any, batch: Dict[str, Any], outputs: Dict[str, Any]):
        """Called after processing each batch"""
        pass
    
    def on_step_end(self, trainer: Any, step: int, loss: float, metrics: Optional[Dict[str, float]] = None):
        """Called after each training step"""
        pass
`;
  }

  /**
   * Get CheckpointCallback
   */
  static getCheckpointCallback(): string {
    return `"""
Checkpoint callback for saving model checkpoints during training
"""

import torch
from pathlib import Path
from typing import Any, Optional
from .base import Callback


class CheckpointCallback(Callback):
    """
    Saves model checkpoints during training
    
    Features:
    - Save at regular intervals
    - Keep only N most recent checkpoints
    - Save best model based on validation loss
    - Save on training interruption
    """
    
    def __init__(
        self,
        checkpoint_dir: str = 'checkpoints',
        save_interval: int = 1000,
        save_total_limit: int = 3,
        save_best: bool = True,
        verbose: bool = True
    ):
        """
        Initialize checkpoint callback
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_interval: Save checkpoint every N steps
            save_total_limit: Maximum number of checkpoints to keep
            save_best: Whether to save best model
            verbose: Print checkpoint messages
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_interval = save_interval
        self.save_total_limit = save_total_limit
        self.save_best = save_best
        self.verbose = verbose
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_loss = float('inf')
        self.checkpoints = []
    
    def on_train_begin(self, trainer: Any):
        """Setup checkpoint directory"""
        if self.verbose:
            print(f"Checkpoints will be saved to: {self.checkpoint_dir}")
    
    def on_step_end(self, trainer: Any, step: int, loss: float, metrics: Optional[dict] = None):
        """Save checkpoint at intervals"""
        if step % self.save_interval == 0 and step > 0:
            self._save_checkpoint(trainer, step, loss, is_best=False)
    
    def on_train_end(self, trainer: Any):
        """Save final checkpoint"""
        self._save_checkpoint(trainer, trainer.global_step, trainer.last_loss, is_final=True)
    
    def _save_checkpoint(
        self,
        trainer: Any,
        step: int,
        loss: float,
        is_best: bool = False,
        is_final: bool = False
    ):
        """Save a checkpoint"""
        # Determine checkpoint name
        if is_final:
            checkpoint_name = 'checkpoint-final.pt'
        elif is_best:
            checkpoint_name = 'checkpoint-best.pt'
        else:
            checkpoint_name = f'checkpoint-{step}.pt'
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Create checkpoint
        checkpoint = {
            'step': step,
            'epoch': getattr(trainer, 'epoch', 0),
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'loss': loss,
            'config': getattr(trainer, 'config', None),
        }
        
        # Add scheduler if exists
        if hasattr(trainer, 'scheduler') and trainer.scheduler is not None:
            checkpoint['scheduler_state_dict'] = trainer.scheduler.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        if self.verbose:
            print(f"Saved checkpoint: {checkpoint_path}")
        
        # Track checkpoint
        if not is_best and not is_final:
            self.checkpoints.append(checkpoint_path)
            self._cleanup_old_checkpoints()
        
        # Save best model
        if self.save_best and loss < self.best_loss:
            self.best_loss = loss
            best_path = self.checkpoint_dir / 'checkpoint-best.pt'
            torch.save(checkpoint, best_path)
            if self.verbose:
                print(f"New best model saved: {best_path} (loss: {loss:.4f})")
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to keep only save_total_limit"""
        if len(self.checkpoints) > self.save_total_limit:
            # Remove oldest checkpoint
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
                if self.verbose:
                    print(f"Removed old checkpoint: {old_checkpoint}")
    
    def load_checkpoint(self, checkpoint_path: str, trainer: Any):
        """Load a checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and hasattr(trainer, 'scheduler'):
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        trainer.global_step = checkpoint['step']
        trainer.epoch = checkpoint.get('epoch', 0)
        
        if self.verbose:
            print(f"Loaded checkpoint from: {checkpoint_path}")
            print(f"Resuming from step {trainer.global_step}")
        
        return checkpoint
`;
  }

  /**
   * Get LoggingCallback
   */
  static getLoggingCallback(): string {
    return `"""
Logging callback for tracking training metrics
"""

import time
from typing import Any, Dict, Optional
from pathlib import Path
from .base import Callback


class LoggingCallback(Callback):
    """
    Logs training metrics to console and file
    
    Features:
    - Log at regular intervals
    - Track loss, learning rate, tokens/sec
    - Save logs to file
    - Display progress
    """
    
    def __init__(
        self,
        log_interval: int = 100,
        log_dir: str = 'logs',
        log_file: str = 'training.log',
        verbose: bool = True
    ):
        """
        Initialize logging callback
        
        Args:
            log_interval: Log every N steps
            log_dir: Directory for log files
            log_file: Log file name
            verbose: Print to console
        """
        self.log_interval = log_interval
        self.log_dir = Path(log_dir)
        self.log_file = log_file
        self.verbose = verbose
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.start_time = None
        self.step_times = []
        self.losses = []
    
    def on_train_begin(self, trainer: Any):
        """Initialize logging"""
        self.start_time = time.time()
        
        log_path = self.log_dir / self.log_file
        with open(log_path, 'w') as f:
            f.write("step,loss,lr,tokens_per_sec,elapsed_time\\n")
        
        if self.verbose:
            print(f"Training logs will be saved to: {log_path}")
            print(f"{'='*60}")
            print(f"{'Step':<10} {'Loss':<12} {'LR':<12} {'Tokens/s':<12} {'Time'}")
            print(f"{'='*60}")
    
    def on_step_end(self, trainer: Any, step: int, loss: float, metrics: Optional[Dict[str, float]] = None):
        """Log metrics at intervals"""
        self.losses.append(loss)
        
        if step % self.log_interval == 0 and step > 0:
            self._log_metrics(trainer, step, loss, metrics)
    
    def on_train_end(self, trainer: Any):
        """Log final metrics"""
        elapsed = time.time() - self.start_time
        avg_loss = sum(self.losses) / len(self.losses) if self.losses else 0
        
        if self.verbose:
            print(f"{'='*60}")
            print(f"Training completed!")
            print(f"Total time: {self._format_time(elapsed)}")
            print(f"Average loss: {avg_loss:.4f}")
            print(f"{'='*60}")
    
    def _log_metrics(self, trainer: Any, step: int, loss: float, metrics: Optional[dict]):
        """Log metrics to console and file"""
        elapsed = time.time() - self.start_time
        
        # Calculate tokens per second
        tokens_per_sec = 0
        if hasattr(trainer, 'tokens_processed'):
            tokens_per_sec = trainer.tokens_processed / elapsed
        
        # Get learning rate
        lr = trainer.optimizer.param_groups[0]['lr']
        
        # Log to file
        log_path = self.log_dir / self.log_file
        with open(log_path, 'a') as f:
            f.write(f"{step},{loss:.6f},{lr:.6e},{tokens_per_sec:.2f},{elapsed:.2f}\\n")
        
        # Log to console
        if self.verbose:
            time_str = self._format_time(elapsed)
            print(f"{step:<10} {loss:<12.4f} {lr:<12.6e} {tokens_per_sec:<12.0f} {time_str}")
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
`;
  }

  /**
   * Get checkpoint manager utility
   */
  static getCheckpointManager(): string {
    return `"""
Checkpoint management utilities
Handles checkpoint saving, loading, and cleanup
"""

import torch
import signal
import sys
from pathlib import Path
from typing import Any, Dict, Optional, List
from datetime import datetime


class CheckpointManager:
    """
    Manages model checkpoints with advanced features:
    - Automatic saving at intervals
    - Best model tracking
    - Checkpoint rotation
    - Save on interrupt (Ctrl+C)
    - Resume from checkpoint
    """
    
    def __init__(
        self,
        checkpoint_dir: str = 'checkpoints',
        save_total_limit: int = 3,
        save_best: bool = True,
        save_on_interrupt: bool = True,
        verbose: bool = True
    ):
        """
        Initialize checkpoint manager
        
        Args:
            checkpoint_dir: Directory for checkpoints
            save_total_limit: Max checkpoints to keep
            save_best: Track and save best model
            save_on_interrupt: Save on Ctrl+C
            verbose: Print messages
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_total_limit = save_total_limit
        self.save_best = save_best
        self.save_on_interrupt = save_on_interrupt
        self.verbose = verbose
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_loss = float('inf')
        self.checkpoints: List[Path] = []
        self.trainer = None
        
        # Setup interrupt handler
        if save_on_interrupt:
            signal.signal(signal.SIGINT, self._interrupt_handler)
    
    def _interrupt_handler(self, signum, frame):
        """Handle Ctrl+C interrupt"""
        print("\\n\\nTraining interrupted! Saving checkpoint...")
        if self.trainer is not None:
            self.save_checkpoint(
                self.trainer,
                self.trainer.global_step,
                self.trainer.last_loss,
                is_interrupt=True
            )
        print("Checkpoint saved. Exiting...")
        sys.exit(0)
    
    def save_checkpoint(
        self,
        trainer: Any,
        step: int,
        loss: float,
        is_best: bool = False,
        is_final: bool = False,
        is_interrupt: bool = False
    ) -> Path:
        """
        Save a checkpoint
        
        Args:
            trainer: Trainer instance
            step: Current training step
            loss: Current loss
            is_best: Whether this is the best model
            is_final: Whether this is the final checkpoint
            is_interrupt: Whether saved due to interrupt
        
        Returns:
            Path to saved checkpoint
        """
        # Store trainer reference
        self.trainer = trainer
        
        # Determine checkpoint name
        if is_final:
            checkpoint_name = 'checkpoint-final.pt'
        elif is_interrupt:
            checkpoint_name = f'checkpoint-interrupt-{step}.pt'
        elif is_best:
            checkpoint_name = 'checkpoint-best.pt'
        else:
            checkpoint_name = f'checkpoint-{step}.pt'
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Create checkpoint dictionary
        checkpoint = {
            'step': step,
            'epoch': getattr(trainer, 'epoch', 0),
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'loss': loss,
            'best_loss': self.best_loss,
            'timestamp': datetime.now().isoformat(),
            'config': getattr(trainer, 'config', None),
        }
        
        # Add scheduler state if exists
        if hasattr(trainer, 'scheduler') and trainer.scheduler is not None:
            checkpoint['scheduler_state_dict'] = trainer.scheduler.state_dict()
        
        # Add scaler state for mixed precision
        if hasattr(trainer, 'scaler') and trainer.scaler is not None:
            checkpoint['scaler_state_dict'] = trainer.scaler.state_dict()
        
        # Add RNG states for reproducibility
        checkpoint['rng_state'] = {
            'python': None,  # Python random state
            'numpy': None,   # NumPy random state
            'torch': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        if self.verbose:
            print(f"✓ Saved checkpoint: {checkpoint_path}")
        
        # Track regular checkpoints for rotation
        if not is_best and not is_final and not is_interrupt:
            self.checkpoints.append(checkpoint_path)
            self._cleanup_old_checkpoints()
        
        # Update best model
        if self.save_best and loss < self.best_loss:
            self.best_loss = loss
            best_path = self.checkpoint_dir / 'checkpoint-best.pt'
            torch.save(checkpoint, best_path)
            if self.verbose:
                print(f"✓ New best model! Loss: {loss:.4f}")
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        trainer: Any,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        Load a checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint
            trainer: Trainer instance
            load_optimizer: Whether to load optimizer state
            load_scheduler: Whether to load scheduler state
            strict: Strict state dict loading
        
        Returns:
            Checkpoint dictionary
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        if self.verbose:
            print(f"Loading checkpoint from: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        trainer.model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        # Load optimizer state
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if load_scheduler and 'scheduler_state_dict' in checkpoint:
            if hasattr(trainer, 'scheduler') and trainer.scheduler is not None:
                trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state for mixed precision
        if 'scaler_state_dict' in checkpoint:
            if hasattr(trainer, 'scaler') and trainer.scaler is not None:
                trainer.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Restore RNG states
        if 'rng_state' in checkpoint:
            rng_state = checkpoint['rng_state']
            if rng_state['torch'] is not None:
                torch.set_rng_state(rng_state['torch'])
            if rng_state['torch_cuda'] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng_state['torch_cuda'])
        
        # Restore training state
        trainer.global_step = checkpoint['step']
        trainer.epoch = checkpoint.get('epoch', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        if self.verbose:
            print(f"✓ Checkpoint loaded successfully")
            print(f"  Resuming from step: {trainer.global_step}")
            print(f"  Epoch: {trainer.epoch}")
            print(f"  Loss: {checkpoint['loss']:.4f}")
        
        return checkpoint
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to maintain save_total_limit"""
        while len(self.checkpoints) > self.save_total_limit:
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
                if self.verbose:
                    print(f"Removed old checkpoint: {old_checkpoint.name}")
    
    def find_latest_checkpoint(self) -> Optional[Path]:
        """Find the most recent checkpoint"""
        checkpoints = list(self.checkpoint_dir.glob('checkpoint-*.pt'))
        
        # Filter out special checkpoints
        checkpoints = [
            cp for cp in checkpoints
            if not cp.name.startswith('checkpoint-best')
            and not cp.name.startswith('checkpoint-final')
        ]
        
        if not checkpoints:
            return None
        
        # Sort by modification time
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return checkpoints[0]
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints with metadata"""
        checkpoints = []
        
        for cp_path in self.checkpoint_dir.glob('checkpoint-*.pt'):
            try:
                checkpoint = torch.load(cp_path, map_location='cpu')
                checkpoints.append({
                    'path': cp_path,
                    'step': checkpoint.get('step', 0),
                    'loss': checkpoint.get('loss', 0),
                    'timestamp': checkpoint.get('timestamp', 'unknown'),
                })
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not load {cp_path}: {e}")
        
        # Sort by step
        checkpoints.sort(key=lambda x: x['step'], reverse=True)
        return checkpoints


if __name__ == '__main__':
    # Test checkpoint manager
    print("Testing CheckpointManager...")
    
    # Create dummy trainer
    class DummyTrainer:
        def __init__(self):
            self.model = torch.nn.Linear(10, 10)
            self.optimizer = torch.optim.Adam(self.model.parameters())
            self.scheduler = None
            self.global_step = 0
            self.epoch = 0
            self.last_loss = 1.0
    
    trainer = DummyTrainer()
    manager = CheckpointManager(checkpoint_dir='test_checkpoints')
    
    # Save checkpoint
    manager.save_checkpoint(trainer, step=100, loss=0.5)
    
    # List checkpoints
    checkpoints = manager.list_checkpoints()
    print(f"Found {len(checkpoints)} checkpoint(s)")
    
    # Load checkpoint
    if checkpoints:
        manager.load_checkpoint(checkpoints[0]['path'], trainer)
    
    # Cleanup
    import shutil
    shutil.rmtree('test_checkpoints')
    
    print("✓ CheckpointManager tests passed!")
`;
  }

  /**
   * Get callbacks __init__.py
   */
  static getCallbacksInit(): string {
    return `"""
Training callbacks package
"""

from .base import Callback
from .checkpoint import CheckpointCallback
from .logging import LoggingCallback
from .checkpoint_manager import CheckpointManager

__all__ = [
    'Callback',
    'CheckpointCallback',
    'LoggingCallback',
    'CheckpointManager',
]
`;
  }
}
