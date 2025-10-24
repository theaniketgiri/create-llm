/**
 * Python code templates for plugin system
 */

export class PythonPluginTemplates {
  /**
   * Get Plugin base class
   */
  static getPluginBase(): string {
    return `"""
Plugin Base Class
Base class for all create-llm plugins
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class Plugin(ABC):
    """
    Base class for create-llm plugins
    
    Plugins can hook into various stages of the training lifecycle
    to add custom functionality like logging, monitoring, or data generation.
    """
    
    def __init__(self, name: str):
        """
        Initialize plugin
        
        Args:
            name: Plugin name
        """
        self.name = name
        self.enabled = True
        self.config: Dict[str, Any] = {}
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize plugin with configuration
        
        Args:
            config: Full configuration dictionary from llm.config.js
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    def on_train_begin(self, trainer) -> None:
        """
        Called when training begins
        
        Args:
            trainer: Trainer instance
        """
        pass
    
    def on_train_end(self, trainer, final_metrics: Dict[str, Any]) -> None:
        """
        Called when training ends
        
        Args:
            trainer: Trainer instance
            final_metrics: Final training metrics
        """
        pass
    
    def on_epoch_begin(self, trainer, epoch: int) -> None:
        """
        Called at the beginning of each epoch
        
        Args:
            trainer: Trainer instance
            epoch: Current epoch number
        """
        pass
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, Any]) -> None:
        """
        Called at the end of each epoch
        
        Args:
            trainer: Trainer instance
            epoch: Current epoch number
            metrics: Epoch metrics
        """
        pass
    
    def on_step_begin(self, trainer, step: int) -> None:
        """
        Called at the beginning of each training step
        
        Args:
            trainer: Trainer instance
            step: Current step number
        """
        pass
    
    def on_step_end(self, trainer, step: int, metrics: Dict[str, Any]) -> None:
        """
        Called at the end of each training step
        
        Args:
            trainer: Trainer instance
            step: Current step number
            metrics: Step metrics (loss, lr, etc.)
        """
        pass
    
    def on_validation_begin(self, trainer) -> None:
        """
        Called when validation begins
        
        Args:
            trainer: Trainer instance
        """
        pass
    
    def on_validation_end(self, trainer, metrics: Dict[str, Any]) -> None:
        """
        Called when validation ends
        
        Args:
            trainer: Trainer instance
            metrics: Validation metrics
        """
        pass
    
    def on_checkpoint_save(self, trainer, checkpoint_path: str) -> None:
        """
        Called when a checkpoint is saved
        
        Args:
            trainer: Trainer instance
            checkpoint_path: Path to saved checkpoint
        """
        pass
    
    def cleanup(self) -> None:
        """
        Cleanup plugin resources
        Called when plugin is being unloaded
        """
        pass
    
    def __repr__(self) -> str:
        return f"<Plugin: {self.name} (enabled={self.enabled})>"


class PluginError(Exception):
    """Exception raised for plugin-related errors"""
    pass
`;
  }

  /**
   * Get PluginManager class
   */
  static getPluginManager(): string {
    return `"""
Plugin Manager
Manages loading, initialization, and lifecycle of plugins
"""

import importlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from .base import Plugin, PluginError


class PluginManager:
    """
    Manages plugins for the LLM training system
    
    Loads plugins from config, initializes them, and provides
    methods to call plugin lifecycle hooks.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize plugin manager
        
        Args:
            config: Configuration dictionary from llm.config.js
        """
        self.config = config
        self.plugins: Dict[str, Plugin] = {}
        self.failed_plugins: List[str] = []
        
        # Load plugins
        self._load_plugins()
    
    def _load_plugins(self) -> None:
        """Load plugins specified in config"""
        plugin_names = self.config.get('plugins', [])
        
        if not plugin_names:
            print("ℹ️  No plugins configured")
            return
        
        print(f"\\n📦 Loading {len(plugin_names)} plugin(s)...")
        
        for plugin_name in plugin_names:
            try:
                self._load_plugin(plugin_name)
            except Exception as e:
                self.failed_plugins.append(plugin_name)
                print(f"⚠️  Warning: Failed to load plugin '{plugin_name}': {e}")
                print(f"   Training will continue without this plugin")
        
        # Summary
        loaded_count = len(self.plugins)
        failed_count = len(self.failed_plugins)
        
        if loaded_count > 0:
            print(f"✓ Successfully loaded {loaded_count} plugin(s)")
            for name in self.plugins.keys():
                print(f"  • {name}")
        
        if failed_count > 0:
            print(f"⚠️  Failed to load {failed_count} plugin(s): {', '.join(self.failed_plugins)}")
    
    def _load_plugin(self, plugin_name: str) -> None:
        """
        Load a single plugin
        
        Args:
            plugin_name: Name of the plugin to load
        
        Raises:
            PluginError: If plugin cannot be loaded
        """
        # Try to import plugin from plugins directory
        try:
            # Add plugins directory to path if not already there
            plugins_dir = Path('plugins')
            if plugins_dir.exists() and str(plugins_dir) not in sys.path:
                sys.path.insert(0, str(plugins_dir))
            
            # Import plugin module
            module_name = f"{plugin_name}_plugin"
            module = importlib.import_module(module_name)
            
            # Get plugin class (should be named like WandBPlugin, SynthexPlugin, etc.)
            class_name = ''.join(word.capitalize() for word in plugin_name.split('_')) + 'Plugin'
            
            if not hasattr(module, class_name):
                raise PluginError(f"Plugin module '{module_name}' does not have class '{class_name}'")
            
            plugin_class = getattr(module, class_name)
            
            # Instantiate plugin
            plugin = plugin_class(name=plugin_name)
            
            # Initialize plugin
            if not plugin.initialize(self.config):
                raise PluginError(f"Plugin '{plugin_name}' initialization failed")
            
            # Store plugin
            self.plugins[plugin_name] = plugin
            
        except ImportError as e:
            raise PluginError(f"Could not import plugin '{plugin_name}': {e}")
        except Exception as e:
            raise PluginError(f"Error loading plugin '{plugin_name}': {e}")
    
    def get_plugin(self, name: str) -> Optional[Plugin]:
        """
        Get a loaded plugin by name
        
        Args:
            name: Plugin name
        
        Returns:
            Plugin instance or None if not found
        """
        return self.plugins.get(name)
    
    def has_plugin(self, name: str) -> bool:
        """
        Check if a plugin is loaded
        
        Args:
            name: Plugin name
        
        Returns:
            True if plugin is loaded, False otherwise
        """
        return name in self.plugins
    
    def get_all_plugins(self) -> List[Plugin]:
        """
        Get all loaded plugins
        
        Returns:
            List of plugin instances
        """
        return list(self.plugins.values())
    
    # Lifecycle hook methods
    
    def on_train_begin(self, trainer) -> None:
        """Call on_train_begin for all plugins"""
        for plugin in self.plugins.values():
            try:
                plugin.on_train_begin(trainer)
            except Exception as e:
                print(f"⚠️  Plugin '{plugin.name}' error in on_train_begin: {e}")
    
    def on_train_end(self, trainer, final_metrics: Dict[str, Any]) -> None:
        """Call on_train_end for all plugins"""
        for plugin in self.plugins.values():
            try:
                plugin.on_train_end(trainer, final_metrics)
            except Exception as e:
                print(f"⚠️  Plugin '{plugin.name}' error in on_train_end: {e}")
    
    def on_epoch_begin(self, trainer, epoch: int) -> None:
        """Call on_epoch_begin for all plugins"""
        for plugin in self.plugins.values():
            try:
                plugin.on_epoch_begin(trainer, epoch)
            except Exception as e:
                print(f"⚠️  Plugin '{plugin.name}' error in on_epoch_begin: {e}")
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, Any]) -> None:
        """Call on_epoch_end for all plugins"""
        for plugin in self.plugins.values():
            try:
                plugin.on_epoch_end(trainer, epoch, metrics)
            except Exception as e:
                print(f"⚠️  Plugin '{plugin.name}' error in on_epoch_end: {e}")
    
    def on_step_begin(self, trainer, step: int) -> None:
        """Call on_step_begin for all plugins"""
        for plugin in self.plugins.values():
            try:
                plugin.on_step_begin(trainer, step)
            except Exception as e:
                print(f"⚠️  Plugin '{plugin.name}' error in on_step_begin: {e}")
    
    def on_step_end(self, trainer, step: int, metrics: Dict[str, Any]) -> None:
        """Call on_step_end for all plugins"""
        for plugin in self.plugins.values():
            try:
                plugin.on_step_end(trainer, step, metrics)
            except Exception as e:
                print(f"⚠️  Plugin '{plugin.name}' error in on_step_end: {e}")
    
    def on_validation_begin(self, trainer) -> None:
        """Call on_validation_begin for all plugins"""
        for plugin in self.plugins.values():
            try:
                plugin.on_validation_begin(trainer)
            except Exception as e:
                print(f"⚠️  Plugin '{plugin.name}' error in on_validation_begin: {e}")
    
    def on_validation_end(self, trainer, metrics: Dict[str, Any]) -> None:
        """Call on_validation_end for all plugins"""
        for plugin in self.plugins.values():
            try:
                plugin.on_validation_end(trainer, metrics)
            except Exception as e:
                print(f"⚠️  Plugin '{plugin.name}' error in on_validation_end: {e}")
    
    def on_checkpoint_save(self, trainer, checkpoint_path: str) -> None:
        """Call on_checkpoint_save for all plugins"""
        for plugin in self.plugins.values():
            try:
                plugin.on_checkpoint_save(trainer, checkpoint_path)
            except Exception as e:
                print(f"⚠️  Plugin '{plugin.name}' error in on_checkpoint_save: {e}")
    
    def cleanup(self) -> None:
        """Cleanup all plugins"""
        for plugin in self.plugins.values():
            try:
                plugin.cleanup()
            except Exception as e:
                print(f"⚠️  Plugin '{plugin.name}' error in cleanup: {e}")
    
    def __repr__(self) -> str:
        return f"<PluginManager: {len(self.plugins)} plugins loaded>"


def create_plugin_manager(config: Dict[str, Any]) -> PluginManager:
    """
    Create and initialize plugin manager
    
    Args:
        config: Configuration dictionary
    
    Returns:
        PluginManager instance
    """
    return PluginManager(config)
`;
  }

  /**
   * Get example plugin template
   */
  static getExamplePlugin(): string {
    return `"""
Example Plugin
Template for creating custom plugins
"""

from typing import Dict, Any
from .base import Plugin


class ExamplePlugin(Plugin):
    """
    Example plugin demonstrating the plugin interface
    
    To create your own plugin:
    1. Copy this file to plugins/your_plugin_name_plugin.py
    2. Rename the class to YourPluginNamePlugin
    3. Implement the initialize() method and any lifecycle hooks you need
    4. Add 'your_plugin_name' to the plugins list in llm.config.js
    """
    
    def __init__(self, name: str = 'example'):
        super().__init__(name)
        self.step_count = 0
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize plugin with configuration
        
        Args:
            config: Full configuration from llm.config.js
        
        Returns:
            True if successful, False otherwise
        """
        print(f"Initializing {self.name} plugin...")
        
        # Store any plugin-specific config
        self.config = config
        
        # Perform any setup needed
        # Return False if initialization fails
        
        print(f"✓ {self.name} plugin initialized")
        return True
    
    def on_train_begin(self, trainer) -> None:
        """Called when training begins"""
        print(f"[{self.name}] Training started")
        self.step_count = 0
    
    def on_step_end(self, trainer, step: int, metrics: Dict[str, Any]) -> None:
        """Called after each training step"""
        self.step_count += 1
        
        # Example: Log every 100 steps
        if step % 100 == 0:
            print(f"[{self.name}] Step {step}: loss={metrics.get('loss', 0):.4f}")
    
    def on_train_end(self, trainer, final_metrics: Dict[str, Any]) -> None:
        """Called when training ends"""
        print(f"[{self.name}] Training completed after {self.step_count} steps")
        print(f"[{self.name}] Final loss: {final_metrics.get('loss', 0):.4f}")
    
    def cleanup(self) -> None:
        """Cleanup plugin resources"""
        print(f"[{self.name}] Cleaning up...")
`;
  }

  /**
   * Get plugins __init__.py
   */
  static getPluginsInit(): string {
    return `"""
Plugins package
Extensible plugin system for create-llm
"""

from .base import Plugin, PluginError
from .plugin_manager import PluginManager, create_plugin_manager

__all__ = [
    'Plugin',
    'PluginError',
    'PluginManager',
    'create_plugin_manager',
]
`;
  }

  /**
   * Get WandB plugin
   */
  static getWandBPlugin(): string {
    return `"""
Weights & Biases (WandB) Plugin
Logs training metrics and artifacts to Weights & Biases
"""

from typing import Dict, Any, Optional
from .base import Plugin


class WandBPlugin(Plugin):
    """
    Weights & Biases integration plugin
    
    Logs training metrics, model artifacts, and system info to WandB.
    Requires wandb package to be installed and configured.
    """
    
    def __init__(self, name: str = 'wandb'):
        super().__init__(name)
        self.wandb = None
        self.run = None
        self.project_name = None
        self.run_name = None
        self.log_interval = 1
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize WandB plugin
        
        Args:
            config: Full configuration from llm.config.js
        
        Returns:
            True if successful, False otherwise
        """
        try:
            import wandb
            self.wandb = wandb
        except ImportError:
            print(f"⚠️  WandB plugin: wandb package not installed")
            print(f"   Install with: pip install wandb")
            return False
        
        # Get WandB config
        wandb_config = config.get('wandb', {})
        
        # Get project and run names
        self.project_name = wandb_config.get('project', 'llm-training')
        self.run_name = wandb_config.get('run_name', None)
        self.log_interval = wandb_config.get('log_interval', 1)
        
        # Get entity (team/user)
        entity = wandb_config.get('entity', None)
        
        # Get tags
        tags = wandb_config.get('tags', [])
        
        # Get notes
        notes = wandb_config.get('notes', None)
        
        # Check if already logged in
        try:
            # Try to initialize WandB
            self.run = self.wandb.init(
                project=self.project_name,
                name=self.run_name,
                entity=entity,
                tags=tags,
                notes=notes,
                config=config,
                resume='allow'
            )
            
            print(f"✓ WandB plugin initialized")
            print(f"  Project: {self.project_name}")
            if self.run_name:
                print(f"  Run: {self.run_name}")
            print(f"  Dashboard: {self.run.get_url()}")
            
            return True
            
        except Exception as e:
            print(f"⚠️  WandB plugin: Failed to initialize")
            print(f"   Error: {e}")
            print(f"   Make sure you're logged in: wandb login")
            return False
    
    def on_train_begin(self, trainer) -> None:
        """Called when training begins"""
        if not self.run:
            return
        
        try:
            # Log model architecture
            if hasattr(trainer.model, 'count_parameters'):
                self.wandb.log({
                    'model/parameters': trainer.model.count_parameters()
                })
            
            # Watch model (logs gradients and parameters)
            self.wandb.watch(trainer.model, log='all', log_freq=100)
            
            print(f"[{self.name}] Started tracking training")
            
        except Exception as e:
            print(f"⚠️  WandB plugin error in on_train_begin: {e}")
    
    def on_step_end(self, trainer, step: int, metrics: Dict[str, Any]) -> None:
        """
        Called after each training step
        
        Args:
            trainer: Trainer instance
            step: Current step
            metrics: Step metrics
        """
        if not self.run:
            return
        
        # Log at specified interval
        if step % self.log_interval != 0:
            return
        
        try:
            # Prepare metrics for logging
            log_dict = {
                'train/step': step,
            }
            
            # Add all metrics with proper prefixes
            for key, value in metrics.items():
                if key == 'loss':
                    log_dict['train/loss'] = value
                elif key == 'val_loss':
                    log_dict['val/loss'] = value
                elif key == 'lr' or key == 'learning_rate':
                    log_dict['train/learning_rate'] = value
                elif key == 'tokens_per_sec':
                    log_dict['performance/tokens_per_sec'] = value
                elif key == 'gpu_memory_gb':
                    log_dict['system/gpu_memory_gb'] = value
                elif key == 'epoch':
                    log_dict['train/epoch'] = value
                else:
                    log_dict[f'metrics/{key}'] = value
            
            # Log to WandB
            self.wandb.log(log_dict, step=step)
            
        except Exception as e:
            print(f"⚠️  WandB plugin error in on_step_end: {e}")
    
    def on_validation_end(self, trainer, metrics: Dict[str, Any]) -> None:
        """
        Called when validation ends
        
        Args:
            trainer: Trainer instance
            metrics: Validation metrics
        """
        if not self.run:
            return
        
        try:
            # Log validation metrics
            log_dict = {}
            
            for key, value in metrics.items():
                if key == 'loss':
                    log_dict['val/loss'] = value
                elif key == 'perplexity':
                    log_dict['val/perplexity'] = value
                elif key == 'accuracy':
                    log_dict['val/accuracy'] = value
                else:
                    log_dict[f'val/{key}'] = value
            
            if log_dict:
                self.wandb.log(log_dict)
            
        except Exception as e:
            print(f"⚠️  WandB plugin error in on_validation_end: {e}")
    
    def on_checkpoint_save(self, trainer, checkpoint_path: str) -> None:
        """
        Called when a checkpoint is saved
        
        Args:
            trainer: Trainer instance
            checkpoint_path: Path to saved checkpoint
        """
        if not self.run:
            return
        
        try:
            # Log checkpoint as artifact
            artifact = self.wandb.Artifact(
                name=f'model-checkpoint',
                type='model',
                description=f'Model checkpoint at step {trainer.global_step}'
            )
            
            artifact.add_file(checkpoint_path)
            self.run.log_artifact(artifact)
            
            print(f"[{self.name}] Logged checkpoint artifact: {checkpoint_path}")
            
        except Exception as e:
            print(f"⚠️  WandB plugin error in on_checkpoint_save: {e}")
    
    def on_train_end(self, trainer, final_metrics: Dict[str, Any]) -> None:
        """
        Called when training ends
        
        Args:
            trainer: Trainer instance
            final_metrics: Final training metrics
        """
        if not self.run:
            return
        
        try:
            # Log final metrics
            log_dict = {}
            for key, value in final_metrics.items():
                log_dict[f'final/{key}'] = value
            
            if log_dict:
                self.wandb.log(log_dict)
            
            # Log summary
            self.wandb.run.summary['training_complete'] = True
            self.wandb.run.summary['final_loss'] = final_metrics.get('loss', None)
            
            print(f"[{self.name}] Training complete. View results at: {self.run.get_url()}")
            
        except Exception as e:
            print(f"⚠️  WandB plugin error in on_train_end: {e}")
    
    def cleanup(self) -> None:
        """Cleanup WandB resources"""
        if self.run:
            try:
                self.wandb.finish()
                print(f"[{self.name}] Finished WandB run")
            except Exception as e:
                print(f"⚠️  WandB plugin error in cleanup: {e}")
`;
  }

  /**
   * Get HuggingFace plugin
   */
  static getHuggingFacePlugin(): string {
    return `"""
Hugging Face Hub Plugin
Enables easy model and tokenizer upload to Hugging Face Hub
"""

from typing import Dict, Any, Optional
from pathlib import Path
from .base import Plugin


class HuggingFacePlugin(Plugin):
    """
    Hugging Face Hub integration plugin
    
    Provides functionality to upload models and tokenizers to the
    Hugging Face Hub for easy sharing and deployment.
    """
    
    def __init__(self, name: str = 'huggingface'):
        super().__init__(name)
        self.hub = None
        self.repo_id = None
        self.private = False
        self.auto_upload = False
        self.upload_interval = 0
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize HuggingFace plugin
        
        Args:
            config: Full configuration from llm.config.js
        
        Returns:
            True if successful, False otherwise
        """
        try:
            from huggingface_hub import HfApi, create_repo
            self.hub = HfApi()
            self.create_repo = create_repo
        except ImportError:
            print(f"⚠️  HuggingFace plugin: huggingface-hub package not installed")
            print(f"   Install with: pip install huggingface-hub")
            return False
        
        # Get HuggingFace config
        hf_config = config.get('huggingface', {})
        
        # Get repository settings
        self.repo_id = hf_config.get('repo_id', None)
        self.private = hf_config.get('private', False)
        self.auto_upload = hf_config.get('auto_upload', False)
        self.upload_interval = hf_config.get('upload_interval', 0)
        
        if not self.repo_id:
            print(f"⚠️  HuggingFace plugin: No repo_id configured")
            print(f"   Add 'repo_id' to huggingface config in llm.config.js")
            return False
        
        # Check authentication
        try:
            # Try to get user info to verify authentication
            user_info = self.hub.whoami()
            username = user_info['name']
            
            print(f"✓ HuggingFace plugin initialized")
            print(f"  User: {username}")
            print(f"  Repository: {self.repo_id}")
            print(f"  Private: {self.private}")
            
            # Create repository if it doesn't exist
            try:
                self.create_repo(
                    repo_id=self.repo_id,
                    private=self.private,
                    exist_ok=True
                )
                print(f"  ✓ Repository ready")
            except Exception as e:
                print(f"  ⚠️  Could not create repository: {e}")
            
            return True
            
        except Exception as e:
            print(f"⚠️  HuggingFace plugin: Not authenticated")
            print(f"   Error: {e}")
            print(f"   Login with: huggingface-cli login")
            return False
    
    def on_train_begin(self, trainer) -> None:
        """Called when training begins"""
        if not self.hub:
            return
        
        try:
            print(f"[{self.name}] Model will be uploaded to: https://huggingface.co/{self.repo_id}")
            
            # Create model card
            self._create_model_card(trainer)
            
        except Exception as e:
            print(f"⚠️  HuggingFace plugin error in on_train_begin: {e}")
    
    def on_checkpoint_save(self, trainer, checkpoint_path: str) -> None:
        """
        Called when a checkpoint is saved
        
        Args:
            trainer: Trainer instance
            checkpoint_path: Path to saved checkpoint
        """
        if not self.hub or not self.auto_upload:
            return
        
        # Only upload at specified intervals
        if self.upload_interval > 0:
            if trainer.global_step % self.upload_interval != 0:
                return
        
        try:
            print(f"[{self.name}] Uploading checkpoint to Hugging Face Hub...")
            
            # Upload checkpoint
            self.hub.upload_file(
                path_or_fileobj=checkpoint_path,
                path_in_repo=f"checkpoints/{Path(checkpoint_path).name}",
                repo_id=self.repo_id,
                repo_type="model"
            )
            
            print(f"[{self.name}] ✓ Checkpoint uploaded")
            
        except Exception as e:
            print(f"⚠️  HuggingFace plugin error in on_checkpoint_save: {e}")
    
    def on_train_end(self, trainer, final_metrics: Dict[str, Any]) -> None:
        """
        Called when training ends
        
        Args:
            trainer: Trainer instance
            final_metrics: Final training metrics
        """
        if not self.hub:
            return
        
        try:
            print(f"[{self.name}] Uploading final model to Hugging Face Hub...")
            
            # Update model card with final metrics
            self._update_model_card(trainer, final_metrics)
            
            print(f"[{self.name}] ✓ Training complete")
            print(f"[{self.name}] View model at: https://huggingface.co/{self.repo_id}")
            
        except Exception as e:
            print(f"⚠️  HuggingFace plugin error in on_train_end: {e}")
    
    def _create_model_card(self, trainer) -> None:
        """
        Create model card for the repository
        
        Args:
            trainer: Trainer instance
        """
        try:
            # Get model info
            model_params = trainer.model.count_parameters() if hasattr(trainer.model, 'count_parameters') else 'Unknown'
            
            # Create model card content
            model_card = f"""---
language: en
license: apache-2.0
tags:
- llm
- gpt
- create-llm
---

# {self.repo_id.split('/')[-1]}

This model was trained using [create-llm](https://github.com/theaniketgiri/create-llm).

## Model Details

- **Model Type:** Language Model
- **Parameters:** {model_params:,} if isinstance(model_params, int) else model_params
- **Training Framework:** PyTorch
- **Created with:** create-llm

## Training Details

Training is in progress. Final metrics will be updated upon completion.

## Usage

\`\`\`python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{self.repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{self.repo_id}")

# Generate text
inputs = tokenizer("Once upon a time", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
\`\`\`

## Citation

If you use this model, please cite:

\`\`\`bibtex
@misc{{{self.repo_id.replace('/', '-')},
  author = {{Your Name}},
  title = {{{self.repo_id.split('/')[-1]}}},
  year = {{2024}},
  publisher = {{Hugging Face}},
  howpublished = {{\\\\url{{https://huggingface.co/{self.repo_id}}}}}
}}
\`\`\`
"""
            
            # Upload model card
            self.hub.upload_file(
                path_or_fileobj=model_card.encode(),
                path_in_repo="README.md",
                repo_id=self.repo_id,
                repo_type="model"
            )
            
            print(f"[{self.name}] ✓ Model card created")
            
        except Exception as e:
            print(f"⚠️  Error creating model card: {e}")
    
    def _update_model_card(self, trainer, final_metrics: Dict[str, Any]) -> None:
        """
        Update model card with final metrics
        
        Args:
            trainer: Trainer instance
            final_metrics: Final training metrics
        """
        try:
            # Get model info
            model_params = trainer.model.count_parameters() if hasattr(trainer.model, 'count_parameters') else 'Unknown'
            
            # Format metrics
            metrics_str = "\\n".join([f"- **{k}:** {v:.4f}" if isinstance(v, float) else f"- **{k}:** {v}" 
                                     for k, v in final_metrics.items()])
            
            # Create updated model card
            model_card = f"""---
language: en
license: apache-2.0
tags:
- llm
- gpt
- create-llm
---

# {self.repo_id.split('/')[-1]}

This model was trained using [create-llm](https://github.com/theaniketgiri/create-llm).

## Model Details

- **Model Type:** Language Model
- **Parameters:** {model_params:,} if isinstance(model_params, int) else model_params
- **Training Framework:** PyTorch
- **Created with:** create-llm

## Training Details

### Final Metrics

{metrics_str}

## Usage

\`\`\`python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{self.repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{self.repo_id}")

# Generate text
inputs = tokenizer("Once upon a time", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
\`\`\`

## Citation

If you use this model, please cite:

\`\`\`bibtex
@misc{{{self.repo_id.replace('/', '-')},
  author = {{Your Name}},
  title = {{{self.repo_id.split('/')[-1]}}},
  year = {{2024}},
  publisher = {{Hugging Face}},
  howpublished = {{\\\\url{{https://huggingface.co/{self.repo_id}}}}}
}}
\`\`\`
"""
            
            # Upload updated model card
            self.hub.upload_file(
                path_or_fileobj=model_card.encode(),
                path_in_repo="README.md",
                repo_id=self.repo_id,
                repo_type="model"
            )
            
            print(f"[{self.name}] ✓ Model card updated with final metrics")
            
        except Exception as e:
            print(f"⚠️  Error updating model card: {e}")
    
    def upload_model(self, model_path: str) -> bool:
        """
        Upload model to Hugging Face Hub
        
        Args:
            model_path: Path to model directory or file
        
        Returns:
            True if successful, False otherwise
        """
        if not self.hub:
            print(f"⚠️  HuggingFace plugin not initialized")
            return False
        
        try:
            print(f"[{self.name}] Uploading model from {model_path}...")
            
            # Upload model files
            self.hub.upload_folder(
                folder_path=model_path,
                repo_id=self.repo_id,
                repo_type="model"
            )
            
            print(f"[{self.name}] ✓ Model uploaded successfully")
            print(f"[{self.name}] View at: https://huggingface.co/{self.repo_id}")
            
            return True
            
        except Exception as e:
            print(f"⚠️  Error uploading model: {e}")
            return False
    
    def upload_tokenizer(self, tokenizer_path: str) -> bool:
        """
        Upload tokenizer to Hugging Face Hub
        
        Args:
            tokenizer_path: Path to tokenizer file
        
        Returns:
            True if successful, False otherwise
        """
        if not self.hub:
            print(f"⚠️  HuggingFace plugin not initialized")
            return False
        
        try:
            print(f"[{self.name}] Uploading tokenizer from {tokenizer_path}...")
            
            # Upload tokenizer file
            self.hub.upload_file(
                path_or_fileobj=tokenizer_path,
                path_in_repo="tokenizer.json",
                repo_id=self.repo_id,
                repo_type="model"
            )
            
            print(f"[{self.name}] ✓ Tokenizer uploaded successfully")
            
            return True
            
        except Exception as e:
            print(f"⚠️  Error uploading tokenizer: {e}")
            return False
    
    def cleanup(self) -> None:
        """Cleanup HuggingFace resources"""
        if self.hub:
            print(f"[{self.name}] HuggingFace plugin cleanup complete")
`;
  }

  /**
   * Get plugins README
   */
  static getPluginsReadme(): string {
    return `# Plugins

This directory contains plugins for extending create-llm functionality.

## Available Plugins

### Built-in Plugins

- **wandb**: Weights & Biases integration for experiment tracking
- **synthex**: SynthexAI integration for synthetic data generation
- **huggingface**: Hugging Face Hub integration for model sharing

### Custom Plugins

You can create custom plugins by following the example in \`example_plugin.py\`.

## Creating a Custom Plugin

1. Create a new file: \`plugins/my_plugin_plugin.py\`
2. Define a class that inherits from \`Plugin\`:

\`\`\`python
from plugins.base import Plugin

class MyPluginPlugin(Plugin):
    def __init__(self, name='my_plugin'):
        super().__init__(name)
    
    def initialize(self, config):
        # Setup your plugin
        return True
    
    def on_step_end(self, trainer, step, metrics):
        # Do something on each training step
        pass
\`\`\`

3. Add your plugin to \`llm.config.js\`:

\`\`\`javascript
module.exports = {
  // ... other config
  plugins: [
    'my_plugin',
  ],
};
\`\`\`

## Plugin Lifecycle Hooks

Plugins can implement the following hooks:

- \`initialize(config)\`: Called when plugin is loaded
- \`on_train_begin(trainer)\`: Called when training starts
- \`on_train_end(trainer, final_metrics)\`: Called when training ends
- \`on_epoch_begin(trainer, epoch)\`: Called at start of each epoch
- \`on_epoch_end(trainer, epoch, metrics)\`: Called at end of each epoch
- \`on_step_begin(trainer, step)\`: Called at start of each step
- \`on_step_end(trainer, step, metrics)\`: Called at end of each step
- \`on_validation_begin(trainer)\`: Called when validation starts
- \`on_validation_end(trainer, metrics)\`: Called when validation ends
- \`on_checkpoint_save(trainer, checkpoint_path)\`: Called when checkpoint is saved
- \`cleanup()\`: Called when plugin is unloaded

## Error Handling

If a plugin fails to load or encounters an error during execution:
- A warning will be displayed
- Training will continue without the plugin
- Other plugins will not be affected

## Examples

See \`example_plugin.py\` for a complete example of a custom plugin.
`;
  }
}
