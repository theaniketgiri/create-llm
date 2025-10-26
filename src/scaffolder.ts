import * as fs from 'fs';
import * as path from 'path';
import ora, { Ora } from 'ora';
import chalk from 'chalk';
import { Template } from './types/template';
import { ProjectConfig } from './prompts';
import { ConfigGenerator } from './config-generator';

/**
 * ScaffolderEngine creates the project structure and files
 */
export class ScaffolderEngine {
    private projectPath: string;
    private spinner: Ora;

    constructor(projectPath: string) {
        this.projectPath = projectPath;
        this.spinner = ora();
    }

    /**
     * Create complete project structure
     */
    async createProjectStructure(config: ProjectConfig, template: Template): Promise<void> {
        this.spinner.start('Creating project structure...');

        try {
            // Create root directory
            this.createDirectory(this.projectPath);

            // Create directory structure
            this.createDirectories();

            this.spinner.succeed(chalk.green('Project structure created'));
        } catch (error) {
            this.spinner.fail(chalk.red('Failed to create project structure'));
            throw error;
        }
    }

    /**
     * Create all necessary directories
     */
    private createDirectories(): void {
        const directories = [
            'data',
            'data/raw',
            'data/processed',
            'models',
            'models/architectures',
            'tokenizer',
            'training',
            'training/callbacks',
            'training/dashboard',
            'training/dashboard/templates',
            'evaluation',
            'checkpoints',
            'logs',
            'plugins',
            'utils',
            'tests'
        ];

        for (const dir of directories) {
            this.createDirectory(path.join(this.projectPath, dir));
        }
    }

    /**
     * Create a directory if it doesn't exist
     */
    private createDirectory(dirPath: string): void {
        if (!fs.existsSync(dirPath)) {
            fs.mkdirSync(dirPath, { recursive: true });
        }
    }

    /**
     * Create a file with content
     */
    private createFile(filePath: string, content: string): void {
        const fullPath = path.join(this.projectPath, filePath);
        const dir = path.dirname(fullPath);

        // Ensure directory exists
        if (!fs.existsSync(dir)) {
            fs.mkdirSync(dir, { recursive: true });
        }

        fs.writeFileSync(fullPath, content, 'utf-8');
    }

    /**
     * Copy template files to project
     */
    async copyTemplateFiles(config: ProjectConfig, template: Template): Promise<void> {
        this.spinner.start('Generating project files...');

        try {
            // Generate Python files
            this.generatePythonFiles(config, template);

            // Generate configuration files
            this.generateConfigFiles(config, template);

            // Generate documentation
            this.generateDocumentation(config, template);

            this.spinner.succeed(chalk.green('Project files generated'));
        } catch (error) {
            this.spinner.fail(chalk.red('Failed to generate project files'));
            throw error;
        }
    }

    /**
     * Generate Python project files
     */
    private generatePythonFiles(config: ProjectConfig, template: Template): void {
        // Import Python templates
        const { PythonTemplates } = require('./python-templates');
        const { PythonDatasetTemplates } = require('./python-dataset-templates');
        const { PythonCallbackTemplates } = require('./python-callback-templates');
        const { PythonTrainerTemplates } = require('./python-trainer-templates');
        const { PythonDashboardTemplates } = require('./python-dashboard-templates');

        const { PythonPluginTemplates } = require('./python-plugin-templates');
        const { PythonErrorTemplates } = require('./python-error-templates');

        // Create __init__.py files
        this.createFile('models/__init__.py', PythonTemplates.getModelsInit());
        this.createFile('models/architectures/__init__.py', PythonTemplates.getArchitecturesInit());
        this.createFile('data/__init__.py', PythonDatasetTemplates.getDataInit());
        this.createFile('training/__init__.py', PythonTrainerTemplates.getTrainingInit());
        this.createFile('training/callbacks/__init__.py', PythonCallbackTemplates.getCallbacksInit());
        this.createFile('training/dashboard/__init__.py', PythonDashboardTemplates.getDashboardInit());
        this.createFile('evaluation/__init__.py', '');
        this.createFile('plugins/__init__.py', PythonPluginTemplates.getPluginsInit());
        this.createFile('utils/__init__.py', PythonErrorTemplates.getErrorInit());

        // Create model architecture files
        this.createFile('models/architectures/gpt.py', PythonTemplates.getGPTArchitecture());
        this.createFile('models/architectures/nano.py', PythonTemplates.getNanoModel());
        this.createFile('models/architectures/tiny.py', PythonTemplates.getTinyModel());
        this.createFile('models/architectures/small.py', PythonTemplates.getSmallModel());
        this.createFile('models/architectures/base.py', PythonTemplates.getBaseModel());
        this.createFile('models/config.py', PythonTemplates.getModelConfig());

        // Create dataset files
        this.createFile('data/dataset.py', PythonDatasetTemplates.getLLMDataset());

        // Create callback files
        this.createFile('training/callbacks/base.py', PythonCallbackTemplates.getBaseCallback());
        this.createFile('training/callbacks/checkpoint.py', PythonCallbackTemplates.getCheckpointCallback());
        this.createFile('training/callbacks/logging.py', PythonCallbackTemplates.getLoggingCallback());
        this.createFile('training/callbacks/checkpoint_manager.py', PythonCallbackTemplates.getCheckpointManager());

        // Create dashboard files
        this.createFile('training/dashboard/dashboard_server.py', PythonDashboardTemplates.getDashboardServer());
        this.createFile('training/dashboard/dashboard_callback.py', PythonDashboardTemplates.getDashboardCallback());
        this.createFile('training/dashboard/templates/dashboard.html', PythonDashboardTemplates.getDashboardHTML());

        // Create plugin files
        this.createFile('plugins/base.py', PythonPluginTemplates.getPluginBase());
        this.createFile('plugins/plugin_manager.py', PythonPluginTemplates.getPluginManager());
        this.createFile('plugins/example_plugin.py', PythonPluginTemplates.getExamplePlugin());
        this.createFile('plugins/README.md', PythonPluginTemplates.getPluginsReadme());

        // Create built-in plugin files
        this.createFile('plugins/wandb_plugin.py', PythonPluginTemplates.getWandBPlugin());
        this.createFile('plugins/huggingface_plugin.py', PythonPluginTemplates.getHuggingFacePlugin());

        // Create error handling files
        this.createFile('utils/exceptions.py', PythonErrorTemplates.getCustomExceptions());
        this.createFile('utils/handlers.py', PythonErrorTemplates.getErrorHandlers());

        // Create test files
        const { PythonTestTemplates } = require('./python-test-templates');
        this.createFile('pytest.ini', PythonTestTemplates.getPytestConfig());
        this.createFile('tests/conftest.py', PythonTestTemplates.getConftest());
        this.createFile('tests/test_models.py', PythonTestTemplates.getModelTests());
        this.createFile('tests/test_config.py', PythonTestTemplates.getConfigTests());
        this.createFile('tests/test_errors.py', PythonTestTemplates.getErrorTests());
        this.createFile('tests/test_integration.py', PythonTestTemplates.getIntegrationTests());
        this.createFile('tests/README.md', PythonTestTemplates.getTestsReadme());

        // Create trainer file
        this.createFile('training/trainer.py', PythonTrainerTemplates.getTrainer());

        // Create placeholder Python files
        this.createFile('data/prepare.py', this.getDataPrepareTemplate());
        this.createFile('tokenizer/train.py', this.getTokenizerTrainTemplate());
        this.createFile('training/train.py', this.getTrainingScriptTemplate());
        this.createFile('evaluation/evaluate.py', this.getEvaluationScriptTemplate());
        this.createFile('evaluation/generate.py', this.getGenerationScriptTemplate());
        this.createFile('chat.py', this.getChatScriptTemplate());
        this.createFile('deploy.py', this.getDeployScriptTemplate());
        this.createFile('compare.py', this.getCompareScriptTemplate());
    }

    /**
     * Generate configuration files
     */
    private generateConfigFiles(config: ProjectConfig, template: Template): void {
        // Generate llm.config.js
        const configGenerator = new ConfigGenerator();
        const configContent = configGenerator.generateConfigWithTips(config, template);
        this.createFile('llm.config.js', configContent);

        // Generate requirements.txt
        this.createFile('requirements.txt', this.getRequirementsTxt(config));

        // Generate .gitignore
        this.createFile('.gitignore', this.getGitignore());

        // Generate sample data file
        this.createFile('data/raw/sample.txt', this.getSampleData(template));
    }

    /**
     * Generate documentation files
     */
    private generateDocumentation(config: ProjectConfig, template: Template): void {
        // Generate README.md
        this.createFile('README.md', this.getReadmeTemplate(config, template));
    }

    /**
     * Display progress with spinner
     */
    displayProgress(message: string): void {
        this.spinner.text = message;
    }

    // Template getters

    private getDataPrepareTemplate(): string {
        const { PythonDataTemplates } = require('./python-tokenizer-templates');
        return PythonDataTemplates.getDataPrepareScript();
    }

    private getTokenizerTrainTemplate(): string {
        const { PythonTokenizerTemplates } = require('./python-tokenizer-templates');
        return PythonTokenizerTemplates.getTokenizerTrainScript();
    }

    private getTrainingScriptTemplate(): string {
        return `#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main training script
Trains the LLM model with full configuration support
"""

import argparse
import sys
import os
import torch
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import ConfigLoader, load_model_from_config
from data import LLMDataset, create_dataloader
from training import Trainer
from training.callbacks import CheckpointCallback, LoggingCallback, CheckpointManager


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train LLM model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default config
  python training/train.py
  
  # Resume from checkpoint
  python training/train.py --resume checkpoints/checkpoint-5000.pt
  
  # Use custom config
  python training/train.py --config my-config.js
  
  # Enable live dashboard
  python training/train.py --dashboard
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='llm.config.js',
        help='Path to config file (default: llm.config.js)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        help='Resume from checkpoint path'
    )
    parser.add_argument(
        '--dashboard',
        action='store_true',
        help='Enable live training dashboard'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Device to train on (default: auto)'
    )
    
    return parser.parse_args()


def setup_device(device_arg: str) -> str:
    """Setup training device"""
    if device_arg == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_arg
    
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        device = 'cpu'
    
    return device


def load_data(config: ConfigLoader, device: str):
    """Load training and validation data"""
    data_config = config.get_data_config()
    training_config = config.get_training_config()
    
    # Check if processed data exists
    train_path = Path('data/processed/train.pt')
    val_path = Path('data/processed/val.pt')
    
    if not train_path.exists():
        print("❌ Training data not found!")
        print("   Please run: python data/prepare.py")
        sys.exit(1)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = LLMDataset(
        str(train_path),
        max_length=data_config.get('max_length', 512)
    )
    
    val_dataset = None
    if val_path.exists():
        val_dataset = LLMDataset(
            str(val_path),
            max_length=data_config.get('max_length', 512)
        )
    
    # Create data loaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=training_config.get('batch_size', 32),
        shuffle=data_config.get('shuffle', True),
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=(device == 'cuda')
    )
    
    val_loader = None
    if val_dataset:
        val_loader = create_dataloader(
            val_dataset,
            batch_size=training_config.get('batch_size', 32),
            shuffle=False,
            num_workers=0,
            pin_memory=(device == 'cuda')
        )
    
    print(f"✓ Loaded {len(train_dataset)} training examples")
    if val_dataset:
        print(f"✓ Loaded {len(val_dataset)} validation examples")
    
    return train_loader, val_loader


def create_callbacks(config: ConfigLoader, enable_dashboard: bool = False):
    """Create training callbacks"""
    callbacks = []
    
    # Checkpoint callback
    checkpoint_config = config.get_checkpoint_config()
    training_config = config.get_training_config()
    
    checkpoint_callback = CheckpointCallback(
        checkpoint_dir='checkpoints',
        save_interval=training_config.get('save_interval', 5000),
        save_total_limit=checkpoint_config.get('save_total_limit', 3),
        save_best=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Logging callback
    logging_config = config.get_logging_config()
    logging_callback = LoggingCallback(
        log_interval=logging_config.get('log_interval', 100),
        log_dir=logging_config.get('log_dir', 'logs'),
        verbose=True
    )
    callbacks.append(logging_callback)
    
    # Dashboard callback (optional)
    if enable_dashboard:
        try:
            from training.dashboard import DashboardCallback
            dashboard_callback = DashboardCallback(port=5000)
            callbacks.append(dashboard_callback)
            print("\\n📊 Dashboard enabled at http://localhost:5000")
        except ImportError:
            print("\\n⚠️  Dashboard dependencies not installed. Install with: pip install flask flask-socketio")
    
    return callbacks


def main():
    """Main training function"""
    args = parse_args()
    
    print("=" * 60)
    print("🚀 LLM Training")
    print("=" * 60)
    
    try:
        # Load config
        print(f"\\nLoading config from: {args.config}")
        config = ConfigLoader(args.config)
        print("✓ Config loaded successfully")
        
        # Setup device
        device = setup_device(args.device)
        print(f"\\nDevice: {device}")
        if device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Load model
        print("\\nLoading model...")
        model = load_model_from_config(args.config)
        num_params = model.count_parameters()
        print(f"✓ Model loaded: {num_params:,} parameters")
        
        # Load data
        print()
        train_loader, val_loader = load_data(config, device)
        
        # Check for potential overfitting
        num_examples = len(train_loader.dataset)
        params_per_example = num_params / num_examples
        if params_per_example > 1000:
            print(f"\\n⚠️  WARNING: Model may be too large for dataset!")
            print(f"   Model: {num_params:,} parameters")
            print(f"   Data: {num_examples:,} examples")
            print(f"   Ratio: {params_per_example:,.0f} params/example")
            print(f"   Recommendation: Use smaller model or add more data\\n")
        
        # Create callbacks
        callbacks = create_callbacks(config, enable_dashboard=args.dashboard)
        
        # Create trainer
        print("\\nInitializing trainer...")
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config.config,
            callbacks=callbacks,
            device=device
        )
        
        # Resume from checkpoint if specified
        if args.resume:
            print(f"\\nResuming from checkpoint: {args.resume}")
            checkpoint_manager = CheckpointManager()
            checkpoint_manager.load_checkpoint(args.resume, trainer)
        
        # Start training
        print("\\n" + "=" * 60)
        print("Starting training...")
        print("=" * 60)
        trainer.train()
        
        print("\\n✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\\n\\n⚠️  Training interrupted by user")
        sys.exit(0)
    
    except Exception as e:
        print(f"\\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
`;
    }

    private getEvaluationScriptTemplate(): string {
        return `#!/usr/bin/env python3
"""
Model evaluation script
Evaluates trained model on validation set with perplexity and metrics
"""

import argparse
import sys
import torch
import time
import json
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import load_model_from_config
from data import LLMDataset, create_dataloader


class Evaluator:
    """
    Model evaluator for computing metrics
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Log model configuration for diagnostics
        if hasattr(model, 'config'):
            print(f"Model Configuration:")
            print(f"  max_length: {model.config.max_length}")
            print(f"  vocab_size: {model.config.vocab_size}")
            if hasattr(model, 'position_embedding'):
                print(f"  Position embedding size: {model.position_embedding.num_embeddings}")
    
    @torch.no_grad()
    def evaluate(self, dataloader):
        """
        Evaluate model on dataset
        
        Returns:
            dict with metrics: loss, perplexity, tokens_per_sec
        """
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        
        start_time = time.time()
        
        print("\\nEvaluating...")
        for batch in tqdm(dataloader, desc="Evaluation"):
            # Move to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs['loss']
            
            # Accumulate
            total_loss += loss.item()
            total_tokens += batch['input_ids'].numel()
            num_batches += 1
        
        elapsed = time.time() - start_time
        
        # Calculate metrics
        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        tokens_per_sec = total_tokens / elapsed
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'tokens_per_sec': tokens_per_sec,
            'total_tokens': total_tokens,
            'num_batches': num_batches,
            'elapsed_time': elapsed
        }


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Evaluate LLM model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on validation set
  python evaluation/evaluate.py --checkpoint checkpoints/final.pt
  
  # Evaluate on custom data
  python evaluation/evaluate.py --checkpoint checkpoints/final.pt --data data/processed/test.pt
  
  # Save results to file
  python evaluation/evaluate.py --checkpoint checkpoints/final.pt --output results.json
        """
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/processed/val.pt',
        help='Path to evaluation data (default: data/processed/val.pt)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for evaluation (default: 32)'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Device to use (default: auto)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Save results to JSON file'
    )
    
    return parser.parse_args()


def load_checkpoint(checkpoint_path: str, device: str):
    """Load model from checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model
    model = load_model_from_config()
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Model loaded from step {checkpoint['step']}")
    print(f"  Training loss: {checkpoint['loss']:.4f}")
    
    # Validate position embedding size matches config
    if hasattr(model, 'config') and hasattr(model, 'position_embedding'):
        config_max_length = model.config.max_length
        actual_max_length = model.position_embedding.num_embeddings
        
        if config_max_length != actual_max_length:
            print(f"\\n⚠️  Position embedding size mismatch detected!")
            print(f"   Config max_length: {config_max_length} | Actual: {actual_max_length}")
            print(f"   Using actual position embedding size: {actual_max_length}")
            model.config.max_length = actual_max_length
    
    return model, checkpoint


def main():
    """Main evaluation function"""
    args = parse_args()
    
    print("=" * 60)
    print("📊 Model Evaluation")
    print("=" * 60)
    
    try:
        # Setup device
        if args.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = args.device
        
        print(f"\\nDevice: {device}")
        
        # Check data exists
        data_path = Path(args.data)
        if not data_path.exists():
            print(f"❌ Data file not found: {args.data}")
            sys.exit(1)
        
        # Load checkpoint
        model, checkpoint = load_checkpoint(args.checkpoint, device)
        
        # Extract max_length from model configuration
        max_length = model.config.max_length if hasattr(model, 'config') else 512
        print(f"Using max_length: {max_length}")
        
        # Load data
        print(f"\\nLoading data: {args.data}")
        dataset = LLMDataset(str(data_path))
        dataloader = create_dataloader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device == 'cuda'),
            max_length=max_length
        )
        print(f"✓ Loaded {len(dataset)} examples")
        
        # Create evaluator
        evaluator = Evaluator(model, device)
        
        # Evaluate with enhanced error handling
        try:
            metrics = evaluator.evaluate(dataloader)
        except IndexError as e:
            if "index out of range" in str(e):
                print("\\n" + "=" * 60)
                print("❌ POSITION EMBEDDING INDEX ERROR")
                print("=" * 60)
                print(f"Model max_length: {max_length}")
                print(f"\\nThis error occurs when validation sequences exceed the model's maximum length.")
                print(f"\\nSolutions:")
                print(f"  1. Reprocess validation data with max_length={max_length}")
                print(f"  2. Increase model's max_length in config and retrain")
                print(f"  3. Check data preprocessing pipeline")
                print("=" * 60)
            raise
        
        # Display results
        print("\\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        print(f"Loss:           {metrics['loss']:.4f}")
        print(f"Perplexity:     {metrics['perplexity']:.2f}")
        print(f"Tokens/sec:     {metrics['tokens_per_sec']:.0f}")
        print(f"Total tokens:   {metrics['total_tokens']:,}")
        print(f"Batches:        {metrics['num_batches']}")
        print(f"Time:           {metrics['elapsed_time']:.2f}s")
        print("=" * 60)
        
        # Save results if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            results = {
                'checkpoint': args.checkpoint,
                'data': args.data,
                'metrics': metrics,
                'checkpoint_step': checkpoint['step'],
                'checkpoint_loss': checkpoint['loss']
            }
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\\n✓ Results saved to: {output_path}")
        
        print("\\n✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"\\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
`;
    }

    private getGenerationScriptTemplate(): string {
        return `#!/usr/bin/env python3
"""
Text generation script
Generates text using trained model with various sampling strategies
"""

import argparse
import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import load_model_from_config
from tokenizers import Tokenizer


class TextGenerator:
    """
    Text generator with multiple sampling strategies
    """
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str = "",
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        num_return_sequences: int = 1
    ):
        """
        Generate text with various sampling strategies
        
        Args:
            prompt: Starting text
            max_length: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling (0 = disabled)
            top_p: Nucleus sampling (1.0 = disabled)
            num_return_sequences: Number of sequences to generate
        
        Returns:
            List of generated texts
        """
        # Encode prompt
        if prompt:
            encoding = self.tokenizer.encode(prompt)
            input_ids = torch.tensor([encoding.ids], dtype=torch.long, device=self.device)
        else:
            # Start with BOS token if available
            input_ids = torch.tensor([[0]], dtype=torch.long, device=self.device)
        
        # Generate multiple sequences
        generated_sequences = []
        
        for _ in range(num_return_sequences):
            generated = self._generate_sequence(
                input_ids.clone(),
                max_length,
                temperature,
                top_k,
                top_p
            )
            generated_sequences.append(generated)
        
        return generated_sequences
    
    def _generate_sequence(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float,
        top_k: int,
        top_p: float
    ) -> str:
        """Generate a single sequence"""
        for _ in range(max_length):
            # Get model predictions
            outputs = self.model(input_ids)
            logits = outputs['logits']
            
            # Get logits for last token
            next_token_logits = logits[0, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample from distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            # Check for EOS token (assuming token 2 is EOS)
            if next_token.item() == 2:
                break
        
        # Decode
        generated_ids = input_ids[0].tolist()
        generated_text = self.tokenizer.decode(generated_ids)
        
        return generated_text


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Generate text with LLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate with prompt
  python evaluation/generate.py --checkpoint checkpoints/final.pt --prompt "Once upon a time"
  
  # Higher temperature for more randomness
  python evaluation/generate.py --checkpoint checkpoints/final.pt --prompt "Hello" --temperature 1.5
  
  # Use top-k sampling
  python evaluation/generate.py --checkpoint checkpoints/final.pt --prompt "The" --top-k 40
  
  # Generate multiple sequences
  python evaluation/generate.py --checkpoint checkpoints/final.pt --prompt "AI is" --num-sequences 3
        """
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default='',
        help='Starting prompt for generation'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=100,
        help='Maximum tokens to generate (default: 100)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Sampling temperature, higher = more random (default: 1.0)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=50,
        help='Top-k sampling, 0 = disabled (default: 50)'
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.95,
        help='Nucleus sampling, 1.0 = disabled (default: 0.95)'
    )
    parser.add_argument(
        '--num-sequences',
        type=int,
        default=1,
        help='Number of sequences to generate (default: 1)'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Device to use (default: auto)'
    )
    
    return parser.parse_args()


def main():
    """Main generation function"""
    args = parse_args()
    
    print("=" * 60)
    print("✨ Text Generation")
    print("=" * 60)
    
    try:
        # Setup device
        if args.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = args.device
        
        print(f"\\nDevice: {device}")
        
        # Load checkpoint
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        
        # Load model
        model = load_model_from_config()
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model loaded from step {checkpoint['step']}")
        
        # Load tokenizer
        tokenizer_path = Path('tokenizer/tokenizer.json')
        if not tokenizer_path.exists():
            print("❌ Tokenizer not found!")
            print("   Please train tokenizer first: python tokenizer/train.py --data data/raw/sample.txt")
            sys.exit(1)
        
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        print("✓ Tokenizer loaded")
        
        # Create generator
        generator = TextGenerator(model, tokenizer, device)
        
        # Display settings
        print("\\n" + "-" * 60)
        print("Generation Settings:")
        print(f"  Prompt: '{args.prompt}'")
        print(f"  Max length: {args.max_length}")
        print(f"  Temperature: {args.temperature}")
        print(f"  Top-k: {args.top_k}")
        print(f"  Top-p: {args.top_p}")
        print(f"  Sequences: {args.num_sequences}")
        print("-" * 60)
        
        # Generate
        print("\\nGenerating...\\n")
        generated_texts = generator.generate(
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_return_sequences=args.num_sequences
        )
        
        # Display results
        print("=" * 60)
        print("Generated Text:")
        print("=" * 60)
        
        for i, text in enumerate(generated_texts, 1):
            if args.num_sequences > 1:
                print(f"\\n[Sequence {i}]")
            print(text)
            if i < len(generated_texts):
                print("\\n" + "-" * 60)
        
        print("\\n" + "=" * 60)
        print("\\n✅ Generation completed successfully!")
        
    except Exception as e:
        print(f"\\n❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
`;
    }

    private getChatScriptTemplate(): string {
        return `#!/usr/bin/env python3
"""
Interactive chat interface
Chat with your trained LLM model in the terminal

Features:
- Interactive terminal session with conversation history
- Context maintenance across multiple turns
- Loading indicator during generation
- Multiple sampling strategies (temperature, top-k, top-p)
- Commands: exit, quit, clear, reset, help
"""

import argparse
import sys
import torch
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models import load_model_from_config, ConfigLoader
from tokenizers import Tokenizer


class ChatSession:
    """
    Interactive chat session with LLM
    Maintains conversation context and generates responses
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str = 'cuda',
        max_context_length: int = 512,
        context_window: int = 10
    ):
        """
        Initialize chat session
        
        Args:
            model: The LLM model
            tokenizer: Tokenizer for encoding/decoding
            device: Device to run on ('cuda' or 'cpu')
            max_context_length: Maximum token length for context
            context_window: Number of conversation turns to keep
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_context_length = max_context_length
        self.context_window = context_window
        
        self.model.to(device)
        self.model.eval()
        
        # Conversation history
        self.context = []
    
    def generate_response(
        self,
        user_input: str,
        max_length: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95
    ) -> str:
        """
        Generate response to user input
        
        Args:
            user_input: User's message
            max_length: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling (0 = disabled)
            top_p: Nucleus sampling (1.0 = disabled)
        
        Returns:
            Generated response text
        """
        # Add user input to context
        self.context.append(f"User: {user_input}")
        
        # Build context text
        context_text = "\\n".join(self.context) + "\\nAssistant:"
        
        # Encode context
        encoding = self.tokenizer.encode(context_text)
        input_ids = torch.tensor([encoding.ids], dtype=torch.long, device=self.device)
        
        # Trim context if too long
        if input_ids.size(1) > self.max_context_length:
            input_ids = input_ids[:, -self.max_context_length:]
        
        # Generate response
        with torch.no_grad():
            generated_ids = self._generate(
                input_ids,
                max_length,
                temperature,
                top_k,
                top_p
            )
        
        # Decode response (only new tokens)
        response_ids = generated_ids[0, input_ids.size(1):].tolist()
        response = self.tokenizer.decode(response_ids)
        
        # Clean up response (stop at next turn markers)
        response = response.split("User:")[0].strip()
        response = response.split("Assistant:")[0].strip()
        
        # Remove any trailing special tokens
        for stop_word in ["<|endoftext|>", "</s>", "<eos>"]:
            response = response.replace(stop_word, "")
        response = response.strip()
        
        # Add to context
        self.context.append(f"Assistant: {response}")
        
        # Trim context to keep only recent turns
        if len(self.context) > self.context_window * 2:  # 2 messages per turn
            self.context = self.context[-self.context_window * 2:]
        
        return response
    
    def _generate(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float,
        top_k: int,
        top_p: float
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively with sampling
        
        Args:
            input_ids: Input token IDs
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
        
        Returns:
            Generated token IDs
        """
        for _ in range(max_length):
            # Crop to model's max length if needed
            input_ids_cond = input_ids
            if input_ids.size(1) > self.model.config.max_length:
                input_ids_cond = input_ids[:, -self.model.config.max_length:]
            
            # Forward pass
            outputs = self.model(input_ids_cond)
            logits = outputs['logits']
            
            # Get next token logits
            next_token_logits = logits[0, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                top_k_values, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                indices_to_remove = next_token_logits < top_k_values[-1]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift right to keep first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample from distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            # Check for EOS tokens (common EOS token IDs)
            if next_token.item() in [0, 1, 2, 3]:  # Common EOS/PAD tokens
                break
            
            # Check for newline (stop at double newline)
            if next_token.item() == 10:  # Newline
                break
        
        return input_ids
    
    def reset_context(self):
        """Clear conversation history"""
        self.context = []
    
    def get_context_length(self) -> int:
        """Get current context length in tokens"""
        context_text = "\\n".join(self.context)
        encoding = self.tokenizer.encode(context_text)
        return len(encoding.ids)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Interactive chat with your trained LLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands during chat:
  exit, quit    - Exit the chat session
  clear, reset  - Clear conversation history
  help          - Show help message
  
Generation parameters:
  --temperature  Controls randomness (0.1-2.0, default: 0.8)
                 Lower = more focused, Higher = more creative
  --top-k        Top-k sampling (default: 50)
  --top-p        Nucleus sampling (default: 0.95)
  --max-length   Maximum response length in tokens (default: 100)

Examples:
  # Start chat with default settings
  python chat.py --checkpoint checkpoints/final.pt
  
  # More creative responses
  python chat.py --checkpoint checkpoints/final.pt --temperature 1.2
  
  # More focused responses
  python chat.py --checkpoint checkpoints/final.pt --temperature 0.5 --top-k 20
  
  # Longer responses
  python chat.py --checkpoint checkpoints/final.pt --max-length 200
        """
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (e.g., checkpoints/final.pt)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='llm.config.js',
        help='Path to config file (default: llm.config.js)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='Sampling temperature (default: 0.8)'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=100,
        help='Maximum response length in tokens (default: 100)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=50,
        help='Top-k sampling (default: 50, 0 to disable)'
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.95,
        help='Nucleus sampling (default: 0.95, 1.0 to disable)'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Device to use (default: auto)'
    )
    parser.add_argument(
        '--context-window',
        type=int,
        default=10,
        help='Number of conversation turns to keep (default: 10)'
    )
    
    return parser.parse_args()


def main():
    """Main chat function"""
    args = parse_args()
    
    # Print header
    print("=" * 70)
    print("💬  Interactive Chat with LLM")
    print("=" * 70)
    
    try:
        # Setup device
        if args.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = args.device
        
        print(f"\\n📱 Device: {device.upper()}")
        if device == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        
        # Load checkpoint
        print(f"\\n📦 Loading checkpoint: {args.checkpoint}")
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"❌ Checkpoint not found: {args.checkpoint}")
            print("\\nAvailable checkpoints:")
            checkpoint_dir = Path('checkpoints')
            if checkpoint_dir.exists():
                checkpoints = list(checkpoint_dir.glob('*.pt'))
                if checkpoints:
                    for cp in sorted(checkpoints):
                        print(f"   - {cp}")
                    print(f"\\n💡 Try using: python chat.py --checkpoint {checkpoints[0]}")
                else:
                    print("   No checkpoints found")
                    print("\\n💡 Train a model first: python training/train.py")
            sys.exit(1)
        
        checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
        
        # Load model
        print(f"\\n🤖 Loading model...")
        try:
            config = ConfigLoader(args.config)
            model = load_model_from_config(args.config)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Get training info
            step = checkpoint.get('step', 'unknown')
            epoch = checkpoint.get('epoch', 'unknown')
            print(f"   ✓ Model loaded (step: {step}, epoch: {epoch})")
            print(f"   ✓ Parameters: {model.count_parameters():,}")
        except Exception as e:
            print(f"   ❌ Failed to load model: {e}")
            sys.exit(1)
        
        # Load tokenizer
        print(f"\\n📝 Loading tokenizer...")
        tokenizer_path = Path('tokenizer/tokenizer.json')
        if not tokenizer_path.exists():
            print("   ❌ Tokenizer not found!")
            print("   Please train tokenizer first:")
            print("   python tokenizer/train.py --data data/raw/sample.txt")
            sys.exit(1)
        
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        print(f"   ✓ Tokenizer loaded (vocab size: {tokenizer.get_vocab_size()})")
        
        # Create chat session
        max_context = config.get('model.max_length', 512)
        chat = ChatSession(
            model,
            tokenizer,
            device,
            max_context_length=max_context,
            context_window=args.context_window
        )
        
        # Display settings
        print("\\n⚙️  Generation settings:")
        print(f"   Temperature: {args.temperature}")
        print(f"   Top-k: {args.top_k}")
        print(f"   Top-p: {args.top_p}")
        print(f"   Max length: {args.max_length} tokens")
        print(f"   Context window: {args.context_window} turns")
        
        # Display instructions
        print("\\n" + "=" * 70)
        print("Chat started! Type your message and press Enter.")
        print("Commands: 'exit' or 'quit' to exit, 'clear' to reset, 'help' for help")
        print("=" * 70 + "\\n")
        
        # Chat loop
        turn_count = 0
        while True:
            try:
                # Get user input
                user_input = input("\\n\\033[1;36mYou:\\033[0m ").strip()
                
                # Check for commands
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("\\n👋 Goodbye!\\n")
                    break
                
                if user_input.lower() in ['clear', 'reset']:
                    chat.reset_context()
                    turn_count = 0
                    print("\\n✓ Context cleared\\n")
                    continue
                
                if user_input.lower() == 'help':
                    print("\\n📖 Commands:")
                    print("   exit, quit, q  - Exit chat")
                    print("   clear, reset   - Clear conversation history")
                    print("   help           - Show this message")
                    print("\\n💡 Tips:")
                    print("   - The model maintains context across turns")
                    print("   - Use 'clear' if responses become incoherent")
                    print(f"   - Current context: {chat.get_context_length()} tokens")
                    continue
                
                if not user_input:
                    continue
                
                # Show loading indicator
                print("\\n\\033[1;32mAssistant:\\033[0m ", end="", flush=True)
                print("⏳ Thinking...", end="", flush=True)
                
                # Generate response
                response = chat.generate_response(
                    user_input,
                    max_length=args.max_length,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p
                )
                
                # Clear loading indicator and print response
                print("\\r\\033[1;32mAssistant:\\033[0m " + response)
                
                turn_count += 1
                
                # Show context info every 5 turns
                if turn_count % 5 == 0:
                    context_len = chat.get_context_length()
                    print(f"\\n   ℹ️  Context: {context_len}/{max_context} tokens, {len(chat.context)//2} turns")
                
            except KeyboardInterrupt:
                print("\\n\\n👋 Goodbye!\\n")
                break
            
            except Exception as e:
                print(f"\\n\\n❌ Error generating response: {e}")
                print("   Try 'clear' to reset context or 'exit' to quit\\n")
                continue
    
    except Exception as e:
        print(f"\\n❌ Failed to start chat: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
`;
    }

    private getDeployScriptTemplate(): string {
        return `#!/usr/bin/env python3
"""
Model Deployment Script
Deploy trained models to various platforms (Hugging Face Hub, Replicate, etc.)
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


class DeploymentManager:
    """
    Manages model deployment to various platforms
    """
    
    def __init__(self, checkpoint_path: str, tokenizer_path: Optional[str] = None):
        """
        Initialize deployment manager
        
        Args:
            checkpoint_path: Path to model checkpoint
            tokenizer_path: Optional path to tokenizer
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else None
        
        # Validate paths
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        if self.tokenizer_path and not self.tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    
    def deploy_to_huggingface(
        self,
        repo_id: str,
        private: bool = False,
        token: Optional[str] = None
    ) -> str:
        """
        Deploy model to Hugging Face Hub
        
        Args:
            repo_id: Repository ID (username/model-name)
            private: Whether to make repository private
            token: Optional HuggingFace token
        
        Returns:
            Model URL on success
        
        Raises:
            Exception: If deployment fails
        """
        try:
            from huggingface_hub import HfApi, create_repo, login
        except ImportError:
            raise ImportError(
                "huggingface-hub not installed.\\n"
                "Install with: pip install huggingface-hub"
            )
        
        print("\\n" + "=" * 60)
        print("🚀 Deploying to Hugging Face Hub")
        print("=" * 60)
        
        # Login if token provided
        if token:
            print("\\n📝 Logging in with provided token...")
            login(token=token)
        
        # Initialize API
        api = HfApi()
        
        # Verify authentication
        try:
            user_info = api.whoami()
            username = user_info['name']
            print(f"\\n✓ Authenticated as: {username}")
        except Exception as e:
            raise Exception(
                f"Authentication failed: {e}\\n"
                f"Please login with: huggingface-cli login\\n"
                f"Or provide token with --token argument"
            )
        
        # Create repository
        print(f"\\n📦 Creating repository: {repo_id}")
        try:
            create_repo(
                repo_id=repo_id,
                private=private,
                exist_ok=True,
                repo_type="model"
            )
            print(f"✓ Repository ready")
        except Exception as e:
            raise Exception(f"Failed to create repository: {e}")
        
        # Upload model checkpoint
        print(f"\\n⬆️  Uploading model checkpoint...")
        try:
            api.upload_file(
                path_or_fileobj=str(self.checkpoint_path),
                path_in_repo=f"pytorch_model.bin",
                repo_id=repo_id,
                repo_type="model"
            )
            print(f"✓ Model checkpoint uploaded")
        except Exception as e:
            raise Exception(f"Failed to upload model: {e}")
        
        # Upload tokenizer if provided
        if self.tokenizer_path:
            print(f"\\n⬆️  Uploading tokenizer...")
            try:
                api.upload_file(
                    path_or_fileobj=str(self.tokenizer_path),
                    path_in_repo="tokenizer.json",
                    repo_id=repo_id,
                    repo_type="model"
                )
                print(f"✓ Tokenizer uploaded")
            except Exception as e:
                print(f"⚠️  Warning: Failed to upload tokenizer: {e}")
        
        # Create model card
        print(f"\\n📄 Creating model card...")
        try:
            self._create_huggingface_model_card(api, repo_id)
            print(f"✓ Model card created")
        except Exception as e:
            print(f"⚠️  Warning: Failed to create model card: {e}")
        
        # Success
        model_url = f"https://huggingface.co/{repo_id}"
        print("\\n" + "=" * 60)
        print("✅ Deployment successful!")
        print("=" * 60)
        print(f"\\n🔗 Model URL: {model_url}")
        print(f"\\n💡 Next steps:")
        print(f"   • View your model: {model_url}")
        print(f"   • Test inference: Use the Inference API on the model page")
        print(f"   • Share with others: Send them the model URL")
        
        return model_url
    
    def deploy_to_replicate(
        self,
        model_name: str,
        token: Optional[str] = None
    ) -> str:
        """
        Deploy model to Replicate
        
        Args:
            model_name: Model name on Replicate
            token: Optional Replicate API token
        
        Returns:
            Model URL on success
        
        Raises:
            Exception: If deployment fails
        """
        print("\\n" + "=" * 60)
        print("🚀 Deploying to Replicate")
        print("=" * 60)
        
        # Check for Replicate token
        replicate_token = token or os.environ.get('REPLICATE_API_TOKEN')
        
        if not replicate_token:
            raise Exception(
                "Replicate API token not found.\\n"
                "Set REPLICATE_API_TOKEN environment variable or use --token argument.\\n"
                "Get your token at: https://replicate.com/account/api-tokens"
            )
        
        print("\\n📝 Note: Replicate deployment requires:")
        print("   1. A Cog configuration file (cog.yaml)")
        print("   2. A predict.py file with model inference code")
        print("   3. Docker installed on your system")
        print("   4. Cog CLI tool installed")
        
        print("\\n💡 To deploy to Replicate:")
        print("   1. Install Cog: https://github.com/replicate/cog")
        print("   2. Create cog.yaml and predict.py in your project")
        print("   3. Run: cog push r8.im/username/model-name")
        
        print("\\n⚠️  Automatic Replicate deployment is not yet implemented.")
        print("   Please follow the manual steps above.")
        
        # For now, return a placeholder URL
        model_url = f"https://replicate.com/{model_name}"
        return model_url
    
    def _create_huggingface_model_card(self, api, repo_id: str) -> None:
        """
        Create a model card for Hugging Face Hub
        
        Args:
            api: HuggingFace API instance
            repo_id: Repository ID
        """
        model_card = f"""---
language: en
license: apache-2.0
tags:
- llm
- gpt
- create-llm
- pytorch
---

# {repo_id.split('/')[-1]}

This model was trained using [create-llm](https://github.com/theaniketgiri/create-llm).

## Model Description

A language model trained with create-llm framework.

## Usage

\`\`\`python
import torch
from transformers import AutoTokenizer

# Load model
model = torch.load('pytorch_model.bin')
model.eval()

# Load tokenizer (if available)
try:
    tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
except:
    print("Tokenizer not available")

# Generate text
# Add your generation code here
\`\`\`

## Training Details

- **Framework:** PyTorch
- **Tool:** create-llm
- **Deployment:** Hugging Face Hub

## Citation

\`\`\`bibtex
@misc{{{repo_id.replace('/', '-')},
  author = {{Your Name}},
  title = {{{repo_id.split('/')[-1]}}},
  year = {{2024}},
  publisher = {{Hugging Face}},
  howpublished = {{\\\\url{{https://huggingface.co/{repo_id}}}}}
}}
\`\`\`
"""
        
        # Upload model card
        api.upload_file(
            path_or_fileobj=model_card.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model"
        )


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Deploy trained model to various platforms',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Deployment Platforms:
  huggingface    Deploy to Hugging Face Hub (model hosting and sharing)
  replicate      Deploy to Replicate (API-based inference)

Examples:
  # Deploy to Hugging Face Hub
  python deploy.py --checkpoint checkpoints/final.pt --to huggingface --repo-id username/model-name
  
  # Deploy with tokenizer
  python deploy.py --checkpoint checkpoints/final.pt --tokenizer tokenizer/tokenizer.json \\\\
                   --to huggingface --repo-id username/model-name
  
  # Deploy private model
  python deploy.py --checkpoint checkpoints/final.pt --to huggingface \\\\
                   --repo-id username/model-name --private
  
  # Deploy to Replicate
  python deploy.py --checkpoint checkpoints/final.pt --to replicate --model-name username/model-name

Authentication:
  Hugging Face: Run 'huggingface-cli login' or use --token
  Replicate: Set REPLICATE_API_TOKEN environment variable or use --token
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint file'
    )
    parser.add_argument(
        '--to',
        type=str,
        required=True,
        choices=['huggingface', 'replicate'],
        help='Deployment platform'
    )
    
    # Optional arguments
    parser.add_argument(
        '--tokenizer',
        type=str,
        help='Path to tokenizer file (optional)'
    )
    parser.add_argument(
        '--token',
        type=str,
        help='API token for authentication (optional)'
    )
    
    # Hugging Face specific
    parser.add_argument(
        '--repo-id',
        type=str,
        help='Repository ID for Hugging Face (username/model-name)'
    )
    parser.add_argument(
        '--private',
        action='store_true',
        help='Make Hugging Face repository private'
    )
    
    # Replicate specific
    parser.add_argument(
        '--model-name',
        type=str,
        help='Model name for Replicate (username/model-name)'
    )
    
    return parser.parse_args()


def main():
    """Main deployment function"""
    args = parse_args()
    
    print("=" * 60)
    print("🚀 Model Deployment Tool")
    print("=" * 60)
    
    try:
        # Create deployment manager
        manager = DeploymentManager(
            checkpoint_path=args.checkpoint,
            tokenizer_path=args.tokenizer
        )
        
        # Deploy based on platform
        if args.to == 'huggingface':
            # Validate Hugging Face arguments
            if not args.repo_id:
                print("\\n❌ Error: --repo-id is required for Hugging Face deployment")
                print("   Example: --repo-id username/model-name")
                sys.exit(1)
            
            # Deploy to Hugging Face
            url = manager.deploy_to_huggingface(
                repo_id=args.repo_id,
                private=args.private,
                token=args.token
            )
            
        elif args.to == 'replicate':
            # Validate Replicate arguments
            if not args.model_name:
                print("\\n❌ Error: --model-name is required for Replicate deployment")
                print("   Example: --model-name username/model-name")
                sys.exit(1)
            
            # Deploy to Replicate
            url = manager.deploy_to_replicate(
                model_name=args.model_name,
                token=args.token
            )
        
        print("\\n")
        
    except FileNotFoundError as e:
        print(f"\\n❌ Error: {e}")
        print("\\n💡 Troubleshooting:")
        print("   • Check that the checkpoint file exists")
        print("   • Verify the file path is correct")
        print("   • Make sure you're running from the project root")
        sys.exit(1)
        
    except ImportError as e:
        print(f"\\n❌ Error: {e}")
        print("\\n💡 Troubleshooting:")
        print("   • Install required packages: pip install -r requirements.txt")
        sys.exit(1)
        
    except Exception as e:
        print(f"\\n❌ Deployment failed: {e}")
        print("\\n💡 Troubleshooting:")
        print("   • Check your internet connection")
        print("   • Verify your authentication credentials")
        print("   • Make sure you have permission to create repositories")
        print("   • Check the platform's status page for outages")
        sys.exit(1)


if __name__ == '__main__':
    main()
`;
    }

    private getCompareScriptTemplate(): string {
        return `#!/usr/bin/env python3
"""
Model Comparison Tool
Compare multiple model checkpoints side-by-side on the same validation set
"""

import argparse
import sys
import torch
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from tabulate import tabulate

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models import load_model_from_config
from data import LLMDataset, create_dataloader
from tokenizers import Tokenizer


class ModelComparator:
    """
    Compares multiple model checkpoints
    """
    
    def __init__(
        self,
        checkpoint_paths: List[str],
        val_data_path: str,
        tokenizer_path: str,
        device: str = 'auto'
    ):
        """
        Initialize model comparator
        
        Args:
            checkpoint_paths: List of checkpoint paths to compare
            val_data_path: Path to validation data
            tokenizer_path: Path to tokenizer
            device: Device to use (auto/cuda/cpu)
        """
        self.checkpoint_paths = [Path(p) for p in checkpoint_paths]
        self.val_data_path = Path(val_data_path)
        self.tokenizer_path = Path(tokenizer_path)
        
        # Setup device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Validate paths
        for cp in self.checkpoint_paths:
            if not cp.exists():
                raise FileNotFoundError(f"Checkpoint not found: {cp}")
        
        if not self.val_data_path.exists():
            raise FileNotFoundError(f"Validation data not found: {val_data_path}")
        
        if not self.tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
        
        # Load tokenizer
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        
        print(f"\\n{'='*70}")
        print(f"🔍 Model Comparison Tool")
        print(f"{'='*70}")
        print(f"\\n📊 Comparing {len(self.checkpoint_paths)} models")
        print(f"📁 Validation data: {val_data_path}")
        print(f"💻 Device: {self.device.upper()}")
    
    def load_model(self, checkpoint_path: Path) -> tuple:
        """
        Load model from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint
        
        Returns:
            Tuple of (model, checkpoint_info)
        """
        print(f"\\n📦 Loading {checkpoint_path.name}...")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model
        model = load_model_from_config()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        # Get checkpoint info
        info = {
            'step': checkpoint.get('step', 'unknown'),
            'epoch': checkpoint.get('epoch', 'unknown'),
            'loss': checkpoint.get('loss', 'unknown'),
        }
        
        print(f"✓ Loaded (step: {info['step']}, epoch: {info['epoch']})")
        
        return model, info
    
    def evaluate_model(self, model, model_name: str) -> Dict[str, Any]:
        """
        Evaluate model on validation set
        
        Args:
            model: Model to evaluate
            model_name: Name for display
        
        Returns:
            Dictionary of metrics
        """
        print(f"\\n📈 Evaluating {model_name}...")
        
        # Create validation dataloader
        val_dataset = LLMDataset(str(self.val_data_path), max_length=512)
        val_loader = create_dataloader(val_dataset, batch_size=8, shuffle=False)
        
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = model(input_ids, labels=labels)
                loss = outputs['loss']
                
                total_loss += loss.item() * input_ids.size(0)
                total_tokens += input_ids.numel()
        
        avg_loss = total_loss / len(val_dataset)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        metrics = {
            'loss': avg_loss,
            'perplexity': perplexity,
            'tokens_evaluated': total_tokens
        }
        
        print(f"✓ Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
        
        return metrics
    
    def generate_samples(
        self,
        models: List[tuple],
        prompts: List[str],
        max_length: int = 50,
        temperature: float = 0.8
    ) -> Dict[str, List[str]]:
        """
        Generate sample outputs from all models
        
        Args:
            models: List of (model, name) tuples
            prompts: List of prompts to generate from
            max_length: Maximum generation length
            temperature: Sampling temperature
        
        Returns:
            Dictionary mapping model names to generated texts
        """
        print(f"\\n✨ Generating samples from {len(models)} models...")
        
        results = {}
        
        for model, name in models:
            model_results = []
            
            for prompt in prompts:
                # Encode prompt
                encoding = self.tokenizer.encode(prompt)
                input_ids = torch.tensor([encoding.ids], dtype=torch.long, device=self.device)
                
                # Generate
                with torch.no_grad():
                    generated = model.generate(
                        input_ids,
                        max_new_tokens=max_length,
                        temperature=temperature
                    )
                
                # Decode
                generated_ids = generated[0, input_ids.size(1):].tolist()
                generated_text = self.tokenizer.decode(generated_ids)
                
                model_results.append(generated_text)
            
            results[name] = model_results
            print(f"✓ Generated {len(prompts)} samples for {name}")
        
        return results
    
    def compare(
        self,
        sample_prompts: List[str] = None,
        save_report: bool = True
    ) -> Dict[str, Any]:
        """
        Compare all models
        
        Args:
            sample_prompts: Optional list of prompts for generation
            save_report: Whether to save comparison report
        
        Returns:
            Comparison results
        """
        # Default prompts
        if sample_prompts is None:
            sample_prompts = [
                "Once upon a time",
                "The future of AI",
                "In a world where"
            ]
        
        # Load all models
        models = []
        model_infos = []
        
        for cp in self.checkpoint_paths:
            model, info = self.load_model(cp)
            models.append((model, cp.stem))
            model_infos.append(info)
        
        # Evaluate all models
        print(f"\\n{'='*70}")
        print(f"📊 Evaluation Results")
        print(f"{'='*70}")
        
        all_metrics = {}
        for (model, name), info in zip(models, model_infos):
            metrics = self.evaluate_model(model, name)
            metrics.update(info)
            all_metrics[name] = metrics
        
        # Display metrics table
        self._display_metrics_table(all_metrics)
        
        # Generate samples
        print(f"\\n{'='*70}")
        print(f"✨ Sample Generations")
        print(f"{'='*70}")
        
        sample_results = self.generate_samples(models, sample_prompts)
        self._display_samples(sample_prompts, sample_results)
        
        # Prepare comparison results
        results = {
            'timestamp': datetime.now().isoformat(),
            'models': list(all_metrics.keys()),
            'metrics': all_metrics,
            'samples': {
                'prompts': sample_prompts,
                'generations': sample_results
            }
        }
        
        # Save report
        if save_report:
            report_path = self._save_report(results)
            print(f"\\n💾 Comparison report saved to: {report_path}")
        
        # Display winner
        self._display_winner(all_metrics)
        
        return results
    
    def _display_metrics_table(self, metrics: Dict[str, Dict[str, Any]]) -> None:
        """Display metrics in a formatted table"""
        headers = ['Model', 'Step', 'Epoch', 'Loss', 'Perplexity', 'Tokens']
        rows = []
        
        for name, m in metrics.items():
            rows.append([
                name,
                m.get('step', 'N/A'),
                m.get('epoch', 'N/A'),
                f"{m['loss']:.4f}",
                f"{m['perplexity']:.2f}",
                f"{m['tokens_evaluated']:,}"
            ])
        
        print(f"\\n{tabulate(rows, headers=headers, tablefmt='grid')}")
    
    def _display_samples(
        self,
        prompts: List[str],
        results: Dict[str, List[str]]
    ) -> None:
        """Display sample generations"""
        for i, prompt in enumerate(prompts):
            print(f"\\n📝 Prompt {i+1}: \\"{prompt}\\"")
            print(f"{'─'*70}")
            
            for model_name, generations in results.items():
                print(f"\\n🤖 {model_name}:")
                print(f"   {generations[i][:200]}...")
    
    def _display_winner(self, metrics: Dict[str, Dict[str, Any]]) -> None:
        """Display the best performing model"""
        # Find model with lowest loss
        best_model = min(metrics.items(), key=lambda x: x[1]['loss'])
        
        print(f"\\n{'='*70}")
        print(f"🏆 Best Model: {best_model[0]}")
        print(f"{'='*70}")
        print(f"   Loss: {best_model[1]['loss']:.4f}")
        print(f"   Perplexity: {best_model[1]['perplexity']:.2f}")
        print(f"   Step: {best_model[1].get('step', 'N/A')}")
    
    def _save_report(self, results: Dict[str, Any]) -> Path:
        """Save comparison report to logs directory"""
        # Create logs directory
        logs_dir = Path('logs')
        logs_dir.mkdir(exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = logs_dir / f'comparison_{timestamp}.json'
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return report_path


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Compare multiple model checkpoints side-by-side',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two models
  python compare.py checkpoints/model_v1.pt checkpoints/model_v2.pt
  
  # Compare three models with custom prompts
  python compare.py model1.pt model2.pt model3.pt \\\\
                   --prompts "Hello world" "The quick brown fox"
  
  # Compare on specific validation data
  python compare.py model1.pt model2.pt --val-data data/processed/val.pt
  
  # Use CPU for comparison
  python compare.py model1.pt model2.pt --device cpu

Output:
  • Evaluation metrics table (loss, perplexity)
  • Sample generations from each model
  • Best model recommendation
  • Comparison report saved to logs/
        """
    )
    
    # Required arguments
    parser.add_argument(
        'checkpoints',
        nargs='+',
        help='Paths to model checkpoints to compare (2 or more)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--val-data',
        type=str,
        default='data/processed/val.pt',
        help='Path to validation data (default: data/processed/val.pt)'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='tokenizer/tokenizer.json',
        help='Path to tokenizer (default: tokenizer/tokenizer.json)'
    )
    parser.add_argument(
        '--prompts',
        nargs='+',
        help='Custom prompts for sample generation'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=50,
        help='Maximum generation length (default: 50)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='Sampling temperature (default: 0.8)'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cuda', 'cpu'],
        default='auto',
        help='Device to use (default: auto)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save comparison report'
    )
    
    return parser.parse_args()


def main():
    """Main comparison function"""
    args = parse_args()
    
    # Validate number of checkpoints
    if len(args.checkpoints) < 2:
        print("❌ Error: At least 2 checkpoints required for comparison")
        print("   Example: python compare.py model1.pt model2.pt")
        sys.exit(1)
    
    try:
        # Create comparator
        comparator = ModelComparator(
            checkpoint_paths=args.checkpoints,
            val_data_path=args.val_data,
            tokenizer_path=args.tokenizer,
            device=args.device
        )
        
        # Run comparison
        results = comparator.compare(
            sample_prompts=args.prompts,
            save_report=not args.no_save
        )
        
        print(f"\\n{'='*70}")
        print(f"✅ Comparison complete!")
        print(f"{'='*70}\\n")
        
    except FileNotFoundError as e:
        print(f"\\n❌ Error: {e}")
        print(f"\\n💡 Troubleshooting:")
        print(f"   • Check that all checkpoint files exist")
        print(f"   • Verify validation data is prepared: python data/prepare.py")
        print(f"   • Ensure tokenizer is trained: python tokenizer/train.py")
        sys.exit(1)
        
    except Exception as e:
        print(f"\\n❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
`;
    }

    private getRequirementsTxt(config: ProjectConfig): string {
        const lines: string[] = [];

        lines.push('# LLM Training Project Dependencies');
        lines.push(`# Generated by create-llm for ${config.template} template`);
        lines.push('');
        lines.push('# Core deep learning framework');
        lines.push('torch>=2.0.0');
        lines.push('');
        lines.push('# Transformers and tokenizers');
        lines.push('transformers>=4.30.0');
        lines.push('tokenizers>=0.13.0');
        lines.push('');
        lines.push('# Training utilities');
        lines.push('tqdm>=4.65.0');
        lines.push('numpy>=1.24.0');
        lines.push('tabulate>=0.9.0');
        lines.push('');
        lines.push('# Data processing');
        lines.push('datasets>=2.14.0');
        lines.push('');
        lines.push('# Visualization and logging');
        lines.push('tensorboard>=2.13.0');
        lines.push('matplotlib>=3.7.0');
        lines.push('');
        lines.push('# Live training dashboard');
        lines.push('flask>=2.3.0');
        lines.push('flask-socketio>=5.3.0');
        lines.push('');

        // Add plugin-specific dependencies
        if (config.plugins.includes('wandb')) {
            lines.push('# Experiment tracking (WandB plugin enabled)');
            lines.push('wandb>=0.15.0');
            lines.push('');
        } else {
            lines.push('# Optional: Experiment tracking');
            lines.push('# wandb>=0.15.0');
            lines.push('');
        }

        if (config.plugins.includes('huggingface')) {
            lines.push('# Model hub integration (HuggingFace plugin enabled)');
            lines.push('huggingface-hub>=0.16.0');
            lines.push('');
        } else {
            lines.push('# Optional: Model hub integration');
            lines.push('# huggingface-hub>=0.16.0');
            lines.push('');
        }

        if (config.plugins.includes('synthex')) {
            lines.push('# Synthetic data generation (Synthex plugin enabled)');
            lines.push('# synthex-ai>=1.0.0  # Install from synthex.ai');
            lines.push('');
        }


        lines.push('# Optional: Advanced optimizations');
        lines.push('# flash-attn>=2.0.0  # Faster attention (requires CUDA)');
        lines.push('# deepspeed>=0.10.0  # Distributed training');
        lines.push('');
        lines.push('# Development and testing tools');
        lines.push('pytest>=7.4.0');
        lines.push('pytest-cov>=4.1.0');
        lines.push('# black>=23.7.0');
        lines.push('# flake8>=6.1.0');

        return lines.join('\n') + '\n';
    }

    private getGitignore(): string {
        return `# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
ENV/
env/
.venv

# Training artifacts
checkpoints/
logs/
data/processed/
*.pt
*.pth
*.ckpt
*.safetensors

# Tokenizer
tokenizer/tokenizer.json
tokenizer/*.model
tokenizer/*.vocab

# TensorBoard
runs/
events.out.tfevents.*

# Weights & Biases
wandb/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.project
.pydevproject

# OS
.DS_Store
Thumbs.db
*.bak

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Environment and secrets
.env
.env.local
.env.*.local
*.key
*.pem
secrets.json

# Large files (use Git LFS if needed)
*.bin
*.h5

# Temporary files
*.tmp
*.temp
.cache/

# Coverage reports
htmlcov/
.coverage
.coverage.*
coverage.xml
*.cover

# pytest
.pytest_cache/

# mypy
.mypy_cache/
.dmypy.json
dmypy.json
`;
    }

    private getSampleData(template: Template): string {
        const minDataSize = template.config.hardware.can_run_on_cpu ? '1-10MB' :
            template.config.model.parameters < 500_000_000 ? '100MB-1GB' : '1GB+';

        return `# Sample Training Data

This is sample training data for your LLM project. Replace this with your own data to train a custom language model.

## Getting Started

To use your own data, simply replace this file with your training text. You can add multiple text files to the data/raw/ directory, and the preparation script will process all of them.

## Data Requirements for ${template.name.toUpperCase()} Template

The ${template.name.toUpperCase()} template is designed for ${template.config.hardware.can_run_on_cpu ? 'quick experimentation and learning' : 'production-grade training'}:
- Recommended data size: ${minDataSize} of text
- Model size: ${(template.config.model.parameters / 1_000_000).toFixed(0)}M parameters  
- Hardware: ${template.config.hardware.recommended_gpu}
- Training time: ${template.config.hardware.estimated_training_time}

## Data Quality Guidelines

For best training results, follow these guidelines:
- Use clean, well-formatted text without excessive special characters
- Remove unwanted formatting, HTML tags, or markup
- Ensure data represents your target domain or use case
- Include diverse examples for general-purpose models
- Maintain consistent formatting and style throughout

## Example Training Domains

You can train models on various types of text data:

### Code Generation
Train on programming code to create a code completion model. Include comments, function definitions, and complete programs in languages like Python, JavaScript, Java, or C++.

### Literature and Creative Writing  
Use books, articles, short stories, or poetry to train a creative writing assistant. Include diverse writing styles and genres for better generalization.

### Technical Documentation
Train on technical manuals, API documentation, tutorials, or how-to guides to create a technical writing assistant.

### Conversational AI
Use dialogue data, chat logs, or conversational transcripts to train a chatbot or conversational agent.

### Domain-Specific Applications
Train on specialized text like medical records, legal documents, scientific papers, or business reports for domain-specific applications.

## Data Format and Structure

Your training data should be plain text files with UTF-8 encoding. Here's a recommended directory structure:

\`\`\`
data/raw/
├── train.txt          # Main training data
├── val.txt            # Validation data (optional)
├── domain1.txt        # Additional domain-specific data
├── domain2.txt        # More domain data
└── sample.txt         # This file (replace with your data)
\`\`\`

If you don't provide a separate validation file, the data preparation script will automatically create a validation split from your training data.

## Example Training Data

Here's some example text to demonstrate the format. This is just placeholder content - replace it with your actual training data.

Once upon a time, in a land far away, there lived a curious programmer who wanted to train their own language model. They gathered text data from various sources and prepared it carefully.

The programmer learned that data quality matters more than quantity. Clean, well-formatted text produces better models than large amounts of noisy data.

They discovered that even small models can be surprisingly capable when trained on focused, domain-specific data. The key is matching the model size to the task complexity.

After preparing the data, the programmer ran the tokenizer training script to create a vocabulary. Then they started the training process and watched as the loss decreased with each epoch.

The model learned patterns in the text, understanding grammar, style, and domain-specific terminology. With patience and careful tuning, the programmer created a useful language model.

## Data Preparation Steps

Follow these steps to prepare your data for training:

1. Replace this file with your training data (or add new .txt files to data/raw/)
2. Train the tokenizer: \`python tokenizer/train.py --data data/raw/\`
3. Prepare the dataset: \`python data/prepare.py\`
4. Start training: \`python training/train.py\`

## Tips for Better Results

- Start with at least 1MB of text for meaningful results
- Use consistent formatting throughout your dataset
- Remove duplicate or near-duplicate content
- Balance different types of content if training for general use
- Test with small experiments before scaling up
- Monitor training loss to ensure the model is learning

## Advanced Data Techniques

For more advanced users, consider these techniques:

### Data Augmentation
Create variations of your text through paraphrasing, back-translation, or synthetic generation to increase dataset size.

### Data Filtering
Remove low-quality examples, duplicates, or irrelevant content to improve training efficiency.

### Domain Mixing
Combine data from multiple domains to create more versatile models, but maintain balance between domains.

### Curriculum Learning
Start training on simpler examples and gradually introduce more complex ones for better learning.

## Next Steps

Once you've replaced this sample data with your own:

1. Train the tokenizer on your data
2. Prepare the processed dataset  
3. Adjust training hyperparameters in llm.config.js if needed
4. Start training and monitor the loss
5. Evaluate your model and iterate

For more detailed information, see the README.md file in the project root.

Good luck with your language model training!
`;
    }

    private getReadmeTemplate(config: ProjectConfig, template: Template): string {
        const params = (template.config.model.parameters / 1_000_000).toFixed(1);
        const hardware = template.config.hardware.recommended_gpu;
        const trainingTime = template.config.hardware.estimated_training_time;
        const cpuFriendly = template.config.hardware.can_run_on_cpu;
        const minData = config.template === 'nano' ? '100+' : 
                       config.template === 'tiny' ? '1,000+' :
                       config.template === 'small' ? '10,000+' : '100,000+';

        return `# ${config.projectName}

> ${template.config.documentation.description}

Created with [create-llm](https://github.com/theaniketgiri/create-llm) ✨

---

## 📋 Project Overview

| Property | Value |
|----------|-------|
| **Template** | ${config.template.toUpperCase()} |
| **Model** | ${template.config.model.type.toUpperCase()} (~${params}M parameters) |
| **Tokenizer** | ${config.tokenizer.toUpperCase()} |
| **Hardware** | ${hardware} |
| **Training Time** | ${trainingTime} |
| **Min Data** | ${minData} examples |
| **CPU Compatible** | ${cpuFriendly ? '✅ Yes' : '❌ No (GPU required)'} |

---

## 🚀 Quick Start

### Step 1: Install Dependencies

\`\`\`bash
pip install -r requirements.txt
\`\`\`

### Step 2: Add Your Training Data

Place your text files in \`data/raw/\`:

\`\`\`bash
# Example: Download Shakespeare
curl https://www.gutenberg.org/files/100/100-0.txt > data/raw/shakespeare.txt

# Or copy your own files
cp /path/to/your/data.txt data/raw/
\`\`\`

**Data Requirements:**
- Format: Plain text (.txt files)
- Encoding: UTF-8
- Minimum: ${minData} examples
- Recommended: Clean, well-formatted text

### Step 3: Train Tokenizer

\`\`\`bash
python tokenizer/train.py --data data/raw/
\`\`\`

This creates a vocabulary from your data. You'll see:
- Vocabulary size
- Sample encoding
- Tokenizer statistics

### Step 4: Prepare Dataset

\`\`\`bash
python data/prepare.py
\`\`\`

This tokenizes and prepares your data. You'll see:
- Number of training examples
- Number of validation examples
- Total tokens processed

### Step 5: Start Training

\`\`\`bash
# Basic training
python training/train.py

# With live dashboard (recommended)
python training/train.py --dashboard
# Then open http://localhost:5000 in your browser
\`\`\`

**Training will show:**
- Real-time loss
- Learning rate schedule
- Tokens per second
- Estimated time remaining

### Step 6: Monitor Training

Watch for these indicators:

✅ **Good Training:**
- Loss steadily decreasing
- Perplexity: 5-20 (depends on data)
- No warnings

⚠️ **Potential Issues:**
- Perplexity < 1.5: Possible overfitting
- Loss not decreasing: Check learning rate
- "Model too large" warning: Add more data or use smaller template

### Step 7: Evaluate Your Model

\`\`\`bash
python evaluation/evaluate.py --checkpoint checkpoints/checkpoint-best.pt
\`\`\`

You'll see:
- Perplexity score
- Loss metrics
- Performance statistics

### Step 8: Generate Text

\`\`\`bash
python evaluation/generate.py \\
  --checkpoint checkpoints/checkpoint-best.pt \\
  --prompt "Once upon a time" \\
  --temperature 0.8
\`\`\`

**Temperature Guide:**
- 0.1-0.3: Focused, deterministic
- 0.7-0.9: Balanced, creative
- 1.0-1.5: Very creative, diverse

### Step 9: Interactive Chat

\`\`\`bash
python chat.py --checkpoint checkpoints/checkpoint-best.pt
\`\`\`

**Chat Commands:**
- \`/temp <value>\`: Adjust temperature
- \`/clear\`: Clear conversation
- \`/quit\`: Exit chat

---

## 📁 Project Structure

\`\`\`
${config.projectName}/
│
├── 📂 data/
│   ├── raw/                    # ← Put your .txt files here
│   ├── processed/              # Tokenized data (auto-generated)
│   ├── dataset.py              # PyTorch dataset classes
│   └── prepare.py              # Data preprocessing script
│
├── 📂 models/
│   ├── architectures/
│   │   ├── gpt.py             # GPT architecture implementation
│   │   ├── nano.py            # 1M parameter model
│   │   ├── tiny.py            # 6M parameter model
│   │   ├── small.py           # 100M parameter model
│   │   └── base.py            # 1B parameter model
│   ├── __init__.py
│   └── config.py              # Configuration loader
│
├── 📂 tokenizer/
│   ├── train.py               # Tokenizer training script
│   └── tokenizer.json         # Trained tokenizer (auto-generated)
│
├── 📂 training/
│   ├── train.py               # Main training script ⭐
│   ├── trainer.py             # Trainer class
│   ├── callbacks/             # Training callbacks
│   │   ├── base.py
│   │   ├── checkpoint.py      # Checkpoint management
│   │   ├── logging.py         # TensorBoard logging
│   │   └── checkpoint_manager.py
│   └── dashboard/             # Live training dashboard
│       ├── dashboard_server.py
│       └── templates/
│
├── 📂 evaluation/
│   ├── evaluate.py            # Model evaluation
│   └── generate.py            # Text generation
│
├── 📂 plugins/                # Optional integrations
│   ├── wandb_plugin.py        # Weights & Biases
│   ├── huggingface_plugin.py  # HuggingFace Hub
│   └── synthex_plugin.py      # Synthetic data
│
├── 📂 checkpoints/            # Saved models (auto-generated)
├── 📂 logs/                   # Training logs (auto-generated)
│
├── 📄 llm.config.js           # Main configuration ⚙️
├── 📄 requirements.txt        # Python dependencies
├── 📄 chat.py                 # Interactive chat interface
├── 📄 deploy.py               # Deployment script
├── 📄 compare.py              # Model comparison tool
└── 📄 README.md               # This file
\`\`\`

---

## ⚙️ Configuration

All settings are in \`llm.config.js\`:

\`\`\`javascript
module.exports = {
  model: {
    type: '${template.config.model.type}',
    size: '${config.template}',
    vocab_size: ${template.config.model.vocab_size},  // Auto-detected from tokenizer
    max_length: ${template.config.model.max_length},
    layers: ${template.config.model.layers},
    heads: ${template.config.model.heads},
    dim: ${template.config.model.dim},
    dropout: ${template.config.model.dropout},
  },
  training: {
    batch_size: ${template.config.training.batch_size},
    learning_rate: ${template.config.training.learning_rate},
    max_steps: ${template.config.training.max_steps},
    // ... more options
  },
};
\`\`\`

**Common Adjustments:**
- \`batch_size\`: Reduce if out of memory
- \`learning_rate\`: Adjust if loss unstable
- \`dropout\`: Increase if overfitting (0.2-0.4)
- \`max_steps\`: Increase for better quality

---

## 💡 Training Tips

${template.config.documentation.training_tips.map(tip => `- ${tip}`).join('\n')}

---

## 🔧 Advanced Usage

### Resume Training

If training was interrupted:

\`\`\`bash
python training/train.py --resume checkpoints/checkpoint-1000.pt
\`\`\`

### Live Dashboard

Monitor training in real-time:

\`\`\`bash
python training/train.py --dashboard
\`\`\`

Then open http://localhost:5000 to see:
- Real-time loss curves
- Learning rate schedule
- GPU memory usage
- Tokens per second
- Recent checkpoints

### Model Comparison

Compare multiple trained models:

\`\`\`bash
python compare.py checkpoints/model-v1/ checkpoints/model-v2/
\`\`\`

Shows:
- Side-by-side metrics
- Sample generations
- Performance comparison

### Custom Generation

\`\`\`bash
# Adjust creativity
python evaluation/generate.py \\
  --checkpoint checkpoints/checkpoint-best.pt \\
  --prompt "Your prompt" \\
  --temperature 0.8 \\
  --top-k 50 \\
  --top-p 0.95 \\
  --max-length 200
\`\`\`

### Deploy to Hugging Face

\`\`\`bash
python deploy.py \\
  --checkpoint checkpoints/checkpoint-best.pt \\
  --to huggingface \\
  --repo-id username/my-model
\`\`\`

---

## 🔌 Plugins

${config.plugins.length > 0 ? `### Enabled Plugins\n\n${config.plugins.map(p => `- **${p}**: Configured and ready to use`).join('\n')}` : '### No Plugins Enabled\n\nTo enable plugins, edit \`llm.config.js\`:\n\n\`\`\`javascript\nplugins: [\n  \'wandb\',        // Experiment tracking\n  \'huggingface\',  // Model sharing\n  \'synthex\',      // Synthetic data\n]\n\`\`\`'}

---

## 🐛 Troubleshooting

### "Vocab size mismatch detected"
✅ **This is normal!** The tool auto-detects and uses the correct vocab size from your tokenizer.

### "Position embedding index error" or sequences too long
✅ **Automatically handled!** Sequences exceeding max_length are truncated with warnings.
- Check data preprocessing if you see frequent truncation warnings
- Consider increasing \`max_length\` in config if needed (requires retraining)

### "Model may be too large for dataset"
⚠️ **Warning:** Risk of overfitting
- **Solution 1:** Add more training data (recommended)
- **Solution 2:** Use smaller template (nano/tiny)
- **Solution 3:** Increase dropout in llm.config.js

### "Perplexity < 1.5"
❌ **Overfitting detected**
- Model memorized the data
- Add much more data or use smaller model

### "CUDA out of memory"
- Reduce \`batch_size\` in llm.config.js
- Enable \`mixed_precision: true\`
- Increase \`gradient_accumulation_steps\`

### "Training loss not decreasing"
- Check learning rate (try 1e-4 to 1e-3)
- Verify data loaded correctly
- Try longer warmup period

### "Tokenizer not found"
- Run \`python tokenizer/train.py --data data/raw/\` first
- Make sure data/raw/ contains .txt files

---

## 📊 Hardware Requirements

| Component | Requirement |
|-----------|-------------|
| **RAM** | ${template.config.hardware.min_ram} minimum |
| **GPU** | ${hardware} |
| **Storage** | 10GB+ free space |
| **Training Time** | ${trainingTime} |

---

## 📚 Resources

- [create-llm Documentation](https://github.com/theaniketgiri/create-llm)
- [Training Best Practices](https://github.com/theaniketgiri/create-llm/docs/training.md)
- [API Reference](https://github.com/theaniketgiri/create-llm/docs/api.md)
- [Troubleshooting Guide](https://github.com/theaniketgiri/create-llm/docs/troubleshooting.md)

---

## 📝 License

MIT

---

## 🙏 Acknowledgments

This project was created with [create-llm](https://github.com/theaniketgiri/create-llm) - The fastest way to start training your own Language Model.

**Built with:**
- PyTorch
- Transformers
- Tokenizers
- TensorBoard

---

**Happy Training! 🚀**

If you encounter any issues, please check the troubleshooting section above or visit the [create-llm repository](https://github.com/theaniketgiri/create-llm/issues).
`;
    }
}
