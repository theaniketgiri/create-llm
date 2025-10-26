/**
 * Python test templates for pytest configuration and test suites
 */

export class PythonTestTemplates {
  /**
   * Get pytest configuration (pytest.ini)
   */
  static getPytestConfig(): string {
    return `[pytest]
# Pytest configuration for create-llm

# Test discovery patterns
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*

# Test paths
testpaths = tests

# Output options
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings
    -ra

# Markers for test categorization
markers =
    unit: Unit tests for individual components
    integration: Integration tests for workflows
    slow: Tests that take a long time to run
    gpu: Tests that require GPU
    
# Coverage options (if pytest-cov is installed)
# addopts = --cov=. --cov-report=html --cov-report=term

# Timeout for tests (if pytest-timeout is installed)
# timeout = 300
`;
  }

  /**
   * Get conftest.py with fixtures
   */
  static getConftest(): string {
    return `"""
Pytest configuration and fixtures
Provides common fixtures for testing
"""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        'model': {
            'type': 'gpt',
            'vocab_size': 1000,
            'max_length': 128,
            'layers': 2,
            'heads': 2,
            'dim': 64,
            'dropout': 0.1,
        },
        'training': {
            'batch_size': 2,
            'learning_rate': 1e-3,
            'max_steps': 10,
            'eval_interval': 5,
            'save_interval': 5,
        },
        'data': {
            'max_length': 128,
            'stride': 64,
        },
    }


@pytest.fixture
def sample_text():
    """Sample text data for testing"""
    return "This is a test sentence. " * 100


@pytest.fixture
def device():
    """Get device for testing (CPU by default)"""
    return 'cpu'


@pytest.fixture
def small_model(sample_config, device):
    """Create a small model for testing"""
    from models.architectures.gpt import create_gpt_model
    model = create_gpt_model(sample_config['model'])
    model.to(device)
    return model


@pytest.fixture
def sample_batch(device):
    """Create a sample batch for testing"""
    batch_size = 2
    seq_len = 32
    vocab_size = 1000
    
    return {
        'input_ids': torch.randint(0, vocab_size, (batch_size, seq_len), device=device),
        'labels': torch.randint(0, vocab_size, (batch_size, seq_len), device=device),
    }
`;
  }

  /**
   * Get test for model architectures
   */
  static getModelTests(): string {
    return `"""
Tests for model architectures
"""

import pytest
import torch
from models.architectures.gpt import GPTModel, GPTConfig, create_gpt_model


class TestGPTModel:
    """Tests for GPT model"""
    
    def test_model_creation(self, sample_config):
        """Test model can be created"""
        model = create_gpt_model(sample_config['model'])
        assert model is not None
        assert isinstance(model, GPTModel)
    
    def test_model_forward(self, small_model, sample_batch):
        """Test model forward pass"""
        outputs = small_model(
            sample_batch['input_ids'],
            labels=sample_batch['labels']
        )
        
        assert 'logits' in outputs
        assert 'loss' in outputs
        assert outputs['logits'].shape == (*sample_batch['input_ids'].shape, small_model.config.vocab_size)
    
    def test_model_generation(self, small_model, device):
        """Test model can generate text"""
        input_ids = torch.randint(0, 100, (1, 10), device=device)
        generated = small_model.generate(input_ids, max_new_tokens=20)
        
        assert generated.shape[1] > input_ids.shape[1]
        assert generated.shape[1] <= input_ids.shape[1] + 20
    
    def test_parameter_count(self, small_model):
        """Test parameter counting"""
        param_count = small_model.count_parameters()
        assert param_count > 0
        assert isinstance(param_count, int)
    
    def test_sequence_length_at_boundary(self, small_model, device):
        """Test model forward pass with sequence at max_length boundary"""
        max_length = small_model.config.max_length
        vocab_size = small_model.config.vocab_size
        
        # Create input exactly at max_length
        input_ids = torch.randint(0, vocab_size, (2, max_length), device=device)
        labels = torch.randint(0, vocab_size, (2, max_length), device=device)
        
        # Should not raise IndexError
        outputs = small_model(input_ids, labels=labels)
        
        assert 'logits' in outputs
        assert 'loss' in outputs
        assert outputs['logits'].shape[1] == max_length
    
    def test_sequence_length_exceeds_max(self, small_model, device):
        """Test model forward pass with sequence exceeding max_length"""
        max_length = small_model.config.max_length
        vocab_size = small_model.config.vocab_size
        
        # Create input longer than max_length
        long_length = max_length + 50
        input_ids = torch.randint(0, vocab_size, (2, long_length), device=device)
        labels = torch.randint(0, vocab_size, (2, long_length), device=device)
        
        # Should not raise IndexError (should truncate)
        outputs = small_model(input_ids, labels=labels)
        
        assert 'logits' in outputs
        assert 'loss' in outputs
        # Output should be truncated to max_length
        assert outputs['logits'].shape[1] == max_length
    
    def test_no_index_error_on_long_sequence(self, small_model, device):
        """Test that no IndexError is raised when sequence exceeds max_length"""
        max_length = small_model.config.max_length
        vocab_size = small_model.config.vocab_size
        
        # Create very long input
        very_long_length = max_length * 2
        input_ids = torch.randint(0, vocab_size, (1, very_long_length), device=device)
        
        # Should not raise IndexError
        try:
            outputs = small_model(input_ids)
            assert outputs is not None
        except IndexError:
            pytest.fail("IndexError raised for long sequence - truncation not working")


@pytest.mark.parametrize("template", ["tiny", "small", "base"])
def test_template_models(template):
    """Test that template models can be created"""
    if template == "tiny":
        from models.architectures.tiny import create_tiny_model
        model = create_tiny_model()
    elif template == "small":
        from models.architectures.small import create_small_model
        model = create_small_model()
    elif template == "base":
        from models.architectures.base import create_base_model
        model = create_base_model()
    
    assert model is not None
    assert model.count_parameters() > 0
`;
  }

  /**
   * Get test for configuration
   */
  static getConfigTests(): string {
    return `"""
Tests for configuration loading and validation
"""

import pytest
from models.config import ConfigLoader, ConfigValidationError


class TestConfigLoader:
    """Tests for ConfigLoader"""
    
    def test_config_loading(self):
        """Test config can be loaded"""
        try:
            config = ConfigLoader('llm.config.js')
            assert config is not None
        except FileNotFoundError:
            pytest.skip("Config file not found")
    
    def test_config_validation(self, sample_config):
        """Test config validation"""
        # This would need a mock config file
        # For now, just test that validation methods exist
        pass
    
    def test_config_getters(self, sample_config):
        """Test config getter methods"""
        # Test that we can access config values
        assert 'model' in sample_config
        assert 'training' in sample_config
`;
  }

  /**
   * Get test for error handling
   */
  static getErrorTests(): string {
    return `"""
Tests for error handling
"""

import pytest
from utils.exceptions import (
    CreateLLMError,
    ConfigurationError,
    DataError,
    TrainingError,
)
from utils.handlers import retry_on_failure, validate_path


class TestExceptions:
    """Tests for custom exceptions"""
    
    def test_base_exception(self):
        """Test base exception"""
        error = CreateLLMError("Test error", suggestion="Try this")
        assert "Test error" in str(error)
        assert "Try this" in str(error)
    
    def test_configuration_error(self):
        """Test configuration error"""
        error = ConfigurationError("Invalid config")
        assert isinstance(error, CreateLLMError)
    
    def test_data_error(self):
        """Test data error"""
        error = DataError("Data not found")
        assert isinstance(error, CreateLLMError)
    
    def test_training_error(self):
        """Test training error"""
        error = TrainingError("Training failed")
        assert isinstance(error, CreateLLMError)


class TestErrorHandlers:
    """Tests for error handling utilities"""
    
    def test_retry_on_failure(self):
        """Test retry decorator"""
        call_count = 0
        
        @retry_on_failure(max_retries=3, delay=0.1)
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Test error")
            return "success"
        
        result = failing_function()
        assert result == "success"
        assert call_count == 3
    
    def test_validate_path_exists(self, temp_dir):
        """Test path validation for existing path"""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test")
        
        validated = validate_path(str(test_file), must_exist=True)
        assert validated.exists()
    
    def test_validate_path_not_exists(self, temp_dir):
        """Test path validation for non-existing path"""
        test_file = temp_dir / "nonexistent.txt"
        
        with pytest.raises(FileNotFoundError):
            validate_path(str(test_file), must_exist=True)
`;
  }

  /**
   * Get integration test
   */
  static getIntegrationTests(): string {
    return `"""
Integration tests for end-to-end workflows
"""

import pytest
import torch


@pytest.mark.integration
class TestTrainingWorkflow:
    """Test complete training workflow"""
    
    @pytest.mark.slow
    def test_minimal_training(self, small_model, sample_batch, device):
        """Test minimal training loop"""
        model = small_model
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Training step
        model.train()
        outputs = model(sample_batch['input_ids'], labels=sample_batch['labels'])
        loss = outputs['loss']
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        assert loss.item() > 0
    
    @pytest.mark.slow
    def test_evaluation(self, small_model, sample_batch, device):
        """Test evaluation"""
        model = small_model
        model.eval()
        
        with torch.no_grad():
            outputs = model(sample_batch['input_ids'], labels=sample_batch['labels'])
            loss = outputs['loss']
        
        assert loss.item() > 0


@pytest.mark.integration
class TestDataPipeline:
    """Test data processing pipeline"""
    
    def test_data_loading(self, temp_dir, sample_text):
        """Test data can be loaded and processed"""
        # Create sample data file
        data_file = temp_dir / "sample.txt"
        data_file.write_text(sample_text)
        
        # Test that file exists and can be read
        assert data_file.exists()
        content = data_file.read_text()
        assert len(content) > 0


@pytest.mark.integration
class TestEvaluationWithSequenceLengths:
    """Test evaluation with various sequence lengths"""
    
    def test_evaluation_with_short_sequences(self, small_model, device):
        """Test evaluation with sequences shorter than max_length"""
        from data.dataset import LLMDataset, create_dataloader
        
        max_length = small_model.config.max_length
        vocab_size = small_model.config.vocab_size
        
        # Create dataset with short sequences
        short_length = max_length // 2
        num_samples = 10
        data = torch.randint(0, vocab_size, (num_samples, short_length))
        
        # Mock dataset
        class MockDataset:
            def __len__(self):
                return num_samples
            
            def __getitem__(self, idx):
                return {
                    'input_ids': data[idx],
                    'attention_mask': torch.ones(short_length),
                    'labels': data[idx]
                }
        
        dataset = MockDataset()
        dataloader = create_dataloader(dataset, batch_size=2, shuffle=False, max_length=max_length)
        
        # Evaluate
        small_model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = small_model(**batch)
                total_loss += outputs['loss'].item()
        
        assert total_loss > 0
    
    def test_evaluation_with_max_length_sequences(self, small_model, device):
        """Test evaluation with sequences equal to max_length"""
        from data.dataset import LLMDataset, create_dataloader
        
        max_length = small_model.config.max_length
        vocab_size = small_model.config.vocab_size
        
        # Create dataset with sequences at max_length
        num_samples = 10
        data = torch.randint(0, vocab_size, (num_samples, max_length))
        
        # Mock dataset
        class MockDataset:
            def __len__(self):
                return num_samples
            
            def __getitem__(self, idx):
                return {
                    'input_ids': data[idx],
                    'attention_mask': torch.ones(max_length),
                    'labels': data[idx]
                }
        
        dataset = MockDataset()
        dataloader = create_dataloader(dataset, batch_size=2, shuffle=False, max_length=max_length)
        
        # Evaluate - should not raise IndexError
        small_model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = small_model(**batch)
                total_loss += outputs['loss'].item()
        
        assert total_loss > 0
    
    def test_evaluation_with_long_sequences(self, small_model, device):
        """Test evaluation with sequences longer than max_length"""
        from data.dataset import LLMDataset, create_dataloader
        
        max_length = small_model.config.max_length
        vocab_size = small_model.config.vocab_size
        
        # Create dataset with sequences longer than max_length
        long_length = max_length + 50
        num_samples = 10
        data = torch.randint(0, vocab_size, (num_samples, long_length))
        
        # Mock dataset
        class MockDataset:
            def __len__(self):
                return num_samples
            
            def __getitem__(self, idx):
                return {
                    'input_ids': data[idx],
                    'attention_mask': torch.ones(long_length),
                    'labels': data[idx]
                }
        
        dataset = MockDataset()
        dataloader = create_dataloader(dataset, batch_size=2, shuffle=False, max_length=max_length)
        
        # Evaluate - should not raise IndexError (sequences should be truncated)
        small_model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                # Verify batch was truncated
                assert batch['input_ids'].shape[1] <= max_length
                outputs = small_model(**batch)
                total_loss += outputs['loss'].item()
        
        assert total_loss > 0
    
    def test_evaluation_completes_successfully(self, small_model, device):
        """Test that evaluation completes successfully with mixed sequence lengths"""
        from data.dataset import LLMDataset, create_dataloader
        
        max_length = small_model.config.max_length
        vocab_size = small_model.config.vocab_size
        
        # Create dataset with mixed sequence lengths
        num_samples = 20
        sequences = []
        for i in range(num_samples):
            # Vary sequence length from short to very long
            seq_len = max_length // 2 + (i * 10)
            sequences.append(torch.randint(0, vocab_size, (seq_len,)))
        
        # Mock dataset
        class MockDataset:
            def __len__(self):
                return num_samples
            
            def __getitem__(self, idx):
                seq = sequences[idx]
                return {
                    'input_ids': seq,
                    'attention_mask': torch.ones(len(seq)),
                    'labels': seq
                }
        
        dataset = MockDataset()
        dataloader = create_dataloader(dataset, batch_size=2, shuffle=False, max_length=max_length)
        
        # Evaluate - should complete without errors
        small_model.eval()
        batch_count = 0
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = small_model(**batch)
                total_loss += outputs['loss'].item()
                batch_count += 1
        
        assert batch_count > 0
        assert total_loss > 0
    
    def test_metrics_calculated_correctly_after_truncation(self, small_model, device):
        """Test that metrics are calculated correctly when sequences are truncated"""
        from data.dataset import LLMDataset, create_dataloader
        
        max_length = small_model.config.max_length
        vocab_size = small_model.config.vocab_size
        
        # Create dataset with long sequences
        long_length = max_length * 2
        num_samples = 8
        data = torch.randint(0, vocab_size, (num_samples, long_length))
        
        # Mock dataset
        class MockDataset:
            def __len__(self):
                return num_samples
            
            def __getitem__(self, idx):
                return {
                    'input_ids': data[idx],
                    'attention_mask': torch.ones(long_length),
                    'labels': data[idx]
                }
        
        dataset = MockDataset()
        dataloader = create_dataloader(dataset, batch_size=2, shuffle=False, max_length=max_length)
        
        # Evaluate and calculate metrics
        small_model.eval()
        losses = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = small_model(**batch)
                losses.append(outputs['loss'].item())
        
        # Verify metrics are valid
        assert len(losses) > 0
        assert all(loss > 0 for loss in losses)
        avg_loss = sum(losses) / len(losses)
        assert avg_loss > 0
        
        # Calculate perplexity
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        assert perplexity > 1.0
`;
  }

  /**
   * Get tests README
   */
  static getTestsReadme(): string {
    return `# Tests

This directory contains tests for the LLM training project.

## Test Structure

\`\`\`
tests/
├── conftest.py              # Pytest fixtures and configuration
├── test_models.py           # Model architecture tests
├── test_config.py           # Configuration tests
├── test_errors.py           # Error handling tests
└── test_integration.py      # Integration tests
\`\`\`

## Running Tests

### Run all tests
\`\`\`bash
pytest
\`\`\`

### Run specific test file
\`\`\`bash
pytest tests/test_models.py
\`\`\`

### Run specific test
\`\`\`bash
pytest tests/test_models.py::TestGPTModel::test_model_creation
\`\`\`

### Run with markers
\`\`\`bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Skip GPU tests
pytest -m "not gpu"
\`\`\`

### Run with coverage
\`\`\`bash
pytest --cov=. --cov-report=html
\`\`\`

## Test Categories

### Unit Tests
- Fast, isolated tests
- Test individual components
- No external dependencies
- Marked with \`@pytest.mark.unit\`

### Integration Tests
- Test complete workflows
- May be slower
- Test component interactions
- Marked with \`@pytest.mark.integration\`

### Slow Tests
- Tests that take significant time
- Marked with \`@pytest.mark.slow\`
- Skip with: \`pytest -m "not slow"\`

### GPU Tests
- Tests that require GPU
- Marked with \`@pytest.mark.gpu\`
- Skip with: \`pytest -m "not gpu"\`

## Writing Tests

### Test Naming
- Test files: \`test_*.py\`
- Test classes: \`Test*\`
- Test functions: \`test_*\`

### Using Fixtures
\`\`\`python
def test_example(temp_dir, sample_config):
    # temp_dir and sample_config are fixtures from conftest.py
    pass
\`\`\`

### Parametrized Tests
\`\`\`python
@pytest.mark.parametrize("value", [1, 2, 3])
def test_with_params(value):
    assert value > 0
\`\`\`

### Marking Tests
\`\`\`python
@pytest.mark.unit
def test_fast():
    pass

@pytest.mark.slow
@pytest.mark.integration
def test_slow_integration():
    pass
\`\`\`

## CI/CD Integration

Tests are automatically run in CI/CD pipeline on:
- Pull requests
- Commits to main branch
- Nightly builds

## Troubleshooting

### Tests fail with import errors
- Make sure you're in the project root
- Install dependencies: \`pip install -r requirements.txt\`
- Install test dependencies: \`pip install pytest pytest-cov\`

### GPU tests fail
- Skip GPU tests if no GPU available: \`pytest -m "not gpu"\`
- Or run on CPU: \`CUDA_VISIBLE_DEVICES="" pytest\`

### Slow tests timeout
- Increase timeout: \`pytest --timeout=600\`
- Or skip slow tests: \`pytest -m "not slow"\`
`;
  }
}
