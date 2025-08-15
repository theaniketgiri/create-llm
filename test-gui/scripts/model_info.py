#!/usr/bin/env python3
"""
Display model information and statistics.
"""

import argparse
import torch
from model import TransformerLM, ModelConfig

def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size_mb(model):
    """Get model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb

def main():
    parser = argparse.ArgumentParser(description="Display model information")
    parser.add_argument("--config", "-c", help="Path to model config file")
    parser.add_argument("--checkpoint", help="Path to model checkpoint")
    
    args = parser.parse_args()
    
    if args.checkpoint:
        # Load from checkpoint
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        config = checkpoint['config']
        model_config = ModelConfig(
            vocab_size=config.get('vocab_size', 50257),
            n_positions=config.get('max_length', 1024),
            n_embd=config.get('n_embd', 768),
            n_layer=config.get('n_layer', 12),
            n_head=config.get('n_head', 12),
            model_type="gpt"
        )
    else:
        # Use default config
        model_config = ModelConfig()
    
    # Create model
    model = TransformerLM(model_config)
    
    # Calculate statistics
    num_params = count_parameters(model)
    model_size = get_model_size_mb(model)
    
    print("Model Information:")
    print(f"Architecture: {model_config.model_type.upper()}")
    print(f"Vocabulary size: {model_config.vocab_size:,}")
    print(f"Max sequence length: {model_config.n_positions:,}")
    print(f"Embedding dimension: {model_config.n_embd:,}")
    print(f"Number of layers: {model_config.n_layer}")
    print(f"Number of attention heads: {model_config.n_head}")
    print(f"Total parameters: {num_params:,}")
    print(f"Model size: {model_size:.2f} MB")
    
    # Memory requirements
    batch_size = 1
    seq_len = model_config.n_positions
    memory_per_sample = (batch_size * seq_len * model_config.n_embd * 4) / 1024**2  # 4 bytes per float32
    print(f"Memory per sample (batch_size=1): {memory_per_sample:.2f} MB")

if __name__ == "__main__":
    main()
