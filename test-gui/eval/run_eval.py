#!/usr/bin/env python3
"""
Evaluation script for the trained LLM.

Usage:
    python run_eval.py --model checkpoints/best.pt --data data/processed/test.txt
    python run_eval.py --model checkpoints/best.pt --generate --prompt "Hello, world!"
"""

import argparse
import os
import yaml
import torch
import json
from pathlib import Path

from model import TransformerLM, ModelConfig
from tokenizer import CustomTokenizer
from data import create_test_dataloader
from eval.evaluator import Evaluator
from eval.metrics import compute_perplexity, compute_accuracy

def load_model(checkpoint_path: str, tokenizer_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model config
    config = checkpoint['config']
    model_config = ModelConfig(
        vocab_size=config.get('vocab_size', 50257),
        n_positions=config.get('max_length', 1024),
        n_embd=config.get('n_embd', 768),
        n_layer=config.get('n_layer', 12),
        n_head=config.get('n_head', 12),
        model_type="gpt"
    )
    
    # Create model
    model = TransformerLM(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, config

def evaluate_model(model, dataloader, device):
    """Evaluate model on test set."""
    evaluator = Evaluator(model, device)
    
    # Compute perplexity
    perplexity = evaluator.compute_perplexity(dataloader)
    
    # Compute accuracy
    accuracy = evaluator.compute_accuracy(dataloader)
    
    # Compute other metrics
    metrics = evaluator.compute_all_metrics(dataloader)
    
    return {
        'perplexity': perplexity,
        'accuracy': accuracy,
        **metrics
    }

def generate_text(model, tokenizer, prompt: str, max_length: int = 100, device: torch.device = None):
    """Generate text from prompt."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return generated_text

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained LLM")
    parser.add_argument("--model", "-m", required=True, help="Path to model checkpoint")
    parser.add_argument("--data", "-d", help="Path to test data")
    parser.add_argument("--tokenizer", "-t", default="tokenizer/tokenizer.json", help="Path to tokenizer")
    parser.add_argument("--output", "-o", default="eval_results.json", help="Output file for results")
    parser.add_argument("--generate", "-g", action="store_true", help="Generate text instead of evaluating")
    parser.add_argument("--prompt", "-p", default="Hello, world!", help="Prompt for text generation")
    parser.add_argument("--max-length", "-l", type=int, default=100, help="Maximum generation length")
    parser.add_argument("--batch-size", "-b", type=int, default=8, help="Batch size for evaluation")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = CustomTokenizer(args.tokenizer)
    
    # Load model
    model, config = load_model(args.model, args.tokenizer, device)
    print(f"Loaded model from {args.model}")
    
    if args.generate:
        # Generate text
        print(f"Generating text with prompt: '{args.prompt}'")
        generated_text = generate_text(
            model, tokenizer, args.prompt, args.max_length, device
        )
        print(f"Generated text: {generated_text}")
        
    else:
        # Evaluate model
        if not args.data:
            print("Error: Test data path required for evaluation")
            return 1
        
        print(f"Evaluating model on {args.data}")
        
        # Create test dataloader
        test_dataloader = create_test_dataloader(
            tokenizer=tokenizer,
            test_path=args.data,
            batch_size=args.batch_size,
            max_length=config.get('max_length', 1024),
            num_workers=4
        )
        
        # Evaluate
        results = evaluate_model(model, test_dataloader, device)
        
        # Print results
        print("\nEvaluation Results:")
        print(f"Perplexity: {results['perplexity']:.4f}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Loss: {results['loss']:.4f}")
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {args.output}")
    
    return 0

if __name__ == "__main__":
    exit(main())
