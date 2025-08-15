#!/usr/bin/env python3
"""
Text generation script for the trained LLM.

Usage:
    python generate.py --model checkpoints/best.pt --prompt "Hello, world!"
    python generate.py --model checkpoints/best.pt --interactive
"""

import argparse
import torch
import json
from typing import List, Dict

from model import TransformerLM, ModelConfig
from tokenizer import CustomTokenizer

def load_model(checkpoint_path: str, device: torch.device):
    """Load trained model from checkpoint."""
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
    
    return model

def generate_text(
    model: TransformerLM,
    tokenizer: CustomTokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    do_sample: bool = True,
    num_return_sequences: int = 1
) -> List[str]:
    """
    Generate text from prompt.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        prompt: Input prompt
        max_length: Maximum generation length
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Top-p sampling
        do_sample: Whether to use sampling
        num_return_sequences: Number of sequences to generate
        
    Returns:
        List of generated texts
    """
    device = next(model.parameters()).device
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    generated_texts = []
    
    for _ in range(num_return_sequences):
        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        generated_texts.append(generated_text)
    
    return generated_texts

def interactive_mode(model: TransformerLM, tokenizer: CustomTokenizer):
    """Run interactive text generation."""
    print("Interactive text generation mode. Type 'quit' to exit.")
    print("-" * 50)
    
    while True:
        try:
            prompt = input("\nEnter prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            
            if not prompt:
                continue
            
            # Generate text
            generated_texts = generate_text(
                model, tokenizer, prompt,
                max_length=100,
                temperature=0.8,
                top_k=50,
                top_p=0.9
            )
            
            print(f"\nGenerated text:")
            print(f"{generated_texts[0]}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nGoodbye!")

def main():
    parser = argparse.ArgumentParser(description="Generate text with trained LLM")
    parser.add_argument("--model", "-m", required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer", "-t", default="tokenizer/tokenizer.json", help="Path to tokenizer")
    parser.add_argument("--prompt", "-p", help="Input prompt")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--max-length", "-l", type=int, default=100, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--num-sequences", "-n", type=int, default=1, help="Number of sequences to generate")
    parser.add_argument("--output", "-o", help="Output file for generated text")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = CustomTokenizer(args.tokenizer)
    
    # Load model
    model = load_model(args.model, device)
    print(f"Loaded model from {args.model}")
    
    if args.interactive:
        # Interactive mode
        interactive_mode(model, tokenizer)
    else:
        # Single generation
        if not args.prompt:
            print("Error: Prompt required for non-interactive mode")
            return 1
        
        print(f"Generating text with prompt: '{args.prompt}'")
        
        generated_texts = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_return_sequences=args.num_sequences
        )
        
        # Print results
        for i, text in enumerate(generated_texts):
            print(f"\nGenerated text {i+1}:")
            print(f"{text}")
        
        # Save to file if specified
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump({
                    'prompt': args.prompt,
                    'generated_texts': generated_texts,
                    'parameters': {
                        'max_length': args.max_length,
                        'temperature': args.temperature,
                        'top_k': args.top_k,
                        'top_p': args.top_p
                    }
                }, f, indent=2, ensure_ascii=False)
            
            print(f"\nResults saved to {args.output}")
    
    return 0

if __name__ == "__main__":
    exit(main())
