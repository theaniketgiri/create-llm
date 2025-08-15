#!/usr/bin/env python3
"""
Simple model testing script for the web GUI.
"""

import sys
import json
import argparse
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def load_model(model_path):
    """Load a trained model checkpoint."""
    try:
        # Try to import model components
        from model import TransformerLM, ModelConfig
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Create model config
        config = ModelConfig(
            vocab_size=checkpoint.get('vocab_size', 50257),
            n_embd=checkpoint.get('n_embd', 768),
            n_layer=checkpoint.get('n_layer', 12),
            n_head=checkpoint.get('n_head', 12),
            max_length=checkpoint.get('max_length', 1024)
        )
        
        # Create model
        model = TransformerLM(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, config
    except Exception as e:
        return None, None

def generate_text(model, config, prompt, max_length=100, temperature=0.7):
    """Generate text using the model."""
    try:
        # Simple tokenization (split by words for demo)
        tokens = prompt.split()
        
        # Convert to tensor
        input_ids = torch.tensor([[hash(token) % config.vocab_size for token in tokens]], dtype=torch.long)
        
        with torch.no_grad():
            # Generate
            generated = []
            for _ in range(max_length):
                # Get logits
                logits = model(input_ids)
                next_token_logits = logits[0, -1, :] / temperature
                
                # Sample
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Add to generated
                generated.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                
                # Stop if we hit max length
                if input_ids.shape[1] >= config.max_length:
                    break
        
        # Convert back to text (simplified)
        response = " ".join([f"token_{i}" for i in generated[:max_length]])
        return response
        
    except Exception as e:
        return f"Error generating text: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description='Test a trained LLM model')
    parser.add_argument('--prompt', type=str, required=True, help='Input prompt')
    parser.add_argument('--model', type=str, default='checkpoints/latest.pt', help='Model checkpoint path')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum generation length')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    
    args = parser.parse_args()
    
    # Load model
    model, config = load_model(args.model)
    
    if model is None:
        print(json.dumps({
            'error': 'Could not load model. Make sure the model checkpoint exists and is valid.'
        }))
        return
    
    # Generate text
    response = generate_text(
        model, config, 
        args.prompt, 
        max_length=args.max_length, 
        temperature=args.temperature
    )
    
    # Return JSON response
    print(json.dumps({
        'response': response,
        'prompt': args.prompt,
        'model_path': args.model
    }))

if __name__ == '__main__':
    main()
