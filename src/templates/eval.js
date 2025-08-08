const fs = require('fs-extra');
const path = require('path');

async function createEvalFiles(projectPath, options) {
  const evalPath = path.join(projectPath, 'eval');
  
  // Create __init__.py
  await fs.writeFile(path.join(evalPath, '__init__.py'), `from .evaluator import Evaluator
from .metrics import compute_perplexity, compute_accuracy

__all__ = ['Evaluator', 'compute_perplexity', 'compute_accuracy']
`);

  // Create main evaluation script
  await createEvalScript(evalPath, options);
  
  // Create evaluator class
  await createEvaluatorClass(evalPath, options);
  
  // Create metrics
  await createMetrics(evalPath, options);
  
  // Create generation script
  await createGenerationScript(evalPath, options);
}

async function createEvalScript(evalPath, options) {
  const scriptContent = `#!/usr/bin/env python3
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
        model_type="${options.template}"
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
        print("\\nEvaluation Results:")
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
`;
  
  await fs.writeFile(path.join(evalPath, 'run_eval.py'), scriptContent);
}

async function createEvaluatorClass(evalPath, options) {
  const evaluatorContent = `import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional
import math

class Evaluator:
    """Evaluator class for LLM evaluation."""
    
    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model
            device: Device to evaluate on
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    def compute_perplexity(self, dataloader: DataLoader) -> float:
        """
        Compute perplexity on dataset.
        
        Args:
            dataloader: Data loader for evaluation
            
        Returns:
            Perplexity score
        """
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs['logits']
                
                # Shift for next token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # Compute loss
                loss_fct = nn.CrossEntropyLoss(reduction='sum')
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                # Count non-padding tokens
                num_tokens = (shift_labels != -100).sum().item()
                
                total_loss += loss.item()
                total_tokens += num_tokens
        
        # Compute perplexity
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        return perplexity
    
    def compute_accuracy(self, dataloader: DataLoader) -> float:
        """
        Compute accuracy on dataset.
        
        Args:
            dataloader: Data loader for evaluation
            
        Returns:
            Accuracy score
        """
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs['logits']
                
                # Shift for next token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # Get predictions
                predictions = torch.argmax(shift_logits, dim=-1)
                
                # Count correct predictions (ignore padding)
                mask = (shift_labels != -100)
                correct = (predictions == shift_labels) & mask
                
                correct_predictions += correct.sum().item()
                total_predictions += mask.sum().item()
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        return accuracy
    
    def compute_loss(self, dataloader: DataLoader) -> float:
        """
        Compute average loss on dataset.
        
        Args:
            dataloader: Data loader for evaluation
            
        Returns:
            Average loss
        """
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs['logits']
                
                # Shift for next token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # Compute loss
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def compute_all_metrics(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Compute all evaluation metrics.
        
        Args:
            dataloader: Data loader for evaluation
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Compute basic metrics
        metrics['loss'] = self.compute_loss(dataloader)
        metrics['perplexity'] = self.compute_perplexity(dataloader)
        metrics['accuracy'] = self.compute_accuracy(dataloader)
        
        return metrics
    
    def generate_samples(
        self,
        tokenizer,
        prompts: List[str],
        max_length: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        num_samples: int = 1
    ) -> List[str]:
        """
        Generate text samples from prompts.
        
        Args:
            tokenizer: Tokenizer instance
            prompts: List of input prompts
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            num_samples: Number of samples per prompt
            
        Returns:
            List of generated texts
        """
        generated_texts = []
        
        for prompt in prompts:
            for _ in range(num_samples):
                # Encode prompt
                input_ids = tokenizer.encode(prompt, add_special_tokens=True)
                input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
                
                # Generate
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        input_ids=input_ids,
                        max_length=max_length,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                
                # Decode
                generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                generated_texts.append(generated_text)
        
        return generated_texts
`;
  
  await fs.writeFile(path.join(evalPath, 'evaluator.py'), evaluatorContent);
}

async function createMetrics(evalPath, options) {
  const metricsContent = `import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple
import math

def compute_perplexity(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> float:
    """
    Compute perplexity from logits and labels.
    
    Args:
        logits: Model output logits [batch_size, seq_len, vocab_size]
        labels: Target labels [batch_size, seq_len]
        ignore_index: Index to ignore in loss computation
        
    Returns:
        Perplexity score
    """
    # Shift for next token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Compute loss
    loss_fct = nn.CrossEntropyLoss(reduction='sum', ignore_index=ignore_index)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    # Count non-ignored tokens
    num_tokens = (shift_labels != ignore_index).sum().item()
    
    # Compute perplexity
    avg_loss = loss.item() / num_tokens if num_tokens > 0 else 0.0
    perplexity = math.exp(avg_loss)
    
    return perplexity

def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> float:
    """
    Compute accuracy from logits and labels.
    
    Args:
        logits: Model output logits [batch_size, seq_len, vocab_size]
        labels: Target labels [batch_size, seq_len]
        ignore_index: Index to ignore in accuracy computation
        
    Returns:
        Accuracy score
    """
    # Shift for next token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Get predictions
    predictions = torch.argmax(shift_logits, dim=-1)
    
    # Count correct predictions (ignore specified index)
    mask = (shift_labels != ignore_index)
    correct = (predictions == shift_labels) & mask
    
    accuracy = correct.sum().item() / mask.sum().item() if mask.sum().item() > 0 else 0.0
    return accuracy

def compute_bleu_score(predictions: List[str], references: List[str]) -> float:
    """
    Compute BLEU score for text generation.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        
    Returns:
        BLEU score
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from nltk.tokenize import word_tokenize
    except ImportError:
        print("NLTK not available. Install with: pip install nltk")
        return 0.0
    
    smoothie = SmoothingFunction().method1
    
    total_bleu = 0.0
    for pred, ref in zip(predictions, references):
        pred_tokens = word_tokenize(pred.lower())
        ref_tokens = word_tokenize(ref.lower())
        
        # Compute BLEU-4
        bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
        total_bleu += bleu
    
    avg_bleu = total_bleu / len(predictions) if predictions else 0.0
    return avg_bleu

def compute_rouge_score(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute ROUGE scores for text generation.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        
    Returns:
        Dictionary of ROUGE scores
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("rouge-score not available. Install with: pip install rouge-score")
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    total_scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for key in total_scores:
            total_scores[key] += scores[key].fmeasure
    
    # Average scores
    num_samples = len(predictions)
    avg_scores = {key: score / num_samples for key, score in total_scores.items()}
    
    return avg_scores

def compute_diversity_metrics(generated_texts: List[str]) -> Dict[str, float]:
    """
    Compute diversity metrics for generated texts.
    
    Args:
        generated_texts: List of generated texts
        
    Returns:
        Dictionary of diversity metrics
    """
    if not generated_texts:
        return {'distinct_1': 0.0, 'distinct_2': 0.0, 'entropy': 0.0}
    
    # Tokenize all texts
    all_tokens = []
    for text in generated_texts:
        tokens = text.lower().split()
        all_tokens.extend(tokens)
    
    if not all_tokens:
        return {'distinct_1': 0.0, 'distinct_2': 0.0, 'entropy': 0.0}
    
    # Compute distinct-1 (unique unigrams)
    unique_unigrams = set(all_tokens)
    distinct_1 = len(unique_unigrams) / len(all_tokens)
    
    # Compute distinct-2 (unique bigrams)
    bigrams = []
    for i in range(len(all_tokens) - 1):
        bigrams.append((all_tokens[i], all_tokens[i + 1]))
    
    unique_bigrams = set(bigrams)
    distinct_2 = len(unique_bigrams) / len(bigrams) if bigrams else 0.0
    
    # Compute entropy
    token_counts = {}
    for token in all_tokens:
        token_counts[token] = token_counts.get(token, 0) + 1
    
    total_tokens = len(all_tokens)
    entropy = 0.0
    for count in token_counts.values():
        prob = count / total_tokens
        entropy -= prob * math.log2(prob)
    
    return {
        'distinct_1': distinct_1,
        'distinct_2': distinct_2,
        'entropy': entropy
    }

def compute_fluency_score(generated_texts: List[str]) -> float:
    """
    Compute fluency score using a simple heuristic.
    
    Args:
        generated_texts: List of generated texts
        
    Returns:
        Fluency score (0-1)
    """
    if not generated_texts:
        return 0.0
    
    total_score = 0.0
    
    for text in generated_texts:
        # Simple fluency heuristics
        score = 0.0
        
        # Check for reasonable sentence length
        sentences = text.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        if 5 <= avg_sentence_length <= 30:
            score += 0.3
        
        # Check for proper capitalization
        if text and text[0].isupper():
            score += 0.2
        
        # Check for reasonable word length
        words = text.split()
        avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
        if 3 <= avg_word_length <= 8:
            score += 0.2
        
        # Check for punctuation
        if any(p in text for p in ['.', '!', '?', ',']):
            score += 0.2
        
        # Check for no excessive repetition
        if len(set(words)) / len(words) > 0.7 if words else True:
            score += 0.1
        
        total_score += score
    
    avg_fluency = total_score / len(generated_texts)
    return avg_fluency
`;
  
  await fs.writeFile(path.join(evalPath, 'metrics.py'), metricsContent);
}

async function createGenerationScript(evalPath, options) {
  const generationContent = `#!/usr/bin/env python3
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
        model_type="${options.template}"
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
            prompt = input("\\nEnter prompt: ").strip()
            
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
            
            print(f"\\nGenerated text:")
            print(f"{generated_texts[0]}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\\nGoodbye!")

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
            print(f"\\nGenerated text {i+1}:")
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
            
            print(f"\\nResults saved to {args.output}")
    
    return 0

if __name__ == "__main__":
    exit(main())
`;
  
  await fs.writeFile(path.join(evalPath, 'generate.py'), generationContent);
}

module.exports = { createEvalFiles }; 