#!/usr/bin/env python3
"""
Train a custom tokenizer for your LLM.

Usage:
    python train_tokenizer.py --input data/raw.txt --output tokenizer/
    python train_tokenizer.py --input data/raw.txt --output tokenizer/ --vocab-size 50000
"""

import argparse
import os
import json
from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from tokenizers.processors import TemplateProcessing

def create_tokenizer(tokenizer_type: str = "bpe"):
    """Create a tokenizer with the specified type."""
    
    if tokenizer_type == "bpe":
        tokenizer = Tokenizer(models.BPE())
        trainer = trainers.BpeTrainer(
            vocab_size=50000,
            special_tokens=["<s>", "</s>", "<unk>", "<pad>"]
        )
    elif tokenizer_type == "wordpiece":
        tokenizer = Tokenizer(models.WordPiece())
        trainer = trainers.WordPieceTrainer(
            vocab_size=50000,
            special_tokens=["<s>", "</s>", "<unk>", "<pad>"]
        )
    elif tokenizer_type == "unigram":
        tokenizer = Tokenizer(models.Unigram())
        trainer = trainers.UnigramTrainer(
            vocab_size=50000,
            special_tokens=["<s>", "</s>", "<unk>", "<pad>"]
        )
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
    
    # Pre-tokenizer
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # Decoder
    tokenizer.decoder = decoders.ByteLevel()
    
    # Post-processor
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A:0 $B:1 </s>",
        special_tokens=[
            ("<s>", 0),
            ("</s>", 1),
        ],
    )
    
    return tokenizer, trainer

def train_tokenizer(input_file: str, output_dir: str, vocab_size: int = 50000, tokenizer_type: str = "bpe"):
    """Train a tokenizer on the input file."""
    
    print(f"Training {tokenizer_type.upper()} tokenizer...")
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    print(f"Vocabulary size: {vocab_size}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create tokenizer
    tokenizer, trainer = create_tokenizer(tokenizer_type)
    
    # Update trainer vocab size
    if hasattr(trainer, 'vocab_size'):
        trainer.vocab_size = vocab_size
    
    # Train the tokenizer
    print("Training tokenizer...")
    tokenizer.train([input_file], trainer)
    
    # Save the tokenizer
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    
    # Save vocabulary
    vocab = tokenizer.get_vocab()
    vocab_path = os.path.join(output_dir, "vocab.json")
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    # Save special tokens
    special_tokens = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "vocab_size": len(vocab)
    }
    special_tokens_path = os.path.join(output_dir, "special_tokens.json")
    with open(special_tokens_path, 'w', encoding='utf-8') as f:
        json.dump(special_tokens, f, ensure_ascii=False, indent=2)
    
    print(f"Tokenizer saved to {tokenizer_path}")
    print(f"Vocabulary saved to {vocab_path}")
    print(f"Special tokens saved to {special_tokens_path}")
    
    # Test the tokenizer
    test_text = "Hello, world! This is a test of the tokenizer."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded.ids)
    
    print(f"\nTest encoding:")
    print(f"Input: {test_text}")
    print(f"Tokens: {encoded.tokens}")
    print(f"IDs: {encoded.ids}")
    print(f"Decoded: {decoded}")
    
    return tokenizer

def main():
    parser = argparse.ArgumentParser(description="Train a custom tokenizer")
    parser.add_argument("--input", "-i", required=True, help="Input text file")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--vocab-size", "-v", type=int, default=50000, help="Vocabulary size")
    parser.add_argument("--type", "-t", choices=["bpe", "wordpiece", "unigram"], 
                       default="bpe", help="Tokenizer type")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist")
        return 1
    
    try:
        train_tokenizer(args.input, args.output, args.vocab_size, args.type)
        return 0
    except Exception as e:
        print(f"Error training tokenizer: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
