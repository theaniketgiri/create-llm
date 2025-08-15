#!/usr/bin/env python3
"""
Advanced tokenizer training with multiple files and options.
"""

import argparse
import os
import glob
from pathlib import Path
from train_tokenizer import train_tokenizer

def find_text_files(input_dir: str, pattern: str = "*.txt"):
    """Find all text files in the input directory."""
    pattern = os.path.join(input_dir, "**", pattern)
    files = glob.glob(pattern, recursive=True)
    return files

def train_on_multiple_files(input_dir: str, output_dir: str, vocab_size: int = 50000, 
                          tokenizer_type: str = "bpe", file_pattern: str = "*.txt"):
    """Train tokenizer on multiple files."""
    
    # Find all text files
    files = find_text_files(input_dir, file_pattern)
    
    if not files:
        print(f"No files found matching pattern: {file_pattern}")
        return None
    
    print(f"Found {len(files)} files to train on:")
    for f in files[:5]:  # Show first 5 files
        print(f"  - {f}")
    if len(files) > 5:
        print(f"  ... and {len(files) - 5} more files")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Import tokenizer creation function
    from train_tokenizer import create_tokenizer
    tokenizer, trainer = create_tokenizer(tokenizer_type)
    
    # Update trainer vocab size
    if hasattr(trainer, 'vocab_size'):
        trainer.vocab_size = vocab_size
    
    # Train on all files
    print(f"Training {tokenizer_type.upper()} tokenizer on {len(files)} files...")
    tokenizer.train(files, trainer)
    
    # Save the tokenizer
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    
    print(f"Tokenizer saved to {tokenizer_path}")
    
    return tokenizer

def main():
    parser = argparse.ArgumentParser(description="Train tokenizer on multiple files")
    parser.add_argument("--input-dir", "-i", required=True, help="Input directory with text files")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory")
    parser.add_argument("--vocab-size", "-v", type=int, default=50000, help="Vocabulary size")
    parser.add_argument("--type", "-t", choices=["bpe", "wordpiece", "unigram"], 
                       default="bpe", help="Tokenizer type")
    parser.add_argument("--pattern", "-p", default="*.txt", help="File pattern to match")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return 1
    
    try:
        train_on_multiple_files(args.input_dir, args.output_dir, args.vocab_size, 
                              args.type, args.pattern)
        return 0
    except Exception as e:
        print(f"Error training tokenizer: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
