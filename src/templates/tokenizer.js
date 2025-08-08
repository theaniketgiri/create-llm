const fs = require('fs-extra');
const path = require('path');

async function createTokenizerFiles(projectPath, options) {
  const tokenizerPath = path.join(projectPath, 'tokenizer');
  
  // Create __init__.py
  await fs.writeFile(path.join(tokenizerPath, '__init__.py'), `from .tokenizer import CustomTokenizer

__all__ = ['CustomTokenizer']
`);

  // Create main tokenizer script
  await createTokenizerScript(tokenizerPath, options);
  
  // Create training script
  await createTrainingScript(tokenizerPath, options);
  
  // Create tokenizer class
  await createTokenizerClass(tokenizerPath, options);
}

async function createTokenizerScript(tokenizerPath, options) {
  const scriptContent = `#!/usr/bin/env python3
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
    
    print(f"\\nTest encoding:")
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
                       default="${options.tokenizer}", help="Tokenizer type")
    
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
`;
  
  await fs.writeFile(path.join(tokenizerPath, 'train_tokenizer.py'), scriptContent);
}

async function createTrainingScript(tokenizerPath, options) {
  const trainingContent = `#!/usr/bin/env python3
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
                       default="${options.tokenizer}", help="Tokenizer type")
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
`;
  
  await fs.writeFile(path.join(tokenizerPath, 'train_multiple.py'), trainingContent);
}

async function createTokenizerClass(tokenizerPath, options) {
  const classContent = `import json
import os
from typing import List, Dict, Optional
from tokenizers import Tokenizer as HFTokenizer

class CustomTokenizer:
    """Custom tokenizer wrapper for easy use in training."""
    
    def __init__(self, tokenizer_path: str):
        """
        Initialize the tokenizer.
        
        Args:
            tokenizer_path: Path to the tokenizer.json file
        """
        self.tokenizer = HFTokenizer.from_file(tokenizer_path)
        self.tokenizer_path = tokenizer_path
        
        # Load special tokens
        special_tokens_path = os.path.join(os.path.dirname(tokenizer_path), "special_tokens.json")
        if os.path.exists(special_tokens_path):
            with open(special_tokens_path, 'r', encoding='utf-8') as f:
                self.special_tokens = json.load(f)
        else:
            self.special_tokens = {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>"
            }
        
        # Set token IDs
        self.bos_token_id = self.tokenizer.token_to_id(self.special_tokens["bos_token"])
        self.eos_token_id = self.tokenizer.token_to_id(self.special_tokens["eos_token"])
        self.unk_token_id = self.tokenizer.token_to_id(self.special_tokens["unk_token"])
        self.pad_token_id = self.tokenizer.token_to_id(self.special_tokens["pad_token"])
        
        # Vocabulary
        self.vocab = self.tokenizer.get_vocab()
        self.vocab_size = len(self.vocab)
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add special tokens
            
        Returns:
            List of token IDs
        """
        encoded = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return encoded.ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def encode_batch(self, texts: List[str], add_special_tokens: bool = True) -> List[List[int]]:
        """
        Encode a batch of texts.
        
        Args:
            texts: List of input texts
            add_special_tokens: Whether to add special tokens
            
        Returns:
            List of token ID lists
        """
        encoded = self.tokenizer.encode_batch(texts, add_special_tokens=add_special_tokens)
        return [e.ids for e in encoded]
    
    def decode_batch(self, token_id_lists: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
        """
        Decode a batch of token ID lists.
        
        Args:
            token_id_lists: List of token ID lists
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            List of decoded texts
        """
        return [self.decode(ids, skip_special_tokens) for ids in token_id_lists]
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into tokens.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        encoded = self.tokenizer.encode(text)
        return encoded.tokens
    
    def get_vocab(self) -> Dict[str, int]:
        """Get the vocabulary mapping."""
        return self.vocab.copy()
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size."""
        return self.vocab_size
    
    def save(self, path: str):
        """Save the tokenizer to a file."""
        self.tokenizer.save(path)
    
    @classmethod
    def from_pretrained(cls, path: str):
        """Load a tokenizer from a saved path."""
        return cls(path)
`;
  
  await fs.writeFile(path.join(tokenizerPath, 'tokenizer.py'), classContent);
}

module.exports = { createTokenizerFiles }; 