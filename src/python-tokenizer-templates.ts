/**
 * Python tokenizer training templates
 */

export class PythonDataTemplates {
  /**
   * Get data preparation script
   */
  static getDataPrepareScript(): string {
    return `#!/usr/bin/env python3
"""
Data preprocessing pipeline
Loads, tokenizes, and prepares text data for training
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple
import torch
from tokenizers import Tokenizer


class DataPreprocessor:
    """Data preprocessing pipeline for LLM training"""
    
    def __init__(self, tokenizer_path: str, max_length: int = 512, stride: int = 256):
        self.tokenizer = self._load_tokenizer(tokenizer_path)
        self.max_length = max_length
        self.stride = stride
    
    def _load_tokenizer(self, path: str) -> Tokenizer:
        """Load trained tokenizer"""
        tokenizer_path = Path(path)
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found: {path}\\n"
                f"Train a tokenizer first: python tokenizer/train.py --data data/raw/"
            )
        return Tokenizer.from_file(str(tokenizer_path))
    
    def load_text_files(self, data_path: str) -> str:
        """Load and concatenate all text files"""
        path = Path(data_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Data path not found: {data_path}")
        
        texts = []
        if path.is_file():
            with open(path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
        elif path.is_dir():
            files = sorted(path.glob('**/*.txt'))
            if not files:
                raise ValueError(f"No .txt files found in {data_path}")
            
            for file in files:
                with open(file, 'r', encoding='utf-8') as f:
                    texts.append(f.read())
        
        return '\\n'.join(texts)
    
    def create_examples(self, text: str) -> List[List[int]]:
        """Create training examples using sliding window"""
        # Tokenize entire text
        encoding = self.tokenizer.encode(text)
        token_ids = encoding.ids
        
        print(f"Total tokens: {len(token_ids):,}")
        
        # Create sliding window examples
        examples = []
        for i in range(0, len(token_ids) - self.max_length + 1, self.stride):
            example = token_ids[i:i + self.max_length]
            if len(example) == self.max_length:
                examples.append(example)
        
        # Add final example if there's remaining text
        if len(token_ids) > self.max_length:
            final_start = len(token_ids) - self.max_length
            if final_start > 0 and (not examples or examples[-1] != token_ids[final_start:]):
                examples.append(token_ids[final_start:])
        
        return examples
    
    def split_train_val(
        self, examples: List[List[int]], val_split: float = 0.1
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """Split data into train and validation sets"""
        n_val = int(len(examples) * val_split)
        n_train = len(examples) - n_val
        
        train_examples = examples[:n_train]
        val_examples = examples[n_train:]
        
        return train_examples, val_examples
    
    def save_examples(self, examples: List[List[int]], output_path: str):
        """Save examples as PyTorch tensor"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to tensor
        tensor = torch.tensor(examples, dtype=torch.long)
        
        # Save
        torch.save(tensor, output_path)
        print(f"Saved {len(examples):,} examples to {output_path}")
    
    def display_statistics(
        self, train_examples: List[List[int]], val_examples: List[List[int]]
    ):
        """Display dataset statistics"""
        vocab_size = self.tokenizer.get_vocab_size()
        
        print(f"\\n{'='*60}")
        print("Dataset Statistics")
        print(f"{'='*60}")
        print(f"Vocabulary size: {vocab_size:,}")
        print(f"Max sequence length: {self.max_length}")
        print(f"Stride: {self.stride}")
        print(f"\\nTraining examples: {len(train_examples):,}")
        print(f"Validation examples: {len(val_examples):,}")
        print(f"Total examples: {len(train_examples) + len(val_examples):,}")
        
        # Calculate total tokens
        total_train_tokens = len(train_examples) * self.max_length
        total_val_tokens = len(val_examples) * self.max_length
        print(f"\\nTraining tokens: {total_train_tokens:,}")
        print(f"Validation tokens: {total_val_tokens:,}")
        print(f"Total tokens: {total_train_tokens + total_val_tokens:,}")
        print(f"{'='*60}\\n")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare training data for LLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare data with default settings
  python data/prepare.py --input data/raw/ --tokenizer tokenizer/tokenizer.json
  
  # Prepare with custom max length and stride
  python data/prepare.py --input data/raw/train.txt --max-length 1024 --stride 512
  
  # Prepare without validation split
  python data/prepare.py --input data/raw/ --val-split 0
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='data/raw/',
        help='Input data path (file or directory)'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='tokenizer/tokenizer.json',
        help='Path to trained tokenizer'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed/',
        help='Output directory for processed data'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=512,
        help='Maximum sequence length (default: 512)'
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=256,
        help='Sliding window stride (default: 256)'
    )
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.1,
        help='Validation split ratio (default: 0.1)'
    )
    
    args = parser.parse_args()
    
    try:
        print("Loading data and tokenizer...")
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor(
            args.tokenizer,
            max_length=args.max_length,
            stride=args.stride
        )
        
        # Load text files
        print(f"Loading text from: {args.input}")
        text = preprocessor.load_text_files(args.input)
        print(f"Loaded {len(text):,} characters")
        
        # Create examples
        print(f"\\nCreating examples (max_length={args.max_length}, stride={args.stride})...")
        examples = preprocessor.create_examples(text)
        print(f"Created {len(examples):,} examples")
        
        # Split train/val
        if args.val_split > 0:
            print(f"\\nSplitting data (val_split={args.val_split})...")
            train_examples, val_examples = preprocessor.split_train_val(
                examples, args.val_split
            )
        else:
            train_examples = examples
            val_examples = []
        
        # Save examples
        print(f"\\nSaving processed data to: {args.output_dir}")
        preprocessor.save_examples(
            train_examples,
            f"{args.output_dir}/train.pt"
        )
        if val_examples:
            preprocessor.save_examples(
                val_examples,
                f"{args.output_dir}/val.pt"
            )
        
        # Display statistics
        preprocessor.display_statistics(train_examples, val_examples)
        
        print("✓ Data preparation complete!")
        print(f"\\nNext steps:")
        print(f"  1. Start training: python training/train.py")
        
    except Exception as e:
        print(f"\\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
`;
  }
}

export class PythonTokenizerTemplates {
  /**
   * Get tokenizer training script
   */
  static getTokenizerTrainScript(): string {
    return `#!/usr/bin/env python3
"""
Tokenizer training script
Trains BPE, WordPiece, or Unigram tokenizers on your data
"""

import argparse
import sys
from pathlib import Path
from typing import List
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece, Unigram
from tokenizers.trainers import BpeTrainer, WordPieceTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence


def load_text_files(data_path: str) -> List[str]:
    """Load all text files from directory or single file"""
    path = Path(data_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")
    
    files = []
    if path.is_file():
        files = [str(path)]
    elif path.is_dir():
        files = [str(f) for f in path.glob('**/*.txt')]
        if not files:
            raise ValueError(f"No .txt files found in {data_path}")
    else:
        raise ValueError(f"Invalid path: {data_path}")
    
    print(f"Found {len(files)} text file(s)")
    return files


def train_bpe_tokenizer(
    files: List[str],
    vocab_size: int,
    min_frequency: int,
    special_tokens: List[str]
) -> Tokenizer:
    """Train BPE (Byte Pair Encoding) tokenizer"""
    print("\\nTraining BPE tokenizer...")
    
    # Initialize tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    
    # Set normalizer and pre-tokenizer
    tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    
    # Configure trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True
    )
    
    # Train
    tokenizer.train(files, trainer)
    
    return tokenizer


def train_wordpiece_tokenizer(
    files: List[str],
    vocab_size: int,
    min_frequency: int,
    special_tokens: List[str]
) -> Tokenizer:
    """Train WordPiece tokenizer"""
    print("\\nTraining WordPiece tokenizer...")
    
    # Initialize tokenizer
    tokenizer = Tokenizer(WordPiece(unk_token="<unk>"))
    
    # Set normalizer and pre-tokenizer
    tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    
    # Configure trainer
    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True
    )
    
    # Train
    tokenizer.train(files, trainer)
    
    return tokenizer


def train_unigram_tokenizer(
    files: List[str],
    vocab_size: int,
    min_frequency: int,
    special_tokens: List[str]
) -> Tokenizer:
    """Train Unigram tokenizer"""
    print("\\nTraining Unigram tokenizer...")
    
    # Initialize tokenizer
    tokenizer = Tokenizer(Unigram())
    
    # Set normalizer and pre-tokenizer
    tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    
    # Configure trainer
    trainer = UnigramTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        unk_token="<unk>"
    )
    
    # Train
    tokenizer.train(files, trainer)
    
    return tokenizer


def display_statistics(tokenizer: Tokenizer, sample_text: str):
    """Display tokenizer statistics"""
    vocab_size = tokenizer.get_vocab_size()
    
    print(f"\\n{'='*60}")
    print("Tokenizer Statistics")
    print(f"{'='*60}")
    print(f"Vocabulary size: {vocab_size:,}")
    
    # Test encoding
    encoding = tokenizer.encode(sample_text)
    tokens = encoding.tokens
    ids = encoding.ids
    
    print(f"\\nSample encoding:")
    print(f"Text: {sample_text}")
    print(f"Tokens: {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
    print(f"Token count: {len(tokens)}")
    print(f"{'='*60}\\n")


def main():
    parser = argparse.ArgumentParser(
        description='Train a tokenizer on your text data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train BPE tokenizer on single file
  python tokenizer/train.py --data data/raw/train.txt --type bpe
  
  # Train WordPiece tokenizer on directory
  python tokenizer/train.py --data data/raw/ --type wordpiece --vocab-size 50000
  
  # Train with custom special tokens
  python tokenizer/train.py --data data/raw/ --type unigram --special-tokens "<s>" "</s>" "<pad>"
        """
    )
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to training data (file or directory)'
    )
    parser.add_argument(
        '--type',
        type=str,
        default='bpe',
        choices=['bpe', 'wordpiece', 'unigram'],
        help='Tokenizer type (default: bpe)'
    )
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=32000,
        help='Vocabulary size (default: 32000)'
    )
    parser.add_argument(
        '--min-frequency',
        type=int,
        default=2,
        help='Minimum token frequency (default: 2)'
    )
    parser.add_argument(
        '--special-tokens',
        nargs='+',
        default=['<pad>', '<unk>', '<s>', '</s>'],
        help='Special tokens (default: <pad> <unk> <s> </s>)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='tokenizer/tokenizer.json',
        help='Output path for trained tokenizer (default: tokenizer/tokenizer.json)'
    )
    
    args = parser.parse_args()
    
    try:
        # Load text files
        print(f"Loading data from: {args.data}")
        files = load_text_files(args.data)
        
        # Calculate total size
        total_size = sum(Path(f).stat().st_size for f in files)
        print(f"Total data size: {total_size / (1024**2):.2f} MB")
        
        # Train tokenizer based on type
        if args.type == 'bpe':
            tokenizer = train_bpe_tokenizer(
                files, args.vocab_size, args.min_frequency, args.special_tokens
            )
        elif args.type == 'wordpiece':
            tokenizer = train_wordpiece_tokenizer(
                files, args.vocab_size, args.min_frequency, args.special_tokens
            )
        elif args.type == 'unigram':
            tokenizer = train_unigram_tokenizer(
                files, args.vocab_size, args.min_frequency, args.special_tokens
            )
        else:
            raise ValueError(f"Unknown tokenizer type: {args.type}")
        
        # Display statistics
        sample_text = "This is a sample text to test the tokenizer."
        display_statistics(tokenizer, sample_text)
        
        # Save tokenizer
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(output_path))
        
        print(f"✓ Tokenizer saved to: {output_path}")
        print(f"\\nNext steps:")
        print(f"  1. Prepare your data: python data/prepare.py")
        print(f"  2. Start training: python training/train.py")
        
    except Exception as e:
        print(f"\\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
`;
  }
}
