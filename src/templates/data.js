const fs = require('fs-extra');
const path = require('path');

async function createDataFiles(projectPath, options) {
  const dataPath = path.join(projectPath, 'data');
  
  // Create __init__.py
  await fs.writeFile(path.join(dataPath, '__init__.py'), `from .dataset import TextDataset, DataLoader
from .preprocessing import preprocess_text

__all__ = ['TextDataset', 'DataLoader', 'preprocess_text']
`);

  // Create dataset class
  await createDatasetClass(dataPath, options);
  
  // Create preprocessing script
  await createPreprocessingScript(dataPath, options);
  
  // Create data loader
  await createDataLoader(dataPath, options);
  
  // Create dataset downloader
  await createDatasetDownloader(dataPath, options);
}

async function createDatasetClass(dataPath, options) {
  const datasetContent = `import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple
import json
import os
from pathlib import Path

from tokenizer import CustomTokenizer

class TextDataset(Dataset):
    """Dataset for text data with tokenization."""
    
    def __init__(
        self,
        tokenizer: CustomTokenizer,
        data_path: str,
        max_length: int = 1024,
        stride: int = 512,
        add_special_tokens: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            tokenizer: Custom tokenizer instance
            data_path: Path to text file or directory
            max_length: Maximum sequence length
            stride: Stride for sliding window
            add_special_tokens: Whether to add special tokens
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.add_special_tokens = add_special_tokens
        
        # Load and tokenize data
        self.data = self._load_data(data_path)
        self.examples = self._create_examples()
    
    def _load_data(self, data_path: str) -> List[str]:
        """Load text data from file or directory."""
        if os.path.isfile(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                return [f.read()]
        elif os.path.isdir(data_path):
            texts = []
            for file_path in Path(data_path).rglob("*.txt"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    texts.append(f.read())
            return texts
        else:
            raise ValueError(f"Data path {data_path} does not exist")
    
    def _create_examples(self) -> List[Dict[str, torch.Tensor]]:
        """Create training examples with sliding window."""
        examples = []
        
        for text in self.data:
            # Tokenize the text
            token_ids = self.tokenizer.encode(text, add_special_tokens=self.add_special_tokens)
            
            # Create sliding window examples
            for i in range(0, len(token_ids) - self.max_length + 1, self.stride):
                input_ids = token_ids[i:i + self.max_length]
                
                # Create labels (shifted by 1 for next token prediction)
                labels = token_ids[i + 1:i + self.max_length + 1]
                
                # Pad if necessary
                if len(input_ids) < self.max_length:
                    pad_length = self.max_length - len(input_ids)
                    input_ids.extend([self.tokenizer.pad_token_id] * pad_length)
                    labels.extend([-100] * pad_length)  # -100 is ignored in loss
                
                if len(labels) < self.max_length:
                    pad_length = self.max_length - len(labels)
                    labels.extend([-100] * pad_length)
                
                examples.append({
                    'input_ids': torch.tensor(input_ids, dtype=torch.long),
                    'labels': torch.tensor(labels, dtype=torch.long),
                    'attention_mask': torch.ones(self.max_length, dtype=torch.long)
                })
        
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]

class StreamingTextDataset(Dataset):
    """Streaming dataset for large files that don't fit in memory."""
    
    def __init__(
        self,
        tokenizer: CustomTokenizer,
        data_path: str,
        max_length: int = 1024,
        buffer_size: int = 10000
    ):
        """
        Initialize streaming dataset.
        
        Args:
            tokenizer: Custom tokenizer instance
            data_path: Path to text file
            max_length: Maximum sequence length
            buffer_size: Number of examples to keep in memory
        """
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.max_length = max_length
        self.buffer_size = buffer_size
        
        # Get file size for length estimation
        self.file_size = os.path.getsize(data_path)
        
        # Initialize buffer
        self.buffer = []
        self.buffer_start = 0
        
        # Load initial buffer
        self._load_buffer()
    
    def _load_buffer(self):
        """Load examples into buffer."""
        self.buffer = []
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            f.seek(self.buffer_start)
            
            # Read chunk of text
            chunk = f.read(self.buffer_size * self.max_length * 4)  # Rough estimate
            
            if not chunk:
                return
            
            # Tokenize chunk
            token_ids = self.tokenizer.encode(chunk, add_special_tokens=True)
            
            # Create examples
            for i in range(0, len(token_ids) - self.max_length + 1, self.max_length // 2):
                input_ids = token_ids[i:i + self.max_length]
                labels = token_ids[i + 1:i + self.max_length + 1]
                
                if len(input_ids) < self.max_length:
                    pad_length = self.max_length - len(input_ids)
                    input_ids.extend([self.tokenizer.pad_token_id] * pad_length)
                    labels.extend([-100] * pad_length)
                
                if len(labels) < self.max_length:
                    pad_length = self.max_length - len(labels)
                    labels.extend([-100] * pad_length)
                
                self.buffer.append({
                    'input_ids': torch.tensor(input_ids, dtype=torch.long),
                    'labels': torch.tensor(labels, dtype=torch.long),
                    'attention_mask': torch.ones(self.max_length, dtype=torch.long)
                })
                
                if len(self.buffer) >= self.buffer_size:
                    break
            
            # Update buffer start position
            self.buffer_start = f.tell()
    
    def __len__(self) -> int:
        # Estimate total length based on file size
        return self.file_size // (self.max_length * 4)  # Rough estimate
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # If we've exhausted the buffer, load more
        if idx >= len(self.buffer):
            self._load_buffer()
            
            # If still no data, we've reached the end
            if not self.buffer:
                raise IndexError("Dataset exhausted")
        
        return self.buffer[idx % len(self.buffer)]
`;
  
  await fs.writeFile(path.join(dataPath, 'dataset.py'), datasetContent);
}

async function createPreprocessingScript(dataPath, options) {
  const preprocessingContent = `#!/usr/bin/env python3
"""
Data preprocessing script for LLM training.

Usage:
    python prepare_dataset.py --input data/raw/ --output data/processed/
    python prepare_dataset.py --input data/raw/ --output data/processed/ --dataset wikitext
"""

import argparse
import os
import json
import re
from pathlib import Path
from typing import List, Dict, Optional
import datasets
from datasets import load_dataset

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove extra whitespace
    text = re.sub(r'\\s+', ' ', text)
    
    # Remove special characters (keep basic punctuation)
    text = re.sub(r'[^\\w\\s.,!?;:()\\[\\]{}"\'-]', '', text)
    
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    # Remove multiple periods
    text = re.sub(r'\\.{2,}', '.', text)
    
    return text.strip()

def preprocess_wikitext(data_dir: str, output_dir: str):
    """Preprocess WikiText dataset."""
    print("Loading WikiText-103 dataset...")
    
    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-103-v1")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process train split
    train_texts = []
    for example in dataset['train']:
        if example['text'].strip():
            cleaned_text = clean_text(example['text'])
            if cleaned_text:
                train_texts.append(cleaned_text)
    
    # Save train data
    train_path = os.path.join(output_dir, 'train.txt')
    with open(train_path, 'w', encoding='utf-8') as f:
        for text in train_texts:
            f.write(text + '\\n\\n')
    
    # Process validation split
    val_texts = []
    for example in dataset['validation']:
        if example['text'].strip():
            cleaned_text = clean_text(example['text'])
            if cleaned_text:
                val_texts.append(cleaned_text)
    
    # Save validation data
    val_path = os.path.join(output_dir, 'validation.txt')
    with open(val_path, 'w', encoding='utf-8') as f:
        for text in val_texts:
            f.write(text + '\\n\\n')
    
    # Process test split
    test_texts = []
    for example in dataset['test']:
        if example['text'].strip():
            cleaned_text = clean_text(example['text'])
            if cleaned_text:
                test_texts.append(cleaned_text)
    
    # Save test data
    test_path = os.path.join(output_dir, 'test.txt')
    with open(test_path, 'w', encoding='utf-8') as f:
        for text in test_texts:
            f.write(text + '\\n\\n')
    
    print(f"Processed {len(train_texts)} training examples")
    print(f"Processed {len(val_texts)} validation examples")
    print(f"Processed {len(test_texts)} test examples")
    print(f"Data saved to {output_dir}")

def preprocess_c4(data_dir: str, output_dir: str, max_examples: int = 100000):
    """Preprocess C4 dataset."""
    print("Loading C4 dataset...")
    
    # Load dataset (subset for memory efficiency)
    dataset = load_dataset("c4", "en", streaming=True)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process train split
    train_texts = []
    for i, example in enumerate(dataset['train']):
        if i >= max_examples:
            break
        
        text = example['text']
        if text.strip():
            cleaned_text = clean_text(text)
            if cleaned_text and len(cleaned_text) > 100:  # Filter short texts
                train_texts.append(cleaned_text)
    
    # Save train data
    train_path = os.path.join(output_dir, 'train.txt')
    with open(train_path, 'w', encoding='utf-8') as f:
        for text in train_texts:
            f.write(text + '\\n\\n')
    
    # Process validation split
    val_texts = []
    for i, example in enumerate(dataset['validation']):
        if i >= max_examples // 10:  # Smaller validation set
            break
        
        text = example['text']
        if text.strip():
            cleaned_text = clean_text(text)
            if cleaned_text and len(cleaned_text) > 100:
                val_texts.append(cleaned_text)
    
    # Save validation data
    val_path = os.path.join(output_dir, 'validation.txt')
    with open(val_path, 'w', encoding='utf-8') as f:
        for text in val_texts:
            f.write(text + '\\n\\n')
    
    print(f"Processed {len(train_texts)} training examples")
    print(f"Processed {len(val_texts)} validation examples")
    print(f"Data saved to {output_dir}")

def preprocess_custom(input_dir: str, output_dir: str):
    """Preprocess custom text files."""
    print(f"Processing custom data from {input_dir}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all text files
    text_files = list(Path(input_dir).rglob("*.txt"))
    
    if not text_files:
        print(f"No .txt files found in {input_dir}")
        return
    
    # Process all files
    all_texts = []
    for file_path in text_files:
        print(f"Processing {file_path}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split into paragraphs
        paragraphs = text.split('\\n\\n')
        
        for paragraph in paragraphs:
            if paragraph.strip():
                cleaned_text = clean_text(paragraph)
                if cleaned_text and len(cleaned_text) > 50:  # Filter very short texts
                    all_texts.append(cleaned_text)
    
    # Split into train/validation
    split_idx = int(len(all_texts) * 0.9)
    train_texts = all_texts[:split_idx]
    val_texts = all_texts[split_idx:]
    
    # Save train data
    train_path = os.path.join(output_dir, 'train.txt')
    with open(train_path, 'w', encoding='utf-8') as f:
        for text in train_texts:
            f.write(text + '\\n\\n')
    
    # Save validation data
    val_path = os.path.join(output_dir, 'validation.txt')
    with open(val_path, 'w', encoding='utf-8') as f:
        for text in val_texts:
            f.write(text + '\\n\\n')
    
    print(f"Processed {len(train_texts)} training examples")
    print(f"Processed {len(val_texts)} validation examples")
    print(f"Data saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset for LLM training")
    parser.add_argument("--input", "-i", required=True, help="Input directory or dataset name")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--dataset", "-d", choices=["wikitext", "c4", "custom"], 
                       default="${options.dataset}", help="Dataset type")
    parser.add_argument("--max-examples", "-m", type=int, default=100000, 
                       help="Maximum examples for C4 dataset")
    
    args = parser.parse_args()
    
    try:
        if args.dataset == "wikitext":
            preprocess_wikitext(args.input, args.output)
        elif args.dataset == "c4":
            preprocess_c4(args.input, args.output, args.max_examples)
        elif args.dataset == "custom":
            preprocess_custom(args.input, args.output)
        
        print("Preprocessing completed successfully!")
        return 0
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
`;
  
  await fs.writeFile(path.join(dataPath, 'prepare_dataset.py'), preprocessingContent);
}

async function createDataLoader(dataPath, options) {
  const dataLoaderContent = `import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from typing import Optional, Tuple

from .dataset import TextDataset, StreamingTextDataset
from tokenizer import CustomTokenizer

def create_dataloaders(
    tokenizer: CustomTokenizer,
    train_path: str,
    val_path: str,
    batch_size: int = 8,
    max_length: int = 1024,
    num_workers: int = 4,
    streaming: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.
    
    Args:
        tokenizer: Custom tokenizer instance
        train_path: Path to training data
        val_path: Path to validation data
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of workers for data loading
        streaming: Whether to use streaming dataset
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    
    # Create datasets
    if streaming:
        train_dataset = StreamingTextDataset(
            tokenizer=tokenizer,
            data_path=train_path,
            max_length=max_length
        )
        val_dataset = StreamingTextDataset(
            tokenizer=tokenizer,
            data_path=val_path,
            max_length=max_length
        )
    else:
        train_dataset = TextDataset(
            tokenizer=tokenizer,
            data_path=train_path,
            max_length=max_length
        )
        val_dataset = TextDataset(
            tokenizer=tokenizer,
            data_path=val_path,
            max_length=max_length
        )
    
    # Create samplers
    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_dataloader, val_dataloader

def create_test_dataloader(
    tokenizer: CustomTokenizer,
    test_path: str,
    batch_size: int = 8,
    max_length: int = 1024,
    num_workers: int = 4
) -> DataLoader:
    """
    Create test dataloader.
    
    Args:
        tokenizer: Custom tokenizer instance
        test_path: Path to test data
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of workers for data loading
        
    Returns:
        Test dataloader
    """
    
    test_dataset = TextDataset(
        tokenizer=tokenizer,
        data_path=test_path,
        max_length=max_length
    )
    
    test_sampler = SequentialSampler(test_dataset)
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return test_dataloader
`;
  
  await fs.writeFile(path.join(dataPath, 'dataloader.py'), dataLoaderContent);
}

async function createDatasetDownloader(dataPath, options) {
  const downloaderContent = `#!/usr/bin/env python3
"""
Download and prepare datasets for LLM training.
"""

import argparse
import os
import requests
import zipfile
import tarfile
from pathlib import Path
from typing import Optional

def download_file(url: str, output_path: str):
    """Download a file from URL."""
    print(f"Downloading {url} to {output_path}...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                
                # Print progress
                if total_size > 0:
                    progress = (downloaded / total_size) * 100
                    print(f"\\rProgress: {progress:.1f}%", end='', flush=True)
    
    print()  # New line after progress

def extract_archive(archive_path: str, extract_dir: str):
    """Extract archive file."""
    print(f"Extracting {archive_path} to {extract_dir}...")
    
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_dir)
    elif archive_path.endswith('.tar'):
        with tarfile.open(archive_path, 'r') as tar_ref:
            tar_ref.extractall(extract_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")

def download_wikitext(output_dir: str):
    """Download WikiText-103 dataset."""
    print("Downloading WikiText-103 dataset...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Download URLs
    urls = {
        'train': 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip',
    }
    
    for split, url in urls.items():
        output_path = os.path.join(output_dir, f'{split}.zip')
        
        # Download if not exists
        if not os.path.exists(output_path):
            download_file(url, output_path)
        
        # Extract
        extract_dir = os.path.join(output_dir, split)
        if not os.path.exists(extract_dir):
            extract_archive(output_path, output_dir)
    
    print("WikiText-103 download completed!")

def download_openwebtext(output_dir: str):
    """Download OpenWebText dataset (subset)."""
    print("Downloading OpenWebText dataset...")
    
    # Note: OpenWebText is very large, so we'll provide instructions
    print("OpenWebText is a large dataset (~40GB).")
    print("Please download it manually from:")
    print("https://github.com/jcpeterson/openwebtext")
    print(f"Then extract it to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Download datasets for LLM training")
    parser.add_argument("--dataset", "-d", choices=["wikitext", "openwebtext"], 
                       default="${options.dataset}", help="Dataset to download")
    parser.add_argument("--output", "-o", default="data/raw", help="Output directory")
    
    args = parser.parse_args()
    
    try:
        if args.dataset == "wikitext":
            download_wikitext(args.output)
        elif args.dataset == "openwebtext":
            download_openwebtext(args.output)
        
        print("Download completed successfully!")
        return 0
    except Exception as e:
        print(f"Error during download: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
`;
  
  await fs.writeFile(path.join(dataPath, 'download_dataset.py'), downloaderContent);
}

module.exports = { createDataFiles }; 