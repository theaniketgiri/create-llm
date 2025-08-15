import torch
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
