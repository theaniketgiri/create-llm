/**
 * Python dataset templates
 */

export class PythonDatasetTemplates {
  /**
   * Get LLM dataset class
   */
  static getLLMDataset(): string {
    return `"""
PyTorch dataset for LLM training
Handles tokenized data loading and batching
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Optional
import json


class LLMDataset(Dataset):
    """
    Dataset for language model training
    
    Loads pre-tokenized data and creates training examples with:
    - Input IDs
    - Attention masks
    - Labels (shifted input IDs for next-token prediction)
    """
    
    def __init__(
        self,
        data_path: str,
        max_length: int = 512,
        pad_token_id: int = 0
    ):
        """
        Initialize dataset
        
        Args:
            data_path: Path to tokenized data file (.pt or .json)
            max_length: Maximum sequence length
            pad_token_id: Token ID for padding
        """
        self.data_path = Path(data_path)
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Load data
        self.data = self._load_data()
        
        print(f"Loaded {len(self.data)} examples from {data_path}")
    
    def _load_data(self):
        """Load tokenized data from file"""
        if self.data_path.suffix == '.pt':
            # Load PyTorch tensor
            data = torch.load(self.data_path)
            if isinstance(data, torch.Tensor):
                # Check if data is already in correct format (2D tensor)
                if data.dim() == 2:
                    # Data is already [num_examples, seq_len]
                    return data.tolist()
                elif data.dim() == 1:
                    # Data is 1D, need to split into sequences
                    return self._split_into_sequences(data)
                else:
                    raise ValueError(f"Expected 1D or 2D tensor, got {data.dim()}D")
            return data
        elif self.data_path.suffix == '.json':
            # Load JSON
            with open(self.data_path, 'r') as f:
                data = json.load(f)
            return data
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
    
    def _split_into_sequences(self, tensor: torch.Tensor):
        """Split long 1D tensor into sequences of max_length"""
        sequences = []
        for i in range(0, len(tensor) - self.max_length + 1, self.max_length):
            seq = tensor[i:i + self.max_length]
            if len(seq) == self.max_length:
                sequences.append(seq.tolist())
        return sequences
    
    def __len__(self) -> int:
        """Return number of examples in dataset"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training example
        
        Returns:
            Dictionary with:
                - input_ids: Token IDs [seq_len]
                - attention_mask: Attention mask [seq_len]
                - labels: Labels for next-token prediction [seq_len]
        """
        # Get sequence
        sequence = self.data[idx]
        
        # Convert to tensor if needed
        if not isinstance(sequence, torch.Tensor):
            sequence = torch.tensor(sequence, dtype=torch.long)
        
        # Ensure sequence is not longer than max_length
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        
        # Pad if needed
        seq_len = len(sequence)
        if seq_len < self.max_length:
            padding = torch.full(
                (self.max_length - seq_len,),
                self.pad_token_id,
                dtype=torch.long
            )
            sequence = torch.cat([sequence, padding])
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.ones(self.max_length, dtype=torch.long)
        if seq_len < self.max_length:
            attention_mask[seq_len:] = 0
        
        # Create labels (shifted input_ids for next-token prediction)
        # Labels are the same as input_ids, model will shift internally
        # Set padding tokens to -100 (ignored in loss calculation)
        labels = sequence.clone()
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': sequence,
            'attention_mask': attention_mask,
            'labels': labels
        }


class StreamingLLMDataset(Dataset):
    """
    Memory-efficient streaming dataset for very large datasets
    Loads data on-the-fly instead of keeping everything in memory
    """
    
    def __init__(
        self,
        data_path: str,
        max_length: int = 512,
        pad_token_id: int = 0,
        stride: int = 256
    ):
        """
        Initialize streaming dataset
        
        Args:
            data_path: Path to tokenized data file
            max_length: Maximum sequence length
            pad_token_id: Token ID for padding
            stride: Stride for sliding window
        """
        self.data_path = Path(data_path)
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.stride = stride
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Load full data once to calculate length
        self.full_data = torch.load(self.data_path)
        
        # Calculate number of examples
        self.num_examples = max(1, (len(self.full_data) - max_length) // stride + 1)
        
        print(f"Streaming dataset with {self.num_examples} examples")
    
    def __len__(self) -> int:
        return self.num_examples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get example using sliding window"""
        start_idx = idx * self.stride
        end_idx = start_idx + self.max_length
        
        # Get sequence
        sequence = self.full_data[start_idx:end_idx]
        
        # Pad if needed
        seq_len = len(sequence)
        if seq_len < self.max_length:
            padding = torch.full(
                (self.max_length - seq_len,),
                self.pad_token_id,
                dtype=torch.long
            )
            sequence = torch.cat([sequence, padding])
        
        # Create attention mask
        attention_mask = torch.ones(self.max_length, dtype=torch.long)
        if seq_len < self.max_length:
            attention_mask[seq_len:] = 0
        
        # Create labels
        labels = sequence.clone()
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': sequence,
            'attention_mask': attention_mask,
            'labels': labels
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    max_length: int = None
):
    """
    Create DataLoader with optimal settings
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer
        max_length: Maximum sequence length (truncate if exceeded)
    
    Returns:
        DataLoader instance
    """
    from torch.utils.data import DataLoader
    
    def collate_fn(batch):
        """Custom collate function with optional max_length truncation"""
        # Stack tensors from batch
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        # Apply max_length constraint if specified
        if max_length is not None:
            input_ids = input_ids[:, :max_length]
            attention_mask = attention_mask[:, :max_length]
            labels = labels[:, :max_length]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True,  # Drop incomplete batches
        collate_fn=collate_fn if max_length is not None else None
    )


if __name__ == '__main__':
    # Test dataset
    print("Testing LLMDataset...")
    
    # Create dummy data
    dummy_data = torch.randint(0, 1000, (10000,))
    torch.save(dummy_data, 'test_data.pt')
    
    # Create dataset
    dataset = LLMDataset('test_data.pt', max_length=512)
    
    print(f"Dataset length: {len(dataset)}")
    
    # Get sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Attention mask shape: {sample['attention_mask'].shape}")
    print(f"Labels shape: {sample['labels'].shape}")
    
    # Test dataloader
    dataloader = create_dataloader(dataset, batch_size=4)
    batch = next(iter(dataloader))
    print(f"\\nBatch input_ids shape: {batch['input_ids'].shape}")
    print(f"Batch attention_mask shape: {batch['attention_mask'].shape}")
    print(f"Batch labels shape: {batch['labels'].shape}")
    
    # Clean up
    import os
    os.remove('test_data.pt')
    
    print("\\n✓ Dataset tests passed!")
`;
  }

  /**
   * Get data __init__.py
   */
  static getDataInit(): string {
    return `"""
Data package
"""

from .dataset import LLMDataset, StreamingLLMDataset, create_dataloader

__all__ = [
    'LLMDataset',
    'StreamingLLMDataset',
    'create_dataloader',
]
`;
  }
}
