import torch
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
