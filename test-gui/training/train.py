#!/usr/bin/env python3
"""
Main training script for the LLM.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --resume checkpoints/latest.pt
"""

import argparse
import os
import yaml
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import logging
from pathlib import Path

from model import TransformerLM, ModelConfig
from tokenizer import CustomTokenizer
from data import create_dataloaders
from training.trainer import Trainer
from training.config import TrainingConfig

def setup_logging(log_dir: str):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )

def setup_distributed():
    """Setup distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        return rank, world_size
    return 0, 1

def main():
    parser = argparse.ArgumentParser(description="Train LLM")
    parser.add_argument("--config", "-c", required=True, help="Path to config file")
    parser.add_argument("--resume", "-r", help="Path to checkpoint to resume from")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = TrainingConfig(**config_dict)
    
    # Setup distributed training
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        rank, world_size = setup_distributed()
    else:
        rank, world_size = 0, 1
    
    # Setup logging
    if rank == 0:
        setup_logging(config.log_dir)
        logging.info(f"Starting training with config: {config}")
    
    # Load tokenizer
    tokenizer = CustomTokenizer(config.tokenizer_path)
    
    # Create model
    model_config = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(),
        n_positions=config.max_length,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
        model_type="gpt"
    )
    
    model = TransformerLM(model_config)
    
    # Move to GPU
    device = torch.device(f"cuda:{args.local_rank}" if args.local_rank != -1 else "cuda:0")
    model = model.to(device)
    
    # Setup distributed training
    if world_size > 1:
        model = DDP(model, device_ids=[args.local_rank])
    
    # Create dataloaders
    train_dataloader, val_dataloader = create_dataloaders(
        tokenizer=tokenizer,
        train_path=config.train_data_path,
        val_path=config.val_data_path,
        batch_size=config.batch_size,
        max_length=config.max_length,
        num_workers=config.num_workers,
        streaming=config.streaming
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config,
        device=device,
        rank=rank
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
