#!/usr/bin/env python3
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
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters (keep basic punctuation)
    text = re.sub(r'[^\w\s.,!?;:()\[\]{}"'\-]', '', text)
    
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    # Remove multiple periods
    text = re.sub(r'\.{2,}', '.', text)
    
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
            f.write(text + '\n\n')
    
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
            f.write(text + '\n\n')
    
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
            f.write(text + '\n\n')
    
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
            f.write(text + '\n\n')
    
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
            f.write(text + '\n\n')
    
    print(f"Processed {len(train_texts)} training examples")
    print(f"Processed {len(val_texts)} validation examples")
    print(f"Data saved to {output_dir}")

def preprocess_custom(input_dir: str, output_dir: str):
    """Preprocess custom text files."""
    print(f"Processing custom data from {input_dir}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if input is a file or directory
    if os.path.isfile(input_dir):
        # Process single file
        text_files = [input_dir]
    elif os.path.isdir(input_dir):
        # Process directory
        text_files = list(Path(input_dir).rglob("*.txt"))
    else:
        print(f"Input path {input_dir} does not exist")
        return
    
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
        paragraphs = text.split('

')
        
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
            f.write(text + '

')
    
    # Save validation data
    val_path = os.path.join(output_dir, 'validation.txt')
    with open(val_path, 'w', encoding='utf-8') as f:
        for text in val_texts:
            f.write(text + '

')
    
    print(f"Processed {len(train_texts)} training examples")
    print(f"Processed {len(val_texts)} validation examples")
    print(f"Data saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset for LLM training")
    parser.add_argument("--input", "-i", required=True, help="Input directory or dataset name")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--dataset", "-d", choices=["wikitext", "c4", "custom"], 
                       default="wikitext", help="Dataset type")
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
