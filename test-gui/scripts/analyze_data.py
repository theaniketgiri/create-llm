#!/usr/bin/env python3
"""
Analyze training data statistics.
"""

import argparse
import os
import re
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

def analyze_text_file(file_path: str):
    """Analyze a single text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Basic statistics
    lines = text.split('\n')
    paragraphs = text.split('\n\n')
    sentences = re.split(r'[.!?]+', text)
    words = re.findall(r'\b\w+\b', text.lower())
    characters = len(text)
    
    # Word frequency
    word_freq = Counter(words)
    most_common = word_freq.most_common(10)
    
    # Average lengths
    avg_sentence_length = sum(len(s.split()) for s in sentences if s.strip()) / len([s for s in sentences if s.strip()])
    avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
    
    return {
        'file': file_path,
        'characters': characters,
        'lines': len(lines),
        'paragraphs': len(paragraphs),
        'sentences': len([s for s in sentences if s.strip()]),
        'words': len(words),
        'unique_words': len(word_freq),
        'avg_sentence_length': avg_sentence_length,
        'avg_word_length': avg_word_length,
        'most_common_words': most_common
    }

def main():
    parser = argparse.ArgumentParser(description="Analyze training data")
    parser.add_argument("--input", "-i", required=True, help="Input file or directory")
    parser.add_argument("--output", "-o", help="Output directory for plots")
    
    args = parser.parse_args()
    
    if os.path.isfile(args.input):
        # Analyze single file
        results = [analyze_text_file(args.input)]
    else:
        # Analyze directory
        results = []
        for file_path in Path(args.input).rglob("*.txt"):
            results.append(analyze_text_file(str(file_path)))
    
    # Print summary
    print("Data Analysis Summary:")
    print("=" * 50)
    
    total_chars = sum(r['characters'] for r in results)
    total_words = sum(r['words'] for r in results)
    total_sentences = sum(r['sentences'] for r in results)
    
    print(f"Total files analyzed: {len(results)}")
    print(f"Total characters: {total_chars:,}")
    print(f"Total words: {total_words:,}")
    print(f"Total sentences: {total_sentences:,}")
    print(f"Average words per sentence: {total_words / total_sentences:.2f}")
    
    # Show per-file statistics
    for result in results:
        print(f"\nFile: {result['file']}")
        print(f"  Characters: {result['characters']:,}")
        print(f"  Words: {result['words']:,}")
        print(f"  Sentences: {result['sentences']:,}")
        print(f"  Unique words: {result['unique_words']:,}")
        print(f"  Avg sentence length: {result['avg_sentence_length']:.2f}")
        print(f"  Avg word length: {result['avg_word_length']:.2f}")
    
    # Generate plots if output directory specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        
        # Word frequency plot
        all_words = []
        for result in results:
            all_words.extend([word for word, _ in result['most_common_words']])
        
        word_freq = Counter(all_words)
        top_words = word_freq.most_common(20)
        
        plt.figure(figsize=(12, 6))
        words, counts = zip(*top_words)
        plt.bar(range(len(words)), counts)
        plt.xticks(range(len(words)), words, rotation=45)
        plt.title('Most Common Words')
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output, 'word_frequency.png'))
        plt.close()
        
        print(f"\nPlots saved to: {args.output}")

if __name__ == "__main__":
    main()
