import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple
import math

def compute_perplexity(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> float:
    """
    Compute perplexity from logits and labels.
    
    Args:
        logits: Model output logits [batch_size, seq_len, vocab_size]
        labels: Target labels [batch_size, seq_len]
        ignore_index: Index to ignore in loss computation
        
    Returns:
        Perplexity score
    """
    # Shift for next token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Compute loss
    loss_fct = nn.CrossEntropyLoss(reduction='sum', ignore_index=ignore_index)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    # Count non-ignored tokens
    num_tokens = (shift_labels != ignore_index).sum().item()
    
    # Compute perplexity
    avg_loss = loss.item() / num_tokens if num_tokens > 0 else 0.0
    perplexity = math.exp(avg_loss)
    
    return perplexity

def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> float:
    """
    Compute accuracy from logits and labels.
    
    Args:
        logits: Model output logits [batch_size, seq_len, vocab_size]
        labels: Target labels [batch_size, seq_len]
        ignore_index: Index to ignore in accuracy computation
        
    Returns:
        Accuracy score
    """
    # Shift for next token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Get predictions
    predictions = torch.argmax(shift_logits, dim=-1)
    
    # Count correct predictions (ignore specified index)
    mask = (shift_labels != ignore_index)
    correct = (predictions == shift_labels) & mask
    
    accuracy = correct.sum().item() / mask.sum().item() if mask.sum().item() > 0 else 0.0
    return accuracy

def compute_bleu_score(predictions: List[str], references: List[str]) -> float:
    """
    Compute BLEU score for text generation.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        
    Returns:
        BLEU score
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from nltk.tokenize import word_tokenize
    except ImportError:
        print("NLTK not available. Install with: pip install nltk")
        return 0.0
    
    smoothie = SmoothingFunction().method1
    
    total_bleu = 0.0
    for pred, ref in zip(predictions, references):
        pred_tokens = word_tokenize(pred.lower())
        ref_tokens = word_tokenize(ref.lower())
        
        # Compute BLEU-4
        bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
        total_bleu += bleu
    
    avg_bleu = total_bleu / len(predictions) if predictions else 0.0
    return avg_bleu

def compute_rouge_score(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute ROUGE scores for text generation.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        
    Returns:
        Dictionary of ROUGE scores
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("rouge-score not available. Install with: pip install rouge-score")
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    total_scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for key in total_scores:
            total_scores[key] += scores[key].fmeasure
    
    # Average scores
    num_samples = len(predictions)
    avg_scores = {key: score / num_samples for key, score in total_scores.items()}
    
    return avg_scores

def compute_diversity_metrics(generated_texts: List[str]) -> Dict[str, float]:
    """
    Compute diversity metrics for generated texts.
    
    Args:
        generated_texts: List of generated texts
        
    Returns:
        Dictionary of diversity metrics
    """
    if not generated_texts:
        return {'distinct_1': 0.0, 'distinct_2': 0.0, 'entropy': 0.0}
    
    # Tokenize all texts
    all_tokens = []
    for text in generated_texts:
        tokens = text.lower().split()
        all_tokens.extend(tokens)
    
    if not all_tokens:
        return {'distinct_1': 0.0, 'distinct_2': 0.0, 'entropy': 0.0}
    
    # Compute distinct-1 (unique unigrams)
    unique_unigrams = set(all_tokens)
    distinct_1 = len(unique_unigrams) / len(all_tokens)
    
    # Compute distinct-2 (unique bigrams)
    bigrams = []
    for i in range(len(all_tokens) - 1):
        bigrams.append((all_tokens[i], all_tokens[i + 1]))
    
    unique_bigrams = set(bigrams)
    distinct_2 = len(unique_bigrams) / len(bigrams) if bigrams else 0.0
    
    # Compute entropy
    token_counts = {}
    for token in all_tokens:
        token_counts[token] = token_counts.get(token, 0) + 1
    
    total_tokens = len(all_tokens)
    entropy = 0.0
    for count in token_counts.values():
        prob = count / total_tokens
        entropy -= prob * math.log2(prob)
    
    return {
        'distinct_1': distinct_1,
        'distinct_2': distinct_2,
        'entropy': entropy
    }

def compute_fluency_score(generated_texts: List[str]) -> float:
    """
    Compute fluency score using a simple heuristic.
    
    Args:
        generated_texts: List of generated texts
        
    Returns:
        Fluency score (0-1)
    """
    if not generated_texts:
        return 0.0
    
    total_score = 0.0
    
    for text in generated_texts:
        # Simple fluency heuristics
        score = 0.0
        
        # Check for reasonable sentence length
        sentences = text.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        if 5 <= avg_sentence_length <= 30:
            score += 0.3
        
        # Check for proper capitalization
        if text and text[0].isupper():
            score += 0.2
        
        # Check for reasonable word length
        words = text.split()
        avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
        if 3 <= avg_word_length <= 8:
            score += 0.2
        
        # Check for punctuation
        if any(p in text for p in ['.', '!', '?', ',']):
            score += 0.2
        
        # Check for no excessive repetition
        if len(set(words)) / len(words) > 0.7 if words else True:
            score += 0.1
        
        total_score += score
    
    avg_fluency = total_score / len(generated_texts)
    return avg_fluency
