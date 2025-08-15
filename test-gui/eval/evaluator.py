import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional
import math

class Evaluator:
    """Evaluator class for LLM evaluation."""
    
    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model
            device: Device to evaluate on
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    def compute_perplexity(self, dataloader: DataLoader) -> float:
        """
        Compute perplexity on dataset.
        
        Args:
            dataloader: Data loader for evaluation
            
        Returns:
            Perplexity score
        """
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs['logits']
                
                # Shift for next token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # Compute loss
                loss_fct = nn.CrossEntropyLoss(reduction='sum')
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                # Count non-padding tokens
                num_tokens = (shift_labels != -100).sum().item()
                
                total_loss += loss.item()
                total_tokens += num_tokens
        
        # Compute perplexity
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        return perplexity
    
    def compute_accuracy(self, dataloader: DataLoader) -> float:
        """
        Compute accuracy on dataset.
        
        Args:
            dataloader: Data loader for evaluation
            
        Returns:
            Accuracy score
        """
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs['logits']
                
                # Shift for next token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # Get predictions
                predictions = torch.argmax(shift_logits, dim=-1)
                
                # Count correct predictions (ignore padding)
                mask = (shift_labels != -100)
                correct = (predictions == shift_labels) & mask
                
                correct_predictions += correct.sum().item()
                total_predictions += mask.sum().item()
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        return accuracy
    
    def compute_loss(self, dataloader: DataLoader) -> float:
        """
        Compute average loss on dataset.
        
        Args:
            dataloader: Data loader for evaluation
            
        Returns:
            Average loss
        """
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs['logits']
                
                # Shift for next token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # Compute loss
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def compute_all_metrics(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Compute all evaluation metrics.
        
        Args:
            dataloader: Data loader for evaluation
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Compute basic metrics
        metrics['loss'] = self.compute_loss(dataloader)
        metrics['perplexity'] = self.compute_perplexity(dataloader)
        metrics['accuracy'] = self.compute_accuracy(dataloader)
        
        return metrics
    
    def generate_samples(
        self,
        tokenizer,
        prompts: List[str],
        max_length: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        num_samples: int = 1
    ) -> List[str]:
        """
        Generate text samples from prompts.
        
        Args:
            tokenizer: Tokenizer instance
            prompts: List of input prompts
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            num_samples: Number of samples per prompt
            
        Returns:
            List of generated texts
        """
        generated_texts = []
        
        for prompt in prompts:
            for _ in range(num_samples):
                # Encode prompt
                input_ids = tokenizer.encode(prompt, add_special_tokens=True)
                input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
                
                # Generate
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        input_ids=input_ids,
                        max_length=max_length,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                
                # Decode
                generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                generated_texts.append(generated_text)
        
        return generated_texts
