import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from .config import ModelConfig
from .attention import MultiHeadAttention
from .feed_forward import FeedForward

class TransformerBlock(nn.Module):
    """A single transformer block."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.ln2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # Attention and feed forward
        self.attn = MultiHeadAttention(config)
        self.ffn = FeedForward(config)
        
        # Dropout
        self.dropout = nn.Dropout(config.resid_pdrop)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        residual = x
        x = self.ln1(x)
        x = self.attn(x, attention_mask=attention_mask)
        x = self.dropout(x)
        x = residual + x
        
        # Feed forward
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = residual + x
        
        return x

class TransformerLM(nn.Module):
    """GPT-style transformer language model."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        
        # Dropout
        self.drop = nn.Dropout(config.embd_pdrop)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        
        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Tie weights
        self.lm_head.weight = self.wte.weight
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_dict: bool = True
    ) -> dict:
        """
        Forward pass of the transformer.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len]
            return_dict: Whether to return a dict or tuple
            
        Returns:
            Dictionary with logits and optionally hidden states
        """
        batch_size, seq_len = input_ids.size()
        
        # Get position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings
        tok_emb = self.wte(input_ids)
        pos_emb = self.wpe(position_ids)
        
        # Combine embeddings
        x = tok_emb + pos_emb
        x = self.drop(x)
        
        # Create causal attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device))
            attention_mask = attention_mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Pass through transformer blocks
        hidden_states = []
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
            hidden_states.append(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Get logits
        logits = self.lm_head(x)
        
        if return_dict:
            return {
                'logits': logits,
                'hidden_states': hidden_states if self.config.output_hidden_states else None
            }
        else:
            return logits
    
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        pad_token_id: int = 0,
        eos_token_id: int = 50256
    ) -> torch.LongTensor:
        """
        Generate text using the model.
        
        Args:
            input_ids: Starting sequence [batch_size, seq_len]
            max_length: Maximum length to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            do_sample: Whether to use sampling
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            
        Returns:
            Generated sequence [batch_size, new_seq_len]
        """
        self.eval()
        batch_size = input_ids.size(0)
        current_ids = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                # Get model predictions
                outputs = self(current_ids)
                next_token_logits = outputs['logits'][:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to sequence
                current_ids = torch.cat([current_ids, next_token], dim=-1)
                
                # Check for EOS
                if (next_token == eos_token_id).any():
                    break
        
        return current_ids
