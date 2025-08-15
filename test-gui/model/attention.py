import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from .config import ModelConfig

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Ensure n_embd is divisible by n_head
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        # Linear projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        
        # Scaling factor
        self.scale = self.head_dim ** -0.5
        
        # Sliding window for Mistral
        self.sliding_window = getattr(config, 'sliding_window', None)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, n_embd = x.size()
        
        # Project to Q, K, V
        qkv = self.c_attn(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.n_head, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, n_head, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply sliding window if specified
        if self.sliding_window is not None:
            k, v = self._apply_sliding_window(k, v, seq_len)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Apply softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, n_embd)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        
        return attn_output
    
    def _apply_sliding_window(self, k: torch.Tensor, v: torch.Tensor, seq_len: int) -> tuple:
        """Apply sliding window attention (for Mistral-style models)."""
        if seq_len <= self.sliding_window:
            return k, v
        
        # Keep only the last sliding_window tokens
        k = k[:, :, -self.sliding_window:, :]
        v = v[:, :, -self.sliding_window:, :]
        
        return k, v
