const fs = require('fs-extra');
const path = require('path');

async function createModelFiles(projectPath, options) {
  const modelPath = path.join(projectPath, 'model');
  
  // Create __init__.py
  await fs.writeFile(path.join(modelPath, '__init__.py'), `from .transformer import TransformerLM
from .config import ModelConfig

__all__ = ['TransformerLM', 'ModelConfig']
`);

  // Create model config
  await createModelConfig(modelPath, options);
  
  // Create transformer architecture
  await createTransformer(modelPath, options);
  
  // Create attention mechanism
  await createAttention(modelPath, options);
  
  // Create feed forward
  await createFeedForward(modelPath, options);
}

async function createModelConfig(modelPath, options) {
  const configContent = `import dataclasses
from typing import Optional

@dataclasses.dataclass
class ModelConfig:
    """Configuration for the transformer model."""
    
    # Model dimensions
    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    
    # Dropout
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    
    # Layer norm
    layer_norm_epsilon: float = 1e-5
    
    # Initialization
    initializer_range: float = 0.02
    
    # Model type specific
    model_type: str = "${options.template}"
    
    # For Mistral/Mixtral
    sliding_window: Optional[int] = None
    num_experts: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    
    # For RWKV
    rwkv_emb_scale: Optional[float] = None
    
    def __post_init__(self):
        if self.model_type == "mistral":
            self.sliding_window = 4096
        elif self.model_type == "mixtral":
            self.sliding_window = 4096
            self.num_experts = 8
            self.num_experts_per_tok = 2
        elif self.model_type == "rwkv":
            self.rwkv_emb_scale = 0.1
`;
  
  await fs.writeFile(path.join(modelPath, 'config.py'), configContent);
}

async function createTransformer(modelPath, options) {
  const transformerContent = `import torch
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
`;
  
  await fs.writeFile(path.join(modelPath, 'transformer.py'), transformerContent);
}

async function createAttention(modelPath, options) {
  const attentionContent = `import torch
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
`;
  
  await fs.writeFile(path.join(modelPath, 'attention.py'), attentionContent);
}

async function createFeedForward(modelPath, options) {
  const feedForwardContent = `import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig

class FeedForward(nn.Module):
    """Feed-forward network with optional MoE support."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Standard feed-forward
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(config.resid_pdrop)
        
        # MoE support for Mixtral
        self.num_experts = getattr(config, 'num_experts', None)
        self.num_experts_per_tok = getattr(config, 'num_experts_per_tok', None)
        
        if self.num_experts is not None:
            self._setup_moe()
    
    def _setup_moe(self):
        """Setup mixture of experts."""
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.config.n_embd, 4 * self.config.n_embd, bias=False),
                nn.GELU(),
                nn.Linear(4 * self.config.n_embd, self.config.n_embd, bias=False)
            ) for _ in range(self.num_experts)
        ])
        
        self.gate = nn.Linear(self.config.n_embd, self.num_experts, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_experts is not None:
            return self._forward_moe(x)
        else:
            return self._forward_standard(x)
    
    def _forward_standard(self, x: torch.Tensor) -> torch.Tensor:
        """Standard feed-forward forward pass."""
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
    def _forward_moe(self, x: torch.Tensor) -> torch.Tensor:
        """Mixture of experts forward pass."""
        batch_size, seq_len, n_embd = x.size()
        
        # Get expert weights
        gate_logits = self.gate(x)  # [batch_size, seq_len, num_experts]
        
        # Select top-k experts per token
        top_k_weights, top_k_indices = torch.topk(
            gate_logits, self.num_experts_per_tok, dim=-1
        )
        top_k_weights = F.softmax(top_k_weights, dim=-1)
        
        # Reshape for expert computation
        x_flat = x.view(-1, n_embd)  # [batch_size * seq_len, n_embd]
        top_k_indices_flat = top_k_indices.view(-1, self.num_experts_per_tok)
        top_k_weights_flat = top_k_weights.view(-1, self.num_experts_per_tok)
        
        # Compute expert outputs
        expert_outputs = []
        for i in range(self.num_experts_per_tok):
            expert_idx = top_k_indices_flat[:, i]
            expert_weight = top_k_weights_flat[:, i:i+1]
            
            # Get expert output
            expert_output = torch.zeros_like(x_flat)
            for expert_id in range(self.num_experts):
                mask = (expert_idx == expert_id)
                if mask.any():
                    expert_output[mask] = self.experts[expert_id](x_flat[mask])
            
            expert_outputs.append(expert_output * expert_weight)
        
        # Combine expert outputs
        output = sum(expert_outputs)
        output = output.view(batch_size, seq_len, n_embd)
        output = self.dropout(output)
        
        return output
`;
  
  await fs.writeFile(path.join(modelPath, 'feed_forward.py'), feedForwardContent);
}

module.exports = { createModelFiles }; 