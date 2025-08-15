import torch
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
