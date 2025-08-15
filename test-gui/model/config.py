import dataclasses
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
    model_type: str = "gpt"
    
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
