import json
import os
from typing import List, Dict, Optional
from tokenizers import Tokenizer as HFTokenizer

class CustomTokenizer:
    """Custom tokenizer wrapper for easy use in training."""
    
    def __init__(self, tokenizer_path: str):
        """
        Initialize the tokenizer.
        
        Args:
            tokenizer_path: Path to the tokenizer.json file
        """
        self.tokenizer = HFTokenizer.from_file(tokenizer_path)
        self.tokenizer_path = tokenizer_path
        
        # Load special tokens
        special_tokens_path = os.path.join(os.path.dirname(tokenizer_path), "special_tokens.json")
        if os.path.exists(special_tokens_path):
            with open(special_tokens_path, 'r', encoding='utf-8') as f:
                self.special_tokens = json.load(f)
        else:
            self.special_tokens = {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>"
            }
        
        # Set token IDs
        self.bos_token_id = self.tokenizer.token_to_id(self.special_tokens["bos_token"])
        self.eos_token_id = self.tokenizer.token_to_id(self.special_tokens["eos_token"])
        self.unk_token_id = self.tokenizer.token_to_id(self.special_tokens["unk_token"])
        self.pad_token_id = self.tokenizer.token_to_id(self.special_tokens["pad_token"])
        
        # Vocabulary
        self.vocab = self.tokenizer.get_vocab()
        self.vocab_size = len(self.vocab)
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add special tokens
            
        Returns:
            List of token IDs
        """
        encoded = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return encoded.ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def encode_batch(self, texts: List[str], add_special_tokens: bool = True) -> List[List[int]]:
        """
        Encode a batch of texts.
        
        Args:
            texts: List of input texts
            add_special_tokens: Whether to add special tokens
            
        Returns:
            List of token ID lists
        """
        encoded = self.tokenizer.encode_batch(texts, add_special_tokens=add_special_tokens)
        return [e.ids for e in encoded]
    
    def decode_batch(self, token_id_lists: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
        """
        Decode a batch of token ID lists.
        
        Args:
            token_id_lists: List of token ID lists
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            List of decoded texts
        """
        return [self.decode(ids, skip_special_tokens) for ids in token_id_lists]
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into tokens.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        encoded = self.tokenizer.encode(text)
        return encoded.tokens
    
    def get_vocab(self) -> Dict[str, int]:
        """Get the vocabulary mapping."""
        return self.vocab.copy()
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size."""
        return self.vocab_size
    
    def save(self, path: str):
        """Save the tokenizer to a file."""
        self.tokenizer.save(path)
    
    @classmethod
    def from_pretrained(cls, path: str):
        """Load a tokenizer from a saved path."""
        return cls(path)
