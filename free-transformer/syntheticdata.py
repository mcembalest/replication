"""
Synthetic Dataset from "The Free Transformer" (Fleuret, 2025) Section 4.1

Each sequence has:
- A "target" of 8 repeated letters at a random position
- Random noise (!) with probability 1/16
- A prompt indicating which letter to look for

Example: "K>!_________!_______!____________!_______KKKKKKKK_________"

The latent structure:
- Target position (0 to seq_length - target_length)
- Noise pattern (binary mask)

This lets us verify that the Free Transformer learns to encode
meaningful latent variables in Z.
"""

import random
import string
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class SyntheticConfig:
    """Configuration for synthetic dataset generation."""
    seq_length: int = 64          # Length of the main sequence (underscores)
    target_length: int = 8        # Length of the repeated letter target
    noise_prob: float = 1/16      # Probability of replacing a char with '!'
    seed: Optional[int] = None    # Random seed for reproducibility


class SyntheticTokenizer:
    """
    Character-level tokenizer for the synthetic dataset.
    
    Vocabulary (29 tokens):
        - A-Z (26 letters)
        - '>' (prompt separator)
        - '_' (background)
        - '!' (noise)
    """
    
    # Special tokens
    PAD_TOKEN = '<PAD>'
    
    def __init__(self):
        # Build vocabulary: A-Z, then special chars
        self.chars = list(string.ascii_uppercase) + ['>', '_', '!']
        self.char_to_id = {c: i for i, c in enumerate(self.chars)}
        self.id_to_char = {i: c for i, c in enumerate(self.chars)}
        
        # Add padding token
        self.pad_id = len(self.chars)
        self.char_to_id[self.PAD_TOKEN] = self.pad_id
        self.id_to_char[self.pad_id] = self.PAD_TOKEN
        
        self.vocab_size = len(self.char_to_id)
    
    def encode(self, text: str) -> list[int]:
        """Convert string to list of token IDs."""
        return [self.char_to_id[c] for c in text]
    
    def decode(self, ids: list[int]) -> str:
        """Convert token IDs back to string."""
        return ''.join(self.id_to_char[i] for i in ids if i != self.pad_id)
    
    def __repr__(self):
        return f"SyntheticTokenizer(vocab_size={self.vocab_size})"


def generate_single_sequence(config: SyntheticConfig) -> tuple[str, dict]:
    """
    Generate one training sequence.
    
    Returns:
        sequence: The full sequence string (e.g., "K>_____KKKKKKKK_____!")
        metadata: Dict with 'letter', 'target_position', 'noise_positions'
    """
    # Start with underscores
    chars = ['_'] * config.seq_length
    
    # Pick random uppercase letter and position
    letter = random.choice(string.ascii_uppercase)
    max_start = config.seq_length - config.target_length
    target_pos = random.randint(0, max_start)
    
    # Place the target (letter repeated target_length times)
    for i in range(config.target_length):
        chars[target_pos + i] = letter
    
    # Add noise (replace with '!' with given probability)
    noise_positions = []
    for i in range(config.seq_length):
        if random.random() < config.noise_prob:
            chars[i] = '!'
            noise_positions.append(i)
    
    # Create prompt + sequence
    sequence = letter + '>' + ''.join(chars)
    
    metadata = {
        'letter': letter,
        'target_position': target_pos,
        'noise_positions': noise_positions,
        'noise_count': len(noise_positions),
    }
    
    return sequence, metadata


class SyntheticDataset(Dataset):
    """
    PyTorch Dataset for the synthetic sequences.
    
    Each item is a dict with:
        - 'input_ids': Tensor of token IDs
        - 'labels': Same as input_ids (for LM training, shifted internally)
        - 'metadata': Dict with ground truth latent info
    """
    
    def __init__(
        self, 
        n_samples: int = 10000,
        config: Optional[SyntheticConfig] = None,
        tokenizer: Optional[SyntheticTokenizer] = None,
    ):
        self.config = config or SyntheticConfig()
        self.tokenizer = tokenizer or SyntheticTokenizer()
        self.n_samples = n_samples
        
        # Set seed if provided
        if self.config.seed is not None:
            random.seed(self.config.seed)
        
        # Pre-generate all sequences
        self.sequences = []
        self.metadata = []
        
        for _ in range(n_samples):
            seq, meta = generate_single_sequence(self.config)
            self.sequences.append(seq)
            self.metadata.append(meta)
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> dict:
        seq = self.sequences[idx]
        token_ids = self.tokenizer.encode(seq)
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'labels': torch.tensor(token_ids, dtype=torch.long),
            'metadata': self.metadata[idx],
            'text': seq,
        }
    
    def get_sequence_length(self) -> int:
        """Return the fixed sequence length (prompt + main sequence)."""
        # "X>" + seq_length = 2 + seq_length
        return 2 + self.config.seq_length


def create_dataloader(
    n_samples: int = 10000,
    batch_size: int = 32,
    config: Optional[SyntheticConfig] = None,
    shuffle: bool = True,
    num_workers: int = 0,
) -> tuple[DataLoader, SyntheticTokenizer]:
    """
    Convenience function to create a DataLoader and tokenizer.
    
    Returns:
        dataloader: PyTorch DataLoader
        tokenizer: The tokenizer used
    """
    tokenizer = SyntheticTokenizer()
    dataset = SyntheticDataset(n_samples=n_samples, config=config, tokenizer=tokenizer)
    
    # Custom collate to handle metadata
    def collate_fn(batch):
        return {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'labels': torch.stack([item['labels'] for item in batch]),
            'metadata': [item['metadata'] for item in batch],
            'text': [item['text'] for item in batch],
        }
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    
    return dataloader, tokenizer


# Quick test when run directly
if __name__ == "__main__":
    print("=== Synthetic Dataset Demo ===\n")
    
    config = SyntheticConfig(seed=42)
    tokenizer = SyntheticTokenizer()
    
    print(f"Tokenizer: {tokenizer}")
    print(f"Vocabulary: {tokenizer.chars}\n")
    
    print("Sample sequences:")
    print("-" * 70)
    
    for i in range(10):
        seq, meta = generate_single_sequence(config)
        print(f"{seq}")
        print(f"  → target '{meta['letter']}' at position {meta['target_position']}, "
              f"{meta['noise_count']} noise chars")
    
    print("\n" + "-" * 70)
    print("\nDataset test:")
    dataset = SyntheticDataset(n_samples=100, config=config, tokenizer=tokenizer)
    print(f"Dataset size: {len(dataset)}")
    print(f"Sequence length: {dataset.get_sequence_length()}")
    
    sample = dataset[0]
    print(f"\nSample item:")
    print(f"  text: {sample['text']}")
    print(f"  input_ids shape: {sample['input_ids'].shape}")
    print(f"  metadata: {sample['metadata']}")