"""
Visualization utilities for the Free Transformer synthetic dataset.

Provides:
- Text-based visualization (terminal-friendly)
- Matplotlib plots for analysis
- Distribution plots for latent variables
"""

import random
from collections import Counter
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from syntheticdata import (
    SyntheticConfig, 
    SyntheticDataset, 
    SyntheticTokenizer,
    generate_single_sequence,
)


# Color scheme for visualization
COLORS = {
    '_': '#E8E8E8',   # Light gray for background
    '!': '#FF6B6B',   # Red for noise
    'target': '#4ECDC4',  # Teal for target letters
    '>': '#95A5A6',   # Gray for separator
}


def print_sequence_highlighted(seq: str, metadata: dict) -> None:
    """
    Print a sequence with ANSI color highlighting.
    
    - Target letters: cyan/bold
    - Noise (!): red
    - Background (_): dim
    """
    letter = metadata['letter']
    
    # ANSI codes
    CYAN = '\033[96m\033[1m'
    RED = '\033[91m'
    DIM = '\033[2m'
    RESET = '\033[0m'
    
    result = []
    for char in seq:
        if char == letter:
            result.append(f"{CYAN}{char}{RESET}")
        elif char == '!':
            result.append(f"{RED}{char}{RESET}")
        elif char == '_':
            result.append(f"{DIM}{char}{RESET}")
        else:
            result.append(char)
    
    print(''.join(result))


def print_batch(sequences: list[str], metadata_list: list[dict], n: int = 10) -> None:
    """Print multiple sequences with highlighting."""
    print("\n" + "=" * 70)
    for i, (seq, meta) in enumerate(zip(sequences[:n], metadata_list[:n])):
        print_sequence_highlighted(seq, meta)
    print("=" * 70 + "\n")


def visualize_sequence_grid(
    sequences: list[str],
    metadata_list: list[dict],
    n_rows: int = 15,
    figsize: tuple = (14, 8),
    title: str = "Synthetic Dataset Samples",
) -> plt.Figure:
    """
    Create a grid visualization of sequences using matplotlib.
    
    Each row is one sequence, with characters colored by type.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    n_rows = min(n_rows, len(sequences))
    seq_length = len(sequences[0])
    
    # Create image array
    img = np.ones((n_rows, seq_length, 3))
    
    for row, (seq, meta) in enumerate(zip(sequences[:n_rows], metadata_list[:n_rows])):
        letter = meta['letter']
        for col, char in enumerate(seq):
            if char == letter:
                # Target - teal
                img[row, col] = [0.31, 0.80, 0.77]
            elif char == '!':
                # Noise - red
                img[row, col] = [1.0, 0.42, 0.42]
            elif char == '_':
                # Background - light gray
                img[row, col] = [0.91, 0.91, 0.91]
            elif char == '>':
                # Separator - medium gray
                img[row, col] = [0.58, 0.65, 0.65]
            else:
                # Prompt letter - teal (same as target)
                img[row, col] = [0.31, 0.80, 0.77]
    
    ax.imshow(img, aspect='auto')
    
    # Add sequence text labels on the left
    for row in range(n_rows):
        ax.text(-1, row, sequences[row][:2], ha='right', va='center', 
                fontfamily='monospace', fontsize=8)
    
    ax.set_xlim(-0.5, seq_length - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_xlabel('Position', fontsize=10)
    ax.set_ylabel('Sample', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Remove y-axis ticks
    ax.set_yticks([])
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=[0.31, 0.80, 0.77], label='Target'),
        mpatches.Patch(facecolor=[1.0, 0.42, 0.42], label='Noise (!)'),
        mpatches.Patch(facecolor=[0.91, 0.91, 0.91], label='Background (_)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_position_distribution(
    metadata_list: list[dict],
    config: SyntheticConfig,
    figsize: tuple = (10, 4),
) -> plt.Figure:
    """Plot the distribution of target positions."""
    positions = [m['target_position'] for m in metadata_list]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    max_pos = config.seq_length - config.target_length
    bins = np.arange(0, max_pos + 2) - 0.5
    
    ax.hist(positions, bins=bins, edgecolor='white', color='#4ECDC4', alpha=0.8)
    ax.set_xlabel('Target Position', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title('Distribution of Target Positions', fontsize=12, fontweight='bold')
    ax.axhline(y=len(metadata_list) / (max_pos + 1), color='red', 
               linestyle='--', label='Expected (uniform)')
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_noise_distribution(
    metadata_list: list[dict],
    config: SyntheticConfig,
    figsize: tuple = (10, 4),
) -> plt.Figure:
    """Plot the distribution of noise counts."""
    noise_counts = [m['noise_count'] for m in metadata_list]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    counter = Counter(noise_counts)
    x = sorted(counter.keys())
    y = [counter[k] for k in x]
    
    ax.bar(x, y, edgecolor='white', color='#FF6B6B', alpha=0.8)
    ax.set_xlabel('Number of Noise Characters (!)', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title('Distribution of Noise Count per Sequence', fontsize=12, fontweight='bold')
    
    # Add expected value line (binomial)
    expected = config.seq_length * config.noise_prob
    ax.axvline(x=expected, color='blue', linestyle='--', 
               label=f'Expected mean: {expected:.1f}')
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_letter_distribution(
    metadata_list: list[dict],
    figsize: tuple = (12, 4),
) -> plt.Figure:
    """Plot the distribution of target letters."""
    letters = [m['letter'] for m in metadata_list]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    counter = Counter(letters)
    x = sorted(counter.keys())
    y = [counter[k] for k in x]
    
    ax.bar(x, y, edgecolor='white', color='#4ECDC4', alpha=0.8)
    ax.set_xlabel('Target Letter', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title('Distribution of Target Letters', fontsize=12, fontweight='bold')
    ax.axhline(y=len(metadata_list) / 26, color='red', 
               linestyle='--', label='Expected (uniform)')
    ax.legend()
    
    plt.tight_layout()
    return fig


def create_full_analysis(
    n_samples: int = 1000,
    config: Optional[SyntheticConfig] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Generate a complete visual analysis of the dataset.
    
    Creates a multi-panel figure with:
    - Sample sequences grid
    - Position distribution
    - Noise distribution
    - Letter distribution
    """
    config = config or SyntheticConfig(seed=42)
    
    # Generate data
    sequences = []
    metadata_list = []
    for _ in range(n_samples):
        seq, meta = generate_single_sequence(config)
        sequences.append(seq)
        metadata_list.append(meta)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Grid of sequences (top)
    ax1 = fig.add_subplot(2, 2, (1, 2))
    
    n_rows = 20
    seq_length = len(sequences[0])
    img = np.ones((n_rows, seq_length, 3))
    
    for row, (seq, meta) in enumerate(zip(sequences[:n_rows], metadata_list[:n_rows])):
        letter = meta['letter']
        for col, char in enumerate(seq):
            if char == letter:
                img[row, col] = [0.31, 0.80, 0.77]
            elif char == '!':
                img[row, col] = [1.0, 0.42, 0.42]
            elif char == '_':
                img[row, col] = [0.91, 0.91, 0.91]
            elif char == '>':
                img[row, col] = [0.58, 0.65, 0.65]
            else:
                img[row, col] = [0.31, 0.80, 0.77]
    
    ax1.imshow(img, aspect='auto')
    ax1.set_title(f'Sample Sequences (n={n_samples} total, showing {n_rows})', 
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Position')
    ax1.set_yticks([])
    
    legend_elements = [
        mpatches.Patch(facecolor=[0.31, 0.80, 0.77], label='Target'),
        mpatches.Patch(facecolor=[1.0, 0.42, 0.42], label='Noise (!)'),
        mpatches.Patch(facecolor=[0.91, 0.91, 0.91], label='Background (_)'),
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # Position distribution (bottom left)
    ax2 = fig.add_subplot(2, 2, 3)
    positions = [m['target_position'] for m in metadata_list]
    max_pos = config.seq_length - config.target_length
    bins = np.arange(0, max_pos + 2) - 0.5
    ax2.hist(positions, bins=bins, edgecolor='white', color='#4ECDC4', alpha=0.8)
    ax2.set_xlabel('Target Position')
    ax2.set_ylabel('Count')
    ax2.set_title('Target Position Distribution', fontsize=11, fontweight='bold')
    ax2.axhline(y=len(metadata_list) / (max_pos + 1), color='red', 
                linestyle='--', label='Expected')
    ax2.legend()
    
    # Noise distribution (bottom right)
    ax3 = fig.add_subplot(2, 2, 4)
    noise_counts = [m['noise_count'] for m in metadata_list]
    counter = Counter(noise_counts)
    x = sorted(counter.keys())
    y = [counter[k] for k in x]
    ax3.bar(x, y, edgecolor='white', color='#FF6B6B', alpha=0.8)
    ax3.set_xlabel('Number of Noise Characters')
    ax3.set_ylabel('Count')
    ax3.set_title('Noise Count Distribution', fontsize=11, fontweight='bold')
    expected = config.seq_length * config.noise_prob
    ax3.axvline(x=expected, color='blue', linestyle='--', 
                label=f'Expected: {expected:.1f}')
    ax3.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    return fig


if __name__ == "__main__":
    print("=== Dataset Visualization Demo ===\n")
    
    config = SyntheticConfig(seed=42)
    
    # Generate samples
    sequences = []
    metadata_list = []
    for _ in range(20):
        seq, meta = generate_single_sequence(config)
        sequences.append(seq)
        metadata_list.append(meta)
    
    # Terminal visualization
    print("Terminal view (with ANSI colors):")
    print("-" * 70)
    for seq, meta in zip(sequences[:10], metadata_list[:10]):
        print_sequence_highlighted(seq, meta)
    print("-" * 70)
    
    # Save matplotlib visualization
    print("\nGenerating matplotlib visualization...")
    fig = create_full_analysis(n_samples=1000, config=config, save_path="dataset_analysis.png")
    plt.close(fig)
    
    print("\nDone! Check dataset_analysis.png")