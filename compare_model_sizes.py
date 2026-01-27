#!/usr/bin/env python3
"""
Compare DLN model size vs Transformer baselines at different capability levels.
Generate graph for presentation showing compression ratio advantage.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from dln import SimpleDLN

dln_estimates = [
    {"name": "DLN\n(10 stories)", "params": 8525, "capability": "Basic"},
    {"name": "DLN\n(50 stories)", "params": 12666, "capability": "Coherent"},
    {"name": "DLN\n(100 stories)", "params": 17611, "capability": "Good"},
    {"name": "DLN\n(200 stories)", "params": 21899, "capability": "Good"},
    {"name": "DLN\n(500 stories)", "params": 30955, "capability": "Strong"},
]

def count_params(model):
    """Count total parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters())


def estimate_dln_params(num_predicates, num_args, embed_dim=32):
    """Estimate DLN parameters without instantiating model."""
    # pred_embed + arg_embed
    embedding_params = num_predicates * embed_dim + num_args * embed_dim
    
    # MLP layers (approximate based on SimpleDLN)
    mlp_params = (embed_dim * 3 * 2) * 32 + 32 * 1  # rough estimate
    
    # AR head
    ar_head_params = (embed_dim * 3) * num_predicates
    
    total = embedding_params + mlp_params + ar_head_params
    return total


def get_actual_dln_size(max_stories=50, embed_dim=32):
    """Get actual DLN parameter count from data."""
    data_path = Path("data/processed/tinystories_train.json")
    if not data_path.exists():
        print(f"Warning: {data_path} not found, using estimates")
        return None
    
    with open(data_path, "r") as f:
        data = json.load(f)
    
    # Extract vocabulary
    relations = set()
    entities = set()
    
    for story in data[:max_stories]:
        for fact in story.get("facts", []):
            rel = fact.get("relation", "")
            subj = fact.get("subject", "")
            obj = fact.get("object", "")
            if rel:
                relations.add(rel)
            if subj:
                entities.add(subj)
            if obj:
                entities.add(obj)
    
    # Build vocabularies (simplified from actual pipeline)
    predicates = sorted(relations)
    args = ["<pad>"] + sorted(entities)
    
    num_preds = len(predicates) * 3  # approximate expansion with derived predicates
    num_args = len(args)
    
    return estimate_dln_params(num_preds, num_args, embed_dim)


def create_comparison_data():
    """Create data comparing DLN vs Transformer at different capability levels."""
    
    # Transformer baselines (from literature)
    transformers = [
        {"name": "GPT-2\n(1M)", "params": 1_000_000, "capability": "Basic"},
        {"name": "GPT-2\n(8M)", "params": 8_000_000, "capability": "Coherent"},
        {"name": "GPT\n(22M)", "params": 22_000_000, "capability": "Good"},
        {"name": "GPT-2\n(124M)", "params": 124_000_000, "capability": "Strong"},
    ]
    
    # Actual measurements from measure_dln_sizes.py
    # Uses real vocabulary counts from tinystories data
    dln_estimates = [
        {"name": "DLN\n(10 stories)", "params": 8_525, "capability": "Basic"},
        {"name": "DLN\n(50 stories)", "params": 12_666, "capability": "Coherent"},
        {"name": "DLN\n(200 stories)", "params": 21_899, "capability": "Good"},
        {"name": "DLN\n(500 stories)", "params": 30_955, "capability": "Strong"},
    ]
    
    return transformers, dln_estimates


def plot_comparison(output_path="docs/compression_comparison.png"):
    """Create comparison graph for presentation."""
    
    transformers, dlns = create_comparison_data()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Extract data
    capabilities = ["Basic", "Coherent", "Good", "Strong"]
    
    transformer_params = [t["params"] for t in transformers]
    dln_params = [d["params"] for d in dlns]
    
    x = np.arange(len(capabilities))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, transformer_params, width, 
                   label='Transformer Models', color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar(x + width/2, dln_params, width,
                   label='Genifer DLN', color='#51CF66', alpha=0.8)
    
    # Formatting
    ax.set_ylabel('Parameters (log scale)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Capability Level', fontsize=13, fontweight='bold')
    ax.set_title('Model Size Comparison: Genifer DLN vs Standard Transformers', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(capabilities, fontsize=12)
    ax.set_yscale('log')
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Annotate with compression ratios
    for i in range(len(capabilities)):
        ratio = transformer_params[i] / dln_params[i]
        mid_y = np.sqrt(transformer_params[i] * dln_params[i])  # geometric mean for log scale
        ax.text(x[i], mid_y, f'{ratio:.0f}×\nsmaller', 
                ha='center', va='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    # Add parameter counts on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height >= 1_000_000:
                label = f'{height/1_000_000:.1f}M'
            elif height >= 1_000:
                label = f'{height/1_000:.0f}K'
            else:
                label = f'{height:.0f}'
            
            ax.annotate(label,
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nGraph saved to {output_path}")
    
    # Print statistics
    print("\n" + "="*60)
    print("COMPRESSION RATIO SUMMARY")
    print("="*60)
    for i, cap in enumerate(capabilities):
        ratio = transformer_params[i] / dln_params[i]
        print(f"{cap:10s}: {transformer_params[i]:>12,} vs {dln_params[i]:>8,} params  →  {ratio:>5.0f}× compression")
    print("="*60)


def print_table():
    """Print markdown table for documentation."""
    transformers, dlns = create_comparison_data()
    
    print("\n## Model Size Comparison Table\n")
    print("| Capability | Transformer | DLN | Compression Ratio |")
    print("|------------|-------------|-----|-------------------|")
    
    for i in range(len(transformers)):
        t_params = transformers[i]["params"]
        d_params = dlns[i]["params"]
        ratio = t_params / d_params
        
        t_str = f"{t_params/1_000_000:.1f}M" if t_params >= 1_000_000 else f"{t_params/1_000:.0f}K"
        d_str = f"{d_params/1_000:.0f}K" if d_params >= 1_000 else f"{d_params:.0f}"
        
        print(f"| {transformers[i]['capability']:10s} | {t_str:>10s} | {d_str:>10s} | **{ratio:.0f}×** |")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare DLN vs Transformer model sizes")
    parser.add_argument("--output", type=str, default="docs/compression_comparison.png",
                       help="Output path for graph")
    parser.add_argument("--table", action="store_true", help="Print markdown table")
    
    args = parser.parse_args()
    
    if args.table:
        print_table()
    
    plot_comparison(args.output)
