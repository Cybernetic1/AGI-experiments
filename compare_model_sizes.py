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
    
    # Transformer baselines (from literature - for reference)
    transformers_reference = [
        {"name": "GPT-2\n(1M)", "params": 1_000_000, "capability": "Basic"},
        {"name": "GPT-2\n(8M)", "params": 8_000_000, "capability": "Coherent"},
        {"name": "GPT\n(22M)", "params": 22_000_000, "capability": "Good"},
        {"name": "GPT-2\n(124M)", "params": 124_000_000, "capability": "Strong"},
    ]
    
    # ACTUAL MEASURED RESULTS from compare_tinystories_simple.py
    # Task: Logical inference on TinyStories facts
    # Date: 2026-01-27
    actual_results = [
        {
            "scale": "Small (10 stories)",
            "transformer_params": 27_393,
            "transformer_acc": 66.7,
            "dln_params": 9_217,
            "dln_acc": 66.7,
            "compression": 3.0
        },
        {
            "scale": "Medium (50 stories)",
            "transformer_params": 29_921,
            "transformer_acc": 67.9,
            "dln_params": 11_745,
            "dln_acc": 75.0,
            "compression": 2.5
        },
        {
            "scale": "Large (200 stories)",
            "transformer_params": 37_025,
            "transformer_acc": 71.7,
            "dln_params": 18_849,
            "dln_acc": 69.7,
            "compression": 2.0
        }
    ]
    
    return transformers_reference, actual_results


def plot_comparison(output_path="docs/compression_comparison.png"):
    """Create comparison graph for presentation."""
    
    transformers_ref, actual_results = create_comparison_data()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Extract data from actual results
    scales = [r["scale"] for r in actual_results]
    transformer_params = [r["transformer_params"] for r in actual_results]
    dln_params = [r["dln_params"] for r in actual_results]
    compressions = [r["compression"] for r in actual_results]
    
    x = np.arange(len(scales))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, transformer_params, width, 
                   label='Transformer Baseline', color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar(x + width/2, dln_params, width,
                   label='Genifer DLN', color='#51CF66', alpha=0.8)
    
    # Formatting
    ax.set_ylabel('Parameters', fontsize=13, fontweight='bold')
    ax.set_xlabel('Dataset Scale', fontsize=13, fontweight='bold')
    ax.set_title('Measured Performance: Genifer DLN vs Transformer Baseline\nLogical Inference Task on TinyStories', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(scales, fontsize=11)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Annotate with compression ratios and accuracies
    for i in range(len(scales)):
        mid_y = max(transformer_params[i], dln_params[i]) * 1.1
        ax.text(x[i], mid_y, f'{compressions[i]:.1f}×\nsmaller', 
                ha='center', va='bottom', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
        
        # Add accuracy labels
        trans_acc = actual_results[i]["transformer_acc"]
        dln_acc = actual_results[i]["dln_acc"]
        ax.text(x[i] - width/2, transformer_params[i] * 0.5, f'{trans_acc:.1f}%', 
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        ax.text(x[i] + width/2, dln_params[i] * 0.5, f'{dln_acc:.1f}%', 
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    # Add parameter counts on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height >= 1_000_000:
                label = f'{height/1_000_000:.1f}M'
            elif height >= 1_000:
                label = f'{height/1_000:.1f}K'
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
    print("\n" + "="*70)
    print("ACTUAL MEASURED COMPRESSION RATIOS")
    print("="*70)
    for r in actual_results:
        print(f"{r['scale']:20s}: {r['transformer_params']:>8,} vs {r['dln_params']:>8,} params  →  {r['compression']:.1f}× compression")
        print(f"                      Transformer: {r['transformer_acc']:.1f}% acc, DLN: {r['dln_acc']:.1f}% acc")
    print("="*70)
    print("\nTask: Logical inference (predicting truth values of propositions)")
    print("Method: Neural training only, no ILP/GA/rule injection")
    print("Date: 2026-01-27")


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
