#!/usr/bin/env python3
"""
Calculate exact parameter counts for DLN models.
"""
import json
from pathlib import Path
from typing import Dict, List

import torch
from dln import SimpleDLN


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count trainable and total parameters in a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Break down by layer
    breakdown = {}
    for name, param in model.named_parameters():
        breakdown[name] = param.numel()
    
    return {
        "total": total,
        "trainable": trainable,
        "breakdown": breakdown
    }


def estimate_from_tinystories_data(
    stories_path: str = "data/processed/tinystories_train.json",
    max_stories: int = 50,
    embed_dim: int = 32
):
    """Estimate DLN size based on TinyStories data."""
    
    fpath = Path(stories_path)
    if not fpath.exists():
        print(f"Data file not found: {stories_path}")
        return None
    
    with open(fpath, "r") as f:
        data = json.load(f)
    
    # Extract vocabulary from data
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
    
    # Build predicate and arg vocabularies (as done in tinystories_pipeline)
    predicates = sorted(relations)
    # Add derived predicates
    predicates += [f"{r}_inferred" for r in predicates]
    if len(predicates) >= 2:
        for i, r1 in enumerate(sorted(relations)):
            for r2 in sorted(relations)[i + 1:]:
                predicates.append(f"{r1}_{r2}_combo")
    predicates += [f"not_{r}" for r in sorted(relations)]
    
    args = ["<pad>"] + sorted(entities)
    
    print(f"\nVocabulary from {max_stories} stories:")
    print(f"  Base relations: {len(relations)}")
    print(f"  Total predicates: {len(predicates)}")
    print(f"  Total args: {len(args)}")
    
    # Create model and count
    model = SimpleDLN(predicates, args, embed_dim=embed_dim)
    counts = count_parameters(model)
    
    print(f"\nModel parameters (embed_dim={embed_dim}):")
    print(f"  Total parameters: {counts['total']:,}")
    print(f"  Trainable parameters: {counts['trainable']:,}")
    print(f"\nBreakdown:")
    for name, count in sorted(counts['breakdown'].items()):
        print(f"    {name}: {count:,}")
    
    # Calculate memory footprint
    memory_mb = counts['total'] * 4 / (1024 * 1024)  # assuming float32
    print(f"\nMemory footprint: {memory_mb:.2f} MB (float32)")
    
    # Compare to baselines
    print(f"\nComparison to Transformer baselines:")
    baselines = [
        ("GPT-2 (1M)", 1_000_000),
        ("GPT-2 (8M)", 8_000_000),
        ("GPT (22M)", 22_000_000),
        ("GPT-2 (124M)", 124_000_000),
    ]
    for name, params in baselines:
        ratio = params / counts['total']
        print(f"  {name}: {ratio:.1f}× larger")
    
    return counts


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Count DLN parameters")
    parser.add_argument("--max-stories", type=int, default=50, help="Stories to analyze")
    parser.add_argument("--embed-dim", type=int, default=32, help="Embedding dimension")
    parser.add_argument("--data-path", type=str, default="data/processed/tinystories_train.json")
    args = parser.parse_args()
    
    estimate_from_tinystories_data(
        stories_path=args.data_path,
        max_stories=args.max_stories,
        embed_dim=args.embed_dim
    )
