#!/usr/bin/env python3
"""
FAIR COMPARISON: DLN vs Transformer with MATCHED PARAMETERS
============================================================

Key insight: Compare models with ~same parameter count
Measure: Convergence speed and final accuracy

This is the most honest comparison:
- Fix parameter budget (e.g., 10K, 30K, 100K params)
- Tune architecture (num_rules for DLN, num_layers for Transformer)
- Train both for same epochs
- Compare: convergence speed + final accuracy

If DLN converges faster or reaches higher accuracy with same params,
that's a real architectural advantage!
"""

import torch
import torch.nn as nn
import json
from pathlib import Path
import argparse
import random
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict


def load_facts(max_stories=50):
    """Load facts from preprocessed TinyStories."""
    data_path = Path("data/processed/tinystories_train.json")
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    all_facts = []
    for story in data[:max_stories]:
        for fact in story.get('facts', []):
            all_facts.append((
                fact.get('relation', ''),
                fact.get('subject', ''),
                fact.get('object', '')
            ))
    
    return all_facts


def build_vocabularies(facts):
    """Build predicate and argument vocabularies."""
    predicates = {'<PAD>': 0}
    args = {'<PAD>': 0}
    
    for rel, subj, obj in facts:
        if rel and rel not in predicates:
            predicates[rel] = len(predicates)
        if subj and subj not in args:
            args[subj] = len(args)
        if obj and obj not in args:
            args[obj] = len(args)
    
    return predicates, args


def create_train_data(facts, predicates, args, negative_ratio=0.5):
    """Create training examples with positive and negative samples."""
    examples = []
    
    # Positive examples
    for rel, subj, obj in facts:
        pred_id = predicates.get(rel, 0)
        arg1_id = args.get(subj, 0)
        arg2_id = args.get(obj, 0)
        if pred_id > 0 and arg1_id > 0 and arg2_id > 0:
            examples.append(([pred_id, arg1_id, arg2_id], 1.0))
    
    # Negative examples
    num_negatives = int(len(examples) * negative_ratio)
    pred_ids = list(predicates.values())[1:]
    arg_ids = list(args.values())[1:]
    
    for _ in range(num_negatives):
        pred_id = random.choice(pred_ids)
        arg1_id = random.choice(arg_ids)
        arg2_id = random.choice(arg_ids)
        
        # Verify it's not a real fact
        is_real = any(
            predicates.get(rel, 0) == pred_id and
            args.get(subj, 0) == arg1_id and
            args.get(obj, 0) == arg2_id
            for rel, subj, obj in facts
        )
        if not is_real:
            examples.append(([pred_id, arg1_id, arg2_id], 0.0))
    
    return examples


class TransformerLogic(nn.Module):
    """Transformer with configurable layers."""
    
    def __init__(self, num_predicates, num_args, embed_dim=32, num_layers=2):
        super().__init__()
        
        self.pred_embed = nn.Embedding(num_predicates, embed_dim)
        self.arg_embed = nn.Embedding(num_args, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=max(1, embed_dim // 16),  # Adaptive heads
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.output = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, triple):
        pred_emb = self.pred_embed(triple[:, 0:1])
        arg1_emb = self.arg_embed(triple[:, 1:2])
        arg2_emb = self.arg_embed(triple[:, 2:3])
        
        x = torch.cat([pred_emb, arg1_emb, arg2_emb], dim=1)
        x = self.transformer(x)
        x = x.mean(dim=1)
        prob = self.output(x).squeeze(-1)
        
        return prob


class DLNLogic(nn.Module):
    """DLN with configurable rules."""
    
    def __init__(self, num_predicates, num_args, embed_dim=32, num_rules=8):
        super().__init__()
        
        self.num_rules = num_rules
        self.pred_embed = nn.Embedding(num_predicates, embed_dim)
        self.arg_embed = nn.Embedding(num_args, embed_dim)
        
        # Explicit rule patterns
        self.rule_patterns = nn.Parameter(torch.randn(num_rules, embed_dim * 3))
        
        # Rule combination
        self.rule_combiner = nn.Sequential(
            nn.Linear(num_rules, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, triple):
        pred_emb = self.pred_embed(triple[:, 0])
        arg1_emb = self.arg_embed(triple[:, 1])
        arg2_emb = self.arg_embed(triple[:, 2])
        
        # Concatenate proposition representation
        prop = torch.cat([pred_emb, arg1_emb, arg2_emb], dim=-1)  # (batch, 3*embed_dim)
        
        # Match against rule patterns
        rule_activations = []
        for i in range(self.num_rules):
            pattern = self.rule_patterns[i:i+1]  # (1, 3*embed_dim)
            # Cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                prop, pattern.expand(prop.shape[0], -1), dim=-1
            )
            rule_activations.append(similarity.unsqueeze(-1))
        
        rule_activations = torch.cat(rule_activations, dim=-1)  # (batch, num_rules)
        rule_activations = torch.sigmoid(rule_activations)
        
        # Combine rule outputs
        prob = self.rule_combiner(rule_activations).squeeze(-1)
        
        return prob


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def find_matching_configs(num_predicates, num_args, target_params, embed_dim=32):
    """Find DLN and Transformer configs with similar parameter counts."""
    
    configs = []
    
    # Try different num_layers for Transformer
    for num_layers in range(1, 6):
        trans = TransformerLogic(num_predicates, num_args, embed_dim, num_layers)
        trans_params = count_parameters(trans)
        
        # Find matching DLN with different num_rules
        for num_rules in range(2, 32):
            dln = DLNLogic(num_predicates, num_args, embed_dim, num_rules)
            dln_params = count_parameters(dln)
            
            # Check if params are within 20% of each other
            if abs(trans_params - dln_params) / max(trans_params, dln_params) < 0.2:
                configs.append({
                    'transformer': {
                        'num_layers': num_layers,
                        'params': trans_params
                    },
                    'dln': {
                        'num_rules': num_rules,
                        'params': dln_params
                    },
                    'avg_params': (trans_params + dln_params) / 2
                })
                break
    
    # Sort by parameter count
    configs.sort(key=lambda x: x['avg_params'])
    
    return configs


def train_and_track(model, train_data, epochs, lr, batch_size, device):
    """Train model and track convergence."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    history = {'epoch': [], 'loss': [], 'accuracy': []}
    
    for epoch in range(epochs):
        model.train()
        random.shuffle(train_data)
        
        total_loss = 0
        correct = 0
        total = 0
        
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            if len(batch) == 0:
                continue
            
            X = torch.tensor([x[0] for x in batch], dtype=torch.long, device=device)
            y = torch.tensor([x[1] for x in batch], dtype=torch.float32, device=device)
            
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            correct += ((pred > 0.5) == (y > 0.5)).sum().item()
            total += len(batch)
        
        avg_loss = total_loss / (len(train_data) / batch_size)
        acc = correct / total * 100
        
        history['epoch'].append(epoch + 1)
        history['loss'].append(avg_loss)
        history['accuracy'].append(acc)
    
    return history


def plot_convergence(results, output_path="docs/convergence_comparison.png"):
    """Plot convergence curves."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for result in results:
        label = f"{result['name']} ({result['params']:,} params)"
        color = result['color']
        
        # Plot accuracy
        ax1.plot(result['history']['epoch'], result['history']['accuracy'],
                label=label, color=color, linewidth=2)
        
        # Plot loss
        ax2.plot(result['history']['epoch'], result['history']['loss'],
                label=label, color=color, linewidth=2)
    
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Convergence: Accuracy', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Convergence: Loss', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nConvergence plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stories", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    print("=" * 70)
    print("PARAMETER-MATCHED COMPARISON: DLN vs Transformer")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading {args.stories} stories...")
    facts = load_facts(max_stories=args.stories)
    predicates, args_vocab = build_vocabularies(facts)
    
    print(f"  Predicates: {len(predicates)}, Arguments: {len(args_vocab)}")
    
    # Create dataset
    all_data = create_train_data(facts, predicates, args_vocab)
    random.shuffle(all_data)
    split_idx = int(0.8 * len(all_data))
    train_data = all_data[:split_idx]
    
    print(f"  Training examples: {len(train_data)}")
    
    device = torch.device(args.device)
    
    # Find matching configurations
    print("\nFinding parameter-matched configurations...")
    configs = find_matching_configs(len(predicates), len(args_vocab), 
                                     target_params=None, embed_dim=args.embed_dim)
    
    if len(configs) < 3:
        print("  Warning: Could only find {} matching configs".format(len(configs)))
    
    # Pick 3 configs at different scales
    selected_configs = [configs[0], configs[len(configs)//2], configs[-1]]
    
    print("\nSelected configurations:")
    for i, cfg in enumerate(selected_configs):
        print(f"  Config {i+1}: ~{cfg['avg_params']:.0f} params")
        print(f"    Transformer: {cfg['transformer']['num_layers']} layers, {cfg['transformer']['params']:,} params")
        print(f"    DLN:         {cfg['dln']['num_rules']} rules, {cfg['dln']['params']:,} params")
    
    # Train all models
    results = []
    colors = [('#FF6B6B', '#FF9999'), ('#FFA500', '#FFD700'), ('#51CF66', '#90EE90')]
    
    for i, cfg in enumerate(selected_configs):
        print(f"\n{'='*70}")
        print(f"Config {i+1}: ~{cfg['avg_params']:.0f} parameters")
        print('='*70)
        
        # Train Transformer
        print(f"\nTraining Transformer ({cfg['transformer']['num_layers']} layers)...")
        trans = TransformerLogic(len(predicates), len(args_vocab), 
                                 args.embed_dim, cfg['transformer']['num_layers'])
        trans_history = train_and_track(trans, train_data, args.epochs, 
                                        0.001, 32, device)
        
        results.append({
            'name': f"Transformer-{cfg['transformer']['num_layers']}L",
            'params': cfg['transformer']['params'],
            'history': trans_history,
            'color': colors[i][0],
            'final_acc': trans_history['accuracy'][-1]
        })
        
        # Train DLN
        print(f"\nTraining DLN ({cfg['dln']['num_rules']} rules)...")
        dln = DLNLogic(len(predicates), len(args_vocab),
                       args.embed_dim, cfg['dln']['num_rules'])
        dln_history = train_and_track(dln, train_data, args.epochs,
                                      0.001, 32, device)
        
        results.append({
            'name': f"DLN-{cfg['dln']['num_rules']}R",
            'params': cfg['dln']['params'],
            'history': dln_history,
            'color': colors[i][1],
            'final_acc': dln_history['accuracy'][-1]
        })
    
    # Plot convergence
    plot_convergence(results)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<20} {'Parameters':<15} {'Final Acc':<12} {'Epochs to 60%'}")
    print("-" * 70)
    
    for r in results:
        # Find epoch where accuracy reaches 60%
        epochs_to_60 = next((e for e, acc in zip(r['history']['epoch'], r['history']['accuracy']) 
                            if acc >= 60.0), args.epochs)
        print(f"{r['name']:<20} {r['params']:<15,} {r['final_acc']:>10.1f}% {epochs_to_60:>12}")
    
    print("\n" + "=" * 70)
    print("Key findings:")
    print("1. Models with similar parameter counts compared")
    print("2. Convergence speed shows architectural efficiency")
    print("3. Final accuracy shows learning capacity")
    print("=" * 70)


if __name__ == "__main__":
    main()
