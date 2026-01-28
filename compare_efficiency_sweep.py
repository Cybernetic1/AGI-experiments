#!/usr/bin/env python3
"""
Parameter Sweep: Real DLN vs Transformer
=========================================

Vary model size (# rules for DLN, # layers for Transformer)
Plot: Parameters vs Test Accuracy
Shows architectural efficiency at different scales
"""

import torch
import torch.nn as nn
import json
from pathlib import Path
import argparse
import random
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

from dln import SimpleDLN
from logic_core import Proposition


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
            rel = fact.get('relation', '')
            subj = fact.get('subject', '')
            obj = fact.get('object', '')
            if rel and subj and obj:
                all_facts.append(Proposition(rel, [subj, obj]))
    
    return all_facts


def build_vocabularies(facts):
    """Build predicate and argument vocabularies."""
    predicates = set()
    args = set()
    
    for prop in facts:
        predicates.add(prop.predicate)
        for arg in prop.args:
            args.add(arg)
    
    return sorted(predicates), sorted(args)


def create_train_data(facts, negative_ratio=0.5):
    """Create training examples."""
    examples = []
    
    if len(facts) < 3:
        return examples
    
    # Positive examples
    for i in range(len(facts)):
        conclusion = facts[i]
        premises = random.sample([f for j, f in enumerate(facts) if j != i], 
                                min(3, len(facts) - 1))
        examples.append((premises, conclusion, 1.0))
    
    # Negative examples
    predicates = list(set(f.predicate for f in facts))
    all_args = list(set(arg for f in facts for arg in f.args))
    
    num_negatives = int(len(examples) * negative_ratio)
    for _ in range(num_negatives):
        pred = random.choice(predicates)
        arg1 = random.choice(all_args)
        arg2 = random.choice(all_args)
        fake_conclusion = Proposition(pred, [arg1, arg2])
        
        is_real = any(
            f.predicate == pred and 
            len(f.args) >= 2 and
            f.args[0] == arg1 and 
            f.args[1] == arg2
            for f in facts
        )
        
        if not is_real:
            premises = random.sample(facts, min(3, len(facts)))
            examples.append((premises, fake_conclusion, 0.0))
    
    return examples


class TransformerLogic(nn.Module):
    """Transformer baseline."""
    
    def __init__(self, predicates, args, embed_dim=32, num_layers=2):
        super().__init__()
        
        self.pred_vocab = {p: i for i, p in enumerate(predicates)}
        self.arg_vocab = {a: i for i, a in enumerate(args)}
        
        self.pred_embed = nn.Embedding(len(predicates), embed_dim)
        self.arg_embed = nn.Embedding(len(args), embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim * 3,
            nhead=max(1, (embed_dim * 3) // 16),
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.output = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
    
    def encode_prop(self, prop: Proposition):
        device = self.pred_embed.weight.device
        pred_idx = self.pred_vocab.get(prop.predicate, 0)
        pred_emb = self.pred_embed(torch.tensor([pred_idx], device=device))
        
        arg_embs = []
        for i in range(2):
            arg = prop.args[i] if i < len(prop.args) else list(self.arg_vocab.keys())[0]
            arg_idx = self.arg_vocab.get(arg, 0)
            arg_embs.append(self.arg_embed(torch.tensor([arg_idx], device=device)))
        
        return torch.cat([pred_emb] + arg_embs, dim=-1)
    
    def forward(self, premises: List[Proposition], conclusion: Proposition):
        prop_embs = [self.encode_prop(p) for p in premises]
        prop_embs.append(self.encode_prop(conclusion))
        
        seq = torch.cat(prop_embs, dim=0).unsqueeze(0)
        transformed = self.transformer(seq)
        out = transformed[0, -1, :]
        prob = self.output(out.unsqueeze(0)).squeeze()
        
        return prob


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(model, train_data, test_data, epochs, lr, device, verbose=False):
    """Train model and return test accuracy."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    for epoch in range(epochs):
        model.train()
        random.shuffle(train_data)
        
        for premises, conclusion, label in train_data:
            optimizer.zero_grad()
            
            pred = model(premises, conclusion)
            if pred.dim() > 0:
                pred = pred.squeeze()
            
            target = torch.tensor(label, dtype=torch.float32, device=device)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
    
    # Evaluate on test
    model.eval()
    correct = 0
    
    with torch.no_grad():
        for premises, conclusion, label in test_data:
            pred = model(premises, conclusion)
            if pred.dim() > 0:
                pred = pred.squeeze()
            
            correct += int((pred.item() > 0.5) == (label > 0.5))
    
    test_acc = correct / len(test_data) * 100 if len(test_data) > 0 else 0
    
    if verbose:
        print(f"  Test accuracy: {test_acc:.1f}%")
    
    return test_acc


def run_sweep(predicates, args, train_data, test_data, epochs, lr, device):
    """Run parameter sweep for both architectures."""
    
    results = {
        'dln': {'params': [], 'accuracy': []},
        'transformer': {'params': [], 'accuracy': []}
    }
    
    # DLN sweep: SimpleDLN has fixed architecture, params depend on vocab
    # We can't vary "num_rules" in SimpleDLN - it doesn't have that parameter
    # So we'll just test the one configuration
    print("\nTesting Real DLN...")
    dln = SimpleDLN(predicates, args, embed_dim=32)
    dln_params = count_parameters(dln)
    dln_acc = train_model(dln, train_data, test_data, epochs, lr, device, verbose=True)
    
    results['dln']['params'].append(dln_params)
    results['dln']['accuracy'].append(dln_acc)
    
    print(f"  DLN: {dln_params:,} params → {dln_acc:.1f}% accuracy")
    
    # Transformer sweep: vary num_layers
    print("\nTesting Transformers with different layers...")
    for num_layers in [1, 2, 3, 4, 5]:
        print(f"\n  Transformer ({num_layers} layers)...")
        trans = TransformerLogic(predicates, args, embed_dim=32, num_layers=num_layers)
        trans_params = count_parameters(trans)
        trans_acc = train_model(trans, train_data, test_data, epochs, lr, device, verbose=True)
        
        results['transformer']['params'].append(trans_params)
        results['transformer']['accuracy'].append(trans_acc)
        
        print(f"    {trans_params:,} params → {trans_acc:.1f}% accuracy")
    
    return results


def plot_results(results, output_path="docs/efficiency_comparison.png"):
    """Plot parameter efficiency comparison."""
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot DLN
    ax.scatter(results['dln']['params'], results['dln']['accuracy'],
              s=200, color='#51CF66', marker='o', label='Genifer DLN',
              edgecolors='black', linewidth=2, zorder=3)
    
    # Plot Transformers
    ax.scatter(results['transformer']['params'], results['transformer']['accuracy'],
              s=100, color='#FF6B6B', marker='s', label='Transformer Baseline',
              edgecolors='black', linewidth=1, alpha=0.7)
    
    # Connect transformer points
    ax.plot(results['transformer']['params'], results['transformer']['accuracy'],
           color='#FF6B6B', linestyle='--', alpha=0.5)
    
    # Annotations
    for i, (p, a) in enumerate(zip(results['dln']['params'], results['dln']['accuracy'])):
        ax.annotate(f'DLN\n{p/1000:.1f}K\n{a:.1f}%',
                   xy=(p, a), xytext=(10, 10), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='#51CF66', alpha=0.3))
    
    for i, (p, a) in enumerate(zip(results['transformer']['params'], 
                                    results['transformer']['accuracy'])):
        layers = i + 1
        ax.annotate(f'{layers}L\n{p/1000:.0f}K',
                   xy=(p, a), xytext=(5, -15), textcoords='offset points',
                   fontsize=8, alpha=0.8)
    
    ax.set_xlabel('Parameters', fontsize=13, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Parameter Efficiency: Real DLN vs Transformer\n(Logical Inference Task)', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stories", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    
    print("=" * 70)
    print("PARAMETER EFFICIENCY SWEEP: Real DLN vs Transformer")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading {args.stories} stories...")
    facts = load_facts(max_stories=args.stories)
    print(f"  Loaded {len(facts)} facts")
    
    if len(facts) < 10:
        print("Not enough facts for meaningful comparison")
        return
    
    predicates, arg_vocab = build_vocabularies(facts)
    print(f"  Predicates: {len(predicates)}, Arguments: {len(arg_vocab)}")
    
    # Create dataset
    all_data = create_train_data(facts, negative_ratio=0.5)
    random.shuffle(all_data)
    
    split_idx = int(0.8 * len(all_data))
    train_data = all_data[:split_idx]
    test_data = all_data[split_idx:]
    
    print(f"  Train: {len(train_data)}, Test: {len(test_data)}")
    
    device = torch.device(args.device)
    
    # Run sweep
    results = run_sweep(predicates, arg_vocab, train_data, test_data,
                       args.epochs, args.lr, device)
    
    # Plot
    plot_results(results)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nReal DLN:")
    for p, a in zip(results['dln']['params'], results['dln']['accuracy']):
        print(f"  {p:,} params → {a:.1f}% test accuracy")
    
    print("\nTransformer (varying layers):")
    for i, (p, a) in enumerate(zip(results['transformer']['params'], 
                                    results['transformer']['accuracy'])):
        print(f"  {i+1} layers: {p:,} params → {a:.1f}% test accuracy")
    
    print("\nKey Finding:")
    dln_p = results['dln']['params'][0]
    dln_a = results['dln']['accuracy'][0]
    
    # Find transformer with closest accuracy
    best_trans_idx = min(range(len(results['transformer']['accuracy'])),
                         key=lambda i: abs(results['transformer']['accuracy'][i] - dln_a))
    trans_p = results['transformer']['params'][best_trans_idx]
    trans_a = results['transformer']['accuracy'][best_trans_idx]
    
    if trans_p > dln_p:
        compression = trans_p / dln_p
        print(f"  At ~{dln_a:.0f}% accuracy: DLN uses {compression:.1f}× fewer parameters")
        print(f"  ({dln_p:,} vs {trans_p:,})")
    else:
        print(f"  DLN achieves {dln_a:.1f}% with {dln_p:,} params")
        print(f"  Transformer needs {best_trans_idx+1} layers for similar performance")


if __name__ == "__main__":
    main()
