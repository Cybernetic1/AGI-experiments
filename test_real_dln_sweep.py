#!/usr/bin/env python3
"""
REAL DLN COMPARISON: Using neural_logic_core.py with cylindrification
======================================================================

Test the actual LogicNetwork with variable num_rules against Transformer.
"""

import torch
import torch.nn as nn
import json
from pathlib import Path
import argparse
import random
import matplotlib.pyplot as plt
from typing import List

from neural_logic_core import LogicNetwork
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


def encode_proposition(prop, pred_vocab, arg_vocab, device='cpu'):
    """
    Encode proposition as a vector.
    Simple one-hot encoding for now.
    """
    pred_idx = pred_vocab.get(prop.predicate, 0)
    arg1_idx = arg_vocab.get(prop.args[0] if len(prop.args) > 0 else "", 0)
    arg2_idx = arg_vocab.get(prop.args[1] if len(prop.args) > 1 else "", 0)
    
    # Create simple encoding: [pred_onehot, arg1_onehot, arg2_onehot]
    vec = torch.zeros(len(pred_vocab) + 2 * len(arg_vocab), device=device)
    vec[pred_idx] = 1.0
    vec[len(pred_vocab) + arg1_idx] = 1.0
    vec[len(pred_vocab) + len(arg_vocab) + arg2_idx] = 1.0
    
    return vec


def create_train_data(facts, pred_vocab, arg_vocab, negative_ratio=0.5):
    """Create training examples: premises + conclusion → label."""
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


class RealDLNWrapper(nn.Module):
    """Wrapper around LogicNetwork to match our task format."""
    
    def __init__(self, pred_vocab, arg_vocab, num_rules=8, num_premises=3, var_slots=4):
        super().__init__()
        
        self.pred_vocab = pred_vocab
        self.arg_vocab = arg_vocab
        
        # Proposition encoding length
        self.prop_length = len(pred_vocab) + 2 * len(arg_vocab)
        
        # Real DLN with cylindrification
        self.dln = LogicNetwork(
            prop_length=self.prop_length,
            num_props=num_premises + 1,  # premises + conclusion
            output_dim=1,  # binary classification
            num_rules=num_rules,
            num_premises=num_premises,
            var_slots=var_slots
        )
        
        self.output_layer = nn.Sigmoid()
    
    def forward(self, premises: List[Proposition], conclusion: Proposition):
        """
        Args:
            premises: List of premise propositions
            conclusion: Conclusion proposition to verify
        Returns:
            prob: Probability that conclusion is true
        """
        device = next(self.parameters()).device
        
        # Encode all propositions on correct device
        prop_vecs = []
        for p in premises:
            prop_vecs.append(encode_proposition(p, self.pred_vocab, self.arg_vocab, device))
        prop_vecs.append(encode_proposition(conclusion, self.pred_vocab, self.arg_vocab, device))
        
        # Stack as working memory: (1, num_props, prop_length)
        working_memory = torch.stack(prop_vecs).unsqueeze(0)
        
        # Pass through DLN
        output = self.dln(working_memory)  # (1, 1)
        prob = self.output_layer(output).squeeze()
        
        return prob


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


def run_sweep(predicates, args, pred_vocab, arg_vocab, train_data, test_data, epochs, lr, device):
    """Run parameter sweep for both architectures."""
    
    results = {
        'dln': {'params': [], 'accuracy': [], 'config': []},
        'transformer': {'params': [], 'accuracy': [], 'config': []}
    }
    
    # Real DLN sweep: vary num_rules
    print("\nTesting Real DLN (with cylindrification) - varying num_rules...")
    for num_rules in [2, 4, 8, 16, 32]:
        print(f"\n  DLN ({num_rules} rules)...")
        dln = RealDLNWrapper(pred_vocab, arg_vocab, num_rules=num_rules, 
                             num_premises=3, var_slots=4)
        dln_params = count_parameters(dln)
        dln_acc = train_model(dln, train_data, test_data, epochs, lr, device, verbose=True)
        
        results['dln']['params'].append(dln_params)
        results['dln']['accuracy'].append(dln_acc)
        results['dln']['config'].append(f'{num_rules}R')
        
        print(f"    {dln_params:,} params → {dln_acc:.1f}% accuracy")
    
    # Transformer sweep: vary num_layers
    print("\nTesting Transformers - varying layers...")
    for num_layers in [1, 2, 3, 4, 5]:
        print(f"\n  Transformer ({num_layers} layers)...")
        trans = TransformerLogic(predicates, args, embed_dim=32, num_layers=num_layers)
        trans_params = count_parameters(trans)
        trans_acc = train_model(trans, train_data, test_data, epochs, lr, device, verbose=True)
        
        results['transformer']['params'].append(trans_params)
        results['transformer']['accuracy'].append(trans_acc)
        results['transformer']['config'].append(f'{num_layers}L')
        
        print(f"    {trans_params:,} params → {trans_acc:.1f}% accuracy")
    
    return results


def plot_results(results, output_path="docs/real_dln_efficiency.png"):
    """Plot parameter efficiency comparison."""
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot both curves
    ax.plot(results['dln']['params'], results['dln']['accuracy'],
           'o-', color='#51CF66', linewidth=2, markersize=10,
           label='Real DLN (with cylindrification)', markeredgecolor='black', markeredgewidth=2)
    
    ax.plot(results['transformer']['params'], results['transformer']['accuracy'],
           's--', color='#FF6B6B', linewidth=2, markersize=8,
           label='Transformer Baseline', markeredgecolor='black', markeredgewidth=1, alpha=0.7)
    
    # Annotate points
    for p, a, cfg in zip(results['dln']['params'], results['dln']['accuracy'], results['dln']['config']):
        ax.annotate(f'{cfg}\n{p/1000:.0f}K',
                   xy=(p, a), xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold', color='#2d7a3e')
    
    for p, a, cfg in zip(results['transformer']['params'], results['transformer']['accuracy'], 
                         results['transformer']['config']):
        ax.annotate(f'{cfg}\n{p/1000:.0f}K',
                   xy=(p, a), xytext=(5, -15), textcoords='offset points',
                   fontsize=8, color='#c92a2a')
    
    ax.set_xlabel('Parameters', fontsize=13, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Parameter Efficiency: Real DLN (Cylindrification) vs Transformer\n(Logical Inference on TinyStories)', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
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
    print("REAL DLN (Cylindrification) vs Transformer - Parameter Sweep")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading {args.stories} stories...")
    facts = load_facts(max_stories=args.stories)
    print(f"  Loaded {len(facts)} facts")
    
    if len(facts) < 10:
        print("Not enough facts")
        return
    
    predicates, arg_vocab_list = build_vocabularies(facts)
    pred_vocab = {p: i for i, p in enumerate(predicates)}
    arg_vocab = {a: i for i, a in enumerate(arg_vocab_list)}
    
    print(f"  Predicates: {len(predicates)}, Arguments: {len(arg_vocab)}")
    
    # Create dataset
    all_data = create_train_data(facts, pred_vocab, arg_vocab, negative_ratio=0.5)
    random.shuffle(all_data)
    
    split_idx = int(0.8 * len(all_data))
    train_data = all_data[:split_idx]
    test_data = all_data[split_idx:]
    
    print(f"  Train: {len(train_data)}, Test: {len(test_data)}")
    
    device = torch.device(args.device)
    
    # Run sweep
    results = run_sweep(predicates, arg_vocab_list, pred_vocab, arg_vocab,
                       train_data, test_data, args.epochs, args.lr, device)
    
    # Plot
    plot_results(results)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nReal DLN (with cylindrification):")
    for p, a, cfg in zip(results['dln']['params'], results['dln']['accuracy'], results['dln']['config']):
        print(f"  {cfg}: {p:,} params → {a:.1f}% test accuracy")
    
    print("\nTransformer:")
    for p, a, cfg in zip(results['transformer']['params'], results['transformer']['accuracy'], 
                         results['transformer']['config']):
        print(f"  {cfg}: {p:,} params → {a:.1f}% test accuracy")


if __name__ == "__main__":
    main()
