#!/usr/bin/env python3
"""
REAL DLN COMPARISON: Using actual SimpleDLN from dln.py
========================================================

Test the real DLN architecture with premise-conclusion structure
against Transformer baseline on same task.
"""

import torch
import torch.nn as nn
import json
from pathlib import Path
import argparse
import random
import matplotlib.pyplot as plt
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
    """
    Create training examples: Given N-1 facts as premises, predict truth of Nth fact.
    Also create negative examples with shuffled arguments.
    """
    examples = []
    
    # Positive examples: use random subsets as premises, predict another fact
    if len(facts) < 3:
        return examples
    
    for i in range(len(facts)):
        # Use some facts as premises, predict another
        conclusion = facts[i]
        premises = random.sample([f for j, f in enumerate(facts) if j != i], 
                                min(3, len(facts) - 1))
        examples.append((premises, conclusion, 1.0))
    
    # Negative examples: shuffle arguments
    predicates = list(set(f.predicate for f in facts))
    all_args = list(set(arg for f in facts for arg in f.args))
    
    num_negatives = int(len(examples) * negative_ratio)
    for _ in range(num_negatives):
        pred = random.choice(predicates)
        arg1 = random.choice(all_args)
        arg2 = random.choice(all_args)
        fake_conclusion = Proposition(pred, [arg1, arg2])
        
        # Check it's not real
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
    """Transformer baseline matching SimpleDLN input format."""
    
    def __init__(self, predicates, args, embed_dim=32, num_layers=2):
        super().__init__()
        
        self.pred_vocab = {p: i for i, p in enumerate(predicates)}
        self.arg_vocab = {a: i for i, a in enumerate(args)}
        
        self.pred_embed = nn.Embedding(len(predicates), embed_dim)
        self.arg_embed = nn.Embedding(len(args), embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim * 3,  # [pred, arg1, arg2]
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
        """Encode proposition to tensor."""
        device = self.pred_embed.weight.device
        pred_idx = self.pred_vocab.get(prop.predicate, 0)
        pred_emb = self.pred_embed(torch.tensor([pred_idx], device=device))
        
        arg_embs = []
        for i in range(2):
            arg = prop.args[i] if i < len(prop.args) else args[0]
            arg_idx = self.arg_vocab.get(arg, 0)
            arg_embs.append(self.arg_embed(torch.tensor([arg_idx], device=device)))
        
        return torch.cat([pred_emb] + arg_embs, dim=-1)  # (1, 3*embed_dim)
    
    def forward(self, premises: List[Proposition], conclusion: Proposition):
        """
        Args:
            premises: List of premise propositions
            conclusion: Conclusion proposition to verify
        Returns:
            prob: Probability that conclusion is true given premises
        """
        # Encode all propositions
        prop_embs = [self.encode_prop(p) for p in premises]
        prop_embs.append(self.encode_prop(conclusion))
        
        # Stack as sequence
        seq = torch.cat(prop_embs, dim=0).unsqueeze(0)  # (1, num_props, 3*embed_dim)
        
        # Transform
        transformed = self.transformer(seq)  # (1, num_props, 3*embed_dim)
        
        # Use last position (conclusion) for prediction
        out = transformed[0, -1, :]  # (3*embed_dim,)
        prob = self.output(out.unsqueeze(0)).squeeze()  # scalar
        
        return prob


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_and_track(model, train_data, epochs, lr, device, model_name):
    """Train model and track convergence."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    history = {'epoch': [], 'loss': [], 'accuracy': []}
    
    print(f"\nTraining {model_name}...")
    
    for epoch in range(epochs):
        model.train()
        random.shuffle(train_data)
        
        total_loss = 0
        correct = 0
        total = 0
        
        for premises, conclusion, label in train_data:
            optimizer.zero_grad()
            
            pred = model(premises, conclusion)
            # Ensure pred is scalar
            if pred.dim() > 0:
                pred = pred.squeeze()
            
            target = torch.tensor(label, dtype=torch.float32, device=device)
            
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            correct += int((pred.item() > 0.5) == (label > 0.5))
            total += 1
        
        avg_loss = total_loss / len(train_data)
        acc = correct / total * 100
        
        history['epoch'].append(epoch + 1)
        history['loss'].append(avg_loss)
        history['accuracy'].append(acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}: Loss = {avg_loss:.4f}, Acc = {acc:.1f}%")
    
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stories", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    
    print("=" * 70)
    print("REAL DLN vs Transformer Comparison")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading {args.stories} stories...")
    facts = load_facts(max_stories=args.stories)
    print(f"  Loaded {len(facts)} facts")
    
    if len(facts) < 10:
        print("Not enough facts for meaningful comparison")
        return
    
    # Build vocabularies
    predicates, arg_vocab = build_vocabularies(facts)
    print(f"  Predicates: {len(predicates)}")
    print(f"  Arguments: {len(arg_vocab)}")
    
    # Create training data
    print("\nCreating train/test splits...")
    all_data = create_train_data(facts, negative_ratio=0.5)
    random.shuffle(all_data)
    
    split_idx = int(0.8 * len(all_data))
    train_data = all_data[:split_idx]
    test_data = all_data[split_idx:]
    
    print(f"  Train examples: {len(train_data)}")
    print(f"  Test examples: {len(test_data)}")
    
    device = torch.device(args.device)
    
    # Create models
    print("\nCreating models...")
    dln = SimpleDLN(predicates, arg_vocab, embed_dim=args.embed_dim)
    transformer = TransformerLogic(predicates, arg_vocab, 
                                   embed_dim=args.embed_dim, num_layers=2)
    
    # Count parameters
    dln_params = count_parameters(dln)
    trans_params = count_parameters(transformer)
    
    print(f"\nModel Sizes:")
    print(f"  Real DLN:    {dln_params:,} parameters")
    print(f"  Transformer: {trans_params:,} parameters")
    print(f"  Ratio:       {trans_params / dln_params:.2f}×")
    
    # Train models
    dln_history = train_and_track(dln, train_data, args.epochs, args.lr, device, "Real DLN")
    trans_history = train_and_track(transformer, train_data, args.epochs, args.lr, device, "Transformer")
    
    # Evaluate on test set
    print("\n" + "=" * 70)
    print("TEST SET EVALUATION")
    print("=" * 70)
    
    dln.eval()
    transformer.eval()
    
    dln_correct = 0
    trans_correct = 0
    
    with torch.no_grad():
        for premises, conclusion, label in test_data:
            dln_pred = dln(premises, conclusion).squeeze()
            trans_pred = transformer(premises, conclusion)
            if trans_pred.dim() > 0:
                trans_pred = trans_pred.squeeze()
            
            dln_correct += int((dln_pred.item() > 0.5) == (label > 0.5))
            trans_correct += int((trans_pred.item() > 0.5) == (label > 0.5))
    
    dln_test_acc = dln_correct / len(test_data) * 100
    trans_test_acc = trans_correct / len(test_data) * 100
    
    print(f"\nReal DLN:    {dln_test_acc:.1f}% test accuracy")
    print(f"Transformer: {trans_test_acc:.1f}% test accuracy")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<20} {'Parameters':<15} {'Train Acc':<12} {'Test Acc'}")
    print("-" * 70)
    print(f"{'Real DLN':<20} {dln_params:<15,} {dln_history['accuracy'][-1]:>10.1f}% {dln_test_acc:>10.1f}%")
    print(f"{'Transformer':<20} {trans_params:<15,} {trans_history['accuracy'][-1]:>10.1f}% {trans_test_acc:>10.1f}%")
    
    if dln_params < trans_params:
        print(f"\n✓ Real DLN uses {trans_params/dln_params:.1f}× fewer parameters")
    
    if dln_test_acc >= trans_test_acc * 0.95:
        print(f"✓ Real DLN achieves comparable accuracy")
    else:
        print(f"• DLN needs improvement (gap: {trans_test_acc - dln_test_acc:.1f}%)")


if __name__ == "__main__":
    main()
