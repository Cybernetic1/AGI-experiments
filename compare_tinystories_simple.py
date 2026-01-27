#!/usr/bin/env python3
"""
FAIR COMPARISON: DLN vs Transformer on TinyStories
===================================================

Simplest possible test using your existing data:
- Task: Predict proposition truth values from story facts
- Data: Already preprocessed TinyStories with facts
- No ILP, no GA, no rule injection - just neural training
- Compare at matched accuracy

This is the most honest comparison we can make.
"""

import torch
import torch.nn as nn
import json
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
import random


def load_facts(max_stories=50):
    """Load facts from preprocessed TinyStories."""
    data_path = Path("data/processed/tinystories_train.json")
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found. Run preprocessing first.")
    
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
    """
    Create training examples: (predicate, arg1, arg2) -> 1 (true) or 0 (false)
    Add negative examples by shuffling arguments.
    """
    examples = []
    
    # Positive examples (actual facts)
    for rel, subj, obj in facts:
        pred_id = predicates.get(rel, 0)
        arg1_id = args.get(subj, 0)
        arg2_id = args.get(obj, 0)
        if pred_id > 0 and arg1_id > 0 and arg2_id > 0:
            examples.append(([pred_id, arg1_id, arg2_id], 1.0))
    
    # Negative examples (randomly shuffle arguments)
    num_negatives = int(len(examples) * negative_ratio)
    pred_ids = list(predicates.values())[1:]  # Skip PAD
    arg_ids = list(args.values())[1:]
    
    for _ in range(num_negatives):
        pred_id = random.choice(pred_ids)
        arg1_id = random.choice(arg_ids)
        arg2_id = random.choice(arg_ids)
        triple = (pred_id, arg1_id, arg2_id)
        
        # Check it's not a real fact
        is_real = any(
            predicates.get(rel, 0) == pred_id and
            args.get(subj, 0) == arg1_id and
            args.get(obj, 0) == arg2_id
            for rel, subj, obj in facts
        )
        if not is_real:
            examples.append(([pred_id, arg1_id, arg2_id], 0.0))
    
    return examples


class SimpleTransformerLogic(nn.Module):
    """Baseline Transformer for logical inference."""
    
    def __init__(self, num_predicates, num_args, embed_dim=32):
        super().__init__()
        
        self.pred_embed = nn.Embedding(num_predicates, embed_dim)
        self.arg_embed = nn.Embedding(num_args, embed_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=2,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Output: binary classification
        self.output = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, triple):
        """
        Args:
            triple: (batch, 3) - [pred_id, arg1_id, arg2_id]
        Returns:
            probs: (batch,) - probability of being true
        """
        pred_emb = self.pred_embed(triple[:, 0:1])  # (batch, 1, embed_dim)
        arg1_emb = self.arg_embed(triple[:, 1:2])
        arg2_emb = self.arg_embed(triple[:, 2:3])
        
        # Concatenate as sequence
        x = torch.cat([pred_emb, arg1_emb, arg2_emb], dim=1)  # (batch, 3, embed_dim)
        
        # Transform
        x = self.transformer(x)  # (batch, 3, embed_dim)
        
        # Pool and predict
        x = x.mean(dim=1)  # (batch, embed_dim)
        prob = self.output(x).squeeze(-1)  # (batch,)
        
        return prob


class SimpleDLNLogic(nn.Module):
    """DLN-style model with explicit embeddings only."""
    
    def __init__(self, num_predicates, num_args, embed_dim=32):
        super().__init__()
        
        self.pred_embed = nn.Embedding(num_predicates, embed_dim)
        self.arg_embed = nn.Embedding(num_args, embed_dim)
        
        # Simple MLP to combine
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, triple):
        """
        Args:
            triple: (batch, 3)
        Returns:
            probs: (batch,)
        """
        pred_emb = self.pred_embed(triple[:, 0])  # (batch, embed_dim)
        arg1_emb = self.arg_embed(triple[:, 1])
        arg2_emb = self.arg_embed(triple[:, 2])
        
        # Concatenate and predict
        x = torch.cat([pred_emb, arg1_emb, arg2_emb], dim=-1)  # (batch, 3*embed_dim)
        prob = self.mlp(x).squeeze(-1)  # (batch,)
        
        return prob


def train_model(model, train_data, epochs=50, lr=0.001, batch_size=32, device='cpu', model_name="Model"):
    """Train model on logical inference."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    print(f"\nTraining {model_name}...")
    
    # Shuffle and batch
    random.shuffle(train_data)
    
    for epoch in range(epochs):
        model.train()
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
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}: Loss = {avg_loss:.4f}, Acc = {acc:.1f}%")
    
    return model


def evaluate_model(model, test_data, batch_size=32, device='cpu'):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i+batch_size]
            if len(batch) == 0:
                continue
            
            X = torch.tensor([x[0] for x in batch], dtype=torch.long, device=device)
            y = torch.tensor([x[1] for x in batch], dtype=torch.float32, device=device)
            
            pred = model(X)
            correct += ((pred > 0.5) == (y > 0.5)).sum().item()
            total += len(batch)
    
    return correct / total * 100 if total > 0 else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stories", type=int, default=50, help="Number of stories")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    print("=" * 70)
    print("FAIR COMPARISON: Transformer vs DLN on TinyStories Facts")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading {args.stories} stories...")
    facts = load_facts(max_stories=args.stories)
    print(f"  Loaded {len(facts)} facts")
    
    # Build vocabularies
    predicates, args_vocab = build_vocabularies(facts)
    print(f"  Predicates: {len(predicates)}")
    print(f"  Arguments: {len(args_vocab)}")
    
    # Create training data
    print("\nCreating train/test splits...")
    all_data = create_train_data(facts, predicates, args_vocab, negative_ratio=0.5)
    random.shuffle(all_data)
    
    split_idx = int(0.8 * len(all_data))
    train_data = all_data[:split_idx]
    test_data = all_data[split_idx:]
    
    print(f"  Train examples: {len(train_data)}")
    print(f"  Test examples: {len(test_data)}")
    
    device = torch.device(args.device)
    
    # Create models
    transformer = SimpleTransformerLogic(len(predicates), len(args_vocab), 
                                         embed_dim=args.embed_dim).to(device)
    dln = SimpleDLNLogic(len(predicates), len(args_vocab),
                         embed_dim=args.embed_dim).to(device)
    
    # Count parameters
    trans_params = sum(p.numel() for p in transformer.parameters())
    dln_params = sum(p.numel() for p in dln.parameters())
    
    print(f"\nModel Sizes:")
    print(f"  Transformer: {trans_params:,} parameters")
    print(f"  DLN:         {dln_params:,} parameters")
    print(f"  Ratio:       {trans_params / dln_params:.2f}×")
    
    # Train models
    transformer = train_model(transformer, train_data, epochs=args.epochs,
                             device=device, model_name="Transformer")
    dln = train_model(dln, train_data, epochs=args.epochs,
                     device=device, model_name="DLN")
    
    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)
    
    trans_acc = evaluate_model(transformer, test_data, device=device)
    dln_acc = evaluate_model(dln, test_data, device=device)
    
    print(f"\nTransformer: {trans_acc:.1f}% accuracy ({trans_params:,} params)")
    print(f"DLN:         {dln_acc:.1f}% accuracy ({dln_params:,} params)")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY FOR PRESENTATION")
    print("=" * 70)
    print(f"\n{'Model':<15} {'Parameters':<15} {'Accuracy':<12} {'Compression'}")
    print("-" * 70)
    print(f"{'Transformer':<15} {trans_params:<15,} {trans_acc:>10.1f}% {'baseline'}")
    
    if dln_params < trans_params:
        compression = trans_params / dln_params
        print(f"{'DLN':<15} {dln_params:<15,} {dln_acc:>10.1f}% {compression:.1f}×")
        
        print(f"\n✓ At {args.stories} stories ({len(predicates)} predicates, {len(args_vocab)} entities):")
        print(f"  DLN achieves {dln_acc:.1f}% accuracy with {compression:.1f}× fewer parameters")
    else:
        print(f"{'DLN':<15} {dln_params:<15,} {dln_acc:>10.1f}% larger")


if __name__ == "__main__":
    main()
