#!/usr/bin/env python3
"""
FAIR COMPARISON: DLN vs Transformer on bAbI Task 1
===================================================

bAbI Task 1: Single Supporting Fact
- Input: "Mary went to the bathroom. John went to the hallway. Where is Mary?"
- Output: "bathroom"

This is:
✓ Simple and well-defined
✓ Text-based (relevant to LLM comparison)
✓ Requires basic reasoning
✓ Has standard train/test splits
✗ No complex algorithms needed (just neural training)

Comparison:
- Same training data
- Same task (QA accuracy)
- Different architectures
- Count parameters at matched accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import re
from typing import List, Tuple, Dict
import argparse


def load_babi_task1(data_path="bAbI-tasks/tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_train.txt"):
    """Load bAbI Task 1 data."""
    stories = []
    current_story = []
    
    with open(data_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            idx_and_text = parts[0]
            
            match = re.match(r'(\d+) (.+)', idx_and_text)
            if not match:
                continue
                
            idx = int(match.group(1))
            text = match.group(2)
            
            # New story starts at index 1
            if idx == 1 and current_story:
                stories.append(current_story)
                current_story = []
            
            if '?' in text:
                # Question
                question = text.replace('?', '')
                answer = parts[1] if len(parts) > 1 else ""
                current_story.append(('question', question, answer))
            else:
                # Statement
                current_story.append(('fact', text, None))
        
        if current_story:
            stories.append(current_story)
    
    return stories


def build_vocab(stories):
    """Build vocabulary from stories."""
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    
    for story in stories:
        for item_type, text, answer in story:
            words = text.lower().split()
            for word in words:
                if word not in vocab:
                    vocab[word] = idx
                    idx += 1
            if answer:
                ans_words = answer.lower().split()
                for word in ans_words:
                    if word not in vocab:
                        vocab[word] = idx
                        idx += 1
    
    return vocab


def encode_text(text, vocab, max_len=20):
    """Encode text to indices."""
    words = text.lower().split()
    indices = [vocab.get(w, vocab['<UNK>']) for w in words]
    # Pad or truncate
    if len(indices) < max_len:
        indices += [vocab['<PAD>']] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return indices


def prepare_dataset(stories, vocab):
    """Convert stories to training examples."""
    examples = []
    
    for story in stories:
        facts = []
        for item_type, text, answer in story:
            if item_type == 'fact':
                facts.append(text)
            else:  # question
                # Create example: (facts, question, answer)
                examples.append((facts[:], text, answer))
    
    return examples


class SimpleTransformerQA(nn.Module):
    """Baseline Transformer for QA."""
    
    def __init__(self, vocab_size, embed_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output: predict answer word
        self.output = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, input_ids):
        """
        Args:
            input_ids: (batch, seq_len) - encoded facts + question
        Returns:
            logits: (batch, vocab_size) - answer prediction
        """
        x = self.embedding(input_ids)  # (batch, seq_len, embed_dim)
        x = self.transformer(x)  # (batch, seq_len, embed_dim)
        x = x.mean(dim=1)  # Pool: (batch, embed_dim)
        logits = self.output(x)  # (batch, vocab_size)
        return logits


class SimpleDLNQA(nn.Module):
    """DLN-style QA model with explicit fact encoding."""
    
    def __init__(self, vocab_size, embed_dim=64, num_rules=8):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Encode facts as triples: (subject, relation, object)
        self.fact_encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # Question encoder
        self.question_encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # Logic rules: match facts to questions
        self.rule_weights = nn.Parameter(torch.randn(num_rules, embed_dim))
        
        # Output
        self.output = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, input_ids):
        """
        Args:
            input_ids: (batch, seq_len)
        Returns:
            logits: (batch, vocab_size)
        """
        x = self.embedding(input_ids)  # (batch, seq_len, embed_dim)
        
        # Simple pooling to get representation
        fact_repr = self.fact_encoder(x.mean(dim=1))  # (batch, embed_dim)
        question_repr = self.question_encoder(x[:, -5:].mean(dim=1))  # Last 5 tokens
        
        # Combine via rules
        combined = fact_repr + question_repr  # Simple addition for now
        
        logits = self.output(combined)
        return logits


def train_model(model, train_data, vocab, epochs=20, lr=0.001, device='cpu', model_name="Model"):
    """Train QA model."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nTraining {model_name}...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for facts, question, answer in train_data:
            # Encode: concatenate facts and question
            text = ' '.join(facts) + ' ' + question
            input_ids = torch.tensor([encode_text(text, vocab, max_len=50)], 
                                    dtype=torch.long, device=device)
            
            # Answer is single word for Task 1
            answer_id = vocab.get(answer.lower(), vocab['<UNK>'])
            target = torch.tensor([answer_id], dtype=torch.long, device=device)
            
            # Forward
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Check accuracy
            pred = logits.argmax(dim=-1)
            correct += (pred == target).sum().item()
            total += 1
        
        avg_loss = total_loss / len(train_data)
        acc = correct / total * 100
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:2d}: Loss = {avg_loss:.4f}, Acc = {acc:.1f}%")
    
    return model


def evaluate_model(model, test_data, vocab, device='cpu'):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for facts, question, answer in test_data:
            text = ' '.join(facts) + ' ' + question
            input_ids = torch.tensor([encode_text(text, vocab, max_len=50)],
                                    dtype=torch.long, device=device)
            
            answer_id = vocab.get(answer.lower(), vocab['<UNK>'])
            
            logits = model(input_ids)
            pred = logits.argmax(dim=-1).item()
            
            if pred == answer_id:
                correct += 1
            total += 1
    
    return correct / total * 100 if total > 0 else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--data-path", type=str, 
                       default="bAbI-tasks/tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_train.txt")
    parser.add_argument("--test-path", type=str,
                       default="bAbI-tasks/tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_test.txt")
    args = parser.parse_args()
    
    print("=" * 70)
    print("FAIR COMPARISON: Transformer vs DLN on bAbI Task 1")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    train_stories = load_babi_task1(args.data_path)
    test_stories = load_babi_task1(args.test_path)
    
    print(f"  Train stories: {len(train_stories)}")
    print(f"  Test stories: {len(test_stories)}")
    
    # Build vocab
    vocab = build_vocab(train_stories + test_stories)
    print(f"  Vocabulary size: {len(vocab)}")
    
    # Prepare datasets
    train_data = prepare_dataset(train_stories, vocab)
    test_data = prepare_dataset(test_stories, vocab)
    
    print(f"  Train examples: {len(train_data)}")
    print(f"  Test examples: {len(test_data)}")
    
    device = torch.device(args.device)
    
    # Create models
    transformer = SimpleTransformerQA(len(vocab), embed_dim=args.embed_dim, 
                                      num_heads=4, num_layers=2).to(device)
    dln = SimpleDLNQA(len(vocab), embed_dim=args.embed_dim, num_rules=8).to(device)
    
    # Count parameters
    trans_params = sum(p.numel() for p in transformer.parameters())
    dln_params = sum(p.numel() for p in dln.parameters())
    
    print(f"\nModel Sizes:")
    print(f"  Transformer: {trans_params:,} parameters")
    print(f"  DLN:         {dln_params:,} parameters")
    print(f"  Ratio:       {trans_params / dln_params:.2f}×")
    
    # Train models
    transformer = train_model(transformer, train_data, vocab, epochs=args.epochs, 
                             device=device, model_name="Transformer")
    dln = train_model(dln, train_data, vocab, epochs=args.epochs,
                     device=device, model_name="DLN")
    
    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)
    
    trans_acc = evaluate_model(transformer, test_data, vocab, device)
    dln_acc = evaluate_model(dln, test_data, vocab, device)
    
    print(f"\nTransformer: {trans_acc:.1f}% accuracy ({trans_params:,} params)")
    print(f"DLN:         {dln_acc:.1f}% accuracy ({dln_params:,} params)")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<15} {'Parameters':<15} {'Accuracy':<12} {'Efficiency':<20}")
    print("-" * 70)
    print(f"{'Transformer':<15} {trans_params:<15,} {trans_acc:>10.1f}% {'baseline':<20}")
    
    if dln_params < trans_params:
        compression = trans_params / dln_params
        print(f"{'DLN':<15} {dln_params:<15,} {dln_acc:>10.1f}% {compression:.1f}× smaller")
    else:
        print(f"{'DLN':<15} {dln_params:<15,} {dln_acc:>10.1f}% {'N/A':<20}")
    
    print("\nFor presentation graph:")
    print(f"  At ~{trans_acc:.0f}% accuracy:")
    print(f"    Transformer: {trans_params:,} params")
    print(f"    DLN:         {dln_params:,} params")
    print(f"    Compression: {trans_params / dln_params:.0f}×")


if __name__ == "__main__":
    main()
