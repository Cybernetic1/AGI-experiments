#!/usr/bin/env python3
"""
Train Real DLN with Unification Task
=====================================

Task: Given partial semantic forms with variables, predict the missing entities.
This exploits DLN's core strength: unification and logical reasoning.

Example:
  Input:  [('e1', 'type', 'walk'), ('e1', 'agent', '?X'), ('?X', 'type', 'person')]
  Output: ?X = 'girl' (or 'boy', 'man', etc.)

This is fundamentally different from sequential prediction - it requires
understanding logical constraints and variable binding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from typing import List, Tuple, Dict
import argparse
from tqdm import tqdm
import random

from neural_logic_core import LogicNetwork
from davidsonian_extraction import DavidsonianExtractor


class UnificationDataset(Dataset):
    """
    Create unification tasks from semantic forms.
    Each example: (partial_logic_with_variables, target_entity)
    """
    
    def __init__(self, stories_file, extractor, model, max_stories=None):
        self.examples = []
        self.extractor = extractor
        self.model = model
        
        # Load and process stories
        stories_path = Path(stories_file)
        if not stories_path.exists():
            # Try alternate location
            stories_path = Path('data/tinystories/stories_10000.txt')
        
        with open(stories_path) as f:
            stories = [line.strip() for line in f if line.strip()]
        
        if max_stories:
            stories = stories[:max_stories]
        
        print(f"Creating unification tasks from {len(stories)} stories...")
        
        for story in tqdm(stories):
            sentences = [s.strip() for s in story.split('.') if s.strip()]
            
            for sent in sentences:
                triples = extractor.extract(sent)
                if len(triples) < 2:
                    continue
                
                # Create multiple unification tasks per sentence
                entities = set()
                for subj, rel, obj in triples:
                    if not subj.startswith('e') and subj != 'pron_she' and subj != 'pron_he':
                        entities.add(subj)
                    if not obj.startswith('e') and obj != 'past' and obj != 'exists':
                        entities.add(obj)
                
                # For each entity, create a task to predict it
                for target_entity in entities:
                    # Replace target with variable
                    masked_triples = []
                    for subj, rel, obj in triples:
                        new_subj = '?X' if subj == target_entity else subj
                        new_obj = '?X' if obj == target_entity else obj
                        masked_triples.append((new_subj, rel, new_obj))
                    
                    # Only add if ?X appears (we masked something)
                    if any('?X' in str(t) for t in masked_triples):
                        self.examples.append((masked_triples, target_entity))
        
        print(f"  Created {len(self.examples)} unification tasks")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


class DLNUnification(nn.Module):
    """
    DLN model for unification tasks.
    Predicts entities that satisfy logical constraints.
    """
    
    def __init__(self, num_rules=6, embed_dim=16):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Vocabularies
        self.relations = ['type', 'agent', 'patient', 'recipient', 'manner', 
                         'location', 'instrument', 'time', 'tense', 'theme', 'goal', 'quantifier']
        self.relation_to_idx = {r: i for i, r in enumerate(self.relations)}
        
        self.entity_vocab = {'?X': 0}  # Variable placeholder
        self.idx_to_entity = {0: '?X'}
        self.next_entity_idx = 1
        self.max_entities = 500
        
        # Embeddings
        self.relation_embed = nn.Embedding(len(self.relations) + 1, embed_dim)
        self.entity_embed = nn.Embedding(self.max_entities, embed_dim)
        
        self.prop_length = embed_dim * 3
        
        # DLN (with 2 variable slots for unification)
        self.dln = LogicNetwork(
            prop_length=self.prop_length,
            num_props=10,
            output_dim=embed_dim * 2,
            num_rules=num_rules,
            num_premises=3,
            var_slots=2  # Perfect for unification with 2 vars
        )
        
        # Prediction head: which entity fills ?X
        hidden_dim = embed_dim * 2
        self.entity_head = nn.Linear(hidden_dim, self.max_entities)
    
    def get_entity_idx(self, entity):
        """Get or create entity index."""
        if entity not in self.entity_vocab:
            if self.next_entity_idx >= self.max_entities:
                return 0
            self.entity_vocab[entity] = self.next_entity_idx
            self.idx_to_entity[self.next_entity_idx] = entity
            self.next_entity_idx += 1
        return self.entity_vocab[entity]
    
    def encode_proposition(self, entity, relation, value, device):
        """Encode proposition as concatenated embeddings."""
        rel_idx = self.relation_to_idx.get(relation, len(self.relation_to_idx))
        ent_idx = self.get_entity_idx(entity)
        val_idx = self.get_entity_idx(value)
        
        rel_emb = self.relation_embed(torch.tensor([rel_idx], device=device))
        ent_emb = self.entity_embed(torch.tensor([ent_idx], device=device))
        val_emb = self.entity_embed(torch.tensor([val_idx], device=device))
        
        return torch.cat([rel_emb, ent_emb, val_emb], dim=-1).squeeze(0)
    
    def forward(self, logic_forms):
        """
        Args:
            logic_forms: List of lists of (subj, rel, obj) triples with ?X variables
        Returns:
            entity_logits: (batch_size, max_entities) - probabilities for each entity
        """
        batch_size = len(logic_forms)
        device = next(self.parameters()).device
        
        # Encode each example
        encoded_batch = []
        for triples in logic_forms:
            # Pad to fixed size
            props = []
            for subj, rel, obj in triples[:10]:  # Max 10 props
                props.append(self.encode_proposition(subj, rel, obj, device))
            
            # Pad to 10 props
            while len(props) < 10:
                props.append(torch.zeros(self.prop_length, device=device))
            
            encoded_batch.append(torch.stack(props))
        
        # Stack into batch tensor
        wm = torch.stack(encoded_batch)  # (batch, 10, prop_length)
        
        # Pass through DLN
        output = self.dln(wm)  # (batch, hidden_dim)
        
        # Predict entity
        entity_logits = self.entity_head(output)  # (batch, max_entities)
        
        return entity_logits


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for logic_forms, target_entities in dataloader:
        # Get target indices
        target_indices = []
        for ent in target_entities:
            idx = model.get_entity_idx(ent)
            target_indices.append(idx)
        target_indices = torch.tensor(target_indices, device=device)
        
        # Forward
        entity_logits = model(logic_forms)
        
        # Loss
        loss = criterion(entity_logits, target_indices)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Stats
        total_loss += loss.item()
        predictions = entity_logits.argmax(dim=1)
        correct += (predictions == target_indices).sum().item()
        total += len(target_entities)
    
    return total_loss / len(dataloader), correct / total * 100


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for logic_forms, target_entities in dataloader:
            target_indices = []
            for ent in target_entities:
                idx = model.get_entity_idx(ent)
                target_indices.append(idx)
            target_indices = torch.tensor(target_indices, device=device)
            
            entity_logits = model(logic_forms)
            predictions = entity_logits.argmax(dim=1)
            
            correct += (predictions == target_indices).sum().item()
            total += len(target_entities)
    
    # Random baseline
    num_entities = model.next_entity_idx
    random_acc = 100.0 / num_entities if num_entities > 0 else 0
    
    return correct / total * 100, random_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/TinyStories-train.txt')
    parser.add_argument('--num-stories', type=int, default=200)
    parser.add_argument('--num-rules', type=int, default=6)
    parser.add_argument('--embed-dim', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize
    extractor = DavidsonianExtractor()
    model = DLNUnification(num_rules=args.num_rules, embed_dim=args.embed_dim).to(device)
    
    # Create dataset
    dataset = UnificationDataset(args.data, extractor, model, max_stories=args.num_stories)
    
    # Split train/test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    print(f"\nDataset:")
    print(f"  Train: {len(train_dataset)} examples")
    print(f"  Test:  {len(test_dataset)} examples")
    print(f"  Vocabulary size: {model.next_entity_idx} entities")
    
    # Count parameters
    dln_params = sum(p.numel() for p in model.dln.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel:")
    print(f"  DLN parameters: {dln_params:,}")
    print(f"  Total parameters: {total_params:,}")
    
    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        if epoch % 5 == 0:
            test_acc, random_acc = evaluate(model, test_loader, device)
            print(f"Epoch {epoch:3d}: Loss={train_loss:.4f}, Train={train_acc:.1f}%, Test={test_acc:.1f}% (Random={random_acc:.1f}%)")
    
    # Final evaluation
    print("\n" + "="*60)
    test_acc, random_acc = evaluate(model, test_loader, device)
    print(f"FINAL RESULTS:")
    print(f"  Test Accuracy:   {test_acc:.1f}%")
    print(f"  Random Baseline: {random_acc:.1f}%")
    print(f"  Improvement:     {test_acc/random_acc:.1f}× better")
    print("="*60)


if __name__ == '__main__':
    main()
