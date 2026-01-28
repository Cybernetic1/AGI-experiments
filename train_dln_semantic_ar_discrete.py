#!/usr/bin/env python3
"""
Train Real DLN with Semantic-AR - DISCRETE PREDICTION VERSION
==============================================================

The DLN should predict actual next logic form (discrete tokens),
not just smooth embeddings.

Task: Given current sentence's logic, predict next sentence's:
  - Relations (type, agent, patient, etc.)
  - Entities (actual entity names)
  
This is much harder and tests real reasoning capability.
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

from neural_logic_core import LogicNetwork
from davidsonian_extraction import DavidsonianExtractor


class DLNSemanticARDiscrete(nn.Module):
    """
    DLN that predicts discrete next logic form.
    """
    
    def __init__(self, num_rules=6, embed_dim=16):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Vocabularies
        self.relations = ['type', 'agent', 'patient', 'recipient', 'manner', 
                         'location', 'instrument', 'time', 'tense', 'theme', 'goal', 'quantifier']
        self.relation_to_idx = {r: i for i, r in enumerate(self.relations)}
        
        self.entity_vocab = {}
        self.idx_to_entity = {}
        self.next_entity_idx = 0
        self.max_entities = 500
        
        # Embeddings
        self.relation_embed = nn.Embedding(len(self.relations) + 1, embed_dim)
        self.entity_embed = nn.Embedding(self.max_entities, embed_dim)
        
        self.prop_length = embed_dim * 3
        
        # DLN encoder (with 2 variable slots for efficiency)
        self.dln = LogicNetwork(
            prop_length=self.prop_length,
            num_props=10,
            output_dim=embed_dim * 2,  # Compressed representation (reduced from *4)
            num_rules=num_rules,
            num_premises=3,
            var_slots=2  # Reduced from 4 to minimize parameters
        )
        
        # Prediction heads for next propositions
        # Predict up to 8 propositions in next sentence
        self.num_output_props = 8
        
        # For each output proposition, predict relation and 2 entities
        hidden_dim = embed_dim * 2  # Matches DLN output (reduced from *4)
        self.relation_head = nn.Linear(hidden_dim, len(self.relations) + 1)
        self.entity1_head = nn.Linear(hidden_dim, self.max_entities)
        self.entity2_head = nn.Linear(hidden_dim, self.max_entities)
        self.num_props_head = nn.Linear(hidden_dim, self.num_output_props + 1)  # How many props
    
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
        """Encode proposition."""
        rel_idx = self.relation_to_idx.get(relation, len(self.relation_to_idx))
        ent_idx = self.get_entity_idx(entity)
        val_idx = self.get_entity_idx(value)
        
        rel_emb = self.relation_embed(torch.tensor([rel_idx], device=device))
        ent_emb = self.entity_embed(torch.tensor([ent_idx], device=device))
        val_emb = self.entity_embed(torch.tensor([val_idx], device=device))
        
        return torch.cat([rel_emb, ent_emb, val_emb], dim=-1).squeeze(0)
    
    def forward(self, current_logic):
        """
        Predict next logic form.
        Supports both single example and batched inputs.
        
        Args:
            current_logic: List[Tuple] for single, or List[List[Tuple]] for batch
        
        Returns:
            num_props_logits, relation_logits, entity1_logits, entity2_logits
        """
        device = next(self.parameters()).device
        
        # Check if batched
        is_batch = isinstance(current_logic[0], list) if current_logic else False
        
        if not is_batch:
            current_logic = [current_logic]  # Make it a batch of 1
        
        batch_size = len(current_logic)
        
        # Encode entire batch at once
        batch_wm = []
        for logic in current_logic:
            if len(logic) == 0:
                prop_vecs = torch.zeros(10, self.prop_length, device=device)
            else:
                prop_vecs = []
                for entity, relation, value in logic[:10]:
                    prop_vecs.append(self.encode_proposition(entity, relation, value, device))
                prop_vecs = torch.stack(prop_vecs)
                # Pad to 10
                if len(prop_vecs) < 10:
                    padding = torch.zeros(10 - len(prop_vecs), self.prop_length, device=device)
                    prop_vecs = torch.cat([prop_vecs, padding], dim=0)
            batch_wm.append(prop_vecs)
        
        working_memory = torch.stack(batch_wm)  # (batch_size, 10, prop_length)
        
        # Process through DLN (batched!)
        repr = self.dln(working_memory)  # (batch_size, embed_dim * 2)
        
        # Predict (batched!)
        num_props_logits = self.num_props_head(repr)  # (batch_size, num_output_props+1)
        relation_logits = self.relation_head(repr)    # (batch_size, num_relations+1)
        entity1_logits = self.entity1_head(repr)      # (batch_size, max_entities)
        entity2_logits = self.entity2_head(repr)      # (batch_size, max_entities)
        
        if not is_batch:
            # Return single example
            return (num_props_logits[0], relation_logits[0], 
                    entity1_logits[0], entity2_logits[0])
        
        return num_props_logits, relation_logits, entity1_logits, entity2_logits


def collate_fn(batch):
    """Custom collate."""
    return [item[0] for item in batch], [item[1] for item in batch]


class TinyStoriesDataset(Dataset):
    """Dataset for semantic AR."""
    
    def __init__(self, data_path, max_stories=200):
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        self.examples = []
        extractor = DavidsonianExtractor()
        
        print(f"Extracting semantic forms from {max_stories} stories...")
        for story in tqdm(data[:max_stories]):
            text = story.get('text', '')
            if not text:
                continue
            
            sentences = [s.strip() + '.' for s in text.replace('\n', ' ').split('.') if s.strip()]
            
            logic_forms = []
            for sent in sentences:
                logic = extractor.extract(sent)
                if logic:
                    logic_forms.append(logic)
            
            for i in range(len(logic_forms) - 1):
                self.examples.append((logic_forms[i], logic_forms[i+1]))
        
        print(f"  Created {len(self.examples)} examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def train_epoch(model, dataloader, optimizer, device):
    """Train with discrete prediction - TRUE BATCHING."""
    model.train()
    total_loss = 0
    num_updates = 0
    
    for curr_batch, next_batch in dataloader:
        # Process entire batch at once!
        optimizer.zero_grad()
        
        # Batched forward pass
        num_props_logits, rel_logits, ent1_logits, ent2_logits = model(curr_batch)
        # Shape: (batch_size, num_classes)
        
        # Prepare targets for entire batch
        batch_size = len(curr_batch)
        target_rels = []
        target_ent1s = []
        target_ent2s = []
        target_nums = []
        
        for next_logic in next_batch:
            if len(next_logic) > 0:
                target_entity, target_rel, target_value = next_logic[0]
                target_rel_idx = model.relation_to_idx.get(target_rel, len(model.relation_to_idx))
                target_ent1_idx = model.get_entity_idx(target_entity)
                target_ent2_idx = model.get_entity_idx(target_value)
                target_num = min(len(next_logic), model.num_output_props)
            else:
                target_rel_idx = len(model.relation_to_idx)
                target_ent1_idx = 0
                target_ent2_idx = 0
                target_num = 0
            
            target_rels.append(target_rel_idx)
            target_ent1s.append(target_ent1_idx)
            target_ent2s.append(target_ent2_idx)
            target_nums.append(target_num)
        
        # Convert to tensors
        target_rels = torch.tensor(target_rels, device=device)
        target_ent1s = torch.tensor(target_ent1s, device=device)
        target_ent2s = torch.tensor(target_ent2s, device=device)
        target_nums = torch.tensor(target_nums, device=device)
        
        # Batched loss computation
        loss = (F.cross_entropy(rel_logits, target_rels) +
               F.cross_entropy(ent1_logits, target_ent1s) +
               F.cross_entropy(ent2_logits, target_ent2s) +
               F.cross_entropy(num_props_logits, target_nums))
        
        if not torch.isnan(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_updates += 1
    
    return total_loss / num_updates if num_updates > 0 else 0.0


def evaluate(model, dataloader, device):
    """Evaluate discrete predictions - BATCHED."""
    model.eval()
    
    correct_rel = 0
    correct_ent1 = 0
    correct_ent2 = 0
    total = 0
    
    with torch.no_grad():
        for curr_batch, next_batch in dataloader:
            # Batched forward pass
            _, rel_logits, ent1_logits, ent2_logits = model(curr_batch)
            # Shape: (batch_size, num_classes)
            
            # Get predictions for entire batch
            pred_rels = rel_logits.argmax(dim=1).cpu().numpy()
            pred_ent1s = ent1_logits.argmax(dim=1).cpu().numpy()
            pred_ent2s = ent2_logits.argmax(dim=1).cpu().numpy()
            
            # Compare with targets
            for i, next_logic in enumerate(next_batch):
                if len(next_logic) == 0:
                    continue
                
                target_entity, target_rel, target_value = next_logic[0]
                target_rel_idx = model.relation_to_idx.get(target_rel, len(model.relation_to_idx))
                target_ent1_idx = model.get_entity_idx(target_entity)
                target_ent2_idx = model.get_entity_idx(target_value)
                
                if pred_rels[i] == target_rel_idx:
                    correct_rel += 1
                if pred_ent1s[i] == target_ent1_idx:
                    correct_ent1 += 1
                if pred_ent2s[i] == target_ent2_idx:
                    correct_ent2 += 1
                
                total += 1
    
    return {
        'relation_acc': correct_rel / total * 100 if total > 0 else 0,
        'entity1_acc': correct_ent1 / total * 100 if total > 0 else 0,
        'entity2_acc': correct_ent2 / total * 100 if total > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stories", type=int, default=200)
    parser.add_argument("--num-rules", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=20)  # Reduced from 50
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--early-stop-patience", type=int, default=5)  # Stop if no improvement for 5 epochs
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    print("="*70)
    print("TRAIN REAL DLN - DISCRETE PREDICTION")
    print("="*70)
    
    # Load dataset
    dataset = TinyStoriesDataset("data/processed/tinystories_train.json", max_stories=args.stories)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=32, collate_fn=collate_fn)
    
    print(f"  Train: {len(train_set)}, Test: {len(test_set)}")
    
    # Create model
    model = DLNSemanticARDiscrete(num_rules=args.num_rules, embed_dim=16)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    dln_params = sum(p.numel() for p in model.dln.parameters())
    print(f"\n  Total params: {total_params:,}")
    print(f"  DLN params: {dln_params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training
    print("\n" + "="*70)
    print("TRAINING (with early stopping)")
    print("="*70)
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, optimizer, device)
        
        # Early stopping check
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0:
            metrics = evaluate(model, test_loader, device)
            print(f"Epoch {epoch+1:3d}: Loss={loss:.4f}, Rel={metrics['relation_acc']:.1f}%, "
                  f"Ent1={metrics['entity1_acc']:.1f}%, Ent2={metrics['entity2_acc']:.1f}%")
        else:
            print(f"Epoch {epoch+1:3d}: Loss={loss:.4f}")
        
        # Early stopping
        if patience_counter >= args.early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {args.early_stop_patience} epochs)")
            break
    
    # Final eval
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    
    final_metrics = evaluate(model, test_loader, device)
    print(f"\nRelation accuracy: {final_metrics['relation_acc']:.1f}%")
    print(f"Entity1 accuracy:  {final_metrics['entity1_acc']:.1f}%")
    print(f"Entity2 accuracy:  {final_metrics['entity2_acc']:.1f}%")
    
    print(f"\nRandom baselines:")
    print(f"  Relations: {100.0/len(model.relations):.1f}%")
    print(f"  Entities: {100.0/model.next_entity_idx:.1f}%")


if __name__ == "__main__":
    main()
