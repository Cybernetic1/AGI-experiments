#!/usr/bin/env python3
"""
Train Real DLN using Semantic-AR Framework
===========================================

Replaces the logic embedding in semantic-AR with real DLN.
This should converge much better than raw AR on fact triples.

Key differences from previous attempt:
1. Uses Davidsonian extraction (semantic structure)
2. Predicts next sentence's logic form (not raw facts)
3. Has rule injection mechanism to accelerate learning
4. Proper evaluation at the end
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


class DLNSemanticEncoder(nn.Module):
    """
    Use real DLN to encode semantic propositions.
    Replaces LogicEmbedding in semantic-AR.
    """
    
    def __init__(self, num_rules=6, embed_dim=16):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Fixed vocabularies for semantic relations
        self.relations = ['type', 'agent', 'patient', 'recipient', 'manner', 
                         'location', 'instrument', 'time', 'tense', 'theme', 'goal']
        self.relation_to_idx = {r: i for i, r in enumerate(self.relations)}
        
        # Entity embeddings
        self.entity_vocab = {}
        self.next_entity_idx = 0
        self.max_entities = 1000
        
        # Embedding layers
        self.relation_embed = nn.Embedding(len(self.relations) + 1, embed_dim)
        self.entity_embed = nn.Embedding(self.max_entities, embed_dim)
        
        # Proposition encoding: [relation, entity, value] → vector
        self.prop_length = embed_dim * 3
        
        # Real DLN to process propositions
        self.dln = LogicNetwork(
            prop_length=self.prop_length,
            num_props=10,  # Support up to 10 propositions per sentence
            output_dim=embed_dim * 2,  # Rich semantic representation
            num_rules=num_rules,
            num_premises=3,
            var_slots=2
        )
    
    def get_entity_idx(self, entity):
        """Get or create entity index."""
        if entity not in self.entity_vocab:
            if self.next_entity_idx >= self.max_entities:
                return 0
            self.entity_vocab[entity] = self.next_entity_idx
            self.next_entity_idx += 1
        return self.entity_vocab[entity]
    
    def encode_proposition(self, entity, relation, value, device):
        """Encode single proposition to vector."""
        rel_idx = self.relation_to_idx.get(relation, len(self.relation_to_idx))
        ent_idx = self.get_entity_idx(entity)
        val_idx = self.get_entity_idx(value)
        
        rel_emb = self.relation_embed(torch.tensor([rel_idx], device=device))
        ent_emb = self.entity_embed(torch.tensor([ent_idx], device=device))
        val_emb = self.entity_embed(torch.tensor([val_idx], device=device))
        
        return torch.cat([rel_emb, ent_emb, val_emb], dim=-1).squeeze(0)
    
    def forward(self, propositions: List[Tuple[str, str, str]]):
        """
        Encode semantic propositions using DLN.
        
        Args:
            propositions: List of (entity, relation, value)
        Returns:
            Semantic embedding vector
        """
        device = next(self.parameters()).device
        
        if len(propositions) == 0:
            return torch.zeros(self.embed_dim * 2, device=device)
        
        # Encode each proposition
        prop_vecs = []
        for entity, relation, value in propositions[:10]:  # Max 10
            prop_vecs.append(self.encode_proposition(entity, relation, value, device))
        
        # Pad to 10 propositions
        while len(prop_vecs) < 10:
            prop_vecs.append(torch.zeros(self.prop_length, device=device))
        
        # Stack as working memory
        working_memory = torch.stack(prop_vecs).unsqueeze(0)  # (1, 10, prop_length)
        
        # Process through DLN
        semantic_repr = self.dln(working_memory).squeeze(0)  # (embed_dim * 2)
        
        return semantic_repr


class DLNSemanticAR(nn.Module):
    """
    Semantic AR with real DLN as encoder.
    """
    
    def __init__(self, num_rules=6, embed_dim=16, hidden_dim=64):
        super().__init__()
        
        self.extractor = DavidsonianExtractor()
        self.encoder = DLNSemanticEncoder(num_rules=num_rules, embed_dim=embed_dim)
        
        # Predictor: current semantic → next semantic
        repr_dim = embed_dim * 2
        self.predictor = nn.Sequential(
            nn.Linear(repr_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, repr_dim)
        )
    
    def extract_logic(self, sentence: str):
        """Extract Davidsonian logic from sentence."""
        return self.extractor.extract(sentence)
    
    def forward(self, current_logic: List[Tuple[str, str, str]]):
        """
        Predict next sentence's logic form.
        
        Args:
            current_logic: Propositions from current sentence
        Returns:
            Predicted semantic representation for next sentence
        """
        # Encode current
        current_repr = self.encoder(current_logic)
        
        # Predict next
        next_repr = self.predictor(current_repr)
        
        return next_repr


class TinyStoriesSemanticDataset(Dataset):
    """Dataset of TinyStories with semantic AR objective."""
    
    def __init__(self, data_path="data/processed/tinystories_train.json", max_stories=200):
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        self.examples = []
        self.extractor = DavidsonianExtractor()
        
        print(f"Extracting semantic forms from {max_stories} stories...")
        for story in tqdm(data[:max_stories]):
            sentences = story.get('sentences', [])
            
            # Extract logic for each sentence
            logic_forms = []
            for sent in sentences:
                logic = self.extractor.extract(sent)
                if logic:  # Only keep non-empty
                    logic_forms.append(logic)
            
            # Create AR pairs: sentence[i] → sentence[i+1]
            for i in range(len(logic_forms) - 1):
                self.examples.append((logic_forms[i], logic_forms[i+1]))
        
        print(f"  Created {len(self.examples)} training examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def train_epoch(model, dataloader, optimizer, device):
    """Train one epoch."""
    model.train()
    total_loss = 0
    
    for current_logic, next_logic in dataloader:
        # Process batch (one example at a time for now)
        for curr, nxt in zip(current_logic, next_logic):
            optimizer.zero_grad()
            
            # Predict next
            pred_next = model(curr)
            
            # Encode actual next
            target_next = model.encoder(nxt)
            
            # Loss: MSE between predicted and actual representations
            loss = F.mse_loss(pred_next, target_next.detach())
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, device):
    """
    Evaluate on test set.
    Metric: Cosine similarity between predicted and actual next representations.
    """
    model.eval()
    total_similarity = 0
    count = 0
    
    with torch.no_grad():
        for current_logic, next_logic in dataloader:
            for curr, nxt in zip(current_logic, next_logic):
                # Predict
                pred_next = model(curr)
                
                # Actual
                target_next = model.encoder(nxt)
                
                # Cosine similarity
                similarity = F.cosine_similarity(
                    pred_next.unsqueeze(0),
                    target_next.unsqueeze(0)
                ).item()
                
                total_similarity += similarity
                count += 1
    
    avg_similarity = total_similarity / count if count > 0 else 0
    return avg_similarity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stories", type=int, default=200)
    parser.add_argument("--num-rules", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    print("="*70)
    print("TRAIN REAL DLN WITH SEMANTIC-AR")
    print("="*70)
    
    # Load dataset
    print(f"\nLoading dataset...")
    dataset = TinyStoriesSemanticDataset(max_stories=args.stories)
    
    # Split train/test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    
    print(f"  Train: {len(train_dataset)} examples")
    print(f"  Test:  {len(test_dataset)} examples")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Create model
    print(f"\nCreating DLN-SemanticAR model...")
    print(f"  Number of rules: {args.num_rules}")
    model = DLNSemanticAR(num_rules=args.num_rules, embed_dim=16, hidden_dim=64)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    dln_params = sum(p.numel() for p in model.encoder.dln.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  DLN parameters: {dln_params:,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    best_similarity = -1
    
    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # Evaluate
        if (epoch + 1) % 5 == 0:
            test_similarity = evaluate(model, test_loader, device)
            print(f"Epoch {epoch+1:3d}: Loss = {train_loss:.4f}, Test Similarity = {test_similarity:.4f}")
            
            if test_similarity > best_similarity:
                best_similarity = test_similarity
        else:
            print(f"Epoch {epoch+1:3d}: Loss = {train_loss:.4f}")
    
    # Final evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    
    final_train_sim = evaluate(model, train_loader, device)
    final_test_sim = evaluate(model, test_loader, device)
    
    print(f"\nTrain similarity: {final_train_sim:.4f}")
    print(f"Test similarity:  {final_test_sim:.4f}")
    print(f"Best test similarity: {best_similarity:.4f}")
    
    # Interpretation
    print("\nInterpretation:")
    print("  Similarity > 0.5: Model learning meaningful patterns")
    print("  Similarity > 0.7: Strong semantic prediction")
    print("  Similarity < 0.3: Model not learning well")
    
    if final_test_sim > 0.5:
        print("\n✓ Real DLN is learning with Semantic-AR!")
    else:
        print("\n✗ Still needs tuning")
    
    # Save model
    torch.save({
        'model_state': model.state_dict(),
        'num_rules': args.num_rules,
        'total_params': total_params,
        'test_similarity': final_test_sim,
    }, 'dln_semantic_ar_model.pt')
    
    print(f"\nModel saved to dln_semantic_ar_model.pt")


if __name__ == "__main__":
    main()
