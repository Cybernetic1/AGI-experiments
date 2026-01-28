#!/usr/bin/env python3
"""
Compare Real DLN vs Transformer on Semantic-AR task.
Both predict discrete next logic forms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from train_dln_semantic_ar_discrete import TinyStoriesDataset, collate_fn, DLNSemanticARDiscrete
from torch.utils.data import DataLoader
import argparse


class TransformerSemanticAR(nn.Module):
    """Transformer baseline for comparison."""
    
    def __init__(self, num_relations, num_entities, embed_dim=64, num_layers=2):
        super().__init__()
        
        self.num_relations = num_relations
        self.num_entities = num_entities
        self.embed_dim = embed_dim
        
        # Embeddings
        self.relation_embed = nn.Embedding(num_relations + 1, embed_dim)
        self.entity_embed = nn.Embedding(num_entities, embed_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim * 3,
            nhead=3,
            dim_feedforward=embed_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Prediction heads
        self.relation_head = nn.Linear(embed_dim * 3, num_relations + 1)
        self.entity1_head = nn.Linear(embed_dim * 3, num_entities)
        self.entity2_head = nn.Linear(embed_dim * 3, num_entities)
    
    def forward(self, logic_encoded):
        """
        Args:
            logic_encoded: (seq_len, embed_dim * 3) encoded propositions
        """
        if logic_encoded.size(0) == 0:
            device = next(self.parameters()).device
            dummy = torch.zeros(1, self.embed_dim * 3, device=device)
            repr = self.transformer(dummy.unsqueeze(0)).squeeze(0).mean(dim=0)
        else:
            # Process through transformer
            repr = self.transformer(logic_encoded.unsqueeze(0)).squeeze(0)
            repr = repr.mean(dim=0)  # Pool over sequence
        
        # Predict
        rel_logits = self.relation_head(repr)
        ent1_logits = self.entity1_head(repr)
        ent2_logits = self.entity2_head(repr)
        
        return None, rel_logits, ent1_logits, ent2_logits


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compare_models(args):
    """Compare DLN vs Transformer."""
    device = torch.device(args.device)
    
    print("="*70)
    print("DLN vs TRANSFORMER COMPARISON")
    print("="*70)
    
    # Load dataset
    dataset = TinyStoriesDataset("data/processed/tinystories_train.json", max_stories=args.stories)
    train_size = int(0.8 * len(dataset))
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=8, collate_fn=collate_fn)
    
    # Create DLN
    print("\n" + "="*70)
    print(f"TESTING DLN ({args.num_rules} rules)")
    print("="*70)
    
    dln_model = DLNSemanticARDiscrete(num_rules=args.num_rules, embed_dim=16).to(device)
    dln_params = count_parameters(dln_model)
    print(f"  Parameters: {dln_params:,}")
    
    # Train DLN
    from train_dln_semantic_ar_discrete import train_epoch, evaluate
    dln_optimizer = torch.optim.Adam(dln_model.parameters(), lr=args.lr)
    
    for epoch in range(args.epochs):
        loss = train_epoch(dln_model, train_loader, dln_optimizer, device)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Loss={loss:.4f}")
    
    dln_metrics = evaluate(dln_model, test_loader, device)
    
    print(f"\nDLN Final Results:")
    print(f"  Relation acc: {dln_metrics['relation_acc']:.1f}%")
    print(f"  Entity1 acc:  {dln_metrics['entity1_acc']:.1f}%")
    print(f"  Entity2 acc:  {dln_metrics['entity2_acc']:.1f}%")
    
    # Create Transformer with similar params
    print("\n" + "="*70)
    print("TESTING TRANSFORMER (matched parameters)")
    print("="*70)
    
    # Estimate num_layers to match DLN params
    # DLN has ~100K params, Transformer layer has ~40K each
    # So use 2-3 layers
    
    num_relations = len(dln_model.relations)
    num_entities = dln_model.next_entity_idx
    
    # Try different sizes
    for num_layers in [1, 2, 3]:
        trans_model = TransformerSemanticAR(
            num_relations=num_relations,
            num_entities=num_entities,
            embed_dim=64,
            num_layers=num_layers
        ).to(device)
        
        trans_params = count_parameters(trans_model)
        
        if abs(trans_params - dln_params) < dln_params * 0.5:  # Within 50%
            print(f"\nUsing {num_layers}-layer Transformer: {trans_params:,} params")
            break
    
    print(f"  DLN params:         {dln_params:,}")
    print(f"  Transformer params: {trans_params:,}")
    print(f"  Ratio: {trans_params/dln_params:.2f}×")
    
    # TODO: Train transformer (needs adaptation of training loop)
    print("\n  (Transformer training not implemented yet)")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nDLN ({dln_params:,} params):")
    print(f"  Relation: {dln_metrics['relation_acc']:.1f}%")
    print(f"  Entity1:  {dln_metrics['entity1_acc']:.1f}%")
    print(f"  Entity2:  {dln_metrics['entity2_acc']:.1f}%")
    
    print(f"\nTransformer ({trans_params:,} params):")
    print(f"  (To be implemented)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stories", type=int, default=200)
    parser.add_argument("--num-rules", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    compare_models(args)


if __name__ == "__main__":
    main()
