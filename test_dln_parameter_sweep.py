#!/usr/bin/env python3
"""
Parameter sweep: Test DLN with varying number of rules.
Create data for presentation graph.
"""

import torch
from train_dln_semantic_ar_discrete import (
    DLNSemanticARDiscrete, TinyStoriesDataset, collate_fn, 
    train_epoch, evaluate
)
from torch.utils.data import DataLoader
import argparse
import json


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def test_dln_with_rules(num_rules, dataset, device, epochs=30, lr=0.001):
    """Test DLN with specific number of rules."""
    
    train_size = int(0.8 * len(dataset))
    train_set, test_set = torch.utils.data.random_split(
        dataset, [train_size, len(dataset) - train_size]
    )
    
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=8, collate_fn=collate_fn)
    
    # Create model
    model = DLNSemanticARDiscrete(num_rules=num_rules, embed_dim=16).to(device)
    params = count_parameters(model)
    
    print(f"\n{'='*70}")
    print(f"DLN with {num_rules} rules ({params:,} params)")
    print('='*70)
    
    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        loss = train_epoch(model, train_loader, optimizer, device)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Loss={loss:.4f}")
    
    # Evaluate
    metrics = evaluate(model, test_loader, device)
    
    print(f"\nResults:")
    print(f"  Relation: {metrics['relation_acc']:.1f}%")
    print(f"  Entity1:  {metrics['entity1_acc']:.1f}%")
    print(f"  Entity2:  {metrics['entity2_acc']:.1f}%")
    
    # Composite score (weighted average)
    composite = (metrics['relation_acc'] + metrics['entity1_acc'] + metrics['entity2_acc']) / 3
    
    return {
        'num_rules': num_rules,
        'params': params,
        'relation_acc': metrics['relation_acc'],
        'entity1_acc': metrics['entity1_acc'],
        'entity2_acc': metrics['entity2_acc'],
        'composite_score': composite
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stories", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    print("="*70)
    print("DLN PARAMETER SWEEP")
    print("="*70)
    print(f"\nDataset: {args.stories} stories")
    print(f"Training: {args.epochs} epochs")
    
    # Load dataset once
    print("\nLoading dataset...")
    dataset = TinyStoriesDataset("data/processed/tinystories_train.json", max_stories=args.stories)
    
    # Test with different numbers of rules
    rule_counts = [2, 4, 6, 8, 12]
    results = []
    
    for num_rules in rule_counts:
        result = test_dln_with_rules(num_rules, dataset, device, epochs=args.epochs)
        results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\n{'Rules':<8} {'Params':<12} {'Relation':<10} {'Entity1':<10} {'Entity2':<10} {'Composite':<10}")
    print("-"*70)
    
    for r in results:
        print(f"{r['num_rules']:<8} {r['params']:<12,} "
              f"{r['relation_acc']:<10.1f} {r['entity1_acc']:<10.1f} "
              f"{r['entity2_acc']:<10.1f} {r['composite_score']:<10.1f}")
    
    # Save results
    with open('dln_parameter_sweep_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to dln_parameter_sweep_results.json")
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    # Find best composite score per parameter budget
    best = max(results, key=lambda x: x['composite_score'])
    print(f"\nBest performance: {best['num_rules']} rules")
    print(f"  Parameters: {best['params']:,}")
    print(f"  Composite score: {best['composite_score']:.1f}%")
    
    # Parameter efficiency
    print("\nParameter efficiency (composite score per 10K params):")
    for r in results:
        efficiency = r['composite_score'] / (r['params'] / 10000)
        print(f"  {r['num_rules']} rules: {efficiency:.2f}")


if __name__ == "__main__":
    main()
