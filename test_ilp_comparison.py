#!/usr/bin/env python3
"""
Test and compare different ILP algorithms for rule discovery.

Usage:
    python test_ilp_comparison.py [--max-stories N] [--max-rules N]
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pipelines.tinystories_pipeline import load_tinystories_facts
from core.ilp_algorithms import compare_algorithms, mine_frequency_based, mine_foil_style, mine_confidence_based
from core.rule_tracker import RuleTracker
from label_utils import _collect_labels
from logic_core import SymbolicEngine
from core.train_utils import _train_on_labels, _eval_on_labels
from dln import SimpleDLN
import torch


def test_single_algorithm(algorithm_name: str, facts, max_rules: int = 20, min_support: int = 2):
    """Test a single ILP algorithm and report results."""
    print(f"\n{'='*70}")
    print(f"TESTING: {algorithm_name.upper()}")
    print(f"{'='*70}")
    
    # Mine rules
    if algorithm_name == 'frequency':
        rules, pred_names = mine_frequency_based(facts, max_rules, min_support)
    elif algorithm_name == 'foil':
        rules, pred_names = mine_foil_style(facts, max_rules, min_support)
    elif algorithm_name == 'confidence':
        rules, pred_names = mine_confidence_based(facts, max_rules, min_support, min_confidence=0.3)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    print(f"\nMined {len(rules)} rules")
    
    # Show some example rules
    print(f"\nSample rules:")
    for i, rule in enumerate(rules[:5]):
        print(f"  {i+1}. {rule}")
    if len(rules) > 5:
        print(f"  ... and {len(rules) - 5} more")
    
    # Setup DLN
    print(f"\n[setup] Creating DLN with {len(pred_names)} predicates...")
    # Collect all entities from facts
    all_args = set()
    for fact in facts[:5000]:
        all_args.update(fact.args)
    arg_list = sorted(list(all_args))
    
    model = SimpleDLN(
        predicates=pred_names,
        args=arg_list,
        embed_dim=32,
    )
    
    # Generate labels
    print(f"[label generation] Generating training labels...")
    labels_dict = _collect_labels(facts[:5000], rules, log_progress=False)
    # Convert dict to list of tuples
    labels = [(pred_args, truth) for pred_args, truth in labels_dict.items()]
    print(f"  Generated {len(labels)} labels")
    
    if len(labels) == 0:
        print("⚠️  No labels generated! Rules may not fire on this data.")
        return None
    
    # Split train/eval
    split_idx = int(0.8 * len(labels))
    train_labels = labels[:split_idx]
    eval_labels = labels[split_idx:]
    
    # Train
    print(f"[training] Training DLN for 20 steps on {len(train_labels)} labels...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Convert labels back to dict format
    train_labels_dict = {pred_args: truth for pred_args, truth in train_labels}
    train_mse = _train_on_labels(model, optimizer, facts[:5000], train_labels_dict, steps=20, batch_size=None)
    print(f"  Final train MSE: {train_mse:.6f}")
    
    # Evaluate
    print(f"[evaluation] Evaluating on {len(eval_labels)} labels...")
    eval_labels_dict = {pred_args: truth for pred_args, truth in eval_labels}
    eval_mse, eval_mae = _eval_on_labels(model, facts[:5000], eval_labels_dict)
    print(f"  Eval MSE: {eval_mse:.6f}, MAE: {eval_mae:.6f}")
    
    return {
        'algorithm': algorithm_name,
        'num_rules': len(rules),
        'num_labels': len(labels),
        'train_mse': train_mse,
        'eval_mse': eval_mse,
        'eval_mae': eval_mae,
    }


def main():
    parser = argparse.ArgumentParser(description='Compare ILP algorithms')
    parser.add_argument('--max-stories', type=int, default=50,
                       help='Max stories to load for testing (default: 50)')
    parser.add_argument('--max-rules', type=int, default=20,
                       help='Max rules per algorithm (default: 20)')
    parser.add_argument('--min-support', type=int, default=2,
                       help='Minimum support for rules (default: 2)')
    parser.add_argument('--algorithm', type=str, choices=['frequency', 'foil', 'confidence', 'all'],
                       default='all', help='Which algorithm to test (default: all)')
    args = parser.parse_args()
    
    print("="*70)
    print("ILP ALGORITHM COMPARISON TEST")
    print("="*70)
    print(f"Loading {args.max_stories} stories...")
    
    # Load facts
    facts = load_tinystories_facts(max_stories=args.max_stories)
    print(f"Loaded {len(facts)} facts from stories")
    
    # Compare algorithms first
    if args.algorithm == 'all':
        print("\n" + "="*70)
        print("PHASE 1: MINING COMPARISON")
        print("="*70)
        results = compare_algorithms(facts, max_rules=args.max_rules, min_support=args.min_support)
        
        print("\n" + "="*70)
        print("PHASE 2: TRAINING COMPARISON")
        print("="*70)
        
        # Test each algorithm
        all_results = []
        for alg_name in ['frequency', 'foil', 'confidence']:
            result = test_single_algorithm(alg_name, facts, max_rules=args.max_rules, min_support=args.min_support)
            if result:
                all_results.append(result)
        
        # Summary
        print("\n" + "="*70)
        print("SUMMARY COMPARISON")
        print("="*70)
        print(f"\n{'Algorithm':<15} {'Rules':<8} {'Labels':<10} {'Train MSE':<12} {'Eval MSE':<12} {'Eval MAE':<12}")
        print("-" * 70)
        for r in all_results:
            print(f"{r['algorithm']:<15} {r['num_rules']:<8} {r['num_labels']:<10} "
                  f"{r['train_mse']:<12.6f} {r['eval_mse']:<12.6f} {r['eval_mae']:<12.6f}")
        
        # Determine winner
        best_eval = min(all_results, key=lambda x: x['eval_mse'])
        print(f"\n🏆 Best performing algorithm: {best_eval['algorithm'].upper()}")
        print(f"   Eval MSE: {best_eval['eval_mse']:.6f}, MAE: {best_eval['eval_mae']:.6f}")
        
    else:
        # Test single algorithm
        result = test_single_algorithm(args.algorithm, facts, max_rules=args.max_rules, min_support=args.min_support)
        if result:
            print(f"\n✅ {args.algorithm.upper()} results:")
            print(f"   Rules: {result['num_rules']}")
            print(f"   Labels: {result['num_labels']}")
            print(f"   Train MSE: {result['train_mse']:.6f}")
            print(f"   Eval MSE: {result['eval_mse']:.6f}, MAE: {result['eval_mae']:.6f}")


if __name__ == '__main__':
    main()
