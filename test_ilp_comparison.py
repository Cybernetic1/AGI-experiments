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
    # Collect all predicates (base + mined) and entities
    all_predicates = set(pred_names)  # Mined predicates
    all_args = set()
    for fact in facts[:5000]:
        all_predicates.add(fact.predicate)  # Base predicates from facts
        all_args.update(fact.args)
    
    pred_list = sorted(list(all_predicates))
    arg_list = sorted(list(all_args))
    
    print(f"\n[setup] Creating DLN with {len(pred_list)} predicates ({len(pred_names)} mined + base)...")
    model = SimpleDLN(
        predicates=pred_list,
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
    eval_mse = _eval_on_labels(model, facts[:5000], eval_labels_dict)
    print(f"  Eval MSE: {eval_mse:.6f}")
    
    return {
        'algorithm': algorithm_name,
        'num_rules': len(rules),
        'num_labels': len(labels),
        'train_mse': train_mse,
        'eval_mse': eval_mse,
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
    parser.add_argument('--num-runs', type=int, default=1,
                       help='Number of runs per algorithm (default: 1)')
    parser.add_argument('--random-seed', type=int, default=None,
                       help='Random seed for reproducibility (default: None)')
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.random_seed is not None:
        torch.manual_seed(args.random_seed)
        import random
        random.seed(args.random_seed)
    
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
        
        # Test each algorithm (multiple runs if requested)
        all_results = []
        for alg_name in ['frequency', 'foil', 'confidence']:
            if args.num_runs > 1:
                print(f"\n[info] Running {alg_name} {args.num_runs} times for statistical analysis...")
                run_results = []
                for run_idx in range(args.num_runs):
                    print(f"\n--- Run {run_idx + 1}/{args.num_runs} ---")
                    result = test_single_algorithm(alg_name, facts, max_rules=args.max_rules, min_support=args.min_support)
                    if result:
                        run_results.append(result)
                
                if run_results:
                    # Compute statistics across runs
                    avg_result = {
                        'algorithm': alg_name,
                        'num_rules': run_results[0]['num_rules'],
                        'num_labels': run_results[0]['num_labels'],
                        'train_mse': sum(r['train_mse'] for r in run_results) / len(run_results),
                        'eval_mse': sum(r['eval_mse'] for r in run_results) / len(run_results),
                        'train_mse_std': (sum((r['train_mse'] - sum(r['train_mse'] for r in run_results) / len(run_results))**2 for r in run_results) / len(run_results))**0.5,
                        'eval_mse_std': (sum((r['eval_mse'] - sum(r['eval_mse'] for r in run_results) / len(run_results))**2 for r in run_results) / len(run_results))**0.5,
                        'num_runs': len(run_results),
                    }
                    all_results.append(avg_result)
            else:
                result = test_single_algorithm(alg_name, facts, max_rules=args.max_rules, min_support=args.min_support)
                if result:
                    all_results.append(result)
        
        # Summary
        print("\n" + "="*70)
        print("SUMMARY COMPARISON")
        print("="*70)
        
        if args.num_runs > 1:
            print(f"\n{'Algorithm':<15} {'Rules':<8} {'Labels':<10} {'Train MSE':<20} {'Eval MSE':<20}")
            print("-" * 85)
            for r in all_results:
                train_str = f"{r['train_mse']:.4f} ± {r['train_mse_std']:.4f}" if 'train_mse_std' in r else f"{r['train_mse']:.6f}"
                eval_str = f"{r['eval_mse']:.4f} ± {r['eval_mse_std']:.4f}" if 'eval_mse_std' in r else f"{r['eval_mse']:.6f}"
                print(f"{r['algorithm']:<15} {r['num_rules']:<8} {r['num_labels']:<10} "
                      f"{train_str:<20} {eval_str:<20}")
            
            # Find most stable (lowest variance)
            most_stable = min(all_results, key=lambda x: x.get('eval_mse_std', float('inf')))
            print(f"\n⭐ Most stable algorithm: {most_stable['algorithm'].upper()}")
            print(f"   Eval MSE: {most_stable['eval_mse']:.4f} ± {most_stable['eval_mse_std']:.4f}")
        else:
            print(f"\n{'Algorithm':<15} {'Rules':<8} {'Labels':<10} {'Train MSE':<12} {'Eval MSE':<12}")
            print("-" * 70)
            for r in all_results:
                print(f"{r['algorithm']:<15} {r['num_rules']:<8} {r['num_labels']:<10} "
                      f"{r['train_mse']:<12.6f} {r['eval_mse']:<12.6f}")
        
        # Determine winner
        best_eval = min(all_results, key=lambda x: x['eval_mse'])
        print(f"\n🏆 Best average eval MSE: {best_eval['algorithm'].upper()}")
        if 'eval_mse_std' in best_eval:
            print(f"   Eval MSE: {best_eval['eval_mse']:.4f} ± {best_eval['eval_mse_std']:.4f} ({best_eval['num_runs']} runs)")
        else:
            print(f"   Eval MSE: {best_eval['eval_mse']:.6f}")
        
    else:
        # Test single algorithm
        result = test_single_algorithm(args.algorithm, facts, max_rules=args.max_rules, min_support=args.min_support)
        if result:
            print(f"\n✅ {args.algorithm.upper()} results:")
            print(f"   Rules: {result['num_rules']}")
            print(f"   Labels: {result['num_labels']}")
            print(f"   Train MSE: {result['train_mse']:.6f}")
            print(f"   Eval MSE: {result['eval_mse']:.6f}")


if __name__ == '__main__':
    main()
