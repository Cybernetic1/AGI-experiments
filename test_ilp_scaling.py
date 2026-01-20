"""
ILP-Only Scaling Test: Can Pure ILP Handle Large Datasets?
===========================================================

Tests ILP rule discovery (no GA) on increasing dataset scales:
- Small: 100 stories, 10K facts
- Medium: 500 stories, 50K facts  
- Large: 1000 stories, 100K facts
- XLarge: 2000 stories, 200K facts

Measures:
- Rule discovery time
- Number of rules discovered
- Rule quality (eval MSE)
- Label expansion
- Memory usage

Optional: Test hierarchical rule organization (general → specific)

Usage:
    # Quick test (CPU)
    python test_ilp_scaling.py --scale small
    
    # Full scaling test (GPU recommended)
    python test_ilp_scaling.py --scale all --device cuda
    
    # Test specific algorithm
    python test_ilp_scaling.py --scale medium --ilp-algorithm confidence
"""

import torch
import json
import time
import psutil
import os
from pathlib import Path
from typing import List, Dict, Tuple
import argparse

from pipelines.tinystories_pipeline import load_tinystories_facts
from logic_core import Proposition, Rule
from label_utils import _collect_labels
from dln import SimpleDLN
from core.train_utils import _train_on_labels, _eval_on_labels
from core.ilp_algorithms import mine_frequency_based, mine_foil_style, mine_confidence_based


SCALE_CONFIGS = {
    'tiny': {'stories': 50, 'facts': 5000, 'rules': 20},
    'small': {'stories': 100, 'facts': 10000, 'rules': 30},
    'medium': {'stories': 500, 'facts': 50000, 'rules': 50},
    'large': {'stories': 1000, 'facts': 100000, 'rules': 75},
    'xlarge': {'stories': 2000, 'facts': 200000, 'rules': 100},
}


def get_memory_usage_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def organize_rules_hierarchically(rules: List[Rule]) -> Dict[str, List[Rule]]:
    """
    Organize rules by generality: general → specific.
    
    Heuristic:
    - General: Few premises (1), broad predicates
    - Specific: Many premises (2+), narrow predicates
    """
    hierarchy = {
        'general': [],      # 1 premise
        'intermediate': [], # 2 premises
        'specific': []      # 3+ premises
    }
    
    for rule in rules:
        num_premises = len(rule.premises)
        if num_premises == 1:
            hierarchy['general'].append(rule)
        elif num_premises == 2:
            hierarchy['intermediate'].append(rule)
        else:
            hierarchy['specific'].append(rule)
    
    return hierarchy


def mine_ilp_rules(
    facts: List[Proposition],
    algorithm: str,
    max_rules: int,
    verbose: bool = True
) -> Tuple[List[Rule], float, float]:
    """Mine rules using ILP algorithm and measure time/memory."""
    
    start_time = time.time()
    start_memory = get_memory_usage_mb()
    
    if algorithm == 'frequency':
        rules, _ = mine_frequency_based(facts, max_rules, min_support=2)
    elif algorithm == 'foil':
        rules, _ = mine_foil_style(facts, max_rules, min_support=2)
    elif algorithm == 'confidence':
        rules, _ = mine_confidence_based(facts, max_rules, min_support=2, min_confidence=0.3)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    mining_time = time.time() - start_time
    memory_delta = get_memory_usage_mb() - start_memory
    
    if verbose:
        print(f"  ✅ Mined {len(rules)} rules in {mining_time:.1f}s (memory: +{memory_delta:.1f}MB)")
    
    return rules, mining_time, memory_delta


def evaluate_rules_fast(
    rules: List[Rule],
    facts: List[Proposition],
    embed_dim: int = 64,
    training_steps: int = 50,
    device: str = "cpu",
    verbose: bool = True
) -> Dict:
    """Fast evaluation focusing on key metrics."""
    
    start_time = time.time()
    
    # Generate labels
    if verbose:
        print(f"  Generating labels...")
    
    label_start = time.time()
    labels_dict = _collect_labels(
        facts,
        rules,
        log_progress=False
    )
    label_time = time.time() - label_start
    
    if len(labels_dict) == 0:
        if verbose:
            print(f"  ❌ No labels generated")
        return {
            'num_labels': 0,
            'expansion_ratio': 0.0,
            'eval_mse': float('inf'),
            'label_time': label_time,
            'training_time': 0.0
        }
    
    if verbose:
        print(f"  ✅ Generated {len(labels_dict)} labels in {label_time:.1f}s")
    
    # Collect vocabularies
    all_predicates = set()
    all_args = set()
    for fact in facts:
        all_predicates.add(fact.predicate)
        all_args.update(fact.args)
    for rule in rules:
        all_predicates.add(rule.conclusion.predicate)
    
    # Create DLN
    model = SimpleDLN(
        predicates=list(all_predicates),
        args=list(all_args),
        embed_dim=embed_dim
    )
    
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    
    # Split labels
    labels_list = list(labels_dict.items())
    split_idx = int(0.8 * len(labels_list))
    train_labels = {k: v for k, v in labels_list[:split_idx]}
    eval_labels = {k: v for k, v in labels_list[split_idx:]}
    
    if len(eval_labels) == 0:
        eval_labels = train_labels
    
    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_start = time.time()
    for _ in range(training_steps):
        _train_on_labels(
            model,
            optimizer,
            facts,
            train_labels,
            steps=1,
            batch_size=None,
            device=device
        )
    training_time = time.time() - train_start
    
    # Evaluate
    eval_mse = _eval_on_labels(model, facts, eval_labels, device=device)
    
    if verbose:
        print(f"  ✅ Eval MSE: {eval_mse:.6f} (training: {training_time:.1f}s)")
    
    return {
        'num_labels': len(labels_dict),
        'expansion_ratio': len(labels_dict) / len(facts) if facts else 0,
        'eval_mse': eval_mse,
        'label_time': label_time,
        'training_time': training_time,
        'total_time': time.time() - start_time
    }


def test_scale(
    scale_name: str,
    config: Dict,
    ilp_algorithm: str,
    corpus_path: str,
    device: str,
    embed_dim: int,
    training_steps: int,
    test_hierarchy: bool,
    verbose: bool
) -> Dict:
    """Test ILP at a specific scale."""
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"SCALE: {scale_name.upper()} ({config['stories']} stories, {config['facts']} facts, {config['rules']} rules)")
        print(f"{'='*70}")
    
    # Load data
    if verbose:
        print(f"\n[1] Loading corpus...")
    
    load_start = time.time()
    facts = load_tinystories_facts(
        max_stories=config['stories'],
        max_facts=config['facts'],
        path=corpus_path
    )
    load_time = time.time() - load_start
    
    if verbose:
        print(f"  ✅ Loaded {len(facts)} facts in {load_time:.1f}s")
    
    # Mine rules
    if verbose:
        print(f"\n[2] Mining ILP rules ({ilp_algorithm})...")
    
    rules, mining_time, memory_delta = mine_ilp_rules(
        facts,
        ilp_algorithm,
        config['rules'],
        verbose=verbose
    )
    
    # Organize hierarchically if requested
    hierarchy = None
    if test_hierarchy and rules:
        if verbose:
            print(f"\n[3] Organizing rules hierarchically...")
        hierarchy = organize_rules_hierarchically(rules)
        if verbose:
            print(f"  General rules (1 premise): {len(hierarchy['general'])}")
            print(f"  Intermediate rules (2 premises): {len(hierarchy['intermediate'])}")
            print(f"  Specific rules (3+ premises): {len(hierarchy['specific'])}")
            
            # Show sample from each level
            if hierarchy['general']:
                print(f"  Sample general: {hierarchy['general'][0]}")
            if hierarchy['intermediate']:
                print(f"  Sample intermediate: {hierarchy['intermediate'][0]}")
            if hierarchy['specific']:
                print(f"  Sample specific: {hierarchy['specific'][0]}")
    
    # Evaluate
    if verbose:
        print(f"\n[{'4' if test_hierarchy else '3'}] Evaluating rules...")
    
    eval_metrics = evaluate_rules_fast(
        rules,
        facts,
        embed_dim=embed_dim,
        training_steps=training_steps,
        device=device,
        verbose=verbose
    )
    
    # Compile results
    results = {
        'scale': scale_name,
        'config': config,
        'facts_loaded': len(facts),
        'num_rules': len(rules),
        'load_time': load_time,
        'mining_time': mining_time,
        'memory_delta_mb': memory_delta,
        **eval_metrics
    }
    
    if hierarchy:
        results['hierarchy'] = {
            'general': len(hierarchy['general']),
            'intermediate': len(hierarchy['intermediate']),
            'specific': len(hierarchy['specific'])
        }
    
    # Summary
    if verbose:
        print(f"\n{'='*70}")
        print(f"SUMMARY: {scale_name.upper()}")
        print(f"{'='*70}")
        print(f"  Facts loaded: {len(facts):,}")
        print(f"  Rules mined: {len(rules)}")
        print(f"  Labels generated: {eval_metrics['num_labels']:,}")
        print(f"  Label expansion: {eval_metrics['expansion_ratio']:.2f}×")
        print(f"  Eval MSE: {eval_metrics['eval_mse']:.6f}")
        print(f"  Time breakdown:")
        print(f"    - Data loading: {load_time:.1f}s")
        print(f"    - Rule mining: {mining_time:.1f}s")
        print(f"    - Label generation: {eval_metrics['label_time']:.1f}s")
        print(f"    - DLN training: {eval_metrics['training_time']:.1f}s")
        print(f"    - Total: {eval_metrics['total_time']:.1f}s")
        print(f"  Memory delta: +{memory_delta:.1f}MB")
        if hierarchy:
            print(f"  Rule hierarchy:")
            print(f"    - General: {results['hierarchy']['general']}")
            print(f"    - Intermediate: {results['hierarchy']['intermediate']}")
            print(f"    - Specific: {results['hierarchy']['specific']}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='ILP-Only Scaling Test')
    parser.add_argument('--scale', default='small', 
                       choices=['tiny', 'small', 'medium', 'large', 'xlarge', 'all'],
                       help='Dataset scale to test (or "all" for full sweep)')
    parser.add_argument('--corpus', default='data/processed/tinystories_train.json', 
                       help='Corpus path')
    parser.add_argument('--ilp-algorithm', default='frequency',
                       choices=['frequency', 'foil', 'confidence'],
                       help='ILP algorithm')
    parser.add_argument('--embed-dim', type=int, default=64, help='DLN embedding dimension')
    parser.add_argument('--training-steps', type=int, default=50, help='DLN training steps')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='Device')
    parser.add_argument('--test-hierarchy', action='store_true', 
                       help='Test hierarchical rule organization')
    parser.add_argument('--output-dir', default='outputs/ilp_scaling', help='Output directory')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    if verbose:
        print("\n" + "="*70)
        print("ILP-ONLY SCALING TEST")
        print("="*70)
        print(f"\nConfiguration:")
        print(f"  Algorithm: {args.ilp_algorithm}")
        print(f"  Device: {args.device}")
        print(f"  DLN: {args.embed_dim}d embeddings, {args.training_steps} steps")
        print(f"  Test hierarchy: {args.test_hierarchy}")
    
    # Determine which scales to test
    if args.scale == 'all':
        scales_to_test = ['tiny', 'small', 'medium', 'large', 'xlarge']
    else:
        scales_to_test = [args.scale]
    
    # Run tests
    all_results = {}
    for scale_name in scales_to_test:
        config = SCALE_CONFIGS[scale_name]
        
        try:
            results = test_scale(
                scale_name,
                config,
                args.ilp_algorithm,
                args.corpus,
                args.device,
                args.embed_dim,
                args.training_steps,
                args.test_hierarchy,
                verbose
            )
            all_results[scale_name] = results
        except Exception as e:
            if verbose:
                print(f"\n❌ Failed at scale {scale_name}: {e}")
            all_results[scale_name] = {'error': str(e)}
    
    # Final comparison
    if len(all_results) > 1 and verbose:
        print("\n" + "="*70)
        print("SCALING COMPARISON")
        print("="*70)
        
        print(f"\n{'Scale':<10} {'Facts':<10} {'Rules':<8} {'Labels':<10} {'Expansion':<12} {'Eval MSE':<12} {'Time (s)':<10}")
        print("-" * 80)
        
        for scale_name in scales_to_test:
            if scale_name in all_results and 'error' not in all_results[scale_name]:
                r = all_results[scale_name]
                print(f"{scale_name:<10} {r['facts_loaded']:<10,} {r['num_rules']:<8} "
                      f"{r['num_labels']:<10,} {r['expansion_ratio']:<12.2f} "
                      f"{r['eval_mse']:<12.6f} {r['total_time']:<10.1f}")
        
        print("\n" + "="*70)
        print("INSIGHTS")
        print("="*70)
        
        # Check if performance degrades with scale
        successful = [r for r in all_results.values() if 'error' not in r and r['eval_mse'] != float('inf')]
        if len(successful) > 1:
            mses = [r['eval_mse'] for r in successful]
            times = [r['total_time'] for r in successful]
            
            print(f"\n[Performance vs Scale]")
            print(f"  MSE range: {min(mses):.6f} to {max(mses):.6f}")
            print(f"  Time range: {min(times):.1f}s to {max(times):.1f}s")
            
            if max(mses) < min(mses) * 2:
                print(f"  ✅ Performance stable across scales")
            else:
                print(f"  ⚠️  Performance degrades at larger scales")
            
            # Time scaling
            if len(times) >= 2:
                time_ratio = times[-1] / times[0]
                fact_ratio = successful[-1]['facts_loaded'] / successful[0]['facts_loaded']
                print(f"\n[Time Complexity]")
                print(f"  Facts scaled by: {fact_ratio:.1f}×")
                print(f"  Time scaled by: {time_ratio:.1f}×")
                
                if time_ratio < fact_ratio * 1.5:
                    print(f"  ✅ Sub-quadratic scaling (good!)")
                elif time_ratio < fact_ratio * fact_ratio * 1.5:
                    print(f"  ⚠️  Quadratic scaling (manageable)")
                else:
                    print(f"  ❌ Super-quadratic scaling (problematic)")
    
    # Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / 'scaling_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    if verbose:
        print(f"\n✅ Results saved to {output_path / 'scaling_results.json'}")


if __name__ == '__main__':
    main()
