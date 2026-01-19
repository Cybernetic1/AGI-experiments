"""
Baseline Comparison: DLN Training With vs Without Rules
========================================================

Compares two conditions:
1. **With Rules**: Train DLN on rule-inferred labels (current approach)
2. **Without Rules (Baseline)**: Train DLN on raw facts only

Measures:
- Convergence speed (steps to target MSE)
- Final train/eval MSE
- Training time

Usage:
    python compare_ar_baseline.py --stories 1000 --facts 50000 --steps 100
"""

import torch
import torch.nn as nn
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
import argparse

from pipelines.tinystories_pipeline import load_tinystories_facts
from hybrid_ga_ilp_dln import HybridRuleDiscovery
from logic_core import Proposition, Rule
from label_utils import _collect_labels
from dln import SimpleDLN
from core.train_utils import _train_on_labels, _eval_on_labels


def train_baseline_without_rules(
    facts: List[Proposition],
    embed_dim: int = 64,
    training_steps: int = 100,
    verbose: bool = True
) -> Tuple[SimpleDLN, Dict]:
    """Train DLN on raw facts only (no rule inference)."""
    
    if verbose:
        print("\n" + "="*70)
        print("BASELINE: Training WITHOUT Rules")
        print("="*70)
    
    # Collect vocabularies
    all_predicates = set()
    all_args = set()
    for fact in facts:
        all_predicates.add(fact.predicate)
        all_args.update(fact.args)
    
    if verbose:
        print(f"\nVocabulary: {len(all_predicates)} predicates, {len(all_args)} entities")
        print(f"Training on {len(facts)} raw facts...")
    
    # Create labels from raw facts (no inference)
    baseline_labels = {
        (f.predicate, f.args): f.truth
        for f in facts
    }
    
    if verbose:
        print(f"Generated {len(baseline_labels)} baseline labels (1:1 with facts)")
    
    # Create DLN model
    model = SimpleDLN(
        predicates=list(all_predicates),
        args=list(all_args),
        embed_dim=embed_dim
    )
    
    # Split train/eval
    labels_list = list(baseline_labels.items())
    split_idx = int(0.8 * len(labels_list))
    train_labels = {k: v for k, v in labels_list[:split_idx]}
    eval_labels = {k: v for k, v in labels_list[split_idx:]}
    
    if verbose:
        print(f"Train: {len(train_labels)} labels, Eval: {len(eval_labels)} labels")
        print(f"Training for {training_steps} steps...")
    
    # Train DLN
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    start_time = time.time()
    history = []
    
    for step in range(training_steps):
        train_mse = _train_on_labels(
            model,
            optimizer,
            facts,
            train_labels,
            steps=1,
            batch_size=None
        )
        
        if verbose and (step + 1) % max(1, training_steps // 10) == 0:
            eval_mse = _eval_on_labels(model, facts, eval_labels)
            elapsed = time.time() - start_time
            print(f"  Step {step + 1}/{training_steps}: Train MSE={train_mse:.6f}, Eval MSE={eval_mse:.6f}, Time={elapsed:.1f}s")
            history.append({
                'step': step + 1,
                'train_mse': train_mse,
                'eval_mse': eval_mse,
                'time': elapsed
            })
    
    training_time = time.time() - start_time
    
    # Final evaluation
    final_train_mse = _eval_on_labels(model, facts, train_labels)
    final_eval_mse = _eval_on_labels(model, facts, eval_labels)
    
    metrics = {
        'num_facts': len(facts),
        'num_labels': len(baseline_labels),
        'expansion_ratio': 1.0,
        'train_mse': final_train_mse,
        'eval_mse': final_eval_mse,
        'training_time': training_time,
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'history': history
    }
    
    if verbose:
        print(f"\n✅ Baseline training complete")
        print(f"  Final Train MSE: {final_train_mse:.6f}")
        print(f"  Final Eval MSE: {final_eval_mse:.6f}")
        print(f"  Training time: {training_time:.1f}s")
        print(f"  Model parameters: {metrics['num_parameters']:,}")
    
    return model, metrics


def train_with_rules(
    facts: List[Proposition],
    rules: List[Rule],
    embed_dim: int = 64,
    training_steps: int = 100,
    verbose: bool = True
) -> Tuple[SimpleDLN, Dict]:
    """Train DLN with rule-inferred labels."""
    
    if verbose:
        print("\n" + "="*70)
        print("WITH RULES: Training with Rule-Inferred Labels")
        print("="*70)
    
    # Generate labels using rules
    if verbose:
        print(f"\nGenerating labels with {len(rules)} rules...")
    
    labels_dict = _collect_labels(
        facts,
        rules,
        log_progress=verbose
    )
    
    if verbose:
        print(f"Generated {len(labels_dict)} labels from {len(facts)} facts")
    
    # Collect vocabularies
    all_predicates = set()
    all_args = set()
    for fact in facts:
        all_predicates.add(fact.predicate)
        all_args.update(fact.args)
    for rule in rules:
        all_predicates.add(rule.conclusion.predicate)
    
    if verbose:
        print(f"Vocabulary: {len(all_predicates)} predicates, {len(all_args)} entities")
    
    # Create DLN model
    model = SimpleDLN(
        predicates=list(all_predicates),
        args=list(all_args),
        embed_dim=embed_dim
    )
    
    # Split train/eval
    labels_list = list(labels_dict.items())
    split_idx = int(0.8 * len(labels_list))
    train_labels = {k: v for k, v in labels_list[:split_idx]}
    eval_labels = {k: v for k, v in labels_list[split_idx:]}
    
    if verbose:
        print(f"Train: {len(train_labels)} labels, Eval: {len(eval_labels)} labels")
        print(f"Training for {training_steps} steps...")
    
    # Train DLN
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    start_time = time.time()
    history = []
    
    for step in range(training_steps):
        train_mse = _train_on_labels(
            model,
            optimizer,
            facts,
            train_labels,
            steps=1,
            batch_size=None
        )
        
        if verbose and (step + 1) % max(1, training_steps // 10) == 0:
            eval_mse = _eval_on_labels(model, facts, eval_labels)
            elapsed = time.time() - start_time
            print(f"  Step {step + 1}/{training_steps}: Train MSE={train_mse:.6f}, Eval MSE={eval_mse:.6f}, Time={elapsed:.1f}s")
            history.append({
                'step': step + 1,
                'train_mse': train_mse,
                'eval_mse': eval_mse,
                'time': elapsed
            })
    
    training_time = time.time() - start_time
    
    # Final evaluation
    final_train_mse = _eval_on_labels(model, facts, train_labels)
    final_eval_mse = _eval_on_labels(model, facts, eval_labels)
    
    metrics = {
        'num_facts': len(facts),
        'num_labels': len(labels_dict),
        'expansion_ratio': len(labels_dict) / len(facts) if facts else 0,
        'train_mse': final_train_mse,
        'eval_mse': final_eval_mse,
        'training_time': training_time,
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'history': history
    }
    
    if verbose:
        print(f"\n✅ Rule-based training complete")
        print(f"  Final Train MSE: {final_train_mse:.6f}")
        print(f"  Final Eval MSE: {final_eval_mse:.6f}")
        print(f"  Training time: {training_time:.1f}s")
        print(f"  Model parameters: {metrics['num_parameters']:,}")
    
    return model, metrics


def main():
    parser = argparse.ArgumentParser(description='Compare DLN Training: With vs Without Rules')
    parser.add_argument('--corpus', default='data/processed/tinystories_train.json', help='Corpus path')
    parser.add_argument('--discovery-stories', type=int, default=100, help='Stories for rule discovery')
    parser.add_argument('--discovery-facts', type=int, default=10000, help='Facts for rule discovery')
    parser.add_argument('--ilp-algorithm', choices=['frequency', 'foil', 'confidence'], default='frequency')
    parser.add_argument('--ilp-rules', type=int, default=30, help='Initial ILP rules')
    parser.add_argument('--ga-generations', type=int, default=10, help='GA generations (reduced for speed)')
    parser.add_argument('--stories', type=int, default=1000, help='Stories for training comparison')
    parser.add_argument('--facts', type=int, default=50000, help='Facts for training comparison')
    parser.add_argument('--embed-dim', type=int, default=64, help='DLN embedding dimension')
    parser.add_argument('--steps', type=int, default=100, help='DLN training steps')
    parser.add_argument('--output-dir', default='outputs/ar_baseline_comparison', help='Output directory')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    if verbose:
        print("\n" + "="*70)
        print("AR BASELINE COMPARISON EXPERIMENT")
        print("="*70)
        print(f"\nConfiguration:")
        print(f"  Rule discovery: {args.discovery_stories} stories, {args.discovery_facts} facts")
        print(f"  ILP: {args.ilp_algorithm}, {args.ilp_rules} rules")
        print(f"  GA: {args.ga_generations} generations")
        print(f"  Training: {args.stories} stories, {args.facts} facts")
        print(f"  DLN: {args.embed_dim}d embeddings, {args.steps} steps")
    
    # Load training facts
    if verbose:
        print(f"\n[Stage 1] Loading training corpus...")
    
    training_facts = load_tinystories_facts(
        max_stories=args.stories,
        max_facts=args.facts,
        path=args.corpus
    )
    
    if verbose:
        print(f"  ✅ Loaded {len(training_facts)} facts")
    
    # Baseline: Train without rules
    baseline_model, baseline_metrics = train_baseline_without_rules(
        training_facts,
        embed_dim=args.embed_dim,
        training_steps=args.steps,
        verbose=verbose
    )
    
    # Discover rules for comparison
    if verbose:
        print(f"\n[Stage 2] Rule Discovery...")
    
    discovery_facts = load_tinystories_facts(
        max_stories=args.discovery_stories,
        max_facts=args.discovery_facts,
        path=args.corpus
    )
    
    discoverer = HybridRuleDiscovery(
        ilp_algorithm=args.ilp_algorithm,
        ilp_rules=args.ilp_rules,
        ga_generations=args.ga_generations,
        sample_facts_for_fitness=1000,
        verbose=verbose
    )
    
    rules = discoverer.discover(discovery_facts)
    
    if verbose:
        print(f"  ✅ Discovered {len(rules)} rules")
    
    # Train with rules
    rules_model, rules_metrics = train_with_rules(
        training_facts,
        rules,
        embed_dim=args.embed_dim,
        training_steps=args.steps,
        verbose=verbose
    )
    
    # Comparison summary
    if verbose:
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        
        print("\n[Baseline - No Rules]")
        print(f"  Labels: {baseline_metrics['num_labels']} (expansion: {baseline_metrics['expansion_ratio']:.2f}×)")
        print(f"  Train MSE: {baseline_metrics['train_mse']:.6f}")
        print(f"  Eval MSE: {baseline_metrics['eval_mse']:.6f}")
        print(f"  Training time: {baseline_metrics['training_time']:.1f}s")
        
        print("\n[With Rules]")
        print(f"  Labels: {rules_metrics['num_labels']} (expansion: {rules_metrics['expansion_ratio']:.2f}×)")
        print(f"  Train MSE: {rules_metrics['train_mse']:.6f}")
        print(f"  Eval MSE: {rules_metrics['eval_mse']:.6f}")
        print(f"  Training time: {rules_metrics['training_time']:.1f}s")
        
        print("\n[Improvement]")
        eval_improvement = (baseline_metrics['eval_mse'] - rules_metrics['eval_mse']) / baseline_metrics['eval_mse'] * 100
        time_ratio = rules_metrics['training_time'] / baseline_metrics['training_time']
        
        if eval_improvement > 0:
            print(f"  ✅ Eval MSE improved by {eval_improvement:.1f}%")
        else:
            print(f"  ❌ Eval MSE worsened by {-eval_improvement:.1f}%")
        
        print(f"  Training time ratio: {time_ratio:.2f}× (rules/baseline)")
        
        if rules_metrics['expansion_ratio'] > 1.1:
            print(f"  ℹ️  Rules provided {rules_metrics['expansion_ratio']:.2f}× label expansion")
        else:
            print(f"  ⚠️  Limited label expansion ({rules_metrics['expansion_ratio']:.2f}×)")
        
        print("\n" + "="*70)
    
    # Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    comparison = {
        'baseline': baseline_metrics,
        'with_rules': rules_metrics,
        'rule_discovery': {
            'num_rules': len(rules),
            'best_fitness': discoverer.best_fitness,
        },
        'improvement': {
            'eval_mse_change_percent': (baseline_metrics['eval_mse'] - rules_metrics['eval_mse']) / baseline_metrics['eval_mse'] * 100,
            'train_time_ratio': rules_metrics['training_time'] / baseline_metrics['training_time'],
        }
    }
    
    with open(output_path / 'comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    if verbose:
        print(f"\n✅ Results saved to {output_path / 'comparison.json'}")


if __name__ == '__main__':
    main()
