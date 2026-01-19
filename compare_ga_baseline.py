"""
GA Validation: Does Evolution Actually Help?
=============================================

Compares three rule discovery approaches:
1. **ILP-only**: Raw ILP rules without evolution
2. **GA-evolved**: ILP seed + genetic algorithm evolution
3. **Random**: Random rules as sanity check

Measures:
- Rule quality (fitness, eval MSE)
- Label expansion
- Training performance

Usage:
    python compare_ga_baseline.py --stories 100 --facts 10000 --ga-generations 10
    
For GPU server:
    python compare_ga_baseline.py --stories 500 --facts 50000 --ga-generations 20 --device cuda
"""

import torch
import torch.nn as nn
import json
import time
import random
from pathlib import Path
from typing import List, Dict, Tuple
import argparse

from pipelines.tinystories_pipeline import load_tinystories_facts
from hybrid_ga_ilp_dln import HybridRuleDiscovery
from logic_core import Proposition, Rule
from label_utils import _collect_labels
from dln import SimpleDLN
from core.train_utils import _train_on_labels, _eval_on_labels
from core.ilp_algorithms import mine_frequency_based, mine_foil_style, mine_confidence_based


def evaluate_rules(
    rules: List[Rule],
    facts: List[Proposition],
    embed_dim: int = 64,
    training_steps: int = 50,
    device: str = "cpu",
    verbose: bool = True,
    label: str = "Rules"
) -> Dict:
    """Evaluate a set of rules by training DLN and measuring performance."""
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"{label}")
        print(f"{'='*70}")
    
    # Generate labels using rules
    if verbose:
        print(f"\nGenerating labels with {len(rules)} rules...")
    
    start_time = time.time()
    labels_dict = _collect_labels(
        facts,
        rules,
        log_progress=False  # Suppress for cleaner output
    )
    label_time = time.time() - start_time
    
    if verbose:
        print(f"  Generated {len(labels_dict)} labels in {label_time:.1f}s")
    
    if len(labels_dict) == 0:
        if verbose:
            print(f"  ❌ No labels generated - rules don't fire!")
        return {
            'num_rules': len(rules),
            'num_labels': 0,
            'expansion_ratio': 0.0,
            'train_mse': float('inf'),
            'eval_mse': float('inf'),
            'training_time': 0.0,
            'label_time': label_time,
            'total_time': label_time
        }
    
    # Collect vocabularies
    all_predicates = set()
    all_args = set()
    for fact in facts:
        all_predicates.add(fact.predicate)
        all_args.update(fact.args)
    for rule in rules:
        all_predicates.add(rule.conclusion.predicate)
    
    # Create DLN model
    model = SimpleDLN(
        predicates=list(all_predicates),
        args=list(all_args),
        embed_dim=embed_dim
    )
    
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
        if verbose:
            print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Split train/eval
    labels_list = list(labels_dict.items())
    split_idx = int(0.8 * len(labels_list))
    train_labels = {k: v for k, v in labels_list[:split_idx]}
    eval_labels = {k: v for k, v in labels_list[split_idx:]}
    
    if len(eval_labels) == 0:
        eval_labels = train_labels  # Fallback
    
    if verbose:
        print(f"  Train: {len(train_labels)} labels, Eval: {len(eval_labels)} labels")
        print(f"  Training for {training_steps} steps...")
    
    # Train DLN
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_start = time.time()
    history = []
    
    for step in range(training_steps):
        train_mse = _train_on_labels(
            model,
            optimizer,
            facts,
            train_labels,
            steps=1,
            batch_size=None,
            device=device
        )
        
        if verbose and (step + 1) % max(1, training_steps // 5) == 0:
            eval_mse = _eval_on_labels(model, facts, eval_labels, device=device)
            elapsed = time.time() - train_start
            print(f"    Step {step + 1}/{training_steps}: Train MSE={train_mse:.6f}, Eval MSE={eval_mse:.6f}, Time={elapsed:.1f}s")
            history.append({
                'step': step + 1,
                'train_mse': train_mse,
                'eval_mse': eval_mse,
                'time': elapsed
            })
    
    training_time = time.time() - train_start
    
    # Final evaluation
    final_train_mse = _eval_on_labels(model, facts, train_labels, device=device)
    final_eval_mse = _eval_on_labels(model, facts, eval_labels, device=device)
    
    metrics = {
        'num_rules': len(rules),
        'num_labels': len(labels_dict),
        'expansion_ratio': len(labels_dict) / len(facts) if facts else 0,
        'train_mse': final_train_mse,
        'eval_mse': final_eval_mse,
        'training_time': training_time,
        'label_time': label_time,
        'total_time': label_time + training_time,
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'history': history
    }
    
    if verbose:
        print(f"\n  ✅ Evaluation complete")
        print(f"    Labels: {len(labels_dict)} (expansion: {metrics['expansion_ratio']:.2f}×)")
        print(f"    Train MSE: {final_train_mse:.6f}")
        print(f"    Eval MSE: {final_eval_mse:.6f}")
        print(f"    Time: {metrics['total_time']:.1f}s")
    
    return metrics


def generate_random_rules(
    facts: List[Proposition],
    num_rules: int,
    seed: int = 42
) -> List[Rule]:
    """Generate random rules as a sanity check baseline."""
    random.seed(seed)
    
    # Collect available predicates and entities
    predicates = list(set(f.predicate for f in facts))
    all_args = set()
    for f in facts:
        all_args.update(f.args)
    args_list = list(all_args)
    
    if len(predicates) < 2 or len(args_list) < 2:
        return []
    
    rules = []
    for i in range(num_rules):
        # Random premises (1-2)
        num_premises = random.choice([1, 2])
        premises = []
        for _ in range(num_premises):
            pred = random.choice(predicates)
            # Use variable-like args for binding
            args = tuple(f"?x{j}" for j in range(random.randint(1, 3)))
            premises.append(Proposition(pred, args, 1.0))
        
        # Random conclusion
        concl_pred = f"random_rule_{i}"
        concl_args = tuple(f"?x{j}" for j in range(random.randint(1, 3)))
        conclusion = Proposition(concl_pred, concl_args, 1.0)
        
        # Random weight
        weight = random.uniform(0.3, 1.0)
        
        rules.append(Rule(premises=premises, conclusion=conclusion, weight=weight))
    
    return rules


def main():
    parser = argparse.ArgumentParser(description='GA Validation: Evolution vs ILP vs Random')
    parser.add_argument('--corpus', default='data/processed/tinystories_train.json', help='Corpus path')
    parser.add_argument('--stories', type=int, default=100, help='Stories for evaluation')
    parser.add_argument('--facts', type=int, default=10000, help='Facts for evaluation')
    parser.add_argument('--ilp-algorithm', choices=['frequency', 'foil', 'confidence'], default='frequency')
    parser.add_argument('--ilp-rules', type=int, default=30, help='Number of ILP rules')
    parser.add_argument('--ga-generations', type=int, default=10, help='GA generations')
    parser.add_argument('--embed-dim', type=int, default=64, help='DLN embedding dimension')
    parser.add_argument('--training-steps', type=int, default=50, help='DLN training steps per condition')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='Device (cpu or cuda)')
    parser.add_argument('--output-dir', default='outputs/ga_validation', help='Output directory')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    # Set random seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    if verbose:
        print("\n" + "="*70)
        print("GA VALIDATION EXPERIMENT")
        print("="*70)
        print(f"\nConfiguration:")
        print(f"  Dataset: {args.stories} stories, {args.facts} facts")
        print(f"  ILP: {args.ilp_algorithm}, {args.ilp_rules} rules")
        print(f"  GA: {args.ga_generations} generations")
        print(f"  DLN: {args.embed_dim}d embeddings, {args.training_steps} steps")
        print(f"  Device: {args.device}")
        print(f"  Random seed: {args.seed}")
    
    # Load facts
    if verbose:
        print(f"\n[Stage 1] Loading corpus...")
    
    facts = load_tinystories_facts(
        max_stories=args.stories,
        max_facts=args.facts,
        path=args.corpus
    )
    
    if verbose:
        print(f"  ✅ Loaded {len(facts)} facts")
    
    # ============================================================================
    # Condition 1: ILP-only (no evolution)
    # ============================================================================
    
    if verbose:
        print(f"\n[Stage 2] Condition 1: ILP-only (no GA evolution)")
    
    if args.ilp_algorithm == 'frequency':
        ilp_rules, _ = mine_frequency_based(facts, args.ilp_rules, min_support=2)
    elif args.ilp_algorithm == 'foil':
        ilp_rules, _ = mine_foil_style(facts, args.ilp_rules, min_support=2)
    elif args.ilp_algorithm == 'confidence':
        ilp_rules, _ = mine_confidence_based(facts, args.ilp_rules, min_support=2, min_confidence=0.3)
    else:
        raise ValueError(f"Unknown ILP algorithm: {args.ilp_algorithm}")
    
    if verbose:
        print(f"  Mined {len(ilp_rules)} ILP rules")
        print(f"  Sample rules:")
        for i, rule in enumerate(ilp_rules[:3], 1):
            print(f"    {i}. {rule}")
    
    ilp_metrics = evaluate_rules(
        ilp_rules,
        facts,
        embed_dim=args.embed_dim,
        training_steps=args.training_steps,
        device=args.device,
        verbose=verbose,
        label="[Condition 1: ILP-only Rules]"
    )
    
    # ============================================================================
    # Condition 2: GA-evolved (ILP seed + evolution)
    # ============================================================================
    
    if verbose:
        print(f"\n[Stage 3] Condition 2: GA-evolved (ILP + {args.ga_generations} generations)")
    
    discoverer = HybridRuleDiscovery(
        ilp_algorithm=args.ilp_algorithm,
        ilp_rules=args.ilp_rules,
        ga_generations=args.ga_generations,
        sample_facts_for_fitness=min(1000, len(facts)),
        verbose=verbose
    )
    
    ga_rules = discoverer.discover(facts)
    
    if verbose:
        print(f"  Evolved {len(ga_rules)} rules")
        print(f"  Best fitness: {discoverer.best_fitness:.4f}")
        print(f"  Sample rules:")
        for i, rule in enumerate(ga_rules[:3], 1):
            print(f"    {i}. {rule}")
    
    ga_metrics = evaluate_rules(
        ga_rules,
        facts,
        embed_dim=args.embed_dim,
        training_steps=args.training_steps,
        device=args.device,
        verbose=verbose,
        label="[Condition 2: GA-Evolved Rules]"
    )
    
    ga_metrics['ga_best_fitness'] = discoverer.best_fitness
    ga_metrics['ga_history'] = discoverer.history
    
    # ============================================================================
    # Condition 3: Random rules (sanity check)
    # ============================================================================
    
    if verbose:
        print(f"\n[Stage 4] Condition 3: Random rules (sanity check)")
    
    random_rules = generate_random_rules(facts, args.ilp_rules, seed=args.seed)
    
    if verbose:
        print(f"  Generated {len(random_rules)} random rules")
        print(f"  Sample rules:")
        for i, rule in enumerate(random_rules[:3], 1):
            print(f"    {i}. {rule}")
    
    random_metrics = evaluate_rules(
        random_rules,
        facts,
        embed_dim=args.embed_dim,
        training_steps=args.training_steps,
        device=args.device,
        verbose=verbose,
        label="[Condition 3: Random Rules]"
    )
    
    # ============================================================================
    # Comparison Summary
    # ============================================================================
    
    if verbose:
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        
        # Table header
        print(f"\n{'Condition':<20} {'Labels':<12} {'Expansion':<12} {'Eval MSE':<12} {'Time (s)':<12}")
        print("-" * 70)
        
        # ILP-only
        print(f"{'1. ILP-only':<20} {ilp_metrics['num_labels']:<12} {ilp_metrics['expansion_ratio']:<12.2f} {ilp_metrics['eval_mse']:<12.6f} {ilp_metrics['total_time']:<12.1f}")
        
        # GA-evolved
        print(f"{'2. GA-evolved':<20} {ga_metrics['num_labels']:<12} {ga_metrics['expansion_ratio']:<12.2f} {ga_metrics['eval_mse']:<12.6f} {ga_metrics['total_time']:<12.1f}")
        
        # Random
        print(f"{'3. Random':<20} {random_metrics['num_labels']:<12} {random_metrics['expansion_ratio']:<12.2f} {random_metrics['eval_mse']:<12.6f} {random_metrics['total_time']:<12.1f}")
        
        print("\n" + "="*70)
        print("ANALYSIS")
        print("="*70)
        
        # Find best performer
        conditions = [
            ('ILP-only', ilp_metrics['eval_mse']),
            ('GA-evolved', ga_metrics['eval_mse']),
            ('Random', random_metrics['eval_mse'])
        ]
        conditions.sort(key=lambda x: x[1])
        
        print(f"\n[Performance Ranking by Eval MSE]")
        for i, (name, mse) in enumerate(conditions, 1):
            marker = "🏆" if i == 1 else "  "
            print(f"  {marker} {i}. {name:<15} MSE: {mse:.6f}")
        
        # GA vs ILP comparison
        print(f"\n[GA Evolution Impact]")
        if ga_metrics['eval_mse'] < ilp_metrics['eval_mse']:
            improvement = (ilp_metrics['eval_mse'] - ga_metrics['eval_mse']) / ilp_metrics['eval_mse'] * 100
            print(f"  ✅ GA improved eval MSE by {improvement:.1f}% over ILP-only")
        else:
            degradation = (ga_metrics['eval_mse'] - ilp_metrics['eval_mse']) / ilp_metrics['eval_mse'] * 100
            print(f"  ❌ GA worsened eval MSE by {degradation:.1f}% vs ILP-only")
        
        # Label expansion comparison
        print(f"\n[Label Expansion]")
        print(f"  ILP-only:   {ilp_metrics['expansion_ratio']:.2f}×")
        print(f"  GA-evolved: {ga_metrics['expansion_ratio']:.2f}×")
        print(f"  Random:     {random_metrics['expansion_ratio']:.2f}×")
        
        # Sanity check
        print(f"\n[Sanity Check]")
        if random_metrics['eval_mse'] > ilp_metrics['eval_mse'] and random_metrics['eval_mse'] > ga_metrics['eval_mse']:
            print(f"  ✅ Both ILP and GA beat random baseline (as expected)")
        else:
            print(f"  ⚠️  Random rules perform unexpectedly well - check data/rules")
        
        # Time analysis
        ga_overhead = ga_metrics['total_time'] - ilp_metrics['total_time']
        print(f"\n[Computational Cost]")
        print(f"  GA overhead: +{ga_overhead:.1f}s ({ga_metrics['total_time']/ilp_metrics['total_time']:.2f}× vs ILP)")
        
        # Conclusion
        print("\n" + "="*70)
        print("CONCLUSION")
        print("="*70)
        
        if ga_metrics['eval_mse'] < ilp_metrics['eval_mse'] * 0.95:  # 5% threshold
            print("\n✅ GA evolution provides meaningful improvement over ILP-only")
            print(f"   Recommendation: Use GA evolution ({args.ga_generations} generations)")
        elif ga_metrics['eval_mse'] < ilp_metrics['eval_mse'] * 1.05:  # Within 5%
            print("\n⚠️  GA evolution provides marginal benefit")
            print(f"   Recommendation: ILP-only may be sufficient (faster)")
        else:
            print("\n❌ GA evolution does not improve over ILP-only")
            print(f"   Recommendation: Use ILP-only or tune GA hyperparameters")
        
        print("\n" + "="*70)
    
    # Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    comparison = {
        'configuration': {
            'stories': args.stories,
            'facts_loaded': len(facts),
            'ilp_algorithm': args.ilp_algorithm,
            'ilp_rules': args.ilp_rules,
            'ga_generations': args.ga_generations,
            'embed_dim': args.embed_dim,
            'training_steps': args.training_steps,
            'device': args.device,
            'seed': args.seed
        },
        'ilp_only': ilp_metrics,
        'ga_evolved': ga_metrics,
        'random': random_metrics,
        'winner': conditions[0][0],
        'ga_improves_ilp': ga_metrics['eval_mse'] < ilp_metrics['eval_mse']
    }
    
    with open(output_path / 'ga_validation.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    if verbose:
        print(f"\n✅ Results saved to {output_path / 'ga_validation.json'}")


if __name__ == '__main__':
    main()
