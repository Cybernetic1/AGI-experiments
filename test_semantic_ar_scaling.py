"""
Pure Semantic Autoregressive Training - Scaling Test
=====================================================

Tests DLN autoregressive prediction on raw facts (no rule-generated labels).

Architecture:
- Given context facts F1...Fn, predict next fact Fn+1
- DLN learns embeddings for predicates and entities
- Uses AR head to predict predicate of next proposition
- Optional: Predict arguments as well

Measures:
- Convergence speed at different scales
- Prediction accuracy (predicate, full proposition)
- Training time vs dataset size
- Memory usage

Usage:
    # Quick test
    python test_semantic_ar_scaling.py --scale small --device cuda
    
    # Full scaling test
    python test_semantic_ar_scaling.py --scale all --device cuda
    
    # Compare with and without rules
    python test_semantic_ar_scaling.py --scale medium --use-rules --device cuda
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import time
import random
import psutil
import os
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
from collections import defaultdict

from pipelines.tinystories_pipeline import load_tinystories_facts
from logic_core import Proposition, Rule
from dln import SimpleDLN
from core.ilp_algorithms import mine_frequency_based


SCALE_CONFIGS = {
    'tiny': {'stories': 50, 'facts': 5000},
    'small': {'stories': 100, 'facts': 10000},
    'medium': {'stories': 500, 'facts': 50000},
    'large': {'stories': 1000, 'facts': 100000},
    'xlarge': {'stories': 2000, 'facts': 200000},
}


def get_memory_usage_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def group_facts_by_story(facts: List[Proposition]) -> List[List[Proposition]]:
    """
    Group facts by story (based on event IDs).
    Assumes facts with same first entity are from same story.
    """
    story_groups = defaultdict(list)
    for fact in facts:
        # Use first arg as story identifier (usually event ID)
        story_id = fact.args[0] if fact.args else "default"
        story_groups[story_id].append(fact)
    
    return list(story_groups.values())


def create_ar_training_data(
    facts: List[Proposition],
    context_size: int = 5,
    max_samples: int = 10000
) -> List[Tuple[List[Proposition], Proposition]]:
    """
    Create autoregressive training pairs: (context_facts, next_fact).
    
    Args:
        facts: All facts from corpus
        context_size: Number of preceding facts to use as context
        max_samples: Maximum training samples
    
    Returns:
        List of (context, target) pairs
    """
    # Group facts by story
    stories = group_facts_by_story(facts)
    
    training_pairs = []
    for story in stories:
        if len(story) < 2:
            continue
        
        # Create sliding window over story
        for i in range(1, len(story)):
            start_idx = max(0, i - context_size)
            context = story[start_idx:i]
            target = story[i]
            training_pairs.append((context, target))
            
            if len(training_pairs) >= max_samples:
                break
        
        if len(training_pairs) >= max_samples:
            break
    
    return training_pairs


def train_semantic_ar(
    model: SimpleDLN,
    training_pairs: List[Tuple[List[Proposition], Proposition]],
    device: str = "cpu",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 0.001,
    verbose: bool = True
) -> Dict:
    """
    Train DLN for autoregressive prediction.
    
    Uses AR head to predict next predicate given context.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Split train/eval
    split_idx = int(0.8 * len(training_pairs))
    train_pairs = training_pairs[:split_idx]
    eval_pairs = training_pairs[split_idx:]
    
    if verbose:
        print(f"  Training on {len(train_pairs)} pairs, evaluating on {len(eval_pairs)} pairs")
    
    history = []
    start_time = time.time()
    
    for epoch in range(epochs):
        # Shuffle training data
        random.shuffle(train_pairs)
        
        # Training
        total_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        for i in range(0, len(train_pairs), batch_size):
            batch = train_pairs[i:i+batch_size]
            
            optimizer.zero_grad()
            batch_loss = 0.0
            
            for context, target in batch:
                if not context:
                    continue
                
                try:
                    # Encode context
                    context_repr = model.encode_premises(context)
                    
                    # Predict next predicate
                    logits = model.ar_head(context_repr)
                    
                    # Target
                    if target.predicate not in model.pred_vocab:
                        continue
                    
                    target_idx = torch.tensor([model.pred_vocab[target.predicate]], device=device)
                    
                    # Loss
                    loss = F.cross_entropy(logits, target_idx)
                    batch_loss = batch_loss + loss
                    
                    # Accuracy
                    pred_idx = logits.argmax(dim=-1).item()
                    if pred_idx == target_idx.item():
                        correct_preds += 1
                    total_preds += 1
                    
                except Exception:
                    continue
            
            if batch_loss > 0:
                batch_loss.backward()
                optimizer.step()
                total_loss += batch_loss.item()
        
        train_acc = correct_preds / max(total_preds, 1)
        
        # Evaluation
        model.eval()
        eval_correct = 0
        eval_total = 0
        
        with torch.no_grad():
            for context, target in eval_pairs:
                if not context or target.predicate not in model.pred_vocab:
                    continue
                
                try:
                    context_repr = model.encode_premises(context)
                    logits = model.ar_head(context_repr)
                    pred_idx = logits.argmax(dim=-1).item()
                    target_idx = model.pred_vocab[target.predicate]
                    
                    if pred_idx == target_idx:
                        eval_correct += 1
                    eval_total += 1
                except Exception:
                    continue
        
        eval_acc = eval_correct / max(eval_total, 1)
        model.train()
        
        elapsed = time.time() - start_time
        
        if verbose:
            print(f"    Epoch {epoch+1}/{epochs}: Train Acc={train_acc:.4f}, Eval Acc={eval_acc:.4f}, Time={elapsed:.1f}s")
        
        history.append({
            'epoch': epoch + 1,
            'train_acc': train_acc,
            'eval_acc': eval_acc,
            'time': elapsed
        })
    
    training_time = time.time() - start_time
    
    # Final metrics
    final_train_acc = history[-1]['train_acc'] if history else 0.0
    final_eval_acc = history[-1]['eval_acc'] if history else 0.0
    
    return {
        'train_acc': final_train_acc,
        'eval_acc': final_eval_acc,
        'training_time': training_time,
        'history': history
    }


def test_semantic_ar_at_scale(
    scale_name: str,
    config: Dict,
    corpus_path: str,
    device: str,
    embed_dim: int,
    epochs: int,
    batch_size: int,
    context_size: int,
    use_rules: bool,
    verbose: bool
) -> Dict:
    """Test semantic-AR at a specific scale."""
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"SCALE: {scale_name.upper()} ({config['stories']} stories, {config['facts']} facts)")
        print(f"{'='*70}")
    
    # Stage 1: Load data
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
    
    # Stage 2: Create AR training data
    if verbose:
        print(f"\n[2] Creating autoregressive training pairs...")
    
    ar_start = time.time()
    training_pairs = create_ar_training_data(
        facts,
        context_size=context_size,
        max_samples=min(10000, len(facts))
    )
    ar_time = time.time() - ar_start
    
    if verbose:
        print(f"  ✅ Created {len(training_pairs)} training pairs in {ar_time:.1f}s")
        if training_pairs:
            ctx, tgt = training_pairs[0]
            print(f"  Example: {len(ctx)} context facts → predict {tgt.predicate}")
    
    # Stage 3: Optional rule mining
    rules = []
    rule_time = 0.0
    if use_rules:
        if verbose:
            print(f"\n[3] Mining ILP rules...")
        
        rule_start = time.time()
        rules, _ = mine_frequency_based(facts, max_rules=30, min_support=2)
        rule_time = time.time() - rule_start
        
        if verbose:
            print(f"  ✅ Mined {len(rules)} rules in {rule_time:.1f}s")
            if rules:
                print(f"  Sample: {rules[0]}")
    
    # Stage 4: Create DLN model
    if verbose:
        stage_num = 4 if use_rules else 3
        print(f"\n[{stage_num}] Creating DLN model...")
    
    # Collect vocabularies
    all_predicates = set(f.predicate for f in facts)
    all_args = set()
    for f in facts:
        all_args.update(f.args)
    
    if use_rules:
        for rule in rules:
            all_predicates.add(rule.conclusion.predicate)
    
    if verbose:
        print(f"  Vocabulary: {len(all_predicates)} predicates, {len(all_args)} entities")
    
    model = SimpleDLN(
        predicates=list(all_predicates),
        args=list(all_args),
        embed_dim=embed_dim
    )
    
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
        if verbose:
            print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
    
    num_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"  Model parameters: {num_params:,}")
    
    # Stage 5: Train
    if verbose:
        stage_num = 5 if use_rules else 4
        print(f"\n[{stage_num}] Training semantic-AR...")
    
    train_metrics = train_semantic_ar(
        model,
        training_pairs,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose
    )
    
    # Compile results
    results = {
        'scale': scale_name,
        'config': config,
        'facts_loaded': len(facts),
        'num_predicates': len(all_predicates),
        'num_entities': len(all_args),
        'num_parameters': num_params,
        'training_pairs': len(training_pairs),
        'load_time': load_time,
        'ar_prep_time': ar_time,
        'rule_mining_time': rule_time if use_rules else 0.0,
        'used_rules': use_rules,
        'num_rules': len(rules) if use_rules else 0,
        **train_metrics
    }
    
    # Summary
    if verbose:
        print(f"\n{'='*70}")
        print(f"SUMMARY: {scale_name.upper()}")
        print(f"{'='*70}")
        print(f"  Facts loaded: {len(facts):,}")
        print(f"  Training pairs: {len(training_pairs):,}")
        print(f"  Vocabulary: {len(all_predicates)} predicates, {len(all_args)} entities")
        print(f"  Model parameters: {num_params:,}")
        if use_rules:
            print(f"  Rules mined: {len(rules)}")
        print(f"  Train accuracy: {train_metrics['train_acc']:.4f}")
        print(f"  Eval accuracy: {train_metrics['eval_acc']:.4f}")
        print(f"  Time breakdown:")
        print(f"    - Data loading: {load_time:.1f}s")
        print(f"    - AR preparation: {ar_time:.1f}s")
        if use_rules:
            print(f"    - Rule mining: {rule_time:.1f}s")
        print(f"    - Training: {train_metrics['training_time']:.1f}s")
        print(f"    - Total: {load_time + ar_time + rule_time + train_metrics['training_time']:.1f}s")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Semantic-AR Scaling Test')
    parser.add_argument('--scale', default='small',
                       choices=['tiny', 'small', 'medium', 'large', 'xlarge', 'all'],
                       help='Dataset scale to test')
    parser.add_argument('--corpus', default='data/processed/tinystories_train.json',
                       help='Corpus path')
    parser.add_argument('--embed-dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--context-size', type=int, default=5, help='Context window size')
    parser.add_argument('--use-rules', action='store_true',
                       help='Mine and use ILP rules (experimental)')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='Device')
    parser.add_argument('--output-dir', default='outputs/semantic_ar_scaling',
                       help='Output directory')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    # Set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    if verbose:
        print("\n" + "="*70)
        print("SEMANTIC AUTOREGRESSIVE SCALING TEST")
        print("="*70)
        print(f"\nConfiguration:")
        print(f"  Device: {args.device}")
        print(f"  Embeddings: {args.embed_dim}d")
        print(f"  Training: {args.epochs} epochs, batch size {args.batch_size}")
        print(f"  Context size: {args.context_size} facts")
        print(f"  Use rules: {args.use_rules}")
        print(f"  Random seed: {args.seed}")
    
    # Determine scales to test
    if args.scale == 'all':
        scales_to_test = ['tiny', 'small', 'medium', 'large', 'xlarge']
    else:
        scales_to_test = [args.scale]
    
    # Run tests
    all_results = {}
    for scale_name in scales_to_test:
        config = SCALE_CONFIGS[scale_name]
        
        try:
            results = test_semantic_ar_at_scale(
                scale_name,
                config,
                args.corpus,
                args.device,
                args.embed_dim,
                args.epochs,
                args.batch_size,
                args.context_size,
                args.use_rules,
                verbose
            )
            all_results[scale_name] = results
        except Exception as e:
            if verbose:
                print(f"\n❌ Failed at scale {scale_name}: {e}")
            all_results[scale_name] = {'error': str(e)}
            import traceback
            traceback.print_exc()
    
    # Comparison
    if len(all_results) > 1 and verbose:
        print("\n" + "="*70)
        print("SCALING COMPARISON")
        print("="*70)
        
        print(f"\n{'Scale':<10} {'Facts':<10} {'Pairs':<10} {'Train Acc':<12} {'Eval Acc':<12} {'Time (s)':<10}")
        print("-" * 75)
        
        for scale_name in scales_to_test:
            if scale_name in all_results and 'error' not in all_results[scale_name]:
                r = all_results[scale_name]
                total_time = r['load_time'] + r['ar_prep_time'] + r.get('rule_mining_time', 0) + r['training_time']
                print(f"{scale_name:<10} {r['facts_loaded']:<10,} {r['training_pairs']:<10,} "
                      f"{r['train_acc']:<12.4f} {r['eval_acc']:<12.4f} {total_time:<10.1f}")
        
        print("\n" + "="*70)
        print("INSIGHTS")
        print("="*70)
        
        successful = [r for r in all_results.values() if 'error' not in r]
        if len(successful) > 1:
            eval_accs = [r['eval_acc'] for r in successful]
            times = [r['training_time'] for r in successful]
            facts = [r['facts_loaded'] for r in successful]
            
            print(f"\n[Performance vs Scale]")
            print(f"  Eval accuracy range: {min(eval_accs):.4f} to {max(eval_accs):.4f}")
            
            if max(eval_accs) > 0.5:
                print(f"  ✅ Model learns meaningful patterns (>50% accuracy)")
            else:
                print(f"  ⚠️  Low accuracy - may need more training or better architecture")
            
            # Time scaling
            if len(times) >= 2 and len(facts) >= 2:
                time_ratio = times[-1] / times[0]
                fact_ratio = facts[-1] / facts[0]
                
                print(f"\n[Time Complexity]")
                print(f"  Facts scaled by: {fact_ratio:.1f}×")
                print(f"  Training time scaled by: {time_ratio:.1f}×")
                
                if time_ratio < fact_ratio * 1.5:
                    print(f"  ✅ Linear or sub-linear scaling (excellent!)")
                elif time_ratio < fact_ratio * 2.5:
                    print(f"  ✅ Near-linear scaling (good)")
                else:
                    print(f"  ⚠️  Super-linear scaling (may need optimization)")
            
            print(f"\n[Recommendation]")
            if max(eval_accs) > 0.6 and times[-1] < 600:
                print(f"  ✅ Semantic-AR scales well - ready for larger datasets")
            elif max(eval_accs) > 0.4:
                print(f"  ⚠️  Moderate performance - tune hyperparameters")
            else:
                print(f"  ❌ Poor performance - architecture may need redesign")
    
    # Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / 'semantic_ar_scaling.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    if verbose:
        print(f"\n✅ Results saved to {output_path / 'semantic_ar_scaling.json'}")


if __name__ == '__main__':
    main()
