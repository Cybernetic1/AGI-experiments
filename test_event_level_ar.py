"""
Event-Level Semantic Autoregressive Training
============================================

Proper semantic-AR: predict next EVENT's propositions, not next atomic predicate.

Architecture:
- Given events E1...En (each event = set of propositions)
- Predict all propositions for event E(n+1)
- Captures sentence-level discourse structure

Example:
  Context: 
    E1: type(e1, 'go'), agent(e1, 'lily'), location(e1, 'park')
    E2: type(e2, 'see'), agent(e2, 'lily'), patient(e2, 'bunny')
  Target:
    E3: type(e3, 'pet'), agent(e3, 'lily'), patient(e3, 'bunny')

Prediction tasks:
1. Next event type (verb): What action happens next?
2. Event arguments: Who/what/where/when for that action?

Usage:
    python test_event_level_ar.py --scale medium --device cuda
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import time
import random
from pathlib import Path
from typing import List, Dict, Tuple, Set
import argparse
from collections import defaultdict

from pipelines.tinystories_pipeline import load_tinystories_facts
from logic_core import Proposition
from dln import SimpleDLN


SCALE_CONFIGS = {
    'tiny': {'stories': 50, 'facts': 5000},
    'small': {'stories': 100, 'facts': 10000},
    'medium': {'stories': 500, 'facts': 50000},
    'large': {'stories': 1000, 'facts': 100000},
    'xlarge': {'stories': 2000, 'facts': 200000},
    'full': {'stories': 999999, 'facts': 9999999},  # Load everything available
}


def group_facts_into_events(facts: List[Proposition]) -> Dict[str, List[Proposition]]:
    """
    Group propositions by event ID.
    
    Returns:
        Dict[event_id, List[Proposition]]
    """
    events = defaultdict(list)
    
    for fact in facts:
        # Event ID is typically first arg for predicates like agent, patient, etc.
        # For type predicates: type(e123, 'verb') - e123 is event ID
        if fact.args and fact.args[0].startswith('e'):
            event_id = fact.args[0]
            events[event_id].append(fact)
        elif fact.predicate == 'type' and len(fact.args) >= 2:
            # Entity type declarations: type(entity_name, 'entity')
            # Group these separately
            events[f"entity_{fact.args[0]}"].append(fact)
    
    return events


def create_event_sequences(events: Dict[str, List[Proposition]], min_event_props: int = 2) -> List[List[List[Proposition]]]:
    """
    Create sequences of events from a story.
    
    Filter out events with too few propositions (incomplete parses).
    
    Returns:
        List of event sequences (each sequence is a story)
    """
    # Sort events by ID to maintain temporal order
    event_ids = sorted([eid for eid in events.keys() if eid.startswith('e')])
    
    # Group consecutive events into stories (assuming e1, e2, e3... are sequential)
    stories = []
    current_story = []
    
    for event_id in event_ids:
        event_props = events[event_id]
        
        # Filter incomplete events
        if len(event_props) < min_event_props:
            continue
        
        current_story.append(event_props)
        
        # Start new story when event IDs have large gap or reach reasonable story length
        if len(current_story) >= 20:  # Max story length
            stories.append(current_story)
            current_story = []
    
    if len(current_story) >= 2:  # Need at least 2 events for context + target
        stories.append(current_story)
    
    return stories


def create_event_ar_training_data(
    facts: List[Proposition],
    context_size: int = 3,
    max_samples: int = 10000
) -> List[Tuple[List[List[Proposition]], List[Proposition]]]:
    """
    Create event-level AR training pairs.
    
    Returns:
        List of (context_events, target_event) pairs
        where context_events is a list of event proposition lists
        and target_event is the next event's propositions
    """
    # Group into events
    events = group_facts_into_events(facts)
    
    # Create event sequences (stories)
    stories = create_event_sequences(events, min_event_props=2)
    
    # Create training pairs
    training_pairs = []
    
    for story in stories:
        if len(story) < 2:
            continue
        
        # Sliding window over story events
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


def train_event_ar(
    model: SimpleDLN,
    training_pairs: List[Tuple[List[List[Proposition]], List[Proposition]]],
    device: str = "cpu",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 0.001,
    verbose: bool = True
) -> Dict:
    """
    Train DLN for event-level autoregressive prediction.
    
    Task: Given context events, predict the TYPE (verb) of next event.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Split train/eval
    split_idx = int(0.8 * len(training_pairs))
    train_pairs = training_pairs[:split_idx]
    eval_pairs = training_pairs[split_idx:]
    
    if verbose:
        print(f"  Training on {len(train_pairs)} event pairs, evaluating on {len(eval_pairs)} pairs")
    
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
            
            for context_events, target_event in batch:
                if not context_events or not target_event:
                    continue
                
                try:
                    # Flatten context events into single proposition list
                    context_props = []
                    for event in context_events:
                        context_props.extend(event)
                    
                    if not context_props:
                        continue
                    
                    # Encode context
                    context_repr = model.encode_premises(context_props)
                    
                    # Find target event's TYPE (verb)
                    target_type = None
                    for prop in target_event:
                        if prop.predicate == 'type' and len(prop.args) >= 2:
                            target_type = prop.args[1]  # The verb
                            break
                    
                    if not target_type or target_type not in model.pred_vocab:
                        continue
                    
                    # Predict next event type
                    logits = model.ar_head(context_repr)
                    target_idx = torch.tensor([model.pred_vocab[target_type]], device=device)
                    
                    # Loss
                    loss = F.cross_entropy(logits, target_idx)
                    batch_loss = batch_loss + loss
                    
                    # Accuracy
                    pred_idx = logits.argmax(dim=-1).item()
                    if pred_idx == target_idx.item():
                        correct_preds += 1
                    total_preds += 1
                    
                except Exception as e:
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
            for context_events, target_event in eval_pairs:
                if not context_events or not target_event:
                    continue
                
                try:
                    context_props = []
                    for event in context_events:
                        context_props.extend(event)
                    
                    if not context_props:
                        continue
                    
                    context_repr = model.encode_premises(context_props)
                    
                    target_type = None
                    for prop in target_event:
                        if prop.predicate == 'type' and len(prop.args) >= 2:
                            target_type = prop.args[1]
                            break
                    
                    if not target_type or target_type not in model.pred_vocab:
                        continue
                    
                    logits = model.ar_head(context_repr)
                    pred_idx = logits.argmax(dim=-1).item()
                    target_idx = model.pred_vocab[target_type]
                    
                    if pred_idx == target_idx:
                        eval_correct += 1
                    eval_total += 1
                except Exception:
                    continue
        
        eval_acc = eval_correct / max(eval_total, 1)
        model.train()
        
        elapsed = time.time() - start_time
        
        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"    Epoch {epoch+1}/{epochs}: Train Acc={train_acc:.4f}, Eval Acc={eval_acc:.4f}, Time={elapsed:.1f}s")
        
        history.append({
            'epoch': epoch + 1,
            'train_acc': train_acc,
            'eval_acc': eval_acc,
            'time': elapsed
        })
    
    training_time = time.time() - start_time
    
    final_train_acc = history[-1]['train_acc'] if history else 0.0
    final_eval_acc = history[-1]['eval_acc'] if history else 0.0
    
    return {
        'train_acc': final_train_acc,
        'eval_acc': final_eval_acc,
        'training_time': training_time,
        'history': history
    }


def test_event_ar_at_scale(
    scale_name: str,
    config: Dict,
    corpus_path: str,
    device: str,
    embed_dim: int,
    epochs: int,
    batch_size: int,
    context_size: int,
    verbose: bool
) -> Dict:
    """Test event-level AR at a specific scale."""
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"EVENT-LEVEL AR: {scale_name.upper()} ({config['stories']} stories, {config['facts']} facts)")
        print(f"{'='*70}")
    
    # Load data
    if verbose:
        print(f"\n[1] Loading corpus...")
    
    from pipelines.tinystories_pipeline import load_tinystories_facts
    
    load_start = time.time()
    facts = load_tinystories_facts(
        max_stories=config['stories'],
        max_facts=config['facts'],
        path=corpus_path
    )
    load_time = time.time() - load_start
    
    if verbose:
        print(f"  ✅ Loaded {len(facts)} facts in {load_time:.1f}s")
    
    # Create event-level training data
    if verbose:
        print(f"\n[2] Creating event-level training pairs...")
    
    ar_start = time.time()
    training_pairs = create_event_ar_training_data(
        facts,
        context_size=context_size,
        max_samples=min(20000, len(facts))
    )
    ar_time = time.time() - ar_start
    
    if verbose:
        print(f"  ✅ Created {len(training_pairs)} event pairs in {ar_time:.1f}s")
        if training_pairs:
            ctx, tgt = training_pairs[0]
            print(f"  Example:")
            print(f"    Context: {len(ctx)} events")
            for i, event in enumerate(ctx[:2], 1):  # Show first 2 events
                print(f"      Event {i}: {len(event)} propositions")
            print(f"    Target: {len(tgt)} propositions")
            target_type = next((p.args[1] for p in tgt if p.predicate == 'type' and len(p.args) >= 2), None)
            print(f"      → Predict next action: {target_type}")
    
    # Create model
    if verbose:
        print(f"\n[3] Creating DLN model...")
    
    all_predicates = set(f.predicate for f in facts)
    all_args = set()
    for f in facts:
        all_args.update(f.args)
    
    # Add verb types to vocabulary
    for fact in facts:
        if fact.predicate == 'type' and len(fact.args) >= 2:
            all_predicates.add(fact.args[1])  # Add verbs
    
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
    
    # Train
    if verbose:
        print(f"\n[4] Training event-level AR...")
    
    train_metrics = train_event_ar(
        model,
        training_pairs,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose
    )
    
    # Results
    results = {
        'scale': scale_name,
        'config': config,
        'facts_loaded': len(facts),
        'num_predicates': len(all_predicates),
        'num_entities': len(all_args),
        'num_parameters': num_params,
        'event_pairs': len(training_pairs),
        'load_time': load_time,
        'ar_prep_time': ar_time,
        **train_metrics
    }
    
    # Summary
    if verbose:
        print(f"\n{'='*70}")
        print(f"SUMMARY: {scale_name.upper()}")
        print(f"{'='*70}")
        print(f"  Facts: {len(facts):,}")
        print(f"  Event pairs: {len(training_pairs):,}")
        print(f"  Vocabulary: {len(all_predicates)} predicates, {len(all_args)} entities")
        print(f"  Model parameters: {num_params:,}")
        print(f"  Train accuracy: {train_metrics['train_acc']:.4f}")
        print(f"  Eval accuracy: {train_metrics['eval_acc']:.4f}")
        print(f"  Total time: {load_time + ar_time + train_metrics['training_time']:.1f}s")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Event-Level Semantic-AR Test')
    parser.add_argument('--scale', default='small',
                       choices=['tiny', 'small', 'medium', 'large', 'xlarge', 'full', 'all'],
                       help='Dataset scale')
    parser.add_argument('--corpus', default='data/processed/tinystories_train.json',
                       help='Corpus path')
    parser.add_argument('--embed-dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--context-size', type=int, default=3, help='Number of context events')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='Device')
    parser.add_argument('--output-dir', default='outputs/event_level_ar',
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
        print("EVENT-LEVEL SEMANTIC AUTOREGRESSIVE TEST")
        print("="*70)
        print(f"\nConfiguration:")
        print(f"  Device: {args.device}")
        print(f"  Embeddings: {args.embed_dim}d")
        print(f"  Training: {args.epochs} epochs, batch size {args.batch_size}")
        print(f"  Context: {args.context_size} events")
        print(f"  Random seed: {args.seed}")
    
    # Determine scales
    if args.scale == 'all':
        scales_to_test = ['tiny', 'small', 'medium', 'large', 'xlarge', 'full']
    else:
        scales_to_test = [args.scale]
    
    # Run tests
    all_results = {}
    for scale_name in scales_to_test:
        config = SCALE_CONFIGS[scale_name]
        
        try:
            results = test_event_ar_at_scale(
                scale_name,
                config,
                args.corpus,
                args.device,
                args.embed_dim,
                args.epochs,
                args.batch_size,
                args.context_size,
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
        print("EVENT-LEVEL AR COMPARISON")
        print("="*70)
        
        print(f"\n{'Scale':<10} {'Facts':<10} {'Events':<10} {'Train Acc':<12} {'Eval Acc':<12} {'Time (s)':<10}")
        print("-" * 75)
        
        for scale_name in scales_to_test:
            if scale_name in all_results and 'error' not in all_results[scale_name]:
                r = all_results[scale_name]
                total_time = r['load_time'] + r['ar_prep_time'] + r['training_time']
                print(f"{scale_name:<10} {r['facts_loaded']:<10,} {r['event_pairs']:<10,} "
                      f"{r['train_acc']:<12.4f} {r['eval_acc']:<12.4f} {total_time:<10.1f}")
        
        print(f"\n{'='*70}")
        print("INSIGHT: Event-level vs Atomic-level AR")
        print(f"{'='*70}")
        print("\nEvent-level AR advantages:")
        print("  ✅ Predicts semantically meaningful units (actions)")
        print("  ✅ Captures discourse structure (story progression)")
        print("  ✅ Clearer evaluation metric (next action type)")
        print("\nCompare eval accuracy to atomic-level test:")
        print("  Previous (atomic): ~59-77% (mixed intra/inter event)")
        print(f"  Current (event): {all_results[scales_to_test[-1]]['eval_acc']:.1%} (pure inter-event)")
    
    # Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / 'event_ar_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    if verbose:
        print(f"\n✅ Results saved to {output_path / 'event_ar_results.json'}")


if __name__ == '__main__':
    main()
