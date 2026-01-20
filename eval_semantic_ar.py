"""
Semantic-AR Detailed Evaluation
================================

Re-train and analyze semantic-AR model with detailed error analysis.

Features:
- Saves best model checkpoint
- Shows per-predicate accuracy
- Analyzes error patterns (what contexts lead to wrong predictions)
- Visualizes confusion matrix
- Exports misclassified examples for inspection

Usage:
    # Train and evaluate at large scale
    python eval_semantic_ar.py --scale large --device cuda
    
    # Load existing model and analyze
    python eval_semantic_ar.py --load outputs/semantic_ar/model.pt --device cuda
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
from collections import defaultdict, Counter

from pipelines.tinystories_pipeline import load_tinystories_facts
from logic_core import Proposition
from dln import SimpleDLN
from test_semantic_ar_scaling import (
    create_ar_training_data,
    SCALE_CONFIGS,
    group_facts_by_story
)


def train_and_save_model(
    facts: List[Proposition],
    device: str,
    embed_dim: int,
    epochs: int,
    batch_size: int,
    context_size: int,
    output_dir: Path,
    verbose: bool
) -> Tuple[SimpleDLN, List[Tuple[List[Proposition], Proposition]]]:
    """Train model and save checkpoint."""
    
    if verbose:
        print(f"\n[1] Creating training data...")
    
    training_pairs = create_ar_training_data(
        facts,
        context_size=context_size,
        max_samples=20000  # More samples for better training
    )
    
    if verbose:
        print(f"  ✅ Created {len(training_pairs)} training pairs")
    
    # Split train/eval
    split_idx = int(0.8 * len(training_pairs))
    train_pairs = training_pairs[:split_idx]
    eval_pairs = training_pairs[split_idx:]
    
    if verbose:
        print(f"\n[2] Creating model...")
    
    # Vocabularies
    all_predicates = set(f.predicate for f in facts)
    all_args = set()
    for f in facts:
        all_args.update(f.args)
    
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
    
    if verbose:
        print(f"\n[3] Training for {epochs} epochs...")
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    best_eval_acc = 0.0
    best_epoch = 0
    
    for epoch in range(epochs):
        random.shuffle(train_pairs)
        
        # Training
        total_correct = 0
        total_preds = 0
        
        for i in range(0, len(train_pairs), batch_size):
            batch = train_pairs[i:i+batch_size]
            
            optimizer.zero_grad()
            batch_loss = 0.0
            
            for context, target in batch:
                if not context or target.predicate not in model.pred_vocab:
                    continue
                
                try:
                    context_repr = model.encode_premises(context)
                    logits = model.ar_head(context_repr)
                    target_idx = torch.tensor([model.pred_vocab[target.predicate]], device=device)
                    
                    loss = F.cross_entropy(logits, target_idx)
                    batch_loss = batch_loss + loss
                    
                    pred_idx = logits.argmax(dim=-1).item()
                    if pred_idx == target_idx.item():
                        total_correct += 1
                    total_preds += 1
                except Exception:
                    continue
            
            if batch_loss > 0:
                batch_loss.backward()
                optimizer.step()
        
        train_acc = total_correct / max(total_preds, 1)
        
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
        
        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"    Epoch {epoch+1}/{epochs}: Train Acc={train_acc:.4f}, Eval Acc={eval_acc:.4f}")
        
        # Save best model
        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'eval_acc': eval_acc,
                'pred_vocab': model.pred_vocab,
                'arg_vocab': model.arg_vocab,
            }, output_dir / 'best_model.pt')
    
    if verbose:
        print(f"\n  ✅ Training complete")
        print(f"  Best eval accuracy: {best_eval_acc:.4f} at epoch {best_epoch}")
    
    return model, eval_pairs


def detailed_evaluation(
    model: SimpleDLN,
    eval_pairs: List[Tuple[List[Proposition], Proposition]],
    device: str,
    output_dir: Path,
    verbose: bool
) -> Dict:
    """Perform detailed error analysis."""
    
    if verbose:
        print(f"\n[4] Detailed evaluation...")
    
    model.eval()
    
    # Per-predicate statistics
    pred_correct = defaultdict(int)
    pred_total = defaultdict(int)
    pred_confusion = defaultdict(Counter)  # actual -> Counter(predicted)
    
    # Error examples
    errors = []
    correct_examples = []
    
    with torch.no_grad():
        for context, target in eval_pairs:
            if not context or target.predicate not in model.pred_vocab:
                continue
            
            try:
                context_repr = model.encode_premises(context)
                logits = model.ar_head(context_repr)
                pred_idx = logits.argmax(dim=-1).item()
                pred_name = model.pred_names[pred_idx]
                target_idx = model.pred_vocab[target.predicate]
                
                # Statistics
                pred_total[target.predicate] += 1
                pred_confusion[target.predicate][pred_name] += 1
                
                if pred_idx == target_idx:
                    pred_correct[target.predicate] += 1
                    if len(correct_examples) < 20:
                        correct_examples.append({
                            'context': [str(p) for p in context],
                            'predicted': pred_name,
                            'actual': target.predicate,
                            'confidence': torch.softmax(logits, dim=-1)[0, pred_idx].item()
                        })
                else:
                    if len(errors) < 50:
                        errors.append({
                            'context': [str(p) for p in context],
                            'predicted': pred_name,
                            'actual': target.predicate,
                            'confidence': torch.softmax(logits, dim=-1)[0, pred_idx].item()
                        })
            except Exception:
                continue
    
    # Compute per-predicate accuracy
    pred_accuracy = {}
    for pred in pred_total:
        acc = pred_correct[pred] / pred_total[pred]
        pred_accuracy[pred] = {
            'accuracy': acc,
            'correct': pred_correct[pred],
            'total': pred_total[pred]
        }
    
    # Overall stats
    total_correct = sum(pred_correct.values())
    total_preds = sum(pred_total.values())
    overall_acc = total_correct / total_preds if total_preds > 0 else 0.0
    
    # Display results
    if verbose:
        print(f"\n{'='*70}")
        print(f"DETAILED EVALUATION RESULTS")
        print(f"{'='*70}")
        print(f"\nOverall accuracy: {overall_acc:.4f} ({total_correct}/{total_preds})")
        
        print(f"\n[Per-Predicate Accuracy]")
        print(f"{'Predicate':<25} {'Accuracy':<12} {'Correct/Total':<15}")
        print("-" * 55)
        
        sorted_preds = sorted(pred_accuracy.items(), key=lambda x: x[1]['accuracy'])
        for pred, stats in sorted_preds:
            print(f"{pred:<25} {stats['accuracy']:<12.4f} {stats['correct']}/{stats['total']}")
        
        print(f"\n[Most Common Confusions]")
        for actual_pred in sorted_preds[:5]:  # Show worst 5
            pred_name = actual_pred[0]
            confusions = pred_confusion[pred_name].most_common(3)
            print(f"\n  {pred_name}:")
            for confused_pred, count in confusions:
                if confused_pred != pred_name:
                    print(f"    → {confused_pred}: {count} times")
        
        print(f"\n[Sample Errors] (showing first 5)")
        for i, err in enumerate(errors[:5], 1):
            print(f"\n  Error {i}:")
            print(f"    Context: {', '.join(err['context'][-3:])}")  # Last 3 for brevity
            print(f"    Predicted: {err['predicted']} (confidence: {err['confidence']:.3f})")
            print(f"    Actual: {err['actual']}")
        
        print(f"\n[Sample Correct] (showing first 3)")
        for i, ex in enumerate(correct_examples[:3], 1):
            print(f"\n  Correct {i}:")
            print(f"    Context: {', '.join(ex['context'][-3:])}")
            print(f"    Predicted: {ex['predicted']} (confidence: {ex['confidence']:.3f})")
    
    # Save detailed results
    results = {
        'overall_accuracy': overall_acc,
        'total_correct': total_correct,
        'total_predictions': total_preds,
        'per_predicate_accuracy': pred_accuracy,
        'confusion_matrix': {k: dict(v) for k, v in pred_confusion.items()},
        'error_examples': errors,
        'correct_examples': correct_examples
    }
    
    with open(output_dir / 'detailed_eval.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    if verbose:
        print(f"\n✅ Detailed results saved to {output_dir / 'detailed_eval.json'}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Semantic-AR Detailed Evaluation')
    parser.add_argument('--scale', default='large',
                       choices=['tiny', 'small', 'medium', 'large', 'xlarge'],
                       help='Dataset scale')
    parser.add_argument('--corpus', default='data/processed/tinystories_train.json',
                       help='Corpus path')
    parser.add_argument('--load', type=str, help='Load existing model checkpoint')
    parser.add_argument('--embed-dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--context-size', type=int, default=5, help='Context window')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='Device')
    parser.add_argument('--output-dir', default='outputs/semantic_ar_eval',
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
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("\n" + "="*70)
        print("SEMANTIC-AR DETAILED EVALUATION")
        print("="*70)
    
    # Load data
    config = SCALE_CONFIGS[args.scale]
    if verbose:
        print(f"\nLoading {args.scale} scale ({config['stories']} stories, {config['facts']} facts)...")
    
    facts = load_tinystories_facts(
        max_stories=config['stories'],
        max_facts=config['facts'],
        path=args.corpus
    )
    
    if verbose:
        print(f"  ✅ Loaded {len(facts)} facts")
    
    # Train or load model
    if args.load:
        if verbose:
            print(f"\n[Loading model from {args.load}]")
        
        checkpoint = torch.load(args.load)
        
        # Recreate vocabularies
        all_predicates = set(f.predicate for f in facts)
        all_args = set()
        for f in facts:
            all_args.update(f.args)
        
        model = SimpleDLN(
            predicates=list(all_predicates),
            args=list(all_args),
            embed_dim=args.embed_dim
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if args.device == "cuda" and torch.cuda.is_available():
            model = model.cuda()
        
        # Create eval pairs
        training_pairs = create_ar_training_data(facts, context_size=args.context_size, max_samples=20000)
        split_idx = int(0.8 * len(training_pairs))
        eval_pairs = training_pairs[split_idx:]
        
        if verbose:
            print(f"  ✅ Loaded model (eval acc: {checkpoint['eval_acc']:.4f})")
    else:
        model, eval_pairs = train_and_save_model(
            facts,
            args.device,
            args.embed_dim,
            args.epochs,
            args.batch_size,
            args.context_size,
            output_path,
            verbose
        )
    
    # Detailed evaluation
    results = detailed_evaluation(
        model,
        eval_pairs,
        args.device,
        output_path,
        verbose
    )
    
    if verbose:
        print(f"\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()
