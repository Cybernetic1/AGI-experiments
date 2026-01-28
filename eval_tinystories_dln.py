#!/usr/bin/env python3
"""
Evaluate trained DLN on simpler tasks to see if it learned anything useful.
"""

import torch
import torch.nn as nn
from train_tinystories_ar_rl import DLNWithHeads, load_story_sequences, build_vocabularies
import random
import argparse


def test_predicate_prediction(model, sequences, device):
    """Test: Can model predict predicate (ignoring exact args)?"""
    print("\n" + "="*70)
    print("TEST 1: Predicate Prediction (Easier)")
    print("="*70)
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for seq in sequences:
            if len(seq) < 2:
                continue
            
            split = random.randint(1, len(seq) - 1)
            context = seq[:split]
            target = seq[split]
            
            # Predict
            pred_logits, _, _ = model.forward_ar(context)
            predicted_pred = pred_logits.argmax().item()
            
            # Check
            pred_vocab = model.pred_vocab
            target_pred_id = pred_vocab.get(target.predicate, 0)
            
            if predicted_pred == target_pred_id:
                correct += 1
            total += 1
    
    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\nPredicate-only accuracy: {accuracy:.1f}% ({correct}/{total})")
    
    # Baseline: random guessing
    num_predicates = len(model.pred_vocab)
    baseline = 100.0 / num_predicates
    print(f"Random baseline: {baseline:.1f}%")
    
    if accuracy > baseline * 1.5:
        print("✓ Model is learning (better than random)")
    else:
        print("✗ Model not significantly better than random")
    
    return accuracy


def test_sequence_coherence(model, sequences, device):
    """Test: Can model distinguish real sequences from shuffled ones?"""
    print("\n" + "="*70)
    print("TEST 2: Sequence Coherence Detection")
    print("="*70)
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for seq in sequences:
            if len(seq) < 3:
                continue
            
            # Real sequence
            real_q = model.forward_rl(seq).item()
            
            # Shuffled sequence
            shuffled = seq[:]
            random.shuffle(shuffled)
            shuffled_q = model.forward_rl(shuffled).item()
            
            # Model should rate real higher
            if real_q > shuffled_q:
                correct += 1
            total += 1
    
    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\nCoherence detection accuracy: {accuracy:.1f}% ({correct}/{total})")
    print(f"Random baseline: 50.0%")
    
    if accuracy > 60:
        print("✓ Model learned coherence patterns")
    else:
        print("✗ Model cannot distinguish coherent from incoherent")
    
    return accuracy


def test_few_shot_learning(model, sequences, device):
    """Test: Given 1 example, can model predict next?"""
    print("\n" + "="*70)
    print("TEST 3: Few-Shot Pattern Recognition")
    print("="*70)
    
    # Group sequences by predicate patterns
    pattern_groups = {}
    for seq in sequences:
        if len(seq) >= 2:
            pattern = tuple(f.predicate for f in seq[:2])
            if pattern not in pattern_groups:
                pattern_groups[pattern] = []
            pattern_groups[pattern].append(seq)
    
    print(f"\nFound {len(pattern_groups)} unique patterns")
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for pattern, seqs in pattern_groups.items():
            if len(seqs) < 2:
                continue
            
            # Use first as example, test on second
            example = seqs[0]
            test = seqs[1]
            
            if len(example) < 2 or len(test) < 2:
                continue
            
            # Show model the pattern from example
            context = example[:1]
            
            # Predict next
            pred_logits, _, _ = model.forward_ar(context)
            predicted_pred = pred_logits.argmax().item()
            
            # Check against test
            target_pred_id = model.pred_vocab.get(test[1].predicate, 0)
            
            if predicted_pred == target_pred_id:
                correct += 1
            total += 1
    
    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\nFew-shot accuracy: {accuracy:.1f}% ({correct}/{total})")
    
    return accuracy


def compare_with_random():
    """Compare trained model vs random initialization."""
    print("\n" + "="*70)
    print("COMPARISON: Trained vs Random Initialization")
    print("="*70)
    print("\nThis would show if training improved the model.")
    print("(Not implemented yet - need to save trained model)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stories", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    print("="*70)
    print("EVALUATE TRAINED DLN")
    print("="*70)
    
    # Load data
    print(f"\nLoading {args.stories} stories...")
    sequences = load_story_sequences(max_stories=args.stories)
    print(f"  Loaded {len(sequences)} sequences")
    
    # Build vocabularies
    predicates, args_list = build_vocabularies(sequences)
    pred_vocab = {p: i for i, p in enumerate(predicates)}
    arg_vocab = {a: i for i, a in enumerate(args_list)}
    
    # Create and load model (in practice, load saved weights)
    # For now, create fresh model
    print("\nCreating fresh model (need to load trained weights in practice)...")
    model = DLNWithHeads(pred_vocab, arg_vocab, num_rules=6, embed_dim=8)
    model = model.to(device)
    
    print("⚠ WARNING: Using untrained model for testing")
    print("In practice, load saved model from train_tinystories_ar_rl.py")
    
    # Run tests
    pred_acc = test_predicate_prediction(model, sequences, device)
    coh_acc = test_sequence_coherence(model, sequences, device)
    fs_acc = test_few_shot_learning(model, sequences, device)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nPredicate prediction: {pred_acc:.1f}%")
    print(f"Coherence detection: {coh_acc:.1f}%")
    print(f"Few-shot learning: {fs_acc:.1f}%")
    
    print("\nNext steps:")
    print("1. Save trained model from train_tinystories_ar_rl.py")
    print("2. Load it here and run evaluation")
    print("3. Compare with Transformer baseline trained same way")
    print("4. Vary num_rules for parameter sweep")


if __name__ == "__main__":
    main()
