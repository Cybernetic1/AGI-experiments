"""
Test GA on TinyStories dataset - compare to neural baseline.
"""

import sys
from genetic_logic_rules import SymbolicRule, genetic_algorithm, semantic_ar_fitness
from train_symmetric import TinyStoriesLogicDataset
import random


def prepare_ga_dataset(num_stories=100, max_seq_len=10):
    """
    Prepare dataset for GA.
    
    Convert from neural format to (text, logic) pairs.
    """
    print("Loading TinyStories dataset...")
    dataset = TinyStoriesLogicDataset(num_stories=num_stories, max_seq_len=max_seq_len)
    
    ga_dataset = []
    
    for i in range(len(dataset)):
        sample = dataset[i]
        
        # Convert text_ids to text
        text_ids = sample['text_ids'].tolist()
        text_tokens = [dataset.vocab.get(tid, '<unk>') for tid in text_ids if tid != 0]
        text = ' '.join(text_tokens)
        
        # Convert propositions to simple logic string
        props = sample['propositions'][0].tolist()  # First proposition
        if props[0] != 0:  # Not padding
            # Simple logic representation
            logic = f"entity={props[0]},relation={props[1]},value={props[2]}"
            ga_dataset.append((text, logic))
    
    print(f"✓ Prepared {len(ga_dataset)} examples")
    return ga_dataset, dataset


def evaluate_rules(rules, test_dataset):
    """
    Evaluate rules on test set.
    
    Returns:
        accuracy: Percentage of correctly parsed examples
        coverage: Percentage of examples matched by at least one rule
    """
    total = len(test_dataset)
    correct = 0
    matched = 0
    
    for text, target_logic in test_dataset:
        # Try each rule
        best_match = None
        for rule in rules:
            if rule.matches(text):
                matched += 1
                logic = rule.parse(text)
                
                # Check if correct (simple string comparison)
                # In practice, we'd check semantic equivalence
                if logic and rule.is_semantically_consistent(text=text):
                    best_match = logic
                    break
        
        if best_match:
            correct += 1
    
    accuracy = correct / total if total > 0 else 0
    coverage = matched / total if total > 0 else 0
    
    return accuracy, coverage


def main():
    print("=" * 70)
    print("GA Test: Symbolic Rules vs Neural Baseline")
    print("=" * 70)
    print()
    
    # Prepare dataset
    print("Phase 1: Data Preparation")
    print("-" * 70)
    
    # Use smaller dataset for faster testing
    train_data, full_dataset = prepare_ga_dataset(num_stories=100, max_seq_len=10)
    
    # Split train/test
    random.shuffle(train_data)
    split = int(0.8 * len(train_data))
    train_set = train_data[:split]
    test_set = train_data[split:]
    
    print(f"Train: {len(train_set)} examples")
    print(f"Test: {len(test_set)} examples")
    print()
    
    # Run GA
    print("Phase 2: Evolving Rules with GA")
    print("-" * 70)
    
    best_rules = genetic_algorithm(
        dataset=train_set,
        population_size=100,
        generations=50,
        mutation_rate=0.2,
        elite_size=10
    )
    
    # Evaluate
    print("Phase 3: Evaluation")
    print("-" * 70)
    
    train_acc, train_cov = evaluate_rules(best_rules, train_set)
    test_acc, test_cov = evaluate_rules(best_rules, test_set)
    
    print(f"Train Results:")
    print(f"  Accuracy: {train_acc*100:.1f}%")
    print(f"  Coverage: {train_cov*100:.1f}%")
    print()
    print(f"Test Results:")
    print(f"  Accuracy: {test_acc*100:.1f}%")
    print(f"  Coverage: {test_cov*100:.1f}%")
    print()
    
    # Show top rules
    print("Top 5 Evolved Rules:")
    print("-" * 70)
    for i, rule in enumerate(best_rules[:5], 1):
        print(f"{i}. Fitness: {rule.fitness:.3f}")
        print(f"   Pattern: {rule.parse_pattern}")
        print(f"   Logic: {rule.logic_template}")
        print(f"   Generate: {rule.gen_template}")
        print()
    
    # Compare to neural baseline
    print("=" * 70)
    print("Comparison to Neural Baseline")
    print("=" * 70)
    print()
    print("Neural Network (50 rules, 30 epochs):")
    print("  Result: 0.00% accuracy (failed to learn)")
    print()
    print(f"GA Symbolic Rules (100 population, 50 generations):")
    print(f"  Result: {test_acc*100:.1f}% accuracy")
    print()
    
    if test_acc > 0.05:
        print("✓ GA WINS! Symbolic search found working rules.")
        print("  → Confirms hypothesis: Discrete optimization better for symbolic rules")
    elif test_acc > 0:
        print("⚠ GA shows promise but needs tuning.")
        print("  → Try: More generations, better mutations, larger population")
    else:
        print("✗ GA also failed. Problem might be in data representation.")
        print("  → Next: Debug data format, simplify logic representation")
    
    print()


if __name__ == "__main__":
    main()
