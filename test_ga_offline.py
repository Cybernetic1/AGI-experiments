"""
Test GA with pre-cached data - no internet required.
Optimized for CPU (GA doesn't benefit from GPU).
"""

import sys
import os
import random

# Force CPU mode (GA doesn't need GPU)
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from genetic_logic_rules import SymbolicRule, genetic_algorithm, semantic_ar_fitness


def create_simple_test_data():
    """
    Create simple synthetic data for offline testing.
    No internet required.
    """
    examples = [
        ("The cat sat", "entity=cat,relation=sat,value=None"),
        ("A dog ran", "entity=dog,relation=ran,value=None"),
        ("The bird flew", "entity=bird,relation=flew,value=None"),
        ("The mouse jumped", "entity=mouse,relation=jumped,value=None"),
        ("A cat walked", "entity=cat,relation=walked,value=None"),
        ("The dog barked", "entity=dog,relation=barked,value=None"),
        ("A bird sang", "entity=bird,relation=sang,value=None"),
        ("The fish swam", "entity=fish,relation=swam,value=None"),
        ("A horse galloped", "entity=horse,relation=galloped,value=None"),
        ("The lion roared", "entity=lion,relation=roared,value=None"),
    ]
    
    return examples


def main():
    print("="*70)
    print("GA Offline Test (CPU-optimized)")
    print("="*70)
    print()
    
    # Create simple dataset
    print("Creating test dataset...")
    dataset = create_simple_test_data()
    print(f"✓ Created {len(dataset)} examples")
    print()
    
    # Show samples
    print("Sample data:")
    for text, logic in dataset[:3]:
        print(f"  Text:  {text}")
        print(f"  Logic: {logic}")
        print()
    
    # Run GA
    print("Running genetic algorithm...")
    print("Population: 20, Generations: 20")
    print()
    
    best_rules = genetic_algorithm(
        dataset=dataset,
        population_size=20,
        generations=20,
        mutation_rate=0.3
    )
    
    print()
    print("="*70)
    print("Results")
    print("="*70)
    print()
    
    print(f"Found {len(best_rules)} rules:")
    for i, rule in enumerate(best_rules[:5], 1):
        fitness = semantic_ar_fitness(rule, dataset)
        print(f"\nRule {i}:")
        print(f"  Parse pattern:  {rule.parse_pattern}")
        print(f"  Logic template: {rule.logic_template}")
        print(f"  Gen template:   {rule.gen_template}")
        print(f"  Fitness:        {fitness:.1%}")
    
    # Test on each example
    print()
    print("Testing best rule on examples:")
    best_rule = best_rules[0]
    
    for text, expected_logic in dataset:
        try:
            parsed_logic = best_rule.parse(text)
            match = "✓" if parsed_logic == expected_logic else "✗"
            print(f"{match} '{text}' → {parsed_logic}")
        except:
            print(f"✗ '{text}' → (parse failed)")


if __name__ == '__main__':
    main()
