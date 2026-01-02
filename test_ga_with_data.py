"""
Test GA-based logic rule learning with real TinyStories data.
Uses cached preprocessed data from ./data/ directory.
Runs on CPU - no GPU needed for GA.
"""

import torch
import pickle
from pathlib import Path
from genetic_logic_rules import genetic_algorithm, SymbolicRule

def load_cached_data():
    """Load preprocessed TinyStories data from cache"""
    data_dir = Path("./data/tinystories_cache")
    
    # Find largest processed file
    pt_files = list(data_dir.glob("processed_*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No processed data found in {data_dir}")
    
    # Use the largest one
    pt_file = max(pt_files, key=lambda p: p.stat().st_size)
    print(f"  Loading: {pt_file.name}")
    
    data = torch.load(pt_file)
    
    return data, None  # No separate val data for now

def main():
    print("="*70)
    print("GA Logic Rule Learning - TinyStories Test")
    print("="*70)
    
    # Load data
    print("\n[1/5] Loading cached TinyStories data...")
    data, _ = load_cached_data()
    
    samples = data['samples']
    vocab = data['vocab']
    
    print(f"  ✓ Total samples: {len(samples)}")
    print(f"  ✓ Vocab size: {len(vocab)}")
    print(f"  ✓ Entity count: {data['next_entity_id']}")
    
    # Use small subset for initial test
    subset_size = min(100, len(samples))
    subset_samples = samples[:subset_size]
    
    # Extract texts
    print("\n  Extracting texts from samples...")
    texts = [s['text'] for s in subset_samples]
    
    print(f"\n  → Using subset of {subset_size} samples for quick test")
    
    # Convert to GA format: (text, logic_string) pairs
    print("\n[2/5] Preparing dataset for GA...")
    ga_dataset = []
    for i in range(len(texts)):
        text = texts[i]
        # Text is already a list of tokens, join them
        if isinstance(text, list):
            text_str = " ".join(text)
        else:
            text_str = text
        
        # Simplify: use first few tokens as "logic"
        first_tokens = text[:3] if isinstance(text, list) else text_str.split()[:3]
        logic_str = " ".join(first_tokens)
        ga_dataset.append((text_str, logic_str))
    
    print(f"  ✓ Prepared {len(ga_dataset)} examples")
    print(f"  Example: '{ga_dataset[0][0][:40]}...' → '{ga_dataset[0][1]}'")
    
    # Run GA
    print("\n[3/5] Running Genetic Algorithm...")
    print("  (This runs on CPU - no GPU acceleration)")
    
    best_rules = genetic_algorithm(
        ga_dataset,
        population_size=30,
        generations=20,
        mutation_rate=0.3,
        elite_size=3
    )
    
    # Evaluate
    print("\n[4/5] Evaluating best rules...")
    print(f"  Found {len(best_rules)} high-fitness rules")
    
    for i, rule in enumerate(best_rules[:3], 1):
        print(f"\n  Rule {i}:")
        print(f"    Pattern: {rule.parse_pattern}")
        print(f"    Logic: {rule.logic_template}")
        print(f"    Fitness: {rule.fitness:.3f}")
        print(f"    Matches: {rule.examples_matched} examples")
    
    # Test on examples
    print("\n[5/5] Testing on sample texts...")
    for i in range(min(5, len(texts))):
        text = texts[i]
        print(f"\n  Example {i+1}: '{text[:50]}...'")
        
        # Try each rule
        for j, rule in enumerate(best_rules[:2], 1):
            if rule.matches(text):
                logic = rule.parse(text)
                gen_text = rule.generate(logic) if logic else None
                print(f"    Rule {j} matches → {logic}")
                if gen_text:
                    print(f"      Generated: '{gen_text}'")
                    consistent = rule.is_semantically_consistent(text=text)
                    print(f"      Consistent: {consistent}")
                break
        else:
            print(f"    No rules matched")
    
    print("\n" + "="*70)
    print("GA Test Complete!")
    print("="*70)

if __name__ == "__main__":
    main()
