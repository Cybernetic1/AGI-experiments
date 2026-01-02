"""
Push GA to optimal TicTacToe play - find minimum population size needed.
"""

import sys
sys.path.insert(0, '.')
from test_ga_tictactoe import *

def run_experiment(population_size, generations, name):
    """Run GA with specific parameters."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {name}")
    print(f"{'='*70}")
    
    training_positions = generate_training_positions()
    
    best_rules = genetic_algorithm(
        training_positions,
        population_size=population_size,
        generations=generations,
        mutation_rate=0.3,
        elite_size=max(10, population_size // 10)
    )
    
    # Test with different rule counts
    print(f"\nTesting different rule subset sizes:")
    for num_rules in [5, 10, 20, 50]:
        if num_rules > len(best_rules):
            break
        win_rate = evaluate_rules(best_rules[:num_rules], num_games=300)
        print(f"  Top {num_rules:2d} rules: {win_rate*100:.1f}% win rate")
    
    return best_rules, win_rate

if __name__ == "__main__":
    results = []
    
    # Test 1: Current baseline
    rules1, win1 = run_experiment(100, 50, "Baseline (100 pop, 50 gen)")
    results.append(("Baseline", 100, 50, win1))
    
    # Test 2: Larger population
    rules2, win2 = run_experiment(300, 50, "Large Pop (300 pop, 50 gen)")
    results.append(("Large Pop", 300, 50, win2))
    
    # Test 3: More generations
    rules3, win3 = run_experiment(100, 200, "Long Train (100 pop, 200 gen)")
    results.append(("Long Train", 100, 200, win3))
    
    # Test 4: Aggressive (both)
    rules4, win4 = run_experiment(500, 100, "Aggressive (500 pop, 100 gen)")
    results.append(("Aggressive", 500, 100, win4))
    
    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY: Finding Optimal Parameters")
    print(f"{'='*70}\n")
    
    results.sort(key=lambda x: x[3], reverse=True)
    
    for name, pop, gen, win in results:
        total_evals = pop * gen
        print(f"{name:15s} | Pop: {pop:3d} | Gen: {gen:3d} | "
              f"Win: {win*100:5.1f}% | Evals: {total_evals:6d}")
    
    print(f"\nOptimal strategy win rate: ~85-90% (perfect play vs random)")
    print(f"Best achieved: {results[0][3]*100:.1f}%")
    
    if results[0][3] > 0.80:
        print("\n✓ NEAR-OPTIMAL: GA learned expert-level TicTacToe!")
    elif results[0][3] > 0.65:
        print("\n⚠ GOOD: GA learned strong play, could improve with more tuning")
    else:
        print("\n✗ SUBOPTIMAL: More training or better fitness function needed")
