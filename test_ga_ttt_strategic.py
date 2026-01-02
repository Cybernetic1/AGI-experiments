"""
Strategic TicTacToe GA - focus on key strategic positions.
"""

import sys
sys.path.insert(0, '.')
from test_ga_tictactoe import *

def generate_strategic_positions():
    """
    Generate training positions focusing on key strategic situations.
    """
    positions = []
    
    # Critical: Winning moves (must recognize)
    win_positions = [
        # Horizontal wins
        ([1, 1, 0, 0, 0, 0, 0, 0, 0], 2),  # XX_
        ([0, 0, 0, 1, 1, 0, 0, 0, 0], 5),  # XX_ row 2
        ([0, 0, 0, 0, 0, 0, 1, 1, 0], 8),  # XX_ row 3
        # Vertical wins
        ([1, 0, 0, 1, 0, 0, 0, 0, 0], 6),  # X over X col 1
        ([0, 1, 0, 0, 1, 0, 0, 0, 0], 7),  # col 2
        # Diagonal wins
        ([1, 0, 0, 0, 1, 0, 0, 0, 0], 8),  # Diagonal \
        ([0, 0, 1, 0, 1, 0, 0, 0, 0], 6),  # Diagonal /
    ]
    
    # Critical: Blocking opponent wins
    block_positions = [
        ([(-1), (-1), 0, 0, 0, 0, 0, 0, 0], 2),  # Block OO_
        ([0, 0, 0, (-1), (-1), 0, 0, 0, 0], 5),  # Block OO_ row 2
        ([(-1), 0, 0, (-1), 0, 0, 0, 0, 0], 6),  # Block vertical
        ([(-1), 0, 0, 0, (-1), 0, 0, 0, 0], 8),  # Block diagonal
    ]
    
    # Strategic: Center and corners
    strategic = [
        ([0]*9, 4),  # Empty -> center
        ([0, 0, 0, 0, 1, 0, 0, 0, 0], 0),  # Center taken -> corner
        ([0, 0, 0, 0, 1, 0, 0, 0, 0], 2),  # Center taken -> corner
        ([0, 0, 0, 0, 1, 0, 0, 0, 0], 6),  # Center taken -> corner
        ([0, 0, 0, 0, 1, 0, 0, 0, 0], 8),  # Center taken -> corner
    ]
    
    # Add all with high weight (duplicate for emphasis)
    for _ in range(5):  # Repeat critical patterns
        positions.extend(win_positions)
        positions.extend(block_positions)
    for _ in range(3):
        positions.extend(strategic)
    
    # Add some random positions for generalization
    for _ in range(30):
        game = TicTacToeGame()
        for _ in range(random.randint(1, 5)):
            if not game.is_terminal():
                game.make_move(game.random_move())
        if not game.is_terminal():
            good_move = random.choice(game.available_moves())
            positions.append((game.board[:], good_move))
    
    return positions


def enhanced_fitness(rule, positions):
    """Enhanced fitness that weights critical positions higher."""
    matches = 0
    score = 0.0
    
    for i, (board, best_action) in enumerate(positions):
        if rule.matches(board):
            matches += 1
            
            # Weight: first 70 positions are critical (win/block), weight 3x
            weight = 3.0 if i < 70 else 1.0
            
            if rule.is_valid_action(board):
                # Exact match gets full points
                if rule.action == best_action:
                    score += weight * 1.0
                # Close action gets partial
                elif abs(rule.action - best_action) <= 2:
                    score += weight * 0.5
                else:
                    score += weight * 0.2
    
    if matches == 0:
        return 0.0
    
    coverage = min(matches / len(positions), 0.3)
    quality = (score / (matches * 3.0)) * 0.7  # Normalized by max possible
    
    return coverage + quality


print("="*70)
print("Strategic TicTacToe GA Training")
print("="*70)

positions = generate_strategic_positions()
print(f"Training positions: {len(positions)} (emphasizing critical patterns)")

# Train with enhanced fitness
population_size = 300
generations = 100

print(f"Population: {population_size}")
print(f"Generations: {generations}")
print()

# Initialize population
population = [TicTacToeRule.random() for _ in range(population_size)]

best_overall = None
best_fitness = 0

for gen in range(generations):
    # Use enhanced fitness
    for rule in population:
        rule.fitness = enhanced_fitness(rule, positions)
    
    population.sort(key=lambda r: r.fitness, reverse=True)
    gen_best = population[0]
    
    if gen_best.fitness > best_fitness:
        best_fitness = gen_best.fitness
        best_overall = gen_best
    
    if gen % 10 == 0 or gen < 5:
        avg_fitness = sum(r.fitness for r in population) / len(population)
        print(f"Gen {gen+1:3d} | Best: {gen_best.fitness:.3f} | Avg: {avg_fitness:.3f}")
    
    if gen_best.fitness > 0.95:
        print(f"\n✓ High fitness achieved at generation {gen+1}!")
        break
    
    # Reproduce
    new_population = []
    elite_size = 30
    new_population.extend(population[:elite_size])
    
    while len(new_population) < population_size:
        parent1 = tournament_selection(population)
        parent2 = tournament_selection(population)
        child = parent1.crossover(parent2)
        if random.random() < 0.3:
            child.mutate()
        new_population.append(child)
    
    population = new_population

# Evaluate
print("\n" + "="*70)
print("Evaluation")
print("="*70)

top_rules = population[:30]

for num_rules in [5, 10, 20, 30]:
    win_rate = evaluate_rules(top_rules[:num_rules], num_games=500)
    print(f"Top {num_rules:2d} rules: {win_rate*100:.1f}% win rate (500 games)")

print(f"\nTop 5 rules:")
for i, rule in enumerate(top_rules[:5], 1):
    print(f"  {i}. {rule}")
