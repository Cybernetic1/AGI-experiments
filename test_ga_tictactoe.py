"""
Test GA on TicTacToe to verify algorithm works before scaling to NL.

Hypothesis: NL requires R_K (Kolmogorov complexity) rules, but if test R << R_K,
we see non-convergence. TTT should converge with small R since it's simpler.

This verifies:
1. GA algorithm implementation is correct
2. Fitness function provides proper gradient
3. Convergence happens when task complexity matches capacity
"""

import random
import numpy as np
from typing import List, Tuple, Optional, Dict


class TicTacToeRule:
    """
    Simple rule for TicTacToe: board_pattern -> action
    
    Board representation: 9 positions, each is -1 (O), 0 (empty), 1 (X)
    Action: position 0-8 to place mark
    """
    
    def __init__(self):
        # Pattern matching: which positions to check (mask + expected values)
        self.position_mask = [False] * 9  # Which positions matter
        self.expected_values = [0] * 9     # What values expected (-1, 0, 1)
        
        # Action: which empty position to choose
        self.action = 0  # Position 0-8
        
        # Fitness tracking
        self.fitness = 0.0
        self.match_count = 0
        self.win_count = 0
    
    @staticmethod
    def random():
        """Create random rule."""
        rule = TicTacToeRule()
        
        # Random mask: check 1-4 positions
        num_check = random.randint(1, 4)
        positions = random.sample(range(9), num_check)
        for pos in positions:
            rule.position_mask[pos] = True
            rule.expected_values[pos] = random.choice([-1, 0, 1])
        
        # Random action
        rule.action = random.randint(0, 8)
        
        return rule
    
    def matches(self, board: List[int]) -> bool:
        """Check if rule pattern matches board state."""
        for i in range(9):
            if self.position_mask[i]:
                if board[i] != self.expected_values[i]:
                    return False
        return True
    
    def is_valid_action(self, board: List[int]) -> bool:
        """Check if action is valid (position is empty)."""
        return board[self.action] == 0
    
    def mutate(self):
        """Mutate rule."""
        mutation_type = random.choice(['mask', 'value', 'action'])
        
        if mutation_type == 'mask':
            # Flip random mask bit
            pos = random.randint(0, 8)
            self.position_mask[pos] = not self.position_mask[pos]
        
        elif mutation_type == 'value':
            # Change expected value for a masked position
            masked_positions = [i for i in range(9) if self.position_mask[i]]
            if masked_positions:
                pos = random.choice(masked_positions)
                self.expected_values[pos] = random.choice([-1, 0, 1])
        
        elif mutation_type == 'action':
            # Change action
            self.action = random.randint(0, 8)
    
    def crossover(self, other: 'TicTacToeRule') -> 'TicTacToeRule':
        """Create child by crossing with another rule."""
        child = TicTacToeRule()
        
        # Inherit mask pattern from one parent
        child.position_mask = random.choice([self.position_mask[:], other.position_mask[:]])
        child.expected_values = random.choice([self.expected_values[:], other.expected_values[:]])
        
        # Inherit action from possibly different parent
        child.action = random.choice([self.action, other.action])
        
        return child
    
    def __str__(self):
        pattern = []
        for i in range(9):
            if self.position_mask[i]:
                val = self.expected_values[i]
                symbol = 'O' if val == -1 else ('X' if val == 1 else '.')
                pattern.append(f"{i}={symbol}")
        pattern_str = ", ".join(pattern) if pattern else "any"
        return f"Rule(if [{pattern_str}] then action={self.action}, fitness={self.fitness:.3f})"


class TicTacToeGame:
    """Simple TicTacToe game for evaluation."""
    
    def __init__(self):
        self.board = [0] * 9  # Empty board
        self.current_player = 1  # X starts
    
    def reset(self):
        self.board = [0] * 9
        self.current_player = 1
    
    def available_moves(self) -> List[int]:
        return [i for i in range(9) if self.board[i] == 0]
    
    def make_move(self, pos: int) -> bool:
        """Make move, return success."""
        if pos < 0 or pos >= 9 or self.board[pos] != 0:
            return False
        self.board[pos] = self.current_player
        self.current_player = -self.current_player
        return True
    
    def check_winner(self) -> Optional[int]:
        """Return winner (1, -1) or None."""
        # Check rows, cols, diagonals
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Cols
            [0, 4, 8], [2, 4, 6]              # Diagonals
        ]
        
        for line in lines:
            values = [self.board[i] for i in line]
            if values[0] != 0 and values[0] == values[1] == values[2]:
                return values[0]
        
        return None
    
    def is_terminal(self) -> bool:
        """Check if game is over."""
        return self.check_winner() is not None or len(self.available_moves()) == 0
    
    def random_move(self) -> int:
        """Random valid move."""
        return random.choice(self.available_moves())


def evaluate_rules(rules: List[TicTacToeRule], num_games: int = 100) -> float:
    """
    Evaluate ruleset by playing games.
    
    Player X uses rules, Player O plays randomly.
    Returns win rate for X.
    """
    wins = 0
    losses = 0
    draws = 0
    
    for _ in range(num_games):
        game = TicTacToeGame()
        
        while not game.is_terminal():
            if game.current_player == 1:  # X uses rules
                # Find matching rule
                action = None
                for rule in rules:
                    if rule.matches(game.board) and rule.is_valid_action(game.board):
                        action = rule.action
                        rule.match_count += 1
                        break
                
                # If no rule matches, play randomly
                if action is None:
                    action = game.random_move()
                
                game.make_move(action)
            else:  # O plays randomly
                action = game.random_move()
                game.make_move(action)
        
        winner = game.check_winner()
        if winner == 1:
            wins += 1
            # Reward rules that matched
            for rule in rules:
                if rule.match_count > 0:
                    rule.win_count += 1
        elif winner == -1:
            losses += 1
        else:
            draws += 1
    
    return wins / num_games


def fitness_function(rule: TicTacToeRule, test_games: List[Tuple[List[int], int]]) -> float:
    """
    Fitness based on:
    1. Does rule match positions?
    2. Is action valid?
    3. Does action lead to winning positions?
    """
    matches = 0
    valid_actions = 0
    good_actions = 0
    
    for board, best_action in test_games:
        if rule.matches(board):
            matches += 1
            
            if rule.is_valid_action(board):
                valid_actions += 1
                
                # Check if action is reasonable (within 2 of best)
                if abs(rule.action - best_action) <= 2:
                    good_actions += 1
    
    if matches == 0:
        return 0.0
    
    # Fitness components
    coverage = min(matches / len(test_games), 0.3)  # Max 30%
    validity = (valid_actions / matches) * 0.4      # Max 40%
    quality = (good_actions / matches) * 0.3        # Max 30%
    
    return coverage + validity + quality


def generate_training_positions() -> List[Tuple[List[int], int]]:
    """
    Generate training positions with reasonable moves.
    
    Returns: List of (board_state, good_action) pairs
    """
    positions = []
    
    # Opening moves (center or corners are good)
    positions.append(([0]*9, 4))  # Empty board -> center
    
    # Block opponent winning
    positions.append(([1, 1, 0, 0, 0, 0, 0, 0, 0], 2))  # Block XX_
    positions.append(([0, 0, 0, 1, 1, 0, 0, 0, 0], 5))  # Block XX_
    
    # Complete own winning line
    positions.append(([1, 0, 1, 0, 0, 0, 0, 0, 0], 1))  # Win X_X
    positions.append(([1, 0, 0, 0, 1, 0, 0, 0, 0], 8))  # Win diagonal
    
    # Generate more random positions
    for _ in range(50):
        game = TicTacToeGame()
        # Play 2-5 random moves
        for _ in range(random.randint(2, 5)):
            if not game.is_terminal():
                game.make_move(game.random_move())
        
        if not game.is_terminal():
            # Pick a reasonable move (random from available)
            good_move = random.choice(game.available_moves())
            positions.append((game.board[:], good_move))
    
    return positions


def tournament_selection(population: List[TicTacToeRule], k: int = 3) -> TicTacToeRule:
    """Select best rule from k random candidates."""
    candidates = random.sample(population, k)
    return max(candidates, key=lambda r: r.fitness)


def genetic_algorithm(training_positions: List[Tuple[List[int], int]],
                     population_size: int = 100,
                     generations: int = 50,
                     mutation_rate: float = 0.3,
                     elite_size: int = 10) -> List[TicTacToeRule]:
    """
    Evolve TicTacToe rules using GA.
    """
    print("=" * 70)
    print("Genetic Algorithm for TicTacToe Rules")
    print("=" * 70)
    print(f"Training positions: {len(training_positions)}")
    print(f"Population: {population_size}")
    print(f"Generations: {generations}")
    print(f"Mutation rate: {mutation_rate}")
    print()
    
    # Initialize population
    print("Initializing random population...")
    population = [TicTacToeRule.random() for _ in range(population_size)]
    
    best_overall = None
    best_fitness = 0
    
    for gen in range(generations):
        # Evaluate fitness
        for rule in population:
            rule.fitness = fitness_function(rule, training_positions)
        
        # Track best
        population.sort(key=lambda r: r.fitness, reverse=True)
        gen_best = population[0]
        
        if gen_best.fitness > best_fitness:
            best_fitness = gen_best.fitness
            best_overall = gen_best
        
        # Print progress
        avg_fitness = sum(r.fitness for r in population) / len(population)
        print(f"Gen {gen+1:3d} | Best: {gen_best.fitness:.3f} | "
              f"Avg: {avg_fitness:.3f} | "
              f"Action: {gen_best.action}")
        
        # Early stopping
        if gen_best.fitness > 0.9:
            print(f"\n✓ High fitness achieved! Stopping early.")
            break
        
        # Selection and reproduction
        new_population = []
        
        # Elitism
        new_population.extend(population[:elite_size])
        
        # Generate offspring
        while len(new_population) < population_size:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            
            child = parent1.crossover(parent2)
            
            if random.random() < mutation_rate:
                child.mutate()
            
            new_population.append(child)
        
        population = new_population
    
    print("\n" + "=" * 70)
    print("Evolution Complete!")
    print("=" * 70)
    
    # Evaluate in actual games
    population.sort(key=lambda r: r.fitness, reverse=True)
    top_rules = population[:20]
    
    print(f"\nEvaluating top 20 rules in actual games...")
    win_rate = evaluate_rules(top_rules, num_games=100)
    print(f"Win rate vs random opponent: {win_rate*100:.1f}%")
    print()
    
    return top_rules


def main():
    print("=" * 70)
    print("TicTacToe GA Test: Verify Algorithm Before Scaling to NL")
    print("=" * 70)
    print()
    print("Hypothesis: GA should converge on TTT (low Kolmogorov complexity)")
    print("           but fail on NL if #rules << R_K (Kolmogorov bound)")
    print()
    
    # Generate training data
    print("Generating training positions...")
    training_positions = generate_training_positions()
    print(f"✓ Generated {len(training_positions)} positions")
    print()
    
    # Run GA
    best_rules = genetic_algorithm(
        training_positions,
        population_size=100,
        generations=50,
        mutation_rate=0.3,
        elite_size=10
    )
    
    # Show top rules
    print("Top 5 Evolved Rules:")
    print("-" * 70)
    for i, rule in enumerate(best_rules[:5], 1):
        print(f"{i}. {rule}")
    print()
    
    # Analysis
    print("=" * 70)
    print("Analysis & Conclusions")
    print("=" * 70)
    print()
    
    # Get final win rate
    win_rate = evaluate_rules(best_rules[:10], num_games=200)
    
    print(f"Final Performance:")
    print(f"  Win rate: {win_rate*100:.1f}% (vs random opponent)")
    print(f"  Top rule fitness: {best_rules[0].fitness:.3f}")
    print()
    
    if win_rate > 0.5:
        print("✓ SUCCESS: GA learned effective TicTacToe rules!")
        print("  → Algorithm implementation is correct")
        print("  → Fitness function provides learning signal")
        print("  → Confirms: Simple tasks (low R_K) can be learned")
        print()
        print("→ IMPLICATION: NL failure is due to R << R_K, not GA algorithm")
        print("  → Need more rules OR simpler NL tasks to test")
    elif win_rate > 0.3:
        print("⚠ PARTIAL: GA shows learning but needs improvement")
        print("  → Try: More generations, better training positions")
        print("  → Or: Refine fitness function")
    else:
        print("✗ FAILURE: GA failed even on simple TicTacToe")
        print("  → Problem is in GA implementation itself")
        print("  → Need to debug: fitness function, mutations, selection")
    print()


if __name__ == "__main__":
    main()
