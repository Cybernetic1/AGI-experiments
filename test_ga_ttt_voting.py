"""
TicTacToe GA with Weighted Voting rule selection.

Instead of first-match, all matching rules vote weighted by fitness.
"""

import sys
sys.path.insert(0, '.')
from test_ga_tictactoe import *
from collections import defaultdict


def evaluate_rules_with_voting(rules: List[TicTacToeRule], num_games: int = 100) -> float:
    """
    Evaluate ruleset using WEIGHTED VOTING for rule selection.
    
    All matching rules vote for their action, weighted by fitness.
    """
    wins = 0
    losses = 0
    draws = 0
    
    for _ in range(num_games):
        game = TicTacToeGame()
        
        while not game.is_terminal():
            if game.current_player == 1:  # X uses rules with voting
                # Collect votes from all matching rules
                votes = defaultdict(float)
                
                for rule in rules:
                    if rule.matches(game.board) and rule.is_valid_action(game.board):
                        # Vote weighted by fitness
                        votes[rule.action] += rule.fitness
                
                # Choose action with highest vote
                if votes:
                    action = max(votes.items(), key=lambda x: x[1])[0]
                else:
                    # No rules match - play randomly
                    action = game.random_move()
                
                game.make_move(action)
            else:  # O plays randomly
                action = game.random_move()
                game.make_move(action)
        
        winner = game.check_winner()
        if winner == 1:
            wins += 1
        elif winner == -1:
            losses += 1
        else:
            draws += 1
    
    return wins / num_games


def evaluate_rules_best_match(rules: List[TicTacToeRule], num_games: int = 100) -> float:
    """
    Evaluate using BEST-MATCH: pick highest fitness rule that matches.
    """
    wins = 0
    
    for _ in range(num_games):
        game = TicTacToeGame()
        
        while not game.is_terminal():
            if game.current_player == 1:
                # Find best matching rule
                best_rule = None
                best_fitness = -1
                
                for rule in rules:
                    if rule.matches(game.board) and rule.is_valid_action(game.board):
                        if rule.fitness > best_fitness:
                            best_fitness = rule.fitness
                            best_rule = rule
                
                if best_rule:
                    action = best_rule.action
                else:
                    action = game.random_move()
                
                game.make_move(action)
            else:
                action = game.random_move()
                game.make_move(action)
        
        if game.check_winner() == 1:
            wins += 1
    
    return wins / num_games


def evaluate_rules_specificity(rules: List[TicTacToeRule], num_games: int = 100) -> float:
    """
    Evaluate using SPECIFICITY: pick most specific (constrained) rule.
    """
    wins = 0
    
    for _ in range(num_games):
        game = TicTacToeGame()
        
        while not game.is_terminal():
            if game.current_player == 1:
                # Find most specific matching rule
                best_rule = None
                max_specificity = -1
                
                for rule in rules:
                    if rule.matches(game.board) and rule.is_valid_action(game.board):
                        specificity = sum(rule.position_mask)  # Count constraints
                        if specificity > max_specificity:
                            max_specificity = specificity
                            best_rule = rule
                
                if best_rule:
                    action = best_rule.action
                else:
                    action = game.random_move()
                
                game.make_move(action)
            else:
                action = game.random_move()
                game.make_move(action)
        
        if game.check_winner() == 1:
            wins += 1
    
    return wins / num_games


def main():
    print("="*70)
    print("Rule Selection Strategy Comparison")
    print("="*70)
    print()
    
    # Train rules
    print("Training ruleset with GA...")
    training_positions = generate_training_positions()
    
    best_rules = genetic_algorithm(
        training_positions,
        population_size=300,
        generations=50,
        mutation_rate=0.3,
        elite_size=30
    )
    
    # Test different selection strategies
    print("\n" + "="*70)
    print("Testing Rule Selection Strategies")
    print("="*70)
    print()
    
    num_games = 500
    
    print(f"Evaluating on {num_games} games each...\n")
    
    # Original first-match
    print("1. FIRST-MATCH (current):")
    win_rate_first = evaluate_rules(best_rules[:20], num_games)
    print(f"   Win rate: {win_rate_first*100:.1f}%")
    print()
    
    # Best-match
    print("2. BEST-MATCH (highest fitness):")
    win_rate_best = evaluate_rules_best_match(best_rules[:20], num_games)
    print(f"   Win rate: {win_rate_best*100:.1f}%")
    print()
    
    # Weighted voting
    print("3. WEIGHTED VOTING (fitness-weighted):")
    win_rate_voting = evaluate_rules_with_voting(best_rules[:20], num_games)
    print(f"   Win rate: {win_rate_voting*100:.1f}%")
    print()
    
    # Specificity
    print("4. SPECIFICITY (most constrained):")
    win_rate_spec = evaluate_rules_specificity(best_rules[:20], num_games)
    print(f"   Win rate: {win_rate_spec*100:.1f}%")
    print()
    
    # Summary
    print("="*70)
    print("Summary")
    print("="*70)
    
    strategies = [
        ("First-Match", win_rate_first),
        ("Best-Match", win_rate_best),
        ("Weighted Voting", win_rate_voting),
        ("Specificity", win_rate_spec)
    ]
    
    strategies.sort(key=lambda x: x[1], reverse=True)
    
    print()
    for rank, (name, win_rate) in enumerate(strategies, 1):
        print(f"{rank}. {name:20s}: {win_rate*100:5.1f}%")
    
    print()
    improvement = (strategies[0][1] - win_rate_first) / win_rate_first * 100
    print(f"Best strategy improves over first-match by: {improvement:+.1f}%")
    print()
    
    if strategies[0][0] == "Weighted Voting":
        print("✓ WEIGHTED VOTING wins - collective intelligence works!")
    elif strategies[0][0] == "Best-Match":
        print("✓ BEST-MATCH wins - quality over quantity!")
    elif strategies[0][0] == "Specificity":
        print("✓ SPECIFICITY wins - expert system approach!")
    else:
        print("⚠ First-Match was best - others need tuning")


if __name__ == "__main__":
    main()
