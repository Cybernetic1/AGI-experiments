"""
Hybrid GA + ILP + DLN Rule Discovery
=====================================

Integrates three approaches:
1. ILP (Inductive Logic Programming) - Fast structured discovery
2. GA (Genetic Algorithm) - Global optimization
3. DLN (Differentiable Logic Network) - Neural evaluation

Architecture:
- ILP mines initial rules (frequency-based, 10 sec)
- GA evolves rules using DLN loss as fitness (5-10 min)
- Best rules used for semantic-AR training
"""

import random
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from copy import deepcopy

from logic_core import Rule, Proposition, SymbolicEngine
from core.ilp_algorithms import mine_frequency_based, mine_foil_style, mine_confidence_based
from dln import SimpleDLN
from label_utils import _collect_labels
from core.train_utils import _train_on_labels, _eval_on_labels


@dataclass
class RuleWithFitness:
    """Wrapper for Rule with fitness tracking."""
    rule: Rule
    fitness: float = 0.0
    train_mse: float = float('inf')
    eval_mse: float = float('inf')
    label_count: int = 0
    generation: int = 0
    
    def __str__(self):
        return f"Rule(fitness={self.fitness:.4f}, train_mse={self.train_mse:.4f}, eval_mse={self.eval_mse:.4f}, labels={self.label_count})"


class HybridRuleDiscovery:
    """
    Hybrid ILP + GA + DLN rule discovery system.
    
    Workflow:
    1. ILP mines seed rules (fast)
    2. GA evolves rules using DLN fitness (iterative)
    3. Returns best rules for final training
    """
    
    def __init__(
        self,
        ilp_algorithm: str = 'frequency',
        ilp_rules: int = 30,
        ga_population: int = 50,
        ga_generations: int = 20,
        elite_size: int = 5,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.7,
        sample_facts_for_fitness: int = 1000,
        verbose: bool = True
    ):
        self.ilp_algorithm = ilp_algorithm
        self.ilp_rules = ilp_rules
        self.ga_population = ga_population
        self.ga_generations = ga_generations
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.sample_facts = sample_facts_for_fitness
        self.verbose = verbose
        
        # Track evolution
        self.history = []
        self.best_rule = None
        self.best_fitness = float('-inf')
    
    def discover(self, facts: List[Proposition]) -> List[Rule]:
        """
        Main discovery pipeline.
        
        Args:
            facts: Training facts
        
        Returns:
            List of best evolved rules
        """
        if self.verbose:
            print("\n" + "="*70)
            print("HYBRID GA + ILP + DLN RULE DISCOVERY")
            print("="*70)
        
        # Phase 1: ILP Seeding
        if self.verbose:
            print(f"\n[Phase 1] ILP Mining ({self.ilp_algorithm})...")
        
        seed_rules = self._ilp_seed(facts)
        
        if self.verbose:
            print(f"  ✅ Generated {len(seed_rules)} seed rules")
        
        # Phase 2: GA Evolution
        if self.verbose:
            print(f"\n[Phase 2] GA Evolution ({self.ga_generations} generations)...")
        
        evolved_rules = self._ga_evolve(seed_rules, facts)
        
        if self.verbose:
            print(f"  ✅ Evolution complete")
            print(f"  Best fitness: {self.best_fitness:.4f}")
            print(f"  Best rule: {self.best_rule}")
        
        # Return top K rules
        top_rules = sorted(evolved_rules, key=lambda r: r.fitness, reverse=True)[:self.ilp_rules]
        
        if self.verbose:
            print(f"\n[Phase 3] Returning top {len(top_rules)} rules")
            for i, r in enumerate(top_rules[:5], 1):
                print(f"  {i}. {r}")
        
        return [r.rule for r in top_rules]
    
    def _ilp_seed(self, facts: List[Proposition]) -> List[RuleWithFitness]:
        """Phase 1: Use ILP to generate seed rules."""
        # Mine rules with selected algorithm
        if self.ilp_algorithm == 'frequency':
            rules, _ = mine_frequency_based(facts, self.ilp_rules, min_support=2)
        elif self.ilp_algorithm == 'foil':
            rules, _ = mine_foil_style(facts, self.ilp_rules, min_support=2)
        elif self.ilp_algorithm == 'confidence':
            rules, _ = mine_confidence_based(facts, self.ilp_rules, min_support=2, min_confidence=0.3)
        else:
            raise ValueError(f"Unknown ILP algorithm: {self.ilp_algorithm}")
        
        # Wrap in fitness tracking
        wrapped = [RuleWithFitness(rule=r, generation=0) for r in rules]
        
        # Add some random variations for diversity
        mutations = []
        for _ in range(min(10, self.ga_population - len(rules))):
            if rules:
                parent = random.choice(wrapped)
                mutated = self._mutate(parent)
                mutated.generation = 0
                mutations.append(mutated)
        
        return wrapped + mutations
    
    def _ga_evolve(self, population: List[RuleWithFitness], facts: List[Proposition]) -> List[RuleWithFitness]:
        """Phase 2: Evolve rules using GA with DLN fitness."""
        # Pad population to desired size
        while len(population) < self.ga_population:
            population.append(deepcopy(random.choice(population[:len(population)//2])))
        
        # Sample facts for faster fitness evaluation
        sampled_facts = facts[:self.sample_facts] if len(facts) > self.sample_facts else facts
        
        # Collect all entities and predicates for DLN
        all_predicates = set()
        all_args = set()
        for fact in sampled_facts:
            all_predicates.add(fact.predicate)
            all_args.update(fact.args)
        
        for generation in range(self.ga_generations):
            if self.verbose:
                print(f"\n  Generation {generation + 1}/{self.ga_generations}")
            
            # Evaluate fitness for all rules
            self._evaluate_population(population, sampled_facts, all_predicates, all_args)
            
            # Track best
            gen_best = max(population, key=lambda r: r.fitness)
            if gen_best.fitness > self.best_fitness:
                self.best_fitness = gen_best.fitness
                self.best_rule = deepcopy(gen_best)
            
            # Statistics
            avg_fitness = sum(r.fitness for r in population) / len(population)
            if self.verbose:
                print(f"    Best: {gen_best.fitness:.4f}, Avg: {avg_fitness:.4f}, Labels: {gen_best.label_count}")
            
            self.history.append({
                'generation': generation,
                'best_fitness': gen_best.fitness,
                'avg_fitness': avg_fitness,
                'best_mse': gen_best.eval_mse
            })
            
            # Create next generation
            population = self._next_generation(population)
        
        # Final evaluation
        self._evaluate_population(population, sampled_facts, all_predicates, all_args)
        
        return population
    
    def _evaluate_population(self, population: List[RuleWithFitness], facts: List[Proposition],
                            all_predicates: set, all_args: set):
        """Evaluate fitness of all rules in population using DLN."""
        for rule_wrapper in population:
            if rule_wrapper.fitness > 0:
                continue  # Already evaluated
            
            try:
                # Generate labels by applying rule
                labels_dict = _collect_labels(facts, [rule_wrapper.rule], log_progress=False)
                rule_wrapper.label_count = len(labels_dict)
                
                if len(labels_dict) == 0:
                    rule_wrapper.fitness = -1.0  # No labels = bad rule
                    continue
                
                # Add rule's conclusion predicate
                rule_preds = list(all_predicates) + [rule_wrapper.rule.conclusion.predicate]
                
                # Create DLN
                dln = SimpleDLN(
                    predicates=list(set(rule_preds)),
                    args=list(all_args),
                    embed_dim=32
                )
                
                # Split labels
                labels_list = [(k, v) for k, v in labels_dict.items()]
                split_idx = int(0.8 * len(labels_list))
                train_labels = {k: v for k, v in labels_list[:split_idx]}
                eval_labels = {k: v for k, v in labels_list[split_idx:]}
                
                if len(eval_labels) == 0:
                    eval_labels = train_labels  # Fallback
                
                # Train DLN briefly
                optimizer = torch.optim.Adam(dln.parameters(), lr=0.001)
                train_mse = _train_on_labels(dln, optimizer, facts, train_labels, steps=10, batch_size=None)
                eval_mse = _eval_on_labels(dln, facts, eval_labels)
                
                rule_wrapper.train_mse = train_mse
                rule_wrapper.eval_mse = eval_mse
                
                # Fitness = negative eval MSE + label count bonus
                label_bonus = min(len(labels_dict) / 10000.0, 0.2)  # Up to 0.2 bonus
                rule_wrapper.fitness = -eval_mse + label_bonus
                
            except Exception as e:
                # Rule failed - assign very low fitness
                rule_wrapper.fitness = -10.0
                if self.verbose and generation == 0:
                    print(f"    ⚠️  Rule failed: {e}")
    
    def _next_generation(self, population: List[RuleWithFitness]) -> List[RuleWithFitness]:
        """Create next generation using selection, crossover, mutation."""
        # Sort by fitness
        population.sort(key=lambda r: r.fitness, reverse=True)
        
        # Elitism: keep best rules
        next_gen = [deepcopy(r) for r in population[:self.elite_size]]
        
        # Fill rest with offspring
        while len(next_gen) < self.ga_population:
            # Tournament selection
            parent1 = self._tournament_select(population)
            parent2 = self._tournament_select(population)
            
            # Crossover
            if random.random() < self.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = deepcopy(parent1)
            
            # Mutation
            if random.random() < self.mutation_rate:
                child = self._mutate(child)
            
            child.fitness = 0.0  # Reset for re-evaluation
            next_gen.append(child)
        
        return next_gen
    
    def _tournament_select(self, population: List[RuleWithFitness], k: int = 3) -> RuleWithFitness:
        """Select best from k random candidates."""
        candidates = random.sample(population, min(k, len(population)))
        return max(candidates, key=lambda r: r.fitness)
    
    def _crossover(self, parent1: RuleWithFitness, parent2: RuleWithFitness) -> RuleWithFitness:
        """Create child by crossing two parent rules."""
        # Simple crossover: mix premises and conclusions
        child_rule = Rule(
            premises=random.choice([parent1.rule.premises, parent2.rule.premises]),
            conclusion=random.choice([parent1.rule.conclusion, parent2.rule.conclusion]),
            weight=random.choice([parent1.rule.weight, parent2.rule.weight])
        )
        
        return RuleWithFitness(
            rule=child_rule,
            generation=max(parent1.generation, parent2.generation) + 1
        )
    
    def _mutate(self, parent: RuleWithFitness) -> RuleWithFitness:
        """Mutate a rule to create variation."""
        mutated = deepcopy(parent.rule)
        
        # Mutation strategies
        mutation_type = random.choice(['weight', 'swap_premises', 'modify_conclusion'])
        
        if mutation_type == 'weight':
            # Adjust rule weight
            mutated.weight = max(0.1, min(1.0, mutated.weight + random.gauss(0, 0.1)))
        
        elif mutation_type == 'swap_premises' and len(mutated.premises) >= 2:
            # Swap premise order
            i, j = random.sample(range(len(mutated.premises)), 2)
            mutated.premises[i], mutated.premises[j] = mutated.premises[j], mutated.premises[i]
        
        elif mutation_type == 'modify_conclusion':
            # Slightly modify conclusion (keep structure, change suffix)
            pred = mutated.conclusion.predicate
            if '_' in pred:
                parts = pred.split('_')
                parts[-1] = random.choice(['evolved', 'refined', 'derived', 'inferred'])
                mutated.conclusion = Proposition('_'.join(parts), mutated.conclusion.args, mutated.conclusion.truth)
        
        return RuleWithFitness(
            rule=mutated,
            generation=parent.generation + 1
        )
    
    def plot_evolution(self, save_path: str = None):
        """Plot fitness evolution over generations."""
        try:
            import matplotlib.pyplot as plt
            
            gens = [h['generation'] for h in self.history]
            best = [h['best_fitness'] for h in self.history]
            avg = [h['avg_fitness'] for h in self.history]
            
            plt.figure(figsize=(10, 6))
            plt.plot(gens, best, 'b-', label='Best Fitness', linewidth=2)
            plt.plot(gens, avg, 'r--', label='Avg Fitness', linewidth=2)
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.title('GA Evolution Progress')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path)
                print(f"Plot saved to {save_path}")
            else:
                plt.show()
        except ImportError:
            print("matplotlib not available for plotting")


def hybrid_discover_rules(
    facts: List[Proposition],
    ilp_algorithm: str = 'frequency',
    ilp_rules: int = 30,
    ga_generations: int = 20,
    verbose: bool = True
) -> List[Rule]:
    """
    Convenience function for hybrid rule discovery.
    
    Args:
        facts: Training facts
        ilp_algorithm: 'frequency', 'foil', or 'confidence'
        ilp_rules: Number of initial ILP rules
        ga_generations: Number of GA generations
        verbose: Print progress
    
    Returns:
        List of best evolved rules
    """
    discoverer = HybridRuleDiscovery(
        ilp_algorithm=ilp_algorithm,
        ilp_rules=ilp_rules,
        ga_generations=ga_generations,
        verbose=verbose
    )
    
    return discoverer.discover(facts)


if __name__ == '__main__':
    # Quick test
    from pipelines.tinystories_pipeline import load_tinystories_facts
    
    print("Loading test data...")
    facts = load_tinystories_facts(max_stories=50, max_facts=5000)
    print(f"Loaded {len(facts)} facts")
    
    print("\nRunning hybrid discovery...")
    rules = hybrid_discover_rules(
        facts,
        ilp_algorithm='frequency',
        ilp_rules=10,
        ga_generations=5,
        verbose=True
    )
    
    print(f"\n✅ Discovered {len(rules)} rules")
    for i, rule in enumerate(rules[:5], 1):
        print(f"{i}. {rule}")
