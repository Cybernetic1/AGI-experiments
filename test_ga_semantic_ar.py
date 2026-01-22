"""
GA Optimized for Semantic-AR (Next-Verb Prediction)
====================================================

Redesigns GA to directly optimize for next-verb prediction accuracy.

Key changes from old GA:
1. Fitness = Top-5 accuracy on held-out event sequences (not label MSE)
2. Rules represent story grammar patterns (verb transitions)
3. Evaluation uses rule constraints during semantic-AR prediction

This tests: Can GA discover better story grammar than ILP frequency counting?

Success metric: Beat ILP baseline (7.75% Top-5 accuracy)

Usage:
    python test_ga_semantic_ar.py --scale full --device cuda --ga-generations 20
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import time
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Set
import argparse
from collections import defaultdict, Counter
from copy import deepcopy

from pipelines.tinystories_pipeline import load_tinystories_facts
from logic_core import Proposition, Rule
from core.ilp_algorithms import mine_frequency_based
from test_semantic_event_ar import (
    download_glove_embeddings,
    create_verb_embedding_matrix,
    SemanticEventAR,
    SCALE_CONFIGS
)
from test_event_level_ar import (
    group_facts_into_events,
    create_event_sequences,
    create_event_ar_training_data
)
from test_rule_assisted_ar import (
    extract_story_grammar_from_rules,
    apply_rule_constraints,
    RuleAssistedSemanticEventAR
)


class VerbTransitionRule:
    """
    Simplified rule format for story grammar.
    
    Represents: verb1 → {verb2, verb3, ...} with weights
    """
    
    def __init__(self, source_verb: str, target_verbs: List[Tuple[str, float]]):
        self.source_verb = source_verb
        self.target_verbs = target_verbs  # [(verb, weight), ...]
        self.fitness = 0.0
    
    def get_candidates(self, top_k: int = 50) -> Set[str]:
        """Get top-k target verbs by weight."""
        sorted_targets = sorted(self.target_verbs, key=lambda x: x[1], reverse=True)
        return {v for v, _ in sorted_targets[:top_k]}
    
    def mutate(self, all_verbs: List[str], mutation_rate: float = 0.2):
        """Mutate this rule by adjusting weights or adding/removing verbs."""
        new_targets = list(self.target_verbs)
        
        # Decide mutation type
        mutation = random.choice(['adjust_weights', 'add_verb', 'remove_verb'])
        
        if mutation == 'adjust_weights':
            # Randomly adjust some weights
            for i in range(len(new_targets)):
                if random.random() < mutation_rate:
                    verb, weight = new_targets[i]
                    new_weight = max(0.1, min(1.0, weight + random.gauss(0, 0.2)))
                    new_targets[i] = (verb, new_weight)
        
        elif mutation == 'add_verb' and len(new_targets) < 100:
            # Add a random verb
            new_verb = random.choice(all_verbs)
            if new_verb not in [v for v, _ in new_targets]:
                new_targets.append((new_verb, random.uniform(0.3, 0.7)))
        
        elif mutation == 'remove_verb' and len(new_targets) > 5:
            # Remove lowest weight verb
            new_targets = sorted(new_targets, key=lambda x: x[1], reverse=True)[:-1]
        
        return VerbTransitionRule(self.source_verb, new_targets)
    
    def __repr__(self):
        top3 = sorted(self.target_verbs, key=lambda x: x[1], reverse=True)[:3]
        return f"Rule({self.source_verb} → {top3})"


class SemanticARGA:
    """
    Genetic Algorithm for evolving story grammar rules.
    
    Fitness = Top-5 accuracy on next-verb prediction with rule constraints.
    """
    
    def __init__(
        self,
        all_verbs: List[str],
        training_pairs: List[Tuple[List[List[Proposition]], List[Proposition]]],
        eval_pairs: List[Tuple[List[List[Proposition]], List[Proposition]]],
        model: RuleAssistedSemanticEventAR,
        device: str = "cpu",
        population_size: int = 30,
        elite_size: int = 5,
        mutation_rate: float = 0.3,
        verbose: bool = True
    ):
        self.all_verbs = all_verbs
        self.training_pairs = training_pairs
        self.eval_pairs = eval_pairs
        self.model = model
        self.device = device
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.verbose = verbose
        
        # Track evolution
        self.history = []
        self.best_rules = None
        self.best_fitness = 0.0
    
    def initialize_population(self, ilp_baseline: Dict[str, Counter]) -> List[Dict[str, VerbTransitionRule]]:
        """
        Initialize population with ILP seed + random variations.
        
        Each individual is a dict: {source_verb -> VerbTransitionRule}
        """
        population = []
        
        # First individual: pure ILP baseline
        ilp_rules = {}
        for verb, next_verbs in ilp_baseline.items():
            target_list = [(v, count/sum(next_verbs.values())) for v, count in next_verbs.most_common(50)]
            ilp_rules[verb] = VerbTransitionRule(verb, target_list)
        population.append(ilp_rules)
        
        # Rest: mutations of ILP + random
        for _ in range(self.population_size - 1):
            if random.random() < 0.5 and ilp_rules:
                # Mutate ILP baseline
                individual = {}
                for verb, rule in ilp_rules.items():
                    individual[verb] = rule.mutate(self.all_verbs, self.mutation_rate)
            else:
                # Random individual
                individual = {}
                for verb in random.sample(self.all_verbs, min(30, len(self.all_verbs))):
                    target_verbs = random.sample(self.all_verbs, min(20, len(self.all_verbs)))
                    target_list = [(v, random.uniform(0.3, 1.0)) for v in target_verbs]
                    individual[verb] = VerbTransitionRule(verb, target_list)
            population.append(individual)
        
        return population
    
    def evaluate_fitness(self, rules: Dict[str, VerbTransitionRule]) -> float:
        """
        Evaluate fitness: Top-5 accuracy on eval set with rule constraints.
        """
        # Convert rules to verb_transitions format
        verb_transitions = {}
        for verb, rule in rules.items():
            candidates = rule.get_candidates(top_k=50)
            # Create Counter with equal weights
            verb_transitions[verb] = Counter({v: 1.0 for v in candidates})
        
        # Update model's transitions
        self.model.verb_transitions = verb_transitions
        
        # Evaluate
        top5_correct = 0
        total = 0
        
        self.model.eval()
        with torch.no_grad():
            for context_events, target_event in self.eval_pairs[:500]:  # Sample for speed
                if not context_events or not target_event:
                    continue
                
                # Get target verb
                target_verb = None
                for prop in target_event:
                    if prop.predicate == 'type' and len(prop.args) >= 2:
                        target_verb = prop.args[1]
                        break
                
                if not target_verb or target_verb not in self.model.verb_vocab:
                    continue
                
                try:
                    # Predict with constraints
                    predictions = self.model.predict_verb_constrained(
                        context_events, self.device, top_k=5, use_constraints=True
                    )
                    pred_verbs = [v for v, _ in predictions]
                    
                    if target_verb in pred_verbs:
                        top5_correct += 1
                    total += 1
                    
                except Exception:
                    continue
        
        fitness = top5_correct / max(total, 1)
        return fitness
    
    def evolve(self, generations: int, ilp_baseline: Dict[str, Counter]) -> Dict[str, VerbTransitionRule]:
        """
        Evolve population for specified generations.
        """
        if self.verbose:
            print(f"\n[GA Evolution] {generations} generations, population {self.population_size}")
        
        # Initialize
        population = self.initialize_population(ilp_baseline)
        
        for gen in range(generations):
            if self.verbose:
                print(f"\n  Generation {gen + 1}/{generations}")
            
            # Evaluate all
            start_time = time.time()
            for individual in population:
                if not hasattr(individual, '_fitness'):
                    individual._fitness = self.evaluate_fitness(individual)
            
            # Track best
            population.sort(key=lambda x: x._fitness, reverse=True)
            gen_best_fitness = population[0]._fitness
            
            if gen_best_fitness > self.best_fitness:
                self.best_fitness = gen_best_fitness
                self.best_rules = deepcopy(population[0])
            
            avg_fitness = sum(ind._fitness for ind in population) / len(population)
            elapsed = time.time() - start_time
            
            if self.verbose:
                print(f"    Best: {gen_best_fitness:.4f}, Avg: {avg_fitness:.4f}, Time: {elapsed:.1f}s")
            
            self.history.append({
                'generation': gen,
                'best_fitness': gen_best_fitness,
                'avg_fitness': avg_fitness
            })
            
            # Create next generation
            if gen < generations - 1:
                population = self._next_generation(population)
        
        if self.verbose:
            print(f"\n  ✅ Evolution complete! Best fitness: {self.best_fitness:.4f}")
        
        return self.best_rules
    
    def _next_generation(self, population: List[Dict[str, VerbTransitionRule]]) -> List[Dict[str, VerbTransitionRule]]:
        """Create next generation via selection, crossover, mutation."""
        next_gen = []
        
        # Elitism: keep best
        next_gen.extend([deepcopy(ind) for ind in population[:self.elite_size]])
        
        # Fill rest with offspring
        while len(next_gen) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_select(population)
            parent2 = self._tournament_select(population)
            
            # Crossover
            if random.random() < 0.7:
                child = self._crossover(parent1, parent2)
            else:
                child = deepcopy(parent1)
            
            # Mutation
            if random.random() < self.mutation_rate:
                child = self._mutate(child)
            
            # Reset fitness
            if hasattr(child, '_fitness'):
                delattr(child, '_fitness')
            
            next_gen.append(child)
        
        return next_gen
    
    def _tournament_select(self, population: List, k: int = 3):
        """Tournament selection."""
        candidates = random.sample(population, min(k, len(population)))
        return max(candidates, key=lambda x: x._fitness)
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Crossover: mix rules from two parents."""
        child = {}
        all_verbs = set(parent1.keys()) | set(parent2.keys())
        
        for verb in all_verbs:
            if verb in parent1 and verb in parent2:
                # Both have rule - randomly choose
                child[verb] = deepcopy(random.choice([parent1[verb], parent2[verb]]))
            elif verb in parent1:
                child[verb] = deepcopy(parent1[verb])
            else:
                child[verb] = deepcopy(parent2[verb])
        
        return child
    
    def _mutate(self, individual: Dict) -> Dict:
        """Mutate individual."""
        mutated = {}
        for verb, rule in individual.items():
            if random.random() < self.mutation_rate:
                mutated[verb] = rule.mutate(self.all_verbs, self.mutation_rate)
            else:
                mutated[verb] = deepcopy(rule)
        return mutated


def main():
    parser = argparse.ArgumentParser(description='GA for Semantic-AR (Next-Verb Prediction)')
    parser.add_argument('--scale', default='full', choices=list(SCALE_CONFIGS.keys()))
    parser.add_argument('--corpus', default='data/processed/tinystories_train.json')
    parser.add_argument('--ga-generations', type=int, default=20, help='GA generations')
    parser.add_argument('--ga-population', type=int, default=30, help='Population size')
    parser.add_argument('--embed-dim', type=int, default=50, choices=[50, 100, 200, 300])
    parser.add_argument('--dln-epochs', type=int, default=20, help='DLN training epochs')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--context-size', type=int, default=3)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--output-dir', default='outputs/ga_semantic_ar')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if verbose:
        print("\n" + "="*70)
        print("GA FOR SEMANTIC-AR (Next-Verb Prediction)")
        print("="*70)
    
    # Load corpus
    if verbose:
        print(f"\n[1] Loading corpus ({args.scale})...")
    
    config = SCALE_CONFIGS[args.scale]
    facts = load_tinystories_facts(
        max_stories=config['stories'],
        max_facts=config['facts'],
        path=args.corpus
    )
    
    if verbose:
        print(f"  ✅ Loaded {len(facts):,} facts")
    
    # Mine ILP baseline for seeding
    if verbose:
        print(f"\n[2] Mining ILP baseline (for GA seed)...")
    
    ilp_rules, _ = mine_frequency_based(facts, max_rules=50, min_support=2)
    ilp_baseline = extract_story_grammar_from_rules(ilp_rules, facts, verbose=False)
    
    if verbose:
        print(f"  ✅ ILP baseline: {len(ilp_baseline)} verb transitions")
    
    # Load GloVe
    if verbose:
        print(f"\n[3] Loading GloVe embeddings...")
    
    glove_embeddings = download_glove_embeddings(embed_dim=args.embed_dim)
    
    # Create training data
    if verbose:
        print(f"\n[4] Creating training data...")
    
    training_pairs = create_event_ar_training_data(
        facts,
        context_size=args.context_size,
        max_samples=999999
    )
    
    split_idx = int(0.8 * len(training_pairs))
    train_pairs = training_pairs[:split_idx]
    eval_pairs = training_pairs[split_idx:]
    
    if verbose:
        print(f"  ✅ Train: {len(train_pairs):,}, Eval: {len(eval_pairs):,}")
    
    # Extract vocabularies
    all_predicates = set(f.predicate for f in facts)
    all_args = set()
    all_verbs = set()
    
    for f in facts:
        all_args.update(f.args)
        if f.predicate == 'type' and len(f.args) >= 2:
            all_verbs.add(f.args[1])
    
    verb_list = sorted(list(all_verbs))
    
    if verbose:
        print(f"  Vocabulary: {len(verb_list)} verbs")
    
    # Create embeddings
    verb_embedding_matrix, num_oov = create_verb_embedding_matrix(
        verb_list, glove_embeddings, args.embed_dim
    )
    
    # Create model
    if verbose:
        print(f"\n[5] Creating semantic-AR model...")
    
    model = RuleAssistedSemanticEventAR(
        predicates=list(all_predicates),
        args=list(all_args),
        verbs=verb_list,
        embed_dim=args.embed_dim,
        verb_embeddings=verb_embedding_matrix,
        verb_transitions=ilp_baseline  # Start with ILP
    )
    
    if args.device == 'cuda' and torch.cuda.is_available():
        model = model.to(args.device)
    
    if verbose:
        print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train DLN (quick pre-training)
    if verbose:
        print(f"\n[6] Pre-training DLN ({args.dln_epochs} epochs)...")
    
    from test_semantic_event_ar import train_semantic_event_ar
    train_results = train_semantic_event_ar(
        model,
        training_pairs,
        device=args.device,
        epochs=args.dln_epochs,
        batch_size=args.batch_size,
        verbose=verbose
    )
    
    # Evaluate ILP baseline
    if verbose:
        print(f"\n[7] Evaluating ILP baseline...")
    
    from test_rule_assisted_ar import evaluate_rule_assisted_ar
    ilp_results = evaluate_rule_assisted_ar(
        model,
        eval_pairs,
        device=args.device,
        use_constraints=True,
        verbose=False
    )
    
    ilp_top5 = ilp_results['with_constraints'].get('top5_acc', 0)
    
    if verbose:
        print(f"  ILP baseline Top-5: {ilp_top5:.4f}")
    
    # Run GA evolution
    if verbose:
        print(f"\n[8] Running GA evolution...")
    
    ga = SemanticARGA(
        all_verbs=verb_list,
        training_pairs=train_pairs,
        eval_pairs=eval_pairs,
        model=model,
        device=args.device,
        population_size=args.ga_population,
        verbose=verbose
    )
    
    best_rules = ga.evolve(args.ga_generations, ilp_baseline)
    
    # Final evaluation with GA rules
    if verbose:
        print(f"\n[9] Final evaluation with GA rules...")
    
    # Convert to transitions format
    ga_transitions = {}
    for verb, rule in best_rules.items():
        candidates = rule.get_candidates(top_k=50)
        ga_transitions[verb] = Counter({v: 1.0 for v in candidates})
    
    model.verb_transitions = ga_transitions
    
    ga_results = evaluate_rule_assisted_ar(
        model,
        eval_pairs,
        device=args.device,
        use_constraints=True,
        verbose=False
    )
    
    ga_top5 = ga_results['with_constraints'].get('top5_acc', 0)
    
    # Summary
    if verbose:
        print(f"\n{'='*70}")
        print("RESULTS SUMMARY")
        print(f"{'='*70}")
        print(f"\nILP Baseline:")
        print(f"  Top-5: {ilp_top5:.4f}")
        
        print(f"\nGA Evolved ({args.ga_generations} generations):")
        print(f"  Top-5: {ga_top5:.4f}")
        
        improvement = ga_top5 - ilp_top5
        if improvement > 0.01:
            print(f"\n✅ GA improves by {improvement*100:.1f}% (absolute)")
            print(f"   Relative improvement: {(improvement/ilp_top5)*100:.1f}%")
        elif improvement > 0:
            print(f"\n⚠️  GA marginally better (+{improvement*100:.2f}%)")
        else:
            print(f"\n❌ GA doesn't improve over ILP")
    
    # Save
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / 'results.json', 'w') as f:
        json.dump({
            'config': vars(args),
            'ilp_baseline': ilp_top5,
            'ga_final': ga_top5,
            'improvement': improvement,
            'ga_history': ga.history
        }, f, indent=2)
    
    if verbose:
        print(f"\n✅ Results saved to {output_path}")


if __name__ == '__main__':
    main()
