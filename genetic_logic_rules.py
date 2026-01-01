"""
Genetic Algorithm for Logic Rules with Semantic-AR Fitness

Fresh approach: Evolve symbolic rules that maintain semantic consistency.
No neural networks - pure symbolic evolution.
"""

import random
import spacy
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import json

# Load spaCy for pattern matching
nlp = spacy.load('en_core_web_sm')


class SymbolicRule:
    """
    A discrete symbolic rule for text ↔ logic transformation.
    
    Parse direction: Text pattern → Logic template
    Generate direction: Logic template → Text pattern
    """
    
    def __init__(self):
        # Parse pattern (POS tags + optional dependencies)
        # e.g., ['NOUN', 'VERB', 'NOUN']
        self.parse_pattern = []
        
        # Logic template (slot names)
        # e.g., ['agent', 'action', 'patient']
        self.logic_template = []
        
        # Generation template (text with slots)
        # e.g., "The {agent} {action} the {patient}"
        self.gen_template = ""
        
        # Fitness tracking
        self.fitness = 0.0
        self.examples_matched = 0
    
    @staticmethod
    def random(max_length=5):
        """Create random rule."""
        rule = SymbolicRule()
        
        # Random POS pattern
        pos_tags = ['NOUN', 'VERB', 'ADJ', 'ADP', 'AUX', 'DET', 'ADV']
        length = random.randint(2, max_length)
        rule.parse_pattern = random.choices(pos_tags, k=length)
        
        # Corresponding logic template
        # Simplification: just extract nouns and verbs as slots
        slot_names = ['entity1', 'action', 'entity2', 'property', 'location']
        num_slots = min(length, 3)
        rule.logic_template = random.sample(slot_names, k=num_slots)
        
        # Simple generation template
        rule.gen_template = " ".join([f"{{{slot}}}" for slot in rule.logic_template])
        
        return rule
    
    def matches(self, text: str) -> bool:
        """Check if parse pattern matches text."""
        doc = nlp(text)
        
        # Extract POS sequence
        pos_seq = [token.pos_ for token in doc]
        
        # Simple substring matching
        pattern_len = len(self.parse_pattern)
        for i in range(len(pos_seq) - pattern_len + 1):
            if pos_seq[i:i+pattern_len] == self.parse_pattern:
                return True
        
        return False
    
    def parse(self, text: str) -> Optional[Dict[str, str]]:
        """
        Parse text to logic representation.
        
        Returns: Dict mapping slot names to values
        """
        if not self.matches(text):
            return None
        
        doc = nlp(text)
        pos_seq = [token.pos_ for token in doc]
        tokens = [token.text.lower() for token in doc]
        
        # Find pattern match
        pattern_len = len(self.parse_pattern)
        for i in range(len(pos_seq) - pattern_len + 1):
            if pos_seq[i:i+pattern_len] == self.parse_pattern:
                # Extract matched tokens
                matched_tokens = tokens[i:i+pattern_len]
                
                # Map to logic slots
                logic = {}
                for j, slot in enumerate(self.logic_template):
                    if j < len(matched_tokens):
                        logic[slot] = matched_tokens[j]
                
                return logic
        
        return None
    
    def generate(self, logic: Dict[str, str]) -> str:
        """
        Generate text from logic representation.
        
        Returns: Generated text string
        """
        try:
            return self.gen_template.format(**logic)
        except KeyError:
            # Missing slots - return partial
            result = self.gen_template
            for key, value in logic.items():
                result = result.replace(f"{{{key}}}", value)
            return result
    
    def is_semantically_consistent(self, text: str = None, logic: Dict[str, str] = None) -> bool:
        """
        Test semantic-AR property:
        - If text given: text → parse → logic → generate → text' → parse → logic'
          Check: logic == logic'
        - If logic given: logic → generate → text → parse → logic'
          Check: logic == logic'
        """
        try:
            if text is not None:
                # Forward test
                logic1 = self.parse(text)
                if logic1 is None:
                    return False
                
                text_gen = self.generate(logic1)
                logic2 = self.parse(text_gen)
                
                return logic2 is not None and logic1 == logic2
            
            elif logic is not None:
                # Backward test
                text_gen = self.generate(logic)
                logic_parsed = self.parse(text_gen)
                
                return logic_parsed is not None and logic == logic_parsed
        
        except Exception:
            return False
        
        return False
    
    def mutate(self):
        """Mutate this rule (in-place)."""
        mutation_type = random.choice(['pattern', 'template', 'generation'])
        
        if mutation_type == 'pattern' and len(self.parse_pattern) > 0:
            # Modify pattern
            pos_tags = ['NOUN', 'VERB', 'ADJ', 'ADP', 'AUX', 'DET', 'ADV']
            action = random.choice(['add', 'remove', 'change'])
            
            if action == 'add' and len(self.parse_pattern) < 6:
                pos = random.choice([0, len(self.parse_pattern)])
                self.parse_pattern.insert(pos, random.choice(pos_tags))
            
            elif action == 'remove' and len(self.parse_pattern) > 2:
                pos = random.randint(0, len(self.parse_pattern) - 1)
                self.parse_pattern.pop(pos)
            
            elif action == 'change':
                pos = random.randint(0, len(self.parse_pattern) - 1)
                self.parse_pattern[pos] = random.choice(pos_tags)
        
        elif mutation_type == 'template' and len(self.logic_template) > 0:
            # Modify logic template
            slot_names = ['entity1', 'action', 'entity2', 'property', 'location']
            action = random.choice(['add', 'remove', 'change'])
            
            if action == 'add' and len(self.logic_template) < 5:
                available = [s for s in slot_names if s not in self.logic_template]
                if available:
                    self.logic_template.append(random.choice(available))
            
            elif action == 'remove' and len(self.logic_template) > 1:
                pos = random.randint(0, len(self.logic_template) - 1)
                self.logic_template.pop(pos)
            
            elif action == 'change':
                pos = random.randint(0, len(self.logic_template) - 1)
                available = [s for s in slot_names if s not in self.logic_template]
                if available:
                    self.logic_template[pos] = random.choice(available)
        
        elif mutation_type == 'generation':
            # Modify generation template
            self.gen_template = " ".join([f"{{{slot}}}" for slot in self.logic_template])
    
    def crossover(self, other: 'SymbolicRule') -> 'SymbolicRule':
        """Create child by crossing with another rule."""
        child = SymbolicRule()
        
        # Inherit pattern from one parent
        child.parse_pattern = random.choice([self.parse_pattern[:], other.parse_pattern[:]])
        
        # Inherit template from possibly different parent
        child.logic_template = random.choice([self.logic_template[:], other.logic_template[:]])
        
        # Inherit generation from possibly different parent
        child.gen_template = random.choice([self.gen_template, other.gen_template])
        
        return child
    
    def __str__(self):
        return f"Rule(pattern={self.parse_pattern}, logic={self.logic_template}, fitness={self.fitness:.3f})"


def semantic_ar_fitness(rule: SymbolicRule, dataset: List[Tuple[str, str]]) -> float:
    """
    Fitness function based on semantic-AR consistency.
    
    Tests:
    1. Can rule parse examples correctly?
    2. Does rule maintain semantic consistency in round-trips?
    
    Args:
        rule: The symbolic rule to evaluate
        dataset: List of (text, logic) pairs
    
    Returns:
        fitness: Score from 0 to 1
    """
    total_score = 0
    matches = 0
    
    for text, target_logic_str in dataset:
        # Test if rule matches this example
        if not rule.matches(text):
            continue
        
        matches += 1
        
        # Test semantic consistency (key!)
        if rule.is_semantically_consistent(text=text):
            total_score += 1.0
        else:
            # Partial credit for matching
            total_score += 0.3
    
    rule.examples_matched = matches
    
    if matches == 0:
        rule.fitness = 0.0
    else:
        # Fitness = consistency + coverage bonus
        consistency = total_score / matches
        coverage = min(matches / len(dataset), 0.2)  # Up to 20% bonus for coverage
        rule.fitness = 0.8 * consistency + 0.2 * coverage
    
    return rule.fitness


def tournament_selection(population: List[SymbolicRule], k: int = 3) -> SymbolicRule:
    """Select best rule from k random candidates."""
    candidates = random.sample(population, k)
    return max(candidates, key=lambda r: r.fitness)


def genetic_algorithm(dataset: List[Tuple[str, str]], 
                      population_size: int = 100,
                      generations: int = 50,
                      mutation_rate: float = 0.2,
                      elite_size: int = 5) -> List[SymbolicRule]:
    """
    Evolve symbolic rules using GA with semantic-AR fitness.
    
    Args:
        dataset: Training examples (text, logic) pairs
        population_size: Number of rules in population
        generations: Number of evolutionary generations
        mutation_rate: Probability of mutation
        elite_size: Number of best rules to keep each generation
    
    Returns:
        Best rules found
    """
    print("=" * 70)
    print("Genetic Algorithm for Semantic-AR Rules")
    print("=" * 70)
    print(f"Dataset size: {len(dataset)}")
    print(f"Population: {population_size}")
    print(f"Generations: {generations}")
    print(f"Mutation rate: {mutation_rate}")
    print()
    
    # Initialize population
    print("Initializing random population...")
    population = [SymbolicRule.random() for _ in range(population_size)]
    
    best_overall = None
    best_fitness = 0
    
    for gen in range(generations):
        # Evaluate fitness
        for rule in population:
            semantic_ar_fitness(rule, dataset)
        
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
              f"Matches: {gen_best.examples_matched} | "
              f"Pattern: {gen_best.parse_pattern}")
        
        # Early stopping
        if gen_best.fitness > 0.95:
            print(f"\n✓ High fitness achieved! Stopping early.")
            break
        
        # Selection and reproduction
        new_population = []
        
        # Elitism: keep best rules
        new_population.extend(population[:elite_size])
        
        # Generate offspring
        while len(new_population) < population_size:
            # Tournament selection
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            
            # Crossover
            child = parent1.crossover(parent2)
            
            # Mutation
            if random.random() < mutation_rate:
                child.mutate()
            
            new_population.append(child)
        
        population = new_population
    
    print("\n" + "=" * 70)
    print("Evolution Complete!")
    print("=" * 70)
    print(f"Best fitness: {best_fitness:.3f}")
    print(f"Best rule: {best_overall}")
    print()
    
    # Return top 10 rules
    population.sort(key=lambda r: r.fitness, reverse=True)
    return population[:10]


if __name__ == "__main__":
    # Test with simple examples
    print("Testing GA with simple examples...\n")
    
    # Simple test dataset
    test_dataset = [
        ("The cat sat", "cat sit"),
        ("The dog ran", "dog run"),
        ("A bird flew", "bird fly"),
        ("The cat jumped", "cat jump"),
        ("A dog walked", "dog walk"),
    ]
    
    # Run GA
    best_rules = genetic_algorithm(
        test_dataset,
        population_size=50,
        generations=30,
        mutation_rate=0.3
    )
    
    # Test best rules
    print("\nTop 5 Rules:")
    print("-" * 70)
    for i, rule in enumerate(best_rules[:5], 1):
        print(f"{i}. {rule}")
        print(f"   Generation: {rule.gen_template}")
        
        # Test on examples
        print(f"   Tests:")
        for text, _ in test_dataset[:3]:
            if rule.matches(text):
                logic = rule.parse(text)
                print(f"     '{text}' → {logic}")
        print()
