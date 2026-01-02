"""
Simple Forward Chainer with RETE-inspired Optimizations
=======================================================

Lightweight production rule system for knowledge base reasoning.
More transparent and controllable than Experta.
"""

from typing import List, Tuple, Dict, Set, Callable, Any
from dataclasses import dataclass
import re


@dataclass
class Fact:
    """A fact is a triple: (entity, relation, value)."""
    entity: str
    relation: str
    value: str
    
    def __hash__(self):
        return hash((self.entity, self.relation, self.value))
    
    def __eq__(self, other):
        return (self.entity == other.entity and 
                self.relation == other.relation and 
                self.value == other.value)
    
    def __repr__(self):
        return f"[{self.entity}, {self.relation}, {self.value}]"


@dataclass
class Rule:
    """A production rule: patterns → action."""
    name: str
    patterns: List[Dict[str, Any]]  # List of patterns to match
    action: Callable  # Function that generates new facts
    
    def __repr__(self):
        return f"Rule({self.name})"


class ForwardChainer:
    """
    Simple forward chaining inference engine.
    Uses incremental matching for efficiency (RETE-inspired).
    """
    
    def __init__(self):
        self.facts: Set[Fact] = set()
        self.rules: List[Rule] = []
        self.rule_index: Dict[str, List[Rule]] = {}  # relation → rules
        self.iterations = 0
        self.max_iterations = 100
        
    def add_fact(self, entity: str, relation: str, value: str):
        """Add a fact to working memory."""
        fact = Fact(entity, relation, value)
        if fact not in self.facts:
            self.facts.add(fact)
            return True
        return False
    
    def add_rule(self, rule: Rule):
        """Add a production rule."""
        self.rules.append(rule)
        
        # Index by first pattern's relation (RETE alpha network idea)
        if rule.patterns:
            first_rel = rule.patterns[0].get('relation')
            if first_rel:
                if first_rel not in self.rule_index:
                    self.rule_index[first_rel] = []
                self.rule_index[first_rel].append(rule)
    
    def match_pattern(self, pattern: Dict[str, Any], bindings: Dict[str, str] = None) -> List[Dict[str, str]]:
        """
        Match a pattern against facts, with variable bindings.
        
        Pattern example: {'entity': '?x', 'relation': 'type', 'value': 'dead'}
        This matches facts like [mary, type, dead] with binding {?x: mary}
        """
        if bindings is None:
            bindings = {}
        
        results = []
        
        for fact in self.facts:
            # Try to match this fact
            new_bindings = bindings.copy()
            match = True
            
            for key in ['entity', 'relation', 'value']:
                pattern_val = pattern.get(key)
                fact_val = getattr(fact, key)
                
                if pattern_val is None:
                    continue
                elif pattern_val.startswith('?'):  # Variable
                    var_name = pattern_val
                    if var_name in new_bindings:
                        # Variable already bound, check consistency
                        if new_bindings[var_name] != fact_val:
                            match = False
                            break
                    else:
                        # Bind variable
                        new_bindings[var_name] = fact_val
                else:  # Constant
                    if pattern_val != fact_val:
                        match = False
                        break
            
            if match:
                results.append(new_bindings)
        
        return results
    
    def match_rule(self, rule: Rule) -> List[Dict[str, str]]:
        """Match all patterns in a rule, return successful bindings."""
        if not rule.patterns:
            return []
        
        # Start with first pattern
        bindings_list = self.match_pattern(rule.patterns[0])
        
        # Match remaining patterns with existing bindings
        for pattern in rule.patterns[1:]:
            new_bindings_list = []
            for bindings in bindings_list:
                matches = self.match_pattern(pattern, bindings)
                new_bindings_list.extend(matches)
            bindings_list = new_bindings_list
        
        return bindings_list
    
    def fire_rule(self, rule: Rule, bindings: Dict[str, str]) -> List[Fact]:
        """Execute rule action with bindings, return new facts."""
        return rule.action(bindings)
    
    def forward_chain(self, verbose: bool = True):
        """Run forward chaining until fixpoint."""
        self.iterations = 0
        changed = True
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Forward Chaining (RETE-inspired)")
            print(f"Initial facts: {len(self.facts)}")
            print(f"{'='*70}")
        
        while changed and self.iterations < self.max_iterations:
            self.iterations += 1
            changed = False
            initial_count = len(self.facts)
            
            # Try each rule
            for rule in self.rules:
                # Match rule against facts
                bindings_list = self.match_rule(rule)
                
                for bindings in bindings_list:
                    # Fire rule
                    new_facts = self.fire_rule(rule, bindings)
                    
                    # Add new facts
                    for fact in new_facts:
                        if self.add_fact(fact.entity, fact.relation, fact.value):
                            changed = True
                            if verbose:
                                print(f"  [Iteration {self.iterations}] {rule.name}")
                                print(f"    Bindings: {bindings}")
                                print(f"    → {fact}")
            
            if verbose and changed:
                print(f"  Facts after iteration {self.iterations}: {len(self.facts)}")
        
        if verbose:
            print(f"{'='*70}")
            print(f"Convergence reached after {self.iterations} iterations")
            print(f"Final facts: {len(self.facts)}")
            print(f"{'='*70}\n")
    
    def get_facts_by_relation(self, relation: str) -> List[Fact]:
        """Query facts by relation."""
        return [f for f in self.facts if f.relation == relation]
    
    def get_facts_by_entity(self, entity: str) -> List[Fact]:
        """Query facts by entity."""
        return [f for f in self.facts if f.entity == entity]


def create_common_sense_rules() -> List[Rule]:
    """
    Create common sense reasoning rules.
    These are reified from natural language descriptions.
    """
    rules = []
    
    # Rule 1: guillotine(X) → dead(X)
    def guillotine_kills(bindings):
        x = bindings['?x']
        return [
            Fact(x, 'state', 'dead'),
        ]
    
    rules.append(Rule(
        name="guillotine_kills",
        patterns=[
            {'entity': '?x', 'relation': 'state', 'value': 'guillotined'}
        ],
        action=guillotine_kills
    ))
    
    # Rule 2: behead(X) → dead(X)
    def behead_kills(bindings):
        x = bindings['?x']
        return [Fact(x, 'state', 'dead')]
    
    rules.append(Rule(
        name="behead_kills",
        patterns=[
            {'entity': '?x', 'relation': 'state', 'value': 'beheaded'}
        ],
        action=behead_kills
    ))
    
    # Rule 3: dead(X) → cannot_act(X)
    def dead_cannot_act(bindings):
        x = bindings['?x']
        return [Fact(x, 'ability', 'cannot_act')]
    
    rules.append(Rule(
        name="dead_cannot_act",
        patterns=[
            {'entity': '?x', 'relation': 'state', 'value': 'dead'}
        ],
        action=dead_cannot_act
    ))
    
    # Rule 4: execute(X) → dead(X)
    def execute_kills(bindings):
        x = bindings['?x']
        return [Fact(x, 'state', 'dead')]
    
    rules.append(Rule(
        name="execute_kills",
        patterns=[
            {'entity': '?x', 'relation': 'state', 'value': 'executed'}
        ],
        action=execute_kills
    ))
    
    return rules


def test_forward_chainer():
    """Test the forward chainer with Mary Queen of Scots example."""
    print("="*70)
    print("Forward Chainer Test: Common Sense Reasoning")
    print("="*70)
    
    # Create knowledge base
    kb = ForwardChainer()
    
    # Add common sense rules
    for rule in create_common_sense_rules():
        kb.add_rule(rule)
    
    print(f"\nAdded {len(kb.rules)} common sense rules:")
    for rule in kb.rules:
        print(f"  - {rule.name}")
    
    # Test 1: Mary was guillotined
    print("\n" + "="*70)
    print("Test 1: Mary Queen of Scots was guillotined")
    print("="*70)
    
    kb.add_fact('mary', 'type', 'person')
    kb.add_fact('mary', 'state', 'guillotined')
    
    print("\nInitial facts:")
    for fact in kb.facts:
        print(f"  {fact}")
    
    kb.forward_chain()
    
    print("Inferred facts:")
    for fact in kb.get_facts_by_entity('mary'):
        print(f"  {fact}")
    
    # Test 2: John was beheaded
    print("\n" + "="*70)
    print("Test 2: John was beheaded")
    print("="*70)
    
    kb2 = ForwardChainer()
    for rule in create_common_sense_rules():
        kb2.add_rule(rule)
    
    kb2.add_fact('john', 'type', 'person')
    kb2.add_fact('john', 'state', 'beheaded')
    
    kb2.forward_chain()
    
    print("Inferred facts:")
    for fact in kb2.get_facts_by_entity('john'):
        print(f"  {fact}")


if __name__ == "__main__":
    test_forward_chainer()
