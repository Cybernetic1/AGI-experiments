# Rule Discovery Module: ILP Integration for AGI Architecture

**Date:** 2026-01-17  
**Insight:** Chain/mined rules are doing ILP - should be a dedicated module

---

## The Key Insight: You're Doing ILP!

### What You're Currently Doing:

```python
# From mine_chain_rules():
def mine_chain_rules(facts, max_rules=10, min_support=2):
    # Count co-occurrences of chained relations
    for f1 in facts:
        for f2 in facts:
            if f1.args[1] == f2.args[0]:  # Chain pattern
                counts[(f1.predicate, f2.predicate)] += 1
    
    # Keep frequent patterns
    for (p1, p2), count in sorted_by_frequency:
        if count >= min_support:
            rules.append(Rule([P(p1,(?x,?y)), P(p2,(?y,?z))], 
                             P(f"{p1}_{p2}", (?x,?z))))
```

**This IS classic ILP!**
- Frequent pattern mining
- Rule induction from examples
- Support threshold (like FOIL, Progol)

### ILP Fundamentals:

```
Inductive Logic Programming:
Input:  Background knowledge + Positive/Negative examples
Output: Logic rules that explain the examples

Example:
Background: parent(X,Y) facts
Positive:   grandparent(john, bob)
Learn:      grandparent(X,Z) :- parent(X,Y), parent(Y,Z)
```

---

## Why Chain Rules Are Actually Helpful

### 1. Transitivity Discovery

**Chains capture transitive relationships:**

```python
# Data has:
friend(alice, bob)
friend(bob, charlie)

# Chain rule discovers:
friend(?x, ?y) ∧ friend(?y, ?z) → friend_chain(?x, ?z)

# Captures: "friend of friend" relationship
```

**This is compositional reasoning!**

### 2. Implicit Relation Discovery

**Finds indirect connections:**

```python
# Data has:
owns(lily, box)
contains(box, toy)

# Chain discovers:
owns(?x, ?y) ∧ contains(?y, ?z) → owns_contains(?x, ?z)

# Semantic meaning: "indirectly possesses"
```

**This creates new relations not explicitly in data!**

### 3. Path Finding

**Multi-hop reasoning:**

```python
# Chain 1: parent(X,Y) ∧ parent(Y,Z) → grandparent(X,Z)
# Chain 2: parent(X,Y) ∧ grandparent(Y,Z) → great_grandparent(X,Z)
# Chain 3: sibling(X,Y) ∧ parent(Y,Z) → uncle(X,Z)
```

**Discovers family relationships automatically!**

### 4. Knowledge Graph Completion

**Infers missing edges:**

```python
# Known: capital_of(paris, france), located_in(eiffel_tower, paris)
# Infer: located_in(eiffel_tower, france)
# Via: located_in(?x,?y) ∧ capital_of(?y,?z) → located_in(?x,?z)
```

**This is what knowledge graphs need!**

---

## Current Implementation: Strengths & Weaknesses

### ✅ Strengths:

1. **Data-driven** - discovers actual patterns
2. **Compositional** - uses variables for abstraction
3. **Efficient** - simple frequent pattern mining
4. **Works!** - generates useful training labels

### ❌ Weaknesses:

1. **Limited to binary chains** - only `P1(?x,?y) ∧ P2(?y,?z)`
2. **No negative examples** - can't learn constraints
3. **No pruning** - doesn't eliminate spurious patterns
4. **No recursion** - can't discover recursive rules
5. **No confidence** - all rules have weight 1.0
6. **No subsumption** - can't generalize chains to patterns

---

## Proposed Architecture: Dedicated ILP Module

### Module Structure:

```
┌─────────────────────────────────────────────────────────────┐
│                    AGI ARCHITECTURE                          │
└─────────────────────────────────────────────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
        ┌───────────▼──────────┐ ┌───────▼────────────┐
        │ NL Parser            │ │ Rule Discovery     │
        │ (Davidsonian)        │ │ (ILP Module)       │ ← NEW!
        │                      │ │                    │
        │ Text → Propositions  │ │ Facts → Rules      │
        └───────────┬──────────┘ └──────┬─────────────┘
                    │                   │
                    └────────┬──────────┘
                             │
                    ┌────────▼──────────┐
                    │ Knowledge Base    │
                    │ (Facts + Rules)   │
                    └────────┬──────────┘
                             │
                ┌────────────┼────────────┐
                │            │            │
     ┌──────────▼──┐  ┌─────▼─────┐  ┌──▼──────────┐
     │ Symbolic    │  │ Neural    │  │ Hybrid      │
     │ Reasoner    │  │ DLN       │  │ Inference   │
     │             │  │           │  │             │
     │ Fast/exact  │  │ Learn     │  │ Best of     │
     │ inference   │  │ patterns  │  │ both        │
     └─────────────┘  └───────────┘  └─────────────┘
```

### ILP Module Interface:

```python
class RuleDiscoveryModule:
    """
    Inductive Logic Programming module for rule discovery.
    Discovers logical rules from facts using various ILP algorithms.
    """
    
    def __init__(self, algorithms=['chain', 'inverse', 'bridge'], 
                 min_support=2, min_confidence=0.7):
        self.algorithms = algorithms
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.discovered_rules = []
    
    def discover_rules(self, facts: List[Proposition], 
                      background_knowledge: List[Rule] = None) -> List[Rule]:
        """
        Discover rules from facts using configured algorithms.
        
        Args:
            facts: Training examples (positive instances)
            background_knowledge: Pre-existing rules to build upon
        
        Returns:
            List of discovered rules with confidence scores
        """
        rules = []
        
        if 'chain' in self.algorithms:
            rules.extend(self._discover_chains(facts))
        
        if 'inverse' in self.algorithms:
            rules.extend(self._discover_inverse(facts))
        
        if 'bridge' in self.algorithms:
            rules.extend(self._discover_bridge(facts))
        
        # Filter by support and confidence
        rules = self._filter_rules(rules, facts)
        
        return rules
    
    def _discover_chains(self, facts):
        """Discover binary chain rules: P1(?x,?y) ∧ P2(?y,?z) → P3(?x,?z)"""
        # Current implementation (improved)
        pass
    
    def _discover_inverse(self, facts):
        """Discover inverse rules: P(?x,?y) → P_inv(?y,?x)"""
        # NEW: Discover symmetry and inverse relations
        pass
    
    def _discover_bridge(self, facts):
        """Discover bridge rules: P1(?x,?y) ∧ P2(?x,?z) → P3(?y,?z)"""
        # NEW: Different variable binding pattern
        pass
    
    def _filter_rules(self, rules, facts):
        """Filter by support, confidence, and semantic validity"""
        # Statistical validation
        # Subsumption checking
        # Negative example checking (if available)
        pass
```

---

## Advanced ILP Algorithms to Integrate

### 1. FOIL (First Order Inductive Learner)

**Classic ILP algorithm:**

```python
def foil_algorithm(positive_examples, negative_examples):
    """
    Learn rules by greedily adding literals that maximize information gain.
    """
    rule = Rule([], target_predicate)
    
    while not covers_all_positive(rule):
        best_literal = None
        best_gain = 0
        
        for candidate in generate_candidate_literals():
            gain = information_gain(rule + candidate, 
                                   positive_examples, 
                                   negative_examples)
            if gain > best_gain:
                best_literal = candidate
                best_gain = gain
        
        rule.add_literal(best_literal)
    
    return rule
```

**Advantages:**
- Uses negative examples (learns what NOT to predict)
- Information gain metric (better than frequency alone)
- Learns complex rules (not just chains)

### 2. Progol (Prolog + ILP)

**Logic programming approach:**

```python
def progol_algorithm(examples, background_knowledge):
    """
    Bottom-up then top-down search for most general rules.
    """
    # Build most specific rule (bottom clause)
    bottom = saturate(examples[0], background_knowledge)
    
    # Generalize to most general rule (top clause)
    top = generalize(bottom)
    
    # Search lattice between bottom and top
    rule = search_lattice(bottom, top, examples)
    
    return rule
```

**Advantages:**
- More systematic search
- Finds most general rules
- Handles complex background knowledge

### 3. Neural-Symbolic ILP (Modern)

**Combine neural networks with ILP:**

```python
class NeuralILP:
    """
    Use neural networks to guide ILP search.
    """
    
    def __init__(self, dln_model):
        self.dln = dln_model  # Your existing DLN!
        self.ilp_engine = ILPEngine()
    
    def discover_rules(self, facts):
        # 1. Neural embedding guides which patterns to try
        embeddings = self.dln.encode_premises(facts)
        candidate_patterns = cluster_embeddings(embeddings)
        
        # 2. ILP validates and refines patterns
        rules = []
        for pattern in candidate_patterns:
            rule = self.ilp_engine.induce_rule(pattern, facts)
            if rule.confidence > threshold:
                rules.append(rule)
        
        return rules
```

**Advantages:**
- Neural network finds promising patterns (fast)
- ILP ensures logical validity (precise)
- Best of both worlds!

---

## Integration with Your Current System

### Step 1: Extract Current Logic

```python
# Currently in tinystories_pipeline.py (lines 93-124)
# Move to dedicated module:

# OLD:
def mine_chain_rules(facts, max_rules=10, min_support=2):
    # ... inline implementation

# NEW:
from rule_discovery import RuleDiscoveryModule

ilp = RuleDiscoveryModule(
    algorithms=['chain', 'inverse', 'bridge'],
    min_support=2,
    min_confidence=0.7
)

discovered_rules = ilp.discover_rules(facts)
```

### Step 2: Extend Capabilities

```python
# Add more rule templates:

class RuleDiscoveryModule:
    
    def _discover_inverse(self, facts):
        """Discover inverse relations"""
        pairs = {}
        for f in facts:
            if len(f.args) == 2:
                fwd = (f.predicate, f.args[0], f.args[1])
                bwd = (f.predicate, f.args[1], f.args[0])
                pairs[fwd] = pairs.get(fwd, 0) + 1
                pairs[bwd] = pairs.get(bwd, 0) + 1
        
        # If P(x,y) and P(y,x) both common → symmetric
        # If P(x,y) common but P(y,x) rare → asymmetric
        
        rules = []
        for (pred, x, y), count in pairs.items():
            reverse_count = pairs.get((pred, y, x), 0)
            if count > min_support and reverse_count > min_support:
                # Symmetric relation
                rules.append(
                    Rule([Prop(pred, (?x,?y))], 
                         Prop(pred, (?y,?x)), 
                         confidence=reverse_count/count)
                )
        
        return rules
    
    def _discover_bridge(self, facts):
        """Bridge pattern: P1(?x,?y) ∧ P2(?x,?z) → P3(?y,?z)"""
        # Find facts that share first argument
        by_first = {}
        for f in facts:
            by_first.setdefault(f.args[0], []).append(f)
        
        counts = {}
        for arg, fs in by_first.items():
            if len(fs) >= 2:
                for f1 in fs:
                    for f2 in fs:
                        if f1.predicate != f2.predicate:
                            key = (f1.predicate, f2.predicate, 
                                   f1.args[1], f2.args[1])
                            counts[key] = counts.get(key, 0) + 1
        
        rules = []
        for (p1, p2, y, z), count in counts.items():
            if count >= min_support:
                rules.append(
                    Rule([Prop(p1, (?x,?y)), Prop(p2, (?x,?z))],
                         Prop(f"{p1}_{p2}_bridge", (?y,?z)),
                         confidence=count/total)
                )
        
        return rules
```

### Step 3: Add Confidence Scoring

```python
def calculate_confidence(rule, facts):
    """
    Calculate rule confidence: P(conclusion | premises)
    """
    # Count how often premises hold
    premise_count = count_matches(rule.premises, facts)
    
    # Count how often conclusion also holds
    both_count = count_matches(rule.premises + [rule.conclusion], facts)
    
    if premise_count == 0:
        return 0.0
    
    confidence = both_count / premise_count
    return confidence
```

---

## Benefits of Modularization

### 1. Separation of Concerns

```
┌─────────────────────┐
│ Rule Discovery      │ ← Logic for finding patterns
│ (ILP Module)        │   Independent of neural network
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Rule Learning       │ ← Neural network learns to predict
│ (DLN)               │   Independent of rule discovery
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Rule Application    │ ← Symbolic engine applies rules
│ (SymbolicEngine)    │   Independent of both
└─────────────────────┘
```

### 2. Swappable Algorithms

```python
# Easy to compare different ILP methods:

# Frequency-based (current)
ilp = RuleDiscoveryModule(algorithms=['chain'])

# Information-gain (FOIL)
ilp = RuleDiscoveryModule(algorithms=['foil'])

# Neural-guided (modern)
ilp = NeuralILP(dln_model=your_dln)

# Hybrid
ilp = RuleDiscoveryModule(algorithms=['chain', 'foil', 'neural'])
```

### 3. Testability

```python
# Can test ILP independently:

def test_chain_discovery():
    facts = [
        Proposition("parent", ("alice", "bob")),
        Proposition("parent", ("bob", "charlie")),
    ]
    
    ilp = RuleDiscoveryModule(algorithms=['chain'])
    rules = ilp.discover_rules(facts)
    
    assert len(rules) == 1
    assert rules[0].conclusion.predicate == "parent_parent_chain"
    # Tests: grandparent discovered!
```

### 4. Extensibility

```python
# Easy to add new algorithms:

class RuleDiscoveryModule:
    
    def discover_rules(self, facts):
        if 'chain' in self.algorithms:
            rules.extend(self._discover_chains(facts))
        
        if 'recursive' in self.algorithms:  # NEW!
            rules.extend(self._discover_recursive(facts))
        
        if 'probabilistic' in self.algorithms:  # NEW!
            rules.extend(self._discover_probabilistic(facts))
        
        return rules
```

---

## Integration Points with AGI Architecture

### 1. Bootstrapping

```python
# Use ILP to bootstrap knowledge:

# Phase 1: Parse text → facts
facts = davidsonian_parser.extract(text_corpus)

# Phase 2: Discover rules from facts
ilp = RuleDiscoveryModule()
discovered_rules = ilp.discover_rules(facts)

# Phase 3: Train DLN on discovered rules
labels = symbolic_engine.infer(facts, discovered_rules)
dln.train(labels)
```

### 2. Active Learning

```python
# ILP suggests which rules to learn next:

current_rules = rule_store.get_all()
performance = evaluate(current_rules)

# Find rules that would improve performance most
candidate_rules = ilp.suggest_candidates(facts, current_rules)

# DLN learns most promising candidates first
for rule in sorted(candidate_rules, key=lambda r: r.expected_gain):
    dln.train_on_rule(rule)
```

### 3. Rule Refinement

```python
# Neural network refines symbolic rules:

# ILP discovers: parent(X,Y) ∧ parent(Y,Z) → grandparent(X,Z)
symbolic_rule = ilp.discover_chains(facts)

# DLN learns exceptions and confidence:
refined_rule = dln.refine_rule(symbolic_rule, facts)
# Output: parent(X,Y) ∧ parent(Y,Z) → grandparent(X,Z) [confidence=0.95]
```

### 4. Interpretable Learning

```python
# DLN predictions → ILP explanation:

prediction = dln.predict(query)
explanation = ilp.explain(prediction, facts, rules)

# Output:
# "Predicted: grandparent(john, bob) [0.94]
#  Because: parent(john, mary) [1.0]
#           parent(mary, bob) [1.0]
#           Rule: parent(X,Y) ∧ parent(Y,Z) → grandparent(X,Z) [0.95]"
```

---

## Recommended Implementation Plan

### Week 1: Extract & Modularize
- Create `rule_discovery.py` module
- Move `mine_chain_rules()` into module
- Add tests for chain discovery
- Update `tinystories_pipeline.py` to use module

### Week 2: Add Rule Types
- Implement `_discover_inverse()` (symmetric/inverse relations)
- Implement `_discover_bridge()` (different binding patterns)
- Add confidence scoring
- Test on TinyStories data

### Week 3: Integrate FOIL
- Implement information gain metric
- Add negative example handling
- Compare frequency vs information gain
- Benchmark on rule quality

### Week 4: Neural-Symbolic Hybrid
- Use DLN embeddings to guide search
- Cluster similar patterns
- Validate with symbolic checking
- Measure improvement over pure ILP

---

## Conclusion

**Your insight is exactly right!**

✅ **Chain rules ARE the valuable part** (compositional reasoning)
✅ **This IS classical ILP** (rule induction from examples)
✅ **Should be a dedicated module** (cleaner architecture)
✅ **Fits naturally into AGI system** (symbolic-neural bridge)

### Key Benefits:

1. **Cleaner separation** - Rule discovery vs rule learning vs rule application
2. **More extensible** - Easy to add new ILP algorithms
3. **Better testing** - Each component testable independently
4. **Richer rules** - Beyond just chains (inverse, bridge, recursive)
5. **Confidence scores** - Not all rules equal weight
6. **Integration point** - Bridge between symbolic and neural

### Next Steps:

1. Extract current rule mining into `RuleDiscoveryModule`
2. Add more rule templates (inverse, bridge)
3. Integrate FOIL for better rule quality
4. Use DLN embeddings to guide ILP search

**This would make your architecture significantly stronger!**
