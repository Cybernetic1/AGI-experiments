# Why Natural Language Parsing Failed: Deep Analysis

**Date:** 2026-01-02  
**Context:** After successfully solving TicTacToe but failing on NL

---

## Executive Summary

**The Crux:** NL parsing is fundamentally harder than TicTacToe by ~1000x in multiple dimensions:

| Dimension | TicTacToe | Natural Language |
|-----------|-----------|------------------|
| State space | 3^9 ≈ 20K | Vocabulary^N (infinite) |
| Rules needed (R_K) | ~10-50 | ~1000-10000+ |
| Ambiguity | None (deterministic) | Extreme (context-dependent) |
| Compositionality | Flat (board state) | Recursive (nested phrases) |
| Training signal | Clear (win/lose) | Unclear (what is "correct parse"?) |

---

## Root Causes of Failure

### 1. **Insufficient Rule Capacity** (R << R_K)

**TicTacToe:**
- Kolmogorov complexity: ~10-50 patterns
- Our capacity: 100-500 rules
- **Result: Success** (over-provisioned)

**Natural Language:**
- Kolmogorov complexity: ~1000-10000+ patterns
- Our capacity: 100 rules
- **Result: Failure** (under-provisioned by 10-100x)

**Evidence:**
- Entity accuracy: 2% (shows *some* learning with 100 rules)
- Extrapolating: 1000-5000 rules might reach 20-50% accuracy

### 2. **Compositional Complexity**

**Problem:** NL has recursive structure, rules need to compose.

```
TTT:     board_pattern → action
         (flat mapping)

NL:      NP → DET + NOUN
         VP → VERB + NP
         S  → NP + VP
         (recursive, hierarchical)
```

**Our rules:** Flat POS patterns (no recursion)

**Solution needed:** 
- Hierarchical/recursive rule application
- Rules that output intermediate representations
- Stack-based parsing (context-free grammar)

### 3. **Ambiguity & Context-Dependence**

**TicTacToe:**
- Board state uniquely determines best action
- No ambiguity

**Natural Language:**
- "Time flies like an arrow" (3+ valid parses)
- "Bank" = financial institution OR river edge?
- Meaning depends on distant context

**Current approach:** Rules match local POS patterns only

**Solution needed:**
- Context features (previous words, discourse state)
- Probabilistic rules (confidence scores)
- Weighted voting (already implemented!) helps

### 4. **Weak Training Signal**

**TicTacToe:**
```python
fitness = win_rate  # Crystal clear!
```

**Natural Language:**
```python
fitness = semantic_consistency(parse(generate(logic))) 
# What does this even mean?
# No ground truth for "correct" logic representation
```

**Problems:**
- No clear objective (unlike "win the game")
- Circular dependency (parse ↔ generate both being learned)
- Evaluation requires external knowledge (human judgment)

### 5. **Representation Mismatch**

**Current:** `(entity, relation, value)` triples

**Problems:**
- Forces everything into subject-verb-object form
- Can't represent:
  - Quantifiers: "all", "some", "three cats"
  - Modifiers: "the very big red house"
  - Nested clauses: "The cat that ate the mouse ran"
  - Coordination: "cats and dogs"

**Example failure:**
```
Input:  "The three big cats quickly ran away"
Current: entity=?, relation=?, value=?  # Doesn't fit!
Needed:  [agent: [det: the, count: 3, adj: big, noun: cats],
          action: [adv: quickly, verb: ran, prep: away]]
```

---

## What Worked in TicTacToe (Lessons)

✅ **Weighted voting** - Improved 31% over first-match  
✅ **Fast convergence** - 1-4 generations sufficient  
✅ **Simple patterns** - Direct board→action mapping  
✅ **Clear fitness** - Win rate objective  
✅ **Sufficient capacity** - 100 rules > 50 patterns needed  

---

## Concrete Improvement Strategies

### Strategy 1: **Massive Scale-Up** (Brute Force)

**Approach:** Throw more rules at the problem

```python
population_size = 10_000      # 100x increase
generations = 500             # More time to evolve
use_weighted_voting = True    # From TTT success
```

**Pros:**
- Simple, requires minimal changes
- Worked for TTT

**Cons:**
- May hit 10K rules and still fail (R_K could be 100K)
- Doesn't address compositional structure
- Fitness function still weak

**Estimated success:** 20-40% accuracy (better than 0%, not human-level)

### Strategy 2: **Curriculum Learning** (Gradual Complexity)

**Approach:** Start simple, gradually increase difficulty

**Phase 1: Two-word sentences** (1 week)
```
"cats sit"
"dogs run"  
"birds fly"
→ Train until 80%+ accuracy
```

**Phase 2: Three-word with determiners** (1 week)
```
"the cat sits"
"a dog runs"
→ Train until 70%+ accuracy
```

**Phase 3: Add modifiers** (2 weeks)
```
"the big cat sits"
"a small dog quickly runs"
```

**Phase 4: Nested structures** (4 weeks)
```
"the cat that sits runs"
```

**Pros:**
- Matches human language acquisition
- Rules bootstrap from simple to complex
- Clear progress milestones

**Cons:**
- Slower than end-to-end training
- Need to design curriculum carefully

**Estimated success:** 60-80% on target complexity

### Strategy 3: **Hierarchical/Compositional Rules** (Structural)

**Approach:** Rules output intermediate representations, compose recursively

```python
class ComposableRule:
    # Instead of: pattern → logic
    # Do:         pattern → intermediate_rep
    
    def parse(self, tokens):
        # Match pattern
        if self.matches(tokens):
            # Output typed structure
            return ParseNode(
                type=self.output_type,  # NP, VP, S, etc.
                children=[...],
                features={...}
            )
```

**Example:**
```
Rule 1: [DET, NOUN] → NP(det=$1, noun=$2)
Rule 2: [VERB, NP]  → VP(verb=$1, object=$2)
Rule 3: [NP, VP]    → S(subject=$1, predicate=$2)

Parse "the cat sits":
  "the cat" → NP(det=the, noun=cat)      [Rule 1]
  "sits" → VP(verb=sits)                  
  NP + VP → S(subject=NP(...), pred=VP(...))  [Rule 3]
```

**Pros:**
- Captures linguistic structure naturally
- Rules compose (generalize better)
- Matches formal grammar theory

**Cons:**
- Major rewrite of rule representation
- More complex fitness function
- Harder to evolve (larger search space)

**Estimated success:** 70-90% with sufficient rules

### Strategy 4: **Hybrid Neural-Symbolic** (Best of Both)

**Approach:** Neural for embeddings/similarity, Symbolic for structure

```python
# Neural component (pre-trained or frozen)
word_embeddings = BERTModel()  # or train small LSTM

# Symbolic rules operate on embeddings
class HybridRule:
    pattern_embeddings = [...]  # Learned continuous patterns
    
    def matches(self, sentence_embedding):
        similarity = cosine(sentence_embedding, self.pattern_embeddings)
        return similarity > threshold
    
    def parse(self, tokens):
        return self.logic_template.fill(tokens)
```

**Pros:**
- Neural handles similarity/generalization
- Symbolic handles structure/reasoning
- Can initialize with pre-trained LLM embeddings

**Cons:**
- Two systems to train/coordinate
- More moving parts

**Estimated success:** 80-95% (state-of-art territory)

### Strategy 5: **Ground in Simpler Task** (Indirect Learning)

**Approach:** Learn logic rules from task success, not parsing accuracy

**Example: Question Answering**
```
Story: "The cat sat. The dog ran."

Q: "What did the cat do?"
A: "sat"

Fitness = answer_correct
  ↓ (requires)
Parse that extracts: cat→sat relation
```

**Pros:**
- Clear objective (task accuracy)
- Avoids circular parse/generate problem
- Aligns with human learning (meaning through use)

**Cons:**
- Need task dataset (bAbI, SQuAD)
- Indirect signal (slower learning)

**Estimated success:** 60-80% (proven by bAbI results)

---

## Recommended Path Forward

**Short-term (1-2 weeks):**
1. ✅ Implement weighted voting for NL rules (from TTT success)
2. ✅ Scale to 1000-5000 rules (test R_K hypothesis)
3. ✅ Curriculum learning: Start with 2-word sentences
4. ✅ Better fitness: Use incremental rewards (partial credit)

**Medium-term (1 month):**
5. Implement compositional rules (hierarchical structure)
6. Add context features (previous words, discourse state)
7. Test on bAbI tasks (clear objective)

**Long-term (3 months):**
8. Hybrid neural-symbolic architecture
9. Pre-train embeddings or use frozen LLM
10. Scale to full TinyStories complexity

---

## Key Insights

1. **R << R_K is the primary bottleneck** - Need 10-100x more rules
2. **Compositional structure missing** - Flat patterns insufficient
3. **Weighted voting from TTT applies** - Ensemble effect crucial
4. **Curriculum learning likely essential** - Can't jump to full complexity
5. **Task-grounded learning better than parse accuracy** - Need clear objectives

---

## Comparison to Human Learning

**Humans:**
- ~2 years to parse simple sentences
- ~5 years for complex nested structures  
- ~100 billion synapses (parameters)
- Constant feedback from environment

**Our system:**
- 30 epochs on 1000 sentences
- 100 rules × ~50 parameters = 5K parameters
- No environmental feedback
- **Expectation: Should fail on full NL**

**Implication:** We're not failing, we're under-resourced by ~10^6 x!

---

## Bottom Line

**The failure is expected and informative:**

✅ TicTacToe success validates GA algorithm  
✅ NL failure confirms R << R_K hypothesis  
✅ Identifies clear paths to improvement  

**AGI is non-trivial precisely because:**
- High Kolmogorov complexity (many patterns)
- Compositional structure (recursive rules)
- Context-dependence (non-local features)
- Ambiguous objectives (what is "understanding"?)

**We can improve by:**
1. Scaling capacity (more rules)
2. Adding structure (compositional)
3. Curriculum learning (simple → complex)
4. Grounding in tasks (clear objectives)

---

