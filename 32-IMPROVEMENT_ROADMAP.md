# NL Parsing Improvement Roadmap

**Priority-ordered actionable steps**

---

## 🚀 Quick Wins (This Week)

### 1. Add Weighted Voting to NL Rules ⏰ 2 hours

**File:** `genetic_logic_rules.py`

```python
def evaluate_with_voting(rules, dataset):
    """Use weighted voting instead of first-match"""
    votes = defaultdict(float)
    for rule in rules:
        if rule.matches(text):
            votes[rule.logic_template] += rule.fitness
    best_parse = max(votes.items(), key=lambda x: x[1])[0]
```

**Expected:** +20-30% accuracy improvement (from TTT results)

---

### 2. Scale to 1000+ Rules ⏰ 1 hour

**Change in `test_ga.py`:**
```python
best_rules = genetic_algorithm(
    dataset=train_set,
    population_size=1000,  # Was: 100
    generations=100,       # Was: 50
    mutation_rate=0.2,
    elite_size=50          # Was: 10
)
```

**Expected:** Test R_K hypothesis, reach 10-20% accuracy

---

### 3. Incremental Fitness Function ⏰ 3 hours

**Current problem:** All-or-nothing (0.0 fitness if wrong)

**New approach:**
```python
def incremental_fitness(rule, dataset):
    score = 0
    for text, target_logic in dataset:
        if rule.matches(text):
            parsed = rule.parse(text)
            
            # Partial credit breakdown
            if parsed:
                # 1. Pattern match bonus (found something)
                score += 0.2
                
                # 2. Slot overlap (how many slots correct?)
                overlap = len(set(parsed.keys()) & set(target_logic.keys()))
                score += 0.3 * (overlap / len(target_logic))
                
                # 3. Value similarity (fuzzy match on values)
                for key in parsed:
                    if key in target_logic:
                        if parsed[key] == target_logic[key]:
                            score += 0.5 / len(target_logic)
                        elif similar(parsed[key], target_logic[key]):
                            score += 0.25 / len(target_logic)
    
    return score / len(dataset)
```

**Expected:** Provide learning gradient, enable convergence

---

## 📚 Curriculum Learning (Week 2)

### 4. Two-Word Corpus ⏰ 4 hours

**Create:** `simple_corpus.txt`
```
cats sit
dogs run
birds fly
fish swim
snakes crawl
...
```

**Generate 1000 examples** of Subject-Verb patterns

**Train until:** 80%+ accuracy

**Why it helps:** Bootstraps basic SV pattern recognition

---

### 5. Three-Word with Determiners ⏰ 4 hours

**Expand corpus:**
```
the cat sits
a dog runs
the bird flies
...
```

**Inject learned rules from Phase 1** as starting population

**Train until:** 70%+ accuracy

---

### 6. Gradual Complexity Ramp ⏰ 1 week

```
Week 1: Subject-Verb (80%+ target)
Week 2: Det-Noun-Verb (70%+ target)  
Week 3: Add adjectives (60%+ target)
Week 4: Simple prepositional phrases (50%+ target)
```

**Each phase:** Previous rules seed next phase population

---

## 🏗️ Structural Improvements (Weeks 3-4)

### 7. Compositional Rules ⏰ 1 week

**New rule type:**
```python
class ComposableRule(SymbolicRule):
    def __init__(self):
        self.input_types = ['TOKEN', 'NP', 'VP']  # What it accepts
        self.output_type = 'S'                     # What it produces
        self.pattern = ...
        self.template = ...
    
    def parse(self, items):
        # Match pattern across typed elements
        if self.matches(items):
            return ParseNode(
                type=self.output_type,
                children=self.extract_children(items),
                features=self.extract_features(items)
            )
```

**Rules compose:**
```
[DET, NOUN] → NP
[NP, VERB]  → S
[ADJ, NOUN] → NOUN
```

**Why it helps:** Captures recursive structure of language

---

### 8. Context Features ⏰ 3 days

**Add to rule matching:**
```python
class ContextualRule(SymbolicRule):
    def matches(self, text, context):
        # Check local pattern
        if not super().matches(text):
            return False
        
        # Check context constraints
        if self.requires_previous_np:
            if 'NP' not in context.previous_constituents:
                return False
        
        return True
```

**Context includes:**
- Previous parsed constituents
- Discourse entities mentioned
- Sentence position (beginning/middle/end)

---

## 🎯 Task-Grounded Learning (Week 5+)

### 9. bAbI Question Answering ⏰ 1 week

**Use existing dataset:** Facebook's bAbI tasks

**Example:**
```
Story: John went to the kitchen. Mary went to the garden.
Q: Where is John?
A: kitchen

Fitness = answer_correctness
```

**Why it helps:**
- Clear objective (can't fake understanding)
- Forces semantic parsing (must extract meaning)
- Incremental difficulty (20 task types)

**Implementation:**
```python
def qa_fitness(rules, qa_dataset):
    correct = 0
    for story, question, answer in qa_dataset:
        # Parse story with rules
        story_logic = parse_with_rules(rules, story)
        
        # Parse question
        query_logic = parse_with_rules(rules, question)
        
        # Answer from logic
        predicted = answer_query(story_logic, query_logic)
        
        if predicted == answer:
            correct += 1
    
    return correct / len(qa_dataset)
```

---

## 🧠 Neural-Symbolic Hybrid (Month 2+)

### 10. Add Pre-trained Embeddings ⏰ 3 days

**Use existing LLM:** sentence-transformers, or small BERT

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

class EmbeddingRule(SymbolicRule):
    def __init__(self):
        super().__init__()
        self.pattern_embedding = None  # Learned
    
    def matches(self, text):
        text_emb = model.encode(text)
        similarity = cosine_sim(text_emb, self.pattern_embedding)
        return similarity > self.threshold
```

**Why it helps:** Generalizes beyond exact POS patterns

---

### 11. Differentiable Rule Selection ⏰ 1 week

**Instead of hard voting, soft attention:**

```python
def soft_rule_application(rules, text):
    # Each rule produces parse + confidence
    parses = []
    confidences = []
    
    for rule in rules:
        if rule.matches(text):
            parse = rule.parse(text)
            conf = rule.fitness * rule.match_strength(text)
            parses.append(parse)
            confidences.append(conf)
    
    # Soft combination (differentiable!)
    weights = softmax(confidences)
    combined_parse = weighted_average(parses, weights)
    
    return combined_parse
```

**Why it helps:** Can backprop through rule selection

---

## 📊 Evaluation Improvements

### 12. Multi-metric Evaluation ⏰ 2 hours

**Current:** 0% accuracy (binary)

**New:**
```python
def evaluate_detailed(rules, test_set):
    metrics = {
        'exact_match': 0,
        'slot_precision': [],
        'slot_recall': [],
        'value_accuracy': [],
        'coverage': 0,  # % examples matched by any rule
    }
    
    for text, gold_logic in test_set:
        pred_logic = apply_rules(rules, text)
        
        # Exact match
        if pred_logic == gold_logic:
            metrics['exact_match'] += 1
        
        # Slot-level metrics
        gold_slots = set(gold_logic.keys())
        pred_slots = set(pred_logic.keys())
        
        precision = len(gold_slots & pred_slots) / len(pred_slots)
        recall = len(gold_slots & pred_slots) / len(gold_slots)
        
        metrics['slot_precision'].append(precision)
        metrics['slot_recall'].append(recall)
        
        # ... etc
    
    return metrics
```

**Why it helps:** Shows where learning is happening

---

## Priority Order

**If you have limited time, do in this order:**

1. ✅ **Weighted voting** (2h, +20-30% guaranteed)
2. ✅ **Incremental fitness** (3h, enables learning)
3. ✅ **Scale to 1000 rules** (1h, test hypothesis)
4. 📚 **Two-word curriculum** (4h, bootstrap learning)
5. 🎯 **bAbI tasks** (1 week, clear objective)
6. 🏗️ **Compositional rules** (1 week, structural fix)

**Skip for now:**
- Neural-symbolic hybrid (complex, can add later)
- Full TinyStories (too hard as first target)

---

## Expected Outcomes

**After Quick Wins (Week 1):**
- 10-20% accuracy on simple sentences
- Clear learning signal in fitness curves
- Validation that more rules help

**After Curriculum (Week 2-3):**
- 80% on two-word sentences
- 60% on three-word sentences
- Understanding of capacity requirements

**After Structural (Week 4-5):**
- 70% on nested structures
- Compositional generalization
- Human-interpretable rules

**After Task-Grounding (Week 6-8):**
- Solve bAbI tasks 1-5 (simple QA)
- Demonstrate semantic understanding
- Path to AGI more clear

---

## Success Criteria

**Minimum viable (declare success):**
- 50%+ accuracy on simple sentence parsing
- Rules compose to handle unseen combinations
- Beats neural baseline on low-data regime

**Strong result (publishable):**
- 70%+ accuracy on moderate complexity
- Solves 10+ bAbI tasks
- Interpretable, human-readable rules

**Breakthrough (AGI-relevant):**
- 90%+ accuracy on TinyStories subset
- Compositional generalization proven
- Scales to novel domains

---

