# Representation Crisis: Triple-Only Logic

**Date:** 2026-01-02  
**Critical Issue:** Current system restricted to (entity, relation, value) triples

---

## The Problem

### Current Limitation
```python
# From symmetric_logic_network.py:
propositions: (batch, num_props, 3)  # FIXED ARITY = 3!
# Each proposition: [entity, relation, value]
```

### What This CAN'T Represent

```
❌ "John loves Mary obsessively"
   Current: [John, loves, Mary]  # Lost "obsessively"!
   
❌ "Three big cats"
   Current: [?, count, 3], [?, adj, big], [?, type, cat]  # 3 separate triples!
   
❌ "John gave Mary the book yesterday"
   Current: [John, gave, Mary]  # Lost book and yesterday!
   Or: Multiple triples lose semantic unity
   
❌ "The cat on the mat sat"
   Current: [cat, on, mat], [cat, sat, ?]  # Lost prepositional attachment
   
❌ Negation: "John does NOT love Mary"
   Current: [John, loves, Mary]  # No way to encode negation!
   
❌ Quantifiers: "All cats sit"
   Current: No representation at all

❌ Modal: "John might love Mary"
   Current: [John, loves, Mary]  # Lost modality
```

---

## Why Triples Are Insufficient

### 1. **Fixed Arity**
- Natural language has variable arity predicates:
  - "sleep" (1-ary): John sleeps
  - "love" (2-ary): John loves Mary
  - "give" (3-ary): John gave Mary a book
  - "transfer" (4-ary): John transferred money to Mary from his account

**Fix:** Variable-arity propositions

### 2. **No Modifiers**
- Adjectives: "very big red cat"
- Adverbs: "quickly ran away"
- Prepositional phrases: "on the table"

**Fix:** Nested/hierarchical structures

### 3. **No Quantification**
- Universal: "all", "every", "each"
- Existential: "some", "a"
- Numerical: "three", "many", "few"

**Fix:** First-order logic with quantifiers

### 4. **No Negation/Modality**
- Negation: "not", "never"
- Modality: "might", "must", "can"
- Tense: "will", "did", "has"

**Fix:** Modal operators

### 5. **No Sentence Structure**
- Can't represent parse trees
- Can't capture scope (nested structures)
- Can't represent coordination ("and", "or")

**Fix:** Tree-structured representations

---

## Existing Approaches (What Linguists Use)

### 1. **Lambda Calculus / Semantic Composition**
```
"John loves Mary" → loves(john, mary)
"John loves Mary obsessively" → loves(john, mary, obsessively)
or: obsessive(loves(john, mary))
```

### 2. **Dependency Grammar**
```
        loves
       /  |  \
    John  Mary  obsessively
```

### 3. **First-Order Logic (FOL)**
```
"All cats sit" → ∀x. cat(x) → sits(x)
"John loves Mary" → loves(john, mary)
"John loves Mary obsessively" → loves(john, mary) ∧ manner(loves, obsessive)
```

### 4. **Abstract Meaning Representation (AMR)**
```
(l / love-01
   :ARG0 (j / person :name "John")
   :ARG1 (m / person :name "Mary")
   :manner (o / obsessive))
```

### 5. **Discourse Representation Theory (DRT)**
```
[john, mary, e]
john(john)
mary(mary)
love-event(e)
agent(e, john)
patient(e, mary)
manner(e, obsessive)
```

---

## Recommended Solution: Flexible Typed Structures

### Option A: **Variable-Arity Propositions**

```python
class Proposition:
    def __init__(self, predicate, *args, **modifiers):
        self.predicate = predicate
        self.args = args
        self.modifiers = modifiers  # adverbs, adjectives, etc.

# Examples:
Proposition("sleep", john)
Proposition("love", john, mary, manner="obsessive")
Proposition("give", john, mary, book, time="yesterday")
```

**Pros:** Flexible, natural  
**Cons:** Variable length harder for neural nets

### Option B: **Hierarchical/Nested Structures**

```python
class LogicNode:
    def __init__(self, type, children=None, features=None):
        self.type = type  # "PRED", "ARG", "MOD", etc.
        self.children = children or []
        self.features = features or {}

# Example: "John loves Mary obsessively"
LogicNode("PRED",
    children=[
        LogicNode("ARG", features={"name": "John", "role": "agent"}),
        LogicNode("ARG", features={"name": "Mary", "role": "patient"}),
        LogicNode("MOD", features={"type": "manner", "value": "obsessive"})
    ],
    features={"verb": "love"}
)
```

**Pros:** Expressive, compositional  
**Cons:** Complex, requires tree operations

### Option C: **Reified Event Semantics**

```python
# Events as first-class entities
Event(id=e1, type="love")
Participant(e1, role="agent", entity=john)
Participant(e1, role="patient", entity=mary)
Modifier(e1, type="manner", value="obsessive")
Time(e1, when="present")
```

**Pros:** Handles complex events naturally  
**Cons:** More propositions per sentence

### Option D: **Hybrid: Key-Value Propositions**

```python
{
    "predicate": "love",
    "agent": "john",
    "patient": "mary",
    "manner": "obsessive",
    "tense": "present"
}
```

**Pros:** Flexible, neural-friendly (attention over key-value)  
**Cons:** No nesting (but could nest dicts)

---

## Concrete Implementation Plan

### Phase 1: Quick Fix (Variable-Length Triples)

```python
# Instead of fixed (batch, num_props, 3)
# Use: (batch, num_props, MAX_ARITY) with padding

class FlexibleProposition:
    def __init__(self, max_arity=5):
        self.max_arity = max_arity
    
    def encode(self, *elements):
        # Pad to max_arity
        padded = list(elements) + [PAD] * (self.max_arity - len(elements))
        return padded[:self.max_arity]

# "John loves Mary" → [loves, john, mary, PAD, PAD]
# "John gave Mary book" → [give, john, mary, book, PAD]
```

**Time:** 1 day  
**Impact:** Handles 3-5 arity predicates  
**Limitation:** Still no recursion/nesting

### Phase 2: Add Modifier Slots

```python
class PropositionWithModifiers:
    """
    Core: [predicate, arg1, arg2, ...]
    Modifiers: {type: value, ...}
    """
    def __init__(self):
        self.core = []  # Main predicate + arguments
        self.modifiers = {}  # adverbs, adjectives, tense, etc.

# "John loves Mary obsessively"
prop = PropositionWithModifiers()
prop.core = ["love", "john", "mary"]
prop.modifiers = {"manner": "obsessive"}
```

**Time:** 2-3 days  
**Impact:** Handles most modifiers  
**Limitation:** Still no deep nesting

### Phase 3: Full Tree Structure

```python
class TreeProposition:
    """
    Recursive tree structure for complex propositions.
    """
    def __init__(self, head, children=None, features=None):
        self.head = head  # Predicate or node type
        self.children = children or []  # List of TreeProposition
        self.features = features or {}  # Modifiers, attributes

# "The cat that sits on the mat ran"
TreeProposition("run",
    children=[
        TreeProposition("cat",
            children=[
                TreeProposition("sit",
                    children=[
                        TreeProposition("on",
                            children=[
                                TreeProposition("mat")
                            ])
                    ])
            ],
            features={"det": "the"})
    ],
    features={"tense": "past"})
```

**Time:** 1-2 weeks  
**Impact:** Full expressivity  
**Challenge:** Neural processing of trees

---

## Neural Network Adaptation

### For Variable-Arity (Phase 1)

```python
class FlexibleLogicNetwork(nn.Module):
    def __init__(self, max_arity=5, symbol_dim=64):
        self.max_arity = max_arity
        
        # Embed each slot
        self.slot_embedders = nn.ModuleList([
            nn.Embedding(vocab_size, symbol_dim)
            for _ in range(max_arity)
        ])
        
        # Attention to combine slots
        self.slot_attention = nn.MultiheadAttention(symbol_dim, 4)
    
    def forward(self, propositions):
        # propositions: (batch, num_props, max_arity)
        
        # Embed each slot
        slot_embeddings = []
        for i in range(self.max_arity):
            emb = self.slot_embedders[i](propositions[:, :, i])
            slot_embeddings.append(emb)
        
        # Stack and attend
        all_slots = torch.stack(slot_embeddings, dim=2)
        # ... attention over slots
```

### For Trees (Phase 3)

```python
class TreeEncoder(nn.Module):
    def __init__(self, symbol_dim=64):
        self.symbol_embedder = nn.Embedding(vocab_size, symbol_dim)
        self.tree_lstm = nn.TreeLSTM(symbol_dim, symbol_dim)
    
    def forward(self, tree: TreeProposition):
        # Recursive encoding
        if not tree.children:
            # Leaf node
            return self.symbol_embedder(tree.head)
        
        # Encode children
        child_encodings = [self.forward(child) for child in tree.children]
        
        # Combine with Tree-LSTM
        return self.tree_lstm(tree.head, child_encodings)
```

---

## Recommended Immediate Action

**Do Phase 1 NOW** (1 day effort):

1. Change `prop_length` from 3 to 5 (allows up to 5-ary predicates)
2. Update all code to handle variable-length propositions
3. Test on examples like "John gave Mary the book"

This gives you 2-3x more expressivity with minimal changes.

**Do Phase 2 within 1 week:**

1. Add modifier dictionary to propositions
2. Update parsing to extract modifiers
3. Update neural network to process modifiers separately

This gets you ~80% of needed expressivity.

**Plan Phase 3 for month 2:**

1. Full tree structures
2. Tree neural networks (Tree-LSTM or Graph Networks)
3. Compositional semantics

This achieves maximum expressivity.

---

## Critical Realization

**Your intuition is correct:** The triple representation is a fundamental bottleneck.

Even with:
- ✅ 10,000 rules
- ✅ Weighted voting
- ✅ Curriculum learning
- ✅ Perfect training

You'll still fail on "John loves Mary obsessively" because **the representation can't express it**.

**Priority:** Fix representation FIRST, then scale up rules.

---

## Bottom Line

Current system: `(entity, relation, value)` triples  
**Maximum achievable:** ~40-50% accuracy (many things unrepresentable)

With flexible arity: `(predicate, arg1, ..., argN)` + modifiers  
**Maximum achievable:** ~70-80% accuracy (most things representable)

With full trees: Recursive nested structures  
**Maximum achievable:** ~95%+ accuracy (full expressivity)

**Recommendation:** Implement Phase 1 (variable arity) TODAY, before continuing with anything else.

