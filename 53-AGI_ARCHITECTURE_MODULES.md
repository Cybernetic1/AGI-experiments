# AGI Architecture: Core Modules Reference

**Date:** 2026-01-03  
**Status:** Architecture design after successful semantic-AR proof-of-concept

## Overview

Our AGI system combines symbolic logic rules with differentiable neural networks, using semantic autoregressive (AR) training on natural language. The key innovation is **reflection**: converting NL text → logical propositions → logical rules, enabling rapid knowledge acquisition.

---

## Core Modules

### 1. **Davidsonian NL Parser** (`davidsonian_extraction.py`)
**Purpose:** Convert natural language sentences to logical form (event-based representations)

**Key Features:**
- POS tagging via spaCy
- Event reification (verbs → event entities)
- Argument structure extraction (agent, patient, instrument, etc.)
- Modifier handling (adjectives, adverbs → properties)
- **Entity resolution** (pronouns, coreference chains)

**Input:** `"The girl found a toy."`  
**Output:** 
```python
[
    ('event', 'e1', 'find'),
    ('agent', 'e1', 'girl1'),
    ('patient', 'e1', 'toy1'),
    ('definite', 'girl1', True),
    ('indefinite', 'toy1', True)
]
```

**Status:** ✅ Basic implementation complete  
**TODO:** 
- Add entity registry for cross-sentence resolution
- Handle complex quantifiers (all, some, none)
- Parse nested clauses

---

### 2. **Entity Registry** (`entity_registry.py`)
**Purpose:** Track entities across discourse, resolve pronouns and references

**Key Features:**
- Unique entity IDs with discourse context
- Pronoun resolution (he/she/it/they)
- Coreference chains ("the girl" = "she" = entity_42)
- Entity properties and relationships

**Data Structure:**
```python
{
    'entity_42': {
        'type': 'person',
        'gender': 'female',
        'mentions': ['the girl', 'she'],
        'properties': [('young', True), ('happy', True)],
        'last_mentioned': sentence_index
    }
}
```

**Status:** ✅ Exists but needs integration  
**TODO:** Integrate with Davidsonian parser for cross-sentence tracking

---

### 3. **Knowledge Base / Rule Engine**
**Purpose:** Store and apply symbolic logic rules efficiently

**Options Evaluated:**
- **Experta** (CLIPS-like, requires Python 3.8)
- **Pyke** (Prolog-like, unmaintained)
- **Custom RETE implementation** (scalable, full control)

**Key Features Needed:**
- Pattern matching with variables
- Forward chaining (facts → infer new facts)
- Backward chaining (query → find supporting facts)
- RETE algorithm for efficient rule matching at scale
- **Gradient transparency** (rules don't block backprop)

**Rule Format:**
```python
# Symbolic rule (hard-coded or learned)
IF   pattern: [('agent', ?e, ?x), ('action', ?e, 'give')]
THEN conclude: [('generous', ?x, True)]
```

**Status:** 🟡 Evaluated options, needs implementation decision  
**TODO:** 
- Choose between custom RETE vs. library
- Implement gradient-transparent rule execution
- Add rule confidence scores (soft rules)

---

### 4. **Differentiable Logic Network** (DLN)
**Purpose:** Learn soft logic rules via gradient descent

**Architecture:**
- Proposition embeddings (atomic facts → vectors)
- Attention-based rule matching
- Soft unification (fuzzy variable binding)
- Confidence scores (0-1) instead of binary True/False

**Key Innovation:** Hybrid execution
- **Symbolic rules:** Fire deterministically, transparent to gradients
- **Soft rules:** Learned weights, fully differentiable

**Existing Code:**
- `neural_logic_core.py` - Core logic operations
- `symmetric_logic_network.py` - Symmetric architecture
- `hierarchical_logic_network.py` - Hierarchical reasoning

**Status:** ✅ Multiple implementations exist  
**TODO:** Integrate with semantic-AR training loop

---

### 5. **Semantic Autoregressive Trainer** (`train_semantic_ar.py`)
**Purpose:** Train the system to predict next sentence in logical space

**Training Loop:**
1. Read NL sentence from TinyStories
2. Parse to logical propositions (Davidsonian)
3. Encode propositions to vector (DLN)
4. Predict next sentence's logical form
5. Parse actual next sentence
6. Compute **graph edit distance loss**
7. Backprop through DLN (symbolic rules are transparent)

**Loss Function:**
```
Loss = GraphEditDistance(predicted_props, actual_props)
     + EntityBindingLoss(predicted_entities, actual_entities)
```

**Status:** ✅ Proof-of-concept with MLP working  
**TODO:** 
- Replace MLP with full DLN
- Implement graph edit distance loss
- Add entity binding loss component

---

### 6. **Reflection Module** (Planned)
**Purpose:** Convert logical propositions back into rules (ILP-style learning)

**Key Idea:** 
- Observe: `[('agent', e1, 'john'), ('action', e1, 'run')]`
- Generalize: `IF [('agent', ?e, ?x)] THEN [('animate', ?x, True)]`
- Store rule in KB for future use

**Methods:**
- Template-based generalization (replace constants with variables)
- Anti-unification (find common structure)
- Rule confidence from frequency
- **Differentiable rule generation** (neural ILP)

**Status:** 🔴 Not implemented  
**TODO:** 
- Design rule extraction heuristics
- Implement variable generalization
- Test on reading NL text about NL parsing rules (meta-learning!)

---

### 7. **Meta-Rules for Domain Transfer** (Future)
**Purpose:** Apply learned rules to new domains via higher-order rules

**Example:**
- Domain 1: Learn Davidsonian parsing rules for English
- Domain 2: Apply similar event-based parsing to actions in a game
- Meta-rule: "Verbs in language correspond to actions in planning"

**Status:** 🔴 Conceptual stage  
**TODO:** Explore after core system stabilizes

---

## Data Flow Architecture

```
NL Text (TinyStories)
    ↓
[Davidsonian Parser] → Logical Propositions (events, entities, properties)
    ↓
[Entity Registry] → Resolve references across sentences
    ↓
[Knowledge Base] → Apply symbolic rules (forward chaining)
    ↓                     ↘
[DLN Encoder]          [Symbolic Rules]
    ↓                     ↓ (transparent)
Proposition Embeddings    New Inferred Facts
    ↓                     ↓
[Predictor Network] ← Soft Rule Matching
    ↓
Predicted Next Logical Form
    ↓
[Loss: Graph Edit Distance + Entity Binding]
    ↓
Backprop (only through DLN, not symbolic rules)
```

---

## Implementation Priority

### Phase 1: Core Semantic-AR (Current Focus)
1. ✅ Davidsonian parser (basic)
2. ✅ Semantic-AR training loop (proof-of-concept)
3. 🟡 Replace MLP with DLN
4. 🟡 Implement graph edit distance loss
5. 🟡 Integrate entity registry

### Phase 2: Rule Engine Integration
6. 🔴 Choose/implement rule engine (RETE)
7. 🔴 Inject Davidsonian meta-rules
8. 🔴 Gradient-transparent rule execution
9. 🔴 Train on larger TinyStories dataset

### Phase 3: Reflection & Meta-Learning
10. 🔴 Implement reflection (props → rules)
11. 🔴 Test: Read NL text, extract rules, apply immediately
12. 🔴 Meta-rules for domain transfer

---

## Key Architectural Decisions

### 1. **Hybrid Symbolic-Neural**
- **Symbolic rules:** Fast, interpretable, gradient-transparent
- **Soft rules:** Learnable, handle ambiguity, fully differentiable

### 2. **Event-Based Semantics (Davidsonian)**
- Universal representation for NL
- Handles modifiers, quantifiers, complex arguments naturally
- Supports entity tracking across discourse

### 3. **Graph Edit Distance Loss**
- Compare logical forms as graphs (nodes = entities, edges = relations)
- More principled than token-level cross-entropy
- Requires efficient approximation (Hungarian algorithm)

### 4. **Reflection as Core Innovation**
- NL → Logic → Rules enables rapid knowledge acquisition
- Bypasses slow gradient descent for symbolic knowledge
- Key advantage over LLMs (they can't do this easily)

---

## Success Metrics

### Short-term (Weeks)
- Parse 80%+ of TinyStories sentences correctly
- Entity resolution accuracy >70%
- Semantic-AR loss decreasing consistently

### Medium-term (Months)
- Read text about NL parsing → immediately improve parsing
- Generalize learned rules to new sentence structures
- Outperform pure neural baseline on same compute budget

### Long-term (Research Goal)
- Demonstrate sample-efficient learning via reflection
- Transfer learned knowledge across domains
- Show interpretable rule extraction from experience

---

## Open Questions

1. **Graph loss efficiency:** Can we approximate graph edit distance in O(n²) reliably?
2. **Rule extraction:** What heuristics work best for generalizing propositions to rules?
3. **RETE implementation:** Custom vs. library? Trade-offs for our use case?
4. **Gradient flow:** How to make symbolic rules truly transparent without information loss?
5. **Scalability:** How does RETE scale to 10k, 100k, 1M rules?

---

## Files Reference

- `davidsonian_extraction.py` - NL → logical form
- `entity_registry.py` - Cross-sentence entity tracking
- `train_semantic_ar.py` - Main training loop
- `neural_logic_core.py` - Differentiable logic operations
- `symmetric_logic_network.py` - DLN architecture
- `knowledge_base.py` - Symbolic rule storage (needs RETE)
- `convergence_system.py` - Proof-of-concept (created today)

---

**Last Updated:** 2026-01-03  
**Next Review:** After implementing graph edit distance loss
