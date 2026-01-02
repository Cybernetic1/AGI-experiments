# Atomic Propositions for Tree Structures

**Date:** 2026-01-02  
**Key Insight:** Can we represent tree structures using atomic propositions?

---

## Your Insight: Trees → Atomic Propositions

**Question:** Can we encode tree structures as collections of atomic propositions?

**Answer:** YES! This is exactly what linguists and logic systems do.

---

## The Tree We Want to Represent

```
"John loves Mary obsessively"

Tree structure:
        love-event (e1)
           /    |    \
       agent  patient  manner
         |      |        |
       John   Mary   obsessive
```

---

## Method 1: Neo-Davidsonian Event Semantics ⭐

**Most principled linguistic approach**

```
Atomic propositions (all triples!):
1. [e1, type, love]           # There's a loving event
2. [e1, agent, john]          # Agent of event is John
3. [e1, patient, mary]        # Patient is Mary
4. [e1, manner, obsessive]    # Manner is obsessive
5. [e1, tense, present]       # Time is present
```

**Key idea:** Events are first-class entities!

**Properties:**
✅ All propositions are triples
✅ Can add unlimited modifiers (each is one triple)
✅ Can query: "How does John love?" → Search [e1, manner, ?]
✅ Compositional: Can combine events

---

## Method 2: Node-Relationship Encoding

**Explicit tree structure**

```
"John loves Mary obsessively"

Atomic propositions:
1. [n1, type, person]         # Node 1 is John
2. [n1, name, john]
3. [n2, type, person]         # Node 2 is Mary
4. [n2, name, mary]
5. [n3, type, event]          # Node 3 is the love event
6. [n3, verb, love]
7. [n3, agent, n1]            # Link: n3's agent is n1
8. [n3, patient, n2]          # Link: n3's patient is n2
9. [n4, type, modifier]       # Node 4 is modifier
10. [n4, value, obsessive]
11. [n3, manner, n4]          # Link: n3's manner is n4
```

**Properties:**
✅ All propositions are triples
✅ Explicit nodes and edges
✅ Can represent any tree
❌ More verbose (11 triples vs 5)

---

## Method 3: Dependency Triple Encoding

**Like dependency parsing**

```
"John loves Mary obsessively"

Atomic propositions (dependency relations):
1. [love, nsubj, john]        # Nominal subject
2. [love, dobj, mary]         # Direct object
3. [love, advmod, obsessive]  # Adverb modifier
4. [john, pos, PROPN]         # POS tag
5. [mary, pos, PROPN]
6. [love, pos, VERB]
7. [obsessive, pos, ADV]
```

**Properties:**
✅ All propositions are triples
✅ Standard linguistic representation
✅ Can use spaCy directly!
❌ Ties you to specific dependency scheme

---

## Method 4: Path-Based Encoding

**Encode tree paths**

```
Tree:
        root(love)
        /    |    \
     arg0  arg1  manner
      |     |      |
    john  mary  obsessive

Propositions (paths from root):
1. [root, isa, love]
2. [root.arg0, isa, john]
3. [root.arg1, isa, mary]
4. [root.manner, isa, obsessive]
```

**Properties:**
✅ All propositions are triples
✅ Natural tree structure
❌ Paths can get long for deep trees
❌ Less standard

---

## Comparison: Which to Use?

| Method | Triples Only? | Composable? | Standard? | Best For |
|--------|---------------|-------------|-----------|----------|
| **Neo-Davidsonian** | ✅ Yes | ✅ Yes | ✅ Linguistics | General NL |
| Node-Relationship | ✅ Yes | ⚠️ Verbose | ⚠️ Custom | Complex graphs |
| Dependency | ✅ Yes | ⚠️ Limited | ✅ spaCy | Parse trees |
| Path-Based | ✅ Yes | ⚠️ Paths | ❌ No | Hierarchies |

**Recommendation: Neo-Davidsonian Event Semantics** ⭐

---

## Why Neo-Davidsonian is Best

### 1. Linguistic Foundation
- Developed by philosopher Donald Davidson
- Extended by linguists (Parsons, etc.)
- Standard in formal semantics

### 2. Natural Composition
```
"John loves Mary" →
  [e1, type, love], [e1, agent, john], [e1, patient, mary]

Add "obsessively" →
  Just add: [e1, manner, obsessive]

Add "yesterday" →
  Just add: [e1, time, yesterday]

No need to rewrite existing propositions!
```

### 3. Handles All Linguistic Phenomena

**Quantifiers:**
```
"Every boy loves some girl"

∀x. [x, type, boy] → ∃e,y. [e, type, love] ∧ [e, agent, x] ∧ 
                           [y, type, girl] ∧ [e, patient, y]
```

**Negation:**
```
"John does not love Mary"

¬∃e. [e, type, love] ∧ [e, agent, john] ∧ [e, patient, mary]
```

**Modality:**
```
"John might love Mary"

◇∃e. [e, type, love] ∧ [e, agent, john] ∧ [e, patient, mary]
```

**Relative clauses:**
```
"The cat that John loves sat"

∃e1,e2,x. [x, type, cat] ∧ [e1, type, love] ∧ [e1, agent, john] ∧ 
          [e1, patient, x] ∧ [e2, type, sit] ∧ [e2, agent, x]
```

### 4. Queryable
```
Q: "How does John love Mary?"
Query: [e, type, love], [e, agent, john], [e, patient, mary], [e, manner, ?]
Answer: obsessive

Q: "Who loves Mary?"
Query: [e, type, love], [e, patient, mary], [e, agent, ?]
Answer: john
```

---

## Implementation Strategy

### Current System (Broken)
```python
# train_symmetric.py, line 236:
propositions.append([
    entity_id,      # Subject
    relation_id,    # Verb
    value_id        # Object
])
# Problem: Can't add modifiers!
```

### Neo-Davidsonian System (Fixed)
```python
def extract_propositions_event_based(sent):
    """
    Extract propositions using event semantics.
    """
    propositions = []
    
    for token in sent:
        if token.pos_ == "VERB":
            # Create event entity
            event_id = create_event_id(token)
            
            # Event type
            propositions.append([
                event_id,
                self.vocab.get_or_create("type"),
                self.vocab.get_or_create(token.lemma_)
            ])
            
            # Agent (subject)
            for child in token.children:
                if child.dep_ in ["nsubj", "nsubjpass"]:
                    propositions.append([
                        event_id,
                        self.vocab.get_or_create("agent"),
                        self.entity_registry.get_or_create(child.text)
                    ])
                
                # Patient (object)
                elif child.dep_ in ["dobj", "pobj"]:
                    propositions.append([
                        event_id,
                        self.vocab.get_or_create("patient"),
                        self.entity_registry.get_or_create(child.text)
                    ])
                
                # Manner (adverbs)
                elif child.dep_ == "advmod":
                    propositions.append([
                        event_id,
                        self.vocab.get_or_create("manner"),
                        self.vocab.get_or_create(child.text)
                    ])
                
                # Location (prepositional phrases)
                elif child.dep_ == "prep":
                    for prep_child in child.children:
                        if prep_child.dep_ == "pobj":
                            propositions.append([
                                event_id,
                                self.vocab.get_or_create("location"),
                                self.entity_registry.get_or_create(prep_child.text)
                            ])
            
            # Tense
            propositions.append([
                event_id,
                self.vocab.get_or_create("tense"),
                self.vocab.get_or_create(get_tense(token))
            ])
    
    return propositions
```

---

## Example: Complex Sentence

**Input:** "Yesterday, John quickly gave Mary the big red book in the garden"

**Neo-Davidsonian Encoding:**
```python
e1 = create_event_id()  # give-event

propositions = [
    # Event core
    [e1, type, give],
    [e1, agent, john],
    [e1, recipient, mary],
    [e1, theme, book_entity],
    
    # Modifiers on event
    [e1, manner, quickly],
    [e1, time, yesterday],
    [e1, location, garden_entity],
    [e1, tense, past],
    
    # Properties of book
    [book_entity, type, book],
    [book_entity, size, big],
    [book_entity, color, red],
    
    # Properties of garden
    [garden_entity, type, garden]
]
```

**All triples! Each adds one piece of information!**

---

## Benefits for Your System

### 1. Universal Representation ✅
- Can represent ANY sentence structure
- No information loss
- Compositional (add modifiers freely)

### 2. Still Uses Triples ✅
- No need to change neural architecture!
- `[entity, relation, value]` format preserved
- Just use more triples per sentence

### 3. GA-Friendly ✅
```python
# Rules can match partial patterns
Rule: "If [e, type, give] and [e, agent, X] then..."
Rule: "If [e, manner, quickly] then..."

# Compositional matching
```

### 4. Query-Based Reasoning ✅
```python
# "Who gave something?"
query([?, type, give], [?, agent, X])

# "What did John do?"
query([?, agent, john], [?, type, X])
```

---

## Challenges & Solutions

### Challenge 1: More Propositions per Sentence

**Problem:** 
- Before: "John loves Mary" → 1 triple
- Now: "John loves Mary" → 3 triples

**Solution:** This is GOOD! More information = better learning signal

### Challenge 2: Need Event IDs

**Problem:** How to generate unique event IDs?

**Solution:** 
```python
# Simple: sequential numbering
event_counter = 0

def create_event_id():
    global event_counter
    event_counter += 1
    return f"e{event_counter}"

# Or: hash-based (deterministic)
def create_event_id(verb_token):
    return hash((verb_token.text, verb_token.i, id(sent)))
```

### Challenge 3: spaCy Integration

**Problem:** spaCy doesn't give you event semantics directly

**Solution:** Extract from dependency parse (shown in code above)

---

## Implementation Timeline

**Week 1: Core Event Extraction**
- [ ] Implement `extract_propositions_event_based()`
- [ ] Handle verbs → events + thematic roles
- [ ] Test on 100 sentences

**Week 2: Modifiers**
- [ ] Add manner (adverbs)
- [ ] Add time expressions
- [ ] Add location (PPs)
- [ ] Add tense/aspect

**Week 3: Entity Properties**
- [ ] Handle adjectives → entity properties
- [ ] Handle determiners → definiteness
- [ ] Handle quantifiers

**Week 4: Complex Structures**
- [ ] Relative clauses
- [ ] Coordination ("and", "or")
- [ ] Negation
- [ ] Modal verbs

---

## Key Realization

**Your intuition is EXACTLY right!**

Trees → Collections of atomic triples (Neo-Davidsonian)

This is:
1. ✅ Principled (standard linguistics)
2. ✅ Universal (represents any structure)
3. ✅ Neural-friendly (still triples!)
4. ✅ Compositional (add modifiers freely)
5. ✅ Query-based (enables reasoning)

**No need to change neural architecture** - just extract better propositions!

---

## Recommended Action

1. **Implement Neo-Davidsonian extraction** (3 days)
2. **Test on your current dataset** (1 day)
3. **Compare to old extraction** (should see immediate improvement)

**Expected result:**
- More propositions per sentence (3-10 instead of 1-2)
- Each proposition is simpler (easier to learn)
- Modifiers preserved (no information loss)
- GA can learn compositional rules

---

## Bottom Line

**You don't need to change your representation format!**

`[entity, relation, value]` triples are sufficient.

**You just need to extract them properly:**
- Old way: One triple per sentence (loses modifiers)
- **New way (Neo-Davidsonian): Multiple triples per event (preserves everything)**

This is the standard solution in formal semantics.
Your intuition led you to the right answer! 🎯

