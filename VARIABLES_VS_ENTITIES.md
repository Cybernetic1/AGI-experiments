# Variables vs Entities: Clarifying the Scoping Problem

## The Fundamental Distinction

You've identified a critical issue in the SCALING_TO_AGI.md document. Let me clarify:

### Classical Logic-Based AI View

**Variables (?X, ?Y)**:
- Scope: **Single logic rule**
- Purpose: Pattern matching, unification
- Lifetime: During rule execution only
- Example: `IF [?X, is_a, cat] AND [?X, color, black] THEN ...`
  - ?X binds to whatever matches during this rule execution
  - Once rule completes, binding is discarded

**Entities (cat_1, cat_2)**:
- Scope: **Entire scenario** (multiple rules, multiple time steps)
- Purpose: Referring to specific individuals
- Lifetime: Throughout the discourse/scenario
- Example: 
  ```
  cat_1 is black    (t=0)
  cat_2 is white    (t=1)
  cat_1 chases cat_2 (t=2)
  cat_2 runs away   (t=3)
  ```
  - cat_1 persists across all time steps
  - Different rules can reference the same cat_1

## The Confusion in SCALING_TO_AGI.md

The document conflated these two concepts! It suggested using variables (γ parameters) to track entities, but:

**Problem**: γ parameters are **per-rule**, so bindings don't persist across rules or time steps.

## The Correct Architecture

We need **BOTH** mechanisms working together:

### 1. Entity Memory (Persistent)

```python
# Global entity registry (persists across scenario)
entity_memory = {
    0: {"type": cat_embed, "color": black_embed},  # cat_1
    1: {"type": cat_embed, "color": white_embed},  # cat_2
    2: {"type": mat_embed, "color": green_embed}   # mat_1
}

# Propositions reference entity IDs
working_memory = [
    [0, "is_a", cat_embed],       # Entity 0 is a cat
    [0, "color", black_embed],    # Entity 0 is black
    [1, "is_a", cat_embed],       # Entity 1 is a cat
    [1, "color", white_embed],    # Entity 1 is white
    [0, "chases", 1],             # Entity 0 chases entity 1
]
```

### 2. Variables Within Rules (Temporary)

```python
# Rule: "If something is a cat and is black, mark it as interesting"
rule_premises = [
    [?X, "is_a", cat],    # γ[0] ≈ 0 (variable), γ[1] ≈ 1 (constant)
    [?X, "color", black]  # γ[0] ≈ 0 (same variable), γ[2] ≈ 1 (constant)
]

# During matching against working_memory:
# Step 1: Match first premise
#   - Tries [0, "is_a", cat_embed]: Matches! (entity 0 is a cat)
#   - Binds ?X → 0 (temporarily, for this rule execution)

# Step 2: Match second premise with ?X=0
#   - Tries [0, "color", black_embed]: Matches! (entity 0 is black)
#   - Confirms binding ?X → 0

# Step 3: Execute rule body/head with ?X=0
#   - Mark entity 0 as interesting

# Step 4: Clear bindings (or try next binding)
#   - ?X binding discarded
#   - Rule can match again with ?X → different entity
```

## How They Work Together

### Scenario: "Black cat chases white cat"

**Generation Process**:

```python
# t=0: Create first entity
action = "CREATE_ENTITY"
entity_id = entity_memory.create(type=cat, color=black)  # Returns 0
wm.append([0, "is_a", cat_embed])
wm.append([0, "color", black_embed])

# t=1: Create second entity
action = "CREATE_ENTITY"
entity_id = entity_memory.create(type=cat, color=white)  # Returns 1
wm.append([1, "is_a", cat_embed])
wm.append([1, "color", white_embed])

# t=2: Generate relation
# Logic rules examine WM and recognize: two cat entities exist
# Decision: CREATE_RELATION between them
wm.append([0, "chases", 1])

# Later (t=10): Reference existing entity
# Logic rules examine WM, see entity 0 already exists
# Decision: REFERENCE entity 0 (not create new)
wm.append([0, "runs", somewhere])
```

**Rule Matching** (happens at every time step):

```python
# Rule: "If X chases Y, then Y should flee from X"
rule_premises = [
    [?X, "chases", ?Y]  # γ[0,2] ≈ 0 (variables)
]

# Matches against [0, "chases", 1] in WM:
# - Binds ?X → 0, ?Y → 1
# - Activates rule body: entity 1 should flee from entity 0
# - Rule outputs: concepts indicating "flee" action for entity 1

# This binding (?X→0, ?Y→1) is TEMPORARY
# Only used during this rule execution
# But entities 0 and 1 are PERSISTENT (in entity_memory)
```

## The Key Decision: CREATE vs REFERENCE

**Critical question**: How does the system decide when to create a new entity vs. reference an existing one?

### Option A: Explicit Action

Treat it as a discrete choice:

```python
# Network outputs:
action_type = predict_action_type(concepts)  # ∈ {CREATE, REFERENCE, RELATION}

if action_type == "CREATE":
    entity_id = entity_memory.create(...)
    new_entity_id = max(entity_memory.keys()) + 1
    
elif action_type == "REFERENCE":
    # Which entity to reference?
    entity_scores = attention(query=concepts, keys=entity_embeddings)
    entity_id = argmax(entity_scores)
    
elif action_type == "RELATION":
    # Which two entities to relate?
    subj_entity = select_entity(concepts, role="subject")
    obj_entity = select_entity(concepts, role="object")
```

### Option B: Unification-Based

Let variables discover entities naturally:

```python
# Rule with variable: [?X, is_a, cat]
# Matching against WM:

for proposition in working_memory:
    if matches_pattern(proposition, [?X, is_a, cat]):
        bind(?X, proposition[0])  # Binds to existing entity
        # This REFERENCES existing entity

# If no match found:
# Some rules can CREATE entities:
# Rule: "If context mentions 'a cat', create entity"
if no_binding_found and context_suggests_new_entity:
    new_entity = entity_memory.create(...)
```

### Option C: Attention-Based Resolution

```python
# When generating next proposition:
concepts = logic_rules(wm)

# Decide subject:
# - Attention over existing entities vs "new entity" token
entity_scores = attention(query=concepts, keys=[*entities, NEW_TOKEN])

if argmax(entity_scores) == NEW_TOKEN:
    subject = entity_memory.create(...)
else:
    subject = argmax(entity_scores)  # Reference existing

# Similar for object
```

## What Needs Clarification

1. **Entity creation signal**: 
   - From language cues ("a cat" = new, "the cat" = existing)?
   - From rules (some rules create, others reference)?
   - From attention/unification failure?

2. **Variable-entity binding**:
   - Variables match against entity IDs in propositions
   - This is temporary (per-rule execution)
   - But which entity IDs exist is determined by entity memory

3. **Training**:
   - Ground truth: Dataset annotated with entity IDs
   - Loss: Predict correct entity ID (existing or new)
   - Challenge: No explicit supervision for CREATE/REFERENCE decision

## Proposed Clean Separation

### Layer 1: Entity Memory (Global)
- Maintains all entities and their properties
- Lifetime: Entire scenario
- Operations: `create()`, `get()`, `update()`

### Layer 2: Working Memory (Recent Context)
- Recent propositions (fixed window, e.g., last 20)
- Propositions contain entity IDs (integers)
- Lifetime: Rolling window

### Layer 3: Logic Rules (Per-Execution)
- Variables bind to entity IDs during matching
- Extract concepts from WM
- Bindings are temporary
- Lifetime: Single forward pass

### Layer 4: Generation (Decision Making)
- Uses concepts to decide next proposition
- Chooses: Create entity, reference entity, or assert relation
- Guided by attention over entity memory + NEW token

## Next Steps for Implementation

1. **Implement entity memory**: Simple dict with integer keys
2. **Modify logic rules**: Match against entity IDs, lookup properties as needed
3. **Add CREATE/REFERENCE decision**: Attention-based selection
4. **Annotate training data**: Mark entity IDs explicitly
5. **Train on simple multi-entity scenarios**: Test if distinction works

This separation keeps variables as pattern-matching tools (local scope) while entities are discourse referents (global scope).
