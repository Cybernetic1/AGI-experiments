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

## Long-Term Memory vs Working Memory

### The Key Insight: Entities Persist, Propositions Move

**Entities** are permanent references (numeric IDs). **Propositions** about them can be in:
- **Working Memory (WM)**: Recent/active facts (small, fast)
- **Long-Term Memory (LTM)**: Historical facts (large, retrieved when needed)

### Example: Learning About Napoleon

```python
# t=0: First encounter - create generic entity
entity_1 = create_entity()  # ID = 1
LTM += [
    [1, "is_a", general],
    [1, "nationality", french],
    [1, "era", 1800s]
]
WM = [[1, "is_a", general], [1, "nationality", french]]

# t=50: Learn his name
LTM += [[1, "name", "Napoleon"]]
WM = [[1, "name", "Napoleon"], [1, "is_a", general]]  # Recent context

# t=100: Learn more facts
LTM += [
    [1, "height", short],
    [1, "won_battle", austerlitz],
    [1, "emperor_of", france]
]
WM = [[1, "won_battle", austerlitz], [1, "emperor_of", france]]  # Most recent

# Entity 1 now has accumulated knowledge
# Can be referenced by ID (1) or by name ("Napoleon")
```

### From Variable to Constant: The Transition

**Initially (unknown entity)**:
```python
# Rule with variable
IF [?X, is_a, general] AND [?X, nationality, french]
THEN [?X, likely_skilled, military]

# ?X can bind to entity_1 (or any general)
```

**After learning name (becomes named constant)**:
```python
# Can now write rules with constant
IF [napoleon, located_at, ?Y]  # napoleon = entity_1
THEN [?Y, historically_significant, true]

# System internally: napoleon → entity_1 (lookup)
```

### The Symbol Grounding Solution

**Entity IDs are the ground truth** (numeric). **Names are properties**:

```python
# Name Resolution Layer
name_registry = {
    "Napoleon": 1,      # napoleon → entity_1
    "Wellington": 2,    # wellington → entity_2
    "France": 3         # france → entity_3 (countries are entities too)
}

# In logic rules, constants are syntactic sugar:
rule = IF [napoleon, fought, ?X] THEN ...
# Preprocessed to:
rule = IF [1, fought, ?X] THEN ...

# When generating natural language:
entity_1.properties["name"] = "Napoleon"
# Output: "Napoleon fought Wellington"
# Instead of: "Entity 1 fought Entity 2"
```

### WM and LTM Interaction

```python
class Memory:
    def __init__(self):
        self.ltm = []  # All propositions (persistent)
        self.wm = []   # Recent propositions (rolling window)
        self.name_registry = {}  # Name → entity_id mapping
        
    def add_fact(self, proposition):
        # All facts go to LTM
        self.ltm.append(proposition)
        
        # Recent facts also in WM
        self.wm.append(proposition)
        if len(self.wm) > MAX_WM_SIZE:
            self.wm.pop(0)  # Remove oldest
            
        # If it's a name fact, update registry
        if proposition[1] == "name":
            entity_id, _, name = proposition
            self.name_registry[name] = entity_id
    
    def retrieve(self, query):
        # Search WM first (fast, recent)
        results = [p for p in self.wm if matches(p, query)]
        
        # If insufficient, search LTM (slow, comprehensive)
        if len(results) < THRESHOLD:
            results += [p for p in self.ltm if matches(p, query)]
        
        return results
    
    def resolve_name(self, name):
        # Convert name constant to entity ID
        return self.name_registry.get(name, None)
```

### Elegant Property: Gradual Solidification

Entities evolve naturally from **unknown** → **familiar** → **named constants**:

1. **Unknown**: `entity_42` (just an ID, few properties)
2. **Familiar**: `entity_42` (rich properties, frequently referenced)
3. **Named**: `entity_42` gets `name="Napoleon"` property
4. **Constant**: Rules can use `napoleon` (syntactic sugar for entity_42)

**The beauty**: Same underlying mechanism (entity IDs + propositions), but usage patterns change as knowledge accumulates.

### Why This Is Elegant

✅ **No special "constant" type**: Constants are just well-known entities  
✅ **Names are learned**: Not hardcoded, discovered through experience  
✅ **Unified representation**: Variables bind to IDs, constants resolve to IDs  
✅ **Graceful scaling**: System works whether entity is unknown or famous  
✅ **LTM/WM are storage tiers**: Not different data structures, same propositions  

### Implementation Strategy

```python
# Entity creation (generic)
entity_id = next_id()  # Returns 1, 2, 3, ...

# Property accumulation (gradual)
add_fact([entity_id, "is_a", general])
add_fact([entity_id, "name", "Napoleon"])  # Aha! Now we know the name

# Rule execution (unified)
# Both syntaxes work:
match([?X, is_a, general])        # Variable: matches any general (including entity_1)
match([napoleon, is_a, general])  # Constant: resolve napoleon→1, then match

# The system treats them the same internally (entity IDs)
```

## Training Acceleration via Logical Preprocessing

### Key Advantage Over Traditional LLMs

**Traditional LLMs**: Learn everything from raw tokens
- Must discover: entities, relations, coreference, etc.
- No explicit structure guidance
- Slow, data-hungry learning

**Logic-Structured AGI**: Can leverage NLP preprocessing
- Pre-extract entities and relations from text
- Provide explicit supervision signals
- Faster, more sample-efficient learning

### Preprocessing Pipeline

```python
# Input: Natural language text
text = "A French general named Napoleon fought at Austerlitz. He won the battle."

# Step 1: Entity Extraction (NLP)
entities = [
    {"id": 1, "mentions": ["French general", "Napoleon", "He"], "type": "PERSON"},
    {"id": 2, "mentions": ["Austerlitz"], "type": "LOCATION"},
    {"id": 3, "mentions": ["the battle"], "type": "EVENT"}
]

# Step 2: Relation Extraction (NLP)
relations = [
    [1, "name", "Napoleon"],
    [1, "nationality", "French"],
    [1, "occupation", "general"],
    [1, "fought_at", 2],
    [1, "won", 3],
    [3, "location", 2]
]

# Step 3: Convert to Logic Format (Training Data)
training_sequence = [
    # t=0: Introduce entity
    {"wm": [], "action": "CREATE", "entity_type": "PERSON", 
     "properties": ["nationality=French", "occupation=general"]},
    
    # t=1: Add name property
    {"wm": [[1, "is_a", "person"]], "action": "ADD_PROPERTY",
     "entity": 1, "property": "name", "value": "Napoleon"},
    
    # t=2: Create location entity
    {"wm": [[1, "name", "Napoleon"]], "action": "CREATE",
     "entity_type": "LOCATION", "name": "Austerlitz"},
    
    # t=3: Add relation
    {"wm": [[1, "name", "Napoleon"], [2, "name", "Austerlitz"]],
     "action": "ADD_RELATION", "subject": 1, "relation": "fought_at", "object": 2},
    
    # t=4: Coreference ("He" → entity 1)
    {"wm": [[1, "fought_at", 2]], "action": "ADD_PROPERTY",
     "entity": 1, "property": "won", "value": 3}
]

# Step 4: Train on Logical Sequences
# System learns to predict next action/proposition
# With explicit entity IDs and structure
```

### Why This Accelerates Training

1. **Explicit entity supervision**: System knows which mentions refer to same entity
2. **Relation labels**: Direct supervision on relation types
3. **Coreference resolution**: Pre-solved (He → Napoleon)
4. **Type information**: Entity types guide creation decisions
5. **Structured targets**: Predict logical propositions, not raw tokens

### Comparison

**Traditional LLM Training**:
```
Input:  "A French general named Napoleon fought"
Target: " at Austerlitz"
Loss: Cross-entropy on token prediction
```
→ Must implicitly learn: entities, relations, coreference, reasoning

**Logic-AGI Training**:
```
Input:  WM = [[1, nationality, French], [1, occupation, general]]
Target: Action = ADD_PROPERTY, entity=1, property=name, value=Napoleon
Loss: Classification loss on action + entity + property
```
→ Explicitly supervised on structure

## LTM/WM Interaction: Content-Addressable Retrieval

### The Missing Piece: How LTM Is Queried

**LTM as Content-Addressable Memory**: Use attention/similarity for retrieval

```python
class ContentAddressableLTM:
    def __init__(self):
        self.propositions = []  # All facts: [[subj, rel, obj], ...]
        self.embeddings = []    # Corresponding embeddings
        
    def add(self, proposition):
        self.propositions.append(proposition)
        # Encode proposition as embedding
        emb = encode(proposition)  # Neural encoder
        self.embeddings.append(emb)
    
    def retrieve(self, query, k=10):
        """
        Query: Can be partial pattern or embedding
        Returns: Top-k matching propositions
        """
        # Encode query
        if isinstance(query, list):  # Pattern like [?X, fought_at, ?Y]
            query_emb = encode(query)
        else:  # Already an embedding (from WM context)
            query_emb = query
            
        # Compute similarities
        similarities = cosine_similarity(query_emb, self.embeddings)
        
        # Return top-k
        top_indices = argsort(similarities)[-k:]
        return [self.propositions[i] for i in top_indices]
```

### WM/LTM Interaction Flow

```python
def process_timestep(wm, ltm, logic_rules):
    """
    At each reasoning step:
    1. Logic rules process WM (recent context)
    2. If more context needed → query LTM
    3. Retrieved facts temporarily added to WM
    4. Continue reasoning
    5. New facts added to both WM and LTM
    """
    
    # Step 1: Apply logic rules to WM
    concepts = logic_rules(wm)
    
    # Step 2: Check if retrieval needed
    if needs_more_context(concepts):
        # Generate retrieval query from current concepts
        query = generate_query(concepts)  # e.g., [napoleon, ?, ?]
        
        # Retrieve from LTM
        retrieved_facts = ltm.retrieve(query, k=5)
        
        # Temporarily augment WM
        wm_augmented = wm + retrieved_facts
        
        # Re-apply logic rules with augmented context
        concepts = logic_rules(wm_augmented)
    
    # Step 3: Generate next proposition
    next_prop = generate_proposition(concepts)
    
    # Step 4: Update both memories
    ltm.add(next_prop)  # Permanent storage
    wm.append(next_prop)  # Recent context
    if len(wm) > MAX_WM_SIZE:
        wm.pop(0)  # Keep WM bounded
    
    return wm, ltm
```

### Example: Recalling Napoleon's Victories

```python
# Current WM (recent context)
wm = [
    [1, name, "Napoleon"],
    [1, planning, battle],
    [2, name, "Jena"]
]

# Logic rule activates: "When planning battle, recall past victories"
# → Triggers LTM retrieval

# Query generation
query = [1, "won", ?X]  # "What did Napoleon win?"

# LTM retrieval (content-addressable)
ltm.retrieve(query) → [
    [1, won, austerlitz],  # Retrieved from LTM
    [1, won, jena],
    [1, won, wagram],
    [austerlitz, year, 1805],
    [jena, year, 1806]
]

# Temporarily add to WM for this reasoning step
wm_augmented = wm + retrieved_facts

# Now logic rules can reason with both recent and recalled context
concepts = logic_rules(wm_augmented)
# → Might conclude: "Napoleon has winning track record"
```

### Three-Tier Memory Architecture

```
┌─────────────────────────────────────────┐
│  Working Memory (WM)                    │
│  - Size: ~20 propositions               │
│  - Access: Direct (O(n) scan)           │
│  - Content: Recent context              │
└──────────┬──────────────────────────────┘
           │ retrieve
           ↓ when needed
┌─────────────────────────────────────────┐
│  Long-Term Memory (LTM)                 │
│  - Size: Unlimited                      │
│  - Access: Content-addressable (O(k))   │
│  - Content: All historical facts        │
│  - Structure: Embedding + proposition   │
└──────────┬──────────────────────────────┘
           │ ground
           ↓ properties
┌─────────────────────────────────────────┐
│  Entity Registry                         │
│  - Size: # of entities                  │
│  - Access: Direct lookup by ID/name     │
│  - Content: ID ↔ name mapping          │
└─────────────────────────────────────────┘
```

### Why Content-Addressable?

✅ **Flexible retrieval**: Can query by pattern, similarity, or specific entities  
✅ **Efficient**: Don't scan all of LTM, only retrieve relevant facts  
✅ **Differentiable**: Attention mechanism is trainable (soft retrieval)  
✅ **Graceful scaling**: Works whether LTM has 100 or 1M facts  
✅ **Biological plausibility**: Similar to hippocampal pattern completion  

### Implementation: Soft vs Hard Retrieval

**Hard Retrieval** (like databases):
```python
# Exact pattern matching
ltm.query("[?X, fought_at, austerlitz]")
# Returns: [[napoleon, fought_at, austerlitz], [french_army, fought_at, austerlitz]]
```

**Soft Retrieval** (differentiable, trainable):
```python
# Attention-based similarity
query_emb = encode([?X, fought_at, austerlitz])
attention_weights = softmax(query_emb @ ltm_embeddings.T / sqrt(d))
retrieved_emb = attention_weights @ ltm_embeddings
# Use retrieved_emb as additional context (fuzzy/soft facts)
```

**Hybrid** (best of both):
```python
# Use hard retrieval to filter candidates (fast)
candidates = ltm.filter(subject=napoleon)  # O(n) but small n

# Then use soft attention for final selection (accurate)
attention_weights = softmax(query_emb @ candidate_embs.T)
final_facts = attention_weights @ candidates
```

## Next Steps for Implementation

1. **Implement unified memory**: LTM (all facts) + WM (recent facts)
2. **Add content-addressable LTM**: Embedding + similarity-based retrieval
3. **Add name registry**: Map symbolic names → entity IDs
4. **Implement NLP preprocessing**: Entity + relation extraction from text
5. **Modify logic rules**: Support both variables (?X) and named constants (napoleon)
6. **Implement retrieval mechanism**: WM first, LTM query when needed
7. **Train on entity persistence**: Same entity across multiple contexts
8. **Test name learning**: Entity gets name property later in sequence
9. **Benchmark**: Compare learning speed vs traditional LLM on same data

This separation keeps variables as pattern-matching tools (local scope) while entities are discourse referents (global scope), and names are just special properties that enable constant-like usage. The preprocessing pipeline and content-addressable LTM provide significant advantages over traditional end-to-end neural approaches.
