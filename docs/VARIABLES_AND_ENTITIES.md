# Variables vs Entities: Scoping and Binding

## The Fundamental Distinction

This document clarifies the critical difference between **variables** (pattern-matching tools) and **entities** (discourse referents).

### Variables (?X, ?Y)

**Scope**: Single logic rule  
**Purpose**: Pattern matching, unification  
**Lifetime**: During rule execution only  

**Example**:
```python
IF [?X, "is_a", "cat"] AND [?X, "color", "black"] THEN ...
```
- ?X binds to whatever matches during this rule execution
- Once rule completes, binding is discarded

### Entities (0, 1, 2, ...)

**Scope**: Entire scenario (multiple rules, multiple time steps)  
**Purpose**: Referring to specific individuals  
**Lifetime**: Throughout the discourse/scenario  

**Important**: Entities are represented as **integer IDs** internally (0, 1, 2, ...). Names like "Napoleon" are learned properties stored separately.

**Example**:
```python
# Actual internal representation (integer IDs only)
t=0: [0, 2, 5]      # Entity 0, relation 2, entity 5
t=1: [0, 3, 7]      # Entity 0, relation 3, value 7
t=2: [0, 8, 1]      # Entity 0, relation 8, entity 1
t=3: [1, 9, 12]     # Entity 1, relation 9, location 12

# Human-readable interpretation (via lookup tables)
# Entity 0 has name="cat" (learned)
# Relation 2 = "is_a" (learned)
# Entity 5 represents type "animal" (learned)
# So [0, 2, 5] means "cat is_a animal"
```

**Entity names are learned**:
```python
# Initially: entities are just IDs
entity_0 = 0  # No name yet

# Later: name property is learned from data
name_registry.add(0, "cat")  # Learn that entity 0 is called "cat"

# Now can resolve: "cat" → 0 and 0 → "cat"
```

## How They Work Together

Variables **bind to** entity IDs during rule matching:

```python
# Rule with variable (premise uses relation IDs internally)
rule_premises = [
    [?X, 2, 5],    # Variable in first position (relation 2="is_a", entity 5="cat" type)
    [?X, 3, 7]     # Same variable (relation 3="color", entity 7="black")
]

# Working memory contains (all integers)
wm = [
    [0, 2, 5],     # Entity 0, is_a, cat
    [0, 3, 7],     # Entity 0, color, black
    [1, 2, 5],     # Entity 1, is_a, cat
    [1, 3, 8]      # Entity 1, color, white
]

# Step 1: Match first premise [?X, 2, 5]
# Tries [0, 2, 5]: Matches!
# Binds ?X → 0 (temporarily)

# Step 2: Match second premise [?X, 3, 7] with ?X=0
# Tries [0, 3, 7]: Matches!
# Confirms binding ?X → 0

# Step 3: Execute rule with ?X=0
# Rule body/head uses entity 0

# Step 4: Clear binding
# ?X binding discarded (or try next binding)
# Rule can match again with ?X → different entity
```

**Human-readable version** (for documentation only):
```python
# Same as above, but with labels for readability
wm = [
    [cat_1, "is_a", "cat"],      # Really: [0, 2, 5]
    [cat_1, "color", "black"],   # Really: [0, 3, 7]
]
```
 (just an integer)
add_fact([1, 2, 100])      # [entity_1, is_a, person_type]
add_fact([1, 5, 101])      # [entity_1, nationality, french_type]

# Relation IDs: 2="is_a", 5="nationality"
# Entity IDs: 100="person" type, 101="french" type

# t=50: Learn his name (stored in name registry)
add_fact([1, 6, "Napoleon"])  # [entity_1, name_relation, name_string]
# Update name registry: "Napoleon" → 1

# t=100: Learn more facts
add_fact([1, 7, 200])      # [entity_1, won_battle, austerlitz_entity]
add_fact([1, 8, 102])      # [entity_1, emperor_of, france_entity]

# Entity 1 now has accumulated knowledge
# Can be referenced by ID (1) or by name ("Napoleon" → 1)
```

**How names are learned from text**:
```python
# NLP preprocessing extracts: "Napoleon was a general"
# Creates:
entity_id = create_entity()  # Returns 1
name_registry.add(1, "Napoleon")  # Learn name
add_fact([1, 2, 103])  # [1, is_a, general_type]

# From text to IDs:
text = "Napoleon was a general"
entities_extracted = {"Napoleon": (1, "PERSON")}
relations_extracted = [("Napoleon", "is_a", "general")]

# Convert to integer IDs:
propositions = [[1, 2, 103]]  # All integers
add_fact([1, "name", "Napoleon"])
# Update name registry: "Napoleon" → 1

# t=100: Learn more facts
add_fact([1, "won_battle", "austerlitz"])
add_fact([1, "emperor_of", "france"])

# Entity 1 now has accumulated knowledge
# Can be referenced by ID (1) or by name ("Napoleon")
```

### Using Named Constants in Rules

**Important**: Keep both IDs (for computation) and tags (for inspection)

```python
# Symbol Registries (bidirectional mappings)
class SymbolRegistries:
    def __init__(self):
        self.entity_to_name = {}   # {1: "Napoleon", 2: "Wellington"}
        self.name_to_entity = {}   # {"Napoleon": 1, "Wellington": 2}
        self.relation_to_name = {} # {7: "won", 8: "fought_at"}
        self.name_to_relation = {} # {"won": 7, "fought_at": 8}
    
    def decode_proposition(self, prop_ids):
        """Convert [1, 7, 200] to [Napoleon, won, Austerlitz] for inspection"""
        return [
            self.entity_to_name.get(prop_ids[0], f"entity_{prop_ids[0]}"),
            self.relation_to_name.get(prop_ids[1], f"rel_{prop_ids[1]}"),
            self.entity_to_name.get(prop_ids[2], f"entity_{prop_ids[2]}")
        ]

# Example: Dual representation
registries = SymbolRegistries()
registries.add_entity(1, "Napoleon")
registries.add_relation(7, "won")
registries.add_entity(200, "Austerlitz")

# For neural network: Use integer IDs
proposition_ids = [1, 7, 200]
concepts = logic_network(wm_as_integers)

# For human inspection: Decode to strings
print(registries.decode_proposition(proposition_ids))
# Output: ['Napoleon', 'won', 'Austerlitz']
```

**Benefits**:
✅ **Neural network processes integers** (fast, efficient)  
✅ **Humans inspect strings** (interpretable, debuggable)  
✅ **Examine learned rules** in readable form  
✅ **Trace reasoning steps** for debugging  
✅ **Validate preprocessing** correctness

# Relation vocabulary (learned from text)
relation_vocab = {
    "is_a": 2,          # String relation → integer ID
    "fought_at": 8,
    "won": 7
}

# In logic rules, human-readable constants are converted to IDs:
# Human writes (or we display):
rule = IF ["napoleon", "fought", ?X] THEN ...

# System internally converts to integers:
rule = IF [1, 8, ?X] THEN ...  # napoleon→1, fought→8

# When generating natural language OUTPUT:
proposition = [1, 7, 200]  # Integer IDs internally
# Lookup for display:
subject_name = name_registry.reverse_lookup(1)    # "Napoleon"
relation_name = relation_vocab.reverse_lookup(7)  # "won"
object_name = name_registry.reverse_lookup(200)   # "Austerlitz"
# Output: "Napoleon won Austerlitz"
```

**Learning vocabularies from text**:
```python
# During preprocessing, build vocabularies
relation_counter = Counter()
entity_mentions = {}

for text in corpus:
    # Extract relations
    relations = extract_relations(text)  # NLP tool
    for rel in relations:
        relation_counter[rel] += 1
    
    # Extract entity mentions
    entities = extract_entities(text)  # NLP tool
    for ent in entities:
        if ent.text not in entity_mentions:
            entity_id = next_id()
            entity_mentions[ent.text] = entity_id

# Build vocabularies
relation_vocab = {rel: idx for idx, (rel, _) 
                  in enumerate(relation_counter.most_common())}

# Result:
# relation_vocab = {"is_a": 0, "has": 1, "located_at": 2, ...}
# entity_mentions = {"Napoleon": 0, "France": 1, ...}

**Entity IDs are the ground truth** (numeric). **Names are properties**.

```python
# Name Resolution Layer
name_registry = {
    "Napoleon": 1,      # napoleon → entity_1
    "Wellington": 2,    # wellington → entity_2
    "France": 3         # france → entity_3
}

# In logic rules, constants are syntactic sugar:
rule = IF ["napoleon", "fought", ?X] THEN ...
# Preprocessed to:
rule = IF [1, "fought", ?X] THEN ...

# When generating natural language:
entity_1.get_name() = "Napoleon"
# Output: "Napoleon fought Wellington"
# Instead of: "Entity 1 fought Entity 2"
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
# Rule with variable: [?X, "is_a", "cat"]
# Matching against WM:

for proposition in working_memory:
    if matches_pattern(proposition, [?X, "is_a", "cat"]):
        bind(?X, proposition[0])  # Binds to existing entity
        # This REFERENCES existing entity

# If no match found:
# Some rules can CREATE entities:
if no_binding_found and context_suggests_new_entity:
    new_entity = entity_memory.create(...)
```

### Option C: Attention-Based Resolution (RECOMMENDED)

**Key insight**: Use concepts from logic network to query entities

**Important**: Concepts are NOT triplets - they're learned feature vectors (e.g., 256-dim)

```python
# Step 1: Logic network extracts concepts (aggregate features)
concepts = logic_rules(wm)  # Shape: (batch, 256)
# concepts = [0.8, 0.3, ...] ← Numbers representing scenario features
# NOT a triplet like [Napoleon, born_in, Corsica]

# Step 2: Generate proposition sequentially

# 2a. Select SUBJECT entity
subject_query = subject_projection(concepts)  # (batch, entity_dim)
# Projection learns: "Given scenario, which entity should be subject?"

entity_embeddings = get_entity_embeddings()  # (num_entities, entity_dim)
new_entity_token = trainable_parameter()  # Special "NEW" token
all_keys = concat([entity_embeddings, new_entity_token])

subject_scores = subject_query @ all_keys.T  # (batch, num_entities+1)
subject_id = argmax(subject_scores)

if subject_id == num_entities:  # Selected NEW
    subject_id = entity_memory.create(...)

# 2b. Select RELATION (conditioned on subject)
relation_input = concat([concepts, entity_embeddings[subject_id]])
relation_logits = relation_head(relation_input)  # (batch, num_relations)
relation_id = argmax(relation_logits)

# Note on constants: If the network learns that a specific relation should 
# always appear (e.g., "is_a" for type assertions), the relation_head will 
# consistently output high scores for that relation ID. 
# This effectively acts as a learned constant.

# 2c. Select OBJECT entity (conditioned on subject + relation)
object_input = concat([
    concepts, 
    entity_embeddings[subject_id],
    relation_embeddings[relation_id]
])
object_query = object_projection(object_input)  # (batch, entity_dim)
object_scores = object_query @ all_keys.T
object_id = argmax(object_scores)

if object_id == num_entities:  # Selected NEW
    object_id = entity_memory.create(...)

# Final proposition: [subject_id, relation_id, object_id]
```

**Key insight**: 
- Concepts describe "what's happening in the scenario"
- Different projections for subject vs object role
- Each step conditions on previous choices
- Example: If subject=Napoleon, relation=won, then object likely=battle

**Why this integrates well**:
- ✅ Uses shared concept representation from logic network
- ✅ Sequential generation allows conditioning
- ✅ Separate projections for subject vs object roles
- ✅ End-to-end differentiable
- ✅ Same architecture pattern as TTT (shared concepts → multiple heads)

## Gradual Solidification

Entities evolve naturally as knowledge accumulates:

1. **Unknown**: `entity_42` (just an ID, few properties)
2. **Familiar**: `entity_42` (rich properties, frequently referenced)
3. **Named**: `entity_42` gets `name="Napoleon"` property
4. **Constant**: Rules can use `"napoleon"` (syntactic sugar for entity_42)

**The beauty**: Same underlying mechanism (entity IDs + propositions), but usage patterns change as knowledge accumulates.

✅ **No special "constant" type**: Constants are just well-known entities  
✅ **Names are learned**: Not hardcoded, discovered through experience  
✅ **Unified representation**: Variables bind to IDs, constants resolve to IDs  
✅ **Graceful scaling**: System works whether entity is unknown or famous  

### Generating Constants in Practice

**Q: Can the network generate a constant as subject/object/relation?**

**A: Yes! Constants are just high-confidence selections:**

```python
# Example 1: Generating entity constant "Napoleon" as subject
concepts = logic_rules(wm)  # Context: "The French emperor..."
subject_query = subject_projection(concepts)
subject_scores = subject_query @ entity_embeddings.T
# subject_scores = [0.05, 0.93, 0.01, ...]  ← Very high for entity_1 (Napoleon)
#                        ↑
#                        Napoleon consistently selected → acts as constant

# Example 2: Generating relation constant "is_a" (type assertion)
relation_logits = relation_head(concat([concepts, subject_emb]))
# relation_logits = [0.02, 0.01, 0.95, ...]  ← Very high for relation_2 (is_a)
#                                  ↑
#                                  "is_a" consistently selected → acts as constant

# Example 3: Generating type constant "person" as object
object_query = object_projection(concat([concepts, subject_emb, is_a_emb]))
object_scores = object_query @ entity_embeddings.T
# If entity_5 represents the type "person", it gets high score
# object_scores = [..., 0.91, ...]  ← entity_5 (person type) selected
```

**Key insight**: 
- Constants are **learned patterns**, not hardcoded
- Network learns: "In this context, always select entity X"
- High confidence = behaves like constant
- Still differentiable and learnable

**Types as entities**:
```python
# Types can be entities too!
entity_registry = {
    0: "Napoleon",     # Specific individual
    1: "Wellington",   # Specific individual
    2: "Austerlitz",   # Specific location
    # Types/categories:
    100: "person",     # Type constant
    101: "location",   # Type constant
    102: "battle",     # Type constant
}

# Then: [0, "is_a", 100] = "Napoleon is a person"
# Subject: constant (entity 0 = Napoleon)
# Relation: constant (is_a)
# Object: constant (entity 100 = person type)
```  

## Implementation Strategy

### Architecture Integration

```python
class HierarchicalLogicNetwork(nn.Module):
    def __init__(self, concept_dim=256, entity_dim=128, num_relations=50):
        super().__init__()
        
        # Shared logic network (extracts concepts from WM)
        self.logic_rules = LogicNetwork(concept_dim=concept_dim)
        
        # Entity embeddings
        self.entity_embeddings = nn.Embedding(max_entities, entity_dim)
        self.new_entity_token = nn.Parameter(torch.randn(entity_dim))
        
        # Proposition generation (sequential)
        # Subject selection
        self.subject_projection = nn.Linear(concept_dim, entity_dim)
        
        # Relation selection (conditioned on subject)
        self.relation_head = nn.Linear(concept_dim + entity_dim, num_relations)
        
        # Object selection (conditioned on subject + relation)
        self.object_projection = nn.Linear(
            concept_dim + entity_dim + relation_embed_dim, 
            entity_dim
        )
        
        # Relation embeddings
        self.relation_embeddings = nn.Embedding(num_relations, relation_embed_dim)
    
    def forward(self, wm, entity_registry):
        # Step 1: Extract concepts (shared across all decisions)
        concepts = self.logic_rules(wm)  # (batch, concept_dim=256)
        
        # Step 2: Select SUBJECT entity
        subject_query = self.subject_projection(concepts)
        entity_embs = self.entity_embeddings(entity_registry.get_ids())
        all_keys = torch.cat([entity_embs, self.new_entity_token.unsqueeze(0)])
        
        subject_scores = subject_query @ all_keys.T
        subject_id = torch.argmax(subject_scores, dim=-1)
        
        # Step 3: Select RELATION (conditioned on subject)
        subject_emb = self.entity_embeddings(subject_id)
        relation_input = torch.cat([concepts, subject_emb], dim=-1)
        relation_logits = self.relation_head(relation_input)
        relation_id = torch.argmax(relation_logits, dim=-1)
        
        # Step 4: Select OBJECT entity (conditioned on subject + relation)
        relation_emb = self.relation_embeddings(relation_id)
        object_input = torch.cat([concepts, subject_emb, relation_emb], dim=-1)
        object_query = self.object_projection(object_input)
        object_scores = object_query @ all_keys.T
        object_id = torch.argmax(object_scores, dim=-1)
        
        # Final proposition
        proposition = [subject_id, relation_id, object_id]
        
        return proposition
```

### Entity and Proposition Flow

```python
# Entity creation (generic)
entity_id = next_id()  # Returns 1, 2, 3, ...

# Property accumulation (gradual)
add_fact([entity_id, "is_a", "general"])
add_fact([entity_id, "name", "Napoleon"])  # Aha! Now we know the name

# Rule execution (unified)
# Both syntaxes work:
match([?X, "is_a", "general"])          # Variable: matches any general
match(["napoleon", "is_a", "general"])  # Constant: resolve napoleon→1, then match

# The system treats them the same internally (entity IDs)
```

## Next Steps

1. Implement entity registry with ID/name mapping
2. Implement variable binding mechanism in rule matching
3. Add CREATE/REFERENCE decision logic (attention-based)
4. Annotate training data with entity IDs
5. Train on entity tracking scenarios
6. Test name learning (entity gets name property later)
