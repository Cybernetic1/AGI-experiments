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

**Example**:
```python
# Entity 0 persists across all time steps
t=0: [0, "is_a", "cat"]
t=1: [0, "color", "black"]
t=2: [0, "chases", 1]
t=3: [1, "runs_away", location_2]
```
- Entity 0 persists across all time steps
- Different rules can reference the same entity 0

## How They Work Together

Variables **bind to** entity IDs during rule matching:

```python
# Rule with variable
rule_premises = [
    [?X, "is_a", "cat"],    # Variable in first position
    [?X, "color", "black"]  # Same variable (must match)
]

# Working memory contains:
wm = [
    [0, "is_a", "cat"],
    [0, "color", "black"],
    [1, "is_a", "cat"],
    [1, "color", "white"]
]

# Step 1: Match first premise
# Tries [0, "is_a", "cat"]: Matches!
# Binds ?X → 0 (temporarily)

# Step 2: Match second premise with ?X=0
# Tries [0, "color", "black"]: Matches!
# Confirms binding ?X → 0

# Step 3: Execute rule with ?X=0
# Rule body/head uses entity 0

# Step 4: Clear binding
# ?X binding discarded (or try next binding)
# Rule can match again with ?X → different entity
```

## From Variable to Constant: Learning Names

Entities evolve naturally from **unknown** → **familiar** → **named constants**.

### Example: Learning About Napoleon

```python
# t=0: First encounter - create generic entity
entity_1 = create_entity()  # ID = 1
add_fact([1, "is_a", "general"])
add_fact([1, "nationality", "french"])

# t=50: Learn his name
add_fact([1, "name", "Napoleon"])
# Update name registry: "Napoleon" → 1

# t=100: Learn more facts
add_fact([1, "won_battle", "austerlitz"])
add_fact([1, "emperor_of", "france"])

# Entity 1 now has accumulated knowledge
# Can be referenced by ID (1) or by name ("Napoleon")
```

### Using Named Constants in Rules

**Initially (unknown entity)**:
```python
# Rule with variable
IF [?X, "is_a", "general"] AND [?X, "nationality", "french"]
THEN [?X, "likely_skilled", "military"]
```

**After learning name (becomes named constant)**:
```python
# Can now write rules with constant
IF ["napoleon", "located_at", ?Y]  # napoleon = entity_1
THEN [?Y, "historically_significant", true]

# System internally: "napoleon" → 1 (lookup via entity registry)
```

## Symbol Grounding Solution

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
