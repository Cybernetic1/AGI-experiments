# Logic Network Architecture

## Overview

The logic network is the core reasoning component that processes working memory and extracts high-level **concepts** (learned features) that guide entity selection and proposition generation.

**Key idea**: Logic rules perform **fuzzy pattern matching** against working memory, extracting features that capture the current scenario state.

## Architecture from TTT (Proven to Work)

From the successful TTT experiments, the logic network consists of:

```python
class LogicNetwork(nn.Module):
    def __init__(self, num_rules, premise_dim, concept_dim):
        self.num_rules = num_rules
        
        # Each rule has premises (what to match)
        self.premise_embeddings = nn.Parameter(
            torch.randn(num_rules, max_premises, premise_dim)
        )
        
        # Cylindrification parameters (γ) - variable vs constant
        self.gamma = nn.Parameter(
            torch.randn(num_rules, max_premises, 3)  # For [subj, rel, obj]
        )
        
        # Rule weights (how much each rule contributes to concepts)
        self.rule_weights = nn.Linear(num_rules, concept_dim)
        
    def forward(self, working_memory):
        """
        Args:
            working_memory: (batch, num_props, 3, prop_dim)
                - num_props: number of propositions in WM
                - 3: [subject, relation, object]
                - prop_dim: embedding dimension
        
        Returns:
            concepts: (batch, concept_dim)
                - High-level features extracted by rules
        """
        # Step 1: Match rules against WM (fuzzy unification)
        rule_activations = self.match_rules(working_memory)  # (batch, num_rules)
        
        # Step 2: Aggregate rule activations into concepts
        concepts = self.rule_weights(rule_activations)  # (batch, concept_dim)
        
        return concepts
```

## Key Components

### 1. Fuzzy Unification (Pattern Matching)

Each rule matches against propositions in working memory:

```python
def match_rules(self, wm):
    """
    For each rule, compute activation based on how well
    its premises match propositions in working memory.
    """
    batch_size, num_props, _, prop_dim = wm.shape
    
    rule_activations = []
    
    for rule_idx in range(self.num_rules):
        # Get premises for this rule
        premises = self.premise_embeddings[rule_idx]  # (max_premises, premise_dim)
        gamma = self.gamma[rule_idx]  # (max_premises, 3)
        
        # Match each premise against all WM propositions
        premise_matches = []
        for premise_idx in range(max_premises):
            premise = premises[premise_idx]
            gamma_p = gamma[premise_idx]  # (3,) for [subj, rel, obj]
            
            # Compute similarity between premise and each WM proposition
            # Using cylindrification (γ) to handle variables
            matches = self.unify(premise, gamma_p, wm)  # (batch, num_props)
            
            # Max pooling: premise satisfied if ANY prop matches
            best_match = torch.max(matches, dim=1)[0]  # (batch,)
            premise_matches.append(best_match)
        
        # Rule activates if ALL premises satisfied (AND logic)
        rule_activation = torch.min(torch.stack(premise_matches), dim=0)[0]
        rule_activations.append(rule_activation)
    
    return torch.stack(rule_activations, dim=1)  # (batch, num_rules)
```

### 2. Cylindrification (γ Parameters)

**Key innovation**: γ controls variable vs constant behavior

```python
def unify(self, premise, gamma, wm_props):
    """
    Match premise against WM propositions using cylindrification.
    
    Args:
        premise: (premise_dim,) - What to match
        gamma: (3,) - Variable/constant controls for [subj, rel, obj]
        wm_props: (batch, num_props, 3, prop_dim) - Working memory
    
    γ ≈ 0: Acts as variable (ignores this position, allows any value)
    γ ≈ 1: Acts as constant (must match exactly)
    """
    batch_size, num_props, _, prop_dim = wm_props.shape
    
    # Compute similarity for each position
    similarities = []
    for pos in range(3):  # subject, relation, object
        # Similarity between premise and WM at this position
        sim = cosine_similarity(premise, wm_props[:, :, pos, :])  # (batch, num_props)
        
        # Weight by γ: if γ ≈ 0, this position doesn't matter (variable)
        weighted_sim = gamma[pos] * sim + (1 - gamma[pos]) * 1.0
        similarities.append(weighted_sim)
    
    # Combine: ALL positions must match (considering γ weights)
    overall_match = torch.prod(torch.stack(similarities), dim=0)
    
    return overall_match  # (batch, num_props)
```

### 3. Concept Extraction

Rules that activate contribute to concepts:

```python
# Rule activations: (batch, num_rules)
# Each rule captures a pattern: "X chases Y", "Black pieces in corner", etc.

# Aggregate into concepts (learned linear combination)
concepts = self.rule_weights(rule_activations)  # (batch, concept_dim)

# Concepts are high-level features like:
# - "Aggressive situation detected"
# - "Entity relationships present"
# - "Question being asked"
```

## Adapting to Text/Propositions

### TTT vs Text Differences

| Aspect | TTT | Text/Propositions |
|--------|-----|-------------------|
| WM size | 9 propositions (fixed) | 20+ propositions (variable) |
| Entities | 2 (X, O) | Hundreds (entity IDs) |
| Relations | 1 (occupies) | Many (is_a, has, fought_at, ...) |
| Rules | 6-8 | Potentially thousands |

### Modifications Needed

**1. Variable WM Size**

Use attention/padding to handle variable number of propositions:

```python
def forward(self, working_memory, wm_mask):
    """
    Args:
        working_memory: (batch, max_props, 3, prop_dim)
        wm_mask: (batch, max_props) - 1 for valid props, 0 for padding
    """
    # Apply mask during matching
    matches = self.unify(premise, gamma, working_memory)
    matches = matches * wm_mask  # Zero out padding
    best_match = torch.max(matches, dim=1)[0]
```

**2. Entity Representations**

Subject/object positions contain entity IDs (integers), need embedding:

```python
class WorkingMemoryEncoder(nn.Module):
    def __init__(self, entity_embed_dim, relation_embed_dim):
        self.entity_embeddings = nn.Embedding(max_entities, entity_embed_dim)
        self.relation_embeddings = nn.Embedding(num_relations, relation_embed_dim)
    
    def forward(self, propositions):
        """
        Args:
            propositions: (batch, num_props, 3) - [subj_id, rel_id, obj_id]
        
        Returns:
            embedded: (batch, num_props, 3, embed_dim)
        """
        subjects = self.entity_embeddings(propositions[:, :, 0])
        relations = self.relation_embeddings(propositions[:, :, 1])
        objects = self.entity_embeddings(propositions[:, :, 2])
        
        return torch.stack([subjects, relations, objects], dim=2)
```

**3. Rule Retrieval (for scaling)**

When num_rules > 1000, use retrieval (see [SCALING_CHALLENGES.md](SCALING_CHALLENGES.md)):

```python
# Don't execute all rules, only retrieve relevant ones
relevant_rule_ids = rule_retrieval_network(wm_summary)  # Top-k
rule_activations = execute_rules(relevant_rule_ids, working_memory)
```

## Concept Structure and Positions

**Question**: Do concepts preserve positional information?

**Answer**: Concepts are **position-aware summaries**, not individual triplets

### Important Distinction

```python
# Working Memory contains TRIPLETS (propositions):
wm = [
    [0, "is_a", "person"],        # Entity 0 (Napoleon) is a person
    [0, "name", "Napoleon"],      # Entity 0's name is Napoleon
    [0, "born_in", 2],            # Entity 0 born in entity 2 (Corsica)
    [2, "is_a", "location"]       # Entity 2 is a location
]

# Logic rules process WM and extract CONCEPTS (fixed-size feature vector):
concepts = logic_rules(wm)  # Shape: (batch, 256)
# concepts = [0.8, 0.3, 0.9, 0.1, ...]  ← Just numbers!
#            ↑    ↑    ↑    ↑
#            |    |    |    Rule 4 activation
#            |    |    Rule 3 activation (e.g., "birth relation present")
#            |    Rule 2 activation (e.g., "person entity present")
#            Rule 1 activation (e.g., "entity has name")

# Concepts are NOT "Napoleon born_in Corsica"
# They are learned features like:
# - "High confidence that subject is a person entity"
# - "Birth/origin relation detected"
# - "Location entity present in WM"
# - "Named entity being discussed"
```

### How Entity Selection Works

The key is that we're generating a proposition **step by step**:

```python
# Step 1: Decide WHAT to say (extract concepts from context)
concepts = logic_rules(wm)  # (batch, 256) - "what's happening?"

# Step 2: Decide SUBJECT entity
subject_query = subject_projection(concepts)  # (batch, entity_dim)
# This projection learns: "Given context, what entity should be subject?"

entity_embeddings = get_all_entities()  # (num_entities, entity_dim)
subject_scores = subject_query @ entity_embeddings.T  # (batch, num_entities)
subject_id = argmax(subject_scores)  # Select entity (e.g., entity 0 = Napoleon)

# Step 3: Decide RELATION (given subject)
relation_input = concat([concepts, subject_embedding])
relation_logits = relation_head(relation_input)  # (batch, num_relations)
relation = argmax(relation_logits)  # e.g., "won"

# Step 4: Decide OBJECT entity (given subject + relation)
object_input = concat([concepts, subject_embedding, relation_embedding])
object_query = object_projection(object_input)
object_scores = object_query @ entity_embeddings.T
object_id = argmax(object_scores)  # e.g., entity 3 = battle

# Final proposition: [0, "won", 3] = "Napoleon won battle"
```

### Example: Entity Position Awareness

```python
# Rule 1: "Subject entity is a person"
# Premises: [?X, is_a, person]
# When activated, contributes to concept: "person_subject_present"

# Rule 2: "Object entity is a location"  
# Premises: [?, ?, ?Y], [?Y, is_a, location]
# When activated, contributes to: "location_object_present"

# Concepts combine these:
concepts = [0.8, 0.3, 0.9, ...]  # High-level features
#          ↑    ↑    ↑
#          |    |    Rule 3 (entity chain detected)
#          |    Rule 2 (location as object)
#          Rule 1 (person as subject)

# These concepts guide entity selection:
# "Should create new entity or reference existing one?"
# "If referencing, which entity is most relevant?"
```

## Training the Logic Network

### Autoregressive Training

```python
# Predict next proposition from current WM
for batch in dataloader:
    wm_t = batch.working_memory  # Current state
    target_prop = batch.next_proposition  # Ground truth
    
    # Extract concepts
    concepts = logic_network(wm_t)
    
    # Predict next proposition
    pred_prop = ar_head(concepts)
    
    # Loss
    loss = cross_entropy(pred_prop, target_prop)
    loss.backward()
```

### What the Network Learns

- **Rules learn patterns**: Which combinations of propositions are important
- **γ parameters learn**: Which positions are variables vs constants
- **Concept weights learn**: How to combine rule activations for prediction

### Interpretability

After training, we can inspect:

```python
# Rule 5 has high activation on this WM state
# Check its premises and γ:
premises = logic_network.premise_embeddings[5]
gamma = logic_network.gamma[5]

# Decode: "This rule looks for [ENTITY, 'is_a', PERSON]"
# Because γ[0] ≈ 0 (variable subject), γ[1] ≈ 1, γ[2] ≈ 1
# Interpretation: "Any entity that is a person"
```

## Integration with Entity Selection

From the hierarchical architecture:

```python
class HierarchicalLogicNetwork(nn.Module):
    def __init__(self):
        # Core logic network
        self.logic_rules = LogicNetwork(
            num_rules=100,
            premise_dim=64,
            concept_dim=256
        )
        
        # Entity selection (uses concepts)
        self.entity_query_proj = nn.Linear(256, entity_dim)
        
        # Generation heads (also use concepts)
        self.ar_head = ARHead(concept_dim=256)
        self.rl_head = RLHead(concept_dim=256)
    
    def forward(self, wm, entity_registry):
        # Extract concepts (CORE REASONING)
        concepts = self.logic_rules(wm)  # (batch, 256)
        
        # Use concepts for entity selection
        entity_query = self.entity_query_proj(concepts)
        selected_entity = select_entity(entity_query, entity_registry)
        
        # Use concepts + entity for proposition generation
        output = self.ar_head(concepts, selected_entity)
        
        return output
```

## Summary

**Logic Network Role**:
- Process working memory (propositions)
- Match learned rules (fuzzy unification)
- Extract concepts (high-level features)
- Provide concepts to downstream modules

**Key Properties**:
- ✅ Position-aware (rules can match specific positions)
- ✅ Learnable (all parameters trained end-to-end)
- ✅ Interpretable (can inspect rules and γ values)
- ✅ Scalable (with rule retrieval for large rule sets)
- ✅ Proven (worked in TTT experiments)

**Concepts**:
- Fixed-size vector (e.g., 256-dim)
- Aggregate features from rule activations
- Capture current scenario state
- Used by entity selection, AR head, RL head

This is the core of the reasoning system - everything else builds on these concepts!
