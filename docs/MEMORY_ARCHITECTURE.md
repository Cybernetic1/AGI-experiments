# Memory Architecture: Working Memory, Long-Term Memory, and Entity Registry

## Overview

This document describes the three-tier memory architecture for the logic-based AGI system.

## The Three Memory Tiers

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

## Working Memory (WM)

**Purpose**: Recent context for immediate reasoning

**Properties**:
- Fixed size: ~20 propositions (rolling window)
- Direct access: Logic rules scan all WM propositions
- Fast: Small enough to process every timestep
- Recent: Drops oldest facts when full

**Example**:
```python
wm = [
    [1, "name", "Napoleon"],
    [1, "is_a", "general"],
    [2, "name", "Austerlitz"],
    [1, "fought_at", 2],
    [1, "won", 3]
]
```

## Long-Term Memory (LTM)

**Purpose**: Complete history with efficient retrieval

**Properties**:
- Unbounded size: Stores all propositions permanently
- Content-addressable: Retrieved by pattern/similarity
- Differentiable: Uses attention mechanism
- Indexed: Each proposition has embedding

### Content-Addressable Retrieval

```python
class ContentAddressableLTM:
    def __init__(self):
        self.propositions = []  # All facts: [[subj, rel, obj], ...]
        self.embeddings = []    # Corresponding embeddings
        
    def add(self, proposition):
        self.propositions.append(proposition)
        emb = encode(proposition)  # Neural encoder
        self.embeddings.append(emb)
    
    def retrieve(self, query, k=10):
        """
        Query: Can be partial pattern or embedding
        Returns: Top-k matching propositions
        """
        if isinstance(query, list):  # Pattern like [?X, fought_at, ?Y]
            query_emb = encode(query)
        else:  # Already an embedding
            query_emb = query
            
        # Compute similarities
        similarities = cosine_similarity(query_emb, self.embeddings)
        
        # Return top-k
        top_indices = argsort(similarities)[-k:]
        return [self.propositions[i] for i in top_indices]
```

### Soft vs Hard Retrieval

**Hard Retrieval** (exact pattern matching):
```python
ltm.query("[?X, fought_at, austerlitz]")
# Returns: [[napoleon, fought_at, austerlitz], [french_army, fought_at, austerlitz]]
```

**Soft Retrieval** (differentiable, trainable):
```python
query_emb = encode([?X, fought_at, austerlitz])
attention_weights = softmax(query_emb @ ltm_embeddings.T / sqrt(d))
retrieved_emb = attention_weights @ ltm_embeddings
```

**Hybrid** (best of both):
```python
# Use hard retrieval to filter candidates (fast)
candidates = ltm.filter(subject=napoleon)

# Then use soft attention for final selection (accurate)
attention_weights = softmax(query_emb @ candidate_embs.T)
final_facts = attention_weights @ candidates
```

## Entity Registry

**Purpose**: Map between entity IDs and names

**Properties**:
- Integer keys: Entities are 0, 1, 2, ...
- Name mapping: "Napoleon" → 1, 1 → "Napoleon"
- Persistent: Entities never deleted
- Global scope: All modules access same registry

```python
class EntityRegistry:
    def __init__(self):
        self.next_id = 0
        self.id_to_name = {}  # {0: "Napoleon", 1: "Wellington"}
        self.name_to_id = {}  # {"Napoleon": 0, "Wellington": 1}
        
    def create(self, name=None):
        entity_id = self.next_id
        self.next_id += 1
        
        if name:
            self.id_to_name[entity_id] = name
            self.name_to_id[name] = entity_id
            
        return entity_id
    
    def add_name(self, entity_id, name):
        """Learn name for existing entity"""
        self.id_to_name[entity_id] = name
        self.name_to_id[name] = entity_id
    
    def resolve(self, name_or_id):
        """Resolve name to ID or ID to name"""
        if isinstance(name_or_id, int):
            return self.id_to_name.get(name_or_id)
        else:
            return self.name_to_id.get(name_or_id)
```

## WM/LTM Interaction Flow

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
        # Generate retrieval query
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

## Example: Recalling Napoleon's Victories

```python
# Current WM (recent context)
wm = [
    [1, "name", "Napoleon"],
    [1, "planning", "battle"],
    [2, "name", "Jena"]
]

# Logic rule activates: "When planning battle, recall past victories"
# → Triggers LTM retrieval

# Query generation
query = [1, "won", ?X]  # "What did Napoleon win?"

# LTM retrieval (content-addressable)
ltm.retrieve(query) → [
    [1, "won", "austerlitz"],
    [1, "won", "jena"],
    [1, "won", "wagram"],
    ["austerlitz", "year", 1805],
    ["jena", "year", 1806]
]

# Temporarily add to WM for this reasoning step
wm_augmented = wm + retrieved_facts

# Now logic rules can reason with both recent and recalled context
concepts = logic_rules(wm_augmented)
# → Might conclude: "Napoleon has winning track record"
```

## Why This Architecture?

✅ **Efficiency**: WM is small and fast, LTM only queried when needed  
✅ **Flexibility**: Content-addressable retrieval supports diverse queries  
✅ **Scalability**: LTM can grow without slowing down WM processing  
✅ **Differentiable**: Soft attention enables end-to-end training  
✅ **Biologically plausible**: Similar to hippocampal memory systems  
✅ **Unified representation**: Same proposition format across all tiers  

## Implementation Considerations

### When to Query LTM?

**Option 1: Rule-triggered**
```python
# Specific rules trigger retrieval
IF [?X, "planning", "battle"] THEN query_ltm([?X, "won", ?Y])
```

**Option 2: Uncertainty-based**
```python
# Query when confidence is low
if max(rule_activations) < THRESHOLD:
    query_ltm(generate_query(wm))
```

**Option 3: Explicit gate**
```python
# Learn to predict when retrieval is needed
should_retrieve = retrieval_gate(wm, concepts)
if should_retrieve:
    query_ltm(...)
```

### How Many Facts to Retrieve?

- **Fixed k**: Always retrieve top-5 facts (simple)
- **Dynamic k**: Retrieve until confidence threshold (adaptive)
- **Soft attention**: Use weighted average over all LTM (differentiable)

### How to Integrate Retrieved Facts?

- **Concatenate**: `wm_augmented = wm + retrieved` (simple)
- **Replace oldest**: `wm[0:k] = retrieved` (maintains size)
- **Attention-based mixing**: Weighted combination (sophisticated)

## Next Steps

1. Implement basic `WorkingMemory` class with rolling window
2. Implement `LongTermMemory` with embedding-based retrieval
3. Implement `EntityRegistry` with ID/name mapping
4. Test retrieval mechanism on synthetic scenarios
5. Add retrieval trigger logic (rule-based or learned)
6. Benchmark: retrieval accuracy and speed
