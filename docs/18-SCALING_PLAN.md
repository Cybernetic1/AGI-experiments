# Scaling Up: Larger Corpora & LTM Integration

**Date:** January 1, 2026  
**Current Status:** TinyStories working, ready to scale

---

## Question 1: What Larger Text Corpora After TinyStories?

### Progression Strategy: Gradual Scaling

```
Current:   TinyStories (100-10K stories)
           ↓
Stage 1:   Medium datasets (100K-1M samples)
           ↓
Stage 2:   Large datasets (1M-10M samples)
           ↓
Stage 3:   Massive datasets (10M+ samples)
```

---

## Stage 1: Medium Corpora (Recommended Next Steps)

### 1. Full TinyStories (~2.1M stories)

**Why start here:**
- ✅ Already integrated in our code
- ✅ Simple, clean language
- ✅ Good for testing architecture
- ✅ Diverse narrative patterns

**Download:**
```python
from datasets import load_dataset
ds = load_dataset('roneneldan/TinyStories', split='train')
# 2.1M stories total
```

**Training time (GPU):**
- 100K stories: ~2-3 hours
- 1M stories: ~1 day
- 2.1M stories (full): ~2 days

### 2. Children's Book Test (CBT) - ~688K sentences

**Why:**
- ✅ Children's books (similar to TinyStories)
- ✅ Named entity focused (good for entity tracking)
- ✅ Question answering included
- ✅ Diverse authors (Grimm, Potter, etc.)

**Download:**
```python
from datasets import load_dataset
ds = load_dataset('cbt', 'raw')
```

**Use case:** Test entity persistence across stories

### 3. BookCorpus (~74M sentences)

**Why:**
- ✅ Unpublished novels (more complex than children's)
- ✅ Rich narrative structure
- ✅ Good character development
- ✅ Used to train BERT

**Download:**
```python
# Note: Original BookCorpus is unavailable, use alternative:
from datasets import load_dataset
ds = load_dataset('bookcorpusopen')
# ~17K books, 74M sentences
```

**Training time:** ~1 week on good GPU

### 4. Wikipedia (~6M articles)

**Why:**
- ✅ Factual knowledge
- ✅ Diverse topics
- ✅ Good entity coverage
- ✅ Clean, well-structured

**Download:**
```python
from datasets import load_dataset
ds = load_dataset('wikipedia', '20220301.en')
# 6.4M articles
```

**Use case:** Test entity registry across documents

---

## Stage 2: Large Corpora (After Medium Success)

### 5. C4 (Colossal Clean Crawled Corpus) - ~365M documents

**Why:**
- Diverse web text
- Good for real-world language
- Used to train T5

**Download:**
```python
from datasets import load_dataset
ds = load_dataset('c4', 'en', streaming=True)
# 365M web pages
```

**Training time:** Weeks to months

### 6. The Pile - 825GB

**Why:**
- Most diverse dataset available
- 22 different sources
- Good for general reasoning

**Subsets:**
- PubMed (medical)
- ArXiv (scientific)
- GitHub (code)
- Books3 (literature)
- Wikipedia
- StackExchange
- etc.

**Download:**
```python
from datasets import load_dataset
ds = load_dataset('EleutherAI/pile', streaming=True)
```

### 7. RedPajama - 1.2 trillion tokens

**Why:**
- LLaMA replication dataset
- High quality
- Diverse sources

**Download:**
```python
from datasets import load_dataset
ds = load_dataset('togethercomputer/RedPajama-Data-1T', streaming=True)
```

---

## Stage 3: Specialized Corpora

### For Reasoning:

**8. bAbI Tasks (1K-10K per task)**
```python
from datasets import load_dataset
ds = load_dataset('babi_qa')
# 20 tasks testing different reasoning types
```

**9. OpenBookQA (~6K questions)**
- Science questions requiring reasoning
- Good for testing multi-hop inference

### For Dialogue:

**10. PersonaChat (~160K utterances)**
- Multi-turn conversations
- Character consistency

**11. DailyDialog (~13K dialogues)**
- Natural conversations
- Emotion and act labels

### For Code (if interested):

**12. The Stack - 3TB code**
- 30 programming languages
- Good for program synthesis

---

## Recommended Progression

### Tonight (GPU, Hours 0-6):
```
TinyStories: 10K stories
→ Verify training works
→ Check losses, accuracy
```

### Week 1 (Days 1-7):
```
TinyStories: 100K stories
→ Full symmetric training
→ Evaluate parsing + generation
→ Test multi-hop reasoning
```

### Week 2 (Days 8-14):
```
TinyStories: 1M stories (half of full)
→ Monitor scaling behavior
→ Add entity registry (LTM)
→ Test entity persistence
```

### Week 3 (Days 15-21):
```
Full TinyStories: 2.1M stories
OR
Children's Book Test: 688K sentences
→ Compare performance
→ Integrate VQ-VAE (optional)
```

### Month 2:
```
BookCorpus OR Wikipedia
→ Scale to 10M+ sentences
→ Test knowledge accumulation
```

---

## Question 2: Is LTM (Entity Registry) Integrated?

### Current Status: ❌ NOT YET INTEGRATED

**What exists:**
- ✅ `entity_registry.py` - Complete implementation (300+ lines)
- ✅ Used in earlier experiments (`experiment_task1_with_registry.py`)
- ✅ Tested and working

**What's missing:**
- ❌ Not integrated with `symmetric_logic_network.py`
- ❌ Not used in `train_symmetric.py`
- ❌ Entity IDs are temporary (per-batch)

### Current Entity Handling (Temporary)

In `train_symmetric.py`:
```python
# Current: Entity IDs are created per-story
self.entity_to_id = {}  # Reset for each processing run
self.next_entity_id = 0

# Problem: "cat" in story 1 ≠ "cat" in story 2
# Each gets different ID, no knowledge transfer
```

### What Entity Registry Provides (LTM)

```python
# With registry: Persistent entities across stories
registry = EntityRegistry()

# Story 1: "A cat chased a mouse"
cat_id = registry.get_or_create("cat", "animal")  # ID: 1

# Story 2: "The cat was happy"  
cat_id = registry.get_or_create("cat", "animal")  # Same ID: 1
# ↑ Knowledge from story 1 persists!

# Can query:
cat = registry.get_entity(cat_id)
print(cat.properties)  # All accumulated knowledge
print(cat.relations)   # All seen relationships
```

---

## Integration Plan: Adding Entity Registry

### Phase 1: Basic Integration (1-2 hours)

**Modify `train_symmetric.py`:**

```python
from entity_registry import EntityRegistry

class TinyStoriesLogicDataset(Dataset):
    def __init__(self, ..., use_persistent_entities=True):
        # Create persistent registry
        self.entity_registry = EntityRegistry() if use_persistent_entities else None
        
        # Build vocabulary (unchanged)
        self.vocab = {"<PAD>": 0, "<UNK>": 1, ...}
        
        # NO MORE: self.entity_to_id = {} (temporary)
        # Use registry instead
    
    def _extract_propositions(self, sent):
        propositions = []
        
        for token in sent:
            if token.pos_ in ['NOUN', 'PROPN']:
                entity_text = token.lemma_.lower()
                
                # OLD: Create new ID each time
                # if entity_text not in self.entity_to_id:
                #     self.entity_to_id[entity_text] = self.next_entity_id
                #     self.next_entity_id += 1
                
                # NEW: Use persistent registry
                if self.entity_registry:
                    entity_id = self.entity_registry.get_or_create(
                        name=entity_text,
                        entity_type=token.pos_  # NOUN or PROPN
                    )
                else:
                    # Fallback to temporary IDs
                    entity_id = self._get_temp_entity_id(entity_text)
                
                # Rest unchanged
                propositions.append([entity_id, relation, value])
        
        return propositions
```

### Phase 2: Entity Embeddings (1-2 days)

**Add learnable entity embeddings:**

```python
class SymmetricLogicNetwork(nn.Module):
    def __init__(self, ..., entity_registry=None):
        super().__init__()
        
        # Connect to registry
        self.entity_registry = entity_registry
        
        # Learnable embeddings for entities
        if entity_registry:
            num_entities = entity_registry.num_entities
        else:
            num_entities = 100  # Default
        
        self.entity_embedder = nn.Embedding(num_entities, hidden_dim)
        
        # When entities are added, expand embedding layer
        self.entity_registry.on_new_entity = self.expand_embeddings
    
    def expand_embeddings(self, new_entity_id):
        """Dynamically grow embedding layer when new entity added."""
        current_size = self.entity_embedder.num_embeddings
        
        if new_entity_id >= current_size:
            # Expand embedding matrix
            new_size = max(new_entity_id + 1, current_size * 2)
            old_embeddings = self.entity_embedder.weight.data
            
            self.entity_embedder = nn.Embedding(new_size, self.hidden_dim)
            self.entity_embedder.weight.data[:current_size] = old_embeddings
            # New embeddings initialized randomly
```

### Phase 3: Knowledge Accumulation (3-5 days)

**Store learned knowledge in registry:**

```python
def train_step_with_ltm(self, text_ids, propositions):
    """Training step that updates entity registry."""
    
    # Standard training
    results = self.model(text_ids, propositions)
    loss = results['loss']
    loss.backward()
    
    # Update entity embeddings in registry
    with torch.no_grad():
        for prop in propositions:
            entity_id = prop[0].item()
            relation = prop[1].item()
            value = prop[2].item()
            
            # Get learned embedding
            entity_embedding = self.model.entity_embedder(entity_id)
            
            # Store in registry
            self.entity_registry.update_embedding(
                entity_id, 
                entity_embedding.cpu().numpy()
            )
            
            # Store relation
            self.entity_registry.add_relation(
                entity_id,
                self.vocab_inv[relation],
                value
            )
    
    return loss
```

### Phase 4: Entity Retrieval (Cross-Story)

**Use entity similarity for coreference:**

```python
# Story 1: "A cat chased a mouse"
cat_1 = registry.get_or_create("cat", "animal")

# Story 2: "The feline was happy"
# Can we link "feline" to existing "cat"?

feline_embedding = model.encode_text("feline")

# Find similar entities
similar = registry.find_similar_entities(
    feline_embedding,
    entity_type="animal",
    threshold=0.8
)

if similar and similar[0]['similarity'] > 0.8:
    # Reuse existing cat entity
    entity_id = similar[0]['id']
else:
    # Create new entity
    entity_id = registry.get_or_create("feline", "animal")
```

---

## Benefits of LTM Integration

### 1. Cross-Story Knowledge
```
Story 1: "Cats like milk"        → [cat, likes, milk]
Story 2: "The cat drank milk"    → Uses same cat ID
Story 3: "Cats are carnivores"   → Accumulates more knowledge

Query: "What do cats like?"
Answer: milk (from story 1)
```

### 2. Entity Disambiguation
```
Story 1: "Paris is beautiful" → Paris (city) ID: 42
Story 2: "Paris is a name"    → Paris (person) ID: 43

Same word, different entities - registry tracks both
```

### 3. Few-Shot Learning
```
Story 1-5: Learn about "dragons" (5 examples)
Story 6: "A dragon appeared" → Reuse learned knowledge

Without LTM: Start from scratch each time
With LTM: Accumulate knowledge progressively
```

### 4. Long-Term Memory
```
Save registry to disk:
  registry.save("entities.db")

Load later:
  registry = EntityRegistry.load("entities.db")
  # All knowledge preserved!

Can train incrementally:
  - Train on dataset A
  - Save registry
  - Train on dataset B (keeps knowledge from A)
```

---

## Implementation Priority

### Must-Have (Tonight/Tomorrow):
1. ✅ Test symmetric network on TinyStories (already done!)
2. ⏳ Scale to 10K stories on GPU (tonight at 1am)

### Should-Have (Week 1):
3. ⏳ Integrate entity registry (2-3 hours work)
4. ⏳ Test entity persistence across stories

### Nice-to-Have (Week 2-3):
5. ⏳ Dynamic embedding expansion
6. ⏳ Entity similarity search
7. ⏳ VQ-VAE integration (discrete codes)

---

## Recommended Corpus Progression

```
Week 1:  TinyStories 100K        (test scaling)
Week 2:  TinyStories 1M          (+ entity registry)
Week 3:  TinyStories Full 2.1M   (+ VQ-VAE optional)
Week 4:  Children's Book Test    (test new domain)
Week 5:  Wikipedia (subset)      (factual knowledge)
Week 6:  BookCorpus              (complex narratives)
Month 2: C4 or The Pile          (if still scaling well)
```

**Each step validates:**
- Scaling properties
- Knowledge accumulation
- Cross-domain transfer
- Long-term memory

---

## Conclusion

### Larger Corpora:
- ✅ Many options available (TinyStories → Wikipedia → BookCorpus → C4)
- ✅ Use streaming for datasets > 1GB
- ✅ Progress gradually (validate each step)

### LTM Integration:
- ❌ Not yet integrated (but code exists!)
- ✅ Easy to add (2-3 hours work)
- ✅ Should add after GPU test succeeds
- ✅ Critical for knowledge accumulation

**Recommendation:**
1. **Tonight**: Test on 10K TinyStories (GPU)
2. **Tomorrow**: If successful, integrate entity registry
3. **Week 1**: Scale to 100K-1M TinyStories with LTM
4. **Week 2+**: Try other corpora

**The architecture is ready to scale!** 🚀

