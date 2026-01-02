# Testing Strategy: From Simple to Complex

**Date:** 2026-01-02  
**Goal:** Validate convergence system with increasing complexity

---

## The Problem with TinyStories + AR

From previous session (`NL_PARSING_SESSION_SUMMARY.md`):
- ❌ 30 epochs: 0% accuracy, loss ~300
- ❌ Increased capacity: Still 0%
- ❌ Pattern priming: Still 0%

**Root Causes:**
1. **AR is too strict** - Forces exact token prediction
2. **TinyStories is too open-ended** - Infinite valid continuations
3. **No credit for partial understanding** - All-or-nothing evaluation
4. **Cycle consistency assumption wrong** - Many phrasings → one meaning (not invertible)

---

## Recommended Strategy: 3-Stage Validation

### Stage 1: Simple QA Task (1-2 days) ⭐ START HERE

**Task:** bAbI Task 1 (Single Supporting Fact)

**Example:**
```
Input:  "John went to the kitchen. Mary went to the bedroom."
Query:  "Where is John?"
Answer: "kitchen"
```

**Why This First:**
- ✅ Clear right/wrong answers
- ✅ Tests extraction + inference
- ✅ Small dataset (can iterate quickly)
- ✅ Validates symbolic components work

**Implementation:**
```python
# test_babi_task1.py
from convergence_system import ConvergenceSystem

system = ConvergenceSystem()

# Process story
story = [
    "John went to the kitchen.",
    "Mary went to the bedroom."
]

for sentence in story:
    result = system.forward(sentence)
    # Add facts to working memory

# Answer query
query = "Where is John?"
answer = system.query(query)  # Should find: [john, location, kitchen]

assert answer == "kitchen"
```

**Expected Results:**
- Day 1: 40-60% accuracy (just extraction)
- Day 2: 70-80% accuracy (after adding location rules)

---

### Stage 2: Add Neural Logic Component (2-3 days)

**Goal:** Integrate `neural_logic_core.py` with symbolic system

**Architecture:**
```
Symbolic Layer (Davidsonian + Forward Chaining)
    ↓
    Facts: [entity, relation, value]
    ↓
Neural Logic Layer (Learnable Rules)
    ↓
    Predictions + Confidence
    ↓
Weighted Combination (Learnable)
    ↓
    Final Answer
```

**Implementation:**
```python
class HybridSystem(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Symbolic components
        self.symbolic = ConvergenceSystem()
        
        # Neural logic rules
        self.neural_rules = nn.ModuleList([
            LogicRule(num_premises=2, var_slots=3, 
                     prop_length=768, output_dim=768)
            for _ in range(20)  # 20 learnable rules
        ])
        
        # Combination weights
        self.symbolic_weight = nn.Parameter(torch.tensor(1.0))
        self.neural_weight = nn.Parameter(torch.tensor(0.1))  # Start small
    
    def forward(self, sentence):
        # Get symbolic facts
        symbolic_facts = self.symbolic.forward(sentence)
        
        # Apply neural rules
        neural_predictions = self.apply_neural_rules(symbolic_facts)
        
        # Combine
        combined = (self.symbolic_weight * symbolic_facts + 
                   self.neural_weight * neural_predictions)
        
        return combined
```

**Why This Order:**
- ✅ Symbolic provides structure (fast convergence)
- ✅ Neural adds flexibility (handles exceptions)
- ✅ Small neural weight prevents disruption
- ✅ Can ablate components to verify contribution

**Expected Results:**
- Symbolic alone: 70-80%
- Hybrid: 85-90%
- **Proves neural component adds value!**

---

### Stage 3: Scale to TinyStories (1 week)

**Critical Change: NOT Auto-Regressive!**

**Instead: Semantic Event Prediction**

**Task:**
```
Input:  "Mary went to the kitchen."
Predict: Next likely event/state
Gold:   [mary, location, kitchen], [mary, state, standing], ...
```

**Why This Works:**
- ✅ Semantic matching (many valid next events)
- ✅ Partial credit (match some facts = partial score)
- ✅ Tests understanding (not memorization)
- ✅ Differentiable (cosine similarity between fact sets)

**Implementation:**
```python
def train_on_tinystories(system, data_loader):
    optimizer = optim.Adam(system.parameters(), lr=0.001)
    
    for epoch in range(50):
        for story in data_loader:
            # Process each sentence
            facts_timeline = []
            for sentence in story:
                result = system.forward(sentence)
                facts_timeline.append(result['inferred'])
            
            # Predict next facts from context
            for t in range(len(facts_timeline) - 1):
                context_facts = facts_timeline[:t+1]
                predicted_facts = system.predict_next(context_facts)
                gold_facts = facts_timeline[t+1]
                
                # Semantic matching loss (not token matching!)
                loss = semantic_distance(predicted_facts, gold_facts)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
```

**Loss Function (Semantic):**
```python
def semantic_distance(predicted, gold):
    """
    Compare fact sets semantically.
    Partial matches get partial credit!
    """
    # Embed facts as vectors
    pred_embeddings = embed_facts(predicted)  # [N, D]
    gold_embeddings = embed_facts(gold)       # [M, D]
    
    # Compute bipartite matching (Hungarian algorithm)
    similarity_matrix = cosine_similarity(pred_embeddings, gold_embeddings)
    matching_score = optimal_bipartite_matching(similarity_matrix)
    
    # Higher score = more overlap
    loss = 1.0 - matching_score
    return loss
```

**Expected Results:**
- Week 1: 40-50% semantic overlap
- Week 2: 60-70% semantic overlap
- Week 3: 75-80% semantic overlap

---

## Why This Strategy Works

### Compared to Your Original Approach:

| Aspect | Original (AR) | New (Semantic) |
|--------|---------------|----------------|
| Objective | Predict exact tokens | Predict semantic facts |
| Evaluation | All-or-nothing | Partial credit |
| Convergence | Never (R << R_K) | Gradual (structure helps) |
| Credit assignment | Opaque | Clear |
| Interpretability | Low | High |

### Key Insights:

1. **Start simple to validate** - bAbI proves components work
2. **Add complexity gradually** - Symbolic → Hybrid → Neural-heavy
3. **Use semantic objectives** - Match meaning, not tokens
4. **Leverage priors** - Davidsonian meta-rules accelerate learning
5. **Measure what matters** - Understanding, not memorization

---

## Recommended Timeline

### This Week:
- **Day 1-2:** Implement bAbI test (Stage 1)
- **Day 3-4:** Add neural logic hybrid (Stage 2)
- **Day 5-7:** Initial TinyStories experiments (Stage 3)

### Next Week:
- Tune loss function (semantic matching)
- Scale up neural rules (20 → 100)
- Measure convergence speed

### Week 3:
- Ablation studies (symbolic vs hybrid)
- Domain transfer tests
- Write up results

---

## Alternative: If You Want AR Still...

**Compromise Approach: Constrained AR**

Instead of free-form generation:
1. Extract facts from sentence
2. Convert facts → template
3. AR fills in template slots

**Example:**
```
Extracted: [e1, type, go], [e1, agent, mary], [e1, location, kitchen]
Template:  "<AGENT> went to the <LOCATION>."
AR fills:  "Mary" + "kitchen"
```

This combines:
- ✅ Symbolic structure (extraction)
- ✅ Neural flexibility (AR on slots)
- ✅ Easier learning (constrained output space)

---

## Recommendation

**START WITH STAGE 1 (bAbI)** - It's the fastest way to validate your system works.

Once that's working (70-80% accuracy in 2 days), you'll have:
- ✅ Confidence the architecture is sound
- ✅ Fast feedback loop for debugging
- ✅ Clear path to scaling up

**THEN** add neural logic and scale to TinyStories with semantic objectives.

---

## What Do You Think?

**Option A:** Start with bAbI (my recommendation)  
**Option B:** Jump to TinyStories + semantic objectives  
**Option C:** Try constrained AR approach  
**Option D:** Something else entirely  

Let me know and I'll implement whichever you prefer! 🚀
