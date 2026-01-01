# Architecture Insights: Symmetric Logic Network with Implicit Graph Structure

**Date:** January 1, 2026  
**Status:** Design Complete, Ready for Prototype

---

## Executive Summary

We have converged on an elegant architecture that combines:
1. **Implicit graph structure** via indexed entities (no explicit tree/graph data structures)
2. **Symmetric bidirectional mapping** between Natural Language and Logic (NL ↔ Logic)
3. **Soft differentiable logic rules** with O(N²) computation but O(R) parameters
4. **Hybrid template-based + learned parsing** for robustness to informal language

This architecture is theoretically principled, computationally efficient, and practically implementable.

---

## Key Insight 1: Implicit Graph Structure via Indexed Entities

### The Problem We Solved

**Original concern:** How to maintain tree/graph structure in working memory while keeping logic rules differentiable?

**Solution:** Don't use explicit graph structures. Use entity IDs as implicit links!

### The Architecture

```python
# Working Memory: Flat list of propositions
WM = [
    [cat_1, type, cat],      # Index 0
    [cat_1, on, mat_1],      # Index 1  ← Links cat_1 to mat_1
    [mat_1, type, mat],      # Index 2
    [mat_1, on, floor_1],    # Index 3  ← Links mat_1 to floor_1
    [floor_1, type, floor]   # Index 4
]

# Entity IDs (cat_1, mat_1, floor_1) ARE the graph links!
# No separate graph data structure needed
```

### How It Works

**Graph traversal becomes soft pattern matching:**

```python
# Traditional graph: "What is cat_1 on?"
target = graph.edges[cat_1]["on"]  # Hard lookup (not differentiable)

# Our approach: "What is cat_1 on?"
for i, prop in enumerate(WM):
    match_score = (
        fuzzy_match(prop[0], cat_1) *     # Entity match
        fuzzy_match(prop[1], "on")        # Relation match
    )
    if match_score > threshold:
        target = prop[2]  # Soft result (differentiable!)
```

**Multi-hop reasoning example:**

```python
# Rule: "X on Y, Y on Z → X above Z"
for i in range(N):
    for j in range(N):
        # Soft match both premises
        match_1 = match(WM[i], [?X, on, ?Y])
        match_2 = match(WM[j], [?Y, on, ?Z])
        
        # Check variable binding: Y must match
        if WM[i][2] == WM[j][0]:  # Entity ID comparison
            confidence = match_1 * match_2
            infer: [WM[i][0], above, WM[j][2]]

# Example:
# [cat_1, on, mat_1] + [mat_1, on, floor_1] → [cat_1, above, floor_1]
```

### Computational Analysis

| Approach | Parameters | Computation | Memory |
|----------|-----------|-------------|---------|
| **Explicit graph + GNN** | O(V×H) | O(E×H) | O(V+E) |
| **Our indexed entities + soft logic** | O(R) | O(N²) | O(N) |
| **Attention (transformer)** | O(V×H) | O(N²×H) | O(N²) |

Where:
- V = vocabulary size
- H = hidden dimension  
- E = number of edges
- R = number of rules (small, ~10-100)
- N = WM size

**Key advantages:**
- ✅ **Minimal parameters:** O(R) - just rule templates
- ✅ **Differentiable:** All operations are soft/continuous
- ✅ **Inevitable O(N²):** Graph reasoning requires checking all pairs
- ✅ **Simpler:** No graph data structures, just propositions

### Why O(N²) is Acceptable

**For our use case (TinyStories):**
- Average sentence: 10-15 words
- N² = 225 operations per sentence
- ~0.1ms per operation
- **Total: ~22.5ms per sentence** (acceptable!)

**Comparison to transformers:**
- Transformers also O(N²) for self-attention
- Our approach has FEWER parameters (rules vs. attention weights)

---

## Key Insight 2: Symmetric NL ↔ Logic (Bidirectional Architecture)

### The Symmetry

```
Forward (parsing):
"The cat chases the mouse" → [cat_1, chases, mouse_1]

Backward (generation):
[cat_1, chases, mouse_1] → "The cat chases the mouse"

These are INVERSE FUNCTIONS!
```

### Why This Matters

**Traditional approach (wasteful):**
```python
parser = NLtoLogic()      # 100M parameters
generator = LogicToNL()   # 100M parameters
# Total: 200M parameters
```

**Our approach (efficient):**
```python
bidirectional = SymmetricNLLogic()  # 100M parameters
# Parse: NL → embedding → Logic
# Generate: Logic → embedding → NL
# Total: 100M parameters (50% reduction!)
```

### Three Ways to Exploit Symmetry

#### **1. Cycle Consistency Training**

```python
# Forward cycle: text → logic → text'
loss_forward = distance(text, reconstruct_from_logic)

# Backward cycle: logic → text → logic'
loss_backward = distance(logic, reconstruct_from_text)

# Total loss
loss = supervised_loss + λ * (loss_forward + loss_backward)
```

**Benefits:**
- Can train on unlabeled text (cycle consistency)
- Mutual regularization (both directions must agree)
- Better generalization

#### **2. Reversible Neural Networks**

```python
class ReversibleTransform(nn.Module):
    def forward(self, x, reverse=False):
        if not reverse:
            return self.parse(x)  # NL → Logic
        else:
            return self.generate(x)  # Logic → NL
    
    # Same network, different directions!
```

**Benefits:**
- Theoretically elegant (guaranteed bijection)
- Perfect symmetry

#### **3. Reversible Logic Rules (Our Choice)**

```python
class ReversibleLogicRule(nn.Module):
    def __init__(self):
        # Single pattern works both ways!
        self.pattern = nn.Parameter(torch.randn(3))
    
    def parse_direction(self, text_features):
        """NL → Logic: Match pattern in text"""
        return similarity(text_features, self.pattern)
    
    def generate_direction(self, logic_pattern):
        """Logic → NL: Generate text from pattern"""
        return self.pattern_to_text(self.pattern)
    
    def bidirectional_loss(self, text, logic):
        """Train both directions simultaneously"""
        loss_parse = self.parse_direction(text)
        loss_generate = self.generate_direction(logic)
        return loss_parse + loss_generate
```

**Benefits:**
- Logic rules naturally work both ways
- Simple to implement
- Interpretable (can see what rules do in both directions)

### The Complete Symmetric Architecture

```python
class SymmetricLogicNetwork(nn.Module):
    def __init__(self):
        # Shared logic rules (bidirectional!)
        self.logic_rules = LogicNetwork(...)
        
        # Symmetric encoders
        self.nl_encoder = TextEncoder(...)
        self.logic_encoder = PropositionEncoder(...)
        
        # Shared latent space (NL and Logic map to same space!)
        
        # Symmetric decoders
        self.nl_decoder = TextDecoder(...)
        self.logic_decoder = PropositionDecoder(...)
    
    def parse(self, text):
        """NL → Logic"""
        text_emb = self.nl_encoder(text)
        concepts = self.logic_rules(text_emb)
        logic = self.logic_decoder(concepts)
        return logic
    
    def generate(self, logic):
        """Logic → NL"""
        logic_emb = self.logic_encoder(logic)
        concepts = self.logic_rules(logic_emb)
        text = self.nl_decoder(concepts)
        return text
    
    def train_step(self, text, logic):
        """Symmetric training"""
        # Forward cycle
        pred_logic = self.parse(text)
        recon_text = self.generate(pred_logic)
        
        # Backward cycle
        pred_text = self.generate(logic)
        recon_logic = self.parse(pred_text)
        
        # Symmetric loss
        loss = (
            F.cross_entropy(pred_logic, logic) +
            F.cross_entropy(pred_text, text) +
            F.mse_loss(recon_text, text) +
            F.mse_loss(recon_logic, logic)
        )
        return loss
```

---

## Key Insight 3: Hybrid Template + Learning for Robustness

### The Problem

**Languages have too many patterns for fixed templates:**
- Formal: ~50-100 core patterns (Chomskyan linguistics)
- Informal: Thousands of constructions (slang, memes, etc.)
- Novel: New patterns emerge constantly

### The Solution

**Hybrid approach: Templates + Learning**

```python
class HybridParser(nn.Module):
    def __init__(self):
        # Core templates for common formal patterns (~30)
        self.core_templates = [
            SVO_Template(),           # Subject-Verb-Object
            NP_Template(),            # Noun phrases
            VP_Template(),            # Verb phrases
            # ... ~30 linguistic templates
        ]
        
        # Learned templates for novel patterns
        self.learned_templates = LearnableTemplateBank(slots=100)
        
        # Seq2seq fallback for everything else
        self.fallback_parser = LSTM_TreeDecoder()
    
    def parse(self, sentence):
        # Try core templates first (fast, interpretable)
        for template in self.core_templates:
            if template.matches(sentence, threshold=0.8):
                return template.instantiate(sentence)
        
        # Try learned templates
        learned_match = self.learned_templates.match(sentence)
        if learned_match.confidence > 0.7:
            return learned_match.tree
        
        # Fallback to neural parser
        return self.fallback_parser(sentence)
```

### Coverage Analysis

| Templates | Coverage | Interpretability | Training Data Needed |
|-----------|----------|------------------|---------------------|
| 0 (pure seq2seq) | 100% | Low | 100K+ sentences |
| 20-30 (minimal) | ~60% | High | 10K sentences |
| 50 (standard) | ~80% | High | 5K sentences |
| 100 (extensive) | ~90% | Medium | 1K sentences |
| ∞ (learned) | 100% | Low | 10K+ sentences |

**Our strategy for TinyStories:**
- Start with 20-30 core templates
- Add 100 learned pattern slots
- Seq2seq fallback for remainder
- **Target: 85% template coverage, 15% learned**

### Example: Handling Informal Language

```python
# Input: "cat be like 😂" (informal, no standard template)

# Step 1: Core templates fail
SVO_Template.match("cat be like 😂") → 0.3 confidence (too low)

# Step 2: Learned template succeeds
# Model saw similar patterns in training:
#   "cat be like 😂"
#   "dog be like 🤔"
#   "they be like nope"
LearnedTemplate_47.match("cat be like 😂") → 0.9 confidence!

# Generates parse tree:
#     be
#    /  \
#  cat  like
#         |
#        😂
```

---

## Architectural Summary

### Complete System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Symmetric Logic Network                   │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  NL Text ←→ Encoder ←→ [Shared Logic Rules] ←→ Decoder ←→ Logic Props │
│                              ↓                                │
│                    Implicit Graph Structure                   │
│                    (Entity-indexed WM)                        │
│                              ↓                                │
│                    Soft Pattern Matching                      │
│                    (O(N²) computation)                        │
│                                                               │
└─────────────────────────────────────────────────────────────┘

Components:
1. Working Memory: Flat propositions with entity IDs
2. Logic Rules: Soft, differentiable, bidirectional
3. Parser/Generator: Symmetric (shared weights)
4. Templates: Hybrid (core + learned)
```

### Key Properties

**1. Simplicity:**
- ✅ No explicit tree/graph structures
- ✅ Entity IDs create implicit links
- ✅ Standard neural network operations

**2. Efficiency:**
- ✅ O(R) parameters (minimal)
- ✅ O(N²) computation (inevitable for graph reasoning)
- ✅ Shared weights (50% parameter reduction)

**3. Differentiability:**
- ✅ Soft matching (continuous)
- ✅ Gumbel-softmax for discrete choices
- ✅ End-to-end gradient flow

**4. Robustness:**
- ✅ Templates for common patterns
- ✅ Learning for novel patterns
- ✅ Graceful degradation

**5. Theoretical Elegance:**
- ✅ NL ↔ Logic as inverse functions
- ✅ Cycle consistency
- ✅ Mutual regularization

---

## Prototype Implementation Plan

### Phase 1: Core Architecture (Week 1)

**Files to create:**
1. `symmetric_logic_network.py` - Main architecture
2. `implicit_graph_wm.py` - Working memory with entity indexing
3. `reversible_logic_rules.py` - Bidirectional logic rules

**Minimal implementation:**
```python
# 1. Working Memory with implicit graph
class ImplicitGraphWM:
    def __init__(self):
        self.propositions = []  # Flat list
        self.entity_index = {}  # entity_id → proposition indices
    
    def add_proposition(self, prop):
        idx = len(self.propositions)
        self.propositions.append(prop)
        
        # Index by entity
        entity = prop[0]
        if entity not in self.entity_index:
            self.entity_index[entity] = []
        self.entity_index[entity].append(idx)
    
    def find_links(self, entity_id, relation):
        """Soft graph traversal"""
        matches = []
        for idx in self.entity_index.get(entity_id, []):
            prop = self.propositions[idx]
            if fuzzy_match(prop[1], relation) > 0.8:
                matches.append(prop[2])
        return matches

# 2. Reversible Logic Rule
class ReversibleLogicRule(nn.Module):
    def __init__(self, prop_dim=3, hidden_dim=64):
        super().__init__()
        self.pattern = nn.Parameter(torch.randn(prop_dim, hidden_dim))
        
    def parse_direction(self, text_encoding):
        """NL → Logic"""
        return torch.matmul(text_encoding, self.pattern.T)
    
    def generate_direction(self, logic_encoding):
        """Logic → NL"""
        return torch.matmul(logic_encoding, self.pattern)
    
    def forward(self, x, direction='parse'):
        if direction == 'parse':
            return self.parse_direction(x)
        else:
            return self.generate_direction(x)

# 3. Symmetric Network
class SymmetricLogicNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.logic_rules = nn.ModuleList([
            ReversibleLogicRule() for _ in range(num_rules)
        ])
        self.nl_encoder = nn.LSTM(vocab_size, hidden_dim)
        self.logic_encoder = nn.Linear(prop_dim, hidden_dim)
        self.nl_decoder = nn.LSTM(hidden_dim, vocab_size)
        self.logic_decoder = nn.Linear(hidden_dim, prop_dim)
    
    def parse(self, text):
        enc = self.nl_encoder(text)
        concepts = sum(rule(enc, 'parse') for rule in self.logic_rules)
        return self.logic_decoder(concepts)
    
    def generate(self, logic):
        enc = self.logic_encoder(logic)
        concepts = sum(rule(enc, 'generate') for rule in self.logic_rules)
        return self.nl_decoder(concepts)
```

### Phase 2: Training Infrastructure (Week 1)

**Files to create:**
1. `train_symmetric.py` - Training loop with cycle consistency
2. `template_library.py` - Core templates for TinyStories

**Training setup:**
```python
def train_epoch(model, data):
    for text, logic in data:
        # Supervised losses
        pred_logic = model.parse(text)
        pred_text = model.generate(logic)
        
        loss_parse = F.cross_entropy(pred_logic, logic)
        loss_generate = F.cross_entropy(pred_text, text)
        
        # Cycle consistency losses
        recon_text = model.generate(pred_logic)
        recon_logic = model.parse(pred_text)
        
        loss_cycle_forward = F.mse_loss(recon_text, text)
        loss_cycle_backward = F.mse_loss(recon_logic, logic)
        
        # Total symmetric loss
        loss = (loss_parse + loss_generate + 
                λ * (loss_cycle_forward + loss_cycle_backward))
        
        loss.backward()
        optimizer.step()
```

### Phase 3: Integration with Existing Code (Week 2)

**Connect with:**
1. `learnable_parsing.py` - Use as NL encoder
2. `hierarchical_logic_network.py` - Use logic rules
3. `entity_registry.py` - Entity ID management
4. `vector_quantization.py` - Optional proposition encoding

### Phase 4: Evaluation (Week 2)

**Metrics:**
1. Parsing accuracy (NL → Logic)
2. Generation quality (Logic → NL)
3. Cycle consistency (text → logic → text')
4. Multi-hop reasoning (using implicit graph)

**Test cases:**
```python
# 1. Single proposition
text = "The cat chases the mouse"
logic = [cat_1, chases, mouse_1]

# 2. Multi-hop reasoning
text1 = "The cat is on the mat"
text2 = "The mat is on the floor"
inferred = [cat_1, above, floor_1]  # Should infer this!

# 3. Cycle consistency
text → logic → text'
Assert: similarity(text, text') > 0.9

# 4. Informal language
text = "cat be like 😂"
# Should handle gracefully
```

---

## Success Criteria

**Prototype is successful if:**

1. ✅ **Parsing:** Achieves >90% accuracy on TinyStories
2. ✅ **Generation:** Produces grammatical sentences from logic
3. ✅ **Cycle consistency:** text → logic → text' with >85% similarity
4. ✅ **Multi-hop:** Can infer [A, above, C] from [A, on, B] + [B, on, C]
5. ✅ **Efficiency:** Trains in <5 minutes per epoch on CPU
6. ✅ **Robustness:** Handles informal patterns gracefully

---

## Open Questions for Prototype

1. **Optimal λ for cycle consistency loss?**
   - Start with λ = 0.5, tune based on validation

2. **How many core templates?**
   - Start with 20-30, expand if coverage is low

3. **Entity ID representation?**
   - Option A: Integer IDs (simple)
   - Option B: Learned embeddings (flexible)
   - **Recommendation:** Start with integer IDs

4. **Soft matching threshold?**
   - For template matching: 0.8
   - For rule firing: 0.7
   - Tune based on precision/recall

5. **VQ integration?**
   - Phase 1: Skip (use continuous propositions)
   - Phase 2: Add VQ layer if needed
   - Trade-off: Discreteness vs. differentiability

---

## Next Steps

**Immediate (today):**
1. ✅ Create architecture summary (this document)
2. ⏳ Implement `symmetric_logic_network.py`
3. ⏳ Implement `implicit_graph_wm.py`
4. ⏳ Create minimal training script

**This week:**
1. Train on 500 TinyStories sentences
2. Evaluate parsing + generation
3. Test cycle consistency
4. Measure multi-hop reasoning

**Next week:**
1. Scale to 10K sentences
2. Add template library
3. Handle informal patterns
4. Full integration with existing code

---

## Conclusion

We have a theoretically sound, computationally efficient, and practically implementable architecture:

- **Implicit graphs** via entity IDs (no explicit structures)
- **Symmetric learning** (NL ↔ Logic as inverse functions)
- **Soft differentiable rules** (O(N²) computation, O(R) parameters)
- **Hybrid templates** (core + learned + fallback)

This architecture solves the key challenges:
1. ✅ Differentiability (soft operations)
2. ✅ Scalability (minimal parameters)
3. ✅ Robustness (handles novel patterns)
4. ✅ Elegance (theoretical foundation)

**Ready to implement prototype!**
