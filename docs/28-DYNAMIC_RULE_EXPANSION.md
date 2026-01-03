# Dynamic Rule Expansion: Growing Network Capacity

## Your Observation: Too Few Rules = Poor Convergence

### The Problem

```
Scenario: 16 rules trying to learn 50+ linguistic patterns

Rule 1: Tries to handle SVO, passive, questions (too much!)
Rule 2: Tries to handle adjectives, adverbs, negation (overloaded!)
...

Result: Each rule is "jack of all trades, master of none"
→ Poor convergence, low accuracy
```

**Analogy:** 
- 16 students trying to learn 50 subjects each
- Better: 50 students, each specializing in 1-2 subjects

### Evidence in Our Results

```
16 rules, 1000 stories (~30 different patterns)
Result: 2% accuracy

Why? Rules are competing/interfering with each other:
- Rule 1 learns "SVO" in epoch 5
- Rule 1 forgets "SVO" when learning "passive" in epoch 10
- Catastrophic forgetting!
```

---

## Solution 1: Start with More Rules

### Current
```python
model = SymmetricLogicNetwork(
    num_rules=16  # Too few!
)
```

### Better
```python
model = SymmetricLogicNetwork(
    num_rules=50  # One per major pattern
)
```

**Trade-off:**
- ✓ More capacity, less interference
- ✗ More parameters, slower training
- ✗ Risk of overfitting with small data

---

## Solution 2: Dynamic Rule Expansion (Your Question!)

### Can We Add Rules Without Retraining?

**YES!** This is called **progressive neural networks** or **network expansion**.

### Current Architecture
```python
class LogicNetwork(nn.Module):
    def __init__(self, num_rules=16):
        self.rules = nn.ModuleList([
            LogicRule() for _ in range(num_rules)
        ])
        # Fixed size! Can't grow.
```

### Expandable Architecture
```python
class ExpandableLogicNetwork(nn.Module):
    def __init__(self, initial_rules=16):
        self.rules = nn.ModuleList([
            LogicRule() for _ in range(initial_rules)
        ])
        self.num_active_rules = initial_rules
    
    def add_rule(self, initialize_fn=None):
        """
        Dynamically add a new rule without retraining.
        
        Args:
            initialize_fn: Optional function to initialize the new rule
                          (e.g., with a linguistic pattern)
        """
        new_rule = LogicRule()
        
        if initialize_fn:
            # Initialize with specific pattern (e.g., SVO)
            initialize_fn(new_rule)
        else:
            # Random initialization
            pass
        
        # Add to network
        self.rules.append(new_rule)
        self.num_active_rules += 1
        
        # Old rules keep their trained weights!
        # Only new rule needs training
        return new_rule
    
    def freeze_rules(self, indices):
        """Freeze certain rules (don't update during training)."""
        for i in indices:
            for param in self.rules[i].parameters():
                param.requires_grad = False
```

### Usage: Progressive Training

```python
# Stage 1: Start small
model = ExpandableLogicNetwork(initial_rules=10)
train(model, epochs=20)  # Train 10 rules
print("Accuracy:", evaluate(model))  # Say, 15%

# Stage 2: Add more rules (NO retraining from scratch!)
model.freeze_rules(range(10))  # Keep first 10 rules as learned
model.add_rule(initialize_with_svo_pattern)
model.add_rule(initialize_with_passive_pattern)
model.add_rule(initialize_with_negation_pattern)
# Now have 13 rules total

train(model, epochs=10)  # Only train NEW 3 rules (fast!)
print("Accuracy:", evaluate(model))  # Say, 25% (improved!)

# Stage 3: Keep expanding as needed
model.add_rule(initialize_with_question_pattern)
train(model, epochs=5)
print("Accuracy:", evaluate(model))  # Say, 35%
```

**Key advantage:** Previous learning is preserved!

---

## Solution 3: Rule Specialization Detection

### Auto-detect when rules are overloaded

```python
def analyze_rule_usage(model, dataloader):
    """
    Check which patterns each rule is handling.
    """
    rule_activations = {i: [] for i in range(model.num_active_rules)}
    
    for batch in dataloader:
        activations = model.get_rule_activations(batch)
        
        for rule_idx, activation in enumerate(activations):
            # Record which inputs activated this rule
            rule_activations[rule_idx].extend(activation)
    
    # Analyze diversity
    for rule_idx, patterns in rule_activations.items():
        diversity = compute_pattern_diversity(patterns)
        
        if diversity > THRESHOLD:
            print(f"⚠️  Rule {rule_idx} is handling too many patterns!")
            print(f"   Diversity: {diversity:.2f}")
            print(f"   Suggestion: Add more rules for specialization")

# Usage
analyze_rule_usage(model, val_loader)
# Output:
# ⚠️  Rule 3 is handling too many patterns!
#    Diversity: 0.85 (high = overloaded)
#    Suggestion: Add more rules for specialization
```

### Auto-expand based on analysis

```python
def auto_expand_network(model, dataloader, target_diversity=0.3):
    """
    Automatically add rules when existing ones are overloaded.
    """
    usage = analyze_rule_usage(model, dataloader)
    
    for rule_idx, diversity in usage.items():
        if diversity > target_diversity:
            # This rule is overloaded, split it!
            
            # Analyze what patterns this rule handles
            patterns = identify_patterns_for_rule(model, rule_idx, dataloader)
            
            # Group patterns into clusters
            clusters = cluster_patterns(patterns, num_clusters=2)
            
            # Create new rule for second cluster
            new_rule = model.add_rule()
            
            # Initialize new rule to handle cluster 2
            initialize_rule_for_patterns(new_rule, clusters[1])
            
            # Original rule now specialized for cluster 1
            specialize_rule_for_patterns(model.rules[rule_idx], clusters[0])
            
            print(f"✓ Expanded: Split rule {rule_idx} into two specialized rules")
```

---

## Solution 4: Mixture of Experts (MoE)

### Soft rule assignment (current)
```python
# All rules compute outputs
outputs = [rule(input) for rule in self.rules]

# Weighted combination (all contribute)
final_output = weighted_sum(outputs, weights)
```

**Problem:** All rules active for all inputs (inefficient, interference)

### Hard rule assignment (MoE)
```python
# Router decides which rule to use
router_probs = self.router(input)  # (batch, num_rules)
selected_rule = argmax(router_probs)  # Pick ONE rule

# Only selected rule computes output
output = self.rules[selected_rule](input)
```

**Benefits:**
- Each rule specializes (no interference)
- Sparse computation (only 1 rule active)
- Can have 100s of rules (only use few at a time)

### Implementation

```python
class MixtureOfExpertsLogicNetwork(nn.Module):
    def __init__(self, num_rules=50):
        super().__init__()
        
        # Many rules (50-100+)
        self.rules = nn.ModuleList([
            LogicRule() for _ in range(num_rules)
        ])
        
        # Router: decides which rule to use
        self.router = nn.Linear(hidden_dim, num_rules)
    
    def forward(self, input):
        # Encode input
        encoded = self.encoder(input)
        
        # Router decides which rule to use
        router_logits = self.router(encoded)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Top-k rules (e.g., k=2)
        top_k_probs, top_k_indices = torch.topk(router_probs, k=2, dim=-1)
        
        # Only compute selected rules
        outputs = []
        for i, idx in enumerate(top_k_indices):
            rule_output = self.rules[idx](encoded)
            weighted_output = top_k_probs[i] * rule_output
            outputs.append(weighted_output)
        
        # Combine top-k outputs
        final_output = sum(outputs)
        
        return final_output
```

**Advantage:** Can have 100 rules but only use 2 at a time!
- No interference between rules
- Each rule specializes
- Efficient computation

---

## Practical Implementation Plan

### Phase 1: Increase Initial Capacity (Tonight)

```python
# Current
model = SymmetricLogicNetwork(num_rules=16)  # Too few!

# Better
model = SymmetricLogicNetwork(num_rules=50)  # More capacity

# Prime first 30 with patterns, leave 20 for learning
for i in range(30):
    initialize_rule(model.rules[i], PATTERNS[i])
```

**Effort:** 5 minutes (just change num_rules)  
**Expected:** Better convergence immediately

### Phase 2: Implement Dynamic Expansion (This Week)

```python
class ExpandableLogicNetwork(SymmetricLogicNetwork):
    def add_rule(self, pattern=None):
        """Add rule without retraining from scratch."""
        new_rule = LogicRule()
        if pattern:
            initialize_rule(new_rule, pattern)
        self.rules.append(new_rule)
        return new_rule
    
    def freeze_rules(self, indices):
        """Freeze trained rules."""
        for i in indices:
            for p in self.rules[i].parameters():
                p.requires_grad = False
```

**Effort:** 1 day  
**Benefit:** Incremental learning without retraining

### Phase 3: Add MoE Router (Next Week)

```python
class MoELogicNetwork(ExpandableLogicNetwork):
    def __init__(self, num_rules=50):
        super().__init__(num_rules)
        self.router = nn.Linear(hidden_dim, num_rules)
    
    def forward(self, input):
        # Route to specialized rules
        router_probs = F.softmax(self.router(input), dim=-1)
        top_k = torch.topk(router_probs, k=2)
        
        # Only use top-2 rules
        outputs = [self.rules[i](input) * prob 
                  for i, prob in zip(top_k.indices, top_k.values)]
        return sum(outputs)
```

**Effort:** 2-3 days  
**Benefit:** 100+ rules, no interference

---

## Quick Test Implementation (Tonight)

Let me implement:
1. ✓ Increase num_rules to 50
2. ✓ Prime first 10 with patterns
3. ✓ Test if accuracy improves

```python
def create_primed_network():
    """Create network with 50 rules, first 10 primed."""
    model = SymmetricLogicNetwork(num_rules=50)  # More capacity!
    
    # Prime first 10 rules with common patterns
    patterns = [
        ('SVO', ['NOUN', 'VERB', 'NOUN']),
        ('SV_PP', ['NOUN', 'VERB', 'ADP', 'NOUN']),
        ('copula_ADJ', ['NOUN', 'AUX', 'ADJ']),
        ('ADJ_NOUN', ['ADJ', 'NOUN']),
        ('NOUN_PP', ['NOUN', 'ADP', 'NOUN']),
        ('passive', ['NOUN', 'AUX', 'VERB']),
        ('progressive', ['AUX', 'VERB']),
        ('perfect', ['AUX', 'VERB']),
        ('negation', ['AUX', 'ADV', 'VERB']),
        ('possessive', ['NOUN', 'PART', 'NOUN']),
    ]
    
    for i, (name, pos_pattern) in enumerate(patterns):
        initialize_rule_with_pattern(model.rules[i], name, pos_pattern)
    
    # Remaining 40 rules: random init (for learning novel patterns)
    
    return model
```

---

## Summary: Addressing Your Concerns

### Your Question 1: Too Few Rules → Poor Convergence?

**Answer: YES, absolutely!**

**Solution:**
- Increase from 16 → 50 rules (immediate)
- Use MoE for 100+ rules without interference (later)

### Your Question 2: Expand rules without retraining?

**Answer: YES, we can!**

**Implementation:**
```python
# Train initial network
model = ExpandableLogicNetwork(num_rules=10)
train(model, epochs=20)

# Add more rules later (keeps old weights!)
model.freeze_rules(range(10))
model.add_rule(pattern='SVO')
model.add_rule(pattern='passive')
train(model, epochs=10)  # Only train NEW rules

# Keep expanding...
```

---

## Tonight's Plan

1. ✓ Increase num_rules to 50
2. ✓ Prime first 10 with linguistic patterns
3. ✓ Test on 1000 stories
4. ✓ See if accuracy jumps from 2% → 20%+

**Expected result:** Much better convergence due to:
- More capacity (50 vs 16 rules)
- Better initialization (primed patterns)
- Less interference (each rule can specialize)

Let me implement this now! 🚀
