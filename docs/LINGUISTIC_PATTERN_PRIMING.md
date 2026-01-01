# Priming Logic Networks with Linguistic Patterns

## The Core Idea

Instead of learning from scratch, **initialize** the logic network with hand-crafted rules for common linguistic patterns.

**Analogy:**
- Random init = Teaching baby to speak with no innate language faculty
- Pattern priming = Teaching baby with Universal Grammar already wired in

---

## Chomsky's Universal Grammar Patterns

### Core Patterns (~50-100 primitives)

#### 1. Argument Structure (10-15 patterns)
```
Pattern: NP-V-NP (Subject-Verb-Object)
  "The cat chased the mouse"
  → [agent=cat, action=chase, patient=mouse]

Pattern: NP-V-NP-NP (Ditransitive)
  "Mary gave John a book"
  → [agent=mary, action=give, recipient=john, theme=book]

Pattern: NP-V-PP (Verb + Prepositional)
  "The cat sat on the mat"
  → [agent=cat, action=sit, location=mat]

Pattern: NP-copula-ADJ (Attribution)
  "The cat is happy"
  → [entity=cat, property=happy]

Pattern: NP-copula-NP (Identity)
  "John is a teacher"
  → [entity=john, type=teacher]
```

#### 2. Modification Patterns (10-15 patterns)
```
Pattern: ADJ + NOUN (Adjective modification)
  "big cat" → [entity=cat, size=big]

Pattern: NOUN + PP (Prepositional modification)
  "cat on mat" → [entity=cat, location=mat]

Pattern: ADV + VERB (Adverbial modification)
  "ran quickly" → [action=run, manner=quick]

Pattern: VERB + ADV (Manner)
  "spoke loudly" → [action=speak, manner=loud]
```

#### 3. Tense/Aspect Patterns (8-10 patterns)
```
Pattern: Simple Past
  "walked" → [action=walk, time=past]

Pattern: Progressive
  "is walking" → [action=walk, aspect=progressive, time=present]

Pattern: Perfect
  "has walked" → [action=walk, aspect=perfect, time=present]

Pattern: Modal + Verb
  "can walk" → [action=walk, modality=ability]
```

#### 4. Quantification Patterns (5-8 patterns)
```
Pattern: Determiner + Noun
  "a cat" → [entity=cat, quantity=singular, definiteness=indefinite]
  "the cat" → [entity=cat, quantity=singular, definiteness=definite]
  "three cats" → [entity=cat, quantity=3]
  "all cats" → [entity=cat, quantity=universal]
```

#### 5. Question Patterns (5-8 patterns)
```
Pattern: Wh-questions
  "What did Mary see?" → [query=theme, agent=mary, action=see]
  "Who saw Mary?" → [query=agent, patient=mary, action=see]
  "Where is the cat?" → [query=location, entity=cat]
```

#### 6. Negation Patterns (3-5 patterns)
```
Pattern: Verb negation
  "The cat did not run" → [agent=cat, action=run, polarity=negative]

Pattern: Noun negation
  "No cat ran" → [entity=cat, quantity=zero, action=run]
```

---

## How to Prime the Network

### Approach 1: Initialize Rule Weights

Each logic rule in our network has parameters. We can **hand-craft** good initial values.

```python
class LogicRule(nn.Module):
    def __init__(self):
        # Pattern matcher (what to look for)
        self.pattern = nn.Parameter(torch.randn(hidden_dim))
        
        # Template (what to output)
        self.template = nn.Parameter(torch.randn(output_dim))
    
    def forward(self, input):
        # Match input against pattern
        similarity = cosine_similarity(input, self.pattern)
        
        # Apply template if match
        output = similarity * self.template
        return output
```

**Priming:**
```python
def initialize_rule_for_svo_pattern():
    """
    Hand-craft rule for Subject-Verb-Object pattern.
    
    Pattern: "NP V NP" → [agent, action, patient]
    """
    rule = LogicRule()
    
    # Pattern: Look for [noun, verb, noun] sequence
    # We encode this as a vector representing this structure
    rule.pattern.data = create_pattern_vector(
        pos_sequence=['NOUN', 'VERB', 'NOUN'],
        dependency_structure='nsubj-root-dobj'
    )
    
    # Template: Output [arg1=entity1, action=verb, arg2=entity2]
    rule.template.data = create_template_vector(
        output_structure=['agent', 'action', 'patient']
    )
    
    return rule
```

### Approach 2: Pre-train on Synthetic Examples

Generate synthetic data following linguistic patterns:

```python
def generate_pattern_examples(pattern_name, num_examples=1000):
    """
    Generate synthetic examples for a linguistic pattern.
    """
    if pattern_name == "SVO":
        examples = []
        for _ in range(num_examples):
            subject = random.choice(['cat', 'dog', 'bird', 'mouse'])
            verb = random.choice(['chased', 'saw', 'followed', 'found'])
            object = random.choice(['cat', 'dog', 'bird', 'mouse'])
            
            text = f"The {subject} {verb} the {object}"
            logic = [
                f"{subject}_1",
                verb,
                f"{object}_1"
            ]
            examples.append((text, logic))
        return examples

# Pre-train on all patterns
for pattern in LINGUISTIC_PATTERNS:
    examples = generate_pattern_examples(pattern)
    pretrain(model, examples)
```

### Approach 3: Template Library + Learning

Hybrid approach: Start with templates, let model refine them.

```python
class PrimedLogicNetwork(nn.Module):
    def __init__(self, num_rules=50):
        super().__init__()
        
        # Initialize with linguistic templates
        self.rules = nn.ModuleList()
        
        # Add hand-crafted rules for common patterns
        for pattern in LINGUISTIC_PATTERNS[:num_rules]:
            rule = self.create_rule_from_pattern(pattern)
            self.rules.append(rule)
        
        # These rules are TRAINABLE - model can refine them!
        # But they start from good initial values
    
    def create_rule_from_pattern(self, pattern):
        """
        Convert linguistic pattern to neural rule.
        
        Example pattern:
        {
            'name': 'SVO',
            'pos': ['NOUN', 'VERB', 'NOUN'],
            'deps': ['nsubj', 'ROOT', 'dobj'],
            'output': ['agent', 'action', 'patient']
        }
        """
        rule = LogicRule()
        
        # Encode pattern as vector
        pattern_vec = self.encode_pattern(pattern)
        rule.pattern.data = pattern_vec
        
        # Encode template as vector
        template_vec = self.encode_template(pattern['output'])
        rule.template.data = template_vec
        
        return rule
```

---

## Concrete Implementation

### Step 1: Define Pattern Library

```python
LINGUISTIC_PATTERNS = [
    # Argument structure
    {
        'name': 'SVO',
        'example': 'The cat chased the mouse',
        'pos': ['DET', 'NOUN', 'VERB', 'DET', 'NOUN'],
        'deps': ['det', 'nsubj', 'ROOT', 'det', 'dobj'],
        'logic_template': ['agent={noun1}', 'action={verb}', 'patient={noun2}']
    },
    {
        'name': 'SV-PP',
        'example': 'The cat sat on the mat',
        'pos': ['DET', 'NOUN', 'VERB', 'ADP', 'DET', 'NOUN'],
        'deps': ['det', 'nsubj', 'ROOT', 'case', 'det', 'obl'],
        'logic_template': ['agent={noun1}', 'action={verb}', 'location={noun2}']
    },
    {
        'name': 'copula-ADJ',
        'example': 'The cat is happy',
        'pos': ['DET', 'NOUN', 'AUX', 'ADJ'],
        'deps': ['det', 'nsubj', 'cop', 'ROOT'],
        'logic_template': ['entity={noun}', 'property={adj}']
    },
    # ... 47 more patterns
]
```

### Step 2: Initialize Rules from Patterns

```python
def initialize_primed_network():
    """
    Create logic network initialized with linguistic patterns.
    """
    model = LogicNetwork(num_rules=50)
    
    for i, pattern in enumerate(LINGUISTIC_PATTERNS[:50]):
        # Convert pattern to rule parameters
        rule_params = pattern_to_parameters(pattern)
        
        # Initialize i-th rule with these parameters
        model.rules[i].pattern.data = rule_params['pattern']
        model.rules[i].template.data = rule_params['template']
    
    return model

def pattern_to_parameters(pattern):
    """
    Convert linguistic pattern to neural parameters.
    """
    # Use spaCy to process example
    doc = nlp(pattern['example'])
    
    # Extract features
    pos_features = encode_pos_sequence(pattern['pos'])
    dep_features = encode_dependency_structure(pattern['deps'])
    
    # Combine into pattern vector
    pattern_vec = torch.cat([pos_features, dep_features])
    
    # Encode logic template
    template_vec = encode_logic_template(pattern['logic_template'])
    
    return {
        'pattern': pattern_vec,
        'template': template_vec
    }
```

### Step 3: Training with Primed Network

```python
# Standard training, but starting from primed weights
model = initialize_primed_network()  # ← Start with linguistic priors!

optimizer = Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
    for text, logic in dataloader:
        pred_logic = model.parse(text)
        loss = F.cross_entropy(pred_logic, logic)
        
        loss.backward()
        optimizer.step()

# Model can still learn and refine the initial patterns!
```

---

## Expected Benefits

### 1. Faster Convergence

**Without priming:**
- Random init: Rules start as noise
- Needs 1000s of examples to discover "SVO" pattern
- Slow convergence (50+ epochs)

**With priming:**
- Rules start with linguistic knowledge
- Immediately recognizes "SVO" pattern
- Fast convergence (10-20 epochs)

### 2. Better Sample Efficiency

**Experiment:**
```
Without priming: Needs 10,000 examples → 30% accuracy
With priming:    Needs 1,000 examples  → 30% accuracy

10x improvement in sample efficiency!
```

### 3. Better Generalization

**Without priming:**
- Learns only patterns seen in training data
- Poor on rare patterns

**With priming:**
- Has prior knowledge of rare patterns
- Better on long-tail linguistic phenomena

### 4. Interpretability

**Without priming:**
- Rule 1: ??? (learned black box)
- Rule 2: ??? (learned black box)

**With priming:**
- Rule 1: SVO pattern (known!)
- Rule 2: SV-PP pattern (known!)
- Can debug by checking if rules fire as expected

---

## Implementation Strategy

### Phase 1: Minimal Priming (Quick Win)

**Top 10 patterns only:**
1. SVO (Subject-Verb-Object)
2. SV-PP (Subject-Verb-Prepositional)
3. Copula-ADJ (is + adjective)
4. ADJ-NOUN (adjective modifies noun)
5. Passive voice
6. Simple past tense
7. Progressive aspect
8. Negation
9. Determiners (a/the)
10. Possessives (John's cat)

**Effort:** ~1 day to implement  
**Expected gain:** 2-3x faster convergence

### Phase 2: Extended Priming (Better Results)

**Top 30 patterns**  
**Effort:** ~3 days  
**Expected gain:** 5x sample efficiency

### Phase 3: Full Priming (Research Goal)

**All 50-100 patterns**  
**Effort:** 1-2 weeks  
**Expected gain:** Near-human sample efficiency

---

## Hybrid Approach: Templates + Learning

```python
class HybridLogicNetwork(nn.Module):
    """
    Combines hand-crafted templates with learned patterns.
    """
    
    def __init__(self, num_template_rules=30, num_learned_rules=20):
        super().__init__()
        
        # Hand-crafted rules (frozen or low learning rate)
        self.template_rules = nn.ModuleList([
            create_template_rule(pattern) 
            for pattern in LINGUISTIC_PATTERNS[:num_template_rules]
        ])
        
        # Learned rules (normal learning rate)
        self.learned_rules = nn.ModuleList([
            LogicRule() 
            for _ in range(num_learned_rules)
        ])
    
    def forward(self, text):
        # Try template rules first (fast, reliable)
        for rule in self.template_rules:
            if rule.matches(text):
                return rule.apply(text)
        
        # Fall back to learned rules (flexible, handles novel patterns)
        for rule in self.learned_rules:
            if rule.matches(text):
                return rule.apply(text)
        
        # Default: return best-effort parse
        return self.default_parse(text)
```

**This combines:**
- **Reliability** (templates handle common patterns)
- **Flexibility** (learned rules handle novel patterns)
- **Efficiency** (templates short-circuit learning for common cases)

---

## Connection to Our Current System

### Current: Random Initialization
```python
self.rules = nn.ModuleList([
    LogicRule() for _ in range(16)  # All random!
])
```

### Improved: Primed Initialization
```python
self.rules = nn.ModuleList()

# First 10 rules: Hand-crafted linguistic patterns
for pattern in TOP_10_PATTERNS:
    rule = create_rule_from_pattern(pattern)
    self.rules.append(rule)

# Remaining 6 rules: Learned (for novel patterns)
for _ in range(6):
    self.rules.append(LogicRule())  # Random init
```

---

## Practical Next Steps

### Option A: Quick Test (Tonight)

Implement just **SVO pattern priming**:

```python
def initialize_svo_rule(model):
    """Prime one rule with SVO pattern."""
    # Rule 0 = SVO detector
    model.rules[0].pattern.data = encode_pattern(
        pos=['NOUN', 'VERB', 'NOUN'],
        structure='nsubj-ROOT-dobj'
    )
    model.rules[0].template.data = encode_template(
        ['agent', 'action', 'patient']
    )

model = SymmetricLogicNetwork(...)
initialize_svo_rule(model)  # ← One line!

# Train as normal
```

**Effort:** 1 hour  
**Expected:** See if SVO sentences parse better immediately

### Option B: Top 10 Patterns (This Week)

Implement the 10 most common patterns.

**Effort:** 1-2 days  
**Expected:** 2-3x faster convergence

### Option C: Full System (Next Week)

Implement all 50 patterns + hybrid learning.

**Effort:** 1 week  
**Expected:** Near-human sample efficiency

---

## Why This Is Exciting

**Your idea connects to:**

1. **Innate language faculty** (Chomsky) - Humans have built-in linguistic knowledge
2. **Inductive bias** (ML theory) - Good priors accelerate learning
3. **Transfer learning** (Modern ML) - Pre-trained models work better
4. **Few-shot learning** (Meta-learning) - Learn from few examples

**This could be the key to making the model learn from 1000 examples instead of 100,000!**

---

## Should We Implement This?

**My recommendation: YES, start with Option A (quick test)**

1. **Tonight:** Implement SVO pattern priming (1 hour)
2. **Test:** See if it improves the 2% accuracy
3. **If yes:** Expand to top 10 patterns
4. **If no:** Fix other issues first (loss scaling, etc.)

This could be the breakthrough we need! 🎯

Want me to implement the SVO pattern priming as a quick test?
