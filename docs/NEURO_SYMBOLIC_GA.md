# Genetic Algorithms + Neural Networks for Logic Rules

## Your Idea: GA for Symbolic Rules

**Key insight:** Logic rules are DISCRETE and COMPOSITIONAL - perfect for GA!

```
Neural Network:
- Good at: Continuous optimization (gradient descent)
- Bad at: Discrete symbolic search

Genetic Algorithm:
- Good at: Discrete symbolic search, combinatorial optimization
- Bad at: Fine-grained continuous optimization

Together: Best of both worlds! 🎯
```

---

## Why GA for Logic Rules Makes Sense

### Problem with Pure Neural Approach

```python
# Neural network tries to learn:
Rule: "If NOUN VERB NOUN then [agent, action, patient]"

# How it learns:
Continuous weights → Soft matching → Argmax → Discrete output

# Problem:
- Slow (needs 1000s of examples)
- Opaque (can't inspect the rule)
- Fragile (slight changes break it)
```

### GA Approach

```python
# GA directly searches symbolic space:
Population of rules:
  Rule 1: "NOUN VERB NOUN → [agent, action, patient]"
  Rule 2: "NOUN AUX ADJ → [entity, property]"
  Rule 3: "ADJ NOUN → [entity, has_property]"
  ...

# Evolution:
1. Evaluate fitness (how many sentences does each rule parse correctly?)
2. Select best rules
3. Crossover (combine rules)
4. Mutate (add/remove conditions)
5. Repeat

# Result: Explicit, interpretable rules!
```

---

## Hybrid Architectures: GA + Neural Networks

### Approach 1: Neural Fitness Function

**GA searches symbolic rules, NN evaluates them**

```python
class NeuralFitnessFunction:
    """
    Neural network evaluates how good a symbolic rule is.
    """
    
    def __init__(self):
        # Neural encoder: Text → Embedding
        self.text_encoder = nn.LSTM(...)
        
        # Neural evaluator: (Embedding, Rule) → Score
        self.rule_evaluator = nn.Linear(...)
    
    def evaluate_rule(self, text, rule):
        """
        Score how well a rule applies to this text.
        
        Args:
            text: Input sentence
            rule: Symbolic rule (e.g., "NOUN VERB NOUN → [agent, action, patient]")
        
        Returns:
            score: Fitness (0-1)
        """
        # Encode text
        embedding = self.text_encoder(text)
        
        # Encode rule (symbolic → vector)
        rule_vector = self.encode_rule(rule)
        
        # Predict: Does rule apply here?
        score = self.rule_evaluator(embedding, rule_vector)
        
        return score

# GA uses this to evolve rules
ga = GeneticAlgorithm(fitness_fn=neural_fitness)
best_rules = ga.evolve(population_size=100, generations=50)
```

**Advantage:** 
- GA searches discrete space efficiently
- NN provides smooth, differentiable fitness landscape

### Approach 2: Neural-Guided Mutation

**NN suggests where to mutate rules**

```python
class NeuralMutationGuide:
    """
    Neural network predicts good mutations for rules.
    """
    
    def __init__(self):
        self.mutation_predictor = nn.Transformer(...)
    
    def suggest_mutations(self, rule, training_data):
        """
        Suggest how to improve a rule based on where it fails.
        
        Args:
            rule: Current symbolic rule
            training_data: Examples where rule fails
        
        Returns:
            mutations: List of suggested modifications
        """
        # Find failure cases
        failures = [ex for ex in training_data if not rule.matches(ex)]
        
        # Neural net analyzes failures
        failure_patterns = self.analyze_failures(failures)
        
        # Suggest mutations
        mutations = self.generate_mutations(rule, failure_patterns)
        
        return mutations

# GA uses neural guidance
for generation in range(100):
    # Standard GA selection
    parents = select_best(population)
    
    # Neural-guided mutation (smarter than random!)
    children = []
    for parent in parents:
        mutations = neural_guide.suggest_mutations(parent, training_data)
        child = apply_best_mutation(parent, mutations)
        children.append(child)
    
    population = children
```

**Advantage:**
- Mutations are targeted (not random)
- Learns from failures
- Faster convergence

### Approach 3: Differentiable GA (DARTS-style)

**Make symbolic rules differentiable via relaxation**

```python
class DifferentiableRule:
    """
    Symbolic rule with continuous relaxation.
    """
    
    def __init__(self, num_operations=10):
        # Softmax over possible operations
        self.operation_weights = nn.Parameter(torch.randn(num_operations))
    
    def apply(self, input):
        """
        Apply rule as weighted combination of operations.
        """
        # Soft selection of operations
        op_probs = F.softmax(self.operation_weights, dim=0)
        
        # Each operation processes input
        outputs = [op(input) for op in self.operations]
        
        # Weighted combination (differentiable!)
        result = sum([p * out for p, out in zip(op_probs, outputs)])
        
        return result
    
    def discretize(self):
        """
        Convert to discrete rule (pick highest weight).
        """
        best_op = torch.argmax(self.operation_weights)
        return DiscreteRule(operation=self.operations[best_op])

# Training
model = NeuralNetwork(rules=[DifferentiableRule() for _ in range(50)])

# Gradient descent (continuous)
for epoch in range(100):
    loss = train(model)
    loss.backward()
    optimizer.step()

# After training, discretize
discrete_rules = [rule.discretize() for rule in model.rules]
```

**Advantage:**
- Uses gradient descent (fast!)
- Gets discrete rules at end (interpretable!)
- Best of both worlds

### Approach 4: Neural Program Synthesis

**NN generates rule programs, GA refines them**

```python
class NeuralProgramGenerator:
    """
    Generate symbolic rules using seq2seq model.
    """
    
    def __init__(self):
        self.encoder = nn.LSTM(...)  # Encode examples
        self.decoder = nn.LSTM(...)  # Decode to rule program
    
    def generate_rule(self, examples):
        """
        Given examples, generate a rule that handles them.
        
        Input examples:
          ["The cat sat" → [cat, sit]]
          ["The dog ran" → [dog, run]]
        
        Output rule:
          "NOUN VERB → [agent, action]"
        """
        # Encode examples
        context = self.encoder([ex.text for ex in examples])
        
        # Decode to rule (as sequence of tokens)
        rule_tokens = self.decoder(context)
        
        # Parse tokens into executable rule
        rule = parse_rule_program(rule_tokens)
        
        return rule

# Hybrid pipeline
for iteration in range(100):
    # Neural: Generate candidate rules
    candidates = neural_generator.generate_rule(training_examples)
    
    # GA: Refine and evolve
    population = candidates
    for gen in range(10):
        population = ga_evolve(population)
    
    # Keep best rules
    best_rules.extend(select_top_k(population, k=10))
```

**Advantage:**
- NN does creative generation (explores widely)
- GA does local refinement (exploits)
- Combines exploration + exploitation

---

## Concrete Implementation for Our System

### Phase 1: Simple GA for Rule Discovery

```python
class SymbolicRule:
    """
    A symbolic logic rule (discrete, interpretable).
    """
    
    def __init__(self):
        self.pattern = PatternMatcher()  # e.g., "NOUN VERB NOUN"
        self.template = LogicTemplate()  # e.g., [agent, action, patient]
    
    def matches(self, sentence):
        """Check if pattern matches sentence."""
        return self.pattern.matches(sentence)
    
    def apply(self, sentence):
        """Extract logic from sentence."""
        return self.template.extract(sentence)
    
    def mutate(self):
        """Random mutation (GA operator)."""
        if random.random() < 0.5:
            # Mutate pattern
            self.pattern.add_constraint()
        else:
            # Mutate template
            self.template.modify_slot()
    
    def crossover(self, other):
        """Combine with another rule (GA operator)."""
        child = SymbolicRule()
        child.pattern = random.choice([self.pattern, other.pattern])
        child.template = random.choice([self.template, other.template])
        return child

def genetic_algorithm_for_rules(training_data, generations=100, pop_size=50):
    """
    Evolve symbolic rules using GA.
    """
    # Initialize population
    population = [SymbolicRule.random() for _ in range(pop_size)]
    
    for gen in range(generations):
        # Evaluate fitness
        fitness = []
        for rule in population:
            correct = sum([1 for ex in training_data 
                          if rule.matches(ex.text) and 
                             rule.apply(ex.text) == ex.logic])
            fitness.append(correct / len(training_data))
        
        # Selection (tournament)
        parents = tournament_selection(population, fitness, k=pop_size//2)
        
        # Crossover
        children = []
        for i in range(0, len(parents), 2):
            child = parents[i].crossover(parents[i+1])
            children.append(child)
        
        # Mutation
        for child in children:
            if random.random() < 0.1:
                child.mutate()
        
        # New generation
        population = parents + children
        
        print(f"Gen {gen}: Best fitness = {max(fitness):.2f}")
    
    # Return best rules
    return sorted(population, key=lambda r: fitness[population.index(r)], 
                  reverse=True)[:10]
```

### Phase 2: Hybrid Neural-GA

```python
class HybridNeuralGA:
    """
    Combines neural network with GA.
    
    - Neural network: Learns embeddings and soft patterns
    - GA: Searches for symbolic rules in embedding space
    """
    
    def __init__(self):
        # Neural encoder
        self.encoder = nn.LSTM(...)
        
        # GA population
        self.symbolic_rules = [SymbolicRule() for _ in range(50)]
    
    def train(self, training_data, epochs=100):
        """
        Alternate between neural training and GA evolution.
        """
        for epoch in range(epochs):
            # Phase 1: Train neural encoder (gradient descent)
            for batch in training_data:
                embeddings = self.encoder(batch.text)
                loss = self.neural_loss(embeddings, batch.logic)
                loss.backward()
                optimizer.step()
            
            # Phase 2: Evolve symbolic rules (GA)
            if epoch % 10 == 0:  # Every 10 epochs
                self.symbolic_rules = self.evolve_rules(
                    self.symbolic_rules, 
                    training_data,
                    use_neural_fitness=True  # NN helps evaluate!
                )
    
    def neural_fitness(self, rule, examples):
        """
        Use neural network to evaluate rule fitness.
        """
        scores = []
        for ex in examples:
            # Encode example
            embedding = self.encoder(ex.text)
            
            # Encode rule
            rule_embedding = self.encode_rule(rule)
            
            # Predict match
            score = cosine_similarity(embedding, rule_embedding)
            scores.append(score)
        
        return np.mean(scores)
    
    def predict(self, text):
        """
        Use both neural and symbolic pathways.
        """
        # Neural prediction
        embedding = self.encoder(text)
        neural_pred = self.neural_head(embedding)
        
        # Symbolic prediction (try each rule)
        for rule in self.symbolic_rules:
            if rule.matches(text):
                symbolic_pred = rule.apply(text)
                
                # Combine neural + symbolic
                return self.combine(neural_pred, symbolic_pred)
        
        # Fall back to neural only
        return neural_pred
```

---

## Advantages of GA + Neural Hybrid

### 1. Interpretability

**Pure neural:**
```
Input: "The cat chased the mouse"
Output: [entity=42, action=156, patient=87]
How? 🤷 Black box
```

**GA + Neural:**
```
Input: "The cat chased the mouse"
Matched rule: "NOUN VERB NOUN → [agent, action, patient]"
Output: [cat, chase, mouse]
How? ✓ Explicit rule!
```

### 2. Sample Efficiency

**Pure neural:** Needs 10,000+ examples to discover "NOUN VERB NOUN" pattern

**GA + Neural:** Can discover pattern from 100 examples by explicit search

### 3. Compositional Generalization

**GA rules are modular:**
```
Rule 1: NOUN VERB NOUN → SVO
Rule 2: NOUN AUX ADJ → copula
Rule 3: ADJ NOUN → modification

Can compose: "The big cat chased the small mouse"
  → Rule 3 + Rule 1 (compositional!)
```

### 4. Zero-Shot Transfer

```
GA discovers: "NOUN VERB NOUN → [agent, action, patient]"

New sentence: "The zorg blicked the flib"
Even with novel words (zorg, blicked, flib), rule still applies!

Neural network would fail (unknown words).
```

---

## Implementation Roadmap

### Week 1: Basic GA for Rules (Proof of Concept)

```python
# Implement basic GA
rules = genetic_algorithm(
    training_data=tinystories_1000,
    generations=100,
    population=50
)

# Test on validation set
accuracy = evaluate(rules, val_data)
print(f"GA-discovered rules: {accuracy}% accuracy")
```

**Expected:** 20-30% accuracy (better than our 2%!)

### Week 2: Neural Fitness Function

```python
# Train neural encoder
encoder = train_neural_encoder(training_data)

# Use encoder to guide GA
rules = genetic_algorithm_with_neural_fitness(
    training_data=tinystories_1000,
    encoder=encoder,
    generations=100
)

# Expected improvement
accuracy = evaluate(rules, val_data)
print(f"Neural-guided GA: {accuracy}% accuracy")
```

**Expected:** 40-50% accuracy

### Week 3: Full Hybrid System

```python
# Combine neural + symbolic
model = HybridNeuralGA()
model.train(training_data, epochs=50)

# Can explain predictions!
pred, explanation = model.predict_with_explanation(test_sentence)
print(f"Prediction: {pred}")
print(f"Because rule: {explanation.rule}")
print(f"Neural confidence: {explanation.neural_score}")
```

**Expected:** 50-60% accuracy + interpretability!

---

## Why This Is Exciting

Your idea connects to several cutting-edge research areas:

1. **Neural Program Synthesis** (MIT, OpenAI)
2. **Differentiable Neural Computers** (DeepMind)
3. **Neural Module Networks** (Berkeley)
4. **AlphaCode** (Google) - Uses symbolic search + neural
5. **Neurosymbolic AI** (IBM, MIT-IBM Watson)

**This could be publication-worthy research!**

---

## Immediate Next Steps

While the 50-rule test is running, we could:

1. **Quick experiment:** Implement simple GA for rule discovery
2. **Compare:** GA vs pure neural on same 1000 examples
3. **Analyze:** Which approach learns faster?

**Would you like me to implement a basic GA for rule discovery while we wait for the neural test to complete?**

This could reveal if symbolic search is indeed more sample-efficient! 🎯
