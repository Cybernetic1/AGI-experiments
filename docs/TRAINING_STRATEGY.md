# Training Strategy: Leveraging Logical Structure

## Training Acceleration via Logical Preprocessing

### Key Advantage Over Traditional LLMs

**Traditional LLMs**: Learn everything from raw tokens
- Must discover: entities, relations, coreference, etc.
- No explicit structure guidance
- Slow, data-hungry learning

**Logic-Structured AGI**: Can leverage NLP preprocessing
- Pre-extract entities and relations from text
- Provide explicit supervision signals
- Faster, more sample-efficient learning

## NLP Preprocessing Pipeline

```python
# Input: Natural language text
text = "A French general named Napoleon fought at Austerlitz. He won the battle."

# Step 1: Entity Extraction (NLP)
entities = [
    {"id": 1, "mentions": ["French general", "Napoleon", "He"], "type": "PERSON"},
    {"id": 2, "mentions": ["Austerlitz"], "type": "LOCATION"},
    {"id": 3, "mentions": ["the battle"], "type": "EVENT"}
]

# Step 2: Relation Extraction (NLP)
relations = [
    [1, "name", "Napoleon"],
    [1, "nationality", "French"],
    [1, "occupation", "general"],
    [1, "fought_at", 2],
    [1, "won", 3],
    [3, "location", 2]
]

# Step 3: Convert to Logic Format (Training Data)
training_sequence = [
    # t=0: Introduce entity
    {"wm": [], 
     "action": "CREATE", 
     "entity_type": "PERSON", 
     "properties": {"nationality": "French", "occupation": "general"}},
    
    # t=1: Add name property
    {"wm": [[1, "is_a", "person"]], 
     "action": "ADD_PROPERTY",
     "entity": 1, 
     "property": "name", 
     "value": "Napoleon"},
    
    # t=2: Create location entity
    {"wm": [[1, "name", "Napoleon"]], 
     "action": "CREATE",
     "entity_type": "LOCATION", 
     "name": "Austerlitz"},
    
    # t=3: Add relation
    {"wm": [[1, "name", "Napoleon"], [2, "name", "Austerlitz"]],
     "action": "ADD_RELATION", 
     "subject": 1, 
     "relation": "fought_at", 
     "object": 2},
    
    # t=4: Coreference ("He" → entity 1)
    {"wm": [[1, "fought_at", 2]], 
     "action": "ADD_PROPERTY",
     "entity": 1, 
     "property": "won", 
     "value": 3}
]

# Step 4: Train on Logical Sequences
# System learns to predict next action/proposition
# With explicit entity IDs and structure
```

## Why This Accelerates Training

1. **Explicit entity supervision**: System knows which mentions refer to same entity
2. **Relation labels**: Direct supervision on relation types
3. **Coreference resolution**: Pre-solved (He → Napoleon)
4. **Type information**: Entity types guide creation decisions
5. **Structured targets**: Predict logical propositions, not raw tokens

## Training Comparison

### Traditional LLM Training

```python
Input:  "A French general named Napoleon fought"
Target: " at Austerlitz"
Loss: Cross-entropy on token prediction
```
→ Must implicitly learn: entities, relations, coreference, reasoning

### Logic-AGI Training

```python
Input:  WM = [[1, "nationality", "French"], [1, "occupation", "general"]]
Target: Action = ADD_PROPERTY, entity=1, property="name", value="Napoleon"
Loss: Classification loss on action + entity + property
```
→ Explicitly supervised on structure

## Three-Phase Training Protocol (from TTT)

Based on successful TTT experiments, use hierarchical AR+RL training:

### Phase 1: Autoregressive Pre-training

**Objective**: Learn world model (predict next proposition)

```python
# Loss: Predict next proposition from current WM
for timestep in training_data:
    wm_t = timestep.working_memory
    target_prop = timestep.next_proposition
    
    # Forward pass
    concepts = logic_rules(wm_t)
    pred_prop = ar_head(concepts)
    
    # Loss
    loss = cross_entropy(pred_prop, target_prop)
    loss.backward()
```

**Duration**: ~70% of total training budget  
**Data**: Unlabeled text (preprocessed into logical sequences)

### Phase 2: RL with Frozen AR

**Objective**: Learn goal-directed behavior without forgetting

```python
# AR head frozen, only train RL head
ar_head.requires_grad = False

for episode in task_episodes:
    state = episode.initial_state
    
    while not done:
        # Extract concepts (shared logic rules)
        concepts = logic_rules(state)
        
        # RL action selection
        action = rl_head(concepts)
        
        # Environment step
        next_state, reward = env.step(action)
        
        # RL loss (REINFORCE)
        loss = -log_prob(action) * reward
        loss.backward()
```

**Duration**: ~20% of total training budget  
**Data**: Task-specific (Q&A, reasoning benchmarks)

### Phase 3: Joint Fine-tuning

**Objective**: Refine both heads together

```python
# Both heads trainable
ar_head.requires_grad = True
rl_head.requires_grad = True

# Alternating batches
for batch in mixed_data:
    if batch.type == "AR":
        loss = ar_loss(batch)
    else:  # RL batch
        loss = rl_loss(batch)
    
    loss.backward()
```

**Duration**: ~10% of total training budget  
**Data**: Mixed AR + RL data

## Training Data Format

### Input: Working Memory State

```python
wm_t = [
    [1, "is_a", "cat"],
    [1, "color", "black"],
    [2, "is_a", "mat"],
    [1, "on", 2]
]
# Encoded as embeddings + processed by logic rules
```

### Output: Next Proposition (AR) or Action (RL)

**Autoregressive target**:
```python
next_prop = [1, "sleeping", true]
# Predict: subject_id, relation, object
```

**RL target**:
```python
action = "ANSWER: entity_1"
# Predict: answer to question
```

## Supervision Signals

### 1. Entity IDs (from NLP)

```python
# Preprocessing annotates entity IDs
text: "Napoleon fought. He won."
annotated: [
    [1, "name", "Napoleon"],  # Entity 1
    [1, "fought", true],
    [1, "won", true]          # "He" resolved to entity 1
]
```

### 2. Relation Types (from NLP)

```python
# Use relation extraction tools
# spaCy, OpenIE, etc.
relations = extract_relations(text)
# Provides labels: "fought_at", "won", "located_in", etc.
```

### 3. Temporal Ordering (from text structure)

```python
# Sentence order → temporal order
sentence_1: [1, "name", "Napoleon"]  # t=0
sentence_2: [1, "fought_at", 2]      # t=1
sentence_3: [1, "won", 3]            # t=2
```

## NLP Tools for Preprocessing

### Entity and Relation Extraction

- **spaCy**: `python -m spacy download en_core_web_trf`
- **AllenNLP**: OpenIE relation extraction
- **Stanza**: Coreference resolution
- **Hugging Face**: NER and relation extraction models

### Example Pipeline

```python
import spacy
from allennlp.predictors import Predictor

# Load tools
nlp = spacy.load("en_core_web_trf")
openie = Predictor.from_path("openie-model.tar.gz")

def preprocess_text(text):
    # Entity extraction
    doc = nlp(text)
    entities = {}
    entity_id = 0
    
    for ent in doc.ents:
        if ent.text not in entities:
            entities[ent.text] = entity_id
            entity_id += 1
    
    # Relation extraction
    relations = openie.predict(sentence=text)
    
    # Convert to logical format
    propositions = []
    for rel in relations['verbs']:
        subj = entities.get(rel['arg0'])
        obj = entities.get(rel['arg1'])
        if subj is not None and obj is not None:
            propositions.append([subj, rel['verb'], obj])
    
    return propositions
```

## Training Metrics

### AR Metrics
- **Proposition accuracy**: Correct [subj, rel, obj] prediction
- **Entity accuracy**: Correct entity ID selection
- **Relation accuracy**: Correct relation type

### RL Metrics
- **Task success rate**: % of correctly answered questions
- **Reward**: Average episode reward
- **Sample efficiency**: Tasks solved per training step

### Combined Metrics
- **Rule interpretability**: Can we understand learned rules?
- **Generalization**: Performance on held-out scenarios
- **Sample efficiency**: Learning speed vs traditional LLM

## Next Steps

1. Implement NLP preprocessing pipeline
2. Convert existing text datasets to logical format
3. Annotate entity IDs and relations
4. Implement AR training loop
5. Implement RL training loop (Q&A tasks)
6. Run Phase 1: AR pre-training
7. Run Phase 2: RL with frozen AR
8. Run Phase 3: Joint fine-tuning
9. Evaluate and compare to baseline LLM
