# First Experiment: Dataset Size and Preparation

## Recommended Corpus Size for Initial Experiments

### Phase 1: Proof of Concept (RECOMMENDED START)

**Corpus Size**: **100-500 simple sentences**

**Why this size?**
- Small enough to manually inspect and debug
- Large enough to learn basic patterns
- Fast training iteration (<1 hour per experiment)
- Easy to annotate with entity IDs

**Example sources**:
- Children's books (first-grade reading level)
- Simple Wikipedia articles (simplified English)
- Synthetic data generated with templates

**Entities**: ~20-50 unique entities  
**Relations**: ~10-20 relation types  
**Training time**: Minutes to hours (not days)

### Phase 2: Scaling Up

**Corpus Size**: **5,000-10,000 sentences**

**Why this size?**
- Tests scalability of entity tracking
- Enough diversity to learn robust patterns
- Still manageable for detailed analysis
- Training time: Hours to a day

**Entities**: ~500-1000 unique entities  
**Relations**: ~50-100 relation types  

### Phase 3: Large-Scale

**Corpus Size**: **100,000+ sentences**

**Why this size?**
- Comparable to small language model training
- Tests all scaling mechanisms
- Requires efficient retrieval and VQ codebook

**Entities**: ~10,000+ unique entities  
**Relations**: ~200+ relation types  
**Training time**: Days

## Data Preparation Strategy

### Step 1: Source Selection

**Option A: Existing Datasets**

1. **bAbI Dataset** (Facebook AI)
   - Size: 20 tasks, ~10K examples each
   - Pre-annotated with entity IDs and relations
   - Focus: Simple reasoning
   - **HIGHLY RECOMMENDED** for Phase 1

2. **Children's Book Test (CBT)**
   - Size: ~680K sentences
   - Simple narrative structure
   - Good for entity tracking

3. **Simple Wikipedia**
   - Size: Varies (can select subset)
   - Factual content
   - Good for relation extraction

**Option B: Synthetic Data**

Generate with templates:
```python
templates = [
    "{PERSON} is a {PROFESSION}.",
    "{PERSON} lives in {LOCATION}.",
    "{PERSON} has a {OBJECT}.",
    "{PERSON_1} met {PERSON_2} at {LOCATION}.",
]

entities = {
    "PERSON": ["Alice", "Bob", "Charlie"],
    "PROFESSION": ["teacher", "doctor", "engineer"],
    "LOCATION": ["Paris", "London", "Tokyo"],
    "OBJECT": ["cat", "dog", "book"]
}

# Generate 100-500 sentences
```

**Advantages**:
- Full control over complexity
- Known ground truth
- Easy to annotate
- Can gradually increase difficulty

**Disadvantages**:
- Less realistic
- May not capture full language complexity

### Step 2: Annotation Pipeline

Use NLP tools to pre-process:

```python
import spacy
from collections import defaultdict

nlp = spacy.load("en_core_web_sm")

def annotate_text(text):
    doc = nlp(text)
    
    # Entity extraction
    entities = {}
    entity_counter = 0
    entity_mentions = defaultdict(list)
    
    for ent in doc.ents:
        if ent.text not in entities:
            entities[ent.text] = entity_counter
            entity_counter += 1
        entity_mentions[entities[ent.text]].append(ent.text)
    
    # Relation extraction (simplified)
    relations = []
    for token in doc:
        if token.pos_ == "VERB":
            subj = [c for c in token.children if c.dep_ == "nsubj"]
            obj = [c for c in token.children if c.dep_ == "dobj"]
            
            if subj and obj:
                subj_text = subj[0].text
                obj_text = obj[0].text
                
                if subj_text in entities and obj_text in entities:
                    relations.append([
                        entities[subj_text],
                        token.lemma_,
                        entities[obj_text]
                    ])
    
    return entities, relations

# Example
text = "Napoleon was a general. He fought at Waterloo."
entities, relations = annotate_text(text)
print(entities)  # {"Napoleon": 0, "general": 1, "Waterloo": 2}
print(relations)  # [[0, "be", 1], [0, "fight", 2]]
```

### Step 3: Data Format

**JSON format for training**:

```json
{
  "text": "Napoleon was a general. He fought at Waterloo.",
  "entities": {
    "0": {"name": "Napoleon", "type": "PERSON", "mentions": ["Napoleon", "He"]},
    "1": {"name": "general", "type": "PROFESSION"},
    "2": {"name": "Waterloo", "type": "LOCATION"}
  },
  "sequence": [
    {
      "timestep": 0,
      "wm": [],
      "action": "CREATE",
      "entity_id": 0,
      "entity_type": "PERSON"
    },
    {
      "timestep": 1,
      "wm": [[0, "is_a", "PERSON"]],
      "action": "ADD_PROPERTY",
      "entity_id": 0,
      "property": "name",
      "value": "Napoleon"
    },
    {
      "timestep": 2,
      "wm": [[0, "name", "Napoleon"]],
      "action": "CREATE",
      "entity_id": 1,
      "entity_type": "PROFESSION"
    },
    {
      "timestep": 3,
      "wm": [[0, "name", "Napoleon"], [1, "is_a", "PROFESSION"]],
      "action": "ADD_RELATION",
      "subject": 0,
      "relation": "is_a",
      "object": 1
    }
  ]
}
```

## Concrete Recommendation for First Experiment

### Dataset: bAbI Task 1 (Single Supporting Fact)

**Why?**
- Pre-annotated with entity IDs
- Simple reasoning required
- Only ~1000 examples per task
- Well-studied baseline

**Example**:
```
Mary went to the bathroom.
John moved to the hallway.
Where is Mary? A: bathroom
```

**Preprocessing**:
```python
# Convert to logical format
entities = {"Mary": 0, "John": 1, "bathroom": 2, "hallway": 3}

sequence = [
    [0, "went_to", 2],    # Mary went to bathroom
    [1, "moved_to", 3],   # John moved to hallway
]

question = [0, "located_at", ?]  # Where is Mary?
answer = 2  # bathroom
```

### Training Protocol

1. **Data split**: 80% train, 10% validation, 10% test
2. **AR training**: Predict next proposition
3. **RL training**: Answer questions correctly
4. **Evaluation**: QA accuracy, entity tracking accuracy

### Expected Metrics

**Success criteria**:
- QA accuracy > 90% (bAbI baseline)
- Entity tracking: 100% (only 2-4 entities)
- Training time: < 1 hour on GPU

**If successful**, scale to:
- bAbI Task 2 (two supporting facts)
- bAbI Task 3 (three supporting facts)
- Multiple tasks combined

## Implementation Timeline

### Week 1: Data Preparation
- [ ] Download bAbI dataset
- [ ] Write preprocessing script
- [ ] Convert to JSON format
- [ ] Verify data quality

### Week 2: Model Implementation
- [ ] Implement entity registry
- [ ] Implement WM/LTM
- [ ] Implement logic rules
- [ ] Implement AR/RL heads

### Week 3: Training
- [ ] Run AR pre-training
- [ ] Run RL training (frozen AR)
- [ ] Run joint fine-tuning
- [ ] Evaluate and analyze

### Week 4: Analysis & Iteration
- [ ] Visualize learned rules
- [ ] Debug failure cases
- [ ] Iterate on architecture
- [ ] Plan next experiment

## Expected Outcomes

**If experiment succeeds**:
✅ Proof that logic networks can learn from text  
✅ Entity tracking works  
✅ AR+RL training protocol effective  
✅ Ready to scale up  

**If experiment fails**:
🔍 Analyze why:
- Entity tracking not working?
- Rules not learning?
- AR/RL mismatch?
- Architecture issue?

Then iterate and try again!

## Cost Estimate

### Computational
- **GPU**: ~10-20 hours (Phase 1)
- **Cost**: ~$10-50 (cloud GPU)
- **Alternative**: Free Colab GPU sufficient

### Time (human)
- **Data prep**: 1-2 days
- **Implementation**: 3-5 days
- **Training & debug**: 2-3 days
- **Analysis**: 1-2 days
- **Total**: ~2 weeks part-time

## Next Steps

1. **Download bAbI**: `git clone https://github.com/facebook/bAbI-tasks`
2. **Set up preprocessing**: Write scripts to convert format
3. **Implement core modules**: Entity registry, WM, LTM
4. **Start with Task 1**: Simplest possible test
5. **Iterate**: Improve based on results

Ready to start! 🚀
