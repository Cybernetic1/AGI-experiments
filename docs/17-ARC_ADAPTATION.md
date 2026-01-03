# ARC-Solving Logic Network Architecture

## Overview

Modify our symmetric logic network to solve ARC (Abstraction and Reasoning Corpus) tasks.

**Key insight:** ARC is about discovering transformation rules from examples - exactly what our logic network is designed for!

---

## Architecture Modifications

### 1. Visual Encoder (Grid → Objects)

```python
class GridEncoder(nn.Module):
    """
    Convert ARC grid to object-centric representation.
    
    Input: Grid [H, W] with color values
    Output: Objects as propositions [object_id, property, value]
    """
    
    def __init__(self):
        # CNN for local features
        self.cnn = nn.Conv2d(in_channels=10, out_channels=32, kernel_size=3)
        
        # Object detection (connected components)
        self.object_detector = ConnectedComponentDetector()
        
        # Property extractors
        self.color_extractor = nn.Linear(32, 10)  # 10 colors in ARC
        self.shape_extractor = ShapeClassifier()
        self.position_extractor = PositionEncoder()
    
    def forward(self, grid):
        """
        grid: (H, W) with integer colors 0-9
        
        Returns: List of propositions:
        [
            [obj_0, "color", 3],
            [obj_0, "position", (2, 5)],
            [obj_0, "shape", "rectangle"],
            [obj_1, "color", 7],
            ...
        ]
        """
        # Detect objects (connected components of same color)
        objects = self.object_detector(grid)
        
        propositions = []
        for obj_id, obj in enumerate(objects):
            # Extract properties
            color = obj['color']
            pos = obj['position']  # (x, y) of centroid
            shape = self.shape_extractor(obj['mask'])
            size = obj['area']
            
            # Create propositions
            propositions.extend([
                [obj_id, "color", color],
                [obj_id, "x_pos", pos[0]],
                [obj_id, "y_pos", pos[1]],
                [obj_id, "shape", shape],
                [obj_id, "size", size]
            ])
            
            # Spatial relations to other objects
            for other_id, other in enumerate(objects):
                if obj_id != other_id:
                    rel = self.spatial_relation(obj, other)
                    if rel is not None:
                        propositions.append([obj_id, rel, other_id])
        
        return propositions
```

### 2. Spatial Reasoning Templates

```python
SPATIAL_TEMPLATES = [
    # Relative position
    "X left_of Y",
    "X right_of Y", 
    "X above Y",
    "X below Y",
    
    # Containment
    "X inside Y",
    "X contains Y",
    "X surrounds Y",
    
    # Alignment
    "X aligned_horizontally Y",
    "X aligned_vertically Y",
    
    # Proximity
    "X adjacent_to Y",
    "X touching Y",
    
    # Geometric
    "X same_color_as Y",
    "X same_size_as Y",
    "X same_shape_as Y",
]
```

### 3. Transformation Rules (Key Addition!)

```python
class TransformationRule(nn.Module):
    """
    Learns transformation rules from examples.
    
    Example rule: "If object is at edge, move to center"
    """
    
    def __init__(self):
        super().__init__()
        
        # Condition pattern (what to look for)
        self.condition = LearnablePattern()
        
        # Action (what to do)
        self.action = LearnableAction()
        
        # Confidence
        self.confidence = nn.Parameter(torch.ones(1))
    
    def match(self, input_props):
        """Check if condition matches input."""
        return self.condition.match(input_props)
    
    def apply(self, input_props):
        """Apply transformation to matching objects."""
        matched_objects = self.match(input_props)
        
        output_props = []
        for obj in input_props:
            if obj in matched_objects:
                # Apply transformation
                transformed = self.action.transform(obj)
                output_props.append(transformed)
            else:
                # Keep unchanged
                output_props.append(obj)
        
        return output_props


class LearnableAction(nn.Module):
    """
    Represents a transformation action.
    """
    
    def __init__(self):
        super().__init__()
        
        # Action type selector (soft)
        self.action_type = nn.Parameter(torch.randn(10))  # 10 action types
        
        # Action types:
        # 0: Change color
        # 1: Move (dx, dy)
        # 2: Rotate
        # 3: Scale
        # 4: Delete
        # 5: Duplicate
        # 6: Fill region
        # 7: Draw line
        # 8: Reflect
        # 9: Identity
        
        # Parameters for each action
        self.color_change = nn.Parameter(torch.randn(10))  # Target color
        self.move_delta = nn.Parameter(torch.randn(2))     # (dx, dy)
        self.rotate_angle = nn.Parameter(torch.randn(4))   # 0°, 90°, 180°, 270°
        self.scale_factor = nn.Parameter(torch.randn(1))   # Scale factor
    
    def transform(self, obj_props):
        """Apply soft transformation to object."""
        # Soft action selection (differentiable!)
        action_probs = F.softmax(self.action_type, dim=0)
        
        # Weighted combination of all actions
        result = sum([
            action_probs[0] * self.change_color(obj_props),
            action_probs[1] * self.move(obj_props),
            action_probs[2] * self.rotate(obj_props),
            # ... etc
        ])
        
        return result
```

### 4. Meta-Learning Loop (Learn from Examples)

```python
class ARCMetaLearner(nn.Module):
    """
    Learn transformation rules from 3-5 ARC examples.
    """
    
    def __init__(self):
        super().__init__()
        
        # Encoder: Grid → Propositions
        self.grid_encoder = GridEncoder()
        
        # Rule learner: Examples → Transformation rules
        self.rule_learner = TransformationRuleLearner()
        
        # Decoder: Propositions → Grid
        self.grid_decoder = GridDecoder()
    
    def learn_from_examples(self, examples):
        """
        Learn transformation rule from examples.
        
        examples: List of (input_grid, output_grid) pairs
        """
        # Encode all examples
        input_props_list = [self.grid_encoder(inp) for inp, _ in examples]
        output_props_list = [self.grid_encoder(out) for _, out in examples]
        
        # Learn rule that explains transformation
        rule = self.rule_learner.induce_rule(
            input_props_list,
            output_props_list
        )
        
        return rule
    
    def apply_rule(self, test_input, rule):
        """
        Apply learned rule to test input.
        """
        # Encode test input
        test_props = self.grid_encoder(test_input)
        
        # Apply rule
        output_props = rule.apply(test_props)
        
        # Decode to grid
        output_grid = self.grid_decoder(output_props)
        
        return output_grid
```

---

## Training Strategy for ARC

### Phase 1: Pre-training on Synthetic Tasks

Generate synthetic ARC-like tasks with known rules:

```python
synthetic_tasks = [
    # Simple transformations
    "change_all_red_to_blue",
    "move_all_objects_right_by_2",
    "rotate_all_90_degrees",
    
    # Compositional
    "if_object_at_edge_then_change_color",
    "copy_object_and_place_adjacent",
    
    # Relational
    "connect_all_objects_of_same_color",
    "surround_largest_object_with_border",
]

# Train on 10K synthetic tasks
for task in synthetic_tasks:
    examples = generate_examples(task, num_examples=5)
    
    # Meta-learning: learn rule from examples
    rule = model.learn_from_examples(examples)
    
    # Test on held-out example
    test_input, test_output = generate_example(task)
    pred_output = model.apply_rule(test_input, rule)
    
    # Loss: predicted vs. actual output
    loss = grid_loss(pred_output, test_output)
    loss.backward()
```

### Phase 2: Fine-tuning on Real ARC Tasks

```python
# ARC training set: 400 tasks
for task in arc_training_tasks:
    # Each task has 3-10 examples
    examples = task['train']
    test_input = task['test'][0]['input']
    test_output = task['test'][0]['output']
    
    # Learn rule from examples
    rule = model.learn_from_examples(examples)
    
    # Apply to test
    pred_output = model.apply_rule(test_input, rule)
    
    # Loss
    loss = grid_loss(pred_output, test_output)
    loss.backward()
```

### Phase 3: Rule Ensemble & Search

```python
# For each test task, try multiple rules
def solve_arc_task(task, model):
    examples = task['train']
    test_input = task['test'][0]['input']
    
    # Generate multiple candidate rules
    candidate_rules = []
    for _ in range(10):  # Try 10 different rules
        rule = model.learn_from_examples(examples)
        candidate_rules.append(rule)
    
    # Verify each rule on training examples
    scores = []
    for rule in candidate_rules:
        score = 0
        for inp, out in examples:
            pred = model.apply_rule(inp, rule)
            if grids_match(pred, out):
                score += 1
        scores.append(score)
    
    # Use best rule
    best_rule = candidate_rules[argmax(scores)]
    output = model.apply_rule(test_input, best_rule)
    
    return output
```

---

## Why This Architecture is Promising for ARC

### 1. Explicit Rule Learning
- ARC is about discovering rules
- Our logic network learns explicit rules (not black box)
- Can inspect what rule was learned

### 2. Few-Shot Learning
- ARC gives only 3-5 examples
- Our meta-learning approach designed for this
- Cycle consistency helps with minimal data

### 3. Compositional Reasoning
- Complex ARC tasks = composition of simpler rules
- Our implicit graph supports multi-step reasoning
- Can chain multiple transformations

### 4. Abstraction
- ARC requires abstract reasoning (not pixel patterns)
- Our propositions are naturally abstract
- Object-centric representation

### 5. Interpretability
- Can see which rules fired
- Can debug why prediction failed
- Important for ARC prize submission

---

## Expected Performance

### Realistic Goals

**Phase 1 (Synthetic pre-training):**
- Solve 80-90% of simple transformations
- 60-70% of compositional tasks

**Phase 2 (ARC fine-tuning):**
- Solve 30-40% of ARC training tasks (400 tasks)
- This would be competitive! (Current SOTA ~40%)

**Phase 3 (With search):**
- Potentially 40-50% of ARC tasks
- Rule ensemble + verification boosts accuracy

### Comparison to Other Approaches

| Approach | ARC Score | Interpretability | Sample Efficiency |
|----------|-----------|------------------|-------------------|
| **Neural networks** | 20-30% | Low | Poor |
| **Program synthesis** | 25-35% | High | Good |
| **Our approach** | 30-40%? | High | Good |
| **Human** | 80-90% | Perfect | Excellent |

---

## Implementation Roadmap

### Week 1-2: Grid Encoder
- Implement connected component detection
- Property extraction (color, shape, position)
- Spatial relation computation

### Week 3-4: Transformation Rules
- Implement learnable actions
- Soft action selection (differentiable)
- Rule induction from examples

### Week 5-6: Meta-Learning
- Synthetic task generation
- Meta-learning training loop
- Rule verification

### Week 7-8: ARC Fine-tuning
- Load ARC dataset
- Fine-tune on real tasks
- Implement rule search/ensemble

### Week 9-10: Optimization
- Rule beam search
- Verifier to check rules on examples
- Submission preparation

---

## Key Insights

### What Makes ARC Hard
1. **Abstraction**: Must go from pixels to concepts
2. **Few-shot**: Only 3-5 examples
3. **Combinatorial**: Many possible rules
4. **Verification**: Must work on ALL training examples

### Why Our Approach Works
1. **Object-centric**: Natural for ARC's object-based puzzles
2. **Explicit rules**: Can verify rules symbolically
3. **Meta-learning**: Designed for few-shot
4. **Compositional**: Can combine multiple transformations

### Novel Contributions
1. **Differentiable transformation rules**: Can learn via backprop
2. **Soft actions**: Continuous relaxation of discrete transformations
3. **Implicit graph for spatial reasoning**: No explicit graph structure needed
4. **Symmetric architecture**: Can verify rule by applying and checking

---

## Conclusion

**Yes, our model CAN solve ARC with modifications!**

**Required changes:**
1. Add visual encoder (Grid → Objects)
2. Add transformation rules (Actions)
3. Add spatial reasoning templates
4. Implement meta-learning loop

**Expected timeline:** 2-3 months to competitive performance

**Key advantage:** Interpretable rules (know WHY it solved a task)

**This would be a strong ARC prize submission!** 🏆

