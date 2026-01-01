# Quick Reference: Datasets & LTM

## Larger Corpora (After TinyStories)

### Immediate Next Steps
1. **TinyStories Full** (2.1M stories) - Already integrated ✓
2. **Children's Book Test** (688K sentences) - Similar domain
3. **Wikipedia** (6M articles) - Factual knowledge
4. **BookCorpus** (74M sentences) - Complex narratives

### Long-term
5. **C4** (365M documents) - Diverse web text
6. **The Pile** (825GB) - Most comprehensive
7. **RedPajama** (1.2T tokens) - LLaMA training data

## LTM Status

### ❌ Not Yet Integrated
- `entity_registry.py` exists (300+ lines)
- Used in old experiments
- **NOT in `train_symmetric.py`**

### Entity IDs Currently
- **Temporary**: Reset each run
- **No persistence**: "cat" in story 1 ≠ "cat" in story 2
- **No accumulation**: Knowledge lost between stories

### What LTM Would Add
```python
# With Registry:
Story 1: "cat likes milk"  → cat ID: 42
Story 2: "cat is happy"    → SAME cat ID: 42 (remembers milk!)
Story 3: "cat sleeps"      → SAME cat ID: 42 (accumulates all facts)

# Query anytime:
cat = registry.get(42)
cat.properties → {likes: milk, state: happy, action: sleeps}
```

## Integration Effort
- **Phase 1**: Basic integration → 2-3 hours
- **Phase 2**: Entity embeddings → 1-2 days
- **Phase 3**: Knowledge accumulation → 3-5 days

## Recommendation

**Tonight (1am GPU run):**
- Test TinyStories 10K without LTM
- Validate training pipeline

**Tomorrow (if successful):**
- Add entity registry (2-3 hours)
- Test on 100K stories with persistence

**Week 1:**
- Scale to 1M TinyStories
- LTM working across stories

**Week 2+:**
- Try Wikipedia or BookCorpus
- Long-term knowledge accumulation

---

**Bottom line:**
- ✅ Many corpora available (TinyStories → C4)
- ❌ LTM not integrated yet (easy to add)
- 📅 Add LTM after tonight's GPU test succeeds
