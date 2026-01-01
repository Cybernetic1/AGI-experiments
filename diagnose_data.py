"""
Quick diagnostic script to check data ranges before training.
Helps identify index out of bounds issues.
"""

import torch
from train_symmetric import TinyStoriesLogicDataset

print("=" * 70)
print("Diagnostic: Checking Data Ranges")
print("=" * 70)

# Create small dataset
print("\nCreating dataset with 10 stories...")
try:
    dataset = TinyStoriesLogicDataset(num_stories=10, max_seq_len=20)
    
    print(f"\nDataset created successfully!")
    print(f"  Samples: {len(dataset)}")
    print(f"  Vocabulary size: {len(dataset.vocab)}")
    print(f"  Entity count: {dataset.next_entity_id}")
    
    # Check a few samples
    print(f"\nChecking sample data ranges...")
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        text_ids = sample['text_ids']
        propositions = sample['propositions']
        
        # Check text IDs
        max_text_id = text_ids.max().item()
        min_text_id = text_ids.min().item()
        
        # Check proposition IDs
        max_prop_entity = propositions[:, 0].max().item()
        max_prop_relation = propositions[:, 1].max().item()
        max_prop_value = propositions[:, 2].max().item()
        
        print(f"\n  Sample {i}:")
        print(f"    Text IDs: min={min_text_id}, max={max_text_id}, vocab_size={len(dataset.vocab)}")
        print(f"    Entities: max={max_prop_entity}, entity_count={dataset.next_entity_id}")
        print(f"    Relations: max={max_prop_relation}, vocab_size={len(dataset.vocab)}")
        print(f"    Values: max={max_prop_value}, entity_count={dataset.next_entity_id}")
        
        # Check for out of bounds
        if max_text_id >= len(dataset.vocab):
            print(f"    ⚠️  WARNING: Text ID {max_text_id} >= vocab size {len(dataset.vocab)}")
        
        if max_prop_entity >= dataset.next_entity_id:
            print(f"    ⚠️  WARNING: Entity ID {max_prop_entity} >= entity count {dataset.next_entity_id}")
        
        if max_prop_relation >= len(dataset.vocab):
            print(f"    ⚠️  WARNING: Relation ID {max_prop_relation} >= vocab size {len(dataset.vocab)}")
        
        if max_prop_value >= dataset.next_entity_id and max_prop_value >= len(dataset.vocab):
            print(f"    ⚠️  WARNING: Value ID {max_prop_value} out of bounds")
    
    print("\n" + "=" * 70)
    print("Diagnostic complete!")
    print("=" * 70)
    
    print("\nSummary:")
    print(f"  ✓ Vocabulary size: {len(dataset.vocab)}")
    print(f"  ✓ Entity count: {dataset.next_entity_id}")
    print(f"  ✓ Max text ID seen: {max_text_id}")
    print(f"  ✓ Max entity ID seen: {max_prop_entity}")
    
except Exception as e:
    print(f"\n✗ Error during dataset creation: {e}")
    import traceback
    traceback.print_exc()
