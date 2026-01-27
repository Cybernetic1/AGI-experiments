#!/usr/bin/env python3
"""
Quick script to measure actual DLN parameters at different scales.
Run this to get real numbers for the compression graph.
"""
import sys
from pathlib import Path

# Add venv activation reminder
print("=" * 60)
print("Make sure to activate venv first:")
print("  source venv/bin/activate")
print("=" * 60)

try:
    from count_parameters import estimate_from_tinystories_data
    
    scales = [10, 50, 100, 200, 500]
    
    print("\n" + "="*60)
    print("MEASURING DLN PARAMETER COUNTS AT DIFFERENT SCALES")
    print("="*60)
    
    results = {}
    
    for stories in scales:
        print(f"\n{'='*60}")
        print(f"Testing with {stories} stories:")
        print('='*60)
        counts = estimate_from_tinystories_data(
            max_stories=stories,
            embed_dim=32
        )
        if counts:
            results[stories] = counts['total']
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY FOR GRAPH")
    print("="*60)
    print("\nPaste these into compare_model_sizes.py:")
    print("\ndln_estimates = [")
    
    capability_map = {
        10: "Basic",
        50: "Coherent", 
        100: "Good",
        200: "Good",
        500: "Strong"
    }
    
    for stories in scales:
        if stories in results:
            cap = capability_map.get(stories, "Unknown")
            print(f'    {{"name": "DLN\\n({stories} stories)", "params": {results[stories]}, "capability": "{cap}"}},')
    
    print("]")
    print("\n" + "="*60)

except ImportError as e:
    print(f"\nError: {e}")
    print("\nMake sure you're in the venv and have required dependencies:")
    print("  source venv/bin/activate")
    print("  pip install torch")
    sys.exit(1)

except FileNotFoundError as e:
    print(f"\nError: {e}")
    print("\nMake sure data/processed/tinystories_train.json exists")
    print("Run preprocessing first if needed")
    sys.exit(1)
