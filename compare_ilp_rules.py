#!/usr/bin/env python3
"""
Simple ILP rule comparison - NO label generation, NO training.
Just compares which rules each algorithm discovers.
"""
import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pipelines.tinystories_pipeline import load_tinystories_facts
from core.ilp_algorithms import mine_frequency_based, mine_foil_style, mine_confidence_based


def main():
    parser = argparse.ArgumentParser(description='Compare ILP algorithms (rules only, no training)')
    parser.add_argument('--max-stories', type=int, default=50, help='Stories to load')
    parser.add_argument('--max-facts', type=int, default=None, help='Max facts (None = no limit)')
    parser.add_argument('--max-rules', type=int, default=20, help='Max rules per algorithm')
    parser.add_argument('--min-support', type=int, default=2, help='Min support for rules')
    parser.add_argument('--show-rules', action='store_true', help='Print all rules discovered')
    args = parser.parse_args()
    
    print("="*70)
    print("ILP RULE COMPARISON (NO TRAINING)")
    print("="*70)
    print(f"Loading {args.max_stories} stories...")
    
    # Load facts
    facts = load_tinystories_facts(
        max_stories=args.max_stories,
        max_facts=args.max_facts if args.max_facts else 999999
    )
    print(f"Loaded {len(facts)} facts\n")
    
    print("="*70)
    print("MINING RULES")
    print("="*70)
    
    # Mine rules with each algorithm
    results = {}
    
    print("\n[1] Frequency-based mining...")
    freq_rules, freq_preds = mine_frequency_based(facts, args.max_rules, args.min_support)
    results['Frequency'] = (freq_rules, freq_preds)
    print(f"    ✅ Discovered {len(freq_rules)} rules")
    
    print("\n[2] FOIL-style (information gain)...")
    foil_rules, foil_preds = mine_foil_style(facts, args.max_rules, args.min_support)
    results['FOIL'] = (foil_rules, foil_preds)
    print(f"    ✅ Discovered {len(foil_rules)} rules")
    
    print("\n[3] Confidence-based...")
    conf_rules, conf_preds = mine_confidence_based(facts, args.max_rules, args.min_support, min_confidence=0.3)
    results['Confidence'] = (conf_rules, conf_preds)
    print(f"    ✅ Discovered {len(conf_rules)} rules")
    
    # Show overlap
    freq_set = set(freq_preds)
    foil_set = set([p.replace('_foil', '_mined') for p in foil_preds])
    conf_set = set([p.replace('_conf', '_mined') for p in conf_preds])
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\n{'Algorithm':<15} {'Rules':<10} {'Sample Rules'}")
    print("-" * 70)
    for name, (rules, preds) in results.items():
        sample = ", ".join([r.conclusion.predicate[:20] for r in rules[:3]])
        if len(rules) > 3:
            sample += ", ..."
        print(f"{name:<15} {len(rules):<10} {sample}")
    
    print(f"\nRule overlap:")
    print(f"  Frequency ∩ FOIL: {len(freq_set & foil_set)} rules")
    print(f"  Frequency ∩ Confidence: {len(freq_set & conf_set)} rules")
    print(f"  FOIL ∩ Confidence: {len(foil_set & conf_set)} rules")
    print(f"  All three agree: {len(freq_set & foil_set & conf_set)} rules")
    
    # Show detailed rules if requested
    if args.show_rules:
        print("\n" + "="*70)
        print("DETAILED RULES")
        print("="*70)
        for name, (rules, preds) in results.items():
            print(f"\n{name} ({len(rules)} rules):")
            print("-" * 70)
            for i, rule in enumerate(rules, 1):
                print(f"{i:2d}. {rule}")
    
    print("\n" + "="*70)
    print("✅ COMPARISON COMPLETE (no training needed!)")
    print("="*70)


if __name__ == '__main__':
    main()
