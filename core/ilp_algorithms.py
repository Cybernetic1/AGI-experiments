"""
ILP algorithms for rule discovery.

Implements several classic ILP algorithms for comparison:
1. Frequency-based mining (current approach)
2. FOIL-style (information gain)
3. Simple confidence-based scoring
"""
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
import math
from logic_core import Proposition, Rule


@dataclass
class RuleCandidate:
    """A candidate rule with quality metrics."""
    rule: Rule
    support: int  # How many examples support this rule
    confidence: float  # P(conclusion | premises)
    info_gain: float = 0.0  # Information gain (FOIL-style)
    
    def __str__(self):
        return f"{self.rule} (support={self.support}, conf={self.confidence:.3f}, gain={self.info_gain:.3f})"


def mine_frequency_based(facts: List[Proposition], max_rules: int = 10, min_support: int = 2) -> Tuple[List[Rule], List[str]]:
    """
    Original frequency-based chain rule mining.
    Finds P1(?x,?y) ^ P2(?y,?z) -> P3(?x,?z) patterns by counting co-occurrences.
    """
    counts: Dict[Tuple[str, str], int] = {}
    facts_2 = [f for f in facts if len(f.args) >= 2]
    by_first: Dict[Tuple[str, str], List[Proposition]] = {}
    
    for f in facts_2:
        by_first.setdefault((f.predicate, f.args[0]), []).append(f)
    
    for f1 in facts_2:
        mid = f1.args[1]
        for (pred, a0), lst in by_first.items():
            if a0 != mid:
                continue
            for f2 in lst:
                key = (f1.predicate, f2.predicate)
                counts[key] = counts.get(key, 0) + 1
    
    sorted_pairs = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    rules: List[Rule] = []
    pred_names: List[str] = []
    
    for (p1, p2), c in sorted_pairs:
        if c < min_support or len(rules) >= max_rules:
            continue
        concl = f"{p1}_{p2}_mined"
        pred_names.append(concl)
        rules.append(
            Rule(
                [Proposition(p1, ("?x", "?y")), Proposition(p2, ("?y", "?z"))],
                Proposition(concl, ("?x", "?z")),
                1.0,
            )
        )
    
    return rules, pred_names


def mine_foil_style(facts: List[Proposition], max_rules: int = 10, min_support: int = 2, 
                    min_info_gain: float = 0.01) -> Tuple[List[Rule], List[str]]:
    """
    FOIL-style rule mining using information gain.
    
    FOIL measures how much a rule reduces uncertainty:
    Gain = support * (log2(P+/(P+ + P-)) - log2(p+/(p+ + p-)))
    
    Where:
    - P+, P- = positive/negative examples covered by the rule
    - p+, p- = positive/negative examples covered by rule body (before adding literal)
    
    Simplified version: Score rules by support * confidence * -log(confidence)
    This favors rules that are both frequent AND informative (not too common).
    """
    # Count co-occurrences and build candidates
    candidates: Dict[Tuple[str, str], RuleCandidate] = {}
    facts_2 = [f for f in facts if len(f.args) >= 2]
    by_first: Dict[Tuple[str, str], List[Proposition]] = {}
    
    for f in facts_2:
        by_first.setdefault((f.predicate, f.args[0]), []).append(f)
    
    # Count pattern occurrences
    pattern_counts: Dict[Tuple[str, str], int] = {}
    p1_counts: Dict[str, int] = {}  # How often each predicate appears as first in chain
    
    for f1 in facts_2:
        p1_counts[f1.predicate] = p1_counts.get(f1.predicate, 0) + 1
        mid = f1.args[1]
        
        for (pred, a0), lst in by_first.items():
            if a0 != mid:
                continue
            for f2 in lst:
                key = (f1.predicate, f2.predicate)
                pattern_counts[key] = pattern_counts.get(key, 0) + 1
    
    # Calculate information gain for each pattern
    total_facts = len(facts_2)
    
    for (p1, p2), support in pattern_counts.items():
        if support < min_support:
            continue
        
        # Confidence: P(pattern | p1 appears)
        p1_freq = p1_counts.get(p1, 1)
        confidence = support / p1_freq
        
        # Information content: -log2(confidence)
        # High when rule is specific (low confidence = surprising = informative)
        if confidence > 0 and confidence < 1:
            info_content = -math.log2(confidence)
        else:
            info_content = 0.0
        
        # FOIL-style gain: support * (informativeness of rule)
        # Favors rules that are both frequent (support) and informative (not too obvious)
        info_gain = support * info_content
        
        concl = f"{p1}_{p2}_foil"
        rule = Rule(
            [Proposition(p1, ("?x", "?y")), Proposition(p2, ("?y", "?z"))],
            Proposition(concl, ("?x", "?z")),
            1.0,
        )
        
        candidates[(p1, p2)] = RuleCandidate(
            rule=rule,
            support=support,
            confidence=confidence,
            info_gain=info_gain
        )
    
    # Sort by information gain
    sorted_candidates = sorted(candidates.values(), key=lambda x: x.info_gain, reverse=True)
    
    rules: List[Rule] = []
    pred_names: List[str] = []
    
    for cand in sorted_candidates[:max_rules]:
        if cand.info_gain < min_info_gain:
            continue
        pred_names.append(cand.rule.conclusion.predicate)
        rules.append(cand.rule)
    
    return rules, pred_names


def mine_confidence_based(facts: List[Proposition], max_rules: int = 10, min_support: int = 2,
                          min_confidence: float = 0.5) -> Tuple[List[Rule], List[str]]:
    """
    Confidence-based mining (like association rule mining).
    
    Confidence = P(P2 follows P1) = count(P1 ^ P2) / count(P1)
    
    Filters rules by minimum confidence threshold.
    Favors rules with high predictive power (if P1, then P2 is likely).
    """
    counts: Dict[Tuple[str, str], int] = {}
    p1_counts: Dict[str, int] = {}
    
    facts_2 = [f for f in facts if len(f.args) >= 2]
    by_first: Dict[Tuple[str, str], List[Proposition]] = {}
    
    for f in facts_2:
        by_first.setdefault((f.predicate, f.args[0]), []).append(f)
    
    for f1 in facts_2:
        p1_counts[f1.predicate] = p1_counts.get(f1.predicate, 0) + 1
        mid = f1.args[1]
        
        for (pred, a0), lst in by_first.items():
            if a0 != mid:
                continue
            for f2 in lst:
                key = (f1.predicate, f2.predicate)
                counts[key] = counts.get(key, 0) + 1
    
    # Calculate confidence for each rule
    candidates: List[Tuple[Tuple[str, str], int, float]] = []
    for (p1, p2), support in counts.items():
        if support < min_support:
            continue
        confidence = support / p1_counts.get(p1, 1)
        if confidence >= min_confidence:
            candidates.append(((p1, p2), support, confidence))
    
    # Sort by confidence (then support as tiebreaker)
    sorted_candidates = sorted(candidates, key=lambda x: (x[2], x[1]), reverse=True)
    
    rules: List[Rule] = []
    pred_names: List[str] = []
    
    for (p1, p2), support, conf in sorted_candidates[:max_rules]:
        concl = f"{p1}_{p2}_conf"
        pred_names.append(concl)
        rules.append(
            Rule(
                [Proposition(p1, ("?x", "?y")), Proposition(p2, ("?y", "?z"))],
                Proposition(concl, ("?x", "?z")),
                1.0,
            )
        )
    
    return rules, pred_names


def compare_algorithms(facts: List[Proposition], max_rules: int = 10, min_support: int = 2) -> Dict[str, Tuple[List[Rule], List[str]]]:
    """
    Run all ILP algorithms and return results for comparison.
    
    Returns:
        Dict mapping algorithm name -> (rules, pred_names)
    """
    results = {}
    
    print("\n" + "="*70)
    print("COMPARING ILP ALGORITHMS")
    print("="*70)
    
    # Frequency-based (current)
    print("\n[1] Frequency-based mining...")
    freq_rules, freq_preds = mine_frequency_based(facts, max_rules, min_support)
    results['frequency'] = (freq_rules, freq_preds)
    print(f"  Generated {len(freq_rules)} rules")
    
    # FOIL-style
    print("\n[2] FOIL-style (information gain)...")
    foil_rules, foil_preds = mine_foil_style(facts, max_rules, min_support)
    results['foil'] = (foil_rules, foil_preds)
    print(f"  Generated {len(foil_rules)} rules")
    
    # Confidence-based
    print("\n[3] Confidence-based mining...")
    conf_rules, conf_preds = mine_confidence_based(facts, max_rules, min_support, min_confidence=0.3)
    results['confidence'] = (conf_rules, conf_preds)
    print(f"  Generated {len(conf_rules)} rules")
    
    # Show overlap
    freq_set = set(freq_preds)
    foil_set = set([p.replace('_foil', '_mined') for p in foil_preds])
    conf_set = set([p.replace('_conf', '_mined') for p in conf_preds])
    
    print(f"\nRule overlap:")
    print(f"  Frequency ∩ FOIL: {len(freq_set & foil_set)} rules")
    print(f"  Frequency ∩ Confidence: {len(freq_set & conf_set)} rules")
    print(f"  FOIL ∩ Confidence: {len(foil_set & conf_set)} rules")
    print(f"  All three: {len(freq_set & foil_set & conf_set)} rules")
    
    print("="*70)
    
    return results
