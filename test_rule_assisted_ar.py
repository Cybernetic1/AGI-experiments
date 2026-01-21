"""
Rule-Assisted Semantic Event-Level AR
======================================

Combines symbolic rules with neural semantic prediction:
1. Mine/evolve rules to capture story patterns
2. Use rules to constrain verb predictions (prune search space)
3. Neural model picks best verb from constrained candidates

This tests whether GA/ILP rules help as INFERENCE-TIME CONSTRAINTS
rather than training-time label generators.

Key innovation: Rules guide prediction, not generate training labels.

Usage:
    # With ILP rules (fast)
    python test_rule_assisted_ar.py --scale full --rule-source ilp --device cuda
    
    # With GA-evolved rules (slower, potentially better)
    python test_rule_assisted_ar.py --scale full --rule-source ga --ga-generations 10 --device cuda
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import time
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Set
import argparse
from collections import defaultdict, Counter

from pipelines.tinystories_pipeline import load_tinystories_facts
from logic_core import Proposition, Rule, SymbolicEngine
from core.ilp_algorithms import mine_frequency_based, mine_foil_style, mine_confidence_based
from hybrid_ga_ilp_dln import HybridRuleDiscovery
from test_semantic_event_ar import (
    download_glove_embeddings,
    create_verb_embedding_matrix,
    SemanticEventAR,
    train_semantic_event_ar,
    SCALE_CONFIGS
)
from test_event_level_ar import (
    group_facts_into_events,
    create_event_sequences,
    create_event_ar_training_data
)


def extract_verb_constraints_from_rules(
    rules: List[Rule],
    verbose: bool = False
) -> Dict[str, Dict[str, Set[str]]]:
    """
    Extract verb prediction constraints from rules.
    
    Rules like: agent(e,x) ∧ patient(e,y) → type(e, 'verb')
    Tell us: when we see agent+patient pattern, expect certain verb categories.
    
    Returns:
        Dict mapping (premise_pattern → {verbs, predicate_types})
    """
    constraints = defaultdict(lambda: {'verbs': set(), 'predicates': set()})
    
    for rule in rules:
        # Extract premise pattern (predicate names)
        premise_pattern = tuple(sorted([p.predicate for p in rule.premises]))
        
        # Extract conclusion predicate/verb
        conclusion_pred = rule.conclusion.predicate
        
        # If conclusion is type(e, verb), extract the verb
        if 'type' in conclusion_pred or 'mined' in conclusion_pred:
            # This rule suggests certain verbs given context
            constraints[premise_pattern]['predicates'].add(conclusion_pred)
        
        # Store the rule's conclusion predicate as a constraint
        constraints[premise_pattern]['predicates'].add(conclusion_pred)
    
    if verbose:
        print(f"\n[Rule Constraint Extraction]")
        print(f"  Extracted {len(constraints)} premise patterns")
        for pattern, info in list(constraints.items())[:5]:
            print(f"    {pattern} → {len(info['predicates'])} conclusion types")
    
    return dict(constraints)


def extract_story_grammar_from_rules(
    rules: List[Rule],
    facts: List[Proposition],
    verbose: bool = False
) -> Dict[str, Counter]:
    """
    Extract story grammar: what verbs typically follow what verbs.
    
    Analyzes facts to find verb sequences, then uses rules to generalize.
    
    Returns:
        Dict[verb -> Counter(next_verbs)] with frequencies
    """
    # Extract verb sequences from facts
    events = group_facts_into_events(facts)
    stories = create_event_sequences(events, min_event_props=2)
    
    verb_transitions = defaultdict(Counter)
    
    for story in stories:
        # Get verb sequence for this story
        verb_sequence = []
        for event in story:
            for prop in event:
                if prop.predicate == 'type' and len(prop.args) >= 2:
                    verb_sequence.append(prop.args[1])
                    break
        
        # Count transitions
        for i in range(len(verb_sequence) - 1):
            current_verb = verb_sequence[i]
            next_verb = verb_sequence[i + 1]
            verb_transitions[current_verb][next_verb] += 1
    
    if verbose:
        print(f"\n[Story Grammar Extraction]")
        print(f"  Extracted transitions for {len(verb_transitions)} verbs")
        
        # Show most common transitions
        print(f"  Top verb transitions:")
        for verb, next_verbs in list(verb_transitions.items())[:5]:
            top_next = next_verbs.most_common(3)
            print(f"    {verb} → {top_next}")
    
    return dict(verb_transitions)


def apply_rule_constraints(
    context_events: List[List[Proposition]],
    all_verbs: List[str],
    verb_transitions: Dict[str, Counter],
    top_k_grammar: int = 50,
    fallback_ratio: float = 0.3
) -> Set[str]:
    """
    Use rules to constrain which verbs are plausible given context.
    
    Args:
        context_events: Previous events in story
        all_verbs: Full verb vocabulary
        verb_transitions: Story grammar (verb → next verbs)
        top_k_grammar: How many verbs to consider from grammar
        fallback_ratio: If no grammar match, use this fraction of vocabulary
    
    Returns:
        Set of plausible verbs (constrained search space)
    """
    # Get last event's verb
    last_verb = None
    if context_events:
        last_event = context_events[-1]
        for prop in last_event:
            if prop.predicate == 'type' and len(prop.args) >= 2:
                last_verb = prop.args[1]
                break
    
    # Apply story grammar constraint
    if last_verb and last_verb in verb_transitions:
        # Get most likely next verbs from grammar
        next_verb_counts = verb_transitions[last_verb]
        candidate_verbs = {verb for verb, _ in next_verb_counts.most_common(top_k_grammar)}
        
        # If too few, add fallback
        if len(candidate_verbs) < 10:
            # Add random sample as fallback
            num_fallback = int(len(all_verbs) * fallback_ratio)
            fallback = set(random.sample(all_verbs, min(num_fallback, len(all_verbs))))
            candidate_verbs.update(fallback)
        
        return candidate_verbs
    else:
        # No grammar match - use large fallback
        num_fallback = int(len(all_verbs) * 0.5)  # Use half the vocabulary
        return set(random.sample(all_verbs, min(num_fallback, len(all_verbs))))


class RuleAssistedSemanticEventAR(SemanticEventAR):
    """
    Semantic Event AR with rule-based verb constraints.
    """
    
    def __init__(self, *args, verb_transitions: Dict[str, Counter] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.verb_transitions = verb_transitions or {}
    
    def predict_verb_constrained(
        self,
        context_events: List[List[Proposition]],
        device: str = "cpu",
        top_k: int = 5,
        use_constraints: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Predict verb with rule-based constraints.
        
        If use_constraints=True, only consider verbs allowed by story grammar.
        """
        # Get context representation
        context_props = []
        for event in context_events:
            context_props.extend(event)
        
        if not context_props:
            return []
        
        context_emb = self.encode_context(context_props, device)
        pred_verb_emb = self.pred_head(context_emb)
        
        # Apply rule constraints
        if use_constraints and self.verb_transitions:
            candidate_verbs = apply_rule_constraints(
                context_events,
                self.verb_names,
                self.verb_transitions,
                top_k_grammar=50,
                fallback_ratio=0.3
            )
            candidate_indices = [self.verb_vocab[v] for v in candidate_verbs if v in self.verb_vocab]
        else:
            # No constraints - use all verbs
            candidate_indices = list(range(len(self.verb_names)))
        
        if not candidate_indices:
            # Fallback to all verbs if no candidates
            candidate_indices = list(range(len(self.verb_names)))
        
        # Get embeddings for candidate verbs only
        candidate_embs = self.verb_embed.weight[candidate_indices]  # (num_candidates, embed_dim)
        
        # Compute cosine similarity
        pred_norm = F.normalize(pred_verb_emb, dim=-1)
        candidate_norm = F.normalize(candidate_embs, dim=-1)
        
        similarities = torch.matmul(pred_norm, candidate_norm.T).squeeze(0)  # (num_candidates,)
        
        # Top-k from candidates
        top_similarities, top_local_indices = torch.topk(
            similarities,
            min(top_k, len(candidate_indices))
        )
        
        # Map back to global verb indices
        results = []
        for local_idx, sim in zip(top_local_indices, top_similarities):
            global_idx = candidate_indices[local_idx.item()]
            verb = self.verb_names[global_idx]
            results.append((verb, sim.item()))
        
        return results


def evaluate_rule_assisted_ar(
    model: RuleAssistedSemanticEventAR,
    eval_pairs: List[Tuple[List[List[Proposition]], List[Proposition]]],
    device: str = "cpu",
    use_constraints: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Evaluate with and without rule constraints for comparison.
    """
    model.eval()
    
    results = {
        'with_constraints': {'top1': 0, 'top5': 0, 'similarity': 0.0, 'count': 0},
        'without_constraints': {'top1': 0, 'top5': 0, 'similarity': 0.0, 'count': 0}
    }
    
    with torch.no_grad():
        for context_events, target_event in eval_pairs:
            if not context_events or not target_event:
                continue
            
            # Get target verb
            target_verb = None
            for prop in target_event:
                if prop.predicate == 'type' and len(prop.args) >= 2:
                    target_verb = prop.args[1]
                    break
            
            if not target_verb or target_verb not in model.verb_vocab:
                continue
            
            # Flatten context
            context_props = []
            for event in context_events:
                context_props.extend(event)
            
            if not context_props:
                continue
            
            try:
                # Compute similarity
                pred_emb, target_emb = model(context_props, target_verb, device)
                pred_norm = F.normalize(pred_emb, dim=-1)
                target_norm = F.normalize(target_emb, dim=-1)
                similarity = F.cosine_similarity(pred_norm, target_norm, dim=-1).item()
                
                # Evaluate WITH constraints
                predictions_constrained = model.predict_verb_constrained(
                    context_events, device, top_k=5, use_constraints=True
                )
                pred_verbs_c = [v for v, _ in predictions_constrained]
                
                if pred_verbs_c and pred_verbs_c[0] == target_verb:
                    results['with_constraints']['top1'] += 1
                if target_verb in pred_verbs_c:
                    results['with_constraints']['top5'] += 1
                results['with_constraints']['similarity'] += similarity
                results['with_constraints']['count'] += 1
                
                # Evaluate WITHOUT constraints
                predictions_unconstrained = model.predict_verb_constrained(
                    context_events, device, top_k=5, use_constraints=False
                )
                pred_verbs_u = [v for v, _ in predictions_unconstrained]
                
                if pred_verbs_u and pred_verbs_u[0] == target_verb:
                    results['without_constraints']['top1'] += 1
                if target_verb in pred_verbs_u:
                    results['without_constraints']['top5'] += 1
                results['without_constraints']['similarity'] += similarity
                results['without_constraints']['count'] += 1
                
            except Exception as e:
                continue
    
    # Compute averages
    for key in ['with_constraints', 'without_constraints']:
        count = results[key]['count']
        if count > 0:
            results[key]['top1_acc'] = results[key]['top1'] / count
            results[key]['top5_acc'] = results[key]['top5'] / count
            results[key]['avg_similarity'] = results[key]['similarity'] / count
    
    if verbose:
        print(f"\n{'='*70}")
        print("RULE-ASSISTED EVALUATION")
        print(f"{'='*70}")
        print(f"\nWithout constraints (baseline):")
        print(f"  Top-1: {results['without_constraints'].get('top1_acc', 0):.4f}")
        print(f"  Top-5: {results['without_constraints'].get('top5_acc', 0):.4f}")
        print(f"  Similarity: {results['without_constraints'].get('avg_similarity', 0):.4f}")
        
        print(f"\nWith rule constraints:")
        print(f"  Top-1: {results['with_constraints'].get('top1_acc', 0):.4f}")
        print(f"  Top-5: {results['with_constraints'].get('top5_acc', 0):.4f}")
        print(f"  Similarity: {results['with_constraints'].get('avg_similarity', 0):.4f}")
        
        # Improvement
        top5_improvement = (
            results['with_constraints'].get('top5_acc', 0) - 
            results['without_constraints'].get('top5_acc', 0)
        )
        if top5_improvement > 0.01:
            print(f"\n✅ Rules improve Top-5 by {top5_improvement*100:.1f}%")
        else:
            print(f"\n⚠️  Rules don't significantly improve predictions")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Rule-Assisted Semantic Event AR')
    parser.add_argument('--scale', default='full', choices=list(SCALE_CONFIGS.keys()))
    parser.add_argument('--corpus', default='data/processed/tinystories_train.json')
    parser.add_argument('--rule-source', default='ilp', choices=['ilp', 'ga', 'none'],
                       help='Source of rules: ILP (fast), GA (evolved), or none (baseline)')
    parser.add_argument('--ilp-algorithm', default='frequency',
                       choices=['frequency', 'foil', 'confidence'])
    parser.add_argument('--ilp-rules', type=int, default=50, help='Number of ILP rules')
    parser.add_argument('--ga-generations', type=int, default=10, help='GA generations (if using GA)')
    parser.add_argument('--embed-dim', type=int, default=50, choices=[50, 100, 200, 300])
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--context-size', type=int, default=3)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--output-dir', default='outputs/rule_assisted_ar')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if verbose:
        print("\n" + "="*70)
        print("RULE-ASSISTED SEMANTIC EVENT-LEVEL AR")
        print("="*70)
        print(f"\nRule source: {args.rule_source}")
    
    # Load corpus
    if verbose:
        print(f"\n[1] Loading corpus ({args.scale})...")
    
    config = SCALE_CONFIGS[args.scale]
    facts = load_tinystories_facts(
        max_stories=config['stories'],
        max_facts=config['facts'],
        path=args.corpus
    )
    
    if verbose:
        print(f"  ✅ Loaded {len(facts):,} facts")
    
    # Discover/mine rules
    rules = []
    if args.rule_source != 'none':
        if verbose:
            print(f"\n[2] Rule discovery ({args.rule_source})...")
        
        if args.rule_source == 'ilp':
            if args.ilp_algorithm == 'frequency':
                rules, _ = mine_frequency_based(facts, args.ilp_rules, min_support=2)
            elif args.ilp_algorithm == 'foil':
                rules, _ = mine_foil_style(facts, args.ilp_rules, min_support=2)
            elif args.ilp_algorithm == 'confidence':
                rules, _ = mine_confidence_based(facts, args.ilp_rules, min_support=2, min_confidence=0.3)
            
            if verbose:
                print(f"  ✅ Mined {len(rules)} ILP rules")
        
        elif args.rule_source == 'ga':
            discoverer = HybridRuleDiscovery(
                ilp_algorithm=args.ilp_algorithm,
                ilp_rules=args.ilp_rules,
                ga_generations=args.ga_generations,
                sample_facts_for_fitness=min(2000, len(facts)),
                verbose=verbose
            )
            rules = discoverer.discover(facts)
            
            if verbose:
                print(f"  ✅ Evolved {len(rules)} GA rules (fitness: {discoverer.best_fitness:.4f})")
    
    # Extract story grammar from rules and facts
    if verbose:
        print(f"\n[3] Extracting story grammar...")
    
    verb_transitions = extract_story_grammar_from_rules(rules, facts, verbose=verbose)
    
    # Load GloVe
    if verbose:
        print(f"\n[4] Loading GloVe embeddings...")
    
    glove_embeddings = download_glove_embeddings(embed_dim=args.embed_dim)
    
    # Create training data
    if verbose:
        print(f"\n[5] Creating training data...")
    
    training_pairs = create_event_ar_training_data(
        facts,
        context_size=args.context_size,
        max_samples=999999
    )
    
    if verbose:
        print(f"  ✅ Created {len(training_pairs):,} pairs")
    
    # Vocabularies
    all_predicates = set(f.predicate for f in facts)
    all_args = set()
    all_verbs = set()
    
    for f in facts:
        all_args.update(f.args)
        if f.predicate == 'type' and len(f.args) >= 2:
            all_verbs.add(f.args[1])
    
    verb_list = sorted(list(all_verbs))
    
    if verbose:
        print(f"  Vocabulary: {len(verb_list)} verbs")
    
    # Create embeddings
    verb_embedding_matrix, num_oov = create_verb_embedding_matrix(
        verb_list, glove_embeddings, args.embed_dim
    )
    
    # Create model
    if verbose:
        print(f"\n[6] Creating rule-assisted model...")
    
    model = RuleAssistedSemanticEventAR(
        predicates=list(all_predicates),
        args=list(all_args),
        verbs=verb_list,
        embed_dim=args.embed_dim,
        verb_embeddings=verb_embedding_matrix,
        verb_transitions=verb_transitions
    )
    
    if verbose:
        print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Story grammar: {len(verb_transitions)} verb transitions")
    
    # Train (standard semantic AR training)
    if verbose:
        print(f"\n[7] Training...")
    
    train_results = train_semantic_event_ar(
        model,
        training_pairs,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=verbose
    )
    
    # Evaluate with constraints
    if verbose:
        print(f"\n[8] Evaluating with rule constraints...")
    
    split_idx = int(0.8 * len(training_pairs))
    eval_pairs = training_pairs[split_idx:]
    
    eval_results = evaluate_rule_assisted_ar(
        model,
        eval_pairs,
        device=args.device,
        use_constraints=True,
        verbose=verbose
    )
    
    # Save
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / 'results.json', 'w') as f:
        json.dump({
            'config': vars(args),
            'num_rules': len(rules),
            'num_verb_transitions': len(verb_transitions),
            'training': train_results,
            'evaluation': eval_results
        }, f, indent=2)
    
    if verbose:
        print(f"\n✅ Results saved to {output_path}")


if __name__ == '__main__':
    main()
