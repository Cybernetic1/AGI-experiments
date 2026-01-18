"""
Semantic-AR Training with Hybrid GA+ILP+DLN Rules
==================================================

Integrates hybrid rule discovery with semantic autoregressive training.

Architecture:
1. Load corpus (TinyStories or custom)
2. Extract facts using Davidsonian parser
3. Discover rules using Hybrid GA+ILP+DLN
4. Train semantic-AR model on large corpus
5. Evaluate on held-out test set

This combines:
- Symbolic rule discovery (interpretable)
- Neural learning (scalable)  
- Semantic objectives (meaningful representations)
"""

import torch
import torch.nn as nn
import json
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import argparse

from pipelines.tinystories_pipeline import load_tinystories_facts
from hybrid_ga_ilp_dln import hybrid_discover_rules, HybridRuleDiscovery
from logic_core import Proposition, Rule, SymbolicEngine
from label_utils import _collect_labels
from dln import SimpleDLN
from core.train_utils import _train_on_labels, _eval_on_labels
from davidsonian_extraction import DavidsonianExtractor


class SemanticARPipeline:
    """
    Full pipeline for semantic-AR training with hybrid rule discovery.
    """
    
    def __init__(
        self,
        corpus_path: str = 'data/processed/tinystories_train.json',
        rule_discovery_stories: int = 100,
        rule_discovery_facts: int = 10000,
        ilp_algorithm: str = 'frequency',
        ilp_rules: int = 30,
        ga_generations: int = 20,
        training_stories: int = 1000,
        training_facts: int = 50000,
        dln_embed_dim: int = 64,
        dln_training_steps: int = 100,
        verbose: bool = True
    ):
        self.corpus_path = corpus_path
        self.rule_discovery_stories = rule_discovery_stories
        self.rule_discovery_facts = rule_discovery_facts
        self.ilp_algorithm = ilp_algorithm
        self.ilp_rules = ilp_rules
        self.ga_generations = ga_generations
        self.training_stories = training_stories
        self.training_facts = training_facts
        self.dln_embed_dim = dln_embed_dim
        self.dln_training_steps = dln_training_steps
        self.verbose = verbose
        
        self.extractor = DavidsonianExtractor()
        self.rules = None
        self.dln_model = None
        self.metrics = {}
    
    def run(self):
        """Execute full pipeline."""
        if self.verbose:
            print("\n" + "="*70)
            print("SEMANTIC-AR WITH HYBRID GA+ILP+DLN")
            print("="*70)
        
        # Stage 1: Rule Discovery
        if self.verbose:
            print(f"\n[Stage 1] Rule Discovery on {self.rule_discovery_stories} stories")
        
        discovery_facts = load_tinystories_facts(
            max_stories=self.rule_discovery_stories,
            max_facts=self.rule_discovery_facts,
            path=self.corpus_path
        )
        
        if self.verbose:
            print(f"  Loaded {len(discovery_facts)} facts for rule discovery")
        
        # Discover rules using hybrid approach
        discoverer = HybridRuleDiscovery(
            ilp_algorithm=self.ilp_algorithm,
            ilp_rules=self.ilp_rules,
            ga_generations=self.ga_generations,
            sample_facts_for_fitness=1000,
            verbose=self.verbose
        )
        
        self.rules = discoverer.discover(discovery_facts)
        
        if self.verbose:
            print(f"\n  ✅ Discovered {len(self.rules)} rules")
            print(f"  Best fitness: {discoverer.best_fitness:.4f}")
        
        self.metrics['rule_discovery'] = {
            'num_rules': len(self.rules),
            'best_fitness': discoverer.best_fitness,
            'evolution_history': discoverer.history
        }
        
        # Stage 2: Label Generation on Larger Corpus
        if self.verbose:
            print(f"\n[Stage 2] Label Generation on {self.training_stories} stories")
        
        training_facts = load_tinystories_facts(
            max_stories=self.training_stories,
            max_facts=self.training_facts,
            path=self.corpus_path
        )
        
        if self.verbose:
            print(f"  Loaded {len(training_facts)} facts for training")
            print(f"  Generating labels with {len(self.rules)} rules...")
        
        # Generate labels using discovered rules
        labels_dict = _collect_labels(
            training_facts,
            self.rules,
            log_progress=self.verbose
        )
        
        if self.verbose:
            print(f"  ✅ Generated {len(labels_dict)} labels")
        
        self.metrics['label_generation'] = {
            'num_facts': len(training_facts),
            'num_labels': len(labels_dict),
            'expansion_ratio': len(labels_dict) / len(training_facts) if training_facts else 0
        }
        
        # Stage 3: DLN Training
        if self.verbose:
            print(f"\n[Stage 3] DLN Training on {len(labels_dict)} labels")
        
        # Collect vocabularies
        all_predicates = set()
        all_args = set()
        for fact in training_facts:
            all_predicates.add(fact.predicate)
            all_args.update(fact.args)
        for rule in self.rules:
            all_predicates.add(rule.conclusion.predicate)
        
        if self.verbose:
            print(f"  Vocabulary: {len(all_predicates)} predicates, {len(all_args)} entities")
        
        # Create DLN model
        self.dln_model = SimpleDLN(
            predicates=list(all_predicates),
            args=list(all_args),
            embed_dim=self.dln_embed_dim
        )
        
        # Split train/eval
        labels_list = [(k, v) for k, v in labels_dict.items()]
        split_idx = int(0.8 * len(labels_list))
        train_labels = {k: v for k, v in labels_list[:split_idx]}
        eval_labels = {k: v for k, v in labels_list[split_idx:]}
        
        if self.verbose:
            print(f"  Train: {len(train_labels)} labels, Eval: {len(eval_labels)} labels")
            print(f"  Training for {self.dln_training_steps} steps...")
        
        # Train DLN
        optimizer = torch.optim.Adam(self.dln_model.parameters(), lr=0.001)
        
        # Training with progress
        for step in range(self.dln_training_steps):
            train_mse = _train_on_labels(
                self.dln_model,
                optimizer,
                training_facts,
                train_labels,
                steps=1,
                batch_size=None
            )
            
            if self.verbose and (step + 1) % max(1, self.dln_training_steps // 10) == 0:
                eval_mse = _eval_on_labels(self.dln_model, training_facts, eval_labels)
                print(f"    Step {step + 1}/{self.dln_training_steps}: Train MSE={train_mse:.6f}, Eval MSE={eval_mse:.6f}")
        
        # Final evaluation
        final_train_mse = _train_on_labels(
            self.dln_model,
            optimizer,
            training_facts,
            train_labels,
            steps=0,  # Just evaluate
            batch_size=None
        )
        final_eval_mse = _eval_on_labels(self.dln_model, training_facts, eval_labels)
        
        self.metrics['training'] = {
            'train_mse': final_train_mse,
            'eval_mse': final_eval_mse,
            'num_parameters': sum(p.numel() for p in self.dln_model.parameters())
        }
        
        if self.verbose:
            print(f"\n  ✅ Training complete")
            print(f"  Final Train MSE: {final_train_mse:.6f}")
            print(f"  Final Eval MSE: {final_eval_mse:.6f}")
            print(f"  Model parameters: {self.metrics['training']['num_parameters']:,}")
        
        # Stage 4: Summary
        self._print_summary()
        
        return self.rules, self.dln_model, self.metrics
    
    def _print_summary(self):
        """Print final summary."""
        if not self.verbose:
            return
        
        print("\n" + "="*70)
        print("PIPELINE SUMMARY")
        print("="*70)
        
        print("\n[Rule Discovery]")
        print(f"  Stories: {self.rule_discovery_stories}")
        print(f"  Facts: {self.metrics['rule_discovery'].get('num_rules', 0)}")
        print(f"  Rules discovered: {len(self.rules)}")
        print(f"  Best fitness: {self.metrics['rule_discovery'].get('best_fitness', 0):.4f}")
        
        print("\n[Label Generation]")
        print(f"  Training facts: {self.metrics['label_generation']['num_facts']}")
        print(f"  Labels generated: {self.metrics['label_generation']['num_labels']}")
        print(f"  Expansion ratio: {self.metrics['label_generation']['expansion_ratio']:.1f}×")
        
        print("\n[DLN Training]")
        print(f"  Train MSE: {self.metrics['training']['train_mse']:.6f}")
        print(f"  Eval MSE: {self.metrics['training']['eval_mse']:.6f}")
        print(f"  Parameters: {self.metrics['training']['num_parameters']:,}")
        
        print("\n[Top 5 Evolved Rules]")
        for i, rule in enumerate(self.rules[:5], 1):
            print(f"  {i}. {rule}")
        
        print("\n" + "="*70)
    
    def save(self, output_dir: str = 'outputs/hybrid_semantic_ar'):
        """Save trained model and discovered rules."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save DLN model
        if self.dln_model:
            torch.save(self.dln_model.state_dict(), output_path / 'dln_model.pt')
            if self.verbose:
                print(f"✅ Saved model to {output_path / 'dln_model.pt'}")
        
        # Save rules
        if self.rules:
            rules_json = [{
                'premises': [{'pred': p.predicate, 'args': p.args, 'truth': p.truth} for p in r.premises],
                'conclusion': {'pred': r.conclusion.predicate, 'args': r.conclusion.args, 'truth': r.conclusion.truth},
                'weight': r.weight
            } for r in self.rules]
            
            with open(output_path / 'evolved_rules.json', 'w') as f:
                json.dump(rules_json, f, indent=2)
            if self.verbose:
                print(f"✅ Saved rules to {output_path / 'evolved_rules.json'}")
        
        # Save metrics
        with open(output_path / 'metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)
        if self.verbose:
            print(f"✅ Saved metrics to {output_path / 'metrics.json'}")


def main():
    parser = argparse.ArgumentParser(description='Hybrid GA+ILP+DLN Semantic-AR Training')
    parser.add_argument('--corpus', default='data/processed/tinystories_train.json', help='Corpus path')
    parser.add_argument('--discovery-stories', type=int, default=100, help='Stories for rule discovery')
    parser.add_argument('--discovery-facts', type=int, default=10000, help='Facts for rule discovery')
    parser.add_argument('--ilp-algorithm', choices=['frequency', 'foil', 'confidence'], default='frequency')
    parser.add_argument('--ilp-rules', type=int, default=30, help='Initial ILP rules')
    parser.add_argument('--ga-generations', type=int, default=20, help='GA generations')
    parser.add_argument('--training-stories', type=int, default=1000, help='Stories for training')
    parser.add_argument('--training-facts', type=int, default=50000, help='Facts for training')
    parser.add_argument('--embed-dim', type=int, default=64, help='DLN embedding dimension')
    parser.add_argument('--training-steps', type=int, default=100, help='DLN training steps')
    parser.add_argument('--output-dir', default='outputs/hybrid_semantic_ar', help='Output directory')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = SemanticARPipeline(
        corpus_path=args.corpus,
        rule_discovery_stories=args.discovery_stories,
        rule_discovery_facts=args.discovery_facts,
        ilp_algorithm=args.ilp_algorithm,
        ilp_rules=args.ilp_rules,
        ga_generations=args.ga_generations,
        training_stories=args.training_stories,
        training_facts=args.training_facts,
        dln_embed_dim=args.embed_dim,
        dln_training_steps=args.training_steps,
        verbose=not args.quiet
    )
    
    # Run pipeline
    rules, model, metrics = pipeline.run()
    
    # Save outputs
    pipeline.save(args.output_dir)
    
    print(f"\n✅ Pipeline complete! Output saved to {args.output_dir}")


if __name__ == '__main__':
    main()
