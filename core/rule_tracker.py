"""
Rule performance tracking for ILP evaluation.

Tracks how often each rule fires, its prediction accuracy, and contribution to training.
"""
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
import torch


@dataclass
class RuleStats:
    """Statistics for a single rule."""
    rule_id: str
    rule_text: str
    
    # Usage counts
    fires_train: int = 0  # How many times rule fired during training label generation
    fires_eval: int = 0   # How many times rule fired during eval label generation
    
    # Accuracy metrics
    total_predictions: int = 0
    correct_predictions: int = 0  # Within threshold
    sum_squared_error: float = 0.0
    sum_absolute_error: float = 0.0
    
    @property
    def accuracy(self) -> float:
        """Prediction accuracy (% within threshold)."""
        return self.correct_predictions / self.total_predictions if self.total_predictions > 0 else 0.0
    
    @property
    def mse(self) -> float:
        """Mean squared error for this rule's predictions."""
        return self.sum_squared_error / self.total_predictions if self.total_predictions > 0 else 0.0
    
    @property
    def mae(self) -> float:
        """Mean absolute error for this rule's predictions."""
        return self.sum_absolute_error / self.total_predictions if self.total_predictions > 0 else 0.0


class RuleTracker:
    """
    Tracks performance of individual rules in the system.
    
    Usage:
        tracker = RuleTracker()
        tracker.register_rules(rules)  # During setup
        tracker.record_firing(rule_id, is_train=True)  # During label generation
        tracker.record_prediction(rule_id, pred, target)  # During evaluation
        tracker.print_report()  # Show which rules are useful
    """
    
    def __init__(self):
        self.stats: Dict[str, RuleStats] = {}
        self.threshold = 0.1  # For "correct" classification
    
    def register_rules(self, rules: List[any], rule_type: str = ""):
        """Register rules for tracking."""
        for i, rule in enumerate(rules):
            rule_id = f"{rule_type}_{i}" if rule_type else f"rule_{i}"
            rule_text = str(rule)
            self.stats[rule_id] = RuleStats(rule_id=rule_id, rule_text=rule_text)
    
    def record_firing(self, rule_id: str, is_train: bool = True):
        """Record that a rule fired during label generation."""
        if rule_id in self.stats:
            if is_train:
                self.stats[rule_id].fires_train += 1
            else:
                self.stats[rule_id].fires_eval += 1
    
    def record_prediction(self, rule_id: str, prediction: float, target: float):
        """Record a prediction made using this rule."""
        if rule_id not in self.stats:
            return
        
        stat = self.stats[rule_id]
        stat.total_predictions += 1
        
        error = abs(prediction - target)
        if error < self.threshold:
            stat.correct_predictions += 1
        
        stat.sum_absolute_error += error
        stat.sum_squared_error += error ** 2
    
    def record_predictions_batch(self, rule_ids: List[str], predictions: torch.Tensor, targets: torch.Tensor):
        """Record predictions for a batch of labels."""
        predictions = predictions.detach().cpu()
        targets = targets.detach().cpu()
        
        for rule_id, pred, target in zip(rule_ids, predictions, targets):
            self.record_prediction(rule_id, pred.item(), target.item())
    
    def get_top_rules(self, n: int = 10, metric: str = "fires_train") -> List[Tuple[str, RuleStats]]:
        """Get top N rules by specified metric."""
        sorted_rules = sorted(
            self.stats.items(),
            key=lambda x: getattr(x[1], metric),
            reverse=True
        )
        return sorted_rules[:n]
    
    def get_bottom_rules(self, n: int = 10, metric: str = "fires_train") -> List[Tuple[str, RuleStats]]:
        """Get bottom N rules (least useful)."""
        sorted_rules = sorted(
            self.stats.items(),
            key=lambda x: getattr(x[1], metric)
        )
        return sorted_rules[:n]
    
    def print_report(self, top_n: int = 10):
        """Print comprehensive rule performance report."""
        print("\n" + "="*70)
        print("RULE PERFORMANCE REPORT")
        print("="*70)
        
        total_rules = len(self.stats)
        total_fires_train = sum(s.fires_train for s in self.stats.values())
        total_fires_eval = sum(s.fires_eval for s in self.stats.values())
        
        print(f"\nTotal rules: {total_rules}")
        print(f"Total firings (train): {total_fires_train}")
        print(f"Total firings (eval): {total_fires_eval}")
        
        # Rules that never fired
        never_fired = [rid for rid, stat in self.stats.items() 
                       if stat.fires_train == 0 and stat.fires_eval == 0]
        if never_fired:
            print(f"\n⚠️  {len(never_fired)} rules NEVER fired (potential noise):")
            for rid in never_fired[:10]:
                print(f"  - {rid}: {self.stats[rid].rule_text[:60]}")
            if len(never_fired) > 10:
                print(f"  ... and {len(never_fired) - 10} more")
        
        # Top rules by firing frequency
        print(f"\n🔥 Top {top_n} Most Frequently Used Rules (train):")
        print("-" * 70)
        for rid, stat in self.get_top_rules(top_n, "fires_train"):
            print(f"  {rid}:")
            print(f"    Fires: {stat.fires_train} train, {stat.fires_eval} eval")
            if stat.total_predictions > 0:
                print(f"    Accuracy: {stat.accuracy:.1%}, MSE: {stat.mse:.4f}, MAE: {stat.mae:.4f}")
            print(f"    Rule: {stat.rule_text[:60]}")
        
        # Top rules by accuracy (if predictions available)
        rules_with_preds = [(rid, stat) for rid, stat in self.stats.items() 
                           if stat.total_predictions > 0]
        if rules_with_preds:
            print(f"\n✅ Top {min(top_n, len(rules_with_preds))} Most Accurate Rules:")
            print("-" * 70)
            top_accurate = sorted(rules_with_preds, key=lambda x: x[1].accuracy, reverse=True)[:top_n]
            for rid, stat in top_accurate:
                print(f"  {rid}:")
                print(f"    Accuracy: {stat.accuracy:.1%} ({stat.correct_predictions}/{stat.total_predictions})")
                print(f"    MSE: {stat.mse:.4f}, MAE: {stat.mae:.4f}")
                print(f"    Fires: {stat.fires_train} train, {stat.fires_eval} eval")
        
        # Worst performing rules
        if rules_with_preds:
            print(f"\n❌ Bottom {min(top_n, len(rules_with_preds))} Least Accurate Rules:")
            print("-" * 70)
            bottom_accurate = sorted(rules_with_preds, key=lambda x: x[1].mse, reverse=True)[:top_n]
            for rid, stat in bottom_accurate:
                print(f"  {rid}:")
                print(f"    MSE: {stat.mse:.4f}, MAE: {stat.mae:.4f}, Accuracy: {stat.accuracy:.1%}")
                print(f"    Fires: {stat.fires_train} train, {stat.fires_eval} eval")
                print(f"    Rule: {stat.rule_text[:60]}")
        
        print("\n" + "="*70)
    
    def export_summary(self) -> Dict:
        """Export summary statistics for analysis."""
        return {
            "total_rules": len(self.stats),
            "rules_never_fired": sum(1 for s in self.stats.values() 
                                     if s.fires_train == 0 and s.fires_eval == 0),
            "total_fires_train": sum(s.fires_train for s in self.stats.values()),
            "total_fires_eval": sum(s.fires_eval for s in self.stats.values()),
            "avg_mse": sum(s.mse for s in self.stats.values()) / len(self.stats),
            "avg_accuracy": sum(s.accuracy for s in self.stats.values()) / len(self.stats),
        }
