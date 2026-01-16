"""
Thin CLI wrapper around tinystories_mini_benchmark for modular demos.
Reuses existing pipeline without modifying core logic.
"""

import argparse
from typing import Optional, Set

from benchmarks import tinystories_mini_benchmark


def run_tinystories_demo(
    device: str = "cpu",
    max_stories: int = 50,
    max_facts: int = 1000,
    max_candidate_rules: int = 200,
    use_rule_injection: bool = True,
    ri_steps: int = 4,
    ri_threshold: float = 0.5,
    ri_lr: float = 1e-2,
    label_batch_size: int = 128,
    allowed_predicates: Optional[Set[str]] = None,
):
    return tinystories_mini_benchmark(
        device=device,
        max_stories=max_stories,
        max_facts=max_facts,
        max_candidate_rules=max_candidate_rules,
        use_rule_injection=use_rule_injection,
        ri_train_steps=ri_steps,
        ri_threshold=ri_threshold,
        ri_lr=ri_lr,
        label_batch_size=label_batch_size,
        allowed_predicates=allowed_predicates,
    )


def main():
    parser = argparse.ArgumentParser(description="TinyStories demo runner (modular wrapper)")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--max-stories", type=int, default=50)
    parser.add_argument("--max-facts", type=int, default=1000)
    parser.add_argument("--max-candidate-rules", type=int, default=200)
    parser.add_argument("--no-rule-injection", action="store_true")
    parser.add_argument("--ri-steps", type=int, default=4)
    parser.add_argument("--ri-threshold", type=float, default=0.5)
    parser.add_argument("--ri-lr", type=float, default=1e-2)
    parser.add_argument("--label-batch-size", type=int, default=128)
    parser.add_argument("--allowed-preds", type=str, default="", help="Comma-separated predicates to keep (optional)")
    args = parser.parse_args()

    allowed = None
    if args.allowed_preds:
        allowed = {p.strip() for p in args.allowed_preds.split(",") if p.strip()}

    run_tinystories_demo(
        device=args.device,
        max_stories=args.max_stories,
        max_facts=args.max_facts,
        max_candidate_rules=args.max_candidate_rules,
        use_rule_injection=not args.no_rule_injection,
        ri_steps=args.ri_steps,
        ri_threshold=args.ri_threshold,
        ri_lr=args.ri_lr,
        label_batch_size=args.label_batch_size,
        allowed_predicates=allowed,
    )


if __name__ == "__main__":
    main()
