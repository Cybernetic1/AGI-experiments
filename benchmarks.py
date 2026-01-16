import argparse
from typing import Optional, Set

from pipelines.benchmark_suite import run_all_smoke_tests


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid prototype benchmarks")
    parser.add_argument("--no-tiny", action="store_true", help="Skip TinyStories benchmark")
    parser.add_argument("--no-ga", action="store_true", help="Skip GA benchmark")
    parser.add_argument("--no-para", action="store_true", help="Skip paraconsistency tests")
    parser.add_argument("--save-store", action="store_true", help="Save RuleStore after TinyStories run")
    parser.add_argument("--load-store", action="store_true", help="Load RuleStore before TinyStories run")
    parser.add_argument("--store-path", type=str, default="data/processed/rule_store_tiny.json", help="Path for RuleStore load/save")
    parser.add_argument("--no-mined", action="store_true", help="Disable mined rules in TinyStories benchmark")
    parser.add_argument("--contra-strength", type=float, default=0.8, help="Strength of injected contradiction")
    parser.add_argument("--save-mined", action="store_true", help="Save RuleStore when mined rules are generated")
    parser.add_argument("--no-entity-registry", action="store_true", help="Disable entity registry canonicalization during TinyStories load")
    parser.add_argument("--svo-fallback", action="store_true", help="Use SVO extractor only (disable Davidsonian parser) for TinyStories load")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for DLN (cpu or cuda)")
    parser.add_argument("--no-label-cache", action="store_true", help="Disable disk label cache for symbolic inference")
    parser.add_argument("--max-stories", type=int, default=50, help="Number of TinyStories to load")
    parser.add_argument("--max-facts", type=int, default=1000, help="Cap on facts loaded from TinyStories")
    parser.add_argument("--no-ar-aux", action="store_true", help="Disable auxiliary predicate prediction loss")
    parser.add_argument("--ar-weight", type=float, default=0.1, help="Weight for auxiliary predicate prediction loss")
    parser.add_argument("--max-candidate-rules", type=int, default=200, help="Cap on candidate rules used for label collection")
    parser.add_argument("--no-rule-injection", action="store_true", help="Disable rule injection and per-rule label collection")
    parser.add_argument("--ri-steps", type=int, default=4, help="Gradient steps per injected rule")
    parser.add_argument("--ri-threshold", type=float, default=0.5, help="Confidence threshold for rule injection")
    parser.add_argument("--ri-lr", type=float, default=1e-2, help="Learning rate for rule injection fine-tuning")
    parser.add_argument("--label-batch-size", type=int, default=128, help="Rule batch size for label collection")
    parser.add_argument("--allowed-preds", type=str, default="", help="Comma-separated predicates to keep during label collection (optional)")
    args = parser.parse_args()

    allowed_preds = None
    if args.allowed_preds:
        allowed_preds = {p.strip() for p in args.allowed_preds.split(",") if p.strip()}

    run_all_smoke_tests(
        run_tiny=not args.no_tiny,
        run_ga=not args.no_ga,
        run_para=not args.no_para,
        save_store=args.save_store,
        load_store=args.load_store,
        store_path=args.store_path,
        use_mined=not args.no_mined,
        contra_strength=args.contra_strength,
        save_mined=args.save_mined,
        use_entity_registry=not args.no_entity_registry,
        prefer_davidsonian=not args.svo_fallback,
        device=args.device,
        disk_label_cache=not args.no_label_cache,
        max_stories=args.max_stories,
        max_facts=args.max_facts,
        use_ar_aux=not args.no_ar_aux,
        ar_weight=args.ar_weight,
        max_candidate_rules=args.max_candidate_rules,
        use_rule_injection=not args.no_rule_injection,
        ri_train_steps=args.ri_steps,
        ri_threshold=args.ri_threshold,
        ri_lr=args.ri_lr,
        label_batch_size=args.label_batch_size,
        allowed_predicates=allowed_preds,
    )
