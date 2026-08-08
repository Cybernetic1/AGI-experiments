"""
Evaluate saved clause-pointer best checkpoints and write JSON summary.
"""
from pathlib import Path
import json
import torch
import importlib.util
import os

# Import train_pot_clause.py by path (not as a package)
_this_dir = Path(__file__).resolve().parent
_spec_path = _this_dir / "train_pot_clause.py"
spec = importlib.util.spec_from_file_location("train_pot_clause", str(_spec_path))
tpc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tpc)

DATA_PATH = Path("/tmp/pot_lt_pairs_clean_balanced2.jsonl")
OUT_PATH = Path("/tmp/pot_clause_ckpt_eval.json")

results = {}
for seed in range(10):
    ck_path = Path(f"/tmp/pot_clause_seed{seed}_best.pt")
    if not ck_path.exists():
        results[seed] = {"status": "missing"}
        continue
    rows = tpc.load_rows(DATA_PATH)
    train_rows, eval_rows = tpc.split_rows(rows, holdout=0.2, seed=seed)
    # match training filtering used during experiments
    train_rows = [r for r in train_rows if r.get("agreement")]
    eval_rows = [r for r in eval_rows if r.get("agreement")]
    train_rows = [r for r in train_rows if tpc.build_target_positions(r) is not None]
    eval_rows = [r for r in eval_rows if tpc.build_target_positions(r) is not None]
    input_vocab = tpc.build_input_vocab(rows)
    max_positions = max(len(r["input_props"]) for r in rows)
    model = tpc.PoTPointerDecoder(len(input_vocab), max_positions, hidden_dim=128)
    ck = torch.load(str(ck_path), map_location="cpu")
    model.load_state_dict(ck["model_state"])
    metrics = tpc.evaluate(model, eval_rows, input_vocab)
    results[seed] = {
        "status": "ok",
        "exact": metrics["exact"],
        "clause_acc": metrics["clause_acc"],
        "by_family": metrics.get("by_family", {}),
        "checkpoint_epoch": ck.get("epoch", None),
        "checkpoint_exact": ck.get("exact", None),
    }

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with OUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"Wrote evaluation to {OUT_PATH}")
