"""
Inspect mispredicted evaluation examples for given seeds and save failures.
"""
from pathlib import Path
import json
import importlib.util

_this_dir = Path(__file__).resolve().parent
_spec_path = _this_dir / "train_pot_clause.py"
spec = importlib.util.spec_from_file_location("train_pot_clause", str(_spec_path))
tpc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tpc)

DATA_PATH = Path("/tmp/pot_lt_pairs_clean_balanced2.jsonl")
OUT_DIR = Path('/tmp')
SEEDS = [4, 9]

rows_all = tpc.load_rows(DATA_PATH)
for seed in SEEDS:
    ck_path = OUT_DIR / f"pot_clause_seed{seed}_best.pt"
    out_path = OUT_DIR / f"pot_clause_failures_seed{seed}.json"
    if not ck_path.exists():
        print(f"Missing checkpoint for seed {seed}: {ck_path}")
        continue
    train_rows, eval_rows = tpc.split_rows(rows_all, holdout=0.2, seed=seed)
    train_rows = [r for r in train_rows if r.get('agreement')]
    eval_rows = [r for r in eval_rows if r.get('agreement')]
    train_rows = [r for r in train_rows if tpc.build_target_positions(r) is not None]
    eval_rows = [r for r in eval_rows if tpc.build_target_positions(r) is not None]
    input_vocab = tpc.build_input_vocab(rows_all)
    max_positions = max(len(r['input_props']) for r in rows_all)
    model = tpc.PoTPointerDecoder(len(input_vocab), max_positions, hidden_dim=128)
    ck = __import__('torch').load(str(ck_path), map_location='cpu')
    model.load_state_dict(ck['model_state'])

    failures = []
    for i, row in enumerate(eval_rows):
        gold = [tpc.clause_text(p) for p in row['target_props']]
        input_ids = __import__('torch').tensor([tpc.encode_input(row, input_vocab)], dtype=__import__('torch').long)
        pred_ids = model.beam_decode(
            input_ids,
            max(len(r['input_props']) for r in rows_all) + 1,
            len(row['input_props']),
            beam_width=5,
            max_steps=len(row['input_props']),
            fixed_length=len(row['input_props']),
        )
        pred = tpc.decode_positions(row, pred_ids)
        if pred != gold:
            failures.append({
                'index': i,
                'family': row.get('family'),
                'input_props': [tpc.clause_text(p) for p in row['input_props']],
                'target_props': gold,
                'predicted': pred,
            })
    with out_path.open('w', encoding='utf-8') as f:
        json.dump({'seed': seed, 'n_eval': len(eval_rows), 'n_fail': len(failures), 'failures': failures}, f, indent=2)
    print(f"Seed {seed}: eval={len(eval_rows)} failures={len(failures)} -> {out_path}")
    # print up to 10 failures
    for ex in failures[:10]:
        print(f"- family={ex['family']} input={ex['input_props']}\n  gold={ex['target_props']}\n  pred={ex['predicted']}")
