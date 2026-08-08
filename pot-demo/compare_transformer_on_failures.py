"""
Compare Transformer predictions for seeds 4 and 9 on families location/count/state.
"""
from pathlib import Path
import importlib.util
import torch

_spec_path = Path(__file__).resolve().parent / "train_pot_transformer.py"
spec = importlib.util.spec_from_file_location("tpt", str(_spec_path))
tpt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tpt)

DATA_PATH = Path("/tmp/pot_lt_pairs_clean_balanced2.jsonl")
CKPT_DIR = Path("/tmp")
SEEDS = [4,9]
FAMILIES = ["location","count","state"]

rows = tpt.load_rows(DATA_PATH)
for seed in SEEDS:
    print(f"\n=== TRANSFORMER SEED {seed} ===")
    train_rows, eval_rows = tpt.split_rows(rows, holdout=0.2, seed=seed)
    if True:
        train_rows = [r for r in train_rows if r.get('agreement')]
        eval_rows = [r for r in eval_rows if r.get('agreement')]
    train_rows = [r for r in train_rows if tpt.build_target_positions(r) is not None] if hasattr(tpt, 'build_target_positions') else train_rows
    eval_rows = [r for r in eval_rows if tpt.build_target_positions(r) is not None] if hasattr(tpt, 'build_target_positions') else eval_rows
    src_vocab = tpt.build_input_vocab(rows)
    tgt_vocab = tpt.build_clause_vocab(rows)
    ck_path = CKPT_DIR / f"transformer_seed{seed}_best.pt"
    if not ck_path.exists():
        print('Missing ckpt',ck_path)
        continue
    ck = torch.load(str(ck_path), map_location='cpu')
    # instantiate model with sizes from code
    model = tpt.ClauseTransformer(len(src_vocab), len(tgt_vocab), d_model=128, nhead=4, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=512).to('cpu')
    # checkpoint key might be 'model_state' or direct state_dict
    if 'model_state' in ck:
        state = ck['model_state']
    else:
        # try direct
        state = ck
    try:
        model.load_state_dict(state)
    except Exception as e:
        print('Model load failed:', e)
        # try nested key
        if 'state_dict' in ck:
            model.load_state_dict(ck['state_dict'])
        else:
            raise
    inv = {v:k for k,v in tgt_vocab.items()}
    wrongs = []
    for i,row in enumerate(eval_rows):
        if row.get('family') not in FAMILIES:
            continue
        src_ids = torch.tensor(tpt.encode_input(row, src_vocab), dtype=torch.long).unsqueeze(1)
        pred_ids = model.greedy_decode(src_ids, None, bos_id=1, eos_id=2, max_len=max(1, len(row.get('target_props',[]))+4), device='cpu')
        pred_tokens = [inv.get(i, '<unk>') for i in pred_ids if i not in (0,1,2)]
        gold = [tpt.clause_text(p) for p in row['target_props']]
        ok = pred_tokens == gold
        if not ok:
            wrongs.append({'index': i, 'family': row.get('family'), 'input_props': [tpt.clause_text(p) for p in row['input_props']], 'gold': gold, 'pred': pred_tokens})
    print(f"Eval rows considered: {sum(1 for r in eval_rows if r.get('family') in FAMILIES)} failures: {len(wrongs)}")
    for w in wrongs[:20]:
        print('-', w)
    # save failures
    out = CKPT_DIR / f"transformer_failures_seed{seed}.json"
    with out.open('w', encoding='utf-8') as f:
        import json
        json.dump({'seed': seed, 'n_eval': sum(1 for r in eval_rows if r.get('family') in FAMILIES), 'failures': wrongs}, f, indent=2)
    print('Saved', out)
