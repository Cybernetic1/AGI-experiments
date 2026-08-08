"""
DLN encoder-only diagnostic on PoT dataset.
Encodes clauses via token embeddings, builds working memory, runs LogicNetwork,
and checks whether any rule's attention picks the gold target clauses.
Saves results to /tmp/pot_dln_encoder_diag_seed0.json
"""
from pathlib import Path
import re, json
import importlib.util
import torch
import torch.nn as nn

# load train_pot_seq TOKEN_RE and clause_text helpers
_spec = importlib.util.spec_from_file_location('tps', 'pot-demo/train_pot_seq.py')
tps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tps)

# load LogicNetwork
_spec2 = importlib.util.spec_from_file_location('nlc', '../neural_logic_core.py')
# adjust path
_spec2 = importlib.util.spec_from_file_location('nlc', str(Path(__file__).resolve().parent.parent / 'neural_logic_core.py'))
nlc = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(nlc)

DATA = Path('/tmp/pot_lt_pairs_clean_balanced2.jsonl')
OUT = Path('/tmp/pot_dln_encoder_diag_seed0.json')

rows = []
with DATA.open('r', encoding='utf-8') as f:
    for line in f:
        rows.append(json.loads(line))

# Filter agreement-only and usable rows similar to training
rows = [r for r in rows if r.get('agreement')]
# use build_target_positions from train_pot_clause for filtering
_spec3 = importlib.util.spec_from_file_location('tpc', str(Path(__file__).resolve().parent / 'train_pot_clause.py'))
tpc = importlib.util.module_from_spec(_spec3)
_spec3.loader.exec_module(tpc)
rows = [r for r in rows if tpc.build_target_positions(r) is not None]

# build token vocab from all clause texts
TOKEN_RE = tps.TOKEN_RE
from collections import Counter
counter = Counter()
for r in rows:
    for p in r['input_props'] + r['target_props']:
        for tok in TOKEN_RE.findall(tps.clause_text(p)):
            counter[tok]+=1
vocab = {tok:i for i,(tok,_) in enumerate(counter.most_common(), start=1)}
# reserve 0 for unk

# embedding dim = prop_length
PROP_LEN = 32
token_emb = nn.Embedding(len(vocab)+1, PROP_LEN)

# simple clause encoder: average token embeddings
def encode_clause_text(text):
    toks = TOKEN_RE.findall(text)
    ids = [vocab.get(t,0) for t in toks]
    if not ids:
        return torch.zeros(PROP_LEN)
    emb = token_emb(torch.tensor(ids))
    return emb.mean(dim=0)

# find max input clauses
max_in = max(len(r['input_props']) for r in rows)

# create LogicNetwork
ln = nlc.LogicNetwork(prop_length=PROP_LEN, num_props=max_in, output_dim=PROP_LEN, num_rules=6, num_premises=2, var_slots=3)
ln.eval()

results = {'n_examples': len(rows), 'examples': []}

for ex_idx, row in enumerate(rows):
    input_texts = [tps.clause_text(p) for p in row['input_props']]
    target_texts = [tps.clause_text(p) for p in row['target_props']]
    # encode clauses
    in_vecs = [encode_clause_text(t) for t in input_texts]
    tgt_vecs = [encode_clause_text(t) for t in target_texts]
    # pad working memory
    W = max_in
    wm = torch.zeros(1, W, PROP_LEN)
    for i,v in enumerate(in_vecs):
        wm[0,i,:] = v
    # run logic network with details
    with torch.no_grad():
        out, details = ln(wm, return_details=True)
    # details: list per rule; each has 'attention_weights' list per premise
    # collect which input positions are attended by any rule premise
    attended_positions = set()
    for rule_info in details:
        atts = rule_info['attention_weights']  # list length J, each tensor shape (batch, W)
        for att in atts:
            pos = int(att[0].argmax().item())
            attended_positions.add(pos)
    # For each target_text, check if any input position matching that exact text is in attended_positions
    matched_targets = 0
    matched_by_baseline = 0
    baseline_matches = []
    for tgt_idx, tgt_text in enumerate(target_texts):
        # find all input positions where input_text == tgt_text
        gold_pos = [i for i,t in enumerate(input_texts) if t == tgt_text]
        # baseline nearest neighbor by cosine
        sims = [torch.cosine_similarity(tgt_vecs[tgt_idx].unsqueeze(0), v.unsqueeze(0)).item() for v in in_vecs]
        nn_pos = max(range(len(sims)), key=lambda i:sims[i]) if sims else -1
        if gold_pos and nn_pos in gold_pos:
            matched_by_baseline += 1
        baseline_matches.append({'tgt':tgt_text,'nn_pos':nn_pos,'nn_text': input_texts[nn_pos] if nn_pos!=-1 else None})
        # check attended
        if any(p in attended_positions for p in gold_pos):
            matched_targets += 1
    results['examples'].append({'index': ex_idx, 'family': row.get('family'), 'n_targets': len(target_texts), 'matched_by_dln': matched_targets, 'matched_by_baseline': matched_by_baseline, 'input_texts': input_texts, 'target_texts': target_texts, 'baseline_matches': baseline_matches})

# aggregate
total_targets = sum(e['n_targets'] for e in results['examples'])
total_matched_dln = sum(e['matched_by_dln'] for e in results['examples'])
total_matched_baseline = sum(e['matched_by_baseline'] for e in results['examples'])
results['summary'] = {'total_targets': total_targets, 'dln_matched': total_matched_dln, 'baseline_matched': total_matched_baseline, 'dln_pct': total_matched_dln/total_targets if total_targets else 0.0, 'baseline_pct': total_matched_baseline/total_targets if total_targets else 0.0}

OUT.write_text(json.dumps(results, indent=2))
print('Wrote', OUT)
