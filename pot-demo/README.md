# PoT logical-form demo

This directory holds the spaCy-fronted logical-form prototype.

## Goal

Map natural language to a Prolog-like form with explicit variables, e.g.

```prolog
event(?e1).
predicate(?e1, give).
agent(?e1, ?x1).
recipient(?e1, ?x2).
theme(?e1, ?x3).
```

## Setup

```bash
pip install -r pot-demo/requirements.txt
python -m spacy download en_core_web_sm
```

## Demo

```bash
python pot-demo/spacy_logical_form.py
python pot-demo/generate_dataset.py --core-only --count 100 --output pot-demo/data/pairs.jsonl
python pot-demo/export_lt_examples.py --input pot-demo/data/pairs.jsonl --output pot-demo/data/lt_pairs.jsonl
python pot-demo/train_pot.py --data pot-demo/data/lt_pairs.jsonl
python pot-demo/train_pot_seq.py --data pot-demo/data/lt_pairs.jsonl
python pot-demo/train_pot_clause.py --data pot-demo/data/lt_pairs.jsonl
```

## Use

- spaCy handles tokenization, POS, dependency structure, and basic entity hints.
- LT can then learn the mapping from parsed structure to logical form and program-like supervision.
- The synthetic generator gives a first paired dataset for training and comparison.
- The exporter converts clause strings into LT-friendly proposition dictionaries.
- `train_pot.py` is a minimal baseline that learns the exported clause sets.
- `train_pot_seq.py` is the autoregressive decoder baseline for ordered logical forms.
- `--core-only` removes boilerplate clauses to make the sequence task smaller first.
- `train_pot_clause.py` predicts whole clauses as sequence units.
