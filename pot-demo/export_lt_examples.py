"""
Convert PoT logical-form pairs into LT-friendly structured JSONL.

The input comes from generate_dataset.py and already includes:
  - text
  - logical_form (gold)
  - parser_form (spaCy front-end output)

This script converts the clause strings into proposition dicts that LT can
consume later.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import re


CLAUSE_RE = re.compile(r"^(?P<pred>[a-z_]+)\((?P<args>.*)\)\.$")


def parse_clause_line(line: str):
    match = CLAUSE_RE.match(line.strip())
    if not match:
        return None
    raw_args = match.group("args").strip()
    args = [arg.strip() for arg in raw_args.split(",")] if raw_args else []
    return {"pred": match.group("pred"), "args": args}


def parse_form(text: str):
    clauses = []
    for line in str(text).splitlines():
        clause = parse_clause_line(line)
        if clause is not None:
            clauses.append(clause)
    return clauses


def main():
    parser = argparse.ArgumentParser(description="Export LT-friendly PoT examples")
    parser.add_argument("--input", default="data/pot_pairs.jsonl")
    parser.add_argument("--output", default="data/pot_lt_pairs.jsonl")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input file: {in_path}")

    rows = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            row = {
                "text": ex["text"],
                "input_props": parse_form(ex["parser_form"]),
                "target_props": parse_form(ex["logical_form"]),
                "family": ex.get("family"),
                "agreement": bool(ex.get("agreement")),
            }
            rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows to {out_path}")
    print(f"Exact parser agreement in source: {sum(1 for r in rows if r['agreement'])}/{len(rows)}")


if __name__ == "__main__":
    main()
