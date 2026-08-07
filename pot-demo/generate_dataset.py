"""
Generate paired NL -> logical-form examples for the PoT demo.

The output is a JSONL file with:
  - text: natural language input
  - logical_form: rendered Prolog-like clauses
  - family: template family name
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import random
from typing import Callable, List

from spacy_logical_form import (
    Clause,
    SpacyLogicalFormParser,
    _sanitize_atom,
    canonicalize_form,
    parse_clause_line,
    render_clauses,
)


NAMES = ["John", "Mary", "Alice", "Bob", "Lily", "Tom", "Sarah", "Noah"]
THINGS = ["book", "ball", "apple", "gift", "coin", "flower", "pencil", "ticket"]
PLACES = ["school", "park", "kitchen", "library", "garden", "office", "house"]
NOUNS = ["student", "cat", "worker", "player", "teacher", "dog"]
VERBS = [
    ("wait", "waits"),
    ("smile", "smiles"),
    ("rest", "rests"),
    ("work", "works"),
    ("run", "runs"),
    ("play", "plays"),
]
UNIVERSAL_PAIRS = [
    ("worker", "works", "work"),
    ("player", "plays", "play"),
    ("teacher", "works", "work"),
    ("student", "works", "work"),
    ("cat", "runs", "run"),
]
CORE_DROP_PREDS = {"entity", "type", "tense", "question", "quantifier", "query_kind"}


@dataclass(frozen=True)
class Example:
    text: str
    logical_form: str
    parser_form: str
    family: str
    agreement: bool


def _render(clauses: List[Clause]) -> str:
    return render_clauses(clauses)


def _filter_form(text: str, core_only: bool) -> str:
    if not core_only:
        return canonicalize_form(text)
    clauses = []
    for line in canonicalize_form(text).splitlines():
        parsed = parse_clause_line(line)
        if parsed is None:
            continue
        pred, args = parsed
        if pred in CORE_DROP_PREDS:
            continue
        clauses.append(f"{pred}({', '.join(args)}).")
    return "\n".join(clauses)


def _with_entity(var: str, name: str, kind: str = "person") -> List[Clause]:
    return [
        Clause("entity", (var,)),
        Clause("name", (var, _sanitize_atom(name))),
        Clause("type", (var, _sanitize_atom(kind))),
    ]


def family_give(rng: random.Random) -> Example:
    subj, recip, _ = rng.sample(NAMES, 3)
    thing = rng.choice(THINGS)
    article = "an" if thing[0] in "aeiou" else "a"
    text = f"{subj} gave {recip} {article} {thing}."
    clauses = [
        *_with_entity("?x1", subj),
        *_with_entity("?x2", recip),
        *_with_entity("?x3", thing, "noun"),
        Clause("event", ("?e1",)),
        Clause("predicate", ("?e1", "give")),
        Clause("agent", ("?e1", "?x1")),
        Clause("recipient", ("?e1", "?x2")),
        Clause("patient", ("?e1", "?x3")),
        Clause("quantifier", ("?x3", "exists")),
    ]
    return Example(text, _render(clauses), "", "give", False)


def family_location(rng: random.Random) -> Example:
    subj = rng.choice(NAMES)
    place = rng.choice(PLACES)
    text = f"{subj} is in the {place}."
    clauses = [
        *_with_entity("?x1", subj),
        *_with_entity("?x2", place, "noun"),
        Clause("event", ("?e1",)),
        Clause("predicate", ("?e1", "be")),
        Clause("agent", ("?e1", "?x1")),
        Clause("location", ("?e1", "?x2")),
    ]
    return Example(text, _render(clauses), "", "location", False)


def family_count(rng: random.Random) -> Example:
    subj = rng.choice(NAMES)
    thing = rng.choice(THINGS)
    count = rng.randint(1, 9)
    text = f"How many {thing}s does {subj} have?"
    clauses = [
        Clause("question", ("true",)),
        *_with_entity("?x1", subj),
        *_with_entity("?x2", thing, "noun"),
        Clause("event", ("?e1",)),
        Clause("predicate", ("?e1", "have")),
        Clause("agent", ("?e1", "?x1")),
        Clause("patient", ("?e1", "?x2")),
        Clause("query", ("?e1", "?x3")),
        Clause("query_kind", ("?e1", "quantity")),
    ]
    return Example(text, _render(clauses), "", "count", False)


def family_universal(rng: random.Random) -> Example:
    noun, surface, lemma = rng.choice(UNIVERSAL_PAIRS)
    text = f"Every {noun} {surface}."
    clauses = [
        Clause("quantifier", ("?x1", "forall")),
        *_with_entity("?x1", noun, "noun"),
        Clause("event", ("?e1",)),
        Clause("predicate", ("?e1", _sanitize_atom(lemma))),
        Clause("agent", ("?e1", "?x1")),
    ]
    return Example(text, _render(clauses), "", "universal", False)


FAMILIES: List[Callable[[random.Random], Example]] = [
    family_give,
    family_location,
    family_count,
    family_universal,
]


def build_examples(count: int, seed: int) -> List[Example]:
    return build_examples_core(count, seed, core_only=False)


def build_examples_core(count: int, seed: int, core_only: bool) -> List[Example]:
    rng = random.Random(seed)
    parser = SpacyLogicalFormParser()
    examples: List[Example] = []
    for _ in range(count):
        family = rng.choice(FAMILIES)
        example = family(rng)
        parsed = parser.parse(example.text).render()
        gold = _filter_form(example.logical_form, core_only)
        parsed = _filter_form(parsed, core_only)
        examples.append(
            Example(
                text=example.text,
                logical_form=gold,
                parser_form=parsed,
                family=example.family,
                agreement=parsed == gold,
            )
        )
    return examples


def main():
    parser = argparse.ArgumentParser(description="Generate PoT logical-form pairs")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="data/pot_pairs.jsonl")
    parser.add_argument("--core-only", action="store_true", help="Drop boilerplate clauses like entity/type/tense")
    args = parser.parse_args()

    examples = build_examples_core(args.count, args.seed, args.core_only)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.__dict__, ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} examples to {out_path}")
    print("Families:", ", ".join(sorted({ex.family for ex in examples})))
    agree = sum(1 for ex in examples if ex.agreement)
    print(f"spaCy/template agreement: {agree}/{len(examples)}")


if __name__ == "__main__":
    main()
