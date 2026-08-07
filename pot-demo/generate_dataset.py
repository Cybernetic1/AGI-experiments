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


def family_transitive(rng: random.Random) -> Example:
    subj, obj = rng.sample(NAMES, 2)
    verb = rng.choice(["saw", "liked", "helped", "met"])
    text = f"{subj} {verb} {obj}."
    clauses = [
        *_with_entity("?x1", subj),
        *_with_entity("?x2", obj),
        Clause("event", ("?e1",)),
        Clause("predicate", ("?e1", _sanitize_atom(verb))),
        Clause("agent", ("?e1", "?x1")),
        Clause("patient", ("?e1", "?x2")),
    ]
    return Example(text, _render(clauses), "", "transitive", False)


def family_mixed(rng: random.Random) -> Example:
    subj, recip, _ = rng.sample(NAMES, 3)
    thing = rng.choice(THINGS)
    place = rng.choice(PLACES)
    article = "an" if thing[0] in "aeiou" else "a"
    text = f"{subj} gave {recip} {article} {thing} in the {place}."
    clauses = [
        *_with_entity("?x1", subj),
        *_with_entity("?x2", recip),
        *_with_entity("?x3", thing, "noun"),
        *_with_entity("?x4", place, "noun"),
        Clause("event", ("?e1",)),
        Clause("predicate", ("?e1", "give")),
        Clause("agent", ("?e1", "?x1")),
        Clause("recipient", ("?e1", "?x2")),
        Clause("patient", ("?e1", "?x3")),
        Clause("location", ("?e1", "?x4")),
    ]
    return Example(text, _render(clauses), "", "mixed", False)


def family_state(rng: random.Random) -> Example:
    subj = rng.choice(NAMES)
    adj = rng.choice(["happy", "sad", "tired", "calm"])
    text = f"{subj} is {adj}."
    clauses = [
        *_with_entity("?x1", subj),
        Clause("event", ("?e1",)),
        Clause("predicate", ("?e1", "be")),
        Clause("agent", ("?e1", "?x1")),
        Clause("state", ("?e1", _sanitize_atom(adj))),
    ]
    return Example(text, _render(clauses), "", "state", False)


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
    family_transitive,
    family_mixed,
    family_state,
    family_count,
    family_universal,
]

FAMILY_NAMES = ["give", "location", "transitive", "mixed", "state", "count", "universal"]


def _family_stream(seed: int, balanced: bool):
    rng = random.Random(seed)
    if balanced:
        while True:
            order = list(range(len(FAMILIES)))
            rng.shuffle(order)
            for index in order:
                yield index
    else:
        while True:
            yield rng.randrange(len(FAMILIES))


def build_examples(count: int, seed: int, core_only: bool, balanced: bool = False, agreement_only: bool = False) -> List[Example]:
    return build_examples_core(count, seed, core_only=core_only, balanced=balanced, agreement_only=agreement_only)


def build_examples_core(count: int, seed: int, core_only: bool, balanced: bool, agreement_only: bool) -> List[Example]:
    rng = random.Random(seed)
    parser = SpacyLogicalFormParser()
    examples: List[Example] = []
    attempts = 0
    family_indices = _family_stream(seed, balanced)
    while len(examples) < count:
        attempts += 1
        if attempts > count * 500:
            raise RuntimeError("Could not generate enough aligned examples; try relaxing --agreement-only")
        family_idx = next(family_indices)
        example = FAMILIES[family_idx](rng)
        parsed = parser.parse(example.text).render()
        gold = _filter_form(example.logical_form, core_only)
        parsed = _filter_form(parsed, core_only)
        aligned = parsed == gold
        if agreement_only and not aligned:
            continue
        examples.append(
            Example(
                text=example.text,
                logical_form=gold,
                parser_form=parsed,
                family=example.family,
                agreement=aligned,
            )
        )
    return examples


def main():
    parser = argparse.ArgumentParser(description="Generate PoT logical-form pairs")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="data/pot_pairs.jsonl")
    parser.add_argument("--core-only", action="store_true", help="Drop boilerplate clauses like entity/type/tense")
    parser.add_argument("--balanced", action="store_true", help="Cycle through families to keep the corpus balanced")
    parser.add_argument("--agreement-only", action="store_true", help="Keep only examples where spaCy and gold agree")
    args = parser.parse_args()

    examples = build_examples_core(args.count, args.seed, args.core_only, args.balanced, args.agreement_only)
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
