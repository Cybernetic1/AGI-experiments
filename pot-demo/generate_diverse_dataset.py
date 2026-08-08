"""
Generate paired NL -> logical-form examples for the PoT demo, but with DIVERSE natural language.
This script adds active/passive swaps, synonyms, and structural variation to the synthetic sentences.
The gold logical form remains canonical, forcing the models to learn to map varied 
input semantic structures (from spaCy) to a single canonical output logic.
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
import generate_dataset as gd

def family_give_diverse(rng: random.Random) -> gd.Example:
    subj, recip, _ = rng.sample(gd.NAMES, 3)
    thing = rng.choice(gd.THINGS)
    article = "an" if thing[0] in "aeiou" else "a"
    
    # Structural variations!
    variation = rng.choice(["original", "passive", "synonym", "receive"])
    if variation == "original":
        text = f"{subj} gave {recip} {article} {thing}."
    elif variation == "passive":
        text = f"{article.capitalize()} {thing} was given to {recip} by {subj}."
    elif variation == "synonym":
        text = f"{subj} handed {article} {thing} to {recip}."
    elif variation == "receive":
        text = f"{recip} received {article} {thing} from {subj}."

    # Gold form is canonical! Always 'give' with agent=subj, recip=recip
    clauses = [
        *gd._with_entity("?x1", subj),
        *gd._with_entity("?x2", recip),
        *gd._with_entity("?x3", thing, "noun"),
        Clause("event", ("?e1",)),
        Clause("predicate", ("?e1", "give")),
        Clause("agent", ("?e1", "?x1")),
        Clause("recipient", ("?e1", "?x2")),
        Clause("patient", ("?e1", "?x3")),
        Clause("quantifier", ("?x3", "exists")),
    ]
    return gd.Example(text, gd._render(clauses), "", "give", False)


def family_location_diverse(rng: random.Random) -> gd.Example:
    subj = rng.choice(gd.NAMES)
    place = rng.choice(gd.PLACES)
    
    variation = rng.choice(["original", "passive", "action"])
    if variation == "original":
        text = f"{subj} is in the {place}."
    elif variation == "passive":
        text = f"The {place} contains {subj}."
    elif variation == "action":
        text = f"{subj} went into the {place}."

    clauses = [
        *gd._with_entity("?x1", subj),
        *gd._with_entity("?x2", place, "noun"),
        Clause("event", ("?e1",)),
        Clause("predicate", ("?e1", "be")),
        Clause("agent", ("?e1", "?x1")),
        Clause("location", ("?e1", "?x2")),
    ]
    return gd.Example(text, gd._render(clauses), "", "location", False)


def family_transitive_diverse(rng: random.Random) -> gd.Example:
    subj, obj = rng.sample(gd.NAMES, 2)
    verb = rng.choice(["saw", "liked", "helped", "met"])
    
    variation = rng.choice(["original", "passive", "relative"])
    if variation == "original":
        text = f"{subj} {verb} {obj}."
    elif variation == "passive":
        text = f"{obj} was {verb} by {subj}."
    elif variation == "relative":
        text = f"It was {subj} who {verb} {obj}."

    clauses = [
        *gd._with_entity("?x1", subj),
        *gd._with_entity("?x2", obj),
        Clause("event", ("?e1",)),
        Clause("predicate", ("?e1", gd._sanitize_atom(verb))),
        Clause("agent", ("?e1", "?x1")),
        Clause("patient", ("?e1", "?x2")),
    ]
    return gd.Example(text, gd._render(clauses), "", "transitive", False)


def family_state_diverse(rng: random.Random) -> gd.Example:
    subj = rng.choice(gd.NAMES)
    adj = rng.choice(["happy", "sad", "tired", "calm"])
    
    variation = rng.choice(["original", "feel", "state"])
    if variation == "original":
        text = f"{subj} is {adj}."
    elif variation == "feel":
        text = f"{subj} feels very {adj}."
    elif variation == "state":
        text = f"{subj} appears to be {adj}."

    clauses = [
        *gd._with_entity("?x1", subj),
        Clause("event", ("?e1",)),
        Clause("predicate", ("?e1", "be")),
        Clause("agent", ("?e1", "?x1")),
        Clause("state", ("?e1", gd._sanitize_atom(adj))),
    ]
    return gd.Example(text, gd._render(clauses), "", "state", False)


DIVERSE_FAMILIES = [
    family_give_diverse,
    family_location_diverse,
    family_transitive_diverse,
    family_state_diverse,
]

def build_diverse_examples_core(count: int, seed: int, core_only: bool, balanced: bool, agreement_only: bool) -> List[gd.Example]:
    rng = random.Random(seed)
    parser = SpacyLogicalFormParser()
    examples: List[gd.Example] = []
    attempts = 0
    
    def _family_stream(seed: int, balanced: bool):
        rng_local = random.Random(seed)
        if balanced:
            while True:
                order = list(range(len(DIVERSE_FAMILIES)))
                rng_local.shuffle(order)
                for index in order:
                    yield index
        else:
            while True:
                yield rng_local.randrange(len(DIVERSE_FAMILIES))

    family_indices = _family_stream(seed, balanced)
    
    # We drop agreement_only enforcement for Diverse examples, because spaCy WILL NOT 
    # magically agree with the gold canonical form for passive/synonym variants!
    # The whole point is to force the model to learn the alignment!
    
    while len(examples) < count:
        attempts += 1
        family_idx = next(family_indices)
        example = DIVERSE_FAMILIES[family_idx](rng)
        
        parsed_str = parser.parse(example.text).render()
        gold = gd._filter_form(example.logical_form, core_only)
        parsed = gd._filter_form(parsed_str, core_only)
        
        # We allow them into the dataset even if parsed != gold (which they usually won't)
        examples.append(
            gd.Example(
                text=example.text,
                logical_form=gold,
                parser_form=parsed,
                family=example.family,
                agreement=(parsed == gold), # likely false for variations
            )
        )
    return examples

def main():
    parser = argparse.ArgumentParser(description="Generate DIVERSE PoT logical-form pairs")
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/pot_diverse_pairs.jsonl")
    parser.add_argument("--core-only", action="store_true", help="Drop boilerplate clauses")
    parser.add_argument("--balanced", action="store_true", help="Cycle through families")
    args = parser.parse_args()

    examples = build_diverse_examples_core(args.count, args.seed, args.core_only, args.balanced, agreement_only=False)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.__dict__, ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} DIVERSE examples to {out_path}")
    print("Families:", ", ".join(sorted({ex.family for ex in examples})))
    agree = sum(1 for ex in examples if ex.agreement)
    print(f"spaCy/template exact agreement: {agree}/{len(examples)} (Low is expected and good!)")

if __name__ == "__main__":
    main()
