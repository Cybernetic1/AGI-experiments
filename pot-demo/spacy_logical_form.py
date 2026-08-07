"""
spaCy front-end for a Prolog-like logical form.

The goal is to turn a natural-language sentence into a compact
set of predicates with explicit logic variables so LT can learn
semantic parsing on top of a structured NL representation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import re

import spacy


WH_WORDS = {"who", "what", "which", "whom", "whose", "where", "when", "why", "how"}
ROLE_MAP = {
    "nsubj": "agent",
    "nsubjpass": "patient",
    "obj": "patient",
    "dobj": "patient",
    "iobj": "recipient",
    "dative": "recipient",
}
PREP_ROLE_MAP = {
    "to": "recipient",
    "for": "beneficiary",
    "with": "instrument",
    "in": "location",
    "at": "location",
    "on": "location",
    "from": "source",
    "into": "destination",
    "onto": "destination",
    "over": "location",
    "under": "location",
}
ENTITY_LABELS = {"PERSON", "ORG", "GPE", "LOC", "NORP", "FAC", "PRODUCT", "EVENT"}


def _sanitize_atom(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "x"


def _var_name(prefix: str, index: int) -> str:
    return f"?{prefix}{index}"


@dataclass(frozen=True)
class Clause:
    predicate: str
    args: tuple[str, ...]

    def render(self) -> str:
        return f"{self.predicate}({', '.join(self.args)})."


@dataclass
class LogicalForm:
    text: str
    clauses: List[Clause] = field(default_factory=list)
    entity_vars: Dict[str, str] = field(default_factory=dict)
    event_vars: List[str] = field(default_factory=list)

    def render(self) -> str:
        return "\n".join(clause.render() for clause in self.clauses)


class SpacyLogicalFormParser:
    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model_name)
        except OSError as exc:
            raise RuntimeError(
                f"Missing spaCy model '{model_name}'. Install it with: "
                f"python -m spacy download {model_name}"
            ) from exc

    def _entity_var(self, lf: LogicalForm, key: str) -> str:
        key = _sanitize_atom(key)
        if key not in lf.entity_vars:
            lf.entity_vars[key] = _var_name("x", len(lf.entity_vars) + 1)
        return lf.entity_vars[key]

    def _event_var(self, lf: LogicalForm) -> str:
        var = _var_name("e", len(lf.event_vars) + 1)
        lf.event_vars.append(var)
        return var

    def _add_entity(self, lf: LogicalForm, var: str, label: Optional[str], name: str):
        lf.clauses.append(Clause("entity", (var,)))
        lf.clauses.append(Clause("name", (var, _sanitize_atom(name))))
        if label:
            lf.clauses.append(Clause("type", (var, _sanitize_atom(label))))

    def _add_mention(self, lf: LogicalForm, span_text: str, label: Optional[str] = None) -> str:
        var = self._entity_var(lf, span_text)
        if not any(cl.predicate == "name" and cl.args[0] == var for cl in lf.clauses):
            self._add_entity(lf, var, label, span_text)
        elif label and not any(cl.predicate == "type" and cl.args[0] == var for cl in lf.clauses):
            lf.clauses.append(Clause("type", (var, _sanitize_atom(label))))
        return var

    def _add_event(self, lf: LogicalForm, verb) -> str:
        event_var = self._event_var(lf)
        lf.clauses.append(Clause("event", (event_var,)))
        lf.clauses.append(Clause("predicate", (event_var, _sanitize_atom(verb.lemma_))))
        if verb.tag_ in {"VBD", "VBN"}:
            lf.clauses.append(Clause("tense", (event_var, "past")))
        elif verb.tag_ in {"VBZ", "VBP"}:
            lf.clauses.append(Clause("tense", (event_var, "present")))
        return event_var

    def parse(self, text: str) -> LogicalForm:
        doc = self.nlp(text)
        lf = LogicalForm(text=text)

        for sent in doc.sents:
            sent_has_question = any(tok.lower_ in WH_WORDS for tok in sent)
            if sent_has_question:
                lf.clauses.append(Clause("question", ("true",)))

            for token in sent:
                if token.pos_ in {"NOUN", "PROPN", "PRON", "NUM"} and token.dep_ in {"nsubj", "nsubjpass", "obj", "dobj", "iobj", "dative", "pobj", "attr", "appos"}:
                    self._add_mention(lf, token.text, token.ent_type_ or token.pos_)

            for verb in sent:
                if verb.pos_ not in {"VERB", "AUX"}:
                    continue
                if verb.dep_ not in {"ROOT", "conj", "xcomp", "advcl", "ccomp"} and verb.head != verb:
                    continue

                event_var = self._add_event(lf, verb)

                for child in verb.children:
                    role = ROLE_MAP.get(child.dep_)
                    if role:
                        entity_var = self._add_mention(lf, child.text, child.ent_type_ or child.pos_)
                        lf.clauses.append(Clause(role, (event_var, entity_var)))
                        continue

                    if child.dep_ == "prep":
                        pobj = next((c for c in child.children if c.dep_ == "pobj"), None)
                        if pobj is not None:
                            entity_var = self._add_mention(lf, pobj.text, pobj.ent_type_ or pobj.pos_)
                            role = PREP_ROLE_MAP.get(child.lemma_.lower(), "modifier")
                            lf.clauses.append(Clause(role, (event_var, entity_var)))
                        continue

                    if child.dep_ == "acomp":
                        lf.clauses.append(Clause("state", (event_var, _sanitize_atom(child.text))))
                        continue

                    if child.dep_ == "advmod":
                        lf.clauses.append(Clause("manner", (event_var, _sanitize_atom(child.text))))
                        continue

                if any(tok.lower_ in {"how", "what", "who", "which"} for tok in sent):
                    query_var = self._entity_var(lf, f"query_{event_var}")
                    lf.clauses.append(Clause("query", (event_var, query_var)))
                    if any(tok.lower_ == "how" and tok.nbor(1).lower_ == "many" for tok in sent if tok.i + 1 < len(doc)):
                        lf.clauses.append(Clause("query_kind", (event_var, "quantity")))

        if not lf.clauses:
            lf.clauses.append(Clause("text", (_sanitize_atom(text[:32] or "empty"),)))

        return lf


def demo():
    parser = SpacyLogicalFormParser()
    samples = [
        "John gave Mary a book.",
        "How many apples does John have?",
        "Every student who studied passed the test.",
    ]
    for sample in samples:
        lf = parser.parse(sample)
        print(f"\nTEXT: {sample}")
        print(lf.render())


if __name__ == "__main__":
    demo()
