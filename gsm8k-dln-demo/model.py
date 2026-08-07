import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib
import re
import sys
from typing import Optional
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_logic_core import LogicNetwork

# Minimal LT (DLN-style) model adapted for proposition vectors
class SimpleLT(nn.Module):
    def __init__(self, vocab_preds, vocab_args, embed_dim=32):
        super().__init__()
        self.pred_vocab = {p: i for i, p in enumerate(vocab_preds)}
        self.arg_vocab = {a: i for i, a in enumerate(vocab_args)}
        self.pred_embed = nn.Embedding(len(vocab_preds), embed_dim)
        self.arg_embed = nn.Embedding(len(vocab_args), embed_dim)
        self.prop_dim = embed_dim * 3
        self.mlp = nn.Sequential(
            nn.Linear(self.prop_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),
        )

    def encode_prop(self, prop):
        # prop: dict with 'pred' and args list
        pred_idx = torch.tensor([self.pred_vocab.get(prop['pred'], 0)], dtype=torch.long)
        emb = [self.pred_embed(pred_idx).squeeze(0)]
        for a in prop.get('args', [])[:2]:
            idx = self.arg_vocab.get(a, 0)
            emb.append(self.arg_embed(torch.tensor([idx], dtype=torch.long)).squeeze(0))
        while len(emb) < 3:
            emb.append(torch.zeros_like(emb[0]))
        return torch.cat(emb, dim=-1)  # (prop_dim,)

    def forward(self, premises_vec, conclusion_vec):
        # premises_vec: (batch, prop_dim); conclusion_vec: (batch, prop_dim)
        features = torch.cat([premises_vec, conclusion_vec], dim=-1)
        return self.mlp(features).squeeze(-1)


def _stable_hash(token: str) -> int:
    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def parse_numeric_answer(answer) -> float:
    text = str(answer).strip().replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        raise ValueError(f"Could not parse numeric answer from: {answer!r}")
    return float(match.group(0))


PURE_NUMERIC_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
PURE_INTEGER_RE = re.compile(r"^-?\d+$")


def is_pure_numeric_answer(answer) -> bool:
    text = str(answer).strip().replace(",", "")
    return bool(PURE_NUMERIC_RE.fullmatch(text))


def is_pure_integer_answer(answer) -> bool:
    text = str(answer).strip().replace(",", "")
    return bool(PURE_INTEGER_RE.fullmatch(text))


def normalize_numeric_answer(answer) -> str:
    if isinstance(answer, (int, float)):
        value = float(answer)
    else:
        text = str(answer).strip().replace(",", "")
        if not PURE_NUMERIC_RE.fullmatch(text):
            raise ValueError(f"Could not normalize numeric answer from: {answer!r}")
        value = float(text)
    if float(value).is_integer():
        return str(int(round(value)))
    text = format(float(value), ".6f")
    return text.rstrip("0").rstrip(".")


def numeric_answers_match(predicted, gold) -> bool:
    try:
        pred_norm = normalize_numeric_answer(predicted)
        gold_norm = normalize_numeric_answer(gold)
    except ValueError:
        return False
    return pred_norm == gold_norm


def answer_to_int_string(answer) -> str:
    value = parse_numeric_answer(answer)
    if float(value).is_integer():
        return str(int(value))
    return str(value)


def extract_final_answer_from_cot(cot: str):
    matches = re.findall(r"-?\d+(?:\.\d+)?", str(cot))
    if not matches:
        return None
    return answer_to_int_string(matches[-1])


class CoTAnswerModel(nn.Module):
    """Tiny hashed-text regressor for GSM8K-style answer prediction."""

    def __init__(self, feature_dim: int = 512):
        super().__init__()
        self.feature_dim = feature_dim
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def featurize(self, question: str, cot: str = "") -> torch.Tensor:
        text = f"{question} {cot}".strip().lower()
        vec = torch.zeros(self.feature_dim, dtype=torch.float32)
        for token in re.findall(r"[a-z]+|-?\d+(?:\.\d+)?", text):
            idx = _stable_hash(token) % self.feature_dim
            vec[idx] += 1.0
        return vec / max(1.0, vec.sum())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class LTBootstrapModel(nn.Module):
    """
    Tiny LT-shaped bootstrap: embed proposition atoms, attend over propositions,
    then regress the final answer.
    """

    def __init__(self, feature_dim: int = 32, num_buckets: int = 2048):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_buckets = num_buckets
        self.atom_embed = nn.Embedding(num_buckets, feature_dim)
        self.attn = nn.Linear(feature_dim, 1)
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def _bucket(self, token: str) -> int:
        return _stable_hash(str(token).lower()) % self.num_buckets

    def encode_prop(self, prop) -> torch.Tensor:
        tokens = [prop.get("pred", "")] + list(prop.get("args", []))
        if not tokens:
            tokens = ["<empty>"]
        idx = torch.tensor([self._bucket(tok) for tok in tokens], dtype=torch.long)
        return self.atom_embed(idx).mean(dim=0)

    def encode_props(self, props) -> tuple[torch.Tensor, torch.Tensor]:
        if not props:
            raise ValueError("Need at least one proposition")
        prop_vecs = torch.stack([self.encode_prop(prop) for prop in props], dim=0)
        attn_logits = self.attn(prop_vecs).squeeze(-1)
        attn_weights = torch.softmax(attn_logits, dim=0)
        wm = torch.sum(attn_weights.unsqueeze(-1) * prop_vecs, dim=0)
        return wm, attn_weights

    def forward(self, props) -> torch.Tensor:
        wm, _ = self.encode_props(props)
        return self.head(wm).squeeze(-1)


class LTArithmeticModel(nn.Module):
    """
    LT-shaped arithmetic model:
    - encode propositions into a DLN-style working memory
    - extract a logic state with fuzzy rule matching
    - decode a first arithmetic step
    - condition a second step on that scratchpad state
    - form answer from two soft-selected operands
    """

    def __init__(self, feature_dim: int = 64, num_buckets: int = 2048):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_buckets = num_buckets
        self.prop_length = 16
        self.num_props = 24
        self.answer_buckets = 2001
        self.logic_core = LogicNetwork(
            prop_length=self.prop_length,
            num_props=self.num_props,
            output_dim=feature_dim,
            num_rules=4,
            num_premises=2,
            var_slots=3,
        )
        self.prop_proj = nn.Sequential(
            nn.Linear(self.prop_length, feature_dim),
            nn.ReLU(),
        )
        self.prop_attn = nn.Linear(feature_dim, 1)
        self.context_proj = nn.Sequential(
            nn.Linear(feature_dim * 5 + 9, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.step_bucket_embed = nn.Embedding(self.answer_buckets, feature_dim)
        self.step1_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.answer_buckets),
        )
        self.step_update = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.step2_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.answer_buckets),
        )
        self.num1_selector = nn.Sequential(
            nn.Linear(feature_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.num2_selector = nn.Sequential(
            nn.Linear(feature_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.op_context = nn.Linear(5, feature_dim)
        self.op_head = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
        )
        self.answer_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.answer_buckets),
        )

    def _bucket(self, token: str) -> int:
        return _stable_hash(str(token).lower()) % self.num_buckets

    def encode_prop_features(self, prop) -> torch.Tensor:
        pred = str(prop.get("pred", "")).lower()
        args = [str(a).lower() for a in prop.get("args", [])]
        value = self._prop_number_value(prop)
        numeric_value = 0.0 if value is None else float(value)
        abs_value = abs(numeric_value)
        sign = 1.0 if numeric_value >= 0 else -1.0
        source = 0.0 if pred.startswith("question:") else 1.0 if pred.startswith("cot:") else 0.5
        op_flags = [
            1.0 if "add" in pred or any(a == "add" for a in args) else 0.0,
            1.0 if "sub" in pred or any(a == "sub" for a in args) else 0.0,
            1.0 if "mul" in pred or any(a == "mul" for a in args) else 0.0,
            1.0 if "div" in pred or any(a == "div" for a in args) else 0.0,
        ]
        arg_hashes = [self._bucket(a) / float(self.num_buckets) for a in args[:3]]
        while len(arg_hashes) < 3:
            arg_hashes.append(0.0)
        features = torch.tensor(
            [
                self._bucket(pred) / float(self.num_buckets),
                arg_hashes[0],
                arg_hashes[1],
                arg_hashes[2],
                numeric_value / 100.0,
                torch.log1p(torch.tensor(abs_value, dtype=torch.float32)).item() / 10.0,
                sign,
                1.0 if self._is_number_prop(prop) else 0.0,
                source,
                len(args) / 5.0,
                *op_flags,
                1.0,
                0.0,
            ],
            dtype=torch.float32,
        )
        return features

    def encode_prop(self, prop) -> torch.Tensor:
        return self.prop_proj(self.encode_prop_features(prop))

    @staticmethod
    def _is_number_prop(prop) -> bool:
        pred = str(prop.get("pred", ""))
        return pred.endswith(":number") and len(prop.get("args", [])) >= 2

    @staticmethod
    def _prop_number_value(prop):
        if len(prop.get("args", [])) < 2:
            return None
        try:
            return parse_numeric_answer(prop["args"][1])
        except ValueError:
            return None

    def _summarize_numbers(self, number_entries, device):
        if not number_entries:
            return torch.zeros(4, dtype=torch.float32, device=device)
        values = torch.tensor(number_entries, dtype=torch.float32, device=device)
        mean = values.mean()
        count = torch.tensor(float(values.numel()), dtype=torch.float32, device=device)
        min_val = values.min()
        max_val = values.max()
        return torch.stack([mean / 100.0, count / 10.0, min_val / 100.0, max_val / 100.0], dim=0)

    def forward(self, props, return_info: bool = False, step1_bucket_override: Optional[int] = None):
        if not props:
            raise ValueError("Need at least one proposition")

        prop_features = torch.stack([self.encode_prop_features(prop) for prop in props], dim=0)
        if prop_features.size(0) > self.num_props:
            prop_features = prop_features[: self.num_props]
        elif prop_features.size(0) < self.num_props:
            pad = torch.zeros(self.num_props - prop_features.size(0), self.prop_length, dtype=torch.float32)
            prop_features = torch.cat([prop_features, pad], dim=0)

        prop_vecs = torch.stack([self.encode_prop(prop) for prop in props], dim=0)
        prop_attn_logits = self.prop_attn(prop_vecs).squeeze(-1)
        prop_weights = torch.softmax(prop_attn_logits, dim=0)
        pooled = torch.sum(prop_weights.unsqueeze(-1) * prop_vecs, dim=0)

        logic_state, rule_details = self.logic_core(prop_features.unsqueeze(0), return_details=True)
        logic_state = logic_state.squeeze(0)
        logic_details = rule_details

        number_entries = []
        number_vecs = []
        number_prop_indices = []
        word_vecs = []
        for prop, vec in zip(props, prop_vecs):
            if self._is_number_prop(prop):
                value = self._prop_number_value(prop)
                if value is not None:
                    number_entries.append(value)
                    number_vecs.append(vec)
                    number_prop_indices.append(len(number_prop_indices))
            if str(prop.get("pred", "")).endswith(":word"):
                word_vecs.append(vec)

        if not number_entries:
            number_entries = [0.0]
            number_vecs = [pooled]

        number_vecs = torch.stack(number_vecs, dim=0)
        if word_vecs:
            word_summary = torch.stack(word_vecs, dim=0).mean(dim=0)
        else:
            word_summary = torch.zeros_like(pooled)
        op_logits = self.op_head(logic_state)
        op_probs = torch.softmax(op_logits, dim=-1)
        number_summary = self._summarize_numbers(number_entries, pooled.device)
        top_prop_count = min(2, prop_vecs.size(0))
        top_prop_indices = torch.topk(prop_attn_logits, k=top_prop_count).indices
        top_prop_vecs = prop_vecs[top_prop_indices]
        if top_prop_count < 2:
            pad = pooled.unsqueeze(0).expand(2 - top_prop_count, -1)
            top_prop_vecs = torch.cat([top_prop_vecs, pad], dim=0)
        context_features = torch.cat(
            [logic_state, pooled, top_prop_vecs[0], top_prop_vecs[1], word_summary, number_summary, op_probs],
            dim=-1,
        )
        context = self.context_proj(context_features)

        step1_logits = self.step1_head(context)
        if step1_bucket_override is None:
            step1_probs = torch.softmax(step1_logits, dim=-1)
            step1_state = torch.matmul(step1_probs, self.step_bucket_embed.weight)
        else:
            step1_state = self.step_bucket_embed(torch.tensor(step1_bucket_override, dtype=torch.long, device=pooled.device))
            step1_probs = None
        step2_state = context + self.step_update(torch.cat([context, step1_state], dim=-1))
        step2_logits = self.step2_head(step2_state)
        answer_logits = self.answer_head(step2_state)

        context_expanded = context.unsqueeze(0).expand_as(number_vecs)
        num1_logits = self.num1_selector(torch.cat([number_vecs, context_expanded], dim=-1)).squeeze(-1)
        num2_logits = self.num2_selector(torch.cat([number_vecs, context_expanded], dim=-1)).squeeze(-1)
        num1_weights = torch.softmax(num1_logits, dim=0)
        num2_weights = torch.softmax(num2_logits, dim=0)
        num1 = torch.sum(num1_weights * torch.tensor(number_entries, dtype=torch.float32, device=pooled.device))
        num2 = torch.sum(num2_weights * torch.tensor(number_entries, dtype=torch.float32, device=pooled.device))

        arithmetic_answer = (
            op_probs[0] * (num1 + num2)
            + op_probs[1] * (num1 - num2)
            + op_probs[2] * (num1 * num2)
            + op_probs[3] * (num1 / (num2 + 1e-6))
        )
        answer = arithmetic_answer

        if return_info:
            return answer.squeeze(-1), {
                "op_logits": op_logits,
                "op_probs": op_probs,
                "answer_logits": answer_logits,
                "step1_logits": step1_logits,
                "step2_logits": step2_logits,
                "prop_weights": prop_weights,
                "logic_state": logic_state,
                "logic_details": logic_details,
                "step1_state": step1_state,
                "step2_state": step2_state,
                "step1_probs": step1_probs,
                "context_features": context_features,
                "top_prop_indices": top_prop_indices,
                "number_summary": number_summary,
                "num1_weights": num1_weights,
                "num2_weights": num2_weights,
                "num1_logits": num1_logits,
                "num2_logits": num2_logits,
                "number_prop_indices": number_prop_indices,
                "num1": num1,
                "num2": num2,
                "context": context,
            }
        return answer.squeeze(-1)

    def decode_program(self, props, info):
        number_values = [
            self._prop_number_value(prop)
            for prop in props
            if self._is_number_prop(prop)
        ]
        number_values = [v for v in number_values if v is not None]
        if not number_values:
            number_values = [0.0]

        op_idx = int(info["op_logits"].argmax().item())
        left_idx = int(info["num1_logits"].argmax().item())
        right_idx = int(info["num2_logits"].argmax().item())

        left = number_values[min(left_idx, len(number_values) - 1)]
        right = number_values[min(right_idx, len(number_values) - 1)]

        if op_idx == 0:
            answer = left + right
            op = "add"
        elif op_idx == 1:
            answer = left - right
            op = "sub"
        elif op_idx == 2:
            answer = left * right
            op = "mul"
        elif op_idx == 3:
            answer = left / (right + 1e-6)
            op = "div"
        else:
            answer = left
            op = "unknown"

        return {
            "op_idx": op_idx,
            "op": op,
            "left_idx": left_idx,
            "right_idx": right_idx,
            "left_value": left,
            "right_value": right,
            "answer": answer,
        }

    def decode_answer_bucket(self, info):
        bucket = int(info["answer_logits"].argmax().item())
        return float(bucket)

    def decode_step1_bucket(self, info):
        bucket = int(info["step1_logits"].argmax().item())
        return float(bucket)

    def decode_step2_bucket(self, info):
        bucket = int(info["step2_logits"].argmax().item())
        return float(bucket)
