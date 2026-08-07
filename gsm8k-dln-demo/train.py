"""
Train a tiny LT-shaped bootstrap model on GSM8K-style data.
This validates the proposition extraction and arithmetic supervision path.
"""

from pathlib import Path
import argparse
import random

import torch
import torch.nn.functional as F

from model import (
    LTArithmeticModel,
    answer_to_int_string,
    is_pure_integer_answer,
    parse_numeric_answer,
)
from preprocess_gsm8k import (
    load_jsonl,
    build_example_props,
    extract_gsm8k_arithmetic,
    extract_gsm8k_steps,
    extract_math_steps,
)


DEFAULT_DATA_PATH = Path("data/gsm8k/main_train.jsonl")
FALLBACK_DATA_PATH = Path("synthetic_gsm8k_demo.jsonl")
CHECKPOINT_PATH = Path("checkpoints/lt_gsm8k_demo.pt")


def split_data(data, holdout_ratio=0.0):
    if len(data) < 2 or holdout_ratio <= 0.0:
        return data, data
    cutoff = max(1, int(len(data) * (1.0 - holdout_ratio)))
    cutoff = min(cutoff, len(data) - 1)
    return data[:cutoff], data[cutoff:]


def _op_index(op: str) -> int:
    return {"add": 0, "sub": 1, "mul": 2, "div": 3}.get(op, 4)


def _find_operand_index(props, operand_value):
    if operand_value is None:
        return None
    operand_value = str(operand_value)
    number_pos = 0
    for i, prop in enumerate(props):
        if not str(prop.get("pred", "")).endswith(":number"):
            continue
        args = prop.get("args", [])
        if len(args) >= 2 and str(args[1]) == operand_value:
            return number_pos
        number_pos += 1
    return None


def _gold_arithmetic(example):
    gold = extract_gsm8k_arithmetic(example.get("cot", ""))
    if gold is not None:
        return gold
    return {
        "op": "unknown",
        "left": None,
        "right": None,
        "result": answer_to_int_string(example["answer"]),
    }


def _has_arithmetic_gold(gold):
    return gold.get("op") != "unknown" and gold.get("left") is not None and gold.get("right") is not None


def _answer_bucket(answer: str, max_bucket: int):
    value = int(round(parse_numeric_answer(answer)))
    return max(0, min(max_bucket - 1, value))


def _answer_value(answer: str) -> float:
    return float(parse_numeric_answer(answer))


def _signed_log1p(value: torch.Tensor) -> torch.Tensor:
    return torch.sign(value) * torch.log1p(torch.abs(value))


def _is_math_example(example) -> bool:
    return "subject" in example


def _filter_math_examples(data):
    filtered = []
    skipped = 0
    for ex in data:
        if _is_math_example(ex) and not is_pure_integer_answer(ex["answer"]):
            skipped += 1
            continue
        filtered.append(ex)
    return filtered, skipped


def _step_buckets(example, max_bucket: int):
    steps = extract_math_steps(example.get("cot", "")) if "subject" in example else extract_gsm8k_steps(example.get("cot", ""))
    if not steps:
        return None, _answer_bucket(example["answer"], max_bucket)
    first = steps[0].get("result")
    last = steps[-1].get("result")
    first_bucket = _answer_bucket(str(first), max_bucket) if first is not None else None
    final_bucket = _answer_bucket(str(last), max_bucket) if last is not None else _answer_bucket(example["answer"], max_bucket)
    return first_bucket, final_bucket


def make_props(example):
    return example.get("_props") or build_example_props(example["question"], example.get("cot", ""), example)


def batch_metrics(model, data):
    model.eval()
    losses = []
    answer_correct = 0
    step1_correct = 0
    step1_total = 0
    step2_correct = 0
    step2_total = 0
    bucket_correct = 0
    program_correct = 0
    cot_correct = 0
    op_correct = 0
    op_total = 0
    operand1_correct = 0
    operand1_total = 0
    operand2_correct = 0
    operand2_total = 0

    with torch.no_grad():
        for ex in data:
            props = make_props(ex)
            pred_value, info = model(props, return_info=True)
            pred_bucket = int(info["answer_logits"].argmax().item())
            gold_answer = answer_to_int_string(ex["answer"])
            gold_bucket = _answer_bucket(ex["answer"], model.answer_buckets)
            gold_step1_bucket, gold_step2_bucket = _step_buckets(ex, model.answer_buckets)
            if _is_math_example(ex):
                if pred_bucket == gold_bucket:
                    answer_correct += 1
            else:
                if pred_bucket == gold_bucket:
                    answer_correct += 1
            losses.append(
                F.cross_entropy(
                    info["answer_logits"].unsqueeze(0),
                    torch.tensor([gold_bucket], dtype=torch.long),
                ).item()
            )

            if int(info["answer_logits"].argmax().item()) == gold_bucket:
                bucket_correct += 1
            if gold_step1_bucket is not None:
                step1_total += 1
                if int(info["step1_logits"].argmax().item()) == gold_step1_bucket:
                    step1_correct += 1
            step2_total += 1
            if int(info["step2_logits"].argmax().item()) == gold_step2_bucket:
                step2_correct += 1
            if answer_to_int_string(str(round(float(model.decode_program(props, info)["answer"])))) == gold_answer:
                program_correct += 1

            gold = _gold_arithmetic(ex)
            if gold.get("result") == gold_answer:
                cot_correct += 1
            if _has_arithmetic_gold(gold):
                op_total += 1
                if int(info["op_probs"].argmax().item()) == _op_index(gold["op"]):
                    op_correct += 1

                left_idx = _find_operand_index(props, gold["left"])
                right_idx = _find_operand_index(props, gold["right"])
                if left_idx is not None:
                    operand1_total += 1
                    if int(info["num1_weights"].argmax().item()) == left_idx:
                        operand1_correct += 1
                if right_idx is not None:
                    operand2_total += 1
                    if int(info["num2_weights"].argmax().item()) == right_idx:
                        operand2_correct += 1

    n = max(1, len(data))
    return (
        sum(losses) / n,
        answer_correct / n,
        bucket_correct / n,
        step1_correct / max(1, step1_total),
        step2_correct / max(1, step2_total),
        program_correct / n,
        cot_correct / n,
        op_correct / max(1, op_total),
        operand1_correct / max(1, operand1_total),
        operand2_correct / max(1, operand2_total),
    )


def main():
    parser = argparse.ArgumentParser(description="Train a tiny LT bootstrap model")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--value-loss-weight", type=float, default=0.05)
    parser.add_argument("--holdout-ratio", type=float, default=0.0)
    args = parser.parse_args()

    data_path = Path(args.data_path) if args.data_path else (
        DEFAULT_DATA_PATH if DEFAULT_DATA_PATH.exists() else FALLBACK_DATA_PATH
    )
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data file: {data_path}")

    data = load_jsonl(str(data_path))
    if not data:
        raise ValueError(f"No examples found in {data_path}")

    for ex in data:
        ex["_props"] = build_example_props(ex["question"], ex.get("cot", ""), ex)

    data, skipped = _filter_math_examples(data)
    if skipped:
        print(f"Filtered out {skipped} non-numeric MATH examples")

    random.shuffle(data)
    train_data, eval_data = split_data(data, args.holdout_ratio)

    model = LTArithmeticModel(feature_dim=32)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params}")

    for epoch in range(args.epochs):
        random.shuffle(train_data)
        model.train()
        train_losses = []

        for ex in train_data:
            props = make_props(ex)
            gold = _gold_arithmetic(ex)
            gold_bucket = _answer_bucket(ex["answer"], model.answer_buckets)
            gold_value = torch.tensor([_answer_value(ex["answer"])], dtype=torch.float32)
            gold_step1_bucket, gold_step2_bucket = _step_buckets(ex, model.answer_buckets)
            pred_value, info = model(props, return_info=True, step1_bucket_override=gold_step1_bucket)
            answer_ce_weight = 0.05 if _is_math_example(ex) else 0.5
            loss = answer_ce_weight * F.cross_entropy(info["answer_logits"].unsqueeze(0), torch.tensor([gold_bucket], dtype=torch.long))
            loss = loss + max(args.value_loss_weight, 0.4 if _is_math_example(ex) else 0.0) * F.smooth_l1_loss(
                _signed_log1p(pred_value.unsqueeze(0)),
                _signed_log1p(gold_value),
            )
            if _has_arithmetic_gold(gold):
                loss = loss + 0.2 * F.cross_entropy(info["op_logits"].unsqueeze(0), torch.tensor([_op_index(gold["op"])], dtype=torch.long))
            if gold_step1_bucket is not None:
                loss = loss + 0.3 * F.cross_entropy(info["step1_logits"].unsqueeze(0), torch.tensor([gold_step1_bucket], dtype=torch.long))
            loss = loss + 0.3 * F.cross_entropy(info["step2_logits"].unsqueeze(0), torch.tensor([gold_step2_bucket], dtype=torch.long))

            if _has_arithmetic_gold(gold):
                left_idx = _find_operand_index(props, gold["left"])
                right_idx = _find_operand_index(props, gold["right"])
                if left_idx is not None:
                    loss = loss + 0.2 * F.cross_entropy(info["num1_logits"].unsqueeze(0), torch.tensor([left_idx], dtype=torch.long))
                if right_idx is not None:
                    loss = loss + 0.2 * F.cross_entropy(info["num2_logits"].unsqueeze(0), torch.tensor([right_idx], dtype=torch.long))

            optim.zero_grad()
            loss.backward()
            optim.step()
            train_losses.append(loss.item())

        train_loss = sum(train_losses) / max(1, len(train_losses))
        eval_loss, answer_acc, bucket_acc, step1_acc, step2_acc, program_acc, cot_acc, op_acc, operand1_acc, operand2_acc = batch_metrics(model, eval_data)
        print(
            f"Epoch {epoch + 1:02d} | train_loss={train_loss:.4f} "
            f"| eval_loss={eval_loss:.4f} | answer_acc={answer_acc:.3f} | bucket_acc={bucket_acc:.3f} | step1_acc={step1_acc:.3f} | step2_acc={step2_acc:.3f} | program_acc={program_acc:.3f} "
            f"| cot_acc={cot_acc:.3f} | op_acc={op_acc:.3f} "
            f"| op1_acc={operand1_acc:.3f} | op2_acc={operand2_acc:.3f}"
        )

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_dim": model.feature_dim,
            "num_buckets": model.num_buckets,
        },
        CHECKPOINT_PATH,
    )
    print(f"Saved checkpoint to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
