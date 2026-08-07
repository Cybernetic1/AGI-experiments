"""
Evaluate the tiny LT bootstrap model on GSM8K-style data.
Reports exact-answer accuracy, CoT parse accuracy, and structure metrics.
"""

from pathlib import Path
import argparse

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


DEFAULT_DATA_PATH = Path("data/gsm8k/main_test.jsonl")
FALLBACK_DATA_PATH = Path("synthetic_gsm8k_demo.jsonl")
CHECKPOINT_PATH = Path("checkpoints/lt_gsm8k_demo.pt")


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


def _answer_bucket(answer: str, max_bucket: int):
    value = int(round(parse_numeric_answer(answer)))
    return max(0, min(max_bucket - 1, value))


def _step_buckets(example, max_bucket: int):
    steps = extract_math_steps(example.get("cot", "")) if "subject" in example else extract_gsm8k_steps(example.get("cot", ""))
    if not steps:
        return None, _answer_bucket(example["answer"], max_bucket)
    first = steps[0].get("result")
    last = steps[-1].get("result")
    first_bucket = _answer_bucket(str(first), max_bucket) if first is not None else None
    final_bucket = _answer_bucket(str(last), max_bucket) if last is not None else _answer_bucket(example["answer"], max_bucket)
    return first_bucket, final_bucket


def _has_arithmetic_gold(gold):
    return gold.get("op") != "unknown" and gold.get("left") is not None and gold.get("right") is not None


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


def main():
    parser = argparse.ArgumentParser(description="Evaluate the LT bootstrap model")
    parser.add_argument("--data-path", default=None)
    args = parser.parse_args()

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Missing checkpoint: {CHECKPOINT_PATH}. Run train.py first.")
    data_path = Path(args.data_path) if args.data_path else (
        DEFAULT_DATA_PATH if DEFAULT_DATA_PATH.exists() else FALLBACK_DATA_PATH
    )
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data file: {data_path}")

    data = load_jsonl(str(data_path))
    for ex in data:
        ex["_props"] = build_example_props(ex["question"], ex.get("cot", ""), ex)
    data, skipped = _filter_math_examples(data)
    if skipped:
        print(f"Filtered out {skipped} non-numeric MATH examples")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model = LTArithmeticModel(
        feature_dim=checkpoint["feature_dim"],
        num_buckets=checkpoint["num_buckets"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    answer_correct = 0
    step1_correct = 0
    step1_total = 0
    step2_correct = 0
    step2_total = 0
    bucket_correct = 0
    program_correct = 0
    cot_correct = 0
    op_correct = 0
    op1_correct = 0
    op2_correct = 0
    loss_total = 0.0

    with torch.no_grad():
        for ex in data:
            props = ex.get("_props") or build_example_props(ex["question"], ex.get("cot", ""))
            _pred_value, info = model(props, return_info=True)
            gold_answer = answer_to_int_string(ex["answer"])
            gold_bucket = _answer_bucket(ex["answer"], model.answer_buckets)
            gold_step1_bucket, gold_step2_bucket = _step_buckets(ex, model.answer_buckets)
            pred_bucket = int(info["answer_logits"].argmax().item())
            if _is_math_example(ex):
                if pred_bucket == gold_bucket:
                    answer_correct += 1
            elif pred_bucket == gold_bucket:
                answer_correct += 1
            if int(info["answer_logits"].argmax().item()) == gold_bucket:
                bucket_correct += 1
            loss_total += F.cross_entropy(
                info["answer_logits"].unsqueeze(0),
                torch.tensor([gold_bucket], dtype=torch.long),
            ).item()
            if gold_step1_bucket is not None:
                step1_total += 1
                if int(info["step1_logits"].argmax().item()) == gold_step1_bucket:
                    step1_correct += 1
            step2_total += 1
            if int(info["step2_logits"].argmax().item()) == gold_step2_bucket:
                step2_correct += 1
            if answer_to_int_string(str(round(float(model.decode_program(props, info)["answer"])))) == gold_answer:
                program_correct += 1

            gold = extract_gsm8k_arithmetic(ex.get("cot", "")) or {
                "op": "unknown",
                "left": None,
                "right": None,
                "result": gold_answer,
            }
            if gold.get("result") == gold_answer:
                cot_correct += 1
            if _has_arithmetic_gold(gold):
                if int(info["op_logits"].argmax().item()) == _op_index(gold["op"]):
                    op_correct += 1

                left_idx = _find_operand_index(props, gold["left"])
                right_idx = _find_operand_index(props, gold["right"])
                if left_idx is not None and int(info["num1_logits"].argmax().item()) == left_idx:
                    op1_correct += 1
                if right_idx is not None and int(info["num2_logits"].argmax().item()) == right_idx:
                    op2_correct += 1

    total = max(1, len(data))
    print(f"Loaded {total} examples")
    print(f"Exact-answer accuracy: {answer_correct / total:.3f}")
    print(f"Answer-bucket accuracy: {bucket_correct / total:.3f}")
    print(f"Step-1 accuracy: {step1_correct / max(1, step1_total):.3f}")
    print(f"Step-2 accuracy: {step2_correct / max(1, step2_total):.3f}")
    print(f"Program accuracy: {program_correct / total:.3f}")
    print(f"CoT parse accuracy: {cot_correct / total:.3f}")
    print(f"Operation accuracy: {op_correct / total:.3f}")
    print(f"Operand-1 accuracy: {op1_correct / total:.3f}")
    print(f"Operand-2 accuracy: {op2_correct / total:.3f}")
    print(f"Answer CE: {loss_total / total:.4f}")


if __name__ == "__main__":
    main()
