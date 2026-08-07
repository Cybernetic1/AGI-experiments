"""
Download a MATH benchmark mirror from Hugging Face and convert it to JSONL.

The output keeps:
- question: the original problem statement
- cot: the worked solution text
- answer: the final answer string

Optionally filter to a subset of subjects/levels/types.
"""

from pathlib import Path
import argparse
import json
import re

from datasets import load_dataset


DEFAULT_DATASET = "nlile/hendrycks-MATH-benchmark"
NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?$")
INTEGER_RE = re.compile(r"-?\d+$")


def _parse_csv_list(value):
    if not value:
        return None
    items = [item.strip().lower() for item in value.split(",")]
    return {item for item in items if item}


def _extract_final_answer(example):
    answer = example.get("answer")
    if answer is not None and str(answer).strip():
        return str(answer).strip()

    solution = str(example.get("solution", ""))
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", solution)
    if boxed:
        return boxed[-1].strip()

    match = re.search(r"-?\d+(?:\.\d+)?\s*$", solution.strip())
    if match:
        return match.group(0).strip()

    return solution.strip()


def _is_numeric_answer(answer: str) -> bool:
    text = str(answer).strip().replace(",", "")
    return bool(NUMERIC_RE.search(text))


def _is_integer_answer(answer: str) -> bool:
    text = str(answer).strip().replace(",", "")
    return bool(INTEGER_RE.fullmatch(text))


def _keep_example(example, subjects=None, levels=None, types=None):
    subject = str(example.get("subject", "")).lower()
    level = str(example.get("level", "")).lower()
    problem_type = str(example.get("type", "")).lower()

    if subjects and subject not in subjects:
        return False
    if levels and level not in levels:
        return False
    if types and problem_type not in types:
        return False
    return True


def _write_split(split, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for idx, ex in enumerate(split):
            answer = _extract_final_answer(ex)
            record = {
                "id": str(ex.get("unique_id", idx)),
                "question": str(ex.get("problem", "")).strip(),
                "cot": str(ex.get("solution", "")).strip(),
                "answer": answer,
                "subject": ex.get("subject"),
                "level": ex.get("level"),
                "type": ex.get("type"),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Download and convert MATH")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", default="data/math")
    parser.add_argument("--split-train", default="train")
    parser.add_argument("--split-test", default="test")
    parser.add_argument("--subjects", default="algebra,prealgebra")
    parser.add_argument("--levels", default="")
    parser.add_argument("--types", default="")
    parser.add_argument("--numeric-only", action="store_true", help="Keep only examples with numeric answers")
    parser.add_argument("--integers-only", action="store_true", help="Keep only examples with integer answers")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    args = parser.parse_args()

    subjects = _parse_csv_list(args.subjects)
    levels = _parse_csv_list(args.levels)
    types = _parse_csv_list(args.types)

    ds = load_dataset(args.dataset)
    train_split = ds[args.split_train]
    test_split = ds[args.split_test]

    if subjects or levels or types:
        train_split = train_split.filter(lambda ex: _keep_example(ex, subjects, levels, types))
        test_split = test_split.filter(lambda ex: _keep_example(ex, subjects, levels, types))

    if args.numeric_only:
        train_split = train_split.filter(lambda ex: _is_numeric_answer(_extract_final_answer(ex)))
        test_split = test_split.filter(lambda ex: _is_numeric_answer(_extract_final_answer(ex)))

    if args.integers_only:
        train_split = train_split.filter(lambda ex: _is_integer_answer(_extract_final_answer(ex)))
        test_split = test_split.filter(lambda ex: _is_integer_answer(_extract_final_answer(ex)))

    if args.limit_train > 0:
        train_split = train_split.select(range(min(args.limit_train, len(train_split))))
    if args.limit_test > 0:
        test_split = test_split.select(range(min(args.limit_test, len(test_split))))

    out_dir = Path(args.output_dir)
    train_path = out_dir / "train.jsonl"
    test_path = out_dir / "test.jsonl"

    _write_split(train_split, train_path)
    _write_split(test_split, test_path)

    print(f"Wrote {len(train_split)} train examples to {train_path}")
    print(f"Wrote {len(test_split)} test examples to {test_path}")


if __name__ == "__main__":
    main()
