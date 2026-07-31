"""
Download GSM8K from Hugging Face and convert it to a simple JSONL format.

The output keeps the original question and splits the provided solution into:
- cot: the reasoning text
- answer: the final numeric answer string
"""

from pathlib import Path
import argparse
import json
import re

from datasets import load_dataset


def split_gsm8k_answer(answer: str):
    text = str(answer).strip()
    if "####" in text:
        cot, final = text.rsplit("####", 1)
        return cot.strip(), final.strip()
    match = re.search(r"-?\d+(?:\.\d+)?\s*$", text)
    if match:
        return text[: match.start()].strip(), match.group(0).strip()
    return text, text


def write_split(split, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for idx, ex in enumerate(split):
            cot, final = split_gsm8k_answer(ex["answer"])
            record = {
                "id": str(idx),
                "question": ex["question"].strip(),
                "cot": cot,
                "answer": final,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Download and convert GSM8K")
    parser.add_argument("--config", default="main", choices=["main", "socratic"])
    parser.add_argument("--output-dir", default="data/gsm8k")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    args = parser.parse_args()

    ds = load_dataset("openai/gsm8k", args.config)
    train_split = ds["train"]
    test_split = ds["test"]

    if args.limit_train > 0:
        train_split = train_split.select(range(min(args.limit_train, len(train_split))))
    if args.limit_test > 0:
        test_split = test_split.select(range(min(args.limit_test, len(test_split))))

    out_dir = Path(args.output_dir)
    train_path = out_dir / f"{args.config}_train.jsonl"
    test_path = out_dir / f"{args.config}_test.jsonl"

    write_split(train_split, train_path)
    write_split(test_split, test_path)

    print(f"Wrote {len(train_split)} train examples to {train_path}")
    print(f"Wrote {len(test_split)} test examples to {test_path}")


if __name__ == "__main__":
    main()
