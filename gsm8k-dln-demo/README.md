# Logic Transformer (LT) compact demo

This directory contains a compact LT/DLN-style demo for arithmetic reasoning on GSM8K and MATH.

The current setup is intentionally small and classification-first. It uses:

- proposition extraction from the input text
- a compact LT working-memory core
- bucketed answer prediction
- explicit step heads for GSM8K-style supervision

## What this demo is for

The goal is to show that a small LT-style model can learn a nontrivial arithmetic reasoning path from structured examples.

The model is currently best understood as a compact baseline rather than the final LT architecture. It is useful for:

- GSM8K multi-step arithmetic
- a numeric-only MATH subset
- checking whether the LT path is learning program structure at all

## Setup

```bash
cd gsm8k-dln-demo
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## GSM8K workflow

Download a local slice, train, and evaluate:

```bash
python download_gsm8k.py --config main --limit-train 200 --limit-test 50
python train.py --data-path data/gsm8k/main_train.jsonl
python eval.py --data-path data/gsm8k/main_test.jsonl
```

If you want a quick smoke test, you can also use:

```bash
./run_demo.sh
```

## MATH workflow

The MATH path uses the accessible mirror and filters to integer answers so it matches the current compact LT answer bucket head:

```bash
python download_math.py --numeric-only --integers-only --subjects algebra,prealgebra --limit-train 50 --limit-test 10
python train.py --data-path data/math/train.jsonl
python eval.py --data-path data/math/test.jsonl
```

You can also switch mirrors if needed:

```bash
python download_math.py --dataset nlile/hendrycks-MATH-benchmark
python download_math.py --dataset rasbt/math_full_minus_math500
python download_math.py --dataset qwedsacf/competition_math
```

## Notes on current scope

This demo currently handles integer-answer MATH examples best. That is enough for a compact proof-of-concept, but it does not yet fully cover symbolic answers or a dedicated MATH step parser.

For GSM8K, the step supervision is richer because the dataset includes annotated arithmetic traces. For MATH, the current code mainly learns from the final answer and the worked solution text.

## Files

- `model.py` - compact LT-style model
- `preprocess_gsm8k.py` - proposition and arithmetic extraction
- `download_gsm8k.py` - GSM8K downloader and JSONL converter
- `download_math.py` - MATH mirror downloader and numeric filtering
- `train.py` - training loop
- `eval.py` - evaluation script
- `run_demo.sh` - GSM8K smoke test wrapper

## Practical guidance

If you are deciding whether to improve the code further or just train longer:

- More training/data will help the current compact model.
- Additional code changes will help most if they target MATH normalization, better step supervision, or a stronger decoder state.
- For the current stage, the biggest wins are likely better data coverage and more training, not a full rewrite.

GPU support becomes more useful as you scale up the data or run more sweeps. For the current small slices, CPU is usually sufficient; a GPU mainly helps with faster iteration and larger experiments.
