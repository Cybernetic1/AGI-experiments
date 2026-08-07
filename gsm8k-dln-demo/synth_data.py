"""
Very small synthetic-data generator for math word problems.
This is a placeholder for LLM-generated synthetic data from models like
Llama 3-70B or Qwen-72B that produce GSM8K-style questions plus CoT traces.
"""
import json
import random

SAMPLE = [
    {
        "id": "1",
        "op": "add",
        "a": 3,
        "b": 2,
        "question": "If you have 3 apples and get 2 more, how many apples do you have?",
        "cot": "Start with 3 apples. Add 2 more. 3 + 2 = 5.",
        "answer": "5",
    },
    {
        "id": "2",
        "op": "sub",
        "a": 10,
        "b": 4,
        "question": "John has 10 marbles and gives away 4, how many remain?",
        "cot": "Start with 10 marbles. Subtract 4. 10 - 4 = 6.",
        "answer": "6",
    },
]

PROMPT_TEMPLATE = """You are generating GSM8K-style math word problems for a Logic Transformer demo.
Create a short problem that requires one or two arithmetic steps.

Return JSON with these fields:
- id: a unique string
- question: the math word problem
- cot: a short chain-of-thought explanation
- answer: the final numeric answer as a string

Keep the reasoning brief and ensure the final answer matches the cot.
"""

def make_variant(q):
    # Simple arithmetic perturbation while keeping question/cot/answer consistent.
    # Replace this with LLM-generated examples for the real demo.
    if q.get("op") == "add":
        a = max(1, q["a"] + random.choice([-1, 0, 1]))
        b = max(1, q["b"] + random.choice([-1, 0, 1]))
        answer = a + b
        question = f"If you have {a} apples and get {b} more, how many apples do you have?"
        cot = f"Start with {a} apples. Add {b} more. {a} + {b} = {answer}."
    else:
        a = max(2, q["a"] + random.choice([-1, 0, 1]))
        b = max(1, min(a - 1, q["b"] + random.choice([-1, 0, 1])))
        answer = a - b
        question = f"John has {a} marbles and gives away {b}, how many remain?"
        cot = f"Start with {a} marbles. Subtract {b}. {a} - {b} = {answer}."
    return {
        "id": f"{q['id']}-{random.randint(1000, 9999)}",
        "question": question,
        "cot": cot,
        "answer": str(answer),
    }

if __name__ == '__main__':
    out = []
    for ex in SAMPLE:
        out.append(ex)
        for _ in range(5):
            out.append(make_variant(ex))
    with open('synthetic_gsm8k_demo.jsonl', 'w') as f:
        for ex in out:
            record = {k: v for k, v in ex.items() if k in {"id", "question", "cot", "answer"}}
            f.write(json.dumps(record) + '\n')
    print('Wrote synthetic_gsm8k_demo.jsonl', len(out))
    print("\nPrompt template:\n")
    print(PROMPT_TEMPLATE)
