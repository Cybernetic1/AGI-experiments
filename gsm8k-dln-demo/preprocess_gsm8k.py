"""
Loader hook for GSM8K-like files. Expect JSONL with {id, question, answer, cot}.
This script also provides a tiny proposition extractor for the LT bootstrap.
"""
import json
import re


ADD_HINTS = {
    "add", "plus", "more", "total", "altogether", "buy", "bought", "gets",
    "got", "receive", "receives", "received", "join", "gave", "gives",
}
SUB_HINTS = {
    "left", "remain", "remains", "gave away", "spent", "loss", "lose",
    "lost", "take away", "difference", "fewer", "less",
}


def _detect_operation(text: str) -> str:
    lower = text.lower()
    if any(h in lower for h in SUB_HINTS):
        return "sub"
    if any(h in lower for h in ADD_HINTS):
        return "add"
    return "unknown"


def extract_propositions(text: str, source: str = "text"):
    lower = str(text).lower()
    numbers = re.findall(r"-?\d+(?:\.\d+)?", lower)
    operation = _detect_operation(lower)
    props = []

    for i, num in enumerate(numbers[:6]):
        props.append({
            "pred": f"{source}:number",
            "args": [f"n{i}", num],
            "numeric_value": float(num),
        })

    if operation != "unknown":
        props.append({
            "pred": f"{source}:operation",
            "args": [operation],
        })

    if len(numbers) >= 2 and operation in {"add", "sub"}:
        props.append({
            "pred": f"{source}:candidate",
            "args": [operation, numbers[0], numbers[1]],
            "numeric_value": None,
        })

    if not props:
        props.append({
            "pred": f"{source}:text",
            "args": [lower[:32] or "<empty>"],
            "numeric_value": None,
        })

    return props


def build_example_props(question: str, cot: str = ""):
    return extract_propositions(question, source="question") + extract_propositions(cot, source="cot")


def extract_gsm8k_arithmetic(cot: str):
    """
    Parse the final annotated arithmetic step from GSM8K-style reasoning text.
    Returns a dict with op, left, right, result, or None if parsing fails.
    """
    text = str(cot)
    annotations = re.findall(r"<<([^<>]+)>>", text)
    if not annotations:
        return None

    expr = annotations[-1].strip()
    if "=" in expr:
        expr, result = expr.split("=", 1)
    else:
        result = None

    expr = expr.replace(" ", "")
    match = re.match(r"^(-?\d+(?:\.\d+)?)([+\-*/x])(-?\d+(?:\.\d+)?)$", expr)
    if not match:
        return None

    left = match.group(1)
    op = match.group(2)
    right = match.group(3)

    op_map = {"+": "add", "-": "sub", "*": "mul", "x": "mul", "/": "div"}
    return {
        "op": op_map.get(op, "unknown"),
        "left": left,
        "right": right,
        "result": result.strip() if result is not None else None,
        "expr": expr,
    }


def extract_gsm8k_steps(cot: str):
    """
    Parse all annotated arithmetic steps from GSM8K-style reasoning text.
    Returns a list of dicts in order of appearance.
    """
    text = str(cot)
    annotations = re.findall(r"<<([^<>]+)>>", text)
    steps = []
    for ann in annotations:
        expr = ann.strip()
        if "=" in expr:
            expr, result = expr.split("=", 1)
        else:
            result = None
        expr = expr.replace(" ", "")
        match = re.match(r"^(-?\d+(?:\.\d+)?)([+\-*/x])(-?\d+(?:\.\d+)?)$", expr)
        if not match:
            continue
        left = match.group(1)
        op = match.group(2)
        right = match.group(3)
        op_map = {"+": "add", "-": "sub", "*": "mul", "x": "mul", "/": "div"}
        steps.append({
            "op": op_map.get(op, "unknown"),
            "left": left,
            "right": right,
            "result": result.strip() if result is not None else None,
            "expr": expr,
        })
    return steps

def load_jsonl(path):
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples

if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'synthetic_gsm8k_demo.jsonl'
    ex = load_jsonl(path)
    print('Loaded', len(ex))
    print(ex[0])
    print(build_example_props(ex[0]["question"], ex[0].get("cot", "")))
