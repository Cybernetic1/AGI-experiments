import math
import torch
from logic_core import Proposition, Rule, SymbolicEngine
from dln import SimpleDLN


def _train(model: SimpleDLN, facts, labels, steps: int = 80):
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(steps):
        opt.zero_grad()
        loss = 0.0
        for (pred, args), truth in labels.items():
            premises = [p for p in facts if p.args == args]
            out = model(premises, Proposition(pred, args))
            loss = loss + torch.nn.functional.mse_loss(out, torch.tensor([[truth]], dtype=torch.float32))
        loss.backward()
        opt.step()
    return opt


def _collect_labels(inferred):
    return {(p.predicate, p.args): p.truth for p in inferred if not p.predicate.startswith("not_")}


def test_dln_matches_symbolic_engine():
    torch.manual_seed(0)
    facts = [
        Proposition("A", ("x",), 1.0),
        Proposition("A", ("y",), 0.6),
        Proposition("D", ("y",), 1.0),
    ]
    rules = [
        Rule([Proposition("A", ("?x",))], Proposition("B", ("?x",)), 1.0),
        Rule([Proposition("A", ("?x",)), Proposition("D", ("?x",))], Proposition("C", ("?x",)), 1.0),
        Rule([Proposition("B", ("?x",))], Proposition("E", ("?x",)), 1.0),
    ]

    sym = SymbolicEngine()
    inferred = sym.infer(facts, rules)
    labels = _collect_labels(inferred)
    target_preds = sorted({p for (p, _args) in labels})

    predicates = sorted({p for p, _ in labels.keys()}.union({f.predicate for f in facts}))
    args = ["<pad>"] + sorted({a for f in facts for a in f.args})
    model = SimpleDLN(predicates, args, embed_dim=24)

    _train(model, facts, {k: labels[k] for k in labels if k[0] in target_preds}, steps=80)

    with torch.no_grad():
        for (pred, args_tuple), truth in labels.items():
            premises = [p for p in facts if p.args == args_tuple]
            pred_truth = model(premises, Proposition(pred, args_tuple)).item()
            assert math.isclose(pred_truth, truth, rel_tol=0.25, abs_tol=0.15), (
                f"Pred {pred}{args_tuple}={pred_truth:.3f} vs target {truth:.3f}")

        mse = 0.0
        for (pred, args_tuple), truth in labels.items():
            premises = [p for p in facts if p.args == args_tuple]
            pred_truth = model(premises, Proposition(pred, args_tuple)).item()
            mse += (pred_truth - truth) ** 2
        mse /= max(len(labels), 1)
        assert mse < 0.02, f"MSE too high: {mse}"


if __name__ == "__main__":
    test_dln_matches_symbolic_engine()
    print("✓ DLN matches symbolic engine on toy inference")
