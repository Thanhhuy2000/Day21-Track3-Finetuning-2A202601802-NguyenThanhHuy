#!/usr/bin/env python3
"""Per-item regression predictions for the base model and the fine-tune.

    COMPUTE_TIER=T4 python scripts/dump_regression_preds.py

The four-group verdict is decided by the regression gate more often than by the target
group -- both complete runs of this lab so far FAILED there and nowhere else. But the
regression predictions exist only inside NB2 and NB5 as locals: `baselines_frozen.json`
keeps one mean, `verdict.json` keeps one delta. Nothing on disk says *what the model
answered*, so REPORT.md section 5 can report that general capability fell and section 6
cannot show a single instance of it.

That is the wrong way round. "regression 0.758 -> 0.589" is a number you have to trust;
a base model answering "thủ đô của Việt Nam?" in prose next to a fine-tune answering it
with `{"intent": "hoi_thong_tin", ...}` is the finding itself, and it is what tells you
whether the fix is deck §14.3's replay data or something else entirely.

Writes results/regression_preds.json. Reads nothing that NB5 has not already frozen, so
running it after the fact does not touch the graded comparison.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from labkit import evaluate as ev, generate, report          # noqa: E402
from labkit.config import get_tier                           # noqa: E402


def main() -> int:
    tier = get_tier(os.environ.get("COMPUTE_TIER"))
    adapter = ROOT / "adapters" / "correct"
    if not adapter.exists():
        print(f"missing {adapter} — run NB3 first", file=sys.stderr)
        return 1

    regression = [json.loads(l) for l in
                  (ROOT / "data" / "eval_regression.jsonl").read_text(encoding="utf-8").splitlines()
                  if l.strip()]
    limit = int(os.environ.get("EVAL_LIMIT", "0"))
    if limit:
        regression = regression[:limit]
    prompts = [r["instruction"] for r in regression]
    print(f"tier={tier.name} model={tier.model_id}  {len(prompts)} câu hỏi phổ thông")

    # Exactly NB5's regression call: system=None, 96 new tokens. A different shape here
    # would produce numbers that do not reconcile with verdict.json, which is worse than
    # having no numbers at all.
    def sweep(model, tok, label):
        preds, _ = generate.generate_batch(model, tok, prompts, system=None,
                                           max_new_tokens=96, label=label)
        scores = [ev.keyword_recall(p, r["keywords"]) for p, r in zip(preds, regression)]
        print(f"{label:12s} regression = {sum(scores) / len(scores):.4f}")
        return preds, scores

    model, tok = generate.load_base(tier)
    base_preds, base_scores = sweep(model, tok, "base")
    # Drop every reference before the second load. `del model` alone is what NB6 got
    # wrong: whatever still points at the weights keeps them resident, and the second
    # load_base() then gets offloaded to CPU on a 14.6 GB card.
    del model
    generate.free_memory()

    from peft import PeftModel
    model, tok = generate.load_base(tier)
    model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    ft_preds, ft_scores = sweep(model, tok, "fine-tune")
    del model
    generate.free_memory()

    rows = [{"i": i,
             "question": r["instruction"],
             "keywords": r["keywords"],
             "base_score": round(b, 4),
             "ft_score": round(f, 4),
             "delta": round(f - b, 4),
             "base_pred": bp.replace("\n", " ").strip(),
             "ft_pred": fp.replace("\n", " ").strip()}
            for i, (r, b, f, bp, fp) in enumerate(
                zip(regression, base_scores, ft_scores, base_preds, ft_preds))]
    report.write_json(rows, "regression_preds.json", results_dir=ROOT / "results")

    worse = [r for r in rows if r["delta"] < 0]
    print(f"\n-> results/regression_preds.json")
    print(f"fine-tune tệ hơn ở {len(worse)}/{len(rows)} câu\n")
    print(report.markdown_table(sorted(rows, key=lambda x: x["delta"])[:5],
                                ["i", "question", "base_score", "ft_score", "delta"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
