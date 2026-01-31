#!/usr/bin/env python
from __future__ import annotations

"""Tuning semplice (grid search) di iperparametri ML + pesi CSP.

Obiettivo: massimizzare corrected_exact_acc mantenendo:
- corrected_consistency >= min_consistency
- corrected_budget_violation <= max_budget_violation

Output:
- JSON con i migliori parametri (in results/tune_best.json di default)
- un CSV con tutte le combinazioni provate (results/tune_grid.csv)

Esempio (consigliato):
  py -3 scripts/tune.py --budget 1200 --seeds 0..2 --n-samples 6000 --folds 3 --out results/tune_best.json
"""

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nscc.eval import run_crossval, summarize  # noqa: E402


def _parse_seeds(spec: str) -> List[int]:
    spec = spec.strip()
    if ".." in spec:
        a, b = spec.split("..", 1)
        return list(range(int(a), int(b) + 1))
    if "," in spec:
        return [int(x.strip()) for x in spec.split(",") if x.strip()]
    return [int(spec)]


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="NSCC - tuning (grid search)")
    ap.add_argument("--budget", type=int, default=1200)
    ap.add_argument("--seeds", type=str, default="0..2")
    ap.add_argument("--n-samples", type=int, default=6000)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--label-noise", type=float, default=0.12)

    ap.add_argument("--min-consistency", type=float, default=0.995)
    ap.add_argument("--max-budget-violation", type=float, default=0.02)

    ap.add_argument("--out", type=str, default="results/tune_best.json")
    ap.add_argument("--grid-csv", type=str, default="results/tune_grid.csv")
    args = ap.parse_args()

    seeds = _parse_seeds(args.seeds)

    # Grid piccola ma sensata.
    ml_hiddens: List[Tuple[int, ...]] = [(64, 32), (128, 64), (128, 64, 32)]
    ml_alphas = [1e-4, 3e-4, 1e-3]
    budget_weights = [2.0, 3.0, 5.0]
    spend_weights = [0.0, 0.3, 0.6]

    best = None
    best_score = -1.0

    rows: List[Dict[str, object]] = []

    for hidden in ml_hiddens:
        for alpha in ml_alphas:
            for bw in budget_weights:
                for sw in spend_weights:
                    # Aggregate across seeds by concatenating fold results.
                    all_fold_results = []
                    for seed in seeds:
                        res = run_crossval(
                            n_samples=args.n_samples,
                            folds=args.folds,
                            seed=seed,
                            label_noise=args.label_noise,
                            budget=args.budget,
                            spend_weight=sw,
                            use_user_prefs=True,
                            model_params={"hidden_layer_sizes": hidden, "alpha": alpha, "max_iter": 500},
                            corrector_params={"budget_weight": bw, "spend_weight": sw},
                        )
                        all_fold_results.extend(res)

                    s = summarize(all_fold_results)
                    score = float(s["corrected_exact_acc"]["mean"])
                    cons = float(s["corrected_consistency"]["mean"])
                    viol = float(s["corrected_budget_violation"]["mean"])

                    row = {
                        "hidden": "-".join(map(str, hidden)),
                        "alpha": alpha,
                        "budget_weight": bw,
                        "spend_weight": sw,
                        "score_corrected_exact": score,
                        "consistency": cons,
                        "budget_violation": viol,
                        "raw_exact": float(s["raw_exact_acc"]["mean"]),
                        "raw_consistency": float(s["raw_consistency"]["mean"]),
                    }
                    rows.append(row)

                    feasible = (cons >= args.min_consistency) and (viol <= args.max_budget_violation)
                    if feasible and score > best_score:
                        best_score = score
                        best = {
                            "budget": args.budget,
                            "model_params": {"hidden_layer_sizes": list(hidden), "alpha": alpha, "max_iter": 500},
                            "corrector_params": {"budget_weight": bw, "spend_weight": sw},
                            "metrics": {
                                "corrected_exact_acc": score,
                                "corrected_consistency": cons,
                                "corrected_budget_violation": viol,
                                "raw_exact_acc": float(s["raw_exact_acc"]["mean"]),
                                "raw_consistency": float(s["raw_consistency"]["mean"]),
                            },
                        }

    out_path = ROOT / args.out
    grid_path = ROOT / args.grid_csv
    _write_csv(grid_path, rows)

    if best is None:
        print("Nessuna combinazione soddisfa i vincoli (consistency/budget).")
        # salva comunque la grid
        print(f"Grid salvata in: {grid_path}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(best, indent=2), encoding="utf-8")

    print("\n=== BEST CONFIG ===")
    print(json.dumps(best, indent=2))
    print(f"\nSalvati:\n- best: {out_path}\n- grid: {grid_path}")


if __name__ == "__main__":
    main()
