#!/usr/bin/env python
from __future__ import annotations

"""Benchmark ampio (multi-seed, multi-budget) con grafici per la relazione.

Esegue k-fold cross-validation per più seed e più budget, aggrega tutti i fold
(seed*fold) e produce:
- summary JSON per budget (mean±std)
- CSV fold-level (per tabelle e analisi)
- grafici metriche vs budget (Raw vs Corrected)

Esempio consigliato (abbastanza ampio ma ancora gestibile su laptop):
  py -3 scripts/run_benchmark.py --seeds 0..9 --budgets 800,1000,1200,1400,1600,1800,2000 \
      --n-samples 15000 --folds 5 --outdir results/bench_big --plot

Nota:
- n_samples è PER SEED. Totale esempi processati ≈ len(seeds) * n_samples.
- le metriche finali sono aggregate su len(seeds)*folds fold.
"""

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nscc.eval import FoldResult, run_crossval, summarize  # noqa: E402
from nscc.plots import plot_metric_over_budgets  # noqa: E402


def _parse_seeds(spec: str) -> List[int]:
    spec = spec.strip()
    if ".." in spec:
        a, b = spec.split("..", 1)
        return list(range(int(a), int(b) + 1))
    if "," in spec:
        return [int(x.strip()) for x in spec.split(",") if x.strip()]
    return [int(spec)]


def _parse_int_list(spec: str) -> List[int]:
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def _foldresults_to_rows(budget: int, seed: int, fold_results: List[FoldResult]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for i, r in enumerate(fold_results):
        rows.append({
            "budget": budget,
            "seed": seed,
            "fold": i,
            "raw_exact_acc": r.raw_exact_acc,
            "corrected_exact_acc": r.corrected_exact_acc,
            "raw_consistency": r.raw_consistency,
            "corrected_consistency": r.corrected_consistency,
            "raw_budget_violation": r.raw_budget_violation,
            "corrected_budget_violation": r.corrected_budget_violation,
            "corrected_avg_changes": r.corrected_avg_changes,
            "raw_precision_macro": r.raw_precision_macro,
            "raw_recall_macro": r.raw_recall_macro,
            "raw_accuracy_macro": r.raw_accuracy_macro,
            "corrected_precision_macro": r.corrected_precision_macro,
            "corrected_recall_macro": r.corrected_recall_macro,
            "corrected_accuracy_macro": r.corrected_accuracy_macro,
        })
    return rows


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _pack_ctz(outdir: Path, ctz_path: Path) -> None:
    """Crea un archivio .ctz (in pratica un .zip) con tutti i file in outdir."""
    ctz_path.parent.mkdir(parents=True, exist_ok=True)
    # mettiamo percorsi relativi ad outdir dentro l'archivio
    with zipfile.ZipFile(ctz_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(outdir.rglob("*")):
            if p.is_file():
                z.write(p, arcname=str(p.relative_to(outdir)))


def main() -> None:
    ap = argparse.ArgumentParser(description="NSCC benchmark ampio (multi-seed, multi-budget)")
    ap.add_argument("--seeds", type=str, default="0..4", help="es. '0..9' oppure '0,1,2'")
    ap.add_argument("--budgets", type=str, default="800,1000,1200,1400,1600,1800,2000")
    ap.add_argument("--n-samples", type=int, default=10000, help="campioni per seed")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--label-noise", type=float, default=0.12)

    ap.add_argument("--ml-hidden", type=str, default="64,32")
    ap.add_argument("--ml-alpha", type=float, default=1e-4)
    ap.add_argument("--ml-max-iter", type=int, default=500)
    ap.add_argument("--params-file", type=str, default=None, help="JSON da scripts/tune.py con model_params/corrector_params")

    ap.add_argument("--budget-weight", type=float, default=3.0)
    ap.add_argument("--spend-weight", type=float, default=0.0)
    ap.add_argument("--no-user-prefs", action="store_true")

    ap.add_argument("--outdir", type=str, default="results/bench")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument(
        "--ctz-out",
        type=str,
        default=None,
        help="Se specificato, crea un archivio .ctz (ZIP) con CSV/JSON/PNG prodotti nel --outdir.",
    )
    args = ap.parse_args()

    seeds = _parse_seeds(args.seeds)
    budgets = _parse_int_list(args.budgets)
    hidden = tuple(int(x.strip()) for x in args.ml_hidden.split(",") if x.strip())

    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    model_params = {"hidden_layer_sizes": hidden, "alpha": args.ml_alpha, "max_iter": args.ml_max_iter}
    corrector_params = {"budget_weight": args.budget_weight, "spend_weight": args.spend_weight}

    if args.params_file:
        p = Path(args.params_file)
        if not p.is_absolute():
            p = ROOT / p
        cfg = json.loads(p.read_text(encoding="utf-8"))
        model_params.update(cfg.get("model_params", {}))
        corrector_params.update(cfg.get("corrector_params", {}))
        # normalize hidden_layer_sizes if provided as list
        if isinstance(model_params.get("hidden_layer_sizes"), list):
            model_params["hidden_layer_sizes"] = tuple(int(x) for x in model_params["hidden_layer_sizes"]) 

    budget_to_summary: Dict[int, Dict[str, Dict[str, float]]] = {}

    all_rows: List[Dict[str, object]] = []

    for budget in budgets:
        print(f"\n=== Budget {budget} EUR ===")
        budget_fold_results: List[FoldResult] = []

        for seed in seeds:
            print(f"  - seed={seed}")
            folds_res = run_crossval(
                n_samples=args.n_samples,
                folds=args.folds,
                seed=seed,
                label_noise=args.label_noise,
                budget=budget,
                spend_weight=args.spend_weight,
                use_user_prefs=not args.no_user_prefs,
                model_params=model_params,
                corrector_params=corrector_params,
            )
            budget_fold_results.extend(folds_res)
            all_rows.extend(_foldresults_to_rows(budget, seed, folds_res))

        summary = summarize(budget_fold_results)
        budget_to_summary[budget] = summary

        # salva summary budget-specific
        (outdir / f"summary_budget_{budget}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        # stampa breve
        print(f"  corrected_consistency: {summary['corrected_consistency']['mean']:.4f}±{summary['corrected_consistency']['std']:.4f}")
        print(f"  corrected_exact_acc  : {summary['corrected_exact_acc']['mean']:.4f}±{summary['corrected_exact_acc']['std']:.4f}")
        print(f"  corrected_accuracy_macro: {summary['corrected_accuracy_macro']['mean']:.4f}±{summary['corrected_accuracy_macro']['std']:.4f}")

    # salva aggregati
    (outdir / "benchmark_summary_by_budget.json").write_text(json.dumps(budget_to_summary, indent=2), encoding="utf-8")
    _write_csv(outdir / "fold_results.csv", all_rows)

    if args.plot:
        # grafici principali
        plot_metric_over_budgets(budget_to_summary, "accuracy_macro", outdir / "accuracy_macro_vs_budget.png", "Macro Accuracy vs Budget")
        plot_metric_over_budgets(budget_to_summary, "precision_macro", outdir / "precision_macro_vs_budget.png", "Macro Precision vs Budget")
        plot_metric_over_budgets(budget_to_summary, "recall_macro", outdir / "recall_macro_vs_budget.png", "Macro Recall vs Budget")
        plot_metric_over_budgets(budget_to_summary, "exact_acc", outdir / "exact_acc_vs_budget.png", "Exact-match Accuracy vs Budget")
        plot_metric_over_budgets(budget_to_summary, "consistency", outdir / "consistency_vs_budget.png", "Consistency Rate vs Budget", ylabel="Rate")
        plot_metric_over_budgets(budget_to_summary, "budget_violation", outdir / "budget_violation_vs_budget.png", "Budget Violation Rate vs Budget", ylabel="Rate")

        print(f"\nSaved plots in: {outdir}")

    # opzionale: crea .ctz (zip) con i risultati
    if args.ctz_out:
        ctz = Path(args.ctz_out)
        if not ctz.is_absolute():
            ctz = ROOT / ctz
        if ctz.suffix.lower() != ".ctz":
            ctz = ctz.with_suffix(ctz.suffix + ".ctz") if ctz.suffix else ctz.with_suffix(".ctz")
        _pack_ctz(outdir, ctz)
        print(f"Saved results archive: {ctz}")

    print(f"\nSaved benchmark in: {outdir}")


if __name__ == "__main__":
    main()
