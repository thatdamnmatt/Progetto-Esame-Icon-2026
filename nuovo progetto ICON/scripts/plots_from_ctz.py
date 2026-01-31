#!/usr/bin/env python
from __future__ import annotations

"""Genera grafici a partire da un archivio .ctz.

Il file .ctz che produciamo è in realtà un archivio ZIP. Dentro trovi:
- per benchmark: benchmark_summary_by_budget.json + summary_budget_*.json + fold_results.csv + PNG
- per esperimento: summary.json + PNG

Questo script permette di rigenerare (o ricreare) i grafici senza rilanciare gli esperimenti.

Esempi:
  py -3 scripts/plots_from_ctz.py --ctz results/bench_click/benchmark.ctz --outdir results/bench_click
"""

import argparse
import json
from pathlib import Path
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nscc.plots import plot_summary, plot_metric_over_budgets  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="NSCC - Rigenera grafici da archivio .ctz")
    ap.add_argument("--ctz", required=True, type=str, help="Percorso al file .ctz")
    ap.add_argument("--outdir", required=True, type=str, help="Cartella dove salvare i grafici")
    args = ap.parse_args()

    ctz = Path(args.ctz)
    if not ctz.is_absolute():
        ctz = ROOT / ctz

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    if not ctz.exists():
        raise SystemExit(f"CTZ non trovato: {ctz}")

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        with zipfile.ZipFile(ctz, "r") as z:
            z.extractall(tmpdir)

        # Caso 1: benchmark
        bench = tmpdir / "benchmark_summary_by_budget.json"
        if bench.exists():
            budget_to_summary = json.loads(bench.read_text(encoding="utf-8"))
            plot_metric_over_budgets(budget_to_summary, "accuracy_macro", outdir / "accuracy_macro_vs_budget.png", "Macro Accuracy vs Budget")
            plot_metric_over_budgets(budget_to_summary, "precision_macro", outdir / "precision_macro_vs_budget.png", "Macro Precision vs Budget")
            plot_metric_over_budgets(budget_to_summary, "recall_macro", outdir / "recall_macro_vs_budget.png", "Macro Recall vs Budget")
            plot_metric_over_budgets(budget_to_summary, "exact_acc", outdir / "exact_acc_vs_budget.png", "Exact-match Accuracy vs Budget")
            plot_metric_over_budgets(budget_to_summary, "consistency", outdir / "consistency_vs_budget.png", "Consistency Rate vs Budget", ylabel="Rate")
            plot_metric_over_budgets(budget_to_summary, "budget_violation", outdir / "budget_violation_vs_budget.png", "Budget Violation Rate vs Budget", ylabel="Rate")
            print(f"Grafici benchmark salvati in: {outdir}")
            return

        # Caso 2: esperimento singolo
        summ = tmpdir / "summary.json"
        if summ.exists():
            summary = json.loads(summ.read_text(encoding="utf-8"))
            plot_summary(summary, out_dir=outdir)
            print(f"Grafici esperimento salvati in: {outdir}")
            return

        raise SystemExit("CTZ non riconosciuto: non contiene benchmark_summary_by_budget.json né summary.json")


if __name__ == "__main__":
    main()
