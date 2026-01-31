from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt


def _bar_with_err(names, means, stds, title: str, ylabel: str, out_path: Path) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = range(len(names))
    ax.bar(x, means, yerr=stds, capsize=5)
    ax.set_xticks(list(x), names)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_summary(summary: Dict[str, Dict[str, float]], out_dir: str | Path = "results") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def mstd(key: str):
        return float(summary[key]["mean"]), float(summary[key]["std"])

    # 1) Precision/Recall/Accuracy (macro)
    for metric in ["precision_macro", "recall_macro", "accuracy_macro"]:
        raw_m, raw_s = mstd(f"raw_{metric}")
        cor_m, cor_s = mstd(f"corrected_{metric}")
        _bar_with_err(
            ["Raw", "Corrected"],
            [raw_m, cor_m],
            [raw_s, cor_s],
            title=f"{metric.replace('_', ' ').title()} (macro, media su variabili)",
            ylabel="Score",
            out_path=out_dir / f"{metric}.png",
        )

    # 2) Exact match accuracy
    raw_m, raw_s = mstd("raw_exact_acc")
    cor_m, cor_s = mstd("corrected_exact_acc")
    _bar_with_err(
        ["Raw", "Corrected"],
        [raw_m, cor_m],
        [raw_s, cor_s],
        title="Exact-match Accuracy (tutte le variabili corrette insieme)",
        ylabel="Accuracy",
        out_path=out_dir / "exact_match_accuracy.png",
    )

    # 3) Consistency
    raw_m, raw_s = mstd("raw_consistency")
    cor_m, cor_s = mstd("corrected_consistency")
    _bar_with_err(
        ["Raw", "Corrected"],
        [raw_m, cor_m],
        [raw_s, cor_s],
        title="Consistency Rate (vincoli KB/CSP)",
        ylabel="Rate",
        out_path=out_dir / "consistency_rate.png",
    )


def plot_metric_over_budgets(
    budget_to_summary: Dict[int, Dict[str, Dict[str, float]]],
    metric_key: str,
    out_path: str | Path,
    title: str,
    ylabel: str = "Score",
) -> None:
    """Line plot con barre d'errore: Raw vs Corrected al variare del budget."""

    budgets = sorted(budget_to_summary.keys())
    raw_means = [float(budget_to_summary[b][f"raw_{metric_key}"]["mean"]) for b in budgets]
    raw_stds = [float(budget_to_summary[b][f"raw_{metric_key}"]["std"]) for b in budgets]
    cor_means = [float(budget_to_summary[b][f"corrected_{metric_key}"]["mean"]) for b in budgets]
    cor_stds = [float(budget_to_summary[b][f"corrected_{metric_key}"]["std"]) for b in budgets]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(budgets, raw_means, yerr=raw_stds, marker="o", linestyle="-")
    ax.errorbar(budgets, cor_means, yerr=cor_stds, marker="o", linestyle="-")
    ax.set_title(title)
    ax.set_xlabel("Budget (EUR)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.05)
    ax.legend(["Raw", "Corrected"], loc="best")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
