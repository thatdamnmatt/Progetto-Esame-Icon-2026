from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from .constraints import compile_constraints, is_consistent
from .csp_corrector import correct_prediction
from .data import VARIABLES, Dataset, generate_synthetic_dataset, config_price
from .kg import build_default_kg, extract_constraints
from .model import predict_proba, train_model


@dataclass
class FoldResult:
    # Exact match (tutte le variabili corrette insieme)
    raw_exact_acc: float
    corrected_exact_acc: float

    # Consistenza logica
    raw_consistency: float
    corrected_consistency: float

    # Budget (0 se budget None)
    raw_budget_violation: float
    corrected_budget_violation: float

    # Quante variabili cambiano in media
    corrected_avg_changes: float

    # Metriche di classificazione (macro mediate su variabili)
    raw_precision_macro: float
    raw_recall_macro: float
    raw_accuracy_macro: float

    corrected_precision_macro: float
    corrected_recall_macro: float
    corrected_accuracy_macro: float


def _exact_match(y_true: Dict[str, np.ndarray], y_pred: Dict[str, np.ndarray]) -> np.ndarray:
    ok = np.ones(len(next(iter(y_true.values()))), dtype=bool)
    for v in VARIABLES:
        ok &= (y_true[v] == y_pred[v])
    return ok


def _macro_metrics(y_true: Dict[str, np.ndarray], y_pred: Dict[str, np.ndarray]) -> Tuple[float, float, float]:
    """
    Calcola precision/recall/accuracy:
    - precision e recall: macro-average per variabile (poi media tra variabili)
    - accuracy: accuracy per variabile (poi media tra variabili)
    """
    precs: List[float] = []
    recs: List[float] = []
    accs: List[float] = []

    for v in VARIABLES:
        yt = y_true[v]
        yp = y_pred[v]
        p, r, _, _ = precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)
        precs.append(float(p))
        recs.append(float(r))
        accs.append(float(accuracy_score(yt, yp)))

    return float(np.mean(precs)), float(np.mean(recs)), float(np.mean(accs))


def run_crossval(
    n_samples: int = 2000,
    folds: int = 5,
    seed: int = 0,
    label_noise: float = 0.10,
    budget: Optional[int] = None,
    spend_weight: float = 0.0,
    use_user_prefs: bool = True,
    model_params: Optional[Dict[str, object]] = None,
    corrector_params: Optional[Dict[str, float]] = None,
) -> List[FoldResult]:
    kg = build_default_kg(seed=0)
    spec = extract_constraints(kg)
    cons = compile_constraints(spec)

    ds: Dataset = generate_synthetic_dataset(cons, n_samples=n_samples, seed=seed, label_noise=label_noise)
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

    results: List[FoldResult] = []

    model_params = model_params or {}
    corrector_params = corrector_params or {}

    for train_idx, test_idx in kf.split(ds.X):
        Xtr, Xte = ds.X[train_idx], ds.X[test_idx]
        ytr = {v: ds.y[v][train_idx] for v in VARIABLES}
        yte = {v: ds.y[v][test_idx] for v in VARIABLES}

        model = train_model(Xtr, ytr, seed=seed, **model_params)
        probas = predict_proba(model, Xte)

        raw_pred = {v: np.empty(len(test_idx), dtype=object) for v in VARIABLES}
        corr_pred = {v: np.empty(len(test_idx), dtype=object) for v in VARIABLES}

        raw_cons_ok = np.zeros(len(test_idx), dtype=bool)
        corr_cons_ok = np.zeros(len(test_idx), dtype=bool)

        raw_budget_ok = np.ones(len(test_idx), dtype=bool)
        corr_budget_ok = np.ones(len(test_idx), dtype=bool)

        changes = np.zeros(len(test_idx), dtype=float)

        for i, ex_proba in enumerate(probas):
            raw_cfg = {v: max(ex_proba[v].items(), key=lambda kv: kv[1])[0] for v in VARIABLES}
            raw_cons_ok[i] = is_consistent(raw_cfg, cons)
            if budget is not None:
                raw_budget_ok[i] = config_price(raw_cfg) <= budget

            # Deriva preferenze utente dalle feature (0..1) se richiesto.
            user_prefs = None
            if use_user_prefs:
                user_prefs = {
                    "gpu_need": float(Xte[i, 1]),
                    "cpu_need": float(Xte[i, 2]),
                    "ram_need": float(Xte[i, 3]),
                    "quality": float(Xte[i, 4]),
                }

            corr_cfg = correct_prediction(
                ex_proba,
                cons,
                budget=budget,
                budget_weight=float(corrector_params.get("budget_weight", 3.0)),
                spend_weight=float(corrector_params.get("spend_weight", spend_weight)),
                user_prefs=user_prefs,
            )
            corr_cons_ok[i] = is_consistent(corr_cfg, cons)
            if budget is not None:
                corr_budget_ok[i] = config_price(corr_cfg) <= budget

            changes[i] = sum(1 for v in VARIABLES if corr_cfg[v] != raw_cfg[v])

            for v in VARIABLES:
                raw_pred[v][i] = raw_cfg[v]
                corr_pred[v][i] = corr_cfg[v]

        raw_exact = float(_exact_match(yte, raw_pred).mean())
        corr_exact = float(_exact_match(yte, corr_pred).mean())

        raw_p, raw_r, raw_a = _macro_metrics(yte, raw_pred)
        cor_p, cor_r, cor_a = _macro_metrics(yte, corr_pred)

        results.append(
            FoldResult(
                raw_exact_acc=raw_exact,
                corrected_exact_acc=corr_exact,
                raw_consistency=float(raw_cons_ok.mean()),
                corrected_consistency=float(corr_cons_ok.mean()),
                raw_budget_violation=float(1.0 - raw_budget_ok.mean()) if budget is not None else 0.0,
                corrected_budget_violation=float(1.0 - corr_budget_ok.mean()) if budget is not None else 0.0,
                corrected_avg_changes=float(changes.mean()),
                raw_precision_macro=raw_p,
                raw_recall_macro=raw_r,
                raw_accuracy_macro=raw_a,
                corrected_precision_macro=cor_p,
                corrected_recall_macro=cor_r,
                corrected_accuracy_macro=cor_a,
            )
        )

    return results


def summarize(results: List[FoldResult]) -> Dict[str, Dict[str, float]]:
    def ms(vals: List[float]) -> Dict[str, float]:
        arr = np.array(vals, dtype=float)
        return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0}

    return {
        "raw_exact_acc": ms([r.raw_exact_acc for r in results]),
        "corrected_exact_acc": ms([r.corrected_exact_acc for r in results]),
        "raw_consistency": ms([r.raw_consistency for r in results]),
        "corrected_consistency": ms([r.corrected_consistency for r in results]),
        "raw_budget_violation": ms([r.raw_budget_violation for r in results]),
        "corrected_budget_violation": ms([r.corrected_budget_violation for r in results]),
        "corrected_avg_changes": ms([r.corrected_avg_changes for r in results]),
        "raw_precision_macro": ms([r.raw_precision_macro for r in results]),
        "raw_recall_macro": ms([r.raw_recall_macro for r in results]),
        "raw_accuracy_macro": ms([r.raw_accuracy_macro for r in results]),
        "corrected_precision_macro": ms([r.corrected_precision_macro for r in results]),
        "corrected_recall_macro": ms([r.corrected_recall_macro for r in results]),
        "corrected_accuracy_macro": ms([r.corrected_accuracy_macro for r in results]),
    }
