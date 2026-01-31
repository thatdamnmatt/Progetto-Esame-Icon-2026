from __future__ import annotations

from dataclasses import dataclass
import warnings
from typing import Dict, List, Tuple

import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning

from .data import VARIABLES


@dataclass
class TrainedModel:
    pipeline: Pipeline
    classes_: Dict[str, List[str]]  # var -> list of classes in estimator order


def train_model(
    X_train: np.ndarray,
    y_train: Dict[str, np.ndarray],
    seed: int = 0,
    hidden_layer_sizes: Tuple[int, ...] = (64, 32),
    alpha: float = 1e-4,
    max_iter: int = 500,
) -> TrainedModel:
    """Allena un predittore multi-output (una testa per variabile).

    Scelta tecnica: MultiOutputClassifier con MLPClassifier base.
    """

    Y = np.column_stack([y_train[v] for v in VARIABLES])

    bs = int(min(128, X_train.shape[0]))

    base = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=alpha,
        batch_size=bs,
        learning_rate_init=1e-3,
        max_iter=max_iter,
        random_state=seed,
        early_stopping=False,
        n_iter_no_change=10,
        verbose=False,
    )

    clf = MultiOutputClassifier(base)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        pipe.fit(X_train, Y)

    classes_: Dict[str, List[str]] = {}
    for v, est in zip(VARIABLES, pipe.named_steps["clf"].estimators_):
        classes_[v] = list(est.classes_)

    return TrainedModel(pipeline=pipe, classes_=classes_)


def predict_labels(model: TrainedModel, X: np.ndarray) -> Dict[str, np.ndarray]:
    Y_pred = model.pipeline.predict(X)
    return {v: Y_pred[:, i] for i, v in enumerate(VARIABLES)}


def predict_proba(model: TrainedModel, X: np.ndarray) -> List[Dict[str, Dict[str, float]]]:
    """Ritorna, per ogni esempio, un dict: var -> {label: prob}.

    Nota: MultiOutputClassifier espone predict_proba come lista di matrici,
    una per variabile.
    """

    probas = model.pipeline.named_steps["clf"].predict_proba(model.pipeline.named_steps["scaler"].transform(X))
    # probas: list of (n_samples, n_classes_var)

    out: List[Dict[str, Dict[str, float]]] = []
    n = X.shape[0]
    for i in range(n):
        ex: Dict[str, Dict[str, float]] = {}
        for var_idx, v in enumerate(VARIABLES):
            cls = model.classes_[v]
            pvec = probas[var_idx][i]
            ex[v] = {cls[j]: float(pvec[j]) for j in range(len(cls))}
        out.append(ex)
    return out
