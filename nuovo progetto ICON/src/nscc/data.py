from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .constraints import CompiledConstraints, is_consistent
from .kg import build_default_catalog


CATALOG = build_default_catalog(seed=0)

VARIABLES: List[str] = CATALOG.variables
DOMAINS: Dict[str, List[str]] = CATALOG.domains
PRICES: Dict[str, int] = CATALOG.prices


@dataclass
class Dataset:
    X: np.ndarray
    y: Dict[str, np.ndarray]


def config_price(cfg: Dict[str, str]) -> int:
    return int(sum(PRICES[v] for v in cfg.values()))


def sample_consistent_config(
    cons: CompiledConstraints,
    rng: np.random.Generator,
    max_tries: int = 10000,
) -> Dict[str, str]:
    for _ in range(max_tries):
        cfg = {v: rng.choice(DOMAINS[v]) for v in VARIABLES}
        if is_consistent(cfg, cons):
            return cfg
    raise RuntimeError("Impossibile campionare una configurazione consistente: controlla i vincoli")


def _make_features(cfg: Dict[str, str], rng: np.random.Generator, budget_hint: Optional[int] = None) -> np.ndarray:
    """
    Feature numeriche (semplici ma utili) - 5 feature:
      1) budget_norm        (stimato/rumoroso)
      2) gpu_score_norm     (proxy prestazioni GPU 0..1)
      3) cpu_score_norm     (proxy prestazioni CPU 0..1)
      4) ram_norm           (GB RAM normalizzati)
      5) quality_target     (target utente 0..1, correlato alla build "vera")
    """
    price = config_price(cfg)

    if budget_hint is None:
        budget_hint = int(price + rng.normal(0, 120))

    min_p, max_p = 500, 3000
    budget_norm = np.clip((budget_hint - min_p) / (max_p - min_p), 0.0, 1.0)

    gpu = cfg["GPU"]
    gpu_score_norm = np.clip(CATALOG.gpu_score[gpu] / 100.0, 0.0, 1.0)

    cpu = cfg["CPU"]
    cpu_score_norm = np.clip(CATALOG.cpu_score[cpu] / 100.0, 0.0, 1.0)

    ram = cfg["RAM"]
    ram_norm = np.clip((CATALOG.ram_gb[ram] - 16) / (64 - 16), 0.0, 1.0)

    # target utente: build più forte -> target più alto (ma non perfetto, c'è rumore)
    quality_target = float(np.clip(0.55 * gpu_score_norm + 0.35 * cpu_score_norm + 0.10 * ram_norm, 0.0, 1.0))
    quality_target = float(np.clip(quality_target + rng.normal(0, 0.08), 0.0, 1.0))

    feat = np.array([budget_norm, gpu_score_norm, cpu_score_norm, ram_norm, quality_target], dtype=float)
    feat += rng.normal(0, 0.03, size=feat.shape)
    return np.clip(feat, 0.0, 1.0)


def generate_synthetic_dataset(
    cons: CompiledConstraints,
    n_samples: int,
    seed: int = 0,
    label_noise: float = 0.10,
) -> Dataset:
    """
    Dataset sintetico:
    - ground truth sempre consistente (campionamento con vincoli)
    - feature includono un budget rumoroso
    - label_noise corrompe alcune etichette (simula predizioni/dati imperfetti)
    """
    rng = np.random.default_rng(seed)

    X = np.zeros((n_samples, 5), dtype=float)
    y = {v: np.empty(n_samples, dtype=object) for v in VARIABLES}

    for i in range(n_samples):
        cfg = sample_consistent_config(cons, rng)

        true_price = config_price(cfg)
        user_budget = int(true_price + rng.normal(0, 180))
        user_budget = max(500, min(3000, user_budget))

        X[i] = _make_features(cfg, rng, budget_hint=user_budget)

        noisy = dict(cfg)
        for v in VARIABLES:
            if rng.random() < label_noise:
                noisy[v] = rng.choice(DOMAINS[v])

        for v in VARIABLES:
            y[v][i] = noisy[v]

    return Dataset(X=X, y=y)
