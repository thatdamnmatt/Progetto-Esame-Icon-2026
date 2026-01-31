from __future__ import annotations

"""Correttore CSP per rendere coerenti (e opzionalmente budget-aware) le predizioni ML."""

import math
from typing import Dict, List, Mapping, Optional, Tuple

from .constraints import CompiledConstraints, is_consistent
from .data import DOMAINS, PRICES, VARIABLES, CATALOG


def config_price(cfg: Mapping[str, str]) -> int:
    return int(sum(PRICES[cfg[v]] for v in VARIABLES))


def _partial_violates(partial: Mapping[str, str], cons: CompiledConstraints) -> bool:
    """Potatura su assegnazione parziale."""

    # incompatibilita'
    for (v1, val1), (v2, val2) in cons.incompatible:
        if v1 in partial and v2 in partial:
            if partial[v1] == val1 and partial[v2] == val2:
                return True

    # requires-any: se antecedente assegnato, e conseguente assegnato, controlla
    for (v1, val1), reqs in cons.requires_any.items():
        if partial.get(v1) != val1:
            continue
        for v2, allowed in reqs.items():
            if v2 in partial and partial[v2] not in allowed:
                return True

    return False


def _assignment_cost(val: str, proba: Optional[Mapping[str, float]]) -> float:
    """Costo (da minimizzare) dell'assegnazione in base alle probabilita' ML."""
    if not proba:
        return 0.0
    p = float(proba.get(val, 0.0))
    p = max(p, 1e-9)
    return -math.log(p)


def _budget_penalty(price: int, budget: Optional[int], budget_weight: float) -> float:
    """Penalizza lo sforamento budget (soft-constraint).

    - se price <= budget: 0
    - altrimenti: penalità quadratica, scalata da budget_weight
    """

    if budget is None or budget_weight <= 0:
        return 0.0
    over = max(0, price - int(budget))
    return float(budget_weight) * (over / 250.0) ** 2


def _spend_near_budget_penalty(price: int, budget: Optional[int], spend_weight: float) -> float:
    """Penalizza l'"underuse" del budget, per spingere a spendere vicino al budget.

    Questo serve a spiegare perché con budget alto si può ottenere una build più cara:
    se spend_weight>0, il solver preferisce soluzioni con prezzo vicino a budget.
    """

    if budget is None or spend_weight <= 0:
        return 0.0
    under = max(0, int(budget) - price)
    return float(spend_weight) * (under / 350.0) ** 2


def _prefs_penalty(
    cfg: Mapping[str, str],
    user_prefs: Optional[Mapping[str, float]],
    budget: Optional[int] = None,
) -> float:
    """Penalità (soft) per mismatch con preferenze utente.

    user_prefs attesi (0..1): gpu_need, cpu_need, ram_need, quality
    - gpu_need: quanto conta la GPU (gaming)
    - cpu_need: quanto conta la CPU (produttività)
    - ram_need: desiderio di RAM (0≈16GB, 1≈64GB)
    - quality: target globale, serve a non proporre "4K" a budget bassi.
    """

    if not user_prefs:
        return 0.0

    gpu_need = float(user_prefs.get("gpu_need", 0.5))
    cpu_need = float(user_prefs.get("cpu_need", 0.5))
    ram_need = float(user_prefs.get("ram_need", 0.5))
    quality = float(user_prefs.get("quality", 0.5))

    # Cap del target di qualità in base al budget: evita richieste "4K" con budget bassi.
    # - Sotto ~700€ -> cap ~0.2
    # - Oltre ~2500€ -> cap ~1.0
    if budget is not None:
        budget_cap = max(0.0, min(1.0, (int(budget) - 700) / (2500 - 700)))
        quality = min(quality, float(budget_cap))

    gpu_sc = CATALOG.gpu_score[cfg["GPU"]]  # 0..100
    cpu_sc = CATALOG.cpu_score[cfg["CPU"]]  # 0..100
    ram_gb = CATALOG.ram_gb[cfg["RAM"]]

    # target in 0..100
    # qualità globale sposta tutti i target verso l'alto/basso
    base_target = 25 + 70 * quality  # 25..95
    gpu_target = base_target * (0.6 + 0.4 * gpu_need)
    cpu_target = base_target * (0.6 + 0.4 * cpu_need)
    ram_target_gb = 16 + int(round(48 * ram_need))  # 16..64

    # pesi: GPU pesa di più quando gpu_need è alto, ecc.
    w_gpu = 1.8 * (0.3 + 0.7 * gpu_need)
    w_cpu = 1.3 * (0.3 + 0.7 * cpu_need)
    w_ram = 0.6 * (0.3 + 0.7 * ram_need)

    # costo quadratico normalizzato
    gpu_term = w_gpu * ((gpu_sc - gpu_target) / 35.0) ** 2
    cpu_term = w_cpu * ((cpu_sc - cpu_target) / 30.0) ** 2
    ram_term = w_ram * ((ram_gb - ram_target_gb) / 24.0) ** 2

    # Penalizza anche componenti "overkill" quando l'utente vuole office/base:
    # - PSU troppo sopra il minimo raccomandato
    # - Case full tower quando non richiesto
    psu_over = 0.0
    if "PSU" in cfg and "GPU" in cfg:
        min_psu = CATALOG.gpu_psu_min.get(cfg["GPU"], 650)
        w = CATALOG.psu_watt.get(cfg["PSU"], 650)
        # penalizza solo quando quality è bassa
        psu_over = (max(0, w - min_psu) / 350.0) ** 2 * (1.2 * (1.0 - quality))

    case_over = 0.0
    if cfg.get("Case") == "case_atx_full_tower":
        case_over = 0.6 * (1.0 - quality)

    return float(gpu_term + cpu_term + ram_term + psu_over + case_over)


def correct_prediction(
    pred_proba: Mapping[str, Mapping[str, float]],
    cons: CompiledConstraints,
    budget: Optional[int] = None,
    budget_weight: float = 3.0,
    spend_weight: float = 0.0,
    user_prefs: Optional[Mapping[str, float]] = None,
    domains: Mapping[str, List[str]] | None = None,
) -> Dict[str, str]:
    """Trova la configurazione consistente più vicina alla predizione ML.

    Funzione obiettivo (min):
      - somma di -log(prob) (distanza dal modello)
      - + penalità budget (sforamento)
      - + penalità underuse (opzionale)
      - + mismatch con preferenze utente (prestazioni target)

    Politica budget (importante per l'"uso realistico"):
      1) prova a cercare SOLO soluzioni con prezzo <= budget (hard)
      2) se non esiste, passa a budget soft minimizzando lo sforamento.
    """

    domains = domains or DOMAINS

    # ---------------------------------------------------------------------
    # Conservative policy (safe):
    # If the raw ML argmax is already consistent AND within budget, keep it
    # ONLY when no explicit user preferences are provided.
    #
    # Why? With preferences (e.g., "office"), the user may want a cheaper/
    # lower-tier build even if the raw prediction is valid. Early-returning
    # would prevent the CSP from aligning the configuration to those prefs.
    # ---------------------------------------------------------------------
    raw_cfg: Dict[str, str] = {
        v: (max(pred_proba.get(v, {}).items(), key=lambda kv: kv[1])[0] if pred_proba.get(v) else domains[v][0])
        for v in VARIABLES
    }
    if user_prefs is None:
        if is_consistent(raw_cfg, cons):
            if budget is None or config_price(raw_cfg) <= int(budget):
                return raw_cfg

    best_cfg: Dict[str, str] | None = None
    best_cost: float = float("inf")

    # ordina variabili: prima quelle con output più "certo" (max prob alto)
    def sharpness(v: str) -> float:
        pv = pred_proba.get(v, {})
        return max(pv.values()) if pv else 0.0

    var_order = sorted(VARIABLES, key=sharpness, reverse=True)

    # pre-compute: costo minimo per variabile (per un bound ottimistico)
    min_cost_per_var: Dict[str, float] = {}
    for v in VARIABLES:
        pv = pred_proba.get(v, {})
        if not pv:
            min_cost_per_var[v] = 0.0
        else:
            # minimo -log(p)
            min_cost_per_var[v] = min(_assignment_cost(val, pv) for val in domains[v])

    def optimistic_remaining(i: int) -> float:
        return sum(min_cost_per_var[var_order[j]] for j in range(i, len(var_order)))

    # bound sul prezzo (per pruning budget hard)
    min_price_per_var = {v: min(PRICES[val] for val in domains[v]) for v in VARIABLES}

    def optimistic_price_remaining(i: int) -> int:
        return int(sum(min_price_per_var[var_order[j]] for j in range(i, len(var_order))))

    def backtrack(i: int, partial: Dict[str, str], cost_so_far: float, hard_budget: bool) -> None:
        nonlocal best_cfg, best_cost

        # bound
        if cost_so_far + optimistic_remaining(i) >= best_cost:
            return

        if _partial_violates(partial, cons):
            return

        # pruning budget hard: se anche col minimo possibile si sfora, taglia
        if hard_budget and budget is not None:
            current_price = sum(PRICES[partial[v]] for v in partial)
            if current_price > int(budget):
                return
            if current_price + optimistic_price_remaining(i) > int(budget):
                return

        if i == len(var_order):
            if not is_consistent(partial, cons):
                return
            price = config_price(partial)
            if hard_budget and budget is not None and price > int(budget):
                return
            total = cost_so_far
            total += _budget_penalty(price, budget, budget_weight)
            total += _spend_near_budget_penalty(price, budget, spend_weight)
            total += _prefs_penalty(partial, user_prefs, budget=budget)

            # Price regularization: when the user wants a "base/office" build
            # (quality low), prefer cheaper configs even if the budget is high.
            if user_prefs is not None:
                q = float(user_prefs.get("quality", 0.5))
                # weight in [0.0..1.2] approx
                price_w = 1.2 * max(0.0, min(1.0, 1.0 - q))
                total += price_w * (price / 1000.0) ** 2
            if total < best_cost:
                best_cost = total
                best_cfg = dict(partial)
            return

        var = var_order[i]
        pv = pred_proba.get(var, {})
        values = list(domains[var])
        values.sort(key=lambda x: pv.get(x, 0.0), reverse=True)

        for val in values:
            partial[var] = val
            c = _assignment_cost(val, pv)
            backtrack(i + 1, partial, cost_so_far + c, hard_budget)
            del partial[var]

    # 1) tenta hard budget (se budget dato)
    if budget is not None:
        backtrack(0, {}, 0.0, hard_budget=True)

    # 2) fallback: budget soft (minimizza sforamento)
    if best_cfg is None:
        backtrack(0, {}, 0.0, hard_budget=False)

    # fallback: se vincoli inconsistenti o qualcosa va storto
    if best_cfg is None:
        out: Dict[str, str] = {}
        for v in VARIABLES:
            pv = pred_proba.get(v, {})
            if pv:
                out[v] = max(pv.items(), key=lambda kv: kv[1])[0]
            else:
                out[v] = domains[v][0]
        return out

    return best_cfg
