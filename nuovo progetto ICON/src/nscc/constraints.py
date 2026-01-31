from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from .kg import ConstraintSpec

Config = Dict[str, str]  # e.g. {"CPU": "intel_i5", ...}


@dataclass(frozen=True)
class CompiledConstraints:
    """Vincoli in forma indicizzata per check rapidi."""

    incompatible: List[Tuple[Tuple[str, str], Tuple[str, str]]]
    # requires_any[(var1,val1)][var2] = set(allowed_values)  (OR)
    requires_any: Dict[Tuple[str, str], Dict[str, set[str]]]


def compile_constraints(spec: ConstraintSpec) -> CompiledConstraints:
    requires_any: Dict[Tuple[str, str], Dict[str, set[str]]] = {}
    for (v1, val1), (v2, val2) in spec.requires:
        key = (v1, val1)
        requires_any.setdefault(key, {}).setdefault(v2, set()).add(val2)
    return CompiledConstraints(incompatible=list(spec.incompatible), requires_any=requires_any)


def is_consistent(config: Mapping[str, str], cons: CompiledConstraints) -> bool:
    """Ritorna True se l'assegnazione totale soddisfa tutti i vincoli."""

    # --- incompatibilita' ---
    for (v1, val1), (v2, val2) in cons.incompatible:
        if config.get(v1) == val1 and config.get(v2) == val2:
            return False
        if config.get(v2) == val2 and config.get(v1) == val1:
            return False

    # --- requires (OR su valori alternativi dello stesso var2) ---
    for (v1, val1), reqs in cons.requires_any.items():
        if config.get(v1) != val1:
            continue
        for v2, allowed_vals in reqs.items():
            if config.get(v2) not in allowed_vals:
                return False

    return True
