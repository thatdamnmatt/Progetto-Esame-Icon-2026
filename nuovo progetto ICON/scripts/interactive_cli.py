from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# permette di eseguire lo script senza installare il pacchetto
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nscc.constraints import compile_constraints, is_consistent  # noqa: E402
from nscc.csp_corrector import correct_prediction  # noqa: E402
from nscc.data import VARIABLES, config_price, generate_synthetic_dataset  # noqa: E402
from nscc.kg import build_default_kg, extract_constraints, build_default_catalog  # noqa: E402
from nscc.model import train_model, predict_proba  # noqa: E402


CAT = build_default_catalog(seed=0)


def pretty(val: str) -> str:
    return CAT.display_name.get(val, val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, required=True, help="Budget in EUR, es. 1200")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=2500)
    ap.add_argument("--gpu-need", type=float, default=0.7, help="0..1 (0=office, 1=4K gaming)")
    ap.add_argument("--cpu-need", type=float, default=0.6, help="0..1 (0=office, 1=heavy productivity)")
    ap.add_argument("--ram-need", type=float, default=0.5, help="0..1 (0≈16GB, 1≈64GB)")
    ap.add_argument("--quality", type=float, default=0.7, help="0..1 qualità target globale")
    args = ap.parse_args()

    kg = build_default_kg(seed=0)
    cons = compile_constraints(extract_constraints(kg))

    ds = generate_synthetic_dataset(cons, n_samples=args.n_train, seed=args.seed, label_noise=0.10)
    model = train_model(ds.X, ds.y, seed=args.seed)

    # feature utente (5 feature): budget_norm, gpu_score_norm, cpu_score_norm, ram_norm, quality_target
    min_p, max_p = 500, 3000
    budget_norm = np.clip((args.budget - min_p) / (max_p - min_p), 0.0, 1.0)
    gpu_need = float(np.clip(args.gpu_need, 0.0, 1.0))
    cpu_need = float(np.clip(args.cpu_need, 0.0, 1.0))
    ram_need = float(np.clip(args.ram_need, 0.0, 1.0))
    quality = float(np.clip(args.quality, 0.0, 1.0))
    x_user = np.array([[budget_norm, gpu_need, cpu_need, ram_need, quality]], dtype=float)

    proba = predict_proba(model, x_user)[0]
    raw_cfg = {v: max(proba[v].items(), key=lambda kv: kv[1])[0] for v in VARIABLES}
    corr_cfg = correct_prediction(
        proba,
        cons,
        budget=args.budget,
        budget_weight=8.0,
        user_prefs={"gpu_need": gpu_need, "cpu_need": cpu_need, "ram_need": ram_need, "quality": quality},
    )

    print("\n=== INPUT UTENTE ===")
    print(f"Budget: {args.budget} EUR")

    print("\n=== PREDIZIONE RAW (ML) ===")
    for v in VARIABLES:
        print(f"{v:12s}: {pretty(raw_cfg[v])}")
    print(f"Consistent: {is_consistent(raw_cfg, cons)}")
    print(f"Prezzo: {config_price(raw_cfg)} EUR")

    print("\n=== CORRETTA (CSP + KB + Budget) ===")
    for v in VARIABLES:
        print(f"{v:12s}: {pretty(corr_cfg[v])}")
    print(f"Consistent: {is_consistent(corr_cfg, cons)}")
    print(f"Prezzo: {config_price(corr_cfg)} EUR")


if __name__ == "__main__":
    main()
