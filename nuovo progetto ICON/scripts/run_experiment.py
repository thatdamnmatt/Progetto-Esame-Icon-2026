#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zipfile

# permette di eseguire lo script senza installare il pacchetto
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nscc.eval import run_crossval, summarize  # noqa: E402
from nscc.plots import plot_summary  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="NSCC - Neuro-Symbolic Constraint Checker")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--label-noise", type=float, default=0.12)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--budget-weight", type=float, default=3.0, help="peso penalità sforamento budget (CSP)")
    ap.add_argument("--spend-weight", type=float, default=0.0, help=">0 spinge a spendere vicino al budget")
    ap.add_argument("--no-user-prefs", action="store_true", help="non derivare preferenze utente dalle feature")

    ap.add_argument("--ml-hidden", type=str, default="64,32", help="hidden sizes MLP, es. 128,64")
    ap.add_argument("--ml-alpha", type=float, default=1e-4)
    ap.add_argument("--ml-max-iter", type=int, default=500)
    ap.add_argument("--out", type=str, default="results/summary.json")
    ap.add_argument("--plot", action="store_true", help="Salva grafici in results/")
    ap.add_argument(
        "--ctz-out",
        type=str,
        default=None,
        help="Se specificato, crea un archivio .ctz (ZIP) contenente summary e grafici.",
    )
    args = ap.parse_args()

    hidden = tuple(int(x.strip()) for x in args.ml_hidden.split(",") if x.strip())

    results = run_crossval(
        n_samples=args.n_samples,
        folds=args.folds,
        seed=args.seed,
        label_noise=args.label_noise,
        budget=args.budget,
        spend_weight=args.spend_weight,
        use_user_prefs=not args.no_user_prefs,
        model_params={"hidden_layer_sizes": hidden, "alpha": args.ml_alpha, "max_iter": args.ml_max_iter},
        corrector_params={"budget_weight": args.budget_weight, "spend_weight": args.spend_weight},
    )

    summary = summarize(results)

    print("\n=== Cross-validation summary ===")
    for k, v in summary.items():
        print(f"{k:>24}: mean={v['mean']:.4f}  std={v['std']:.4f}")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")

    if args.plot:
        figdir = out_path.parent
        plot_summary(summary, out_dir=figdir)
        print(f"Saved plots in: {figdir}")

    if args.ctz_out:
        ctz = Path(args.ctz_out)
        if not ctz.is_absolute():
            ctz = ROOT / ctz
        if ctz.suffix.lower() != ".ctz":
            ctz = ctz.with_suffix(ctz.suffix + ".ctz") if ctz.suffix else ctz.with_suffix(".ctz")
        ctz.parent.mkdir(parents=True, exist_ok=True)
        base_dir = out_path.parent
        with zipfile.ZipFile(ctz, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for p in sorted(base_dir.rglob("*")):
                if p.is_file():
                    z.write(p, arcname=str(p.relative_to(base_dir)))
        print(f"Saved results archive: {ctz}")


if __name__ == "__main__":
    main()
