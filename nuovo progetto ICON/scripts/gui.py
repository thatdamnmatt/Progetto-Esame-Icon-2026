from __future__ import annotations

import sys
from pathlib import Path

# permette di eseguire lo script senza installare il pacchetto
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np

from nscc.constraints import compile_constraints, is_consistent
from nscc.csp_corrector import correct_prediction
from nscc.data import VARIABLES, config_price, generate_synthetic_dataset
from nscc.kg import build_default_kg, extract_constraints, build_default_catalog
from nscc.model import train_model, predict_proba


CAT = build_default_catalog(seed=0)


def pretty(val: str) -> str:
    return CAT.display_name.get(val, val)


def estimate_min_price() -> int:
    return int(sum(min(CAT.prices[v] for v in CAT.domains[var]) for var in CAT.variables))


def make_user_features(budget: int, gpu_need: float, cpu_need: float, ram_need: float, quality: float) -> np.ndarray:
    """Crea il vettore feature coerente col dataset sintetico (5 feature)."""
    min_p, max_p = 500, 3000
    budget_norm = np.clip((budget - min_p) / (max_p - min_p), 0.0, 1.0)
    gpu_score_norm = float(np.clip(gpu_need, 0.0, 1.0))
    cpu_score_norm = float(np.clip(cpu_need, 0.0, 1.0))
    ram_norm = float(np.clip(ram_need, 0.0, 1.0))
    quality_target = float(np.clip(quality, 0.0, 1.0))
    return np.array([[budget_norm, gpu_score_norm, cpu_score_norm, ram_norm, quality_target]], dtype=float)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NSCC - PC Configurator (Neuro-Symbolic)")
        self.geometry("980x640")

        kg = build_default_kg(seed=0)
        self.cons = compile_constraints(extract_constraints(kg))

        ds = generate_synthetic_dataset(self.cons, n_samples=2500, seed=0, label_noise=0.10)
        self.model = train_model(ds.X, ds.y, seed=0)

        self.min_budget = estimate_min_price()

        self._build_ui()

    def _build_ui(self):
        frm = ttk.Frame(self, padding=12)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="NSCC — PC Configurator (ML + KB + CSP)", font=("Segoe UI", 14, "bold")).grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 12)
        )

        ttk.Label(frm, text="Budget (EUR):").grid(row=1, column=0, sticky="w")
        self.budget_var = tk.StringVar(value="1200")
        ttk.Entry(frm, textvariable=self.budget_var, width=10).grid(row=1, column=1, sticky="w", padx=(6, 18))

        ttk.Label(frm, text=f"Minimo stimato build completa: ~{self.min_budget} EUR", foreground="#555").grid(
            row=1, column=2, sticky="w"
        )

        ttk.Label(frm, text="GPU demand (0=office, 1=4K gaming):").grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.gpu_need = tk.DoubleVar(value=0.7)
        ttk.Scale(frm, from_=0.0, to=1.0, variable=self.gpu_need, orient="horizontal").grid(
            row=2, column=1, columnspan=3, sticky="we", pady=(10, 0)
        )

        ttk.Label(frm, text="CPU demand (0=office, 1=heavy productivity):").grid(row=3, column=0, sticky="w", pady=(10, 0))
        self.cpu_need = tk.DoubleVar(value=0.6)
        ttk.Scale(frm, from_=0.0, to=1.0, variable=self.cpu_need, orient="horizontal").grid(
            row=3, column=1, columnspan=3, sticky="we", pady=(10, 0)
        )

        ttk.Label(frm, text="RAM need (0=16GB, 1=64GB):").grid(row=4, column=0, sticky="w", pady=(10, 0))
        self.ram_need = tk.DoubleVar(value=0.5)
        ttk.Scale(frm, from_=0.0, to=1.0, variable=self.ram_need, orient="horizontal").grid(
            row=4, column=1, columnspan=3, sticky="we", pady=(10, 0)
        )

        ttk.Label(frm, text="Qualità target (0=base, 1=top-tier):").grid(row=5, column=0, sticky="w", pady=(10, 0))
        self.quality = tk.DoubleVar(value=0.7)
        ttk.Scale(frm, from_=0.0, to=1.0, variable=self.quality, orient="horizontal").grid(
            row=5, column=1, columnspan=3, sticky="we", pady=(10, 0)
        )

        btns = ttk.Frame(frm)
        btns.grid(row=6, column=0, columnspan=4, sticky="w", pady=(14, 10))
        ttk.Button(btns, text="Calcola configurazione", command=self.on_compute).pack(side="left")
        ttk.Button(btns, text="Proponi alternativa", command=self.on_alternative).pack(side="left", padx=8)

        self.out = tk.Text(frm, height=22, width=115)
        self.out.grid(row=7, column=0, columnspan=4, sticky="nsew", pady=(10, 0))
        frm.rowconfigure(7, weight=1)
        frm.columnconfigure(3, weight=1)

        self._write("GUI pronta. Inserisci budget e premi 'Calcola'.\n")

    def _write(self, s: str):
        self.out.insert("end", s)
        self.out.see("end")

    def _parse_budget(self) -> int | None:
        try:
            return int(self.budget_var.get().strip())
        except Exception:
            return None

    def _compute(self, budget: int, jitter: float = 0.0):
        if budget <= 0:
            messagebox.showerror("Errore", "Budget non valido.")
            return

        if budget < self.min_budget:
            messagebox.showwarning(
                "Budget troppo basso",
                f"Budget={budget} EUR, minimo stimato ~{self.min_budget} EUR.\n"
                f"Il CSP cercherà la soluzione con MINIMO sforamento.",
            )

        x = make_user_features(
            budget=budget,
            gpu_need=float(np.clip(self.gpu_need.get() + jitter, 0.0, 1.0)),
            cpu_need=float(np.clip(self.cpu_need.get() + jitter, 0.0, 1.0)),
            ram_need=float(np.clip(self.ram_need.get() + jitter, 0.0, 1.0)),
            quality=float(np.clip(self.quality.get() + jitter, 0.0, 1.0)),
        )

        proba = predict_proba(self.model, x)[0]
        raw_cfg = {v: max(proba[v].items(), key=lambda kv: kv[1])[0] for v in VARIABLES}

        corr_cfg = correct_prediction(
            proba,
            self.cons,
            budget=budget,
            # preferisci rispettare il budget quando possibile
            budget_weight=8.0,
            # preferenze utente (usate come soft constraints/utility)
            user_prefs={
                "gpu_need": float(self.gpu_need.get()),
                "cpu_need": float(self.cpu_need.get()),
                "ram_need": float(self.ram_need.get()),
                "quality": float(self.quality.get()),
            },
        )

        self._write("\n============================\n")
        self._write(f"BUDGET: {budget} EUR\n")

        self._write("--- RAW (ML) ---\n")
        for v in VARIABLES:
            self._write(f"{v:12s}: {pretty(raw_cfg[v])}\n")
        self._write(f"Consistent: {is_consistent(raw_cfg, self.cons)}\n")
        self._write(f"Prezzo: {config_price(raw_cfg)} EUR\n\n")

        self._write("--- CORRETTA (CSP + KB + Budget) ---\n")
        for v in VARIABLES:
            self._write(f"{v:12s}: {pretty(corr_cfg[v])}\n")
        self._write(f"Consistent: {is_consistent(corr_cfg, self.cons)}\n")
        self._write(f"Prezzo: {config_price(corr_cfg)} EUR\n")

        if config_price(corr_cfg) > budget:
            self._write("NOTE: budget NON rispettato (minimizzato lo sforamento).\n")
        else:
            self._write("NOTE: budget rispettato.\n")

    def on_compute(self):
        budget = self._parse_budget()
        if budget is None:
            messagebox.showerror("Errore", "Inserisci un budget intero, es. 1200")
            return
        self._compute(budget, jitter=0.0)

    def on_alternative(self):
        budget = self._parse_budget()
        if budget is None:
            messagebox.showerror("Errore", "Inserisci un budget intero, es. 1200")
            return
        self._compute(budget, jitter=0.08)


if __name__ == "__main__":
    App().mainloop()
