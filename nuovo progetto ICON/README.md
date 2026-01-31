# NSCC — Neuro-Symbolic Constraint Checker (PC Configurator)

Mini-progetto ICon: integrazione **ML + KB(KG) + CSP**.

## Setup (Windows)

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Esecuzione

### Avvio con un click (Windows)

- `start_gui.bat` → installa le dipendenze (se mancano) e avvia la GUI
- `start_experiment.bat` → installa le dipendenze (se mancano) e lancia la cross-validation con grafici

### 1) Cross-validation + grafici

```bat
python scripts\run_experiment.py --seed 0 --n-samples 2000 --folds 5 --plot
```

Output:
- `results/summary.json`
- grafici `.png` in `results/` (precision/recall/accuracy, exact-match, consistency)

### 1b) Benchmark ampio (multi-seed, multi-budget) + grafici per la relazione

Consigliato per documentazione (media ± dev. standard su molti run):

```bat
py -3 scripts/run_benchmark.py --seeds 0..9 --budgets 800,1000,1200,1400,1600,1800,2000 \
  --n-samples 15000 --folds 5 --outdir results/bench_big --plot
```

Output:
- `results/bench_big/summary_by_budget.json`
- `results/bench_big/folds.csv` (fold-level, utile per tabelle)
- grafici `.png` in `results/bench_big/` (metriche vs budget)

### 1c) Tuning (ottimizzazione) iperparametri ML + pesi CSP

Per scegliere parametri “buoni” in modo riproducibile (grid search):

```bat
py -3 scripts/tune.py --budget 1200 --seeds 0..2 --n-samples 6000 --folds 3 --out results/tune_best.json
```

Poi puoi riusare i parametri migliori passando i valori a `run_experiment.py` / `run_benchmark.py`.

### 2) CLI interattiva (budget)

```bat
python scripts\interactive_cli.py --budget 1500 --seed 0 --spend-weight 0.5
```

### 3) GUI (Tkinter)

```bat
python scripts\gui.py
```

Note:
- il correttore è **conservativo** solo quando NON vengono fornite preferenze utente: se la predizione ML è già
  coerente e rispetta il budget, non modifica la configurazione. Con preferenze (es. "office") può invece
  abbassare i tier per allinearsi al profilo scelto.
- `Spend near budget` è pensato per build ad alte prestazioni: nella GUI viene pesato anche
  dalla "GPU demand" per evitare che preferenze low-end forzino comportamenti tipo "spendi tutto".
  L'opzione è "gated" dalla preferenza GPU: con GPU demand bassa l'effetto è ridotto,
  per evitare comportamento "spendi tutto" su build low-end.


## Catalogo ampio (opzionale)

Per usare un dominio **molto più ampio** basato su un dataset online (PCPartPicker), esegui:

```bash
python scripts/fetch_catalog.py --max-per-category 150
```

Questo genera `data/catalog_curated.json`. Da quel momento il progetto userà automaticamente quel catalogo (fallback al catalogo interno se il file non esiste).

Fonte dataset: docyx/pc-part-dataset (MIT) su GitHub.
