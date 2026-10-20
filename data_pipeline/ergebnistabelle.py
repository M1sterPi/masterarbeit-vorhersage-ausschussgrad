"""Kennzahlen aller finalen Modellläufe als Tabelle
(Auswahl: runs_io.waehle_finale_laeufe).

In der Arbeit: Kennzahlen aller finalen Läufe für Abschnitt 5.7 (MiniRocket mit
und ohne z-Normierung), 6.2, 6.4, die Bezugsgrößen in 7.2 und Abschnitt 8.1.

Ergebnis: runs/ergebnisse_final.csv, mit hybrid und znorm aus dem Run-JSON.
Berichtet werden Faltenmittel (cv_*_mean); die hurdle_*-Werte sind gepoolt.

    python data_pipeline/ergebnistabelle.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_prep as D  # noqa: E402
from runs_io import FINALE_KONFIG, EBENEN, waehle_finale_laeufe  # noqa: E402

ZIEL = D.RUNS_DIR / "ergebnisse_final.csv"

SPALTEN = [
    "konfig",
    "ebene",
    "run_id",
    "model",
    "test_level",
    "cv_level",
    "hybrid",
    "znorm",
    "params_source",
    "n_rows",
    "n_train",
    "n_test",
    "n_features",
    "cv_mae_mean",
    "cv_mae_std",
    "cv_r2_mean",
    "cv_r2_std",
    "cv_baseline_median_mae_mean",
    "cv_baseline_mittel_r2_mean",
    "test_mae",
    "test_r2",
    "baseline_median_mae",
    "baseline_mittel_r2",
    "hurdle_mae_alle",
    "hurdle_r2_alle",
    "einzel_mae_alle",
    "einzel_r2_alle",
    "hurdle_mae_ohne",
    "hurdle_r2_ohne",
    "bedingt_mae_ohne",
    "bedingt_r2_ohne",
    "stufe_a_auc",
    "stufe_a_ap",
    "data_sha256",
]


def main() -> None:
    t = waehle_finale_laeufe(D.RUNS_DIR / "runs_summary.csv", D.RUN_ARTIFACTS_DIR)
    if t.empty:
        raise SystemExit("Keine finalen Läufe in runs/runs_summary.csv gefunden.")
    for k in ("hybrid", "znorm"):
        werte = []
        for rid in t.run_id:
            d = json.loads((D.RUN_ARTIFACTS_DIR / f"run_{rid}.json").read_text(encoding="utf-8"))
            mp = d.get("model_params") or {}
            cfg = d.get("config") or {}
            werte.append(mp.get(k, cfg.get(k)))
        t[k] = werte
    t["konfig"] = pd.Categorical(t.konfig, categories=list(FINALE_KONFIG), ordered=True)
    t["ebene"] = pd.Categorical(t.ebene, categories=list(EBENEN), ordered=True)
    t = t.sort_values(["konfig", "ebene"])[[c for c in SPALTEN if c in t.columns]]
    ZIEL.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(ZIEL, index=False, encoding="utf-8")

    fehlend = [
        (k, e)
        for k in FINALE_KONFIG
        for e in EBENEN
        if not ((t.konfig == k) & (t.ebene == e)).any()
    ]
    zeigen = [c for c in ("konfig", "ebene", "run_id", "znorm", "cv_mae_mean", "cv_r2_mean",
                          "hurdle_mae_alle") if c in t.columns]
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(t[zeigen].to_string(index=False))
    if fehlend:
        print("\nOhne Lauf:", ", ".join(f"{k}/{e}" for k, e in fehlend))
    print(f"\ngeschrieben: {ZIEL.relative_to(D.ROOT)} ({len(t)} Zeilen)")


if __name__ == "__main__":
    main()
