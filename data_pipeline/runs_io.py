"""Lesen von runs/runs_summary.csv (auch mit von Hand eingefügten Trennzeilen)
und Auswahl der finalen Läufe je Konfiguration und Bewertungsebene.

In der Arbeit: legt fest, welche Modellläufe in Kapitel 6 und 7 berichtet
werden.

    python data_pipeline/runs_io.py [pfad]   # Datei normalisieren
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pandas as pd

from data_prep import SP


def _felder(zeile: str) -> list[str]:
    return next(csv.reader([zeile]))


def lies_summary(pfad: str | Path) -> pd.DataFrame:
    """runs_summary.csv einlesen, von Hand eingefügte Trennzeilen inbegriffen."""
    roh = Path(pfad).read_text(encoding="utf-8").splitlines()
    if not roh:
        return pd.DataFrame()

    kopf = [k.strip().rstrip(";") for k in _felder(roh[0])]
    n = len(kopf)

    puffer = io.StringIO()
    schreiber = csv.writer(puffer, lineterminator="\n")
    schreiber.writerow(kopf)
    for zeile in roh[1:]:
        if not zeile.strip():
            continue
        f = _felder(zeile)
        if len(f) != n:
            text = ",".join(x for x in f if x.strip() and x.strip() not in ";,")
            f = [text] + [""] * (n - 1)
        schreiber.writerow(f)

    puffer.seek(0)
    return pd.read_csv(puffer)


# Regel: je Konfiguration und Ebene der zuletzt geschriebene passende Lauf auf dem
# endgültigen Datenstand (Zeilenzahl) mit allen Merkmalsquellen.
N_ROWS_FINAL = SP["n_zeilen_final"]

EBENEN = {
    "billet": ("billet", "random"),
    "auftrag": ("auftrag", "auftrag"),
    "artikel": ("artikel", "artikel"),
}

# Konfiguration -> (Modellname, Muster in params_source, Bedingungen an model_params)
FINALE_KONFIG = {
    "XGBoost": ("XGBoost", "best_params_v3", {}),
    "XGBoost (Standard)": ("XGBoost", "default (kein Tuning", {}),
    "CatBoost": ("CatBoost", "catboost_best_params", {}),
    "TabPFN": ("TabPFN", None, {}),
    "ROCKET (Kurven)": (
        "ROCKET",
        "MiniRocket n_kernels=1000",
        {"hybrid": False, "znorm": True},
    ),
    "ROCKET (hybrid)": (
        "ROCKET",
        "MiniRocket n_kernels=1000",
        {"hybrid": True, "znorm": True},
    ),
    "ROCKET (hybrid, ohne z-Norm)": (
        "ROCKET",
        "MiniRocket n_kernels=1000",
        {"hybrid": True, "znorm": False},
    ),
    "HYDRA": ("HYDRA", None, {}),
    "FreshPRINCE": ("FreshPRINCE", None, {}),
    "hurdle": ("hurdle", None, {}),
}


def _model_params(run_dir: Path, run_id) -> dict:
    f = Path(run_dir) / f"run_{run_id}.json"
    if not f.exists():
        return {}
    d = json.loads(f.read_text(encoding="utf-8"))
    return d.get("model_params", {}) or {}


def waehle_finale_laeufe(
    summary: str | Path | pd.DataFrame,
    run_dir: str | Path,
    konfigs=None,
    ebenen=None,
) -> pd.DataFrame:
    """Je Konfiguration × Bewertungsebene die passende Summary-Zeile."""
    d = summary if isinstance(summary, pd.DataFrame) else lies_summary(summary)
    d = d[d.n_rows == N_ROWS_FINAL]
    d = d[d.feature_source.isna() | (d.feature_source == "beides")]
    zeilen = []
    for ebene in ebenen or EBENEN:
        tl, cl = EBENEN[ebene]
        for konfig in konfigs or FINALE_KONFIG:
            modell, muster, bedingung = FINALE_KONFIG[konfig]
            t = d[(d.model == modell) & (d.cv_level == cl)]
            if modell != "hurdle":
                t = t[t.test_level == tl]
            if muster:
                t = t[t.params_source.astype(str).str.contains(muster, regex=False)]
            if bedingung:
                passt = [
                    all(_model_params(run_dir, r).get(k) == v for k, v in bedingung.items())
                    for r in t.run_id
                ]
                t = t[passt]
            if t.empty:
                continue
            r = t.sort_values("run_id").iloc[-1].copy()
            r["konfig"] = konfig
            r["ebene"] = ebene
            zeilen.append(r)
    return pd.DataFrame(zeilen).reset_index(drop=True)


if __name__ == "__main__":
    import sys

    ziel = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else (Path(__file__).resolve().parent.parent / "runs" / "runs_summary.csv")
    )
    vorher = len(ziel.read_text(encoding="utf-8").splitlines())
    df = lies_summary(ziel)
    df.to_csv(ziel, index=False)
    print(
        f"{ziel}: {vorher} Zeilen gelesen → {len(df)} Datenzeilen, "
        f"{len(df.columns)} Spalten geschrieben"
    )
    print(
        "Trennzeilen erhalten:",
        int(df.iloc[:, 0].astype(str).str.contains("___").sum()),
    )
