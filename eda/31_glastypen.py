"""31 — Ausschussgrad je Glastyp.

In der Arbeit: Abschnitt 7.4: Ausschussgrad und Zahl der Billets je Glastyp.

Ausgabe: reports/eda/31_glastypen.csv, runs/eda/31_glastypen.json.

    python eda/31_glastypen.py
"""

from __future__ import annotations

import pandas as pd

import _common as C
from data_prep import SP  # noqa: E402


def main():
    ds = C.lade(alle_arten=True)
    X, ys, df = C.daten_teil(ds)
    t = pd.DataFrame(
        {
            "Glastyp": X["Glastyp"].astype(str),
            "y": ys.to_numpy(),
            "Artikel": df[SP["artikel"]].astype(str).to_numpy(),
        }
    )
    tab = (
        t.groupby("Glastyp")
        .agg(n=("y", "size"), artikel=("Artikel", "nunique"), median=("y", "median"), mittel=("y", "mean"))
        .sort_values("n", ascending=False)
    )
    print(tab.round(2).to_string())
    kennung = t.Glastyp.str.contains(SP["glastyp_kennung"])
    print(
        f"\n{SP['glastyp_kennung']}-Glastypen: {int(kennung.sum())} von {len(t)} Billets, Median je Typ "
        + ", ".join(f"{k} {v:.1f} %" for k, v in tab.loc[tab.index.str.contains(SP['glastyp_kennung']), 'median'].items())
        + f"; gesamt Median {ys.median():.1f} %"
    )
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    tab.to_csv(C.OUT_DIR / "31_glastypen.csv", encoding="utf-8")
    C.speichere_json(
        "31_glastypen",
        {
            "je_glastyp": tab.reset_index().to_dict(orient="records"),
            "billets_mit_kennung": int(kennung.sum()),
            "median_je_typ_mit_kennung": tab.loc[tab.index.str.contains(SP["glastyp_kennung"]), "median"].to_dict(),
            "gesamt_median": float(ys.median()),
            "n": int(len(t)),
        },
    )


if __name__ == "__main__":
    main()
