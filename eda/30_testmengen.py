"""30 — Testmengen.

In der Arbeit: Abschnitt 6.1: Tabelle 10 (Testmengen und Überlappung mit dem
Training).

Größe der Testmengen je Ebene und Überlappung mit dem Training.
Ausgabe: reports/eda/30_testmengen.csv, runs/eda/30_testmengen.json.

    python eda/30_testmengen.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import _common as C
from data_prep import SP  # noqa: E402

EBENEN = {"billet": "billetweise", "auftrag": "auftragsrein", "artikel": "artikelrein"}


def main():
    ds = C.lade(alle_arten=True)
    df = ds.df.reset_index(drop=True)
    bid = ds.batch_ids.astype(str).reset_index(drop=True)
    auf = df[SP["auftrag"]].astype(str)
    art = df[SP["artikel"]].astype(str)

    zeilen = []
    for ebene, name in EBENEN.items():
        pfad = C.D.TEST_IDS_PATHS[ebene]
        test_ids = set(json.loads(pfad.read_text(encoding="utf-8")))
        test = bid.isin(test_ids).to_numpy()
        auf_train, art_train = set(auf[~test]), set(art[~test])
        zeilen.append(
            {
                "Ebene": name,
                "Datei": pfad.name,
                "Training": int((~test).sum()),
                "Test": int(test.sum()),
                "Test-IDs nicht im Bestand": len(test_ids - set(bid)),
                "Testbillets mit bekanntem Auftrag": float(np.mean(auf[test].isin(auf_train))),
                "Testbillets mit bekanntem Artikel": float(np.mean(art[test].isin(art_train))),
            }
        )
    t = pd.DataFrame(zeilen)
    with pd.option_context("display.width", 200):
        print(
            t.assign(
                **{
                    c: (100 * t[c]).round(1).astype(str) + " %"
                    for c in ("Testbillets mit bekanntem Auftrag", "Testbillets mit bekanntem Artikel")
                }
            ).to_string(index=False)
        )
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    t.to_csv(C.OUT_DIR / "30_testmengen.csv", index=False, encoding="utf-8")

    n_je_auf = auf.map(auf.value_counts())
    geschwister = float((n_je_auf >= 2).mean())
    print(f"\nBillets mit mindestens einem Geschwisterbillet im Auftrag: {100 * geschwister:.1f} %")
    C.speichere_json(
        "30_testmengen",
        {"ebenen": zeilen, "anteil_mit_geschwister": geschwister, "n": int(len(df))},
    )


if __name__ == "__main__":
    main()
