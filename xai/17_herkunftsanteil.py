"""17 — Anteil der Datenquellen am SHAP.

In der Arbeit: Abschnitt 7.2: Anteil von Stammdaten und Prozessaufzeichnung am
SHAP-Betrag.

Anteil von Stammdaten und Prozessaufzeichnung am mittleren |SHAP| aus xai/07,
Zuordnung über data_prep.influx_spalten.

    python xai/17_herkunftsanteil.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

D = C.D
EBENEN = ("artikel", "auftrag", "billet")


def anteile(t: pd.DataFrame, spalte: str, influx: set) -> dict:
    w = t[spalte].astype(float)
    gesamt = w.sum()
    prozess = w[[m in influx for m in w.index]].sum()
    abgel = w[[m in D._ABGELEITET for m in w.index]].sum()
    stamm = gesamt - prozess - abgel
    return {
        "prozess": prozess / gesamt,
        "stammdaten": stamm / gesamt,
        "abgeleitet": abgel / gesamt,
        "stammdaten_mit_abgeleitet": (stamm + abgel) / gesamt,
        "alte_zuordnung": (w.groupby(t.herkunft).sum() / gesamt).to_dict(),
    }


def main():
    ds = D.load_split(verbose=False)
    influx = set(D.influx_spalten(ds.X_raw.columns))
    erg = {}
    for e in EBENEN:
        p = C.TAB_DIR / f"07_kreuzvalidierung_xgboost_{e}.csv"
        if not p.exists():
            print(f"{p.name} fehlt — erst TEST_LEVEL={e} python xai/07_kreuzvalidierung.py")
            continue
        t = pd.read_csv(p).set_index("Merkmal")
        erg[e] = {
            "n_merkmale": int(len(t)),
            "n_prozess": int(sum(m in influx for m in t.index)),
            "n_abgeleitet": int(sum(m in D._ABGELEITET for m in t.index)),
            "falten": anteile(t, "Mittel", influx),
            "test_set": anteile(t, "Test-Set", influx),
        }
        a = erg[e]["falten"]
        print(
            f"{e:8s} Faltenmittel: Stammdaten {100 * a['stammdaten']:.1f} % + abgeleitet "
            f"{100 * a['abgeleitet']:.1f} % = {100 * a['stammdaten_mit_abgeleitet']:.1f} %, "
            f"Prozessaufzeichnung {100 * a['prozess']:.1f} %  "
            f"({erg[e]['n_merkmale']} Merkmale, davon {erg[e]['n_prozess']} Prozess)"
        )
    ziel = C.ERG_DIR / "17_herkunftsanteil.json"
    ziel.parent.mkdir(parents=True, exist_ok=True)
    ziel.write_text(json.dumps(erg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"geschrieben: {ziel.relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
