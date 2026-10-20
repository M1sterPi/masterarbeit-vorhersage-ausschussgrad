"""25 — Lichtbogenende im Exportfenster.

In der Arbeit: Abschnitt 5.1: Lage des Lichtbogenendes im Fenster und Billets,
deren Lichtbogen über die Grenze hinaus brennt.

Auswertung von data/_arc_end_scan.csv (data_pipeline/arc_end_scan.py).
Ausgabe: runs/eda/25_lichtbogenende.json.

    python eda/25_lichtbogenende.py
"""

from __future__ import annotations

import pandas as pd

import _common as C
from data_prep import SP  # noqa: E402

SCAN = C.ROOT / "data" / "_arc_end_scan.csv"
GRENZE_H = float(C.D.MAX_BILLET_DURATION_H)
KANAELE = {SP["lichtbogen_leistung"]: SP["lichtbogen_leistung"] + " > 1", SP["lichtbogen_an"]: SP["lichtbogen_an"] + " > 0,5"}


def kennzahlen(s: pd.DataFrame, kanal: str) -> dict:
    ende = s[f"b_{kanal}_end_h"]
    start = s[f"b_{kanal}_start_h"]
    mit = ende.notna()
    ueber = mit & (ende > GRENZE_H)
    ohne = ~mit
    return {
        "n_billets": int(len(s)),
        "n_mit_lichtbogen": int(mit.sum()),
        "ende_median_h": float(ende[mit].median()),
        "ende_p99_h": float(ende[mit].quantile(0.99)),
        "anteil_ende_bis_grenze": float((ende[mit] <= GRENZE_H).mean()),
        "n_ueber_grenze": int(ueber.sum()),
        "anteil_ueber_grenze": float(ueber.sum() / mit.sum()),
        "n_ueber_grenze_start_nach_grenze": int((ueber & (start > GRENZE_H)).sum()),
        "n_ohne_lichtbogen": int(ohne.sum()),
        "fenster_ohne_lichtbogen_median_min": float(s.loc[ohne, "a_span_h"].median() * 60)
        if ohne.any()
        else None,
    }


def main():
    s = pd.read_csv(SCAN, dtype={"billet": str})
    ds = C.D.load_split(verbose=False)
    im_modell = set(
        ds.df[SP["billetnummer"]].astype(str).str.replace(r"\.0$", "", regex=True)
    )
    bestaende = {"alle": s, "modell": s[s.billet.isin(im_modell)]}

    ergebnis = {"grenze_h": GRENZE_H}
    print(f"Lichtbogenende im ungekappten Influx-Fenster (Grenze {GRENZE_H:g} h)\n")
    for kanal, text in KANAELE.items():
        for name, t in bestaende.items():
            k = kennzahlen(t, kanal)
            ergebnis[f"{kanal}_{name}"] = k
            print(
                f"{text:13s} {name:6s} n={k['n_billets']:4d} mit Lichtbogen {k['n_mit_lichtbogen']:4d} | "
                f"Ende Median {k['ende_median_h']:.2f} h, p99 {k['ende_p99_h']:.2f} h | "
                f"bis {GRENZE_H:g} h {100 * k['anteil_ende_bis_grenze']:.1f} % | "
                f"darüber {k['n_ueber_grenze']} ({100 * k['anteil_ueber_grenze']:.1f} %, "
                f"davon Start nach {GRENZE_H:g} h: {k['n_ueber_grenze_start_nach_grenze']}) | "
                f"ohne Lichtbogen {k['n_ohne_lichtbogen']}"
                + (
                    f" (Fenster Median {k['fenster_ohne_lichtbogen_median_min']:.0f} min)"
                    if k["n_ohne_lichtbogen"]
                    else ""
                )
            )
    a = s[f"a_{SP['lichtbogen_leistung']}_end_h"].dropna()
    ergebnis["kontrolle_export_max_h"] = float(a.max())
    print(f"\nKontrolle Export (gekappt): spätestes Lichtbogenende {a.max():.2f} h")
    C.speichere_json("25_lichtbogenende", ergebnis)


if __name__ == "__main__":
    main()
