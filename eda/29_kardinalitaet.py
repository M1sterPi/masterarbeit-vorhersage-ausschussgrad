"""29 — Kodierung und Kardinalitätsschwelle.

In der Arbeit: Abschnitt 5.5: Tabelle 8 (Kardinalitätsschwelle), Textspalten
und die Lücke zwischen beschreibenden und kennzeichnenden Spalten.

Textspalten, Wirkung der Schwelle (XGBoost, artikelreine Falten) und im
Auftrag konstante Merkmale.
Ausgabe: reports/eda/29_kardinalitaet.csv, runs/eda/29_kardinalitaet.json.

    python eda/29_kardinalitaet.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import _common as C
from data_prep import SP  # noqa: E402

D = C.D
SCHWELLEN = [20, 40, 90, 160, 300, 3000]
BEZEICHNUNG = "Bezeichnung"


def textspalten(X):
    obj = [c for c in X.columns if X[c].dtype == object]
    de = [c for c in obj if D._to_num_de(X[c]).notna().sum() >= 0.7 * X[c].notna().sum()]
    return obj, de, [c for c in obj if c not in de]


def main():
    ds = C.lade(alle_arten=True)
    X, ys, df = C.daten_teil(ds)
    art = df[SP["artikel"]].astype(str).to_numpy()
    auf = df[SP["auftrag"]].astype(str).to_numpy()
    erg = {}

    obj, de, kat = textspalten(X)
    erg["A"] = {"merkmale": X.shape[1], "text": len(obj), "deutsches_zahlenformat": de,
                "kategorial": len(kat)}
    print(f"A  {X.shape[1]} Merkmale, davon {len(obj)} Text: {len(de)} Zahlen im deutschen "
          f"Format ({', '.join(de)}), {len(kat)} kategorial")

    nu = X[kat].nunique().sort_values()
    grenze = D.MAX_CARDINALITY
    zugelassen, aus = nu[nu <= grenze], nu[nu > grenze]
    k = X[BEZEICHNUNG].astype(str)
    art_je_wert = pd.Series(art).groupby(k.to_numpy()).nunique()
    eindeutig = float((k.map(art_je_wert) == 1)[X[BEZEICHNUNG].notna()].mean())
    erg["B"] = {
        "auspraegungen": nu.to_dict(),
        "groesste_zugelassene": [zugelassen.index[-1], int(zugelassen.iloc[-1])],
        "kleinste_ausgeschlossene": [aus.index[0], int(aus.iloc[0])],
        "n_zugelassen": int(len(zugelassen)),
        "n_ausgeschlossen": int(len(aus)),
        "ausgeschlossen": list(aus.index),
        "glastyp": int(nu["Glastyp"]),
        "bezeichnung": int(nu[BEZEICHNUNG]),
        "bezeichnung_anteil_nur_ein_artikel": eindeutig,
    }
    print(f"B  Lücke: größte beschreibende Spalte {zugelassen.index[-1]} ({zugelassen.iloc[-1]}), "
          f"kleinste kennzeichnende {aus.index[0]} ({aus.iloc[0]}); bei Schwelle {grenze}: "
          f"{len(zugelassen)} zugelassen, {len(aus)} ausgeschlossen")
    print(f"   Glastyp {nu['Glastyp']} Ausprägungen; {BEZEICHNUNG} {nu[BEZEICHNUNG]}, davon tragen "
          f"{100 * eindeutig:.0f} % der Billets eine Bezeichnung aus nur einem Artikel")
    print("   ausgeschlossen:", ", ".join(f"{c} ({n})" for c, n in aus.items()))

    folds = C.folds_final(df)
    max_zu, min_aus, bez = [], [], []
    for tr, _ in folds:
        n = X.iloc[tr][kat].nunique()
        max_zu.append(int(n[n <= grenze].max()))
        min_aus.append(int(n[n > grenze].min()))
        bez.append(int(n[BEZEICHNUNG]))
    erg["C"] = {"max_zugelassen_je_falte": max_zu, "min_ausgeschlossen_je_falte": min_aus,
                "bezeichnung_je_falte": bez}
    print(f"C  je Falte (Train): zugelassen höchstens {max(max_zu)}, ausgeschlossen mindestens "
          f"{min(min_aus)}; {BEZEICHNUNG} {min(bez)}–{max(bez)} Ausprägungen")

    zeilen = []
    alt = D.MAX_CARDINALITY
    try:
        for s in SCHWELLEN:
            D.MAX_CARDINALITY = s
            p = C.oof_xgb(X, ys, folds)
            zeilen.append({"schwelle": s, "oof_mae": C.mae(ys, p), "oof_r2": C.r2(ys, p)})
            print(f"D  Schwelle {s:5d}: OOF-MAE {zeilen[-1]['oof_mae']:.3f}  "
                  f"OOF-R² {zeilen[-1]['oof_r2']:+.3f}", flush=True)
    finally:
        D.MAX_CARDINALITY = alt
    med = np.full(len(ys), np.nan)
    for tr, va in folds:
        med[va] = ys.iloc[tr].median()
    med_falten = float(np.mean([C.mae(ys.iloc[va], med[va]) for _, va in folds]))
    erg["D"] = {"tabelle": zeilen, "median_vorhersage_mae_gepoolt": C.mae(ys, med),
                "median_vorhersage_mae_faltenmittel": med_falten}
    print(f"   Median-Vorhersage auf denselben Falten: MAE {med_falten:.2f} (Faltenmittel), "
          f"{C.mae(ys, med):.2f} (gepoolt)")
    t = pd.DataFrame(zeilen)
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    t.to_csv(C.OUT_DIR / "29_kardinalitaet.csv", index=False, encoding="utf-8")

    Xp, _, _ = D.process_features(X, X)
    g = Xp.groupby(auf).nunique(dropna=False)
    konst = [c for c in Xp.columns if (g[c] <= 1).all() and Xp[c].nunique(dropna=False) > 1]
    kombi = Xp[konst].astype(str).groupby(auf).first().drop_duplicates()
    erg["E"] = {"konstant_im_auftrag": konst, "n_konstant": len(konst),
                "n_auftraege": int(len(np.unique(auf))), "n_kombinationen": int(len(kombi))}
    print(f"E  {len(konst)} Merkmale konstant im Auftrag; {len(np.unique(auf))} Aufträge bilden "
          f"{len(kombi)} verschiedene Kombinationen")

    C.speichere_json("29_kardinalitaet", erg)


if __name__ == "__main__":
    main()
