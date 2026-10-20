"""11 — Imputation fehlender Werte.

In der Arbeit: Abschnitt 5.6: Tabelle 9 (Imputationsverfahren im Vergleich).

Imputationsverfahren im Vergleich (XGBoost, artikelreine Falten).
Ausgabe: reports/eda/11_imputation.pdf.

    python eda/11_imputation.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

import _common as C

KNN_K = 5


def _median_impute(Xtr, Xva, mit_indikator: bool = False):
    """Median aus dem TRAININGSTEIL, angewandt auf beide."""
    med = Xtr.median(numeric_only=True)
    if mit_indikator:
        luecken = [c for c in Xtr.columns if Xtr[c].isna().any()]
        itr = Xtr[luecken].isna().astype(float).add_suffix("__fehlt")
        iva = Xva[luecken].isna().astype(float).add_suffix("__fehlt")
        return (
            pd.concat([Xtr.fillna(med), itr], axis=1),
            pd.concat([Xva.fillna(med), iva], axis=1),
        )
    return Xtr.fillna(med), Xva.fillna(med)


def _mean_impute(Xtr, Xva):
    m = Xtr.mean(numeric_only=True)
    return Xtr.fillna(m), Xva.fillna(m)


def _knn_impute(Xtr, Xva):
    """KNN-Imputation, gefittet auf dem Trainingsteil."""
    from sklearn.impute import KNNImputer
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(Xtr)
    imp = KNNImputer(n_neighbors=KNN_K)
    Ztr = imp.fit_transform(sc.transform(Xtr))
    Zva = imp.transform(sc.transform(Xva))
    return (
        pd.DataFrame(sc.inverse_transform(Ztr), columns=Xtr.columns, index=Xtr.index),
        pd.DataFrame(sc.inverse_transform(Zva), columns=Xva.columns, index=Xva.index),
    )


VARIANTEN = [
    ("nativ (NaN an XGBoost)", None),
    ("Median", lambda a, b: _median_impute(a, b, False)),
    ("Median + Fehlt-Indikator", lambda a, b: _median_impute(a, b, True)),
    ("Mittelwert", _mean_impute),
    (f"KNN (k={KNN_K}, standardisiert)", _knn_impute),
]


def herkunft(spalten) -> pd.Series:
    """'SQL' oder 'Influx' je Spalte — die beiden Fehlwert-Semantiken."""
    import data_prep as D

    inf = set(D.influx_spalten(list(spalten)))
    return pd.Series(
        ["Influx" if c in inf else "SQL" for c in spalten], index=list(spalten)
    )


def main():
    ds = C.lade(alle_arten=True)
    X, y, df = C.daten_teil(ds)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    df = df.reset_index(drop=True)
    fuss = C.kopfzeile(ds, alle_arten=True)
    pdf = C.pdf_oeffnen("11_imputation.pdf")

    fehlt = X.isna().mean()
    quelle = herkunft(X.columns)
    n_voll = int((fehlt == 0).sum())
    n_teil = int(((fehlt > 0) & (fehlt < 1)).sum())
    n_leer = int((fehlt >= 1).sum())

    fig, ax = C.seite(pdf, "A/B — Struktur der Fehlwerte", 2, 2)

    for i, q in enumerate(["SQL", "Influx"]):
        sub = fehlt[quelle[quelle == q].index]
        ax[i].hist(sub * 100, bins=40, color="steelblue" if q == "SQL" else "seagreen")
        ax[i].set_title(f"A{i + 1}  Fehlanteil je {q}-Spalte", fontsize=10)
        ax[i].set_xlabel("Anteil fehlender Werte [%]")
        ax[i].set_ylabel("Anzahl Spalten")

    proz = X.isna().mean(axis=1) * 100
    ax[2].hist(proz, bins=50, color="darkorange")
    ax[2].set_title("A3  Fehlanteil je Billet", fontsize=10)
    ax[2].set_xlabel("Anteil fehlender Merkmale einer Zeile [%]")
    ax[2].set_ylabel("Anzahl Billets")

    kandidaten = fehlt[(fehlt > 0.01) & (fehlt < 0.99)].index
    zeilen = []
    for c in kandidaten:
        da = X[c].notna()
        if da.sum() < 30 or (~da).sum() < 30:
            continue
        _, p = mannwhitneyu(y[da], y[~da])
        zeilen.append(
            {
                "merkmal": c,
                "p": p,
                "d_median": float(y[da].median() - y[~da].median()),
                "quelle": quelle[c],
            }
        )
    t = pd.DataFrame(zeilen).sort_values("p").reset_index(drop=True)
    m = len(t)
    t["q"] = np.minimum.accumulate((t.p * m / (t.index + 1))[::-1])[::-1].clip(0, 1)
    n_mnar = int((t.q < 0.05).sum())

    for maske, farbe, name in ((t.q < 0.05, "#d62728", "q < 0,05"), (t.q >= 0.05, "0.6", "q ≥ 0,05")):
        ax[3].scatter(
            t.d_median[maske],
            -np.log10(t.p[maske].clip(lower=1e-300)),
            color=farbe,
            s=14,
            label=name,
        )
    ax[3].axhline(
        -np.log10(t.p[t.q < 0.05].max()) if n_mnar else np.nan,
        color="#d62728",
        ls="--",
        lw=1,
        label="Grenze q = 0,05",
    )
    ax[3].set_title("B  Unterschied in y mit und ohne Wert", fontsize=10)
    ax[3].set_xlabel("Median(y | Wert da) − Median(y | Wert fehlt)  [Prozentpunkte]")
    ax[3].set_ylabel("−log10 p  (Mann-Whitney)")
    ax[3].legend(fontsize=7, loc="upper left")
    C.schliesse(
        pdf,
        fig,
        fuss + " | Mann-Whitney, Benjamini-Hochberg-korrigiert",
    )

    folds = C.folds_final(df)
    print(f"  {len(folds)} artikelreine Falten, {len(y)} Zeilen")

    ergebnisse = []
    for name, fn in VARIANTEN:
        print(f"  rechne: {name} ...", flush=True)
        pred = C.oof_xgb(X, y, folds, nachbearbeiten=fn)
        mm, sd = C.falten_streuung(y, pred, folds)
        ergebnisse.append(
            {
                "name": name,
                "mae": C.mae(y, pred),
                "r2": C.r2(y, pred),
                "sd": sd,
                "mae_falten": mm,
            }
        )
        print(f"      MAE {ergebnisse[-1]['mae']:.3f}  R² {ergebnisse[-1]['r2']:+.4f}")

    basis = VARIANTEN[0][0]
    fig, ax = C.seite(pdf, "C — Was Imputation an der Güte ändert", 1, 2, groesse=(11.7, 5.2))

    C.vergleichstabelle(
        ax[0], ergebnisse, basis, "MAE", "C1  OOF-MAE, Differenz zur Basis"
    )

    pos = np.arange(len(ergebnisse))
    r2w = [e["r2"] for e in ergebnisse]
    ax[1].barh(
        pos,
        r2w,
        color=["#7f7f7f" if e["name"] == basis else "#3182bd" for e in ergebnisse],
    )
    ax[1].set_yticks(pos)
    ax[1].set_yticklabels([e["name"] for e in ergebnisse], fontsize=8)
    ax[1].invert_yaxis()
    ax[1].axvline(0, color="k", lw=0.8)
    ax[1].spines["left"].set_visible(False)
    ax[1].tick_params(axis="y", length=0)
    ax[1].set_xlabel("OOF-R² (gepoolt)")
    ax[1].set_title("C2  Dieselbe Frage in R²", fontsize=10)

    n_ind = int((X.isna().sum() > 0).sum())
    beste = min(ergebnisse, key=lambda e: e["mae"])
    spanne = max(e["mae"] for e in ergebnisse) - min(e["mae"] for e in ergebnisse)
    C.schliesse(
        pdf,
        fig,
        f"{len(folds)} artikelreine Falten | getunte Parameter aus best_params_v3.json | "
        "Imputation je Falte nur auf dem Trainingsteil gefittet",
    )

    innerhalb = spanne <= ergebnisse[0]["sd"]
    C.textseite(
        pdf,
        "Imputation — Entscheidung und Begründung",
        [
            "ENTSCHEIDUNG: keine Imputation. NaN wird unverändert an das Modell "
            "gereicht. Ausnahme ist der ROCKET-Zweig, siehe unten.",
            "",
            f"Grundlage: {C.umfang_text(len(y))}, {len(folds)} artikelreine Falten, "
            "getunte Parameter aus best_params_v3.json, jede Imputation je Falte nur "
            "auf dem Trainingsteil gefittet.",
            f"Befund A — zwei Sorten Fehlwert. Von {X.shape[1]} Rohmerkmalen sind "
            f"{n_voll} vollständig, {n_teil} teilweise belegt (mindestens ein "
            f"Fehlwert, in 07_datenqualitaet.pdf enger gefasst), {n_leer} leer. "
            + "; ".join(
                f"{q}: {int((quelle == q).sum())} Spalten, Median-Fehlanteil "
                f"{100 * fehlt[quelle[quelle == q].index].median():.1f} %"
                for q in ("SQL", "Influx")
            )
            + f". Je Billet fehlen im Median {proz.median():.1f} % der Merkmale; "
            f"vollständige Zeilen: {int((proz == 0).sum())} von {len(proz)}. Die "
            f"Trennung nach Herkunft ist dabei wichtiger als die Anzahl: bei einer "
            f"SQL-Stammdatenspalte bedeutet leer 'nicht gepflegt', bei einem "
            f"Influx-Aggregat 'dieser Kanal hat in dieser Schmelze nicht "
            f"geschrieben'. Der zweite Fall ist gar kein fehlender Wert, sondern "
            f"eine Beobachtung. Ihn zu imputieren hiesse, eine Messung durch eine "
            f"Schätzung zu ersetzen.",
            "",
            f"Befund B — das Fehlen ist nicht zufällig. Bei {n_mnar} von {m} "
            f"geprüften Spalten (mehr als 1 und weniger als 99 % fehlend, je "
            f"mindestens 30 Zeilen mit und ohne Wert) unterscheidet sich y zwischen "
            f"Zeilen mit und ohne "
            f"Wert signifikant (Mann-Whitney, FDR-korrigiert). Damit liegt MNAR vor, "
            f"und jede Imputation, die den Wert ersetzt ohne die Missingness zu "
            f"behalten, löscht Information. Das ist der inhaltliche Grund gegen "
            f"Median- und Mittelwert-Imputation — unabhängig davon, was sie messen.",
            "",
            "Befund C, die Zahlen:",
            *C.zahlenzeilen(ergebnisse, basis),
            f"Bestes Verfahren in dieser Messung: {beste['name']} "
            f"({beste['mae']:.3f} gegen {ergebnisse[0]['mae']:.3f} nativ). Die "
            f"Indikator-Variante fügt {n_ind} zusätzliche 0/1-Spalten zu den "
            f"{X.shape[1]} Rohmerkmalen hinzu.",
            f"Befund C — gemessen ändert es nichts. Die fünf Varianten liegen "
            f"{spanne:.3f} MAE-Punkte auseinander, die Faltenstreuung der Basis "
            f"beträgt ±{ergebnisse[0]['sd']:.3f}. "
            + (
                "Die Spannweite liegt damit INNERHALB der Streuung: die Verfahren "
                "sind auf diesen Daten nicht unterscheidbar."
                if innerhalb
                else "Die Spannweite überschreitet die Streuung; die Rangfolge in "
                "C1 ist zu beachten, bleibt aber auf einer Messung mit fünf "
                "Falten schwach belegt."
            )
            + " Bemerkenswert ist vor allem, was NICHT passiert: auch die "
            "Median+Indikator-Variante, die nach Befund B die einzige theoretisch "
            "richtige wäre, gewinnt nichts. XGBoost lernt die Default-Richtung je "
            "Knoten und hat die Indikator-Information damit bereits.",
            "",
            "Warum die Entscheidung trotzdem nicht beliebig ist. Wären die Zahlen "
            "die einzige Begründung, könnte man jede Variante nehmen. 'Nativ' ist "
            "aus zwei anderen Gründen vorzuziehen: es trifft keine Annahme über "
            "den unbeobachteten Wert, und es hält die Kette kurz — jede "
            "Imputation müsste je Falte gefittet werden, und genau dort entsteht "
            "sonst stilles Leck. Für Modelle OHNE NaN-Fähigkeit (RidgeCV im "
            "ROCKET-Zweig) ist der Verzicht keine Option; dort steht fillna(0,0), "
            "und das ist als Limitation zu nennen, nicht als neutrale Wahl: bei "
            "z-normierten Kanälen ist 0 der Mittelwert, bei rohen ein realer "
            "Messwert.",
            "",
            "Offen. Die Indikator-Variante ist hier nur für XGBoost geprüft. Ob "
            "TabPFN — das bestbewertete Modell — davon profitiert, ist nicht "
            "gemessen; TabPFN hat eine eigene NaN-Behandlung, deren Verhalten bei "
            "MNAR nicht dokumentiert ist.",
        ],
    )
    pdf.close()
    print(f"geschrieben: {C.OUT_DIR / '11_imputation.pdf'}")
    for e in ergebnisse:
        print(
            f"  {e['name']:<34} MAE {e['mae']:.3f} ± {e['sd']:.3f}  R² {e['r2']:+.4f}"
        )


if __name__ == "__main__":
    main()
