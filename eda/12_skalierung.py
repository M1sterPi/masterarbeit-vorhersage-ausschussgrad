"""12 — Skalierung und Zielgröße.

In der Arbeit: Abschnitt 5.6 (unplausible Werte im tabellarischen Zweig:
betroffene Zellen und Billets) und 5.7 (Spannweite der Wertebereiche).

Skalierung der Merkmale und Transformation von y im Vergleich.
Ausgabe: reports/eda/12_skalierung.pdf.

    python eda/12_skalierung.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import _common as C


def _sk(cls, **kw):
    def fn(Xtr, Xva, _cls=cls, _kw=kw):
        sc = _cls(**_kw).fit(Xtr)
        return (
            pd.DataFrame(sc.transform(Xtr), columns=Xtr.columns, index=Xtr.index),
            pd.DataFrame(sc.transform(Xva), columns=Xva.columns, index=Xva.index),
        )

    return fn


def _signed_log(Xtr, Xva):
    """sign(x)·log1p(|x|) — monoton, aber stark nichtlinear."""

    def f(X):
        Z = X.astype(float)
        return np.sign(Z) * np.log1p(np.abs(Z))

    return f(Xtr), f(Xva)


def main():
    from sklearn.preprocessing import (
        MinMaxScaler,
        QuantileTransformer,
        StandardScaler,
    )

    varianten = [
        ("roh (Stand der Arbeit)", None),
        ("StandardScaler", _sk(StandardScaler)),
        ("MinMaxScaler", _sk(MinMaxScaler)),
        (
            "QuantileTransformer (normal)",
            _sk(QuantileTransformer, output_distribution="normal", random_state=42),
        ),
        ("sign(x)·log1p(|x|)", _signed_log),
    ]

    ds = C.lade(alle_arten=True)
    X, y, df = C.daten_teil(ds)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    df = df.reset_index(drop=True)
    fuss = C.kopfzeile(ds, alle_arten=True)
    pdf = C.pdf_oeffnen("12_skalierung.pdf")
    folds = C.folds_final(df)

    import data_prep as D

    Xn, _, _ = D.process_features(X, X)
    Xn = Xn.astype(float)
    spanne = (Xn.max() - Xn.min()).replace(0, np.nan).dropna()
    sd = Xn.std().replace(0, np.nan).dropna()

    fig, ax = C.seite(
        pdf, "A/B — Merkmalsskalen und ihre Wirkung auf XGBoost", 1, 3, groesse=(11.7, 5.2)
    )

    ax[0].hist(np.log10(spanne), bins=50, color="steelblue")
    ax[0].set_title("A1  Wertebereich je Merkmal", fontsize=10)
    ax[0].set_xlabel("log10 (max − min)")
    ax[0].set_ylabel("Anzahl Merkmale")

    gross = sd.sort_values(ascending=False).head(12)
    klein = sd.sort_values().head(12)
    zeig = pd.concat([gross, klein])
    pos = np.arange(len(zeig))
    ax[1].barh(
        pos,
        np.log10(zeig.to_numpy()),
        color=["#d62728"] * len(gross) + ["#3182bd"] * len(klein),
    )
    ax[1].set_yticks(pos)
    ax[1].set_yticklabels([s[:34] for s in zeig.index], fontsize=6.5)
    ax[1].invert_yaxis()
    ax[1].set_xlabel("log10 Standardabweichung")
    ax[1].set_title("A2  Grösste und kleinste Skalen", fontsize=10)
    ax[1].axvline(0, color="k", lw=0.8)
    ax[1].spines["left"].set_visible(False)
    ax[1].tick_params(axis="y", length=0)

    erg = []
    for name, fn in varianten:
        print(f"  XGBoost / {name} ...", flush=True)
        pred = C.oof_xgb(X, y, folds, nachbearbeiten=fn)
        mm, s = C.falten_streuung(y, pred, folds)
        erg.append({"name": name, "mae": C.mae(y, pred), "r2": C.r2(y, pred), "sd": s})
        print(f"     MAE {erg[-1]['mae']:.4f}  R² {erg[-1]['r2']:+.5f}")

    C.vergleichstabelle(
        ax[2], erg, varianten[0][0], "MAE", "B1  XGBoost: MAE je Skalierung"
    )

    sp = max(e["mae"] for e in erg) - min(e["mae"] for e in erg)
    C.schliesse(pdf, fig, fuss + " | 5 artikelreine Falten, getunte Parameter")

    from sklearn.linear_model import RidgeCV

    ALPHAS = np.logspace(-1, 8, 19)

    def oof_ridge(skalieren: bool, robust: bool):
        """RidgeCV OOF."""
        pred = np.full(len(y), np.nan)
        alphas = []
        for tr, va in folds:
            Xtr, Xva, _ = D.process_features(X.iloc[tr], X.iloc[va])
            Xtr, Xva = Xtr.astype(float), Xva.astype(float)
            med = Xtr.median()
            Xtr, Xva = Xtr.fillna(med).fillna(0.0), Xva.fillna(med).fillna(0.0)
            if robust:
                behalten = Xtr.columns[Xtr.std() > 0]
                Xtr, Xva = Xtr[behalten], Xva[behalten]
                lo, hi = Xtr.quantile(0.001), Xtr.quantile(0.999)
                Xtr = Xtr.clip(lo, hi, axis=1)
                Xva = Xva.clip(lo, hi, axis=1)
            A, B = Xtr.to_numpy(), Xva.to_numpy()
            if skalieren:
                sc = StandardScaler().fit(A)
                A, B = sc.transform(A), sc.transform(B)
            r = RidgeCV(alphas=ALPHAS).fit(A, y.iloc[tr])
            alphas.append(float(r.alpha_))
            pred[va] = r.predict(B)
        return pred, alphas

    ridge = []
    alpha_log = {}
    for robust in (False, True):
        for skal in (False, True):
            name = ("robust, " if robust else "naiv, ") + (
                "StandardScaler" if skal else "roh"
            )
            print(f"  RidgeCV / {name} ...", flush=True)
            pr, al = oof_ridge(skal, robust)
            alpha_log[name] = al
            ridge.append(
                {
                    "name": name,
                    "mae": C.mae(y, pr),
                    "r2": C.r2(y, pr),
                    "sd": C.falten_streuung(y, pr, folds)[1],
                    "robust": robust,
                }
            )
            print(f"     MAE {ridge[-1]['mae']:.3f}  R² {ridge[-1]['r2']:+.4f}")

    fig, ax = C.seite(
        pdf, "C — Wo Skalierung entscheidet: das lineare Modell", 1, 3, groesse=(11.7, 5.2)
    )

    naiv = [r for r in ridge if not r["robust"]]
    pos = np.arange(2)
    ax[0].barh(pos, [r["mae"] for r in naiv], color=["#d62728", "#8b0000"])
    ax[0].set_xscale("log")
    ax[0].set_yticks(pos)
    ax[0].set_yticklabels([r["name"] for r in naiv], fontsize=8)
    ax[0].invert_yaxis()
    ax[0].set_xlabel("OOF-MAE, logarithmisch")
    ax[0].set_title("C1  RidgeCV ohne Vorarbeit", fontsize=10)
    ax[0].axvline(erg[0]["mae"], color="k", ls="--", lw=1, label="XGBoost")
    ax[0].legend(fontsize=7, loc="lower right")

    rob = [r for r in ridge if r["robust"]]
    ax[1].barh(pos, [r["mae"] for r in rob], color=["#d62728", "#2ca02c"])
    ax[1].set_yticks(pos)
    ax[1].set_yticklabels([r["name"] for r in rob], fontsize=8)
    ax[1].invert_yaxis()
    ax[1].set_xlabel("OOF-MAE")
    ax[1].set_xlim(0, max(r["mae"] for r in rob) * 1.15)
    ax[1].set_title("C2  RidgeCV nach Entschärfung", fontsize=10)
    ax[1].axvline(erg[0]["mae"], color="k", ls="--", lw=1, label="XGBoost")
    ax[1].legend(fontsize=7, loc="lower right")

    for name, farbe in [
        ("robust, roh", "#d62728"),
        ("robust, StandardScaler", "#2ca02c"),
    ]:
        ax[2].scatter(
            np.arange(len(alpha_log[name])),
            alpha_log[name],
            label=name,
            color=farbe,
            s=45,
        )
    ax[2].set_yscale("log")
    ax[2].set_xlabel("Falte")
    ax[2].set_ylabel("von RidgeCV gewähltes alpha")
    ax[2].set_title("C3  Gewähltes alpha je Falte", fontsize=10)
    ax[2].set_xticks(np.arange(len(folds)))
    ax[2].legend(fontsize=7)
    ax[2].grid(alpha=0.3, which="both")

    XA = D.process_features(X, X)[0].astype(float)
    verdacht = []
    for c in XA.columns:
        s = XA[c].dropna()
        if len(s) < 50:
            continue
        q = s.abs().quantile(0.999)
        if q > 0 and s.abs().max() > 1000 * q:
            verdacht.append((c, q, s.abs().max(), int((s.abs() > 1000 * q).sum())))
    vt = pd.DataFrame(verdacht, columns=["spalte", "q999", "max", "zellen"])
    zeilen_mit = pd.Series(False, index=XA.index)
    for _, r in vt.iterrows():
        zeilen_mit |= XA[r.spalte].abs() > 1000 * r.q999
    n_zeilen_sentinel = int(zeilen_mit.sum())
    C.schliesse(
        pdf,
        fig,
        "RidgeCV, alphas 1e-1 .. 1e8, 5 artikelreine Falten | "
        "'robust' = trainingskonstante Spalten entfernt, auf Trainingsquantile "
        "0,1/99,9 % winsorisiert",
    )

    from scipy.stats import skew

    fig, ax = C.seite(pdf, "D — Transformation der Zielgrösse", 1, 3, groesse=(11.7, 5.2))

    ax[0].hist(y, bins=60, color="steelblue")
    ax[0].set_xlabel("y [%]")
    ax[0].set_ylabel("Anzahl Billets")
    ax[0].set_title("D1  y", fontsize=10)
    ax[0].axvline(float(y.median()), color="crimson", lw=1, label="Median")
    ax[0].axvline(float(y.mean()), color="darkorange", lw=1, label="Mittelwert")
    ax[0].legend(fontsize=7)

    ax[1].hist(np.log1p(y), bins=60, color="seagreen")
    ax[1].set_xlabel("log1p(y)")
    ax[1].set_ylabel("Anzahl Billets")
    ax[1].set_title("D2  log1p(y)", fontsize=10)

    ziele = [
        ("y roh", lambda v: v, lambda v: v),
        ("log1p(y)", np.log1p, np.expm1),
        ("sqrt(y)", np.sqrt, lambda v: np.clip(v, 0, None) ** 2),
    ]
    zerg = []
    for name, hin, zurueck in ziele:
        print(f"  Zieltransformation / {name} ...", flush=True)
        yt = pd.Series(hin(y.to_numpy()))
        pred_t = C.oof_xgb(X, yt, folds)
        pred = zurueck(
            pred_t
        )
        zerg.append(
            {
                "name": name,
                "mae": C.mae(y, pred),
                "r2": C.r2(y, pred),
                "sd": C.falten_streuung(y, pred, folds)[1],
            }
        )
        print(f"     MAE {zerg[-1]['mae']:.4f}  R² {zerg[-1]['r2']:+.5f}")

    C.vergleichstabelle(
        ax[2],
        zerg,
        "y roh",
        "MAE",
        "D3  Zieltransformation",
    )

    C.schliesse(
        pdf,
        fig,
        "Bewertung immer auf der Originalskala — eine MAE in log-Einheiten "
        "wäre mit den berichteten Zahlen nicht vergleichbar",
    )

    r_roh = next(r for r in ridge if r["name"] == "robust, roh")
    r_std = next(r for r in ridge if r["name"] == "robust, StandardScaler")
    n_roh = next(r for r in ridge if r["name"] == "naiv, roh")
    n_std = next(r for r in ridge if r["name"] == "naiv, StandardScaler")
    ridge_d = r_roh["mae"] - r_std["mae"]
    besser_z = min(zerg, key=lambda e: e["mae"])
    a_roh, a_std = alpha_log["robust, roh"], alpha_log["robust, StandardScaler"]
    innerhalb_b = sp <= erg[0]["sd"]
    z_d = zerg[0]["mae"] - besser_z["mae"]
    C.textseite(
        pdf,
        "Normalisierung und Skalierung — Entscheidung und Begründung",
        [
            "ENTSCHEIDUNG: keine Skalierung der Merkmale für die baumbasierten "
            "Modelle (XGBoost, CatBoost) und für TabPFN. StandardScaler nur dort, "
            "wo ein lineares Modell dahintersteht — im ROCKET-Zweig. Die Zielgrösse "
            "bleibt untransformiert.",
            "",
            f"Grundlage: {C.umfang_text(len(y))}, {len(folds)} artikelreine Falten, "
            "getunte Parameter aus best_params_v3.json.",
            f"Befund A — die Frage ist nicht akademisch. Die Wertebereiche der "
            f"{len(spanne)} nicht-konstanten Merkmale reichen von "
            f"{spanne.min():.3g} bis {spanne.max():.3g}, liegen also um den Faktor "
            f"{spanne.max() / spanne.min():.3g} auseinander. Wer ein "
            f"distanzbasiertes oder ein regularisiertes Verfahren einsetzt, MUSS "
            f"hier skalieren.",
            "",
            "Befund B, die Zahlen (XGBoost):",
            *C.zahlenzeilen(erg, varianten[0][0]),
            f"Befund B — für Bäume ist es folgenlos. Vier Skalierungen, darunter "
            f"mit sign(x)·log1p(|x|) eine stark nichtlineare, liegen "
            f"{sp:.4f} MAE-Punkte auseinander; die Faltenstreuung beträgt "
            f"±{erg[0]['sd']:.3f}. Erwartet wäre exakt dieselbe Zahl; die kleinen "
            f"Abweichungen sind Implementierung, nicht Statistik — XGBoost mit "
            f"tree_method='hist' legt die Splitkandidaten auf ein Histogrammraster, "
            f"dessen Kanten von der Skala abhängen. Der Grund ist strukturell: "
            f"ein Entscheidungsbaum vergleicht ausschliesslich 'x <= Schwelle', und "
            f"jede streng monotone Abbildung je Merkmal lässt die erzeugte "
            f"Zeilenmenge unverändert. Standardisieren wäre hier kein Fehler, aber "
            f"ein Arbeitsschritt ohne Wirkung — und jeder Arbeitsschritt, der je "
            f"Falte gefittet werden muss, ist eine Gelegenheit für ein Leck.",
            "",
            f"Befund C — bei RidgeCV kippt die Antwort, aber erst nach einer "
            f"Vorarbeit. Entschärft man die Daten (trainingskonstante Spalten "
            f"entfernt, auf die Trainingsquantile 0,1/99,9 % winsorisiert), steht "
            f"roh bei MAE {r_roh['mae']:.3f} / R² {r_roh['r2']:+.4f} und "
            f"standardisiert bei {r_std['mae']:.3f} / {r_std['r2']:+.4f} — "
            f"{abs(ridge_d):.3f} MAE-Punkte und {abs(r_std['r2'] - r_roh['r2']):.3f} "
            f"R² Unterschied, weit ausserhalb jeder Faltenstreuung. Der Mechanismus "
            f"steht in C3: unskaliert wählt RidgeCV alpha zwischen "
            f"{min(a_roh):.0e} und {max(a_roh):.0e}, skaliert zwischen "
            f"{min(a_std):.0e} und {max(a_std):.0e}. Die L2-Strafe wirkt auf die Koeffizienten, und ein Merkmal "
            f"in großen Einheiten braucht einen winzigen Koeffizienten für "
            f"dieselbe Wirkung — es entgeht der Strafe fast vollständig, während "
            f"ein 0/1-Merkmal sie voll trägt. Ohne Skalierung entscheidet die Wahl "
            f"der Messeinheit darüber, welches Merkmal das Modell benutzen darf. "
            f"Genau deshalb steht in rocket_regression.py ein StandardScaler "
            f"zwischen Transformer und RidgeCV, je Falte neu gefittet.",
            "",
            f"Befund C1 — der wichtigere Teil, und er war nicht gesucht. OHNE die "
            f"Entschärfung ist das lineare Modell in BEIDEN Varianten unbrauchbar: "
            f"roh MAE {C.zahl(n_roh['mae'])}, standardisiert "
            f"MAE {C.zahl(n_std['mae'])} (XGBoost: {erg[0]['mae']:.3f}). "
            + (
                "Skalierung macht es also noch SCHLECHTER, nicht besser. "
                if n_std["mae"] > n_roh["mae"]
                else "Skalierung hilft hier nicht über die Unbrauchbarkeit hinweg. "
            )
            + f"Ursache sind Analysendatensätze, in denen {len(vt)} Spalten "
            f"({int(vt.zellen.sum())} Zellen in {n_zeilen_sentinel} Billets) statt "
            f"Messwerten einen Sentinel tragen, grösster Betrag "
            f"{C.zahl(vt['max'].max(), 2)} — um mehr als den Faktor 1000 über dem "
            f"99,9-%-Quantil der jeweiligen Spalte:",
            *[
                f"  {r.spalte}: 99,9-%-Quantil {r.q999:.3f}, Maximum {C.zahl(r['max'], 2)}, "
                f"{r.zellen} Zelle(n)"
                for _, r in vt.sort_values("max", ascending=False).iterrows()
            ],
            f"Kommt eine solche Zeile in die "
            f"Validierungsfalte und war ihre Spalte im Trainingsteil konstant, setzt "
            f"StandardScaler den Massstab auf 1 und reicht den Sentinel "
            f"unverändert durch — in standardisierten Einheiten ein Ausreisser von "
            f"zwei Millionen Sigma. Daraus folgen zwei Dinge: (1) Standardisierung "
            f"ist KEINE Robustheitsmassnahme, sie verstärkt Ausreisser eher; wer "
            f"sie einsetzt, braucht zusätzlich Winsorisierung oder RobustScaler. "
            f"(2) Der Plausibilitätsdeckel MAX_PLAUSIBLE_ABS = 1e12 in data_prep "
            f"greift hier nicht, weil {vt['max'].max():.1e} weit darunter liegt. Ein "
            f"spaltenweiser Deckel — wie ihn rocket_regression.py in GRENZEN je "
            f"Kanal führt — wäre auch für die Stammdaten angebracht. Für die "
            f"berichteten Ergebnisse der Arbeit ist nichts zu korrigieren: XGBoost, "
            f"CatBoost und TabPFN sind gegen einen einzelnen Extremwert unempfindlich "
            f"(er wird zu einem Split, der eine Zeile isoliert). Die Stelle gehört "
            f"aber in die Limitationen, weil sie jede künftige Erweiterung um ein "
            f"lineares oder distanzbasiertes Verfahren betrifft — einschliesslich "
            f"der KNN-Imputation aus 11.",
            "",
            "Befund D — z-Normierung je Kurve (ZNORM) ist eine andere Frage und "
            "steht in dieser Arbeit auf False. ROCKET z-normiert üblicherweise "
            "jede Reihe einzeln und behält damit nur die FORM des Verlaufs. Für "
            "Klassifikationsaufgaben ist das richtig, weil dort die absolute Höhe "
            "oft Geräteoffset ist. Hier ist sie der Prozess: die stärksten "
            "Merkmale der Arbeit sind Energie- und Massengrössen, also genau die "
            "Höhe. z-Normierung würfe sie weg. Die Einstellung geht in den "
            "Cache-Schlüssel des Tensors ein, ist also je Variante getrennt "
            "gerechnet und nicht vermischt.",
            "",
            "Zieltransformation, die Zahlen (zurückgerechnet auf Prozent, "
            "Bewertung immer auf der Originalskala):",
            *C.zahlenzeilen(zerg, "y roh"),
            f"y ist rechtsschief (Schiefe {skew(y):.2f}, nach log1p "
            f"{skew(np.log1p(y)):.2f}; Median {y.median():.1f}, Mittel {y.mean():.1f}). "
            + (
                f"Gemessen senkt {besser_z['name']} die MAE um {z_d:.3f} Punkte "
                f"({besser_z['mae']:.3f} gegen {zerg[0]['mae']:.3f}); das liegt "
                + (
                    "INNERHALB"
                    if z_d <= zerg[0]["sd"]
                    else "AUSSERHALB"
                )
                + f" der Faltenstreuung von ±{zerg[0]['sd']:.3f}. Das R² bewegt "
                f"sich dabei von {zerg[0]['r2']:+.4f} auf {besser_z['r2']:+.4f}. "
                if besser_z["name"] != "y roh"
                else f"Gemessen ist keine Transformation besser als untransformiert "
                f"(MAE {zerg[0]['mae']:.3f}). "
            )
            + "".join(
                f"ACHTUNG: {e['name']} verbessert MAE ({e['mae']:.3f}) UND R² "
                f"({e['r2']:+.4f}) zugleich — für diese Transformation gilt das "
                "Auseinanderlaufen unten NICHT, und die Entscheidung für "
                "untransformiertes y ist bei ihr nicht über R² zu begründen. "
                for e in zerg[1:]
                if e["mae"] < zerg[0]["mae"] and e["r2"] > zerg[0]["r2"]
            )
            + "Bei log1p gehen MAE und R² auseinander, und das ist kein Messfehler, "
            "sondern die Definition der beiden Masse: MAE gewichtet jeden Fehler "
            "gleich, R² quadriert ihn. Im Log-Raum zu optimieren heisst, den "
            "RELATIVEN Fehler zu minimieren — das Modell wird bei kleinen y genauer "
            "und bei den grossen Ausschusswerten schlechter, und genau dort sitzt der "
            "Nutzen der Vorhersage. Die Wahl ist damit keine statistische, sondern "
            "eine fachliche: welcher Fehler kostet in der Fertigung? Die Arbeit "
            "bewertet in Prozentpunkten und bleibt bei untransformiertem y. Dazu "
            "kommt ein praktischer Preis: die Rückabbildung mit expm1 ist nicht "
            "erwartungstreu (Jensen), die Vorhersagen sind systematisch zu niedrig — "
            "was die MAE (deren optimale Konstante der Median ist) begünstigt und "
            "das R² bestraft.",
        ],
    )
    pdf.close()
    print(f"geschrieben: {C.OUT_DIR / '12_skalierung.pdf'}")
    for e in erg + ridge + zerg:
        print(f"  {e['name']:<32} MAE {e['mae']:.4f}  R² {e['r2']:+.5f}")


if __name__ == "__main__":
    main()
