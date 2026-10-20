"""06 — Referenzvorhersagen und Negativkontrolle.

In der Arbeit: nicht berichtet; Referenzvorhersagen und Negativkontrolle als
Gegenprobe zu Abschnitt 6.1.

Konstante Vorhersagen und ein Modell auf permutiertem y.
Ausgabe: reports/eda/06_kontrolle.pdf.

    python eda/06_kontrolle.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import _common as C
from data_prep import SP  # noqa: E402

N_SPLITS = 5
SEED = 42


def r2(wahr, vorher) -> float:
    wahr, vorher = np.asarray(wahr, float), np.asarray(vorher, float)
    ss_res = ((wahr - vorher) ** 2).sum()
    ss_tot = ((wahr - wahr.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def oof_gruppenmittel(y, gruppe, folds):
    """Out-of-fold: sage den im Train gesehenen Gruppenmittelwert vorher."""
    pred = np.full(len(y), np.nan)
    for tr, va in folds:
        mittel = y.iloc[tr].groupby(gruppe.iloc[tr]).mean()
        global_ = y.iloc[tr].mean()
        pred[va] = gruppe.iloc[va].map(mittel).fillna(global_).to_numpy()
    return pred


def oof_modell(X, y, folds):
    """XGBoost out-of-fold, Feature-Processing pro Fold nur auf dem Train-Teil."""
    import xgboost as xgb
    import data_prep as D

    pred = np.full(len(y), np.nan)
    for tr, va in folds:
        Xtr, Xva, _ = D.process_features(X.iloc[tr], X.iloc[va])
        m = xgb.XGBRegressor(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            n_jobs=4,
            verbosity=0,
        )
        m.fit(Xtr, y.iloc[tr])
        pred[va] = m.predict(Xva)
    return pred


def main():
    ds = C.lade(alle_arten=True)
    X, y, df = C.daten_teil(ds)
    fuss = C.kopfzeile(ds, alle_arten=True)
    pdf = C.pdf_oeffnen("06_kontrolle.pdf")

    auftrag = df[SP["auftrag"]].astype(str)
    artikel = df[SP["artikel"]].astype(str)
    y = y.reset_index(drop=True)
    X = X.reset_index(drop=True)

    gkf = GroupKFold(n_splits=N_SPLITS)
    folds = list(gkf.split(X, y, groups=auftrag))

    ergebnisse = []
    ergebnisse.append(("globaler Mittelwert", r2(y, np.full(len(y), y.mean()))))
    ergebnisse.append(("globaler Median", r2(y, np.full(len(y), y.median()))))
    ergebnisse.append(
        ("Artikelmittel (OOF)", r2(y, oof_gruppenmittel(y, artikel, folds)))
    )
    ergebnisse.append(
        ("Auftragsmittel (OOF)", r2(y, oof_gruppenmittel(y, auftrag, folds)))
    )

    print("  rechne XGBoost (echt) ...")
    p_echt = oof_modell(X, y, folds)
    ergebnisse.append(("XGBoost, echtes y", r2(y, p_echt)))

    print("  rechne XGBoost (y global permutiert) ...")
    rng = np.random.default_rng(SEED)
    y_perm = pd.Series(rng.permutation(y.to_numpy()))
    ergebnisse.append(
        ("XGBoost, y permutiert", r2(y_perm, oof_modell(X, y_perm, folds)))
    )

    print("  rechne XGBoost (y innerhalb der Auftraege permutiert) ...")
    y_in = y.copy()
    for _, idx in pd.Series(range(len(y))).groupby(auftrag.to_numpy()):
        i = idx.to_numpy()
        y_in.iloc[i] = rng.permutation(y.iloc[i].to_numpy())
    ergebnisse.append(
        ("XGBoost, y in Aufträgen permutiert", r2(y_in, oof_modell(X, y_in, folds)))
    )

    E = pd.DataFrame(ergebnisse, columns=["verfahren", "r2"])

    r = E.set_index("verfahren").r2
    echt, innen = r["XGBoost, echtes y"], r["XGBoost, y in Aufträgen permutiert"]

    C.textseite(
        pdf,
        "06 — Baselines und Negativkontrolle",
        [
            f"GroupKFold({N_SPLITS}) nach Auftrag, {len(y)} Billets, "
            f"{auftrag.nunique()} Aufträge. XGBoost mit festen, nicht getunten "
            "Parametern (n_estimators 400, max_depth 5, learning_rate 0,05).",
            *[f"  {v}: R² {w:+.3f}" for v, w in r.items()],
            "Wie die Kontrollen zu lesen sind:",
            "  y permutiert — die Zuordnung Merkmal→Ziel ist vollständig zerstört. Jedes "
            "R² deutlich über 0 wäre ein Leck oder ein Fehler im Validierungsaufbau, "
            f"gleich welcher Art. Gemessen: {r['XGBoost, y permutiert']:+.3f}.",
            "  y innerhalb der Aufträge permutiert — die Auftragsmittelwerte bleiben exakt "
            "erhalten, nur welches Billet im Auftrag welchen Wert hat, wird zerstört. "
            f"Gemessen: {innen:+.3f} gegen {echt:+.3f} mit echtem y — es bleiben "
            f"{100 * innen / echt:.0f} % der Güte stehen. Dieser Teil stammt von der "
            "Auftragsebene; der Rest hängt an der Zuordnung einzelner Billets.",
            "Die Baselines sind out-of-fold gerechnet: der Gruppenmittelwert stammt immer "
            "nur aus dem Trainingsteil des jeweiligen Folds, unbekannte Gruppen bekommen "
            "den globalen Mittelwert. Weil die Folds nach Auftrag getrennt sind, ist "
            "JEDER Auftrag im Validierungsteil unbekannt — die Auftragsmittel-Baseline "
            "ist in diesem Aufbau also per Konstruktion der globale Mittelwert "
            f"(R² {r['Auftragsmittel (OOF)']:+.3f}) und nur der Vollständigkeit halber "
            "aufgeführt. Die Artikelmittel-Baseline ist die aussagekräftige: Artikel "
            "kommen über Aufträge hinweg wieder.",
        ],
    )

    fig, ax = C.seite(pdf, "Was erreicht welches Verfahren?", 1, 2)
    farben = [
        "grey",
        "grey",
        "steelblue",
        "steelblue",
        "seagreen",
        "indianred",
        "indianred",
    ]
    ys = np.arange(len(E))[::-1]
    ax[0].barh(ys, E.r2, color=farben[: len(E)])
    ax[0].axvline(0, color="black", lw=0.9)
    ax[0].set_yticks(ys)
    ax[0].set_yticklabels(E.verfahren, fontsize=8)
    ax[0].set_xlabel("R² (out-of-fold)")
    ax[0].set_title("Baselines, Modell und Negativkontrollen", fontsize=10)
    ax[0].grid(alpha=0.3, axis="x")
    for yy, v in zip(ys, E.r2):
        ax[0].text(max(v, 0), yy, f"  {v:+.3f}", va="center", fontsize=8)

    ax[1].scatter(p_echt, y, s=4, alpha=0.3, color="seagreen")
    lim = [
        float(min(np.nanmin(p_echt), y.min())),
        float(max(np.nanmax(p_echt), y.max())),
    ]
    ax[1].plot(lim, lim, color="crimson", ls="--", lw=1)
    ax[1].set_xlabel("Vorhersage [%]")
    ax[1].set_ylabel("y [%]")
    ax[1].set_title("XGBoost: Vorhersage gegen Messwert", fontsize=10)
    ax[1].grid(alpha=0.3)
    C.schliesse(pdf, fig, fuss)

    pdf.close()
    print(f"geschrieben: {C.OUT_DIR / '06_kontrolle.pdf'}\n")
    for _, r in E.iterrows():
        print(f"    {r.verfahren:<38}R² {r.r2:+.4f}")


if __name__ == "__main__":
    main()
