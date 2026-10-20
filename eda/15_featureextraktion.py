"""15 — Kennzahlen aus den Kurven.

In der Arbeit: nicht berichtet; Vorarbeit zum Kennzahlenkatalog in Abschnitt
5.3.

Katalog der Kennzahlen, Vergleich der Extraktionsverfahren, Ablation.
Ausgabe: reports/eda/15_featureextraktion.pdf.

    python eda/15_featureextraktion.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import _common as C

sys.path.insert(0, str(C.ROOT / "data_pipeline"))
import data_prep as D  # noqa: E402
from runs_io import lies_summary  # noqa: E402
from data_prep import SP  # noqa: E402

N_ROWS = SP["n_zeilen_final"]

FAMILIEN = list(D._AGG_SUFFIXES)

EXTRAKTION = {
    "Aggregate (handgewählt)": ["XGBoost", "CatBoost", "TabPFN"],
    "MiniRocket (Faltungskerne)": ["ROCKET"],
    "HYDRA (Kerne + Wörterbuch)": ["HYDRA"],
    "FreshPRINCE (tsfresh)": ["FreshPRINCE"],
}


def familie(spalte: str) -> str | None:
    """Zu welcher Aggregatfamilie gehoert die Spalte? None = Stammdatum."""
    for s in FAMILIEN:
        if spalte.endswith(s):
            return s
    return None


def main():
    ds = C.lade(alle_arten=True)
    X, y, df = C.daten_teil(ds)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    df = df.reset_index(drop=True)
    fuss = C.kopfzeile(ds, alle_arten=True)
    pdf = C.pdf_oeffnen("15_featureextraktion.pdf")
    folds = C.folds_final(df)

    influx = set(D.influx_spalten(list(X.columns)))
    fam = pd.Series({c: familie(c) for c in X.columns})
    num = C.numerische_merkmale(X)
    abw = C.auftragsabweichung(y, df[SP["auftrag"]].astype(str))

    fig, ax = C.seite(
        pdf, "A — Was der Aggregatkatalog enthält", 1, 3, groesse=(11.7, 5.2)
    )

    zahl = fam.value_counts().reindex(FAMILIEN).fillna(0).astype(int)
    n_stamm = int(fam.isna().sum())
    pos = np.arange(len(zahl) + 1)
    ax[0].barh(
        pos,
        list(zahl.to_numpy()) + [n_stamm],
        color=["#3182bd"] * len(zahl) + ["#2ca02c"],
    )
    ax[0].set_yticks(pos)
    ax[0].set_yticklabels(list(zahl.index) + ["(Stammdaten)"], fontsize=8)
    ax[0].invert_yaxis()
    ax[0].set_xlabel("Anzahl Merkmale")
    ax[0].set_title("A1  Merkmale je Familie", fontsize=10)
    ax[0].spines["left"].set_visible(False)
    ax[0].tick_params(axis="y", length=0)

    t_y = C.spearman_tabelle(X, y, num).set_index("merkmal")
    t_a = C.spearman_tabelle(X, abw, num).set_index("merkmal")

    gruppen, daten_y, daten_a = [], [], []
    for f in FAMILIEN + ["(Stammdaten)"]:
        sp = [
            c
            for c in num
            if (familie(c) == f) or (f == "(Stammdaten)" and familie(c) is None)
        ]
        sp_y = [abs(t_y.rho[c]) for c in sp if c in t_y.index]
        sp_a = [abs(t_a.rho[c]) for c in sp if c in t_a.index]
        if len(sp_y) < 3:
            continue
        gruppen.append(f)
        daten_y.append(sp_y)
        daten_a.append(sp_a)

    ax[1].boxplot(daten_y, vert=False, showfliers=False, widths=0.6)
    ax[1].set_yticklabels(gruppen, fontsize=7.5)
    ax[1].set_xlabel("|Spearman ρ| gegen y")
    ax[1].set_title("A2  Welche Familie korreliert mit y?", fontsize=10)
    ax[1].grid(alpha=0.3, axis="x")

    ax[2].boxplot(daten_a, vert=False, showfliers=False, widths=0.6)
    ax[2].set_yticklabels(gruppen, fontsize=7.5)
    ax[2].set_xlabel("|Spearman ρ| gegen y − Auftragsmittel")
    ax[2].set_title("A3  … und mit der Auftragsabweichung?", fontsize=10)
    ax[2].set_xlim(ax[1].get_xlim())
    ax[2].grid(alpha=0.3, axis="x")

    C.schliesse(pdf, fig, fuss + " | Spearman, FDR-korrigiert")

    s = lies_summary(C.ROOT / "runs" / "runs_summary.csv")
    s = s[
        (s.n_rows == N_ROWS) & (s.cv_level == "artikel") & (s.test_level == "artikel")
    ]
    verfahren = []
    for name, modelle in EXTRAKTION.items():
        kand = s[s.model.isin(modelle)]
        if kand.empty:
            continue
        best = kand.sort_values("cv_mae_mean").iloc[0]
        verfahren.append(
            {
                "name": name,
                "modell": best.model,
                "mae": float(best.cv_mae_mean),
                "sd": float(best.cv_mae_std),
                "r2": float(best.oof_r2),
                "n_feat": int(best.n_features),
            }
        )
    basis_mae = (
        float(s.cv_baseline_median_mae_mean.dropna().iloc[0]) if len(s) else np.nan
    )

    fig, ax = C.seite(pdf, "B — Vier Wege, aus einer Schmelze Merkmale zu machen", 2, 2)

    pos = np.arange(len(verfahren))
    ax[0].barh(
        pos,
        [v["mae"] for v in verfahren],
        xerr=[v["sd"] for v in verfahren],
        color="#3182bd",
        capsize=3,
    )
    ax[0].set_yticks(pos)
    ax[0].set_yticklabels([v["name"] for v in verfahren], fontsize=8)
    ax[0].invert_yaxis()
    ax[0].set_xlabel("CV-MAE [Prozentpunkte]")
    ax[0].set_title("B1  CV-MAE je Extraktionsverfahren", fontsize=10)
    if np.isfinite(basis_mae):
        ax[0].axvline(
            basis_mae,
            color="crimson",
            ls="--",
            lw=1.2,
            label="Median-Baseline",
        )
        ax[0].legend(fontsize=7, loc="lower right")

    ax[1].scatter(
        [v["n_feat"] for v in verfahren],
        [v["r2"] for v in verfahren],
        s=90,
        color="#d62728",
    )
    for v in verfahren:
        ax[1].annotate(
            v["name"].split(" (")[0],
            (v["n_feat"], v["r2"]),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=7.5,
        )
    ax[1].axvline(len(y), color="k", ls="--", lw=1.2, label="Zahl der Billets")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("erzeugte Merkmale (logarithmisch)")
    ax[1].set_ylabel("OOF-R²")
    ax[1].axhline(0, color="0.6", lw=0.8)
    ax[1].set_title("B2  Güte gegen Zahl der erzeugten Merkmale", fontsize=10)
    ax[1].legend(fontsize=7, loc="lower left")
    ax[1].grid(alpha=0.3)

    print("  Ablation der Aggregatfamilien ...")
    ablation = []
    pred0 = C.oof_xgb(X, y, folds)
    ablation.append(
        {
            "name": "alle Merkmale",
            "mae": C.mae(y, pred0),
            "r2": C.r2(y, pred0),
            "sd": C.falten_streuung(y, pred0, folds)[1],
            "n": X.shape[1],
        }
    )
    print(f"     alle: MAE {ablation[0]['mae']:.3f}")
    for f in FAMILIEN:
        weg = [c for c in X.columns if familie(c) == f]
        if len(weg) < 3:
            continue
        print(f"     ohne {f} ({len(weg)} Spalten) ...", flush=True)
        pred = C.oof_xgb(X.drop(columns=weg), y, folds)
        ablation.append(
            {
                "name": f"ohne {f}",
                "mae": C.mae(y, pred),
                "r2": C.r2(y, pred),
                "sd": C.falten_streuung(y, pred, folds)[1],
                "n": X.shape[1] - len(weg),
            }
        )
        print(f"        MAE {ablation[-1]['mae']:.3f}")

    C.vergleichstabelle(
        ax[2],
        ablation,
        "alle Merkmale",
        "MAE",
        "C  Ablation der Aggregatfamilien",
    )

    print("  Quellen-Ablation ...")
    quellen = []
    for name, spalten in [
        ("beides (Stand)", list(X.columns)),
        ("nur SQL", [c for c in X.columns if c not in influx]),
        ("nur Influx", [c for c in X.columns if c in influx or c in D._ABGELEITET]),
    ]:
        print(f"     {name} ({len(spalten)} Spalten) ...", flush=True)
        pred = C.oof_xgb(X[spalten], y, folds)
        quellen.append(
            {
                "name": f"{name}  [{len(spalten)}]",
                "mae": C.mae(y, pred),
                "r2": C.r2(y, pred),
                "sd": C.falten_streuung(y, pred, folds)[1],
            }
        )
        print(f"        MAE {quellen[-1]['mae']:.3f}  R² {quellen[-1]['r2']:+.4f}")

    C.vergleichstabelle(
        ax[3],
        quellen,
        quellen[0]["name"],
        "MAE",
        "D  Stammdaten gegen Prozesskanäle",
    )
    C.schliesse(
        pdf,
        fig,
        f"B aus runs_summary.csv (n_rows={N_ROWS}, artikelrein) | "
        f"C und D neu gerechnet, XGBoost mit getunten Parametern auf "
        f"{len(folds)} artikelreinen Falten",
    )

    teuerste = max(ablation[1:], key=lambda e: e["mae"]) if len(ablation) > 1 else None
    billigste = min(ablation[1:], key=lambda e: e["mae"]) if len(ablation) > 1 else None
    q_sql = next(q for q in quellen if q["name"].startswith("nur SQL"))
    q_inf = next(q for q in quellen if q["name"].startswith("nur Influx"))
    q_bei = quellen[0]
    auto = [v for v in verfahren if v is not verfahren[0]]
    agg = verfahren[0]
    gewinn_inf = q_bei["r2"] - q_sql["r2"]
    C.textseite(
        pdf,
        "Feature-Extraktion — Entscheidung und Begründung",
        [
            "ENTSCHEIDUNG: handgewählte Aggregate je Kanal (mean, std, min, max, "
            "last, range, n_samples, known_frac, bei Binärkanälen zusätzlich "
            "frac_on_t und n_edges), keine Vorauswahl unter ihnen. Die "
            "automatischen Extraktionen werden als Vergleich mitgeführt, nicht "
            "als Ersatz.",
            "",
            f"Grundlage: {C.umfang_text(len(y))}, {len(folds)} artikelreine Falten. "
            "Seite B stammt aus runs_summary.csv (je Verfahren der beste Lauf nach "
            "CV-MAE), C und D sind hier neu gerechnet (XGBoost, getunte Parameter).",
            "Merkmale je Familie: "
            + ", ".join(f"{f} {n}" for f, n in zahl.items())
            + f", Stammdaten {n_stamm}.",
            "Mediane |ρ| je Familie, gegen y und gegen die Auftragsabweichung "
            "(Felder A2 und A3):",
            *[
                f"  {f}: n {len(dy)}, gegen y {np.median(dy):.3f}, gegen Abweichung "
                f"{np.median(da):.3f}"
                for f, dy, da in zip(gruppen, daten_y, daten_a)
            ],
            f"Befund A — der Katalog erzeugt aus {len(influx)} Influx-Spalten und "
            f"{X.shape[1] - len(influx)} Stammdatenspalten zusammen {X.shape[1]} "
            f"Rohmerkmale. Die Familien unterscheiden sich in ihrer Korrelation "
            f"mit y deutlich, in ihrer Korrelation mit der AUFTRAGSABWEICHUNG "
            f"dagegen kaum — alle liegen dort nahe null. Das ist dieselbe Aussage "
            f"wie in 03_merkmale.pdf, hier nach Extraktionsfamilie aufgeschlüsselt: "
            f"KEINE Familie des Katalogs trennt Billets innerhalb eines Auftrags. "
            f"Die Extraktion ist damit nicht der Engpass; eine andere Aggregation "
            f"derselben Kanäle würde daran nichts ändern. '_n_samples' ist gesondert "
            f"zu lesen: die Zahl der Messpunkte ist kein Prozesswert, sondern ein "
            f"Schreibverhalten der Anlage — hohe Werte dort sind Erkennung, keine "
            f"Erklärung.",
            "",
            "Befund B, die Zahlen:",
            *[
                f"  {v['name']}: bestes Modell {v['modell']}, CV-MAE {v['mae']:.3f} "
                f"± {v['sd']:.3f}, OOF-R² {v['r2']:+.3f}, {v['n_feat']} Merkmale"
                for v in verfahren
            ],
            f"Median-Baseline {basis_mae:.3f}.",
            f"Befund B — die automatischen Verfahren gewinnen nicht. "
            + ", ".join(v["name"].split(" (")[0] for v in auto)
            + f" erzeugen zwischen {C.zahl(min(v['n_feat'] for v in auto))} und "
            f"{C.zahl(max(v['n_feat'] for v in auto))} Merkmale. "
            + (
                "Keines liegt unter der MAE der "
                f"{agg['n_feat']} handgewählten Aggregate ({agg['mae']:.3f}); "
                if all(v["mae"] >= agg["mae"] for v in auto)
                else "Mindestens eines liegt knapp unter den handgewählten Aggregaten; "
            )
            + "Unterschiede kleiner als die Faltenstreuung (± "
            f"{agg['sd']:.3f}) sind dabei nicht als Rangfolge zu lesen: "
            + ", ".join(
                f"{v['name'].split(' (')[0]} {v['mae'] - agg['mae']:+.3f}" for v in auto
            )
            + ". Das ist kein Argument gegen diese Verfahren an sich — es ist eine "
            f"Aussage über die Datenlage: bei {len(y)} Zeilen ist eine Extraktion mit "
            f"{C.zahl(max(v['n_feat'] for v in auto))} Merkmalen tief im "
            "p-grösser-n-Bereich, und die Regularisierung muss dann so stark sein, "
            "dass vom Signal wenig übrig bleibt. Wer die Verfahren zitiert, sollte die "
            "Merkmalszahl "
            "danebenstellen (Bild B2).",
            "",
            "Befund C, die Zahlen:",
            *C.zahlenzeilen(ablation, "alle Merkmale"),
            (
                f"Befund C — die Ablation bestätigt A. Die teuerste Familie ist "
                f"'{teuerste['name'][5:]}' mit {teuerste['mae'] - ablation[0]['mae']:+.3f} "
                f"MAE, die billigste '{billigste['name'][5:]}' mit "
                f"{billigste['mae'] - ablation[0]['mae']:+.3f}; die Faltenstreuung "
                f"beträgt ±{ablation[0]['sd']:.3f}. Keine einzelne Familie ist "
                f"also unverzichtbar. Das ist die erwartete Folge starker "
                f"Redundanz: die Aggregate desselben Kanals sind untereinander "
                f"hoch korreliert (07_datenqualitaet.pdf), und fällt eine Familie "
                f"weg, übernehmen die übrigen. Für die Arbeit heisst das, dass "
                f"Importance-Ranglisten einzelner Merkmale mit Vorsicht zu lesen "
                f"sind — sie verteilen Anteile innerhalb einer austauschbaren "
                f"Gruppe."
                if teuerste
                else "Befund C — keine Ablation gerechnet."
            ),
            "",
            "Befund D, die Zahlen:",
            *C.zahlenzeilen(quellen, quellen[0]["name"]),
            f"Befund D — gemessen auf dem endgültigen Stand (v6, y ∈ "
            f"{C.y_grenzen_text()}, artikelreine Falten): beides zusammen "
            f"MAE {q_bei['mae']:.3f} / R² {q_bei['r2']:+.4f}, nur Stammdaten "
            f"{q_sql['mae']:.3f} / {q_sql['r2']:+.4f}, nur Prozesskanäle "
            f"{q_inf['mae']:.3f} / {q_inf['r2']:+.4f}. Die Prozesskanäle "
            + (
                f"bringen also {gewinn_inf:+.4f} R² gegenüber den Stammdaten allein "
                f"(MAE {q_bei['mae'] - q_sql['mae']:+.3f}, Faltenstreuung ± "
                f"{q_bei['sd']:.3f}) — sie KOSTEN nicht, sie tragen bei, wenn auch "
                "wenig. Damit hat sich der Befund gegenüber dem Stand in data_prep.py "
                "GEDREHT und muss dort wie in der Arbeit nachgezogen werden."
                if gewinn_inf > 0
                else f"kosten also {gewinn_inf:+.4f} R² gegenüber den Stammdaten allein."
            ),
            "Die ältere Messung galt für einen früheren Datenstand. Seither haben "
            "sich drei Dinge geändert: v6 mit korrigierter "
            "Spaltenklassifikation, die Untergrenze Y_MIN = 1 (die zwei "
            "Artefaktgruppen entfernt, siehe 08_y_untergrenze.pdf) und die "
            "Artikelebene als Berichtsebene. Welcher der drei den Ausschlag gibt, "
            "ist hier nicht getrennt worden — das wäre die nächste Messung. "
            "Festzuhalten ist zweierlei: FEATURE_SOURCE = 'beides' ist auf dem "
            "endgültigen Stand die gemessen beste Einstellung und nicht nur die "
            "bequemste, und der lange Begründungskommentar in data_prep.py "
            "beschreibt eine Messung, die nicht mehr gilt.",
            "",
            "Was daraus für die Feature-Extraktion folgt, und was nicht. Es folgt "
            "NICHT, dass die Extraktion schlecht gewählt ist — B und C zeigen, "
            "dass weder eine automatische Extraktion noch das Weglassen einer "
            "Familie es besser macht. Es folgt auch nicht, dass die Prozesskanäle "
            "den Billet-Ausschuss erklären: nach Befund A korreliert KEINE ihrer "
            "Familien nennenswert mit der Abweichung vom Auftragsmittel, und nach "
            "13_glaettung.pdf und 14_segmentierung.pdf ändern weder Glättung noch "
            "Fensterwahl im Kurvenzweig etwas Deutliches. Der Beitrag der Kanäle "
            "liegt damit wahrscheinlich nicht in der "
            "Physik der Schmelze, sondern in dem, was sie über den Auftrag "
            "verraten — Programm, Maschine, Stillstandsmuster. Ob der Prozess zu "
            "stabil ist oder die entscheidenden Grössen gar nicht gemessen "
            "werden, lässt sich mit diesen Daten nicht unterscheiden; das gehört "
            "in die Limitationen und ist die wichtigste offene Frage der Arbeit.",
        ],
    )
    pdf.close()
    print(f"geschrieben: {C.OUT_DIR / '15_featureextraktion.pdf'}")
    for e in ablation + quellen:
        print(f"  {e['name']:<28} MAE {e['mae']:.3f}  R² {e['r2']:+.4f}")


if __name__ == "__main__":
    main()
