"""10 — Modellvergleich über die drei Bewertungsebenen.

In der Arbeit: Abschnitt 6.4: Abbildung 12 (CV-MAE) und Abbildung 13 (CV-R²)
mit den Abständen zur Median-Vorhersage; Bezugsgrößen in 7.2 und 8.1.

Auswahl der Läufe über runs_io.waehle_finale_laeufe.
Ausgabe: reports/eda/10_modellvergleich.pdf, reports/figures/modellvergleich*.png,
reports/figures/kapitel6/.

    python eda/10_modellvergleich.py
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _common as C

sys.path.insert(0, str(C.ROOT / "data_pipeline"))
from runs_io import N_ROWS_FINAL as N_ROWS, waehle_finale_laeufe  # noqa: E402

EBENEN = [
    ("billet", "billet", "random"),
    ("auftrag", "auftrag", "auftrag"),
    ("artikel", "artikel", "artikel"),
]
KONFIG = [
    "XGBoost",
    "CatBoost",
    "TabPFN",
    "ROCKET (Kurven)",
    "ROCKET (hybrid)",
    "HYDRA",
    "FreshPRINCE",
]
FARBEN = {
    "XGBoost": "#3182bd",
    "CatBoost": "#31a354",
    "TabPFN": "#7b3294",
    "ROCKET (Kurven)": "#f4a582",
    "ROCKET (hybrid)": "#d62728",
    "HYDRA": "#ff7f0e",
    "FreshPRINCE": "#8c564b",
}


def lade() -> pd.DataFrame:
    gewaehlt = waehle_finale_laeufe(
        C.ROOT / "runs" / "runs_summary.csv",
        C.ROOT / "runs" / "single",
        konfigs=KONFIG,
        ebenen=[n for n, _, _ in EBENEN],
    )
    zeilen = []
    for name, _, _ in EBENEN:
        for modell in KONFIG:
            t = gewaehlt[(gewaehlt.ebene == name) & (gewaehlt.konfig == modell)]
            if t.empty:
                continue
            r = t.iloc[0]
            zeilen.append(
                {
                    "modell": modell,
                    "ebene": name,
                    "run_id": r.run_id,
                    "cv_mae": r.cv_mae_mean,
                    "cv_std": r.cv_mae_std,
                    "cv_r2": r.cv_r2_mean,
                    "basis_r2_cv": r.cv_baseline_mittel_r2_mean,
                    "test_r2": r.test_r2,
                    "basis_r2_test": r.baseline_mittel_r2,
                    "basis_cv": r.cv_baseline_median_mae_mean,
                    "test_mae": r.test_mae,
                    "basis_test": r.baseline_median_mae,
                }
            )
    e = pd.DataFrame(zeilen)
    e["gewinn_cv"] = e.basis_cv - e.cv_mae
    e["gewinn_test"] = e.basis_test - e.test_mae
    e["gewinn_r2_cv"] = e.cv_r2 - e.basis_r2_cv
    return e


MAE = {
    "cv": "cv_mae",
    "basis": "basis_cv",
    "gewinn": "gewinn_cv",
    "test": "test_mae",
    "name": "MAE",
    "einheit": " [Prozentpunkte]",
    "basisname": "Median-Baseline",
    "hoeher_besser": False,
    "fmt": "{:.2f}",
    "datei": "modellvergleich.png",
}
R2 = {
    "cv": "cv_r2",
    "basis": "basis_r2_cv",
    "gewinn": "gewinn_r2_cv",
    "test": "test_r2",
    "name": "R²",
    "einheit": "",
    "basisname": "Mittelwert-Baseline",
    "hoeher_besser": True,
    "fmt": "{:+.3f}",
    "datei": "modellvergleich_r2.png",
}


def _reihenfolge(e, k=MAE):
    """Modelle nach ihrer Guete auf der massgeblichen Ebene sortieren."""
    m = e[e.ebene == "auftrag"].set_index("modell")[k["cv"]]
    if m.empty:
        m = e[e.ebene == "artikel"].set_index("modell")[k["cv"]]
    uebrig = [x for x in KONFIG if x not in m.index]
    return list(m.sort_values(ascending=not k["hoeher_besser"]).index) + uebrig


def feld_a(ax, e, k=MAE):
    modelle = _reihenfolge(e, k)
    x = np.arange(len(modelle))
    breite = 0.26
    for i, (name, _, _) in enumerate(EBENEN):
        s = e[e.ebene == name].set_index("modell")
        werte = [s[k["cv"]].get(m, np.nan) for m in modelle]
        balken = ax.bar(
            x + (i - 1) * breite,
            werte,
            breite,
            label=name,
            color=plt_farbe(i),
            edgecolor="white",
            linewidth=0.6,
        )
        _beschrifte(ax, balken, werte, k["fmt"])
        for j, v in enumerate(werte):
            if np.isnan(v):
                ax.annotate(
                    "fehlt",
                    xy=(x[j] + (i - 1) * breite, 0.005 if k["hoeher_besser"] else 0.3),
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=6.5,
                    color="#999999",
                )
    for i, (name, _, _) in enumerate(EBENEN):
        s = e[e.ebene == name]
        if not s.empty:
            ax.axhline(
                s[k["basis"]].iloc[0],
                color=plt_farbe(i),
                ls="--",
                lw=1,
                alpha=0.8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" (", "\n(") for m in modelle], fontsize=7.5)
    ax.set_ylabel(f"CV-{k['name']}{k['einheit']}")
    ax.set_title(f"A  CV-{k['name']} je Modell und Bewertungsebene", fontsize=10)
    if k["hoeher_besser"]:
        lo = min(np.nanmin(e[k["cv"]]), np.nanmin(e[k["basis"]]), 0)
        hi = max(np.nanmax(e[k["cv"]]), 0)
        rand = 0.35 * (hi - lo)
        ax.set_ylim(lo - 0.6 * rand, hi + 1.6 * rand)
        ax.axhline(0, color="black", lw=1)
    else:
        ax.set_ylim(0, max(e[k["cv"]].max(), e[k["basis"]].max()) * 1.35)
    from matplotlib.lines import Line2D

    griffe, _ = ax.get_legend_handles_labels()
    for i, (name, _, _) in enumerate(EBENEN):
        s = e[e.ebene == name]
        if not s.empty:
            griffe.append(
                Line2D(
                    [],
                    [],
                    color=plt_farbe(i),
                    ls="--",
                    lw=1,
                    label=f"{k['basisname']} {name} "
                    f"{s[k['basis']].iloc[0]:{'+.3f' if k['hoeher_besser'] else '.3f'}}"
                    .replace(".", ","),
                )
            )
    ax.legend(handles=griffe, fontsize=7, loc="upper left", ncol=2)
    ax.grid(alpha=0.3, axis="y")


def _beschrifte(ax, balken, werte, fmt):
    """Wert senkrecht ueber (bzw. bei negativen Werten unter) jeden Balken."""
    for b, v in zip(balken, werte):
        if np.isnan(v):
            continue
        ax.annotate(
            fmt.format(v),
            xy=(b.get_x() + b.get_width() / 2, v),
            xytext=(0, 2 if v >= 0 else -2),
            textcoords="offset points",
            ha="center",
            va="bottom" if v >= 0 else "top",
            rotation=90,
            fontsize=6,
        )


def plt_farbe(i):
    return ["#9ecae1", "#6baed6", "#2171b5"][i]


def feld_b(ax, e, k=MAE):
    modelle = _reihenfolge(e, k)
    x = np.arange(len(modelle))
    breite = 0.26
    for i, (name, _, _) in enumerate(EBENEN):
        s = e[e.ebene == name].set_index("modell")
        werte = [s[k["gewinn"]].get(m, np.nan) for m in modelle]
        balken = ax.bar(
            x + (i - 1) * breite,
            werte,
            breite,
            label=name,
            color=plt_farbe(i),
            edgecolor="white",
            linewidth=0.6,
        )
        _beschrifte(ax, balken, werte, "{:+.3f}" if k["hoeher_besser"] else "{:+.2f}")
    ax.axhline(0, color="black", lw=1)
    lo, hi = np.nanmin(e[k["gewinn"]]), np.nanmax(e[k["gewinn"]])
    rand = 0.35 * (max(hi, 0) - min(lo, 0))
    ax.set_ylim(min(lo, 0) - rand, max(hi, 0) + rand)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" (", "\n(") for m in modelle], fontsize=7.5)
    ax.set_ylabel(f"Gewinn gegenüber {k['basisname']}{k['einheit']}")
    ax.set_title(f"B  Gewinn gegenüber der {k['basisname']}", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower left")
    ax.grid(alpha=0.3, axis="y")


def feld_c(ax, e, k=MAE):
    """Spannweite über die Ebenen gegen die Spannweite zwischen den Modellen."""
    voll = [
        m for m in _reihenfolge(e, k) if e[e.modell == m].ebene.nunique() == len(EBENEN)
    ]
    spannen = []
    for m in voll:
        s = e[e.modell == m].set_index("ebene")[k["cv"]]
        spannen.append((m, s.min(), s.max(), s.max() - s.min()))
    y = np.arange(len(spannen))
    for i, (m, lo, hi, _) in enumerate(spannen):
        ax.plot(
            [lo, hi],
            [i, i],
            lw=6,
            color=FARBEN.get(m, "#888888"),
            solid_capstyle="butt",
        )
        ax.plot([lo, hi], [i, i], "|", color="black", ms=10, mew=1.2)
    s = e[e.ebene == "auftrag"]
    if not s.empty and len(s) > 1:
        ax.axvspan(
            s[k["cv"]].min(),
            s[k["cv"]].max(),
            color="#999999",
            alpha=0.18,
            label="Spanne zwischen den Modellen (Auftragsebene)",
        )
        ax.legend(fontsize=7, loc="upper right", framealpha=0.92)
    ax.set_yticks(y)
    ax.set_yticklabels([m for m, _, _, _ in spannen], fontsize=8)
    if k["hoeher_besser"]:
        ax.axvline(0, color="black", lw=1)
    ax.set_xlabel(f"CV-{k['name']}{k['einheit']}")
    ax.set_ylim(-0.6, len(spannen) - 0.2)
    ax.set_title(
        f"C  Spanne {'des' if k['hoeher_besser'] else 'der'} CV-{k['name']} "
        "über die drei Ebenen",
        fontsize=10,
    )
    ax.grid(alpha=0.3, axis="x")


def feld_d(ax, e, k=MAE):
    formen = ["o", "s", "^", "D", "v", "P", "X"]
    modelle = _reihenfolge(e, k)
    form_je_modell = {m: formen[i % len(formen)] for i, m in enumerate(modelle)}
    for name, _, _ in EBENEN:
        s = e[e.ebene == name]
        for _, r in s.iterrows():
            ax.scatter(
                r[k["cv"]],
                r[k["test"]],
                s=70,
                marker=form_je_modell[r.modell],
                color=plt_farbe([n for n, _, _ in EBENEN].index(name)),
                edgecolors="white",
                linewidth=0.8,
                zorder=3,
            )
    pad = 0.05 if k["hoeher_besser"] else 0.5
    lo = min(e[k["cv"]].min(), e[k["test"]].min()) - pad
    hi = max(e[k["cv"]].max(), e[k["test"]].max()) + pad
    ax.plot([lo, hi], [lo, hi], "--", color="#999999", lw=1, zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    if k["hoeher_besser"]:
        ax.axhline(0, color="black", lw=0.8, alpha=0.6)
        ax.axvline(0, color="black", lw=0.8, alpha=0.6)
    ax.set_xlabel(f"CV-{k['name']}{k['einheit']}")
    ax.set_ylabel(f"Test-{k['name']}{k['einheit']}")
    ax.set_title("D  Kreuzvalidierung gegen Test-Set", fontsize=10)

    from matplotlib.lines import Line2D

    l_ebene = [
        Line2D([], [], marker="o", ls="", color=plt_farbe(i), label=n)
        for i, (n, _, _) in enumerate(EBENEN)
    ] + [Line2D([], [], color="#999999", ls="--", lw=1, label="CV = Test")]
    l_modell = [
        Line2D([], [], marker=form_je_modell[m], ls="", color="#555555", label=m)
        for m in modelle
    ]
    erste = ax.legend(
        handles=l_ebene, fontsize=7, loc="lower right" if k["hoeher_besser"] else "upper left"
    )
    ax.add_artist(erste)
    ax.legend(
        handles=l_modell,
        fontsize=6.5,
        loc="upper left" if k["hoeher_besser"] else "lower right",
        ncol=2,
    )
    ax.grid(alpha=0.3)


def einzelfeld(e, k, datei):
    """Feld A allein als eigene Abbildung (Kapitel 6, Abbildungen 12 und 13)."""
    fig, ax = plt.subplots(figsize=(16 / 2.54, 9 / 2.54))
    feld_a(ax, e, k)
    C.aufraeumen(fig)
    ziel = C.ROOT / "reports" / "figures" / "kapitel6" / datei
    ziel.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ziel, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"geschrieben: {ziel.relative_to(C.ROOT)}")


def main():
    e = lade()
    if e.empty:
        raise SystemExit("Keine passenden Läufe gefunden.")

    pdf = C.pdf_oeffnen("10_modellvergleich.pdf")
    fig, ax = C.seite(pdf, "Modellvergleich über die drei Bewertungsebenen", 2, 2)
    feld_a(ax[0], e)
    feld_b(ax[1], e)
    feld_c(ax[2], e)
    feld_d(ax[3], e)
    C.aufraeumen(fig)
    fig.savefig(
        C.ROOT / "reports" / "figures" / MAE["datei"],
        dpi=200,
        bbox_inches="tight",
    )
    fehlend = [
        f"{m}/{lv}"
        for m in KONFIG
        for lv, _, _ in EBENEN
        if e[(e.modell == m) & (e.ebene == lv)].empty
    ]
    C.schliesse(
        pdf,
        fig,
        f"{C.CSV_PATH.name} | {N_ROWS} Billets | je Modell und Ebene der zuletzt "
        f"geschriebene Lauf" + (f" | fehlend: {', '.join(fehlend)}" if fehlend else ""),
    )

    fig, ax = C.seite(pdf, "Modellvergleich über die drei Bewertungsebenen (R²)", 2, 2)
    feld_a(ax[0], e, R2)
    feld_b(ax[1], e, R2)
    feld_c(ax[2], e, R2)
    feld_d(ax[3], e, R2)
    C.aufraeumen(fig)
    fig.savefig(
        C.ROOT / "reports" / "figures" / R2["datei"],
        dpi=200,
        bbox_inches="tight",
    )
    C.schliesse(
        pdf,
        fig,
        f"{C.CSV_PATH.name} | {N_ROWS} Billets | R² gegen die Mittelwert-Baseline, "
        "die das R² definitionsgemäß auf 0 setzt (in der CV leicht darunter)",
    )

    zeilen = []
    for m in _reihenfolge(e):
        for lv, _, _ in EBENEN:
            t = e[(e.modell == m) & (e.ebene == lv)]
            if t.empty:
                zeilen.append(f"  {m}, {lv}: kein Lauf")
                continue
            r = t.iloc[0]
            zeilen.append(
                f"  {m}, {lv}: CV-MAE {r.cv_mae:.3f}, CV-R² "
                f"{r.cv_r2:+.3f}, Gewinn {r.gewinn_cv:+.3f}, Test-MAE {r.test_mae:.3f}"
            )
    spannen = []
    for m in _reihenfolge(e):
        t = e[e.modell == m]
        if t.ebene.nunique() == len(EBENEN):
            spannen.append(f"{m} {t.cv_mae.max() - t.cv_mae.min():.2f}")
    au = e[e.ebene == "auftrag"]
    basis = e.groupby("ebene").basis_cv.first().to_dict()
    einzelfeld(e, MAE, "abb_modellvergleich_mae.png")
    einzelfeld(e, R2, "abb_modellvergleich_r2.png")

    C.textseite(
        pdf,
        "Zahlen und Vorbehalte",
        [
            "Median-Baseline je Ebene (CV): "
            + ", ".join(f"{k} {v:.4f}" for k, v in basis.items())
            + ". Jede Kennzahl ist gegen die Vorhersage gemessen, die sie "
            "definitionsgemäß minimiert — für die MAE ist das der Median.",
            "Je Modell und Ebene (Gewinn = Median-Baseline minus CV-MAE, über 0 "
            "heisst besser als die Baseline; die Zahlen stehen auch an den Balken "
            "in Feld A und B):",
            *zeilen,
            "Spanne der CV-MAE über die drei Ebenen (Feld C): "
            + ", ".join(spannen)
            + (
                f". Zum Vergleich die Spanne zwischen den Modellen auf der "
                f"Auftragsebene (graues Band): {au.cv_mae.max() - au.cv_mae.min():.2f}."
                if len(au) > 1
                else "."
            ),
            "Vorbehalt zur Auswahl. runs_summary.csv enthält auch Ablations- und "
            "Zwischenläufe: ungetuntes XGBoost, ROCKET mit verschiedenen "
            "Kernelzahlen und beiden ZNORM-Einstellungen. Diese Seite nimmt je "
            "Modell und Ebene den zuletzt geschriebenen Lauf der endgültigen "
            "Konfiguration. ROCKET ist MiniRocket mit 1000 Kerneln und ZNORM, "
            "einmal nur auf den Rohkurven (Kurven) und einmal zusätzlich mit den "
            "tabellarischen v6-Merkmalen (hybrid). Transformer und Kernelzahl "
            "stehen in params_source, znorm und hybrid nur im Run-JSON; die "
            "Auswahl liest sie deshalb dort aus.",
            "Vorbehalt zur Vergleichbarkeit. XGBoost und CatBoost sind mit je 100 "
            "Optuna-Versuchen auf der Artikelebene abgestimmt, TabPFN, ROCKET, "
            "HYDRA und FreshPRINCE gar nicht. Bei den faltungsbasierten Verfahren "
            "ist das vertretbar, weil ihr einziger Modellparameter — die "
            "Regularisierung des linearen Kopfes — intern per Kreuzvalidierung "
            "gewählt wird. Bei FreshPRINCE ist es eine echte Einschränkung.",
            "Vorbehalt zu FreshPRINCE. Der Lauf verwendet tsfresh im Modus "
            "minimal. Der Modus comprehensive aus der Veröffentlichung "
            "(Guijo-Rubio et al. 2024) war wegen der Rechenzeit nicht durchführbar.",
            "Vorbehalt zur Streuung. Die Faltenstreuung ist nicht mehr gezeigt; "
            "sie wäre nach "
            "Bates, Hastie und Tibshirani (2021) ohnehin systematisch zu klein, weil jede "
            "Beobachtung in Training und Bewertung vorkommt und die Faltenwerte "
            "dadurch korreliert sind. Unterschiede in der Größenordnung dieser "
            "Streuung sind nicht als gesichert zu lesen.",
        ],
    )
    pdf.close()
    print(f"geschrieben: {C.OUT_DIR / '10_modellvergleich.pdf'}")
    print(f"  {len(e)} Lauf-Zellen, fehlend: {fehlend if fehlend else 'keine'}")


if __name__ == "__main__":
    main()
