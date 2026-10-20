"""02 — Erklärungen einzelner Billets.

In der Arbeit: Abschnitt 7.4: Abbildung 22 (vier Billets der Testmenge) und die
Überlappung der wichtigsten Merkmale zweier Billets. Abbildung zeichnet xai/23.

SHAP-Beiträge ausgewählter Billets und Überlappung der wichtigsten Merkmale.
Ausgaben unter reports/xai/, runs/xai/ und reports/xai/tabellen/.

    TEST_LEVEL=<billet|auftrag|artikel> python xai/02_lokal.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import _common as C

TOP_LOKAL = 12
TOP_SCHWARM = 15
PAARE = 4000


def wasserfall(ax, beitraege, spalten, i, basis, wahr, vorher, titel):
    b = pd.Series(beitraege[i], index=spalten)
    gross = b.reindex(b.abs().sort_values(ascending=False).index[:TOP_LOKAL])
    rest = b.sum() - gross.sum()
    gross = pd.concat([gross, pd.Series({f"übrige {len(b)-TOP_LOKAL} Merkmale": rest})])
    gross = gross.iloc[::-1]
    ax.barh(
        range(len(gross)),
        gross.values,
        color=["#2ca02c" if v < 0 else "#d62728" for v in gross.values],
    )
    ax.set_yticks(range(len(gross)))
    ax.set_yticklabels([m[:30] for m in gross.index], fontsize=6.5)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("SHAP-Beitrag [Prozentpunkte]", fontsize=8)
    ax.set_title(
        f"{titel}\nBasis {basis:.1f} → Vorhersage {vorher:.1f} | "
        f"tatsächlich {wahr:.1f}",
        fontsize=9,
    )
    ax.grid(alpha=0.3, axis="x")
    ax.tick_params(axis="x", labelsize=7)


def beeswarm(ax, beitraege, X, w):
    namen = list(w.head(TOP_SCHWARM).Merkmal)[::-1]
    rng = np.random.default_rng(C.SEED)
    for j, name in enumerate(namen):
        s = beitraege[:, list(X.columns).index(name)]
        v = pd.to_numeric(X[name], errors="coerce").values.astype(float)
        endlich = np.isfinite(v)
        f = np.full(len(v), 0.5)
        if endlich.sum() > 1:
            lo, hi = np.nanpercentile(v[endlich], [5, 95])
            if hi > lo:
                f[endlich] = np.clip((v[endlich] - lo) / (hi - lo), 0, 1)
        ax.scatter(
            s,
            j + rng.uniform(-0.28, 0.28, len(s)),
            s=4,
            alpha=0.45,
            c=f,
            cmap="coolwarm",
            vmin=0,
            vmax=1,
            linewidths=0,
        )
    ax.set_yticks(range(len(namen)))
    ax.set_yticklabels([m[:32] for m in namen], fontsize=7)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("SHAP-Beitrag je Billet [Prozentpunkte]")
    ax.set_title(
        "A  Verteilung der Beiträge  (blau = niedriger Merkmalswert, " "rot = hoher)",
        fontsize=10,
    )
    ax.grid(alpha=0.3, axis="x")


def ueberlappung(ax, beitraege, k=10):
    rang = np.argsort(-np.abs(beitraege), axis=1)[:, :k]
    mengen = [set(r) for r in rang]
    rng = np.random.default_rng(C.SEED)
    n = len(mengen)
    a = rng.integers(0, n, PAARE)
    b = rng.integers(0, n, PAARE)
    j = np.array([len(mengen[x] & mengen[y]) / k for x, y in zip(a, b) if x != y])
    ax.hist(j, bins=np.arange(0, 1.05, 1 / k), color="#3182bd", edgecolor="white")
    ax.axvline(j.mean(), color="#d62728", lw=2)
    ax.annotate(
        f"Mittel {j.mean():.2f}",
        (j.mean(), ax.get_ylim()[1] * 0.9),
        color="#d62728",
        fontsize=9,
        ha="left",
        textcoords="offset points",
        xytext=(6, 0),
    )
    ax.set_xlabel(f"Anteil gemeinsamer Merkmale unter den Top {k}")
    ax.set_ylabel("Billetpaare")
    ax.set_title(
        f"B  Wie individuell sind die Erklärungen?  " f"({len(j)} zufällige Paare)",
        fontsize=10,
    )
    ax.grid(alpha=0.3, axis="y")
    return float(j.mean()), float(np.median(j))


def main():
    ds, X_tr, X_te = C.lade()
    modell, quelle = C.xgb_modell(X_tr, ds.y_train)
    beitraege, basis = C.shap_xgboost(modell, X_te)
    vorher = modell.predict(X_te)
    C.pruefe_additivitaet(beitraege, basis, vorher, "XGBoost")
    wahr = np.asarray(ds.y_test)
    rest = wahr - vorher
    w = C.globale_wichtigkeit(beitraege, X_te.columns)

    gut = np.abs(rest) < np.percentile(np.abs(rest), 25)
    faelle = [
        (
            int(np.argmax(np.where(gut, wahr, -np.inf))),
            "A  hoher Ausschuss, gut getroffen",
        ),
        (
            int(np.argmin(np.where(gut, wahr, np.inf))),
            "B  niedriger Ausschuss, gut getroffen",
        ),
        (int(np.argmax(rest)), "C  stärkste Unterschätzung"),
        (int(np.argmin(rest)), "D  stärkste Überschätzung"),
    ]

    pdf = C.pdf_oeffnen(C.mit_ebene("02_lokal.pdf"))
    fig, ax = C.seite("Billet-individuelle SHAP-Erklärungen (XGBoost, Test-Set)", 2, 2)
    for (i, titel), a in zip(faelle, ax):
        wasserfall(a, beitraege, X_te.columns, i, basis, wahr[i], vorher[i], titel)
    C.schliesse(pdf, fig, C.fusszeile(ds, X_te, quelle))

    fig2, ax2 = C.seite(
        "Wie individuell die Erklärungen sind", 1, 2, groesse=(11.7, 8.3)
    )
    beeswarm(ax2[0], beitraege, X_te, w)
    mittel, median = ueberlappung(ax2[1], beitraege)
    C.schliesse(pdf, fig2, C.fusszeile(ds, X_te, quelle))

    C.textseite(
        pdf,
        "Lesehinweise und Vorbehalte",
        [
            "Auswahl der Einzelfälle. Nicht zufällig gezogen, sondern nach vier "
            "Kriterien bestimmt: die beiden gut getroffenen Fälle zeigen, worauf "
            "das Modell seine Vorhersage stützt, die beiden Fehlschläge, wo diese "
            "Stützung versagt. Als gut getroffen gilt ein Residuum im untersten "
            "Viertel der Beträge.",
            f"Individualität. Zwei zufällig gepaarte Billets teilen im Mittel "
            f"{mittel:.0%} ihrer zehn wichtigsten Merkmale (Median {median:.0%}). "
            f"Je näher dieser Wert an eins liegt, desto weniger trägt eine lokale "
            f"Erklärung über die globale hinaus — die Vorhersagen unterschieden "
            f"sich dann nur in der Stärke, nicht in der Begründung.",
            "Lesart des Beeswarms. Jeder Punkt ist ein Billet. Die Lage auf der "
            "x-Achse ist der Beitrag dieses Merkmals zu diesem Billet, die Farbe "
            "die Ausprägung des Merkmals (blau niedrig, rot hoch). Ein Merkmal, "
            "dessen rote Punkte rechts und blaue links liegen, wirkt monoton; "
            "vermischte Farben deuten auf Wechselwirkungen mit anderen Merkmalen.",
            "Vorbehalt. SHAP-Werte sind Beiträge zur Vorhersage des Modells, keine "
            "Wirkungen im Prozess. Ein Beitrag von +3 Prozentpunkten bedeutet, dass "
            "das Modell für dieses Billet um 3 Punkte höher schätzt als im Mittel — "
            "nicht, dass eine Änderung der Größe den Ausschuss um 3 Punkte senken "
            "würde. Für kausale Aussagen wäre ein Eingriff nötig, kein "
            "Beobachtungsmodell.",
        ],
    )
    pdf.close()

    p = C.speichere(
        C.mit_ebene("02_lokal.json"),
        {
            "modell": "XGBoost",
            "basiswert": basis,
            "ueberlappung_top10_mittel": mittel,
            "ueberlappung_top10_median": median,
            "faelle": [
                {
                    "index": i,
                    "rolle": t,
                    "y_wahr": float(wahr[i]),
                    "y_vorhergesagt": float(vorher[i]),
                    "residuum": float(rest[i]),
                    "beitraege": [
                        {"merkmal": str(X_te.columns[j]), "beitrag": float(beitraege[i, j]),
                         "wert": str(ds.X_test_raw.reset_index(drop=True).iloc[i].get(X_te.columns[j], ""))}
                        for j in np.argsort(-np.abs(beitraege[i]))[:8]
                    ],
                }
                for i, t in faelle
            ],
        },
    )
    print(f"geschrieben: {C.OUT_DIR / C.mit_ebene('02_lokal.pdf')}\n             {p}\n")
    print(
        f"Überlappung der Top-10 zwischen Billets: Mittel {mittel:.3f}, "
        f"Median {median:.3f}"
    )
    for i, t in faelle:
        print(
            f"  {t:<38} y={wahr[i]:6.2f}  ŷ={vorher[i]:6.2f}  " f"Rest {rest[i]:+6.2f}"
        )


if __name__ == "__main__":
    main()
