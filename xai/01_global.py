"""01 — Globale Merkmalswichtigkeit.

In der Arbeit: nicht berichtet; SHAP auf der Testmenge; die Rangliste in
Abschnitt 7.2 kommt aus xai/07.

SHAP-Werte von XGBoost auf dem Test-Set.
Ausgaben unter reports/xai/, runs/xai/ und reports/xai/tabellen/.

    TEST_LEVEL=<billet|auftrag|artikel> python xai/01_global.py
"""

from __future__ import annotations

import numpy as np

import _common as C

TOP = 20
FARBE = {"Stammdaten": "#3182bd", "Prozess": "#e6550d"}


def feld_a(ax, w):
    t = w.head(TOP).iloc[::-1]
    farben = [FARBE[C.herkunft(m)] for m in t.Merkmal]
    ax.barh(range(len(t)), t.mittlerer_betrag, color=farben)
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels([m[:34] for m in t.Merkmal], fontsize=7)
    ax.set_xlabel("mittlerer |SHAP| [Prozentpunkte Ausschuss]")
    ax.set_title(f"A  Die {TOP} wichtigsten Merkmale", fontsize=10)
    ax.grid(alpha=0.3, axis="x")
    from matplotlib.patches import Patch

    ax.legend(
        handles=[Patch(color=FARBE[k], label=k) for k in FARBE],
        fontsize=7.5,
        loc="lower right",
    )


def feld_b(ax, w):
    w = w.copy()
    w["Herkunft"] = w.Merkmal.map(C.herkunft)
    g = w.groupby("Herkunft").agg(
        summe=("mittlerer_betrag", "sum"), anzahl=("Merkmal", "size")
    )
    g["anteil"] = g.summe / g.summe.sum()
    x = np.arange(len(g))
    ax.bar(x, g.anteil, color=[FARBE[i] for i in g.index])
    for i, (_, r) in enumerate(g.iterrows()):
        ax.annotate(
            f"{r.anteil:.1%}\n{int(r.anzahl)} Merkmale",
            (i, r.anteil),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(g.index)
    ax.set_ylim(0, max(g.anteil) * 1.35)
    ax.set_ylabel("Anteil an der gesamten |SHAP|-Summe")
    ax.set_title("B  Woher die erklärte Wirkung stammt", fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    return g


def feld_c(ax, w):
    k = np.cumsum(w.mittlerer_betrag.values)
    k = k / k[-1]
    ax.plot(np.arange(1, len(k) + 1), k, color="black", lw=1.6)
    for anteil in (0.5, 0.8, 0.9):
        n = int(np.searchsorted(k, anteil) + 1)
        ax.axhline(anteil, color="#999999", ls=":", lw=1)
        ax.plot([n], [anteil], "o", ms=7, color="#d62728")
        ax.annotate(
            f"{anteil:.0%} bei {n} Merkmalen",
            (n, anteil),
            textcoords="offset points",
            xytext=(8, -12),
            fontsize=8,
            color="#d62728",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Merkmale, nach Wichtigkeit sortiert (log)")
    ax.set_ylabel("kumulierter Anteil")
    ax.set_title("C  Wie stark sich die Wirkung konzentriert", fontsize=10)
    ax.grid(alpha=0.3)


def feld_d(ax, w):
    t = w.head(TOP).iloc[::-1]
    ax.barh(
        range(len(t)),
        t.mittlerer_beitrag,
        color=["#2ca02c" if v < 0 else "#d62728" for v in t.mittlerer_beitrag],
    )
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels([m[:34] for m in t.Merkmal], fontsize=7)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("mittlerer SHAP-Beitrag (vorzeichenbehaftet)")
    ax.set_title(
        "D  Richtung der Wirkung  (grün senkt, rot erhöht den Ausschuss)", fontsize=10
    )
    ax.grid(alpha=0.3, axis="x")


def main():
    ds, X_tr, X_te = C.lade()
    modell, quelle = C.xgb_modell(X_tr, ds.y_train)
    print(f"Modell: XGBoost, {quelle}, {X_tr.shape[1]} Merkmale")

    beitraege, basis = C.shap_xgboost(modell, X_te)
    C.pruefe_additivitaet(beitraege, basis, modell.predict(X_te), "XGBoost")
    w = C.globale_wichtigkeit(beitraege, X_te.columns)

    pdf = C.pdf_oeffnen(C.mit_ebene("01_global.pdf"))
    fig, ax = C.seite(
        "Globale Merkmalswichtigkeit aus exakten SHAP-Werten " "(XGBoost, Test-Set)",
        2,
        2,
    )
    feld_a(ax[0], w)
    g = feld_b(ax[1], w)
    feld_c(ax[2], w)
    feld_d(ax[3], w)
    C.schliesse(pdf, fig, C.fusszeile(ds, X_te, quelle))

    k = np.cumsum(w.mittlerer_betrag.values)
    k = k / k[-1]
    n50, n80 = int(np.searchsorted(k, 0.5) + 1), int(np.searchsorted(k, 0.8) + 1)
    C.textseite(
        pdf,
        "Lesehinweise und Vorbehalte",
        [
            "Berechnung. Exakte TreeSHAP-Werte über die native Implementierung von "
            "XGBoost (pred_contribs=True), nicht über die shap-Bibliothek: shap "
            "0.49.1 kann XGBoost 3.1.1 nicht laden, da base_score als JSON-Array "
            "serialisiert wird. Die Additivität — Summe der Beiträge plus Basiswert "
            "gleich Vorhersage — wird bei jedem Lauf nachgerechnet.",
            f"Bezugsmenge. Gerechnet wird auf dem eingefrorenen Test-Set "
            f"({X_te.shape[0]} Billets), also auf Daten, die weder in das Training "
            f"noch in die Wahl der Hyperparameter eingeflossen sind. SHAP erklärt "
            f"das Modell, nicht den Prozess: Ein hoher Beitrag heisst, dass das "
            f"Modell die Größe nutzt, nicht dass sie den Ausschuss verursacht.",
            f"Konzentration. Die Hälfte der gesamten erklärten Wirkung entfällt auf "
            f"{n50} Merkmale, 80 Prozent auf {n80} von {X_te.shape[1]}. "
            f"Der Basiswert des Modells liegt bei {basis:.2f} Prozentpunkten.",
            "Herkunft. "
            + "; ".join(
                f"{i}: {r.anteil:.1%} der |SHAP|-Summe aus {int(r.anzahl)} "
                f"Merkmalen"
                for i, r in g.iterrows()
            )
            + ". Diese Aufteilung ist die SHAP-Sicht auf FF2 und ergänzt die "
            "Ablationsrechnung, ersetzt sie aber nicht: Sie misst die Nutzung "
            "durch das Modell, nicht den eigenständigen Beitrag der Quelle.",
            "Vorbehalt zur Gruppenstruktur. Nach den Befunden aus Kapitel 6 liegt "
            "das vorhersagbare Signal weit überwiegend auf Auftrags- und "
            "Artikelebene. Merkmale, die das Erzeugnis beschreiben, erhalten hier "
            "deshalb erwartungsgemäss hohe Beiträge — das bestätigt jene Befunde "
            "und ist kein eigenständiger Hinweis auf einen Prozesszusammenhang.",
        ],
    )
    pdf.close()

    p = C.speichere(
        C.mit_ebene("01_global.json"),
        {
            "modell": "XGBoost",
            "params_quelle": quelle,
            "n_test": int(X_te.shape[0]),
            "n_features": int(X_te.shape[1]),
            "basiswert": basis,
            "n_fuer_50_prozent": n50,
            "n_fuer_80_prozent": n80,
            "herkunft": {
                i: {"anteil": float(r.anteil), "anzahl": int(r.anzahl)}
                for i, r in g.iterrows()
            },
            "top50": w.head(50).to_dict("records"),
            "rangliste_csv": C.mit_ebene("01_global.csv"),
        },
    )
    pcsv = C.speichere_rangliste(C.mit_ebene("01_global.csv"), w)
    print(f"geschrieben: {C.OUT_DIR / C.mit_ebene('01_global.pdf')}\n             {p}\n             {pcsv}\n")
    print(
        f"Basiswert {basis:.2f}   50 % der Wirkung in {n50} Merkmalen, "
        f"80 % in {n80}\n"
    )
    print("Top 15:")
    for i, r in w.head(15).iterrows():
        print(
            f"  {i+1:>2}. {r.Merkmal:<40}{r.mittlerer_betrag:>8.3f}"
            f"  ({C.herkunft(r.Merkmal)})"
        )


if __name__ == "__main__":
    main()
