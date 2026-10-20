"""09 — Redundanz.

In der Arbeit: nicht berichtet; Korrelationsgruppen unter den wichtigsten
Merkmalen.

Korrelationsgruppen unter den wichtigsten Merkmalen und gruppierte SHAP-Werte.
Ausgaben unter reports/xai/, runs/xai/ und reports/xai/tabellen/.

    TEST_LEVEL=<billet|auftrag|artikel> python xai/09_redundanz.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform

import _common as C

TOP = 50
GRENZE = 0.30
FARBE_HERKUNFT = {"Stammdaten": "#3182bd", "Prozess": "#e6550d"}


def shap_rangliste() -> pd.DataFrame:
    p = C.TAB_DIR / C.mit_ebene("07_kreuzvalidierung_xgboost.csv")
    if not p.exists():
        raise SystemExit(
            f"{p.name} fehlt. Erst 07_kreuzvalidierung.py fuer TEST_LEVEL="
            f"{C.D.TEST_LEVEL} laufen lassen."
        )
    return pd.read_csv(p)


def korrelationen(X: pd.DataFrame, merkmale: list[str]) -> pd.DataFrame:
    """|Spearman| zwischen den Merkmalen; konstante Spalten bekommen 0."""
    R = X[merkmale].corr(method="spearman").abs()
    return R.fillna(0.0)


def gruppieren(R: pd.DataFrame):
    """Hierarchische Gruppierung ueber 1 - |rho|."""
    D = 1.0 - R.values
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2
    Z = linkage(squareform(D, checks=False), method="average")
    gruppe = fcluster(Z, t=GRENZE, criterion="distance")
    return Z, pd.Series(gruppe, index=R.index, name="gruppe")


def gruppentabelle(rang: pd.DataFrame, gruppe: pd.Series, R: pd.DataFrame):
    """Je Gruppe: Summe der |SHAP|, staerkstes Mitglied, Mitgliederzahl."""
    w = rang.set_index("Merkmal").Mittel
    zeilen = []
    for g, mitglieder in gruppe.groupby(gruppe):
        namen = list(mitglieder.index)
        werte = w.reindex(namen).fillna(0.0)
        fuehrend = werte.idxmax()
        innen = R.loc[namen, namen].values
        mittel_rho = (
            float((innen.sum() - len(namen)) / (len(namen) * (len(namen) - 1)))
            if len(namen) > 1
            else float("nan")
        )
        zeilen.append(
            {
                "gruppe": int(g),
                "n": len(namen),
                "summe": float(werte.sum()),
                "maximum": float(werte.max()),
                "fuehrend": fuehrend,
                "mittleres_rho": mittel_rho,
                "mitglieder": namen,
            }
        )
    return pd.DataFrame(zeilen).sort_values("summe", ascending=False).reset_index(drop=True)


def seite_struktur(pdf, Z, R, gruppe, fuss):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11.7, 8.3))
    fig.suptitle(
        f"Korrelationsstruktur der Top {TOP} — Ebene {C.D.TEST_LEVEL}",
        fontsize=13,
        y=0.985,
    )
    gs = fig.add_gridspec(
        2, 1, height_ratios=[0.85, 1.35], left=0.19, right=0.94, top=0.92, bottom=0.19,
        hspace=0.62,
    )

    axd = fig.add_subplot(gs[0])
    dd = dendrogram(
        Z,
        labels=list(R.index),
        ax=axd,
        color_threshold=GRENZE,
        above_threshold_color="#999999",
        leaf_font_size=4.6,
    )
    axd.axhline(GRENZE, color="#d62728", ls="--", lw=1)
    axd.set_ylabel("1 − |ρ|", fontsize=8)
    axd.tick_params(axis="y", labelsize=7)
    ordnung = [R.index[i] for i in dd["leaves"]]
    for t, m in zip(axd.get_xmajorticklabels(), ordnung):
        t.set_color(FARBE_HERKUNFT[C.herkunft(m)])
        t.set_text(m[:18])
    axd.set_xticklabels([m[:18] for m in ordnung], fontsize=4.6)
    for t, m in zip(axd.get_xmajorticklabels(), ordnung):
        t.set_color(FARBE_HERKUNFT[C.herkunft(m)])
    axd.set_title(
        f"A  Gruppierung über 1 − |Spearman ρ| (rot: Schnitt bei {GRENZE:.2f}, "
        f"also |ρ| ≥ {1 - GRENZE:.2f}) — blau Stammdaten, orange Prozess",
        fontsize=9.5,
    )
    M = R.loc[ordnung, ordnung]
    axh = fig.add_subplot(gs[1])
    bild = axh.imshow(M.values, vmin=0, vmax=1, cmap="magma_r", aspect="auto")
    grenzen = np.where(
        np.array([gruppe[a] for a in ordnung[:-1]])
        != np.array([gruppe[b] for b in ordnung[1:]])
    )[0]
    for g in grenzen:
        axh.axhline(g + 0.5, color="#00b0f0", lw=0.7)
        axh.axvline(g + 0.5, color="#00b0f0", lw=0.7)
    axh.set_xticks(range(len(ordnung)))
    axh.set_yticks(range(len(ordnung)))
    axh.set_xticklabels([m[:20] for m in ordnung], rotation=90, fontsize=4.2)
    axh.set_yticklabels([m[:24] for m in ordnung], fontsize=4.6)
    axh.tick_params(length=0)
    axh.set_title(
        "B  |ρ| in der Reihenfolge des Dendrogramms, Gruppengrenzen blau",
        fontsize=9.5,
        pad=10,
    )
    cb = fig.colorbar(bild, ax=axh, fraction=0.02, pad=0.012)
    cb.set_label("|Spearman ρ|", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)

    fig.text(0.5, 0.015, fuss, ha="center", fontsize=7.5, style="italic")
    pdf.savefig(fig)
    plt.close(fig)
    return ordnung


def feld_c(ax, G, n_zeigen=14):
    t = G.head(n_zeigen).iloc[::-1]
    y = np.arange(len(t))
    ax.barh(y, t.summe, color="#6baed6", alpha=0.9, label="Summe der Gruppe")
    ax.barh(y, t.maximum, height=0.45, color="#08519c", label="stärkstes Mitglied")
    for yi, r in zip(y, t.itertuples()):
        ax.annotate(
            f"n={r.n}",
            (r.summe, yi),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=6.5,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.fuehrend[:26]} …" if r.n > 1 else r.fuehrend[:28]
                        for r in t.itertuples()], fontsize=6.5)
    ax.set_xlabel("mittlerer |SHAP| [PP]")
    ax.set_title(
        "C  Gruppierte Zuschreibung\nheller Überhang = Beitrag der übrigen "
        "Gruppenmitglieder",
        fontsize=9,
    )
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3, axis="x")


def feld_d(ax, rang, gruppe, R, G, n_zeigen=20):
    w = rang.set_index("Merkmal").Mittel
    summe = {int(r.gruppe): r.summe for r in G.itertuples()}
    zeilen = []
    for m in rang.Merkmal.head(n_zeigen):
        if m not in gruppe.index:
            continue
        andere = [a for a in R.index if a != m]
        zeilen.append(
            {
                "Merkmal": m,
                "max_rho": float(R.loc[m, andere].max()) if andere else 0.0,
                "partner": R.loc[m, andere].idxmax() if andere else "",
                "anteil": float(w[m] / summe[int(gruppe[m])]) if summe[int(gruppe[m])] else 1.0,
                "n_gruppe": int((gruppe == gruppe[m]).sum()),
            }
        )
    t = pd.DataFrame(zeilen).iloc[::-1]
    y = np.arange(len(t))
    ax.barh(y, t.max_rho, color="#bdbdbd", alpha=0.9, label="höchstes |ρ| zu einem anderen Merkmal")
    ax.scatter(t.anteil, y, s=26, color="#d62728", zorder=3, label="eigener Anteil an der Gruppensumme")
    ax.axvline(1 - GRENZE, color="#d62728", ls=":", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([m[:30] for m in t.Merkmal], fontsize=6.5)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("|ρ| bzw. Anteil")
    ax.set_title(
        "D  Steht das Merkmal für sich?\nkleiner Anteil = einer von mehreren "
        "Stellvertretern",
        fontsize=9,
    )
    ax.legend(fontsize=6.5, loc="lower left")
    ax.grid(alpha=0.3, axis="x")
    return t.iloc[::-1]


def main():
    print(f"Ebene: TEST_LEVEL={C.D.TEST_LEVEL}")
    rang = shap_rangliste()
    ds = C.D.load_split(verbose=False)

    X_all, _, _ = C.D.process_features(ds.X_raw, ds.X_raw)
    merkmale = [m for m in rang.Merkmal.head(TOP) if m in X_all.columns]
    print(f"  {len(merkmale)} der Top {TOP} im kodierten Datensatz vorhanden")

    R = korrelationen(X_all, merkmale)
    Z, gruppe = gruppieren(R)
    G = gruppentabelle(rang, gruppe, R)
    mehrere = G[G.n > 1]
    print(f"  {gruppe.nunique()} Gruppen, davon {len(mehrere)} mit mehr als einem Merkmal")

    fuss = (
        f"{C.D.CSV_PATH.name} | Korrelation über alle {len(X_all)} Billets | "
        f"|SHAP| aus 07 (kreuzvalidiert, Ebene {C.D.TEST_LEVEL}) | "
        f"Gruppierung: average linkage auf 1 − |ρ|, Schnitt {GRENZE:.2f}"
    )

    pdf = C.pdf_oeffnen(C.mit_ebene("09_redundanz.pdf"))
    ordnung = seite_struktur(pdf, Z, R, gruppe, fuss)

    fig, ax = C.seite(
        f"Was die Gruppierung an der Rangliste ändert — Ebene {C.D.TEST_LEVEL}", 1, 2
    )
    feld_c(ax[0], G)
    einzeln = feld_d(ax[1], rang, gruppe, R, G)
    C.schliesse(pdf, fig, fuss)

    allein = einzeln[einzeln.anteil > 0.95].Merkmal.tolist()
    geteilt = einzeln[einzeln.anteil < 0.5].Merkmal.tolist()
    groesste = G.iloc[0]
    C.textseite(
        pdf,
        "Lesehinweise und Vorbehalte",
        [
            f"Vorgehen. Die {len(merkmale)} Merkmale der kreuzvalidierten "
            f"SHAP-Spitzengruppe aus 07 werden ueber 1 − |Spearman ρ| hierarchisch "
            f"gruppiert (average linkage, Schnitt bei {GRENZE:.2f}, entspricht "
            f"|ρ| ≥ {1 - GRENZE:.2f} im Mittel). Die Korrelation wird auf allen "
            f"{len(X_all)} Billets nach der Kodierung gerechnet; es wird hier kein "
            f"Modell bewertet, sondern die Struktur der Merkmale beschrieben.",
            f"Ergebnis der Gruppierung. {gruppe.nunique()} Gruppen, davon "
            f"{len(mehrere)} mit mehr als einem Mitglied. Die staerkste Gruppe wird "
            f"von {groesste.fuehrend} angefuehrt, umfasst {groesste.n} Merkmale und "
            f"traegt zusammen {groesste.summe:.3f} PP gegenueber {groesste.maximum:.3f} "
            f"PP des staerksten Einzelmitglieds. Die Differenz ist genau der Betrag, "
            f"den eine Einzelrangliste auf mehrere Zeilen verteilt und damit "
            f"unsichtbar macht.",
            f"Wer steht fuer sich. Anteil an der eigenen Gruppensumme ueber 95 "
            f"Prozent — also praktisch ohne Stellvertreter — haben "
            f"{len(allein)} der geprueften Merkmale"
            + (f": {', '.join(allein[:8])}." if allein else ".")
            + f" Unter 50 Prozent liegen {len(geteilt)}"
            + (f": {', '.join(geteilt[:8])}." if geteilt else "."),
            "Wie das mit 08 zusammenhaengt. Ein Merkmal, das sich eine Gruppe mit "
            "anderen teilt, ist der typische Fall fuer den Unterschied zwischen "
            "Permutation und LOCO: entfernt man es, uebernimmt ein Gruppenmitglied "
            "seine Rolle, und die Guete bleibt. Die Gruppensumme hier und der "
            "LOCO-Wert dort messen zwei Seiten derselben Sache.",
            "Vorbehalt. Spearman erfasst nur monotone paarweise Zusammenhaenge. "
            "Merkmale koennen sich auch ohne hohe paarweise Korrelation vertreten, "
            "etwa wenn drei zusammen eine vierte Groesse bestimmen; solche Faelle "
            "findet dieses Verfahren nicht. Ausserdem sind die kategorialen "
            "Merkmale ordinal kodiert, ihre Korrelationen haengen damit an der "
            "willkuerlichen Reihenfolge der Kategorien und sind nur bei wenigen "
            "Auspraegungen belastbar.",
        ],
    )
    pdf.close()

    tab = (
        rang[["Merkmal", "Mittel"]]
        .rename(columns={"Mittel": "shap_mittel"})
        .merge(gruppe.rename("gruppe").reset_index().rename(columns={"index": "Merkmal"}),
               on="Merkmal", how="inner")
        .merge(einzeln[["Merkmal", "max_rho", "partner", "anteil", "n_gruppe"]],
               on="Merkmal", how="left")
        .merge(G[["gruppe", "summe", "fuehrend", "mittleres_rho"]].rename(
            columns={"summe": "gruppe_summe", "fuehrend": "gruppe_fuehrend"}),
            on="gruppe", how="left")
    )
    pcsv = C.speichere_rangliste(C.mit_ebene("09_redundanz.csv"), tab, "shap_mittel")

    p = C.speichere(
        C.mit_ebene("09_redundanz.json"),
        {
            "test_level": C.D.TEST_LEVEL,
            "n_zeilen": int(len(X_all)),
            "n_merkmale": len(merkmale),
            "grenze": GRENZE,
            "n_gruppen": int(gruppe.nunique()),
            "n_gruppen_mehrgliedrig": int(len(mehrere)),
            "gruppen": G.to_dict("records"),
            "je_merkmal": einzeln.to_dict("records"),
            "reihenfolge_dendrogramm": ordnung,
            "allein_stehend": allein,
            "stark_geteilt": geteilt,
            "rangliste_csv": pcsv.name,
        },
    )
    print(f"\ngeschrieben: {C.OUT_DIR / C.mit_ebene('09_redundanz.pdf')}\n             {p}"
          f"\n             {pcsv}\n")
    for r in G.head(8).itertuples():
        print(f"  Gruppe {r.gruppe:>3} n={r.n:<3} Summe {r.summe:>7.3f}  "
              f"max {r.maximum:>7.3f}  {r.fuehrend}")


if __name__ == "__main__":
    main()
