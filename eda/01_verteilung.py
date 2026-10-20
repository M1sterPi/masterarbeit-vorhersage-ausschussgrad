"""01 — Verteilung des Ausschussgrads y.

In der Arbeit: Abschnitt 4.1.3 (Rohbestand, negative Werte, Totalausfälle) und
4.2.1 (Lage und Streuung des Ausschussgrads); Werte für Tabelle 1.

Verteilung, Wirkung des y-Filters, Kategorien und Glastypen.
Ausgabe: reports/eda/01_verteilung.pdf.

    python eda/01_verteilung.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import _common as C
from data_prep import SP  # noqa: E402


def main():
    ds = C.lade(alle_arten=True)
    X, y, df = C.daten_teil(ds)
    fuss = C.kopfzeile(ds, alle_arten=True)

    roh = pd.read_csv(
        C.CSV_PATH, usecols=["batch_id", "y", SP["masse_soll"]], low_memory=False
    )
    n_soll = C.als_zahl(roh[SP["masse_soll"]]).nunique()

    med, mit = y.median(), y.mean()
    lo, hi = C.bootstrap_ci(y.to_numpy(), np.median)
    weg = roh[~roh.y.between(C.Y_MIN, C.Y_MAX) & roh.y.notna()]
    gruppen = {}
    for spalte in [SP["kategorie"], "Glastyp"]:
        if spalte not in df.columns:
            continue
        vorh = df[spalte].notna()
        r = C.var_anteil_mit_null(y[vorh], df.loc[vorh, spalte], n_perm=200)
        yv = y[vorh]
        r["sd_gesamt"] = yv.std()
        r["sd_innen"] = (yv - yv.groupby(df.loc[vorh, spalte]).transform("mean")).std()
        r["fehlend"] = int((~vorh).sum())
        r["g"] = df[spalte].fillna("(kein Wert)")
        gruppen[spalte] = r
    n_je = df.groupby(SP["auftrag"]).size()
    dev = C.auftragsabweichung(y, df[SP["auftrag"]])

    def sd_satz(spalte):
        r = gruppen[spalte]
        rueck = 100 * (1 - r["sd_innen"] / r["sd_gesamt"])
        ohne = f", {r['fehlend']} Billets ohne Wert nicht mitgerechnet" if r["fehlend"] else ""
        return (
            f"{spalte} ({r['gruppen']} Stufen{ohne}): "
            f"erklärter Varianzanteil {r['beobachtet']:.3f} bei Nullerwartung "
            f"{r['null_mittel']:.3f} ± {r['null_std']:.3f} (Überschuss "
            f"{r['ueberschuss']:+.3f}); die Standardabweichung sinkt dabei nur von "
            f"{r['sd_gesamt']:.1f} auf {r['sd_innen']:.1f} Prozentpunkte, also um "
            f"{rueck:.1f} %."
        )

    pdf = C.pdf_oeffnen("01_verteilung.pdf")

    C.textseite(
        pdf,
        "01 — Verteilung der Zielgrösse",
        [
            "y = (Sollmasse - Istmasse) / Sollmasse * 100, also der Masseverlust in Prozent.",
            f"Rohdaten: {len(roh)} Billets (min {roh.y.min():.1f} %, max "
            f"{roh.y.max():.1f} %). Nach dem Filter {C.y_grenzen_text()} verbleiben "
            f"{int(roh.y.between(C.Y_MIN, C.Y_MAX).sum())}; entfernt werden "
            f"{len(weg)} ({100 * len(weg) / len(roh):.1f} %), davon "
            f"{int((weg.y < C.Y_MIN).sum())} unterhalb und "
            f"{int((weg.y > C.Y_MAX).sum())} oberhalb.",
            f"Nach dem Filter: Median {med:.1f} % (95-%-Bootstrap-Intervall "
            f"[{lo:.1f}; {hi:.1f}]), Mittel {mit:.1f} %, SD {y.std():.1f} Prozentpunkte.",
            "Der Filter entfernt drei GRUNDVERSCHIEDENE Dinge und behandelt sie gleich:",
            f"  - Erfassungsfehler: das Minimum liegt bei {roh.y.min():.1f} %, d. h. "
            "Die Istmasse war ein Vielfaches der Sollmasse. Physikalisch unmöglich.",
            "  - leichte Überfüllung (kleine negative Werte).",
            f"  - den Bereich 0 ≤ y < {C.Y_MIN:g}: y == 0 genau "
            f"{int((roh.y == 0).sum())} Zeilen, 0 < y < {C.Y_MIN:g} weitere "
            f"{int(((roh.y > 0) & (roh.y < C.Y_MIN)).sum())}. Warum diese Zeilen "
            "fallen, ist die alleinige Frage von 08_y_untergrenze.pdf.",
            "Das ist eine begründungspflichtige Entscheidung, keine Selbstverständlichkeit: "
            "die untere Grenze unterstellt, dass Übererfüllung nicht vorkommt.",
            f"Die Sollmasse hat nur {n_soll} verschiedene Werte.",
            "ZUR KENNZAHL eta²: der erklärte Varianzanteil ist ein Anteil an der "
            "VARIANZ, nicht an der Streuung. Varianz ist quadriert, und die Wurzel "
            "dämpft stark. Beim Formulieren in der Arbeit ist der Unterschied "
            "wesentlich — 'erklärt x % der Varianz' klingt nach deutlich mehr, als in "
            "Prozentpunkten von y ankommt.",
            *[sd_satz(sp) for sp in gruppen],
            f"Aufträge: {len(n_je)}, Median {n_je.median():.0f} Billets je Auftrag, "
            f"maximal {n_je.max()}. Einzelbillet-Aufträge: {int((n_je == 1).sum())} "
            f"({100 * (n_je == 1).mean():.0f} %). Billets mit Geschwistern: "
            f"{100 * (n_je[n_je > 1].sum() / n_je.sum()):.0f} %. Die SD von y sinkt "
            f"von {y.std():.1f} auf {dev.std():.1f} Prozentpunkte, wenn man je "
            "Auftrag den Mittelwert abzieht.",
        ],
    )

    fig, ax = C.seite(pdf, "Zielgrösse y — Verteilung und Filterwirkung", 2, 2)

    ax[0].hist(roh.y.clip(-50, 120), bins=120, color="steelblue")
    for g in (C.Y_MIN, C.Y_MAX):
        ax[0].axvline(g, color="crimson", ls="--", lw=1)
    ax[0].set_title("y roh (auf [−50; 120] begrenzt)", fontsize=10)
    ax[0].set_xlabel("y [%]")
    ax[0].set_ylabel("Anzahl Billets")

    ax[1].hist(y, bins=80, color="steelblue")
    ax[1].set_title(f"y nach Filter {C.y_grenzen_text()}", fontsize=10)
    ax[1].set_xlabel("y [%]")
    ax[1].set_ylabel("Anzahl Billets")
    ax[1].axvline(med, color="crimson", lw=1, label="Median")
    ax[1].legend(fontsize=8)

    s = np.sort(y.to_numpy())
    ax[2].plot(s, np.arange(1, len(s) + 1) / len(s), color="steelblue")
    ax[2].set_title("Empirische Verteilungsfunktion", fontsize=10)
    ax[2].set_xlabel("y [%]")
    ax[2].set_ylabel("Anteil ≤ y")
    ax[2].grid(alpha=0.3)

    ax[3].hist(weg.y.clip(-40, 5), bins=60, color="indianred")
    ax[3].set_title("vom Filter entfernt (auf [−40; 5] begrenzt)", fontsize=10)
    ax[3].set_xlabel("y [%]")
    ax[3].set_ylabel("Anzahl Billets")
    C.schliesse(pdf, fig, fuss)

    fig, ax = C.seite(pdf, "Rechtfertigen die Daten eine Eingrenzung?", 2, 2)

    for i, spalte in enumerate([SP["kategorie"], "Glastyp"]):
        if spalte not in gruppen:
            C.leer(ax[i])
            continue
        g = gruppen[spalte]["g"]
        stufen = g.value_counts()
        reihenfolge = y.groupby(g).median().sort_values().index.tolist()
        daten = [y[g == s].to_numpy() for s in reihenfolge]
        bp = ax[i].boxplot(daten, vert=True, showfliers=False, patch_artist=True)
        for kasten, s in zip(bp["boxes"], reihenfolge):
            kasten.set_facecolor("lightsteelblue" if stufen[s] >= 20 else "#eef2f6")
        ax[i].set_xticklabels(
            [f"{s}\n(n = {int(stufen[s])})" for s in reihenfolge],
            fontsize=7,
            rotation=40,
            ha="right",
        )
        ax[i].set_ylabel("y [%]")
        ax[i].set_title(f"y je {spalte}", fontsize=10)
        ax[i].grid(alpha=0.3, axis="y")

    ax[2].hist(n_je, bins=range(1, int(n_je.max()) + 2), color="steelblue")
    ax[2].set_title("Billets je Auftrag", fontsize=10)
    ax[2].set_xlabel("Billets im Auftrag")
    ax[2].set_ylabel("Anzahl Aufträge")

    ax[3].hist(y - y.mean(), bins=70, alpha=0.55, label="y − Gesamtmittel", color="grey")
    ax[3].hist(dev, bins=70, alpha=0.75, label="y − Auftragsmittel", color="steelblue")
    ax[3].legend(fontsize=8)
    ax[3].set_title("Streuung gesamt und innerhalb des Auftrags", fontsize=10)
    ax[3].set_xlabel("Abweichung [Prozentpunkte]")
    ax[3].set_ylabel("Anzahl Billets")
    C.schliesse(pdf, fig, fuss)

    pdf.close()
    print(f"geschrieben: {C.OUT_DIR / '01_verteilung.pdf'}")
    print(f"  SD y {y.std():.2f} -> SD innerhalb Auftrag {dev.std():.2f}")


if __name__ == "__main__":
    main()
