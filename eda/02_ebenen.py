"""02 — Auf welcher Ebene variiert y?.

In der Arbeit: Tabelle 2 (Varianzanteil je Gruppierung gegen das Nullmodell,
Reststreuung) und der Text dazu in Abschnitt 4.2.5.

Varianzanteile je Gruppierung gegen ein Permutations-Nullmodell.
Ausgabe: reports/eda/02_ebenen.pdf.

    python eda/02_ebenen.py
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
    pdf = C.pdf_oeffnen("02_ebenen.pdf")

    auftrag = df[SP["auftrag"]]
    dev = C.auftragsabweichung(y, auftrag)
    am = y.groupby(auftrag).transform("mean")

    ergebnisse = []
    for spalte, name in C.GRUPPIERUNGEN:
        if spalte not in df.columns:
            continue
        r = C.var_anteil_mit_null(y, df[spalte], n_perm=200)
        if np.isnan(r["beobachtet"]):
            continue
        vorh = df[spalte].notna()
        yv = y[vorh]
        r["omega2"] = C.omega2(y, df[spalte])
        r["sd_innen"] = (yv - yv.groupby(df.loc[vorh, spalte]).transform("mean")).std()
        r["kurz"] = name
        r["name"] = f"{name} ({r['gruppen']})"
        ergebnisse.append(r)
    E = pd.DataFrame(ergebnisse)

    r2_mittel = 1 - ((y - am) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    art = df[SP["artikel"]]
    ok = art.notna() & auftrag.notna()
    yy, aa, ff = y[ok], art[ok].astype(str), auftrag[ok].astype(str)
    gesamt = ((yy - yy.mean()) ** 2).sum()
    m_art = yy.groupby(aa).transform("mean")
    m_auf = yy.groupby(ff).transform("mean")
    v_art = ((m_art - yy.mean()) ** 2).sum() / gesamt
    v_auf = ((m_auf - m_art) ** 2).sum() / gesamt
    v_bil = ((yy - m_auf) ** 2).sum() / gesamt

    from scipy.stats import spearmanr

    rr, pp = spearmanr(am, dev)

    def e(kurz):
        return E.set_index("kurz").loc[kurz]

    tabelle = [
        f"  {r['name']}: η² {r.beobachtet:.3f}, Null {r.null_mittel:.3f} ± "
        f"{r.null_std:.3f}, Überschuss {r.ueberschuss:+.3f}, ω² {r.omega2:.3f}, "
        f"Reststreuung {r.sd_innen:.1f} Prozentpunkte"
        for _, r in E.iterrows()
    ]
    ch, au = e("Charge Anode"), e("Auftrag")

    C.textseite(
        pdf,
        "02 — Auf welcher Ebene lebt der Ausschuss?",
        [
            "Der Varianzanteil (eta²) einer Gruppierung sagt für sich genommen NICHTS. "
            "Er wächst mechanisch mit der Zahl der Gruppen: im Grenzfall 'jede Gruppe "
            "ein Billet' erklärt jede beliebige Gruppierung 100 % der Varianz, auch "
            "eine zufällige. Deshalb steht neben jedem beobachteten Wert die "
            "Nullerwartung aus 200 Permutationen von y bei unveränderter "
            "Gruppenstruktur. Berichtet gehört der ÜBERSCHUSS über die Nullerwartung, "
            "nicht der Rohwert. ω² ist ein zweites, analytisches Korrekturverfahren; "
            "wo beide auseinanderliegen, ist die Rangfolge nicht belastbar.",
            f"Alle Zahlen beziehen sich auf {C.umfang_text(len(y))}, ohne Kategorie- "
            "und Glastyp-Filter. Billets ohne Wert in einer Gruppierung fallen für "
            "diese Gruppierung weg. In Klammern die Zahl der Gruppen.",
            *tabelle,
            f"Rohstoffcharge: der Rohwert ({ch.beobachtet:.3f}) ist der höchste und "
            f"wirkt wie der stärkste Erklärer überhaupt — nach Abzug der Nullerwartung "
            f"bleiben {ch.ueberschuss:+.3f} gegen {au.ueberschuss:+.3f} beim Auftrag, "
            f"also {100 * ch.ueberschuss / au.ueberschuss:.0f} % des Auftragseffekts. "
            f"ω² sieht das NICHT so ({ch.omega2:.3f} gegen {au.omega2:.3f}): bei "
            f"{ch.gruppen} Chargen auf {ch.n} Billets sind die meisten Chargen "
            "Einzelbillets, und die Innerhalb-Varianz, auf der ω² beruht, wird aus "
            "sehr wenigen Freiheitsgraden geschätzt. Die Rangfolge Charge gegen "
            "Auftrag ist damit nicht belastbar; belastbar ist nur, dass der Rohwert "
            "der Charge weit überzeichnet ist. Aus demselben Grund ist die "
            "Reststreuung der Charge (Seite 2) mechanisch klein.",
            f"Dreistufige Zerlegung (nur Billets mit Artikel und Auftrag): Artikel "
            f"{v_art:.3f}, Auftrag im Artikel {v_auf:.3f}, Billet im Auftrag {v_bil:.3f}.",
            f"y gegen den Auftragsmittelwert: R² {r2_mittel:+.3f}, die SD sinkt von "
            f"{y.std():.1f} auf {dev.std():.1f} Prozentpunkte. ACHTUNG: das "
            "Auftragsmittel kennt y bereits — die Zahl ist eine Obergrenze, keine "
            "Vorhersageleistung.",
            f"Restabweichung gegen Auftragsniveau: Spearman ρ = {rr:+.3f} (p = {pp:.2g}). "
            "Die diagonalen Linien im Streudiagramm sind Aufträge mit Totalausfällen "
            "(y = 100): dort ist die Abweichung 100 minus Auftragsmittel.",
        ],
    )

    fig, ax = C.seite(pdf, "Trennt der Auftrag die Billets?", 2, 2)

    g = y.groupby(auftrag).agg(["mean", "std", "size"]).sort_values("mean")
    g = g[g["size"] >= 2]
    xs = np.arange(len(g))
    ax[0].errorbar(
        xs,
        g["mean"],
        yerr=g["std"].fillna(0),
        fmt="o",
        ms=1.6,
        lw=0.4,
        alpha=0.5,
        color="steelblue",
        ecolor="lightsteelblue",
        label="Auftragsmittel ± SD",
    )
    ax[0].axhline(y.mean(), color="crimson", ls="--", lw=1, label="Gesamtmittel")
    ax[0].set_title("Aufträge mit ≥ 2 Billets, nach Mittelwert sortiert", fontsize=10)
    ax[0].set_xlabel("Auftrag (sortiert)")
    ax[0].set_ylabel("y [%]")
    ax[0].legend(fontsize=8, loc="upper left")

    ys = np.arange(len(E))
    ax[1].barh(ys, E.beobachtet, color="lightsteelblue", label="η² roh")
    ax[1].barh(ys, E.ueberschuss, color="steelblue", label="Überschuss über Null")
    ax[1].errorbar(
        E.null_mittel,
        ys,
        xerr=2 * E.null_std,
        fmt="|",
        color="crimson",
        ms=9,
        lw=1.4,
        label="Nullerwartung ± 2σ",
    )
    ax[1].scatter(
        E.omega2, ys, marker="D", s=22, color="darkorange", zorder=5, label="ω²"
    )
    ax[1].set_yticks(ys)
    ax[1].set_yticklabels(E.name, fontsize=8)
    ax[1].set_xlabel("Anteil der y-Varianz")
    ax[1].set_title("Varianzanteil je Gruppierung mit Nullmodell", fontsize=10)
    ax[1].legend(fontsize=7, loc="lower right")
    ax[1].grid(alpha=0.3, axis="x")

    ax[2].scatter(am, y, s=3, alpha=0.25, color="steelblue")
    lim = [min(am.min(), y.min()), max(am.max(), y.max())]
    ax[2].plot(lim, lim, color="crimson", lw=1, ls="--")
    ax[2].set_xlabel("Mittelwert des Auftrags [%]")
    ax[2].set_ylabel("y des einzelnen Billets [%]")
    ax[2].set_title("y gegen den Auftragsmittelwert", fontsize=10)

    namen = ["Artikel", "Auftrag im Artikel", "Billet im Auftrag"]
    unten = 0.0
    for wert, name, farbe in zip(
        [v_art, v_auf, v_bil], namen, ["#2c5f8a", "#79a7cc", "#d9d9d9"]
    ):
        ax[3].bar(0, wert, bottom=unten, color=farbe, width=0.5)
        ax[3].text(
            0,
            unten + wert / 2,
            f"{name}\n{wert:.3f}",
            ha="center",
            va="center",
            fontsize=9,
            color="white" if farbe == "#2c5f8a" else "black",
        )
        unten += wert
    ax[3].set_xlim(-0.6, 0.6)
    ax[3].set_xticks([])
    ax[3].spines["bottom"].set_visible(False)
    ax[3].set_ylabel("Anteil der y-Varianz")
    ax[3].set_title("Zerlegung der y-Varianz", fontsize=10)
    C.schliesse(pdf, fig, fuss)

    fig, ax = C.seite(pdf, "Wie viel bleibt innerhalb der Gruppe übrig?", 1, 2)

    ys = np.arange(len(E))
    ax[0].barh(ys, E.sd_innen, color="steelblue")
    ax[0].axvline(y.std(), color="crimson", ls="--", lw=1, label="SD von y gesamt")
    ax[0].set_yticks(ys)
    ax[0].set_yticklabels(E.name, fontsize=8)
    ax[0].invert_yaxis()
    ax[0].set_xlabel("Reststreuung innerhalb der Gruppe [Prozentpunkte]")
    ax[0].set_title("Standardabweichung nach Abzug des Gruppenmittels", fontsize=10)
    ax[0].legend(fontsize=8, loc="lower right")
    ax[0].grid(alpha=0.3, axis="x")

    ax[1].scatter(am, dev, s=3, alpha=0.25, color="seagreen")
    ax[1].axhline(0, color="crimson", lw=1)
    ax[1].set_xlabel("Auftragsmittelwert [%]")
    ax[1].set_ylabel("y − Auftragsmittel [Prozentpunkte]")
    ax[1].set_title("Restabweichung gegen das Auftragsniveau", fontsize=10)
    C.schliesse(pdf, fig, fuss)

    pdf.close()

    print(f"geschrieben: {C.OUT_DIR / '02_ebenen.pdf'}")
    for t in tabelle:
        print(t)
    print(
        f"Dreistufig: Artikel {v_art:.3f} | Auftrag-im-Artikel {v_auf:.3f} | "
        f"Billet-im-Auftrag {v_bil:.3f}"
    )


if __name__ == "__main__":
    main()
