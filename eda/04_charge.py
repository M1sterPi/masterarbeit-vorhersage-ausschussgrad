"""04 — Rohstoffcharge.

In der Arbeit: Abschnitt 4.2.6: Rohstoffcharge innerhalb der Aufträge gegen das
Nullmodell und Rohstoffanalyse innerhalb der Artikel; Abdeckung der
Rohstoffanalyse in Abschnitt 4.2.3.

Rohstoffcharge und Rohstoffanalyse gegen y, mit Nullmodell.
Ausgabe: reports/eda/04_charge.pdf.

    python eda/04_charge.py
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

import _common as C
from data_prep import SP  # noqa: E402

ELEMENTE = ["Al", "Na", "K", "Li", "Ca", "Ti", "Fe", "Zr"]


def main():
    ds = C.lade(alle_arten=True)
    X, y, df = C.daten_teil(ds)
    fuss = C.kopfzeile(ds, alle_arten=True)
    pdf = C.pdf_oeffnen("04_charge.pdf")

    auftrag = df[SP["auftrag"]]
    dev = C.auftragsabweichung(y, auftrag)
    charge = df[SP["rohstoff_charge_anode"]].astype(str).replace("nan", np.nan)

    je_auftrag = df.groupby(SP["auftrag"])[SP["rohstoff_charge_anode"]].nunique()
    groesse = df.groupby(SP["auftrag"]).size()
    mehr = groesse[groesse > 1].index
    anteil_variabel = float((je_auftrag[mehr] > 1).mean())

    r_ges = C.var_anteil_mit_null(y, charge, n_perm=200)
    ok = charge.notna() & auftrag.isin(mehr)
    r_dev = C.var_anteil_mit_null(dev[ok], charge[ok], n_perm=200)

    mit = charge.notna()
    w_ch = C.omega2(y[mit], charge[mit])
    r_au = C.var_anteil_mit_null(y[mit], auftrag[mit], n_perm=200)
    w_au = C.omega2(y[mit], auftrag[mit])
    einzel = int((charge.value_counts() == 1).sum())

    chemie = []
    for el in ELEMENTE:
        spalte = f"chargeA__{el}"
        if spalte not in X.columns:
            continue
        s = C.als_zahl(X[spalte])
        gut = s.notna() & (s > 0)
        if gut.sum() < 30:
            continue
        ry, py = spearmanr(s[gut], y[gut])
        rd, pd_ = spearmanr(s[gut], dev[gut])
        chemie.append(
            {"el": el, "s": s, "gut": gut, "ry": ry, "py": py, "rd": rd, "pd": pd_}
        )

    C.textseite(
        pdf,
        "04 — Rohstoffcharge",
        [
            f"{int(charge.nunique())} verschiedene Anodenchargen auf {int(mit.sum())} "
            f"Billets mit Chargenangabe ({int((~mit).sum())} Billets ohne). "
            f"{einzel} Chargen kommen nur in einem einzigen Billet vor.",
            f"{100 * anteil_variabel:.1f} % der Mehrbillet-Aufträge verwenden mehr als "
            f"eine Charge (im Mittel {je_auftrag[mehr].mean():.2f}). Die Charge ist "
            "damit der einzige Stammdaten-Kandidat, der Geschwisterbillets überhaupt "
            "trennen KANN.",
            "Warum der Rohwert täuscht (nur Billets mit Chargenangabe, Charge gegen "
            "Auftrag auf derselben Menge):",
            f"  Charge: η² {r_ges['beobachtet']:.3f}, Nullerwartung "
            f"{r_ges['null_mittel']:.3f} ± {r_ges['null_std']:.3f}, Überschuss "
            f"{r_ges['ueberschuss']:+.3f}, ω² {w_ch:.3f}",
            f"  Auftrag: η² {r_au['beobachtet']:.3f}, Nullerwartung "
            f"{r_au['null_mittel']:.3f} ± {r_au['null_std']:.3f}, Überschuss "
            f"{r_au['ueberschuss']:+.3f}, ω² {w_au:.3f}",
            "Beide Korrekturen nehmen dem Rohwert der Charge den grössten Teil. Wie "
            "viel übrig bleibt, beurteilen sie aber VERSCHIEDEN: der "
            f"Permutationsüberschuss sieht die Charge bei "
            f"{100 * r_ges['ueberschuss'] / r_au['ueberschuss']:.0f} % des "
            f"Auftrags, ω² bei {100 * w_ch / w_au:.0f} %. Grund ist die Datenlage — "
            "bei so vielen Einzelbillet-Chargen wird die Innerhalb-Varianz, auf der ω² "
            "beruht, aus sehr wenigen Freiheitsgraden geschätzt. Belastbar ist nur: "
            "der Rohwert ist weit überzeichnet. Eine Rangfolge Charge gegen Auftrag "
            "lässt sich daraus nicht ableiten.",
            "Und die eigentliche Frage — erklärt die Charge etwas INNERHALB des Auftrags? "
            f"Varianzanteil an der Auftragsabweichung {r_dev['beobachtet']:.3f}, "
            f"Nullerwartung {r_dev['null_mittel']:.3f} ± {r_dev['null_std']:.3f}, "
            f"Überschuss {r_dev['ueberschuss']:+.3f} "
            f"({r_dev['ueberschuss'] / max(r_dev['null_std'], 1e-9):+.1f} σ).",
            "Spurenelemente (nur Lose mit echter Laboranalyse, Al > 0; Spearman ρ gegen "
            "y und gegen die Auftragsabweichung; die Achsen der Streudiagramme sind auf "
            "den Bereich ohne Ausreisser begrenzt (Tukey-Zaun, 3 IQR), die Zahl der Punkte ausserhalb steht in Klammern):",
            *[
                f"  {c['el']}: n {int(c['gut'].sum())} "
                f"({C.ausserhalb(c['s'][c['gut']], C.robuste_grenzen(c['s'][c['gut']]))} "
                f"ausserhalb), gegen y ρ {c['ry']:+.2f} "
                f"(p {c['py']:.2g}), gegen Abweichung ρ {c['rd']:+.2f} (p {c['pd']:.2g})"
                for c in chemie
            ],
        ],
    )

    fig, ax = C.seite(pdf, "Die Rohstoffcharge, roh und korrigiert", 2, 2)

    namen = ["an y", "an der Auftragsabweichung"]
    beob = [r_ges["beobachtet"], r_dev["beobachtet"]]
    null = [r_ges["null_mittel"], r_dev["null_mittel"]]
    nsd = [r_ges["null_std"], r_dev["null_std"]]
    xs = np.arange(2)
    ax[0].bar(xs, beob, width=0.5, color="lightsteelblue", label="beobachtet")
    ax[0].bar(
        xs,
        [b - n for b, n in zip(beob, null)],
        width=0.5,
        color="steelblue",
        label="Überschuss über Null",
    )
    ax[0].errorbar(
        xs,
        null,
        yerr=[2 * s for s in nsd],
        fmt="_",
        ms=22,
        lw=1.6,
        color="crimson",
        label="Nullerwartung ± 2σ",
    )
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels(namen, fontsize=9)
    ax[0].set_ylabel("Varianzanteil der Charge")
    ax[0].set_ylim(top=1.0)
    ax[0].set_title("Varianzanteil der Charge mit Nullmodell", fontsize=10)
    ax[0].legend(fontsize=8, loc="upper right")
    ax[0].grid(alpha=0.3, axis="y")

    g = y.groupby(charge).agg(["mean", "std", "size"])
    g = g[g["size"] >= 3].sort_values("mean")
    ax[1].errorbar(
        np.arange(len(g)),
        g["mean"],
        yerr=g["std"].fillna(0),
        fmt="o",
        ms=2,
        lw=0.5,
        alpha=0.6,
        color="seagreen",
        ecolor="lightgreen",
        label="Chargenmittel ± SD",
    )
    ax[1].axhline(y.mean(), color="crimson", ls="--", lw=1, label="Gesamtmittel")
    ax[1].set_title("Chargen mit ≥ 3 Billets, nach Mittelwert sortiert", fontsize=10)
    ax[1].set_xlabel("Charge (sortiert)")
    ax[1].set_ylabel("y [%]")
    ax[1].legend(fontsize=8, loc="upper left")

    ax[2].hist(
        je_auftrag[mehr],
        bins=range(1, int(je_auftrag[mehr].max()) + 2),
        color="steelblue",
    )
    ax[2].set_xlabel("verschiedene Chargen im Auftrag")
    ax[2].set_ylabel("Anzahl Aufträge")
    ax[2].set_title("Chargen je Mehrbillet-Auftrag", fontsize=10)

    hc = [c for c in df.columns if c.endswith("has_chemistry")]
    if hc:
        wert = df[hc[0]].astype(str).str.lower().isin(["true", "1"])
        anteile = [int(wert.sum()), int((~wert).sum())]
        ax[3].bar(
            ["Laboranalyse vorhanden", "nie analysiert (Al = 0)"],
            anteile,
            color=["seagreen", "lightgrey"],
        )
        for i, v in enumerate(anteile):
            ax[3].text(
                i,
                v,
                f"{v} ({100 * v / len(wert):.0f} %)",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax[3].set_ylim(0, max(anteile) * 1.12)
        ax[3].set_title("Billets mit Chemieanalyse", fontsize=10)
        ax[3].set_ylabel("Anzahl Billets")
    else:
        C.leer(ax[3])
    C.schliesse(pdf, fig, fuss)

    fig, ax = C.seite(
        pdf, "Spurenelemente gegen y und gegen die Auftragsabweichung", 4, 4
    )
    i = 0
    for c in chemie:
        if i >= len(ax) - 1:
            break
        s, gut = c["s"], c["gut"]
        grenzen = C.robuste_grenzen(s[gut])
        for ziel, name, farbe, ylab in (
            (y, "y", "steelblue", "y [%]"),
            (dev, "Abweichung", "seagreen", "Abweichung [Prozentpunkte]"),
        ):
            a = ax[i]
            i += 1
            a.scatter(s[gut], ziel[gut], s=4, alpha=0.4, color=farbe)
            a.set_xlim(*grenzen)
            a.set_title(f"{c['el']} gegen {name}", fontsize=8)
            a.set_ylabel(ylab, fontsize=6.5)
            a.tick_params(labelsize=6)
    for a in ax[i:]:
        C.leer(a)
    C.schliesse(pdf, fig, fuss + "  |  nur Lose mit echter Laboranalyse (Al > 0)")

    pdf.close()
    print(f"geschrieben: {C.OUT_DIR / '04_charge.pdf'}")
    print(
        f"  Charge  : Ueberschuss {r_ges['ueberschuss']:+.3f}  omega2 {w_ch:.3f}\n"
        f"  Auftrag : Ueberschuss {r_au['ueberschuss']:+.3f}  omega2 {w_au:.3f}\n"
        f"  an der Abweichung: Ueberschuss {r_dev['ueberschuss']:+.3f}"
    )


if __name__ == "__main__":
    main()
