"""09 — Zwei Zustandskanäle und abgebrochene Schmelzen.

In der Arbeit: Abschnitt 4.2.6: Zahl der Billets, die die letzte Programmphase
erreichen, und Kennzeichnung der abgebrochenen Schmelzen.

Bezug zweier Zustandskanäle zu y und Kennzeichnung abgebrochener Schmelzen.
Ausgabe: reports/eda/09_prozesskanaele.pdf, reports/figures/prozesskanaele_*.png.

    python eda/09_prozesskanaele.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import _common as C
from data_prep import SP  # noqa: E402

TL = SP["kanal_a"] + "_n_samples"
PH = SP["phase"] + "_std"
MIN_PRO_AUFTRAG = 3
F_TL = "#d62728"
F_PH = "#7b3294"
F_Y = "#3182bd"


def lade():
    """Merkmale, y und die Gruppenspalten in einem Rahmen."""
    import data_prep as D

    ds = D.load_split(verbose=False)
    X = ds.X_raw.reset_index(drop=True)
    d = pd.DataFrame(
        {
            "batch_id": ds.batch_ids.astype(str).values,
            "y": pd.to_numeric(ds.y, errors="coerce").values,
            "Maschine": X[SP["ofen_kennung"]].astype(str).values,
            "Ofen": X["Ofennummer"].astype(str).values,
        }
    )
    mst = pd.read_csv(
        C.ROOT / "data" / "billets_master.csv",
        low_memory=False,
        usecols=["batch_id", SP["artikel"], SP["auftrag"]],
    )
    mst["batch_id"] = mst["batch_id"].astype(str)
    d = d.merge(mst, on="batch_id", how="left")
    for c in X.columns:
        if c.startswith(SP["kanal_a"]) or c.startswith(SP["phase"]):
            d[c] = pd.to_numeric(X[c], errors="coerce").values
    return d


def eta2(v, g):
    """Anteil der Varianz von v, der zwischen den Gruppen g liegt."""
    s = pd.DataFrame({"v": v, "g": g}).dropna()
    if s.empty or s.v.nunique() < 2:
        return np.nan
    gm = s.groupby("g")["v"]
    ges = ((s.v - s.v.mean()) ** 2).sum()
    return (
        float((((gm.mean() - s.v.mean()) ** 2 * gm.size()).sum()) / ges)
        if ges
        else np.nan
    )


def innerhalb(d, spalte):
    """Spearman nach Zentrierung je Auftrag — der Effekt ohne Auftragsebene."""
    s = d.dropna(subset=[spalte, "y", SP["auftrag"]]).copy()
    s = s[s.groupby(SP["auftrag"])[spalte].transform("size") >= MIN_PRO_AUFTRAG]
    if len(s) < 30:
        return np.nan, np.nan, 0
    yz = s.y - s.groupby(SP["auftrag"])["y"].transform("mean")
    vz = s[spalte] - s.groupby(SP["auftrag"])[spalte].transform("mean")
    return (
        float(spearmanr(s[spalte], s.y).statistic),
        float(spearmanr(vz, yz).statistic),
        int(s[SP["auftrag"]].nunique()),
    )


def feld_a(ax, d):
    """n_samples gegen n_edges — der Kanal ist ereignisbasiert."""
    ns, ne = d[TL], d[SP["kanal_a"] + "_n_edges"]
    ax.scatter(ne, ns, s=10, alpha=0.35, color=F_TL, edgecolors="none")
    lims = [1, max(ns.max(), ne.max()) * 1.3]
    ax.plot(lims, lims, "--", color="#666666", lw=1, label="Winkelhalbierende")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_xlabel(SP["kanal_a"] + "_n_edges (Schaltvorgänge)")
    ax.set_ylabel(SP["kanal_a"] + "_n_samples")
    ax.set_title("A  Messpunkte gegen Schaltvorgänge", fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3, which="both")


def feld_b(ax, d, spalte, farbe, titel, xlabel):
    """y je Quintil des Merkmals, Median mit Quartilsabstand."""
    s = d.dropna(subset=[spalte, "y"])
    try:
        q = pd.qcut(s[spalte], 5, duplicates="drop")
    except ValueError:
        q = pd.cut(s[spalte], 5)
    g = s.groupby(q, observed=True)["y"]
    med, lo, hi, n = g.median(), g.quantile(0.25), g.quantile(0.75), g.size()
    x = np.arange(len(med))
    ax.errorbar(
        x,
        med,
        yerr=[med - lo, hi - med],
        fmt="o-",
        color=farbe,
        lw=1.8,
        ms=7,
        capsize=4,
        label="Median mit Quartilsabstand",
    )
    ax.axhline(s.y.median(), color="#999999", ls="--", lw=1, label="Median über alle")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"≤{iv.right:.3g}" for iv in med.index[:-1]] + [f">{med.index[-1].left:.3g}"],
        fontsize=7,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Ausschussgrad y [%]")
    ax.set_title(titel, fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3, axis="y")
    return n


def feld_c(ax, d, spalten, farben, titel):
    """Varianzanteil je Gruppierungsebene, y als Massstab daneben."""
    ebenen = [
        ("Maschine", "Maschine"),
        ("Ofen", "Ofen"),
        ("Artikel", SP["artikel"]),
        ("Auftrag", SP["auftrag"]),
    ]
    breite = 0.8 / (len(spalten) + 1)
    x = np.arange(len(ebenen))
    for i, (sp, fa) in enumerate(zip(spalten + ["y"], farben + [F_Y])):
        werte = [eta2(d[sp], d[g]) for _, g in ebenen]
        ax.bar(
            x + i * breite,
            werte,
            breite,
            color=fa,
            edgecolor="white",
            linewidth=0.6,
            label=sp if sp != "y" else "y (zum Vergleich)",
        )
    ax.set_xticks(x + breite * len(spalten) / 2)
    ax.set_xticklabels([n for n, _ in ebenen])
    ax.set_ylabel("η²  (Anteil der Varianz zwischen den Gruppen)")
    ax.set_title(titel, fontsize=10)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6.5, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.3, axis="y")


def feld_d(ax, d, paare):
    """Roher gegen auftragsbereinigten Zusammenhang."""
    namen, roh, innen, n_auf = [], [], [], []
    for sp, _ in paare:
        r, i, na = innerhalb(d, sp)
        namen.append(sp)
        roh.append(r)
        innen.append(i)
        n_auf.append(na)
    x = np.arange(len(namen))
    ax.bar(x - 0.2, roh, 0.4, color="#999999", edgecolor="white", label="roh")
    ax.bar(
        x + 0.2,
        innen,
        0.4,
        color="#2ca02c",
        edgecolor="white",
        label="innerhalb Auftrag zentriert",
    )
    for i, (r, v) in enumerate(zip(roh, innen)):
        ax.annotate(
            f"{r:+.3f}",
            xy=(i - 0.2, r),
            xytext=(0, 4 if r > 0 else -11),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
        )
        ax.annotate(
            f"{v:+.3f}",
            xy=(i + 0.2, v),
            xytext=(0, 4 if v > 0 else -11),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
        )
    ax.axhline(0, color="black", lw=0.8)
    alle = [w for w in roh + innen if not np.isnan(w)]
    ax.set_ylim(min(min(alle), 0) * 1.3, max(max(alle), 0) * 1.3)
    ax.set_xticks(x)
    ax.set_xticklabels(namen, fontsize=8)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Spearman ρ zu y")
    ax.set_title("D  Spearman ρ zu y, roh und je Auftrag zentriert", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper right" if roh[0] > 0 else "lower right")
    ax.grid(alpha=0.3, axis="y")


def feld_e(ax, d):
    """Die höchste erreichte Phase ist faktisch konstant — die Abweichler sind Abbrueche."""
    v = d[SP["phase"] + "_max"].value_counts().sort_index()
    farben = ["#d62728" if k != v.idxmax() else F_PH for k in v.index]
    ax.bar(v.index.astype(int).astype(str), v.values, color=farben, edgecolor="white")
    ax.set_yscale("log")
    ax.set_ylim(0.7, v.max() * 3)
    ax.set_xlabel("höchste erreichte Phase")
    ax.set_ylabel("Billets (log)")
    ax.set_title("E  Höchste erreichte Phase je Billet", fontsize=10)
    for i, (k, n) in enumerate(v.items()):
        ax.annotate(
            str(n),
            xy=(i, n),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )
    ax.grid(alpha=0.3, axis="y")


def feld_f(ax, d):
    """y je erreichter Endphase."""
    s = d.dropna(subset=[SP["phase"] + "_max", "y"])
    g = s.groupby(SP["phase"] + "_max", observed=True)["y"]
    med, n = g.median(), g.size()
    farben = ["#d62728" if k != n.idxmax() else F_PH for k in med.index]
    ax.bar(med.index.astype(int).astype(str), med.values, color=farben, edgecolor="white")
    for i, (k, m_) in enumerate(med.items()):
        ax.annotate(
            f"n={n[k]}",
            xy=(i, m_),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )
    ax.axhline(
        s.y.median(),
        color="#999999",
        ls="--",
        lw=1,
        label="Median über alle",
    )
    ax.set_xlabel("höchste erreichte Phase")
    ax.set_ylabel("Median y [%]")
    ax.set_title("F  Median y je höchster erreichter Phase", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(alpha=0.3, axis="y")


def main():
    d = lade()
    pdf = C.pdf_oeffnen("09_prozesskanaele.pdf")

    fig, ax = C.seite(
        pdf, f"{TL}: Messart und Ebene", 2, 2
    )
    feld_a(ax[0], d)
    n_b = feld_b(
        ax[1],
        d,
        TL,
        F_TL,
        "B  y je Klasse der Schaltvorgänge",
        "Schaltvorgänge (Quantilsklassen)",
    )
    feld_c(
        ax[2],
        d,
        [TL, SP["kanal_a"] + "_known_frac"],
        [F_TL, "#fc9272"],
        "C  Varianzanteil je Ebene",
    )
    feld_d(ax[3], d, [(TL, F_TL), (PH, F_PH)])
    C.aufraeumen(fig)
    fig.savefig(
        C.ROOT / "reports" / "figures" / "prozesskanaele_a.png",
        dpi=200,
        bbox_inches="tight",
    )
    C.schliesse(
        pdf,
        fig,
        f"{C.CSV_PATH.name} | {len(d)} Billets | y ∈ [{1:g}; 100]",
    )

    fig2, ax2 = C.seite(pdf, f"{PH}: Phasenverlauf der Schmelze", 2, 2)
    feld_e(ax2[0], d)
    feld_f(ax2[1], d)
    n_g = feld_b(
        ax2[2],
        d,
        PH,
        F_PH,
        "G  y je Klasse der Phasenstreuung",
        "Streuung der Phasennummer (Quantilsklassen)",
    )
    feld_c(
        ax2[3],
        d,
        [PH, SP["phase"] + "_max"],
        [F_PH, "#bcbddc"],
        "H  Varianzanteil je Ebene",
    )
    C.aufraeumen(fig2)
    fig2.savefig(
        C.ROOT / "reports" / "figures" / "prozesskanaele_b.png", dpi=200, bbox_inches="tight"
    )
    C.schliesse(pdf, fig2, f"{C.CSV_PATH.name} | {len(d)} Billets")

    r_tl, i_tl, n_tl = innerhalb(d, TL)
    r_ph, i_ph, _ = innerhalb(d, PH)
    e_tl = eta2(d[TL], d[SP["auftrag"]])
    e_y = eta2(d["y"], d[SP["auftrag"]])
    n_phas = int(d[SP["phase"] + "_max"].notna().sum())
    n8 = int((d[SP["phase"] + "_max"] == SP["letzte_phase"]).sum())
    _s = d.dropna(subset=[TL, "y"])
    _q = _s.groupby(pd.qcut(_s[TL], 5, duplicates="drop"), observed=True)["y"].mean()
    q_von, q_bis = _q.iloc[0], _q.iloc[-1]
    y_ab6 = d.loc[d[SP["phase"] + "_max"] == SP["endphase_kurz"], "y"].median()
    ns_ne = d[[TL, SP["kanal_a"] + "_n_edges"]].dropna()
    rho_ne = spearmanr(ns_ne[TL], ns_ne[SP["kanal_a"] + "_n_edges"]).statistic
    diff_ne = (ns_ne[TL] - ns_ne[SP["kanal_a"] + "_n_edges"]).median()
    rueck = 100 * (1 - i_tl / r_tl) if r_tl else float("nan")
    phasen = d.dropna(subset=[SP["phase"] + "_max", "y"]).groupby(SP["phase"] + "_max")["y"]
    C.textseite(
        pdf,
        "Kennzahlen",
        [
            f"{TL}: Spearman zu den Schaltvorgängen {rho_ne:.4f}, mediane Differenz "
            f"{diff_ne:.0f}; η² über die Maschine {eta2(d[TL], d['Maschine']):.3f}.",
            f"Zusammenhang mit y: {len(n_b)} Quantilsklassen (n je Klasse: "
            f"{', '.join(str(int(x)) for x in n_b)}), mittleres y von {q_von:.1f} bis "
            f"{q_bis:.1f} %; Spearman roh {r_tl:+.4f}.",
            f"Ebene: η² über den Auftrag {e_tl:.3f} (Merkmal) und {e_y:.3f} (y). Je "
            f"Auftrag zentriert sinkt der Zusammenhang von {r_tl:+.4f} auf {i_tl:+.4f} "
            f"({rueck:.0f} %, {n_tl} Aufträge mit mindestens {MIN_PRO_AUFTRAG} Billets).",
            f"Höchste erreichte Phase: {n8} von {n_phas} Billets erreichen die Endphase; "
            "Billets je Endphase: "
            + ", ".join(f"Phase {int(k)}: {n}" for k, n in phasen.size().items())
            + f". Median y bei Endphase {SP['endphase_kurz']}: {y_ab6:.1f} %. Billets, die die Endphase "
            "nicht erreichen, gelten als abgebrochene Schmelzen.",
            f"{PH}: Spearman zu y roh {r_ph:+.4f}, innerhalb des Auftrags {i_ph:+.4f}; "
            f"n je Quantilsklasse: {', '.join(str(int(x)) for x in n_g)}.",
            "Methode: Die Zerlegung nach Ebenen beruht auf η² (Varianzanteile ohne "
            "Modellannahme). Die Zentrierung je Auftrag nutzt nur Aufträge mit "
            "mindestens der genannten Billetzahl.",
        ],
    )
    pdf.close()
    print(f"geschrieben: {C.OUT_DIR / '09_prozesskanaele.pdf'}")
    print(f"  {TL}: roh {r_tl:+.4f}, innerhalb Auftrag {i_tl:+.4f}")
    print(f"  {PH}: roh {r_ph:+.4f}, innerhalb Auftrag {i_ph:+.4f}")


if __name__ == "__main__":
    main()
