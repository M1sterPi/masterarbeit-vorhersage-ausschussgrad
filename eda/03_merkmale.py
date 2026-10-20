"""03 — Merkmale gegen y.

In der Arbeit: nicht berichtet; Vorgänger von eda/24 (Abschnitt 4.2.6).

Rangkorrelation gesamt und innerhalb von Auftrag bzw. Artikel.
Ausgabe: reports/eda/03_merkmale.pdf, 03_merkmale_tabelle.csv,
reports/figures/merkmale_top10_innerhalb_*.png.

    python eda/03_merkmale.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import _common as C
from data_prep import SP  # noqa: E402

TOP_N = 24


def main():
    ds = C.lade(alle_arten=True)
    X, y, df = C.daten_teil(ds)
    fuss = C.kopfzeile(ds, alle_arten=True)
    pdf = C.pdf_oeffnen("03_merkmale.pdf")

    auftrag = df[SP["auftrag"]]
    dev = C.auftragsabweichung(y, auftrag)
    artikel = df[SP["artikel"]].astype(str)
    dev_art = y - y.groupby(artikel).transform("mean")
    num = C.numerische_merkmale(X)

    t_y = C.spearman_tabelle(X, y, num).set_index("merkmal")
    t_d = C.spearman_tabelle(X, dev, num).set_index("merkmal")
    t_a = C.spearman_tabelle(X, dev_art, num).set_index("merkmal")
    w_d = C.within_tabelle(X, y, auftrag, num).set_index("merkmal")
    w_a = C.within_tabelle(X, y, artikel, num).set_index("merkmal")
    gem = t_y.index.intersection(t_d.index).intersection(t_a.index)
    T = pd.DataFrame(
        {
            "rho_y": t_y.loc[gem, "rho"],
            "q_y": t_y.loc[gem, "q"],
            "rho_dev": t_d.loc[gem, "rho"],
            "q_dev": t_d.loc[gem, "q"],
            "rho_art": t_a.loc[gem, "rho"],
            "q_art": t_a.loc[gem, "q"],
            "rho_w_auf": w_d.rho.reindex(gem),
            "q_w_auf": w_d.q.reindex(gem),
            "rho_w_art": w_a.rho.reindex(gem),
            "q_w_art": w_a.q.reindex(gem),
            "n": t_y.loc[gem, "n"],
        }
    )

    def ebene(r):
        if r.q_y >= 0.05:
            return "—"
        if r.q_w_auf < 0.05:
            return "Billet"
        if r.q_w_art < 0.05:
            return "Auftrag"
        return "Artikel"

    T["ebene"] = T.apply(ebene, axis=1)
    T["quelle"] = [
        "Influx"
        if any(
            c.endswith(s)
            for s in (
                "_mean",
                "_std",
                "_min",
                "_max",
                "_last",
                "_range",
                "_n_samples",
                "_known_frac",
                "_frac_on_t",
                "_n_edges",
            )
        )
        else "SQL"
        for c in T.index
    ]
    T = T.reindex(T.rho_y.abs().sort_values(ascending=False).index)

    n_sig_y = int((T.q_y < 0.05).sum())
    n_sig_d = int((T.q_dev < 0.05).sum())
    n_sig_a = int((T.q_art < 0.05).sum())
    schrumpf = (T.rho_dev.abs() / T.rho_y.abs().replace(0, np.nan)).median()
    schrumpf_art = (T.rho_art.abs() / T.rho_y.abs().replace(0, np.nan)).median()
    n_sig_wd = int((T.q_w_auf < 0.05).sum())
    n_sig_wa = int((T.q_w_art < 0.05).sum())
    n_konst_d = int(T.rho_w_auf.isna().sum())
    n_konst_a = int(T.rho_w_art.isna().sum())
    schrumpf_wd = (T.rho_w_auf.abs() / T.rho_y.abs().replace(0, np.nan)).median()
    schrumpf_wa = (T.rho_w_art.abs() / T.rho_y.abs().replace(0, np.nan)).median()
    sig_y = T[T.q_y < 0.05]
    ebenen = sig_y.ebene.value_counts().reindex(["Artikel", "Auftrag", "Billet"]).fillna(0).astype(int)
    T.to_csv(C.OUT_DIR / "03_merkmale_tabelle.csv", encoding="utf-8")

    anteile = []
    for c in num:
        s = C.als_zahl(X[c])
        if s.notna().sum() < 50 or s.nunique() < 3:
            continue
        anteile.append(
            {
                "merkmal": c,
                "anteil": C.var_anteil(s.to_numpy(), auftrag.to_numpy()),
                "quelle": T.quelle.get(c, "SQL"),
            }
        )
    A = pd.DataFrame(anteile).dropna()
    from scipy.stats import spearmanr

    rr, pp = spearmanr(A.anteil, T.rho_y.abs().reindex(A.merkmal).to_numpy(), nan_policy="omit")

    top = T.head(TOP_N).index.tolist()
    grenzen = {c: C.robuste_grenzen(C.als_zahl(X[c])) for c in top}

    C.textseite(
        pdf,
        "03 — Merkmale gegen y und gegen die Auftragsabweichung",
        [
            f"{len(T)} numerische Merkmale, je zweimal gegen ein Ziel geprüft "
            f"({C.umfang_text(len(y))}):",
            f"  gegen y: {n_sig_y} Merkmale mit q < 0,05",
            f"  gegen y − Auftragsmittel: {n_sig_d} Merkmale mit q < 0,05",
            f"  gegen y − Artikelmittel: {n_sig_a} Merkmale mit q < 0,05",
            f"Median-Schrumpfung |ρ_Abweichung| / |ρ_y| = {schrumpf:.2f} (Auftrag), "
            f"{schrumpf_art:.2f} (Artikel).",
            "Within-Korrelation (Merkmal UND y je Gruppe zentriert, auf Rängen, nur "
            "Gruppen mit ≥ 2 Billets, Freiheitsgrade n − k − 1):",
            f"  innerhalb Auftrag: {n_sig_wd} Merkmale mit q < 0,05, {n_konst_d} "
            f"innerhalb konstant, Median-Schrumpfung {schrumpf_wd:.2f}",
            f"  innerhalb Artikel: {n_sig_wa} Merkmale mit q < 0,05, {n_konst_a} "
            f"innerhalb konstant, Median-Schrumpfung {schrumpf_wa:.2f}",
            f"Von den {n_sig_y} mit y verbundenen Merkmalen tragen bis zur Ebene "
            f"Artikel: {ebenen['Artikel']}, Auftrag (innerhalb Artikel, nicht innerhalb "
            f"Auftrag): {ebenen['Auftrag']}, Billet (auch innerhalb Auftrag): "
            f"{ebenen['Billet']}.",
            *[
                f"  {c} [{T.loc[c, 'ebene']}]: ρ_y {T.loc[c, 'rho_y']:+.3f}, "
                f"within Artikel {T.loc[c, 'rho_w_art']:+.3f} (q {T.loc[c, 'q_w_art']:.2g}), "
                f"within Auftrag {T.loc[c, 'rho_w_auf']:+.3f} (q {T.loc[c, 'q_w_auf']:.2g})"
                for c in sig_y[sig_y.ebene != "Artikel"].index
            ],
            *[
                f"  nur within signifikant: {c}: ρ_y {T.loc[c, 'rho_y']:+.3f} "
                f"(q {T.loc[c, 'q_y']:.2g}), within Artikel {T.loc[c, 'rho_w_art']:+.3f} "
                f"(q {T.loc[c, 'q_w_art']:.2g}), within Auftrag "
                f"{T.loc[c, 'rho_w_auf']:+.3f} (q {T.loc[c, 'q_w_auf']:.2g})"
                for c in T[(T.q_y >= 0.05) & ((T.q_w_art < 0.05) | (T.q_w_auf < 0.05))].index
            ],
            "Die Artikelabweichung ist der schwächere Abzug: sie nimmt nur das Rezept "
            "heraus, nicht die Unterschiede zwischen Aufträgen desselben Artikels. Ein "
            "Merkmal, das gegen die Artikel-, aber nicht gegen die Auftragsabweichung "
            "trägt, trennt Aufträge innerhalb eines Artikels.",
            "Lesart: ein Merkmal, das mit y korreliert, aber nicht mit der Abweichung "
            "vom Auftragsmittel, beschreibt die SORTE des Auftrags, nicht das einzelne "
            "Billet. Bricht die zweite Spalte flächendeckend ein, ist die Vorhersage im "
            "Kern eine Auftragsvorhersage. Im Streudiagramm der Ranglisten-Seite liegen "
            "solche Merkmale weit unter der Diagonale.",
            f"Warum FDR und nicht roher p-Wert: bei {len(T)} Tests und alpha = 0,05 sind "
            f"rund {round(0.05 * len(T))} Zufallstreffer zu erwarten. Benjamini-Hochberg "
            "kontrolliert den Anteil falscher Entdeckungen unter den als signifikant "
            "gemeldeten.",
            "Warum Spearman und nicht Pearson: y ist rechtsschief und viele Merkmale "
            "haben Ausreisser; Pearson misst dort die Ausreisser statt des Zusammenhangs.",
            f"Die {TOP_N} stärksten Merkmale (Streudiagramme; die Hochachse ist y, die "
            "Rechtsachse ist auf den Bereich ohne Ausreisser begrenzt — Tukey-Zaun "
            "3 IQR —, die Zahl der Punkte ausserhalb steht dahinter):",
            *[
                f"  {i + 1}. {c} [{T.loc[c, 'quelle']}]: ρ_y {T.loc[c, 'rho_y']:+.2f}, "
                f"ρ_Auftragsabw. {T.loc[c, 'rho_dev']:+.2f}, "
                f"ρ_Artikelabw. {T.loc[c, 'rho_art']:+.2f}, "
                f"{C.ausserhalb(C.als_zahl(X[c]).dropna(), grenzen[c])} ausserhalb"
                for i, c in enumerate(top)
            ],
            "Ebene der Merkmale (letzte Seite): Werte nahe 1 heissen, das Merkmal ist "
            "innerhalb eines Auftrags konstant — es KANN nichts über die Unterschiede "
            f"zwischen Geschwisterbillets sagen. {len(A)} Merkmale, davon "
            f"{int((A.quelle == 'SQL').sum())} SQL und {int((A.quelle == 'Influx').sum())} "
            f"Influx. Zusammenhang zwischen Auftragsanteil und |ρ| gegen y: Spearman "
            f"ρ = {rr:+.3f} (p = {pp:.2g}).",
        ],
    )

    farbe_quelle = {"Influx": "steelblue", "SQL": "darkorange"}

    def quellen_legende(a, **kw):
        from matplotlib.lines import Line2D

        griffe = [
            Line2D([], [], ls="", marker="o", color=f, label=q)
            for q, f in farbe_quelle.items()
        ]
        return griffe

    def ranglisten_seite(spalte: str, abw: str, mittel: str):
        K = T.head(35)
        fig, ax = C.seite(
            pdf, f"Korrelation mit y gegenüber Korrelation mit der {abw}", 1, 2
        )
        ys = np.arange(len(K))[::-1]
        ax[0].barh(ys + 0.2, K.rho_y, height=0.4, color="steelblue", label="gegen y")
        ax[0].barh(
            ys - 0.2,
            K[spalte],
            height=0.4,
            color="seagreen",
            label=f"gegen y − {mittel}",
        )
        ax[0].set_yticks(ys)
        ax[0].set_yticklabels(
            [f"{c[:34]} [{q}]" for c, q in zip(K.index, K.quelle)], fontsize=6.5
        )
        ax[0].axvline(0, color="black", lw=0.8)
        ax[0].set_xlabel("Spearman ρ")
        ax[0].set_title(f"{len(K)} stärkste Merkmale nach |ρ| zu y", fontsize=10)
        ax[0].legend(fontsize=8, loc="lower right")
        ax[0].grid(alpha=0.3, axis="x")

        ax[1].scatter(
            T.rho_y.abs(),
            T[spalte].abs(),
            s=12,
            alpha=0.55,
            c=[farbe_quelle[q] for q in T.quelle],
        )
        lim = float(max(T.rho_y.abs().max(), T[spalte].abs().max())) * 1.08
        (diag,) = ax[1].plot(
            [0, lim], [0, lim], color="crimson", ls="--", lw=1, label="gleich stark"
        )
        ax[1].set_xlim(0, lim)
        ax[1].set_ylim(0, lim)
        ax[1].set_xlabel("|ρ| gegen y")
        ax[1].set_ylabel(f"|ρ| gegen die {abw}")
        ax[1].set_title("Jeder Punkt ein Merkmal", fontsize=10)
        ax[1].legend(
            handles=quellen_legende(ax[1]) + [diag], fontsize=8, loc="upper left"
        )
        ax[1].grid(alpha=0.3)
        C.schliesse(pdf, fig, fuss)

    fig, ax = C.seite(pdf, "Die zehn stärksten Merkmale nach Abzug des Gruppenmittels", 2, 1)
    for a, (spalte, qspalte, gruppe) in zip(
        ax, [("rho_w_art", "q_w_art", "Artikel"), ("rho_w_auf", "q_w_auf", "Auftrag")]
    ):
        K = T[T[spalte].notna()].reindex(
            T[spalte].dropna().abs().sort_values(ascending=False).index
        ).head(10)
        a.axis("off")
        tab = a.table(
            cellText=[
                [c[:40], K.loc[c, "quelle"], C.de(f"{K.loc[c, 'rho_y']:+.3f}"),
                 C.de(f"{K.loc[c, 'q_y']:.2g}"), C.de(f"{K.loc[c, spalte]:+.3f}"),
                 C.de(f"{K.loc[c, qspalte]:.2g}"), K.loc[c, "ebene"]]
                for c in K.index
            ],
            colLabels=["Merkmal", "Quelle", "ρ gegen y", "q (y)",
                       f"ρ innerhalb {gruppe}", "q (innerhalb)", "Ebene"],
            loc="center",
            colWidths=[0.34, 0.09, 0.1, 0.1, 0.14, 0.12, 0.09],
        )
        tab.auto_set_font_size(False)
        tab.set_fontsize(8)
        tab.scale(1, 1.3)
        a.set_title(
            f"Innerhalb {gruppe}: Merkmal und y je {gruppe} zentriert, nach |ρ| sortiert",
            fontsize=10,
        )
    C.schliesse(pdf, fig, fuss + "  |  Ebene '—': kein signifikanter Zusammenhang mit y selbst")

    def top10_balken(a, spalte: str, gruppe: str):
        K = T.loc[T[spalte].dropna().abs().sort_values(ascending=False).index].head(10)
        ys = np.arange(len(K))[::-1]
        a.barh(ys, K[spalte], color="steelblue", height=0.6)
        a.set_yticks(ys)
        a.set_yticklabels(K.index, fontsize=8)
        a.axvline(0, color="black", lw=0.8)
        grenze = float(K[spalte].abs().max()) * 1.25
        a.set_xlim(-grenze, grenze)
        a.set_xlabel(f"Spearman ρ innerhalb des {gruppe}s")
        a.set_title(f"Innerhalb {gruppe}", fontsize=10)
        a.grid(alpha=0.3, axis="x")

    fig, ax = C.seite(pdf, "Die zehn stärksten Merkmale nach Abzug des Gruppenmittels", 1, 2)
    top10_balken(ax[0], "rho_w_art", "Artikel")
    top10_balken(ax[1], "rho_w_auf", "Auftrag")
    C.aufraeumen(fig)
    C.schliesse(pdf, fig, fuss + "  |  Merkmal und y je Gruppe zentriert, nach |ρ| sortiert")

    for spalte, gruppe, datei in [
        ("rho_w_art", "Artikel", "merkmale_top10_innerhalb_artikel.png"),
        ("rho_w_auf", "Auftrag", "merkmale_top10_innerhalb_auftrag.png"),
    ]:
        import matplotlib.pyplot as plt

        f1, a1 = plt.subplots(figsize=(7, 4))
        top10_balken(a1, spalte, gruppe)
        C.aufraeumen(f1)
        png = C.ROOT / "reports" / "figures" / datei
        png.parent.mkdir(parents=True, exist_ok=True)
        f1.savefig(png, dpi=200, bbox_inches="tight")
        plt.close(f1)

    ranglisten_seite("rho_w_auf", "Within-Korrelation im Auftrag", "Auftragsmittel (beide zentriert)")
    ranglisten_seite("rho_w_art", "Within-Korrelation im Artikel", "Artikelmittel (beide zentriert)")
    ranglisten_seite("rho_dev", "Auftragsabweichung (nur y zentriert)", "Auftragsmittel")

    for start in range(0, len(top), 12):
        block = top[start : start + 12]
        fig, ax = C.seite(
            pdf,
            f"y gegen die stärksten Merkmale ({start + 1}–{start + len(block)})",
            3,
            4,
        )
        for a, c in zip(ax, block):
            s = C.als_zahl(X[c])
            ok = s.notna()
            a.scatter(s[ok], y[ok], s=3, alpha=0.25, color="steelblue")
            a.set_xlim(*grenzen[c])
            a.set_title(f"{c[:30]}", fontsize=7.5)
            a.set_ylabel("y [%]", fontsize=6.5)
            a.tick_params(labelsize=6)
        for a in ax[len(block) :]:
            C.leer(a)
        C.schliesse(pdf, fig, fuss + "  |  y auf der Hochachse")

    fig, ax = C.seite(pdf, "Auf welcher Ebene leben die Merkmale?", 1, 2)
    for q in ("SQL", "Influx"):
        teil = A[A.quelle == q]
        ax[0].hist(teil.anteil, bins=40, alpha=0.6, label=q, color=farbe_quelle[q])
    ax[0].set_xlabel("Anteil der Merkmalsvarianz zwischen Aufträgen")
    ax[0].set_ylabel("Anzahl Merkmale")
    ax[0].set_title("Auftragskonstant oder billetindividuell?", fontsize=10)
    ax[0].legend(fontsize=8, loc="upper left")

    ax[1].scatter(
        A.anteil,
        T.rho_y.abs().reindex(A.merkmal).to_numpy(),
        s=12,
        alpha=0.55,
        c=[farbe_quelle[q] for q in A.quelle],
    )
    ax[1].set_xlabel("Anteil der Merkmalsvarianz zwischen Aufträgen")
    ax[1].set_ylabel("|ρ| gegen y")
    ax[1].set_title("Auftragsanteil gegen Stärke der Korrelation", fontsize=10)
    ax[1].legend(handles=quellen_legende(ax[1]), fontsize=8, loc="upper left")
    ax[1].grid(alpha=0.3)
    C.schliesse(pdf, fig, fuss)

    pdf.close()
    print(f"geschrieben: {C.OUT_DIR / '03_merkmale.pdf'}")
    print(
        f"  {len(T)} Merkmale | q<0.05 gegen y: {n_sig_y} | gegen Auftragsabw.: {n_sig_d}"
        f" | gegen Artikelabw.: {n_sig_a}"
    )
    print(f"  Median-Schrumpfung |rho_dev|/|rho_y| = {schrumpf:.3f} (Auftrag), "
          f"{schrumpf_art:.3f} (Artikel)")
    print(f"  within Auftrag: q<0.05 {n_sig_wd}, konstant {n_konst_d}, Schrumpfung {schrumpf_wd:.3f}")
    print(f"  within Artikel: q<0.05 {n_sig_wa}, konstant {n_konst_a}, Schrumpfung {schrumpf_wa:.3f}")
    print(f"  Ebenen der {n_sig_y} y-Merkmale: {ebenen.to_dict()}")


if __name__ == "__main__":
    main()
