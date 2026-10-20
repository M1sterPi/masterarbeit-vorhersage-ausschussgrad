"""06 — TabPFN gegen XGBoost.

In der Arbeit: Abschnitt 7.2: Abbildung 17 und die Rangkorrelation zwischen
TabPFN und XGBoost auf denselben Billets.

Merkmalswichtigkeit beider Verfahren auf denselben Billets (braucht xai/03
und xai/05). Abbildungen auch in reports/figures/ und reports/figures/kapitel7/.

    python xai/06_tabpfn_vs_xgboost.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data_pipeline"))

TABPFN_LAUF = ROOT / "runs" / "xai" / "05_tabpfn.json"

import _common as C  # noqa: E402

TOP = 20
K_GITTER = [5, 10, 20, 30, 50, 75, 100, 150, 200]
F_TAB = "#7b3294"
F_XGB = "#3182bd"
F_REF = "#999999"


def lade_tabpfn():
    """Gespeicherte TabPFN-Beitraege + die erklaerten Zeilenindizes."""
    if not TABPFN_LAUF.exists():
        raise SystemExit(f"TabPFN-Lauf fehlt: {TABPFN_LAUF}")
    d = json.loads(TABPFN_LAUF.read_text(encoding="utf-8"))
    schluessel = sorted(d["beitraege"], key=int)
    B = np.array([d["beitraege"][k] for k in schluessel])
    return B, [int(k) for k in schluessel], d["spalten"], int(d["budget"])


def eigenstabilitaet(B, n_paare=100, seed=C.SEED):
    """Rangkorrelation zweier Bootstrap-Ziehungen ueber die erklaerten Zeilen."""
    rng = np.random.default_rng(seed)
    n = len(B)
    zieh = np.array(
        [np.abs(B[rng.integers(0, n, n)]).mean(axis=0) for _ in range(2 * n_paare)]
    )
    return float(
        np.mean(
            [
                spearmanr(zieh[i], zieh[i + 1]).statistic
                for i in range(0, 2 * n_paare, 2)
            ]
        )
    )


def wiederholziehung(zeilen):
    """Erst- gegen Zweitziehung aus 05: (mittleres rho, Top-20-Deckung, n)."""
    d = json.loads(TABPFN_LAUF.read_text(encoding="utf-8"))
    rhos, deck = [], []
    for k, zweit in d.get("zweitziehung", {}).items():
        if k not in d["beitraege"]:
            continue
        a = np.abs(np.array(d["beitraege"][k]))
        b = np.abs(np.array(zweit))
        rhos.append(spearmanr(a, b).statistic)
        deck.append(len(set(np.argsort(-a)[:20]) & set(np.argsort(-b)[:20])))
    if not rhos:
        return float("nan"), float("nan"), 0
    return float(np.mean(rhos)), float(np.mean(deck)), len(rhos)


def feld_a(ax, wt, wx):
    tt, tx = list(wt.head(TOP).Merkmal), list(wx.head(TOP).Merkmal)
    gemeinsam = set(tt) & set(tx)
    for spalte, (namen, farbe, titel) in enumerate(
        ((tt, F_TAB, "TabPFN (shapiq)"), (tx, F_XGB, "XGBoost (TreeSHAP)"))
    ):
        for rang, name in enumerate(namen):
            fett = name in gemeinsam
            ax.text(
                spalte,
                -rang,
                f"{rang + 1:2d}. {name[:26]}",
                fontsize=7,
                ha="center",
                va="center",
                color=farbe if fett else "#888888",
                weight="bold" if fett else "normal",
                bbox=dict(
                    fc=farbe if fett else "white",
                    alpha=0.14 if fett else 0,
                    ec="none",
                    pad=1.4,
                ),
            )
        ax.text(spalte, 1.6, titel, fontsize=9, ha="center", weight="bold", color=farbe)
    for name in gemeinsam:
        ax.plot(
            [0.34, 0.66],
            [-tt.index(name), -tx.index(name)],
            color="#bbbbbb",
            lw=0.8,
            zorder=0,
        )
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(-TOP + 0.4, 2.4)
    ax.axis("off")
    ax.set_title(
        f"A  Die {TOP} wichtigsten Merkmale je Verfahren "
        f"({len(gemeinsam)} gemeinsam)",
        fontsize=10,
    )


def feld_b(ax, wt, wx, nullgruppe):
    v = wt.merge(wx, on="Merkmal", suffixes=("_t", "_x"))
    ist_null = v.Merkmal.isin(nullgruppe)
    benutzt = v.mittlerer_betrag_x > 0
    boden = float(v.loc[benutzt, "mittlerer_betrag_x"].min()) / 3
    x = v.mittlerer_betrag_x.where(benutzt, boden)
    for maske, farbe, marker, name, z in (
        (benutzt & ~ist_null, "#444444", "o", "von XGBoost benutzt", 3),
        (~benutzt & ~ist_null, "#bbbbbb", "<", "von XGBoost nie benutzt", 2),
        (ist_null, "#d62728", "<", "konstant im Train (wahrer Wert 0)", 4),
    ):
        s = maske.values
        if s.any():
            ax.scatter(
                x[s],
                v.mittlerer_betrag_t[s],
                s=30 if "konstant" in name else 16,
                alpha=0.9 if "konstant" in name else 0.5,
                marker=marker,
                color=farbe,
                edgecolors="none",
                label=f"{name} (n={int(s.sum())})",
                zorder=z,
            )
    ax.axvline(boden * 1.8, color="#cccccc", lw=0.8, ls=":")
    decke = float(v.mittlerer_betrag_t[ist_null].max()) if ist_null.any() else np.inf
    ueber = v.mittlerer_betrag_t > decke
    if ist_null.any():
        ax.axhline(decke, color="#d62728", lw=1, ls="--")
        ax.annotate(
            f"Rauschdecke der Nullgruppe: {decke:.3f}\n"
            f"darüber: {int(ueber.sum())} Merkmale",
            xy=(0.97, decke),
            xycoords=("axes fraction", "data"),
            ha="right",
            va="bottom",
            fontsize=7,
            color="#d62728",
        )
    rho = float(spearmanr(v.mittlerer_betrag_t, v.mittlerer_betrag_x).statistic)
    b = v[benutzt]
    rho_b = float(spearmanr(b.mittlerer_betrag_t, b.mittlerer_betrag_x).statistic)
    u = v[ueber]
    rho_ueber = (
        float(spearmanr(u.mittlerer_betrag_t, u.mittlerer_betrag_x).statistic)
        if len(u) > 3
        else float("nan")
    )
    sp50 = set(wt.head(50).Merkmal) | set(wx.head(50).Merkmal)
    s = v[v.Merkmal.isin(sp50)]
    rho_50 = float(spearmanr(s.mittlerer_betrag_t, s.mittlerer_betrag_x).statistic)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("XGBoost — mittlerer |Beitrag|")
    ax.set_ylabel("TabPFN — mittlerer |Beitrag|")
    ax.set_title("B  Dieselben Merkmale, beide Verfahren", fontsize=10)
    ax.annotate(
        f"ρ = {rho:+.3f}  alle {len(v)} Merkmale\n"
        f"ρ = {rho_b:+.3f}  nur die {int(benutzt.sum())} von XGBoost benutzten\n"
        f"ρ = {rho_ueber:+.3f}  nur die {int(ueber.sum())} über der Rauschdecke\n"
        f"ρ = {rho_50:+.3f}  Vereinigung beider Top-50 (n={len(s)}, "
        f"auswahlbedingt negativ)",
        xy=(0.03, 0.97),
        xycoords="axes fraction",
        va="top",
        fontsize=7,
        weight="bold",
        color="#333333",
        bbox=dict(fc="white", ec="#cccccc", alpha=0.95, pad=3),
        zorder=8,
    )
    ax.legend(fontsize=6.5, loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.3, which="both")
    return (
        rho,
        rho_b,
        rho_50,
        len(v),
        int(benutzt.sum()),
        float(decke) if ist_null.any() else float("nan"),
        int(ueber.sum()) if ist_null.any() else 0,
        rho_ueber,
    )


def feld_c(ax, wt, wx, n_merkmale, referenz):
    anteil = []
    for k in K_GITTER:
        ov = len(set(wt.head(k).Merkmal) & set(wx.head(k).Merkmal))
        anteil.append(ov / k)
    ax.plot(K_GITTER, anteil, "o-", color=F_TAB, lw=1.8, ms=6, label="TabPFN ↔ XGBoost")
    if referenz:
        rk = sorted(int(k) for k in referenz)
        ax.plot(
            rk,
            [referenz[str(k)] for k in rk],
            "s--",
            color=F_REF,
            lw=1.4,
            ms=5,
            label="XGBoost ↔ CatBoost (aus 03)",
        )
    ax.plot(
        K_GITTER,
        [k / n_merkmale for k in K_GITTER],
        ":",
        color="#d62728",
        lw=1.4,
        label="Zufallserwartung",
    )
    ax.set_xscale("log")
    ax.set_xticks(K_GITTER)
    ax.set_xticklabels([str(k) for k in K_GITTER], fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Listenlänge k")
    ax.set_ylabel("Anteil gemeinsamer Merkmale")
    ax.set_title("C  Überlappung der Top-k", fontsize=10)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.3)
    return {str(k): a for k, a in zip(K_GITTER, anteil)}


def feld_d(ax, rho_kreuz, stab_t, stab_x, rho_ref):
    namen = [
        "TabPFN\nmit sich selbst",
        "XGBoost\nmit sich selbst",
        "XGBoost ↔\nCatBoost (03)",
        "TabPFN ↔\nXGBoost",
    ]
    werte = [stab_t, stab_x, rho_ref, rho_kreuz]
    farben = [F_TAB, F_XGB, F_REF, "#d62728"]
    b = ax.bar(namen, werte, color=farben, edgecolor="white", linewidth=0.8)
    for balken, w in zip(b, werte):
        ax.text(
            balken.get_x() + balken.get_width() / 2,
            w + 0.03,
            f"{w:+.3f}",
            ha="center",
            fontsize=8.5,
            weight="bold",
        )
    ax.axhline(stab_t, color=F_TAB, ls="--", lw=1.2)
    ax.annotate(
        "Obergrenze für den Kreuzvergleich:\n"
        "die Eigenstabilität der schwächeren Seite",
        xy=(2.6, stab_t),
        xytext=(0.7, stab_t - 0.28),
        fontsize=7.5,
        color=F_TAB,
        arrowprops=dict(arrowstyle="->", color=F_TAB, lw=1),
    )
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Spearman ρ der Merkmalsränge")
    ax.set_title("D  Womit der Kreuzvergleich zu vergleichen ist", fontsize=10)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.grid(alpha=0.3, axis="y")


def feld_e(ax, wt, wx):
    """Konzentrationskurve: wieviele Merkmale tragen die Attributionsmasse?"""
    for w, farbe, name in ((wt, F_TAB, "TabPFN"), (wx, F_XGB, "XGBoost")):
        v = np.sort(np.asarray(w.mittlerer_betrag))[::-1]
        k = np.cumsum(v) / v.sum()
        ax.plot(np.arange(1, len(k) + 1) / len(k), k, color=farbe, lw=2, label=name)
        n80 = int(np.searchsorted(k, 0.8)) + 1
        ax.annotate(
            f"{name}: 80 % auf {n80} von {len(v)}",
            xy=(n80 / len(v), 0.8),
            xytext=(0.3, 0.5 if name == "TabPFN" else 0.86),
            fontsize=7.5,
            color=farbe,
            arrowprops=dict(arrowstyle="->", color=farbe, lw=1),
        )
    ax.plot([0, 1], [0, 1], ":", color="#d62728", lw=1.4, label="Gleichverteilung")
    ax.axhline(0.8, color="#999999", lw=0.8, ls="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Anteil der Merkmale (nach Wichtigkeit sortiert)")
    ax.set_ylabel("kumulierter Anteil der Attributionsmasse")
    ax.set_title("E  Wie konzentriert ist die Erklärung?", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.3)


def feld_f(ax, wt, wx, nullgruppe):
    """Der Prueftest: unterscheidet TabPFN benutzte von unbenutzten Merkmalen?"""
    from scipy.stats import mannwhitneyu

    v = wt.merge(wx, on="Merkmal", suffixes=("_t", "_x"))
    benutzt = v.mittlerer_betrag_x > 0
    ist_null = v.Merkmal.isin(nullgruppe)
    gruppen = [
        v.mittlerer_betrag_t[benutzt & ~ist_null].values,
        v.mittlerer_betrag_t[~benutzt & ~ist_null].values,
        v.mittlerer_betrag_t[ist_null].values,
    ]
    namen = [
        f"von XGBoost benutzt\n(n={len(gruppen[0])})",
        f"von XGBoost nie benutzt\n(n={len(gruppen[1])})",
        f"konstant im Train\nwahrer Wert 0  (n={len(gruppen[2])})",
    ]
    gruppen = [g for g in gruppen if len(g) > 1]
    namen = namen[: len(gruppen)]
    teile = ax.violinplot(gruppen, showmedians=True, widths=0.7)
    for koerper, farbe in zip(teile["bodies"], (F_XGB, "#bbbbbb", "#d62728")):
        koerper.set_facecolor(farbe)
        koerper.set_alpha(0.55)
    for schluessel in ("cmedians", "cbars", "cmins", "cmaxes"):
        if schluessel in teile:
            teile[schluessel].set_color("#333333")
    for i, g in enumerate(gruppen, start=1):
        ax.annotate(
            f"Median {np.median(g):.3f}",
            xy=(i, np.median(g)),
            xytext=(i + 0.18, np.median(g)),
            fontsize=7.5,
            va="center",
            weight="bold",
        )
    p = float(mannwhitneyu(gruppen[0], gruppen[1]).pvalue)
    p_null = (
        float(mannwhitneyu(gruppen[0], gruppen[2]).pvalue)
        if len(gruppen) > 2
        else float("nan")
    )
    ax.set_xticks(range(1, len(gruppen) + 1))
    ax.set_xticklabels(namen, fontsize=8)
    alle = np.concatenate(gruppen)
    alle = alle[alle > 0]
    ax.set_yscale("log")
    ax.set_ylim(alle.min() * 0.7, alle.max() * 1.6)
    ax.set_ylabel("TabPFN — mittlerer |Beitrag| (log)")
    ax.set_title("F  Trennt TabPFN Benutztes von Unbenutztem?", fontsize=10)
    ax.annotate(
        f"Mann-Whitney p = {p:.1e}\n"
        + (
            "kein Unterschied — die TabPFN-Rangliste\n"
            "unterscheidet nicht zwischen Merkmalen, die\n"
            "XGBoost benutzt, und solchen, die es ignoriert"
            if p >= 0.05
            else "Unterschied nachweisbar: TabPFN bewertet die\n"
            "von XGBoost benutzten Merkmale höher"
        ),
        xy=(0.5, 0.97),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=8,
        weight="bold",
        color="#d62728" if p >= 0.05 else "#2ca02c",
        bbox=dict(fc="white", ec="#cccccc", alpha=0.92, pad=3),
    )
    ax.grid(alpha=0.3, axis="y")
    return p, p_null


def main():
    Bt, zeilen, spalten_lauf, budget = lade_tabpfn()
    ds, X_tr, X_te = C.lade()

    if list(X_te.columns) != spalten_lauf:
        raise SystemExit(
            "Spaltenliste weicht vom gespeicherten TabPFN-Lauf ab "
            f"({X_te.shape[1]} heute gegen {len(spalten_lauf)} im Lauf). "
            "data_prep.py hat sich seit dem Lauf bewegt — Konfiguration "
            "pruefen (TEST_LEVEL, Y_MIN, Leckageliste) oder 05_tabpfn.py neu "
            "rechnen."
        )
    if max(zeilen) >= len(X_te):
        raise SystemExit(
            f"Zeilenindex {max(zeilen)} liegt ausserhalb des Test-Sets "
            f"({len(X_te)} Zeilen). TabPFN-Lauf und Split passen nicht zusammen."
        )

    mx, qx = C.xgb_modell(X_tr, ds.y_train)
    bx_alle, basis = C.shap_xgboost(mx, X_te)
    C.pruefe_additivitaet(bx_alle, basis, mx.predict(X_te), "XGBoost")
    Bx = bx_alle[zeilen]

    wt = C.globale_wichtigkeit(Bt, spalten_lauf)
    wx = C.globale_wichtigkeit(Bx, spalten_lauf)
    nullgruppe = [c for c in spalten_lauf if X_tr[c].nunique(dropna=False) <= 1]
    stab_t, stab_x = eigenstabilitaet(Bt), eigenstabilitaet(Bx)
    wied_rho, wied_top20, n_wied = wiederholziehung(zeilen)

    ref = json.loads((C.ERG_DIR / C.mit_ebene("03_vergleich.json")).read_text("utf-8"))
    print(
        f"  TabPFN:  {len(Bt)} Billets, budget={budget}, Eigenstabilität ρ={stab_t:+.3f}"
    )
    print(f"  XGBoost: dieselben {len(Bx)} Billets, Eigenstabilität ρ={stab_x:+.3f}")
    print(f"  Nullgruppe: {len(nullgruppe)} im Train konstante Spalten")

    pdf = C.pdf_oeffnen("06_tabpfn_vs_xgboost.pdf")
    fig, ax = C.seite(
        "Stimmen TabPFN und XGBoost in der Merkmalswichtigkeit überein?", 2, 2
    )
    feld_a(ax[0], wt, wx)
    rho, rho_b, rho_50, n_m, n_benutzt, decke, n_ueber, rho_ueber = feld_b(
        ax[1], wt, wx, nullgruppe
    )
    ueb = feld_c(ax[2], wt, wx, n_m, ref.get("ueberlappung_top_k"))
    feld_d(ax[3], rho, stab_t, stab_x, ref["spearman_rho"])
    fig.savefig(
        ROOT / "reports" / "figures" / "tabpfn_vs_xgboost.png",
        dpi=200,
        bbox_inches="tight",
    )
    fuss = (
        f"{len(Bt)} Billets aus dem Test-Set, für beide Verfahren dieselben | "
        f"{n_m} Merkmale, davon {len(nullgruppe)} im Train konstant | "
        f"TabPFN budget={budget}, KV-Cache, Basis = Spaltenmedian | "
        f"XGBoost {qx} | Test-Ebene {C.D.TEST_LEVEL}"
    )
    C.schliesse(pdf, fig, fuss)

    import matplotlib.pyplot as plt

    fig_a, ax_a = plt.subplots(figsize=(6.0, 4.4))
    feld_a(ax_a, wt, wx)
    ziel = ROOT / "reports" / "figures" / "kapitel7" / "abb_7_tabpfn_xgboost_top20.png"
    ziel.parent.mkdir(parents=True, exist_ok=True)
    fig_a.savefig(ziel, dpi=200, bbox_inches="tight")
    plt.close(fig_a)

    fig2, ax2 = C.seite(
        "Wieviel Signal trägt die TabPFN-Attribution? — Diagnose", 1, 2
    )
    feld_e(ax2[0], wt, wx)
    p_trenn, p_null = feld_f(ax2[1], wt, wx, nullgruppe)
    fig2.savefig(
        ROOT / "reports" / "figures" / "tabpfn_vs_xgboost_diagnose.png",
        dpi=200,
        bbox_inches="tight",
    )
    C.schliesse(pdf, fig2, fuss)

    top20 = len(set(wt.head(20).Merkmal) & set(wx.head(20).Merkmal))
    top5 = len(set(wt.head(5).Merkmal) & set(wx.head(5).Merkmal))
    ist_null = wt.Merkmal.isin(nullgruppe)
    med_null = float(wt.mittlerer_betrag[ist_null].median())
    med_rest = float(wt.mittlerer_betrag[~ist_null].median())
    v_t = np.sort(np.asarray(wt.mittlerer_betrag))[::-1]
    n80_t = int(np.searchsorted(np.cumsum(v_t) / v_t.sum(), 0.8)) + 1
    v_x = np.sort(np.asarray(wx.mittlerer_betrag))[::-1]
    n80_x = int(np.searchsorted(np.cumsum(v_x) / v_x.sum(), 0.8)) + 1
    summe_t = float(np.abs(Bt).sum(axis=1).mean())
    summe_x = float(np.abs(Bx).sum(axis=1).mean())
    C.textseite(
        pdf,
        "Lesehinweise und Vorbehalte",
        [
            f"Ergebnis. Rangkorrelation über alle {n_m} Merkmale: ρ = {rho:+.3f}. "
            f"Zum Vergleich erreichen XGBoost und CatBoost in 03 auf derselben "
            f"Aufgabe ρ = {ref['spearman_rho']:+.3f}. Von den 20 wichtigsten "
            f"Merkmalen teilen sich TabPFN und XGBoost {top20}, von den ersten "
            f"fünf {top5}. Beschränkt auf die {n_ueber} Merkmale, die sich bei "
            f"TabPFN überhaupt vom Rauschen abheben, bleibt die Korrelation mit "
            f"ρ = {rho_ueber:+.3f} positiv. Die Übereinstimmung ist damit kein "
            f"Artefakt der vielen unbenutzten Merkmale, aber auch deutlich "
            f"schwächer als zwischen zwei Baummodellen.",
            f"Zur Zahl ρ = {rho_50:+.3f} auf der Vereinigung beider Top-50 in "
            f"Feld B. Sie ist KEIN eigenständiger Widerspruchsbeleg: die Auswahl "
            f"bedingt auf die Extreme beider Listen, und ein Merkmal gelangt "
            f"meist deshalb hinein, weil genau eine der beiden Listen es hoch "
            f"führt. Dieser Auswahleffekt erzeugt eine negative Korrelation auch "
            f"dann, wenn beide Ranglisten insgesamt positiv zusammenhängen. "
            f"Massgeblich ist ρ über die Merkmale über der Rauschdecke.",
            f"Wieviel davon ist Rauschen? Das beantwortet die mitgerechnete "
            f"Nullgruppe: {len(nullgruppe)} der {n_m} Spalten sind im "
            f"Trainingssatz konstant, TabPFN entfernt sie in der Vorverarbeitung, "
            f"ihr wahrer Shapley-Wert ist exakt 0. Die Approximation schreibt "
            f"ihnen im Median {med_null:.3f} Prozentpunkte zu und höchstens "
            f"{decke:.3f}. Über dieser Rauschdecke stehen {n_ueber} Merkmale, "
            f"das sind {n_ueber / n_m:.0%} des Merkmalssatzes. Der Median aller "
            f"übrigen Merkmale liegt mit {med_rest:.3f} dagegen praktisch auf "
            f"dem Rauschboden — die Rangfolge im Mittelfeld ist bedeutungslos, "
            f"die Spitzengruppe hebt sich ab.",
            f"Bindungen. {n_m - n_benutzt} der {n_m} Merkmale kommen in keinem "
            f"XGBoost-Split vor und stehen alle auf 0, was Spearman verzerrt. "
            f"Beschränkt man die Korrelation auf die {n_benutzt} tatsächlich "
            f"benutzten Merkmale, ändert sie sich von {rho:+.3f} auf "
            f"{rho_b:+.3f}.",
            f"Feld D — der Massstab für den Kreuzvergleich. Zieht man die "
            f"{len(Bt)} erklärten Billets mit Zurücklegen neu und vergleicht zwei "
            f"solche Ziehungen, erreicht XGBoost ρ = {stab_x:+.3f}, TabPFN "
            f"ρ = {stab_t:+.3f}. Kein Kreuzvergleich kann besser ausfallen als "
            f"die schwächere dieser beiden Zahlen. ACHTUNG: dieser Bootstrap "
            f"zieht dieselben Schätzungen neu und misst nur die Streuung "
            f"ZWISCHEN Billets. Das Koalitionsrauschen innerhalb eines Billets "
            + (
                f"zeigt erst die zweite unabhängige Ziehung aus 05 "
                f"({n_wied} Billets): ρ = {wied_rho:+.3f} über alle Merkmale, "
                f"{wied_top20:.1f} von 20 Spitzenmerkmalen gemeinsam. Die "
                f"Eigenstabilität in Feld D ist daher eine obere Schranke, "
                f"keine Messung."
                if n_wied
                else "sieht er nicht; dafür fehlt im Lauf die Zweitziehung."
            ),
            f"Seite 2, Feld F — trennt TabPFN Benutztes von Unbenutztem? XGBoost "
            f"teilt die Merkmale hart in zwei Lager: {n_benutzt} kommen in "
            f"mindestens einem Split vor, {n_m - n_benutzt} in keinem einzigen. "
            + (
                f"TabPFN bewertet das erste Lager höher, der Unterschied ist mit "
                f"p = {p_trenn:.1e} nachweisbar. Die TabPFN-Rangliste trägt also "
                f"Information darüber, welche Merkmale zählen — auch wenn sie "
                f"eine andere Reihenfolge ergibt als TreeSHAP."
                if p_trenn < 0.05
                else f"TabPFN tut das nicht — der Unterschied ist mit "
                f"p = {p_trenn:.2f} nicht nachweisbar. Die TabPFN-Rangliste "
                f"trägt damit keine erkennbare Information darüber, welche "
                f"Merkmale überhaupt zählen."
            ),
            f"Seite 2, Feld E — Konzentration. Bei XGBoost liegen 80 Prozent der "
            f"Attributionsmasse auf {n80_x} Merkmalen, bei TabPFN auf {n80_t} "
            f"von {n_m}. Je näher die Kurve an der Gleichverteilung liegt, desto "
            f"weniger unterscheidet die Erklärung überhaupt. Die Beträge "
            f"summieren sich je Billet im Mittel auf {summe_t:.1f} "
            f"Prozentpunkte bei TabPFN gegen {summe_x:.1f} bei XGBoost, bei "
            f"einem y-Bereich von 1 bis 100: die Einzelbeiträge heben sich bei "
            f"TabPFN stärker gegenseitig auf. Das ist erwartbar, denn shapiq "
            f"schätzt mit budget={budget} Koalitionen über {n_m} Merkmale, "
            f"während TreeSHAP exakt rechnet.",
            f"Was daraus folgt. Zitierfähig ist die TabPFN-Spitzengruppe — die "
            f"{n_ueber} Merkmale über der Rauschdecke der Nullgruppe —, nicht "
            f"die Rangfolge über alle {n_m} Merkmale. Für den Vergleich mit "
            f"XGBoost heisst das: Übereinstimmung oder Abweichung ist nur "
            f"innerhalb dieser Spitzengruppe interpretierbar. Der Befund aus 03 "
            f"(XGBoost gegen CatBoost, ρ = {ref['spearman_rho']:.2f}) bleibt der "
            f"belastbarere Beleg dafür, dass eine Rangfolge in den Daten liegt "
            f"und nicht im Algorithmus; TabPFN ist die dritte Stimme mit der "
            f"grössten Unsicherheit.",
            f"Datenstand. Dieser Vergleich läuft auf demselben Stand von "
            f"data_prep.py wie der TabPFN-Lauf vom 18.09.2026 "
            f"(Y_MIN = {C.D.Y_MIN:g}, "
            f"TEST_LEVEL = {C.D.TEST_LEVEL}). Das Skript prüft die Spaltenliste "
            f"gegen den gespeicherten Lauf und bricht bei Abweichung ab. Die "
            f"Beträge sind mit Läufen, deren Basisvektor auf dem Spaltenmittel lag, "
            f"NICHT vergleichbar.",
        ],
    )
    pdf.close()

    p = C.speichere(
        "06_tabpfn_vs_xgboost.json",
        {
            "n_billets": len(Bt),
            "n_merkmale": n_m,
            "budget": budget,
            "datenstand": {
                "y_min": C.D.Y_MIN,
                "test_level": C.D.TEST_LEVEL,
                "tabpfn_lauf": TABPFN_LAUF.name,
            },
            "spearman_rho": rho,
            "spearman_rho_nur_benutzte": rho_b,
            "spearman_rho_ueber_rauschdecke": rho_ueber,
            "spearman_rho_top50_vereinigung": rho_50,
            "n_von_xgboost_benutzt": n_benutzt,
            "p_tabpfn_trennt_benutzt_unbenutzt": p_trenn,
            "nullgruppe": {
                "n": len(nullgruppe),
                "median": med_null,
                "rauschdecke": decke,
                "n_merkmale_ueber_decke": n_ueber,
                "median_uebrige_merkmale": med_rest,
                "p_benutzt_gegen_nullgruppe": p_null,
            },
            "zweitziehung": {
                "n_billets": n_wied,
                "rho": wied_rho,
                "top20_gemeinsam": wied_top20,
            },
            "eigenstabilitaet": {"tabpfn": stab_t, "xgboost": stab_x},
            "referenz_xgb_catboost": ref["spearman_rho"],
            "top20_gemeinsam": top20,
            "ueberlappung_top_k": ueb,
            "top20_tabpfn": wt.head(20)[["Merkmal", "mittlerer_betrag"]].to_dict(
                "records"
            ),
            "top20_xgboost": wx.head(20)[["Merkmal", "mittlerer_betrag"]].to_dict(
                "records"
            ),
            "rangliste_csv": C.mit_ebene("06_tabpfn_vs_xgboost.csv"),
        },
    )
    tab = (
        wt[["Merkmal", "mittlerer_betrag"]]
        .rename(columns={"mittlerer_betrag": "tabpfn"})
        .merge(
            wx[["Merkmal", "mittlerer_betrag"]].rename(
                columns={"mittlerer_betrag": "xgboost"}
            ),
            on="Merkmal",
        )
    )
    tab["rang_tabpfn"] = tab.tabpfn.rank(ascending=False, method="min").astype(int)
    tab["rang_xgboost"] = tab.xgboost.rank(ascending=False, method="min").astype(int)
    tab["konstant_im_train"] = tab.Merkmal.isin(nullgruppe)
    tab["ueber_rauschdecke"] = tab.tabpfn > decke
    tab["von_xgboost_benutzt"] = tab.xgboost > 0
    pcsv = C.speichere_rangliste(
        C.mit_ebene("06_tabpfn_vs_xgboost.csv"), tab, "tabpfn"
    )
    print(f"geschrieben: {pcsv}")
    print(f"geschrieben: {C.OUT_DIR / '06_tabpfn_vs_xgboost.pdf'}")
    print(f"geschrieben: {p}")
    print(
        f"  ρ(TabPFN, XGBoost) = {rho:+.3f}   Referenz ρ(XGB, CatBoost) = {ref['spearman_rho']:+.3f}"
    )
    print(
        f"  Rauschdecke der Nullgruppe {decke:.3f} → {n_ueber} von {n_m} "
        f"Merkmalen heben sich ab; Feld F p = {p_trenn:.1e}"
    )


if __name__ == "__main__":
    main()
