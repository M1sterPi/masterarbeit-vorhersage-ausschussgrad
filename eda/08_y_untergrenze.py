"""08 — Untere Grenze von y.

In der Arbeit: Abschnitt 4.1.3: Untergrenze des Ausschussgrads, Abbildung 4.

Begründung des unteren y-Filters.
Ausgabe: reports/eda/08_y_untergrenze.pdf, reports/figures/y_untergrenze*.png,
y_verteilung_balken.png.

    python eda/08_y_untergrenze.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import _common as C
from data_prep import SP  # noqa: E402

Y_MAX = 100.0
SCHRITT = 0.01
KANDIDATEN = [0.001, 0.01, 0.1]
ZOOM_MAX = 5.0
# Faustregel der Fertigung, fester Abzug und erwartete Lücken stehen in data/spalten.json
VORSCHLAG_FIRMA = SP["faustregel_y"]
VORSCHLAG_DATEN = 1.0
ANTEIL_FEST = SP["anteil_fest"]
LUECKEN_ERWARTET = tuple(map(tuple, SP["luecken_erwartet"]))
FARBE_FEST = "#ff7f0e"
FARBE_TICK = "#d62728"
FARBE_ECHT = "#3182bd"


def lade_roh() -> pd.DataFrame:
    """y aus der Merkmalstabelle, Massen aus den Stammdaten — ohne y-Filter."""
    mst = pd.read_csv(C.ROOT / "data" / "billets_master.csv", low_memory=False)
    frm = pd.read_csv(C.CSV_PATH, usecols=["batch_id", "y"], low_memory=False)
    mst["batch_id"] = mst["batch_id"].astype(str)
    frm["batch_id"] = frm["batch_id"].astype(str)
    d = frm.merge(
        mst[["batch_id", SP["masse_soll"], SP["masse_ist"], SP["artikel"]]],
        on="batch_id",
        how="left",
    )
    for c in ("y", SP["masse_soll"], SP["masse_ist"]):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["diff"] = (d[SP["masse_soll"]] - d[SP["masse_ist"]]).round(4)
    d["schritte"] = (d["diff"] / SCHRITT).round()
    d = d[d.y.between(0, Y_MAX)].copy()
    d["gruppe"] = np.select(
        [d.schritte <= 2, d.y < VORSCHLAG_DATEN],
        ["1 Skalenschritte", "2 fester Abzug"],
        default="3 Messung",
    )
    return d


def feld_a(ax, d):
    """Häufigkeit jeder Massendifferenz bis 2 kg — die Quantisierung."""
    v = d[d["diff"].between(0, 2.0)]["diff"].value_counts().sort_index()
    tick = v.index <= 2 * SCHRITT
    ax.bar(
        v.index[tick],
        v.values[tick],
        width=0.009,
        color=FARBE_TICK,
        label="0, 1 oder 2 Skalenschritte",
    )
    ax.bar(
        v.index[~tick],
        v.values[~tick],
        width=0.009,
        color=FARBE_ECHT,
        label="übrige Billets",
    )
    ax.set_yscale("log")
    ax.set_ylim(0.7, v.max() * 8)
    ax.set_xlim(-0.06, 1.95)
    ax.set_xlabel("Sollmasse − Istmasse [kg]")
    ax.set_ylabel("Billets (log)")
    ax.set_title("A  Massendifferenz je Billet", fontsize=10)
    ax.axvspan(0.025, 0.205, color="#999999", alpha=0.18, label="leere Zone")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(alpha=0.3, axis="y")


def feld_b(ax, d):
    """Dieselbe Lücke in y — und wo die drei Kandidaten schneiden."""
    tick = d.schritte <= 2
    pos = d[(d.y > 0) & (d.y <= 1)]
    rng = np.random.default_rng(C.SEED)
    for maske, farbe, name in (
        (tick, FARBE_TICK, "1–2 Skalenschritte"),
        (~tick, FARBE_ECHT, "übrige Billets"),
    ):
        s = pos[maske.reindex(pos.index, fill_value=False)]
        ax.scatter(
            s.y,
            rng.uniform(-1, 1, len(s)),
            s=14,
            alpha=0.65,
            color=farbe,
            label=name,
            zorder=3,
        )
    ax.set_xscale("log")
    ax.set_xlim(6e-4, 1.2)
    ax.set_ylim(-3.6, 3.6)
    ax.set_yticks([])
    ax.set_xlabel("Ausschussgrad y [%], logarithmisch")
    ax.set_title("B  Kandidaten für die Untergrenze", fontsize=10)
    gmax = d.loc[tick, "y"].max()
    gmin = d.loc[~tick & (d.y > 0), "y"].min()
    ax.axvspan(gmax, gmin, color="#2ca02c", alpha=0.13, zorder=1, label="leere Zone")
    texte = {k: f"{k:g}" for k in KANDIDATEN}
    for k in KANDIDATEN:
        gut = gmax < k < gmin
        ax.axvline(
            k,
            color="#2ca02c" if gut else "#d62728",
            lw=2,
            ls="-" if gut else "--",
            zorder=4,
        )
        ax.annotate(
            texte[k],
            xy=(k, -3.2),
            xytext=(3, 0),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=8,
            color="#2ca02c" if gut else "#d62728",
        )
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(alpha=0.3, axis="x")


def feld_c(ax, d):
    """Entfernte Zeilen als Funktion der Schwelle — das Plateau."""
    gitter = np.logspace(-4, 0, 600)
    n = [int((d.y < s).sum()) for s in gitter]
    ax.step(gitter, n, where="post", color="black", lw=1.6)
    ax.set_xscale("log")
    ax.set_xlabel("Untergrenze Y_MIN [%], logarithmisch")
    ax.set_ylabel("entfernte Billets")
    ax.set_title("C  Entfernte Billets je Untergrenze", fontsize=10)
    plateau = int((d.schritte <= 2).sum())
    ax.axhline(plateau, color="#999999", ls=":", lw=1)
    gmax = d.loc[d.schritte <= 2, "y"].max()
    gmin = d.loc[(d.schritte > 2) & (d.y > 0), "y"].min()
    ax.axvspan(gmax, gmin, color="#2ca02c", alpha=0.15)
    ax.set_ylim(40, plateau * 2.2)
    for k, dx, dy in zip(KANDIDATEN, (8, 8, 0), (-14, -14, 16)):
        weg = int((d.y < k).sum())
        gut = gmax < k < gmin
        ax.plot([k], [weg], "o", ms=8, color="#2ca02c" if gut else "#d62728", zorder=5)
        ax.annotate(
            f"{k:g}".replace(".", ",") + f" → {weg}",
            xy=(k, weg),
            textcoords="offset points",
            xytext=(dx, dy),
            ha="left" if dx else "center",
            fontsize=8,
            color="#2ca02c" if gut else "#d62728",
        )
    ax.grid(alpha=0.3)


def feld_d(ax, d):
    """Ein Skalenschritt in y-Einheiten — die Auflösungsgrenze."""
    ms = np.linspace(max(d[SP["masse_soll"]].min(), 20), d[SP["masse_soll"]].quantile(0.995), 300)
    ax.plot(
        ms,
        SCHRITT / ms * 100,
        color="black",
        lw=1.6,
        label=f"ein Skalenschritt ({SCHRITT:g} kg)".replace(".", ","),
    )
    ax.plot(
        ms,
        2 * SCHRITT / ms * 100,
        color="#666666",
        lw=1,
        ls="--",
        label="zwei Skalenschritte",
    )
    betroffen = d[(d.schritte <= 2) & (d.y > 0)]
    ax.scatter(
        betroffen[SP["masse_soll"]],
        betroffen.y,
        s=22,
        color=FARBE_TICK,
        zorder=4,
        label="Billets mit 1–2 Skalenschritten",
    )
    for k in KANDIDATEN:
        gut = k == 0.1
        ax.axhline(
            k, color="#2ca02c" if gut else "#d62728", lw=1.6, ls="-" if gut else "--"
        )
        ax.annotate(
            "Y_MIN = " + f"{k:g}".replace(".", ","),
            xy=(ms[-1], k),
            ha="right",
            va="bottom",
            fontsize=7.5,
            color="#2ca02c" if gut else "#d62728",
        )
    ax.set_yscale("log")
    ax.set_ylim(7e-4, 1)
    ax.set_xlabel("Sollmasse [kg]")
    ax.set_ylabel("Ausschussgrad y [%], logarithmisch")
    ax.set_title("D  Ein Skalenschritt in y-Einheiten", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(alpha=0.3)


_G_FARBE = {
    "1 Skalenschritte": FARBE_TICK,
    "2 fester Abzug": FARBE_FEST,
    "3 Messung": FARBE_ECHT,
}
_G_NAME = {
    "1 Skalenschritte": "Gruppe 1 — 0–2 Skalenschritte",
    "2 fester Abzug": "Gruppe 2 — fester Abzug",
    "3 Messung": "Gruppe 3 — streuende Messung",
}


def luecken(d):
    """Die beiden leeren Zonen, aus den Daten ABGELESEN statt gesetzt."""
    paare = (("1 Skalenschritte", "2 fester Abzug"), ("2 fester Abzug", "3 Messung"))
    return tuple(
        (float(d.loc[d.gruppe == a, "y"].max()), float(d.loc[d.gruppe == b, "y"].min()))
        for a, b in paare
    )


def _schwellen(ax, beschriften=True, hoehe=0.5):
    """Die beiden diskutierten Schwellen, in jedem Zoom-Feld gleich gezeichnet."""
    for v, farbe, txt in (
        (VORSCHLAG_DATEN, "#2ca02c", "Y_MIN = 1"),
        (VORSCHLAG_FIRMA, "#d62728", f"{VORSCHLAG_FIRMA:g} % (Faustregel)"),
    ):
        ax.axvline(v, color=farbe, lw=1.6, zorder=6)
        if beschriften:
            ax.annotate(
                txt,
                xy=(v, hoehe),
                xycoords=("data", "axes fraction"),
                xytext=(-4, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                rotation=90,
                fontsize=7,
                color=farbe,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1),
            )


def feld_e(ax, d):
    """Jedes einzelne Billet unter 5 % — die Luecken werden direkt sichtbar."""
    z = d[d.y < ZOOM_MAX]
    rng = np.random.default_rng(C.SEED)
    for g in sorted(z.gruppe.unique()):
        s = z[z.gruppe == g]
        ax.scatter(
            s.y,
            rng.uniform(-1, 1, len(s)),
            s=26,
            alpha=0.6,
            color=_G_FARBE[g],
            edgecolors="none",
            label=_G_NAME[g],
            zorder=3,
        )
    for nr, (lo, hi) in enumerate(luecken(d)):
        ax.axvspan(lo, hi, color="#2ca02c", alpha=0.14, zorder=1,
                   label="leere Zone" if nr == 0 else None)
    ax.set_xlim(-0.12, ZOOM_MAX + 0.05)
    ax.set_ylim(-1.6, 2.6)
    ax.set_yticks([])
    ax.set_xlabel("Ausschussgrad y [%]")
    ax.set_title("E  Alle Billets unter 5 %", fontsize=10)
    _schwellen(ax, hoehe=0.82)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.3, axis="x")


def feld_f(ax, d):
    """Feines Histogramm — wo die Masse unterhalb 5 % wirklich sitzt."""
    z = d[d.y < ZOOM_MAX]
    kanten = np.arange(0, ZOOM_MAX + 0.05, 0.05)
    unten = np.zeros(len(kanten) - 1)
    for g in sorted(z.gruppe.unique()):
        h, _ = np.histogram(z[z.gruppe == g].y, bins=kanten)
        ax.bar(
            kanten[:-1],
            h,
            width=0.048,
            bottom=unten,
            align="edge",
            color=_G_FARBE[g],
            label=_G_NAME[g],
        )
        unten += h
    hoch = max(unten.max(), 1)
    ax.set_yscale("log")
    ax.set_ylim(0.7, hoch * 14)
    ax.set_xlim(-0.12, ZOOM_MAX)
    ax.set_xlabel("Ausschussgrad y [%], Klassenbreite 0,05 Prozentpunkte")
    ax.set_ylabel("Billets (log)")
    ax.set_title("F  Häufigkeit unter 5 %", fontsize=10)
    _schwellen(ax)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.3, axis="y")


def feld_g(ax, d):
    """Differenz gegen Sollmasse: konstanter Abzug gegen echte Streuung."""
    z = d[d.y < ZOOM_MAX]
    for g in sorted(z.gruppe.unique()):
        s = z[z.gruppe == g]
        ax.scatter(
            s[SP["masse_soll"]],
            s["diff"],
            s=30,
            alpha=0.7,
            color=_G_FARBE[g],
            edgecolors="none",
            label=_G_NAME[g],
            zorder=3,
        )
    hoch = z["diff"].max() if len(z) else 1.0
    xmax = z[SP["masse_soll"]].max() * 1.08
    ms = np.linspace(0, xmax, 200)
    for anteil, stil, txt in (
        (ANTEIL_FEST, "-", f"{100 * ANTEIL_FEST:g} % der Sollmasse".replace(".", ",")),
        (VORSCHLAG_FIRMA / 100, ":", f"{VORSCHLAG_FIRMA:g} % — Faustregel"),
    ):
        ax.plot(ms, anteil * ms, color="#555555", lw=1.2, ls=stil, zorder=2)
        y_rand = min(anteil * xmax, hoch * 1.02)
        oben = y_rand < anteil * xmax - 1e-9
        ax.annotate(
            txt,
            xy=(y_rand / anteil, y_rand),
            xytext=(4, -3) if oben else (-3, 4),
            textcoords="offset points",
            ha="left" if oben else "right",
            va="top" if oben else "bottom",
            fontsize=7,
            color="#555555",
        )
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.6, hoch * 1.14)
    ax.set_xlabel("Sollmasse [kg]")
    ax.set_ylabel("Sollmasse − Istmasse [kg]")
    ax.set_title("G  Massendifferenz gegen Sollmasse", fontsize=10)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.3)


def feld_h(ax, d):
    """Die Schwellenkurve linear — das zweite Plateau und was 5 % kostet."""
    gitter = np.linspace(0, ZOOM_MAX, 1200)
    n = [int((d.y < s).sum()) for s in gitter]
    ax.step(gitter, n, where="post", color="black", lw=1.6, zorder=4)
    lo2, hi2 = luecken(d)[1]
    ax.axvspan(lo2, hi2, color="#2ca02c", alpha=0.15, zorder=1)
    for v, farbe, dy in (
        (VORSCHLAG_DATEN, "#2ca02c", 12),
        (VORSCHLAG_FIRMA, "#d62728", -4),
    ):
        weg = int((d.y < v).sum())
        ax.plot([v], [weg], "o", ms=9, color=farbe, zorder=6)
        ax.annotate(
            f"{v:g} % → {weg}",
            xy=(v, weg),
            textcoords="offset points",
            xytext=(12, dy),
            ha="left",
            fontsize=8,
            color=farbe,
        )
    hoch = int((d.y < ZOOM_MAX).sum())
    ax.set_xlim(0, ZOOM_MAX + 0.9)
    ax.set_ylim(100, hoch * 1.15)
    ax.set_xlabel("Untergrenze Y_MIN [%]")
    ax.set_ylabel("entfernte Billets")
    ax.set_title("H  Entfernte Billets je Untergrenze", fontsize=10)
    ax.grid(alpha=0.3)


BREITE_VOLL = 2.0
BREITE_ZOOM = 0.5
ZOOM2_MAX = 15.0


def _histogramm(ax, d, bis, breite, log):
    """Gestapeltes Histogramm nach Gruppe, mit fuellenden Balken."""
    z = d[d.y <= bis]
    kanten = np.arange(0, bis + breite, breite)
    unten = np.zeros(len(kanten) - 1)
    for g in sorted(z.gruppe.unique()):
        h, _ = np.histogram(z[z.gruppe == g].y, bins=kanten)
        ax.bar(
            kanten[:-1],
            h,
            width=breite,
            bottom=unten,
            align="edge",
            color=_G_FARBE[g],
            edgecolor="white",
            linewidth=0.6,
            label=_G_NAME[g],
        )
        unten += h
    ax.set_xlim(-breite * 0.6, bis + breite * 0.6)
    if log:
        ax.set_yscale("log")
        ax.set_ylim(0.7, max(unten.max(), 1) * 9)
        ax.set_ylabel("Billets (log)")
    else:
        ax.set_ylim(0, max(unten.max(), 1) * 1.15)
        ax.set_ylabel("Billets")
    ax.set_xlabel(f"y [%], Klassenbreite {breite:g} Prozentpunkte")
    ax.grid(alpha=0.3, axis="y")
    return unten, kanten


def feld_i(ax, d):
    """Ganzer Bereich, lineare Haeufigkeit — wo die Masse der Billets liegt."""
    unten, kanten = _histogramm(ax, d, 100.0, BREITE_VOLL, log=False)
    ax.set_title("I  y, ganzer Bereich, linear", fontsize=10)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)


def feld_j(ax, d):
    """Ganzer Bereich, logarithmisch — die Randgruppen werden sichtbar."""
    _histogramm(ax, d, 100.0, BREITE_VOLL, log=True)
    ax.set_title("J  y, ganzer Bereich, logarithmisch", fontsize=10)
    ax.axvspan(0, ZOOM2_MAX, color="#999999", alpha=0.13, zorder=0, label="Ausschnitt K/L")
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)


def feld_k(ax, d):
    """Ausschnitt bis 15 %, linear — der eigentliche Arbeitsbereich."""
    unten, kanten = _histogramm(ax, d, ZOOM2_MAX, BREITE_ZOOM, log=False)
    ax.set_title(f"K  Ausschnitt 0–{ZOOM2_MAX:g} %, linear", fontsize=10)
    _schwellen(ax, hoehe=0.62)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)


def feld_l(ax, d):
    """Ausschnitt bis 15 %, logarithmisch — beide Artefaktgruppen im Bild."""
    _histogramm(ax, d, ZOOM2_MAX, BREITE_ZOOM, log=True)
    ax.set_title(f"L  Ausschnitt 0–{ZOOM2_MAX:g} %, logarithmisch", fontsize=10)
    _schwellen(ax, hoehe=0.42)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)


def abbildung_schwelle(d):
    """Einzelabbildung für die Arbeit: Feld K ohne Gruppenfarben."""
    fig, ax = C.plt.subplots(figsize=(6.3, 3.4))
    z = d[d.y <= ZOOM2_MAX]
    kanten = np.arange(0, ZOOM2_MAX + BREITE_ZOOM, BREITE_ZOOM)
    h, _ = np.histogram(z.y, bins=kanten)
    ax.bar(
        kanten[:-1],
        h,
        width=BREITE_ZOOM,
        align="edge",
        color=FARBE_ECHT,
        edgecolor="white",
        linewidth=0.6,
    )
    n_weg = int((d.y < VORSCHLAG_DATEN).sum())
    ax.axvline(VORSCHLAG_DATEN, color="#d62728", lw=1.6, ls="--", zorder=6)
    ax.annotate(
        f"Untergrenze y = {VORSCHLAG_DATEN:g} %\n{n_weg} Billets entfernt",
        xy=(VORSCHLAG_DATEN, 0.93),
        xycoords=("data", "axes fraction"),
        xytext=(6, 0),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=8,
        color="#d62728",
    )
    ax.set_xlim(-BREITE_ZOOM * 0.6, ZOOM2_MAX + BREITE_ZOOM * 0.6)
    ax.set_ylim(0, h.max() * 1.15)
    breite = f"{BREITE_ZOOM:g}".replace(".", ",")
    ax.set_xlabel(f"y [%], Klassenbreite {breite} Prozentpunkte")
    ax.set_ylabel("Anzahl Billets")
    ax.grid(alpha=0.3, axis="y")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(
        C.ROOT / "reports" / "figures" / "y_untergrenze_1prozent.png",
        dpi=300,
        bbox_inches="tight",
    )
    C.plt.close(fig)


def main():
    d = lade_roh()
    abbildung_schwelle(d)
    if abs(C.Y_MIN - VORSCHLAG_DATEN) > 1e-9:
        raise SystemExit(
            f"data_prep rechnet Y_MIN={C.Y_MIN:g}, diese Seite begruendet "
            f"{VORSCHLAG_DATEN:g}. Entweder VORSCHLAG_DATEN nachziehen und die "
            f"Textseiten pruefen, oder data_prep.Y_MIN zuruecksetzen."
        )
    for (lo, hi), (e_lo, e_hi) in zip(luecken(d), LUECKEN_ERWARTET):
        if abs(lo - e_lo) > 0.01 or abs(hi - e_hi) > 0.01:
            raise SystemExit(
                f"Luecke liegt jetzt bei ({lo:.4f}, {hi:.4f}), erwartet war "
                f"({e_lo}, {e_hi}). LUECKEN_ERWARTET und die Textseite "
                f"nachziehen."
            )
    tick = d.schritte <= 2
    gmax = d.loc[tick, "y"].max()
    gmin = d.loc[~tick & (d.y > 0), "y"].min()
    teil = tick & (d.y > 0)
    n_teil = int(teil.sum())
    n_art_tick = int(d.loc[teil, SP["artikel"]].nunique())
    n_art = int(d[SP["artikel"]].nunique())

    pdf = C.pdf_oeffnen("08_y_untergrenze.pdf")
    fig, ax = C.seite(
        pdf, "Erste Stufe: Skalenschritt-Artefakte direkt über y = 0", 2, 2
    )
    feld_a(ax[0], d)
    feld_b(ax[1], d)
    feld_c(ax[2], d)
    feld_d(ax[3], d)
    png = C.ROOT / "reports" / "figures" / "y_untergrenze.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    C.aufraeumen(fig)
    fig.savefig(png, dpi=200, bbox_inches="tight")

    C.schliesse(
        pdf,
        fig,
        f"{C.CSV_PATH.name} + billets_master.csv | alle {len(d)} Billets "
        f"mit 0 ≤ y ≤ {Y_MAX:g} (ohne y-Filter, sonst wären genau die "
        f"fraglichen Zeilen fort) | Istmasse mit zwei Nachkommastellen",
    )

    fig2, ax2 = C.seite(
        pdf, "Zweite Stufe: der Bereich unter 5 %", 2, 2
    )
    feld_e(ax2[0], d)
    feld_f(ax2[1], d)
    feld_g(ax2[2], d)
    feld_h(ax2[3], d)
    C.aufraeumen(fig2)
    fig2.savefig(
        C.ROOT / "reports" / "figures" / "y_untergrenze_zoom.png",
        dpi=200,
        bbox_inches="tight",
    )
    n_g2 = int((d.gruppe == "2 fester Abzug").sum())
    n_g3 = int(((d.gruppe == "3 Messung") & (d.y < ZOOM_MAX)).sum())
    C.schliesse(
        pdf,
        fig2,
        f"Faustregel y < {VORSCHLAG_FIRMA:g} | Gruppe 2 umfasst "
        f"{n_g2} Billets, Gruppe 3 unterhalb {ZOOM_MAX:g} % noch {n_g3}",
    )

    fig3, ax3 = C.seite(
        pdf, "Verteilung von y mit den Artefaktgruppen", 2, 2
    )
    feld_i(ax3[0], d)
    feld_j(ax3[1], d)
    feld_k(ax3[2], d)
    feld_l(ax3[3], d)
    C.aufraeumen(fig3)
    fig3.savefig(
        C.ROOT / "reports" / "figures" / "y_verteilung_balken.png",
        dpi=200,
        bbox_inches="tight",
    )
    C.schliesse(
        pdf,
        fig3,
        f"Klassenbreite {BREITE_VOLL:g} pp im vollen Bereich, "
        f"{BREITE_ZOOM:g} pp im Ausschnitt | Farben wie auf Seite 2 | "
        f"{int((d.y < ZOOM2_MAX).sum())} der {len(d)} Billets liegen unter "
        f"{ZOOM2_MAX:g} %",
    )

    g2 = d[d.gruppe == "2 fester Abzug"]
    n_art_g2 = int(g2[SP["artikel"]].nunique())
    paare = g2.groupby([SP["masse_soll"], SP["masse_ist"]]).size().sort_values(ascending=False)
    g3 = d[(d.gruppe == "3 Messung") & (d.y < ZOOM_MAX)]
    n_diff_g3 = int(g3["diff"].nunique())
    C.textseite(
        pdf,
        f"Zweite Stufe: Faustregel y < {VORSCHLAG_FIRMA:g} gegen Y_MIN = {VORSCHLAG_DATEN:g}",
        [
            f"Die Faustregel y < {VORSCHLAG_FIRMA:g} entfernt "
            f"{int((d.y < VORSCHLAG_FIRMA).sum())} der {len(d)} Billets "
            f"({100 * (d.y < VORSCHLAG_FIRMA).mean():.1f} %).",
            "Befund E und F: Unterhalb der Faustregel liegen drei Gruppen, getrennt "
            "durch zwei leere Zonen. "
            f"Gruppe 1: {int((d.gruppe == '1 Skalenschritte').sum())} Billets mit 0 bis 2 "
            f"Skalenschritten. Gruppe 2: {len(g2)} Billets zwischen "
            f"{luecken(d)[0][1]:.4f} und {luecken(d)[1][0]:.4f}. Gruppe 3: {n_g3} "
            f"Billets unterhalb {ZOOM_MAX:g} %.",
            f"Befund G: Gruppe 2 besteht aus einem festen Anteil der Sollmasse. "
            f"{int(paare.head(3).sum())} der {len(g2)} Billets verteilen sich auf drei "
            f"Wertepaare (Soll-/Istmasse) aus {n_art_g2} Artikeln. Gruppe 3 hat "
            f"{n_diff_g3} verschiedene Differenzen und streut.",
            f"Befund H: Jede Schwelle zwischen {luecken(d)[1][0]:.3f} und "
            f"{luecken(d)[1][1]:.3f} entfernt dieselben "
            f"{int((d.y < VORSCHLAG_DATEN).sum())} Zeilen (Gruppe 1 und 2). "
            f"{VORSCHLAG_FIRMA:g} % entfernt zusätzlich die {n_g3} Billets der Gruppe 3. "
            f"Anteile: {VORSCHLAG_DATEN:g} % entfernt "
            f"{100 * (d.y < VORSCHLAG_DATEN).mean():.1f} %, {VORSCHLAG_FIRMA:g} % "
            f"entfernt {100 * (d.y < VORSCHLAG_FIRMA).mean():.1f} %. "
            f"{int((d.y < ZOOM2_MAX).sum())} Billets liegen unter {ZOOM2_MAX:g} %, "
            f"{int((d.y >= 100).sum())} bei y = 100.",
            f"Festlegung: Y_MIN = {VORSCHLAG_DATEN:g}.",
        ],
    )

    C.textseite(
        pdf,
        "Erste Stufe: Billets mit null bis zwei Skalenschritten",
        [
            "y = (Sollmasse − Istmasse) / Sollmasse · 100.",
            f"Befund A: Die Istmasse wird mit {SCHRITT:g} kg Auflösung geführt. "
            f"{int((d.schritte == 0).sum())} Billets haben null, "
            f"{int((d.schritte == 1).sum())} einen und {int((d.schritte == 2).sum())} "
            "zwei Skalenschritte Differenz; dahinter folgt eine leere Lücke.",
            f"Befund B und C: Das größte y der Skalenschritt-Gruppe ist {gmax:.4f}, das "
            f"kleinste y aller übrigen Billets {gmin:.4f}. Jede Schwelle dazwischen "
            f"entfernt dieselben {int(tick.sum())} Zeilen. "
            + "; ".join(f"{k:g} entfernt {int((d.y < k).sum())}" for k in KANDIDATEN) + ".",
            f"Befund D: Ein Skalenschritt entspricht je nach Billetmasse einem y zwischen "
            f"{SCHRITT / d[SP['masse_soll']].quantile(0.95) * 100:.4f} und "
            f"{SCHRITT / d[SP['masse_soll']].quantile(0.05) * 100:.4f}; die betroffenen Billets "
            "liegen auf dieser Kurve.",
            f"Die {n_teil} Billets mit ein bis zwei Skalenschritten stammen aus "
            f"{n_art_tick} von {n_art} Artikeln. Y_MAX = 100 bleibt offen "
            "(Totalausfall).",
        ],
    )
    pdf.close()
    print(f"geschrieben: {C.OUT_DIR / '08_y_untergrenze.pdf'}")
    print(f"  Skalenschritt-Gruppe: {int(tick.sum())} Billets, y bis {gmax:.4f}")
    print(f"  naechstes Billet:     y = {gmin:.4f}")
    for k in KANDIDATEN:
        print(f"  Y_MIN={k:<6g} entfernt {int((d.y < k).sum()):>4} Zeilen")


if __name__ == "__main__":
    main()
