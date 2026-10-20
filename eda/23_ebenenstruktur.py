"""23 — Struktur über Artikel, Auftrag und Billet.

In der Arbeit: Abschnitte 4.1.1 bis 4.2.1 und 4.2.7: Tabelle 1, Gruppengrößen,
Abbildung 3, 5, 8 und 9, abgebrochene Schmelzen (4.2.6).

Gruppengrößen, Ebene je Merkmal, abgebrochene Schmelzen.
Ausgabe: reports/eda/23_ebenenstruktur.pdf, runs/eda/23_ebenenstruktur.json,
Abbildungen in reports/figures/.

    python eda/23_ebenenstruktur.py
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

import _common as C
from data_prep import SP  # noqa: E402

D = C.D

FIG_DIR = C.ROOT / "reports" / "figures"
JSON_OUT = C.ROOT / "runs" / "eda" / "23_ebenenstruktur.json"

F_ARTIKEL, F_AUFTRAG, F_BILLET = "#1f78b4", "#a6cee3", "#d95f02"
GRAU = "#6b6b6b"

ABGELEITET = ["melt_weekday", "melt_hour"]


def de(x, n=1):
    return f"{x:.{n}f}".replace(".", ",")


def tausend(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def gruppen(df: pd.DataFrame) -> dict:
    bpa = df.groupby(SP["auftrag"]).size()
    apa = df.groupby(SP["artikel"])[SP["auftrag"]].nunique()
    bpart = df.groupby(SP["artikel"]).size().sort_values(ascending=False)
    verschachtelt = int((df.groupby(SP["auftrag"])[SP["artikel"]].nunique() > 1).sum())
    return {
        "billets": int(len(df)),
        "auftraege": int(df[SP["auftrag"]].nunique()),
        "artikel": int(df[SP["artikel"]].nunique()),
        "billets_ohne_artikel": int(df[SP["artikel"]].isna().sum()),
        "auftraege_in_mehreren_artikeln": verschachtelt,
        "billets_je_auftrag_median": float(bpa.median()),
        "billets_je_auftrag_mittel": float(bpa.mean()),
        "billets_je_auftrag_max": int(bpa.max()),
        "einzelbillet_auftraege": int((bpa == 1).sum()),
        "anteil_billets_mit_geschwistern": float((df[SP["auftrag"]].map(bpa) > 1).mean()),
        "auftraege_je_artikel_median": float(apa.median()),
        "auftraege_je_artikel_max": int(apa.max()),
        "artikel_mit_mehreren_auftraegen": int((apa >= 2).sum()),
        "artikel_mit_einem_auftrag": int((apa == 1).sum()),
        "anteil_billets_in_artikeln_mit_mehreren_auftraegen": float(df[SP["artikel"]].map(apa).ge(2).mean()),
        "billets_je_artikel_median": float(bpart.median()),
        "billets_je_artikel_max": int(bpart.max()),
        "anteil_billets_top10_artikel": float(bpart.head(10).sum() / len(df)),
    }


def merkmalsebenen(X: pd.DataFrame, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    influx = set(D.influx_spalten(X.columns))
    mehr = df.groupby(SP["auftrag"])[SP["auftrag"]].transform("size") > 1
    zeilen = []
    for c in X.columns:
        s = X[c]
        quelle = "Prozessdaten" if c in influx else ("abgeleitet" if c in ABGELEITET else "Stammdaten")
        anteil_auftraege = np.nan
        if s.notna().sum() == 0 or s.nunique(dropna=True) <= 1:
            ebene = "konstant oder leer"
        else:
            v = df.assign(v=s.values)
            je_artikel = v.groupby(SP["artikel"])["v"].nunique(dropna=True)
            je_auftrag = v[mehr].groupby(SP["auftrag"])["v"].nunique(dropna=True)
            if (je_artikel <= 1).all():
                ebene = "zwischen Artikeln"
            elif (je_auftrag <= 1).all():
                ebene = "zwischen Aufträgen"
            else:
                ebene = "zwischen Billets"
                anteil_auftraege = float((je_auftrag > 1).mean())
        zeilen.append({"merkmal": c, "quelle": quelle, "ebene": ebene, "anteil_auftraege_variiert": anteil_auftraege})
    t = pd.DataFrame(zeilen)
    tab = t.pivot_table(index="ebene", columns="quelle", values="merkmal", aggfunc="count", fill_value=0)
    zus = {
        "tabelle": tab.to_dict(),
        "summe": int(len(t)),
        "median_anteil_auftraege_billetebene": {
            q: float(t[(t.ebene == "zwischen Billets") & (t.quelle == q)].anteil_auftraege_variiert.median())
            for q in ("Stammdaten", "Prozessdaten")
        },
        "beispiele": {e: t[t.ebene == e].merkmal.head(30).tolist() for e in t.ebene.unique()},
    }
    return t, zus


def abbrueche(X: pd.DataFrame, y: pd.Series) -> dict:
    pm = X[SP["phase"] + "_max"]
    ab = pm < SP["letzte_phase"]
    return {
        "mit_phasennummer": int(pm.notna().sum()),
        "abbrueche": int(ab.sum()),
        "y_median_abbruch": float(y[ab].median()),
        "y_median_rest": float(y[~ab & pm.notna()].median()),
        "totalausfaelle_unter_abbruechen": int((y[ab] >= 99.99).sum()),
        "totalausfaelle_gesamt": int((y >= 99.99).sum()),
        "anteil_abbrueche_im_oberen_dezil": float(ab[y >= y.quantile(0.9)].mean()),
    }


def kasten(ax, x, y, b, h, text, farbe, schrift=8.5, text_farbe="black"):
    ax.add_patch(FancyBboxPatch((x - b / 2, y - h / 2), b, h, boxstyle="round,pad=0.005,rounding_size=0.012",
                                fc=farbe, ec="none"))
    ax.text(x, y, text, ha="center", va="center", fontsize=schrift, color=text_farbe)


def abb_schema(g: dict, zus: dict):
    tab = zus["tabelle"]

    def n(ebene):
        return sum(tab.get(q, {}).get(ebene, 0) for q in ("Stammdaten", "Prozessdaten", "abgeleitet"))

    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0.05, 0.95)
    ax.axis("off")
    yA, yF, yB = 0.83, 0.5, 0.17
    xA = 0.48
    xF = [0.37, 0.595]
    xB = [[0.31, 0.37, 0.43], [0.56, 0.63]]
    for xf, xbs in zip(xF, xB):
        ax.plot([xA, xf], [yA - 0.07, yF + 0.07], color=GRAU, lw=0.8, zorder=0)
        for xb in xbs:
            ax.plot([xf, xb], [yF - 0.07, yB + 0.07], color=GRAU, lw=0.8, zorder=0)
    ax.text(0.675, yF, "…", fontsize=12, color=GRAU, va="center")
    ax.text(0.675, yB, "…", fontsize=12, color=GRAU, va="center")
    kasten(ax, xA, yA, 0.15, 0.13, "Artikel", F_ARTIKEL, 8.5, "white")
    for i, xf in enumerate(xF, 1):
        kasten(ax, xf, yF, 0.12, 0.13, f"Auftrag {i}", F_AUFTRAG, 8)
    for xbs in xB:
        for xb in xbs:
            kasten(ax, xb, yB, 0.052, 0.13, "Billet", F_BILLET, 6.5, "white")
    links = [
        (yA, "Artikel", f"{g['artikel']} im Datensatz",
         f"im Median {de(g['auftraege_je_artikel_median'], 0)} Aufträge je Artikel"),
        (yF, "Auftrag", f"{g['auftraege']} im Datensatz",
         f"im Median {de(g['billets_je_auftrag_median'], 0)} Billets je Auftrag"),
        (yB, "Billet", f"{tausend(g['billets'])} im Datensatz", ""),
    ]
    for yy, name, z1, z2 in links:
        ax.text(0.0, yy + 0.05, name, fontsize=8.5, weight="bold", va="center")
        ax.text(0.0, yy - 0.01, z1, fontsize=7, color=GRAU, va="center")
        if z2:
            ax.text(0.0, yy - 0.065, z2, fontsize=7, color=GRAU, va="center")
    rechts = [
        (yA, f"fest je Artikel ({n('zwischen Artikeln')} Merkmale):",
         "Glastyp, Schmelzprogramm und\nseine Parameter"),
        (yF, f"fest je Auftrag ({n('zwischen Aufträgen')} Merkmale):",
         "Sollmaße, Ofen, Endartikel"),
        (yB, f"verschieden je Billet ({n('zwischen Billets')} Merkmale):",
         "Prozessverlauf (Kennzahlen der\nKanäle), Rohstoffcharge"),
    ]
    for yy, kopf, text in rechts:
        ax.text(0.725, yy + 0.05, kopf, fontsize=7, weight="bold", va="center")
        ax.text(0.725, yy - 0.035, text, fontsize=7, va="center", linespacing=1.15)
    fig.savefig(FIG_DIR / "ebenen_schema.png", dpi=300, bbox_inches="tight")
    return fig


def beispielartikel(df: pd.DataFrame, y: pd.Series, zeit: pd.Series) -> list:
    """Vier Artikel mit vielen Aufträgen, über das Spektrum der Artikelmittel
    verteilt."""
    d = df.assign(y=y.values, t=zeit.values)
    st = d.groupby(SP["artikel"]).agg(auftraege=(SP["auftrag"], "nunique"), billets=("y", "size"),
                                         mittel=("y", "mean"))
    kand = st[(st.auftraege >= 6) & (st.auftraege <= 14) & (st.billets >= 25)].sort_values("mittel")
    idx = sorted({int(round(q * (len(kand) - 1))) for q in (0.1, 0.4, 0.7, 0.95)})
    return kand.index[idx].tolist()


def abb_beispiel(df: pd.DataFrame, y: pd.Series, zeit: pd.Series):
    d = df.assign(y=y.values, t=zeit.values)
    gesamt = float(y.mean())
    artikel = beispielartikel(df, y, zeit)
    rng = np.random.default_rng(C.SEED)
    fig, axes = plt.subplots(1, len(artikel), figsize=(6.4, 2.9), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    for k, (ax, a) in enumerate(zip(axes, artikel)):
        da = d[d[SP["artikel"]] == a]
        am = da.y.mean()
        reihen = (da.groupby(SP["auftrag"]).t.min().sort_values().index.tolist())
        ax.axhline(gesamt, color=GRAU, lw=0.8, ls=":", zorder=1)
        ax.axhline(am, color=F_ARTIKEL, lw=1.2, ls="--", zorder=2)
        for j, f in enumerate(reihen):
            df_ = da[da[SP["auftrag"]] == f]
            fm = df_.y.mean()
            ax.bar(j, fm - am, bottom=am, width=0.7, color=F_AUFTRAG, edgecolor="white", lw=0.5, zorder=2)
            x = j + rng.uniform(-0.18, 0.18, len(df_))
            ax.scatter(x, df_.y, s=11, color=F_BILLET, edgecolor="white", linewidth=0.4, zorder=3)
        ax.set_xticks(range(len(reihen)), [str(i + 1) for i in range(len(reihen))], fontsize=7)
        ax.set_xlim(-0.6, len(reihen) - 0.4)
        ax.set_title(f"Artikel {'ABCD'[k]}\n{len(reihen)} Aufträge, {len(da)} Billets", fontsize=7.5)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Ausschussgrad y [%]", fontsize=8)
    axes[0].set_ylim(0, 102)
    fig.supxlabel("Aufträge des Artikels in zeitlicher Reihenfolge", fontsize=7.5, y=-0.02)
    griffe = [
        Line2D([], [], marker="o", ls="", color=F_BILLET, markeredgecolor="white", ms=5, label="Billet"),
        Patch(color=F_AUFTRAG, label="Auftragsmittel minus Artikelmittel"),
        Line2D([], [], color=F_ARTIKEL, ls="--", lw=1.2, label="Artikelmittel"),
        Line2D([], [], color=GRAU, ls=":", lw=0.8, label="Gesamtmittel"),
    ]
    fig.legend(handles=griffe, loc="upper center", ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 1.1))
    C.aufraeumen(fig)
    fig.savefig(FIG_DIR / "ebenen_beispiel.png", dpi=300, bbox_inches="tight")
    return fig, artikel


def abb_zeit(y: pd.Series, zeit: pd.Series):
    d = pd.DataFrame({"t": zeit.values, "y": y.values}).dropna().sort_values("t")
    roll = d.set_index("t").y.rolling("30D")
    med = roll.median().where(roll.count() >= 20)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 3.6), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1.1], "hspace": 0.08})
    a1.scatter(d.t, d.y, s=3, color="#4a7fb0", alpha=0.25, linewidth=0, label="Billet")
    a1.plot(med.index, med.values, color="black", lw=1.4, label="gleitender Median (30 Tage)")
    a1.set_ylabel("Ausschussgrad y [%]", fontsize=8)
    a1.set_ylim(0, 102)
    leg = a1.legend(fontsize=7, loc="upper right", markerscale=3, frameon=False)
    leg.legend_handles[0].set_alpha(0.8)
    a1.tick_params(labelsize=7)
    a1.grid(axis="y", alpha=0.3)
    monat = d.set_index("t").y.resample("MS").size()
    a2.bar(monat.index + pd.Timedelta(days=15), monat.values, width=24, color=GRAU)
    a2.set_ylabel("Billets\nje Monat", fontsize=8)
    a2.tick_params(labelsize=7)
    a2.grid(axis="y", alpha=0.3)
    C.aufraeumen(fig)
    fig.savefig(FIG_DIR / "zeitverlauf.png", dpi=300, bbox_inches="tight")
    return fig


def abb_verteilung(y: pd.Series):
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    ax.hist(y, bins=np.arange(1, 101.5, 1), color="#4a7fb0", edgecolor="white", linewidth=0.3)
    med, mit = float(y.median()), float(y.mean())
    ax.axvline(med, color="black", lw=1.2)
    ax.axvline(mit, color="black", lw=1.2, ls="--")
    oben = ax.get_ylim()[1] * 1.12
    ax.set_ylim(0, oben)
    ax.text(med - 1, oben * 0.99, f"Median {de(med)} %", ha="right", va="top", fontsize=7)
    ax.text(mit + 1, oben * 0.99, f"Mittelwert {de(mit)} %", ha="left", va="top", fontsize=7)
    n100 = int((y >= 99.99).sum())
    ax.annotate(f"Totalausfall\n(y = 100 %, {n100} Billets)", xy=(99.5, n100), xytext=(86, oben * 0.55),
                fontsize=7, ha="center", arrowprops={"arrowstyle": "-", "color": GRAU, "lw": 0.7})
    ax.set_xlabel("Ausschussgrad y [%], Klassenbreite 1 Prozentpunkt", fontsize=8)
    ax.set_ylabel("Anzahl Billets", fontsize=8)
    ax.set_xlim(0, 101)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", alpha=0.3)
    C.aufraeumen(fig)
    fig.savefig(FIG_DIR / "y_verteilung.png", dpi=300, bbox_inches="tight")
    return fig


def abb_fehlanteil(X: pd.DataFrame):
    influx = set(D.influx_spalten(X.columns))
    anteil = X.isna().mean()
    stamm = anteil[[c not in influx for c in X.columns]]
    proz = anteil[[c in influx for c in X.columns]]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    bins = np.linspace(0, 1, 41)
    ax.hist([stamm, proz], bins=bins, stacked=True, color=["#1f78b4", "#d95f02"],
            label=[f"Stammdaten ({len(stamm) - len(ABGELEITET)} Merkmale, dazu {len(ABGELEITET)} abgeleitete)",
                   f"Prozessdaten ({len(proz)} Merkmale)"],
            edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Anteil der Billets ohne Wert", fontsize=8)
    ax.set_ylabel("Anzahl Merkmale", fontsize=8)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}".replace("%", " %")))
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, frameon=False, loc="upper center")
    ax.grid(axis="y", alpha=0.3)
    C.aufraeumen(fig)
    fig.savefig(FIG_DIR / "fehlanteil_merkmale.png", dpi=300, bbox_inches="tight")
    return fig


def main():
    ds = C.lade()
    X, y, df = C.daten_teil(ds)
    zeit = pd.Series(ds.melt_end_ts[C.eda_maske(ds)]).reset_index(drop=True)

    g = gruppen(df)
    g["maschinen"] = int(df[SP["ofen_kennung"]].nunique())
    g["oefen"] = int(df["Ofennummer"].nunique())
    g["schmelzende_von"] = str(zeit.min().date()) if zeit.notna().any() else None
    g["schmelzende_bis"] = str(zeit.max().date()) if zeit.notna().any() else None
    t, zus = merkmalsebenen(X, df)
    ab = abbrueche(X, y)

    with C.pdf_oeffnen("23_ebenenstruktur.pdf") as pdf:
        tab = pd.DataFrame(zus["tabelle"]).fillna(0).astype(int)
        C.textseite(pdf, "23  Ebenenstruktur: Artikel, Auftrag, Billet", [
            f"{tausend(g['billets'])} Billets in {g['auftraege']} Aufträgen und {g['artikel']} Artikeln "
            f"({g['billets_ohne_artikel']} Billet ohne Artikelangabe). Kein Auftrag gehört zu mehr als einem "
            f"Artikel ({g['auftraege_in_mehreren_artikeln']}): die Ebenen sind verschachtelt.",
            f"Billets je Auftrag: Median {de(g['billets_je_auftrag_median'], 0)}, Mittel "
            f"{de(g['billets_je_auftrag_mittel'], 2)}, höchstens {g['billets_je_auftrag_max']}; "
            f"{g['einzelbillet_auftraege']} Einzelbillet-Aufträge; "
            f"{de(100 * g['anteil_billets_mit_geschwistern'])} % der Billets haben Geschwister.",
            f"Aufträge je Artikel: Median {de(g['auftraege_je_artikel_median'], 0)}, höchstens "
            f"{g['auftraege_je_artikel_max']}; {g['artikel_mit_mehreren_auftraegen']} Artikel mit mehreren, "
            f"{g['artikel_mit_einem_auftrag']} mit einem Auftrag; "
            f"{de(100 * g['anteil_billets_in_artikeln_mit_mehreren_auftraegen'])} % der Billets liegen in Artikeln "
            f"mit mehreren Aufträgen. Billets je Artikel: Median {de(g['billets_je_artikel_median'], 0)}, höchstens "
            f"{g['billets_je_artikel_max']}; die zehn größten Artikel stellen "
            f"{de(100 * g['anteil_billets_top10_artikel'])} % der Billets.",
            "Ebene, auf der sich ein Merkmal ändert (Spalten: Quelle):\n" + tab.to_string(),
            "Lesart 'zwischen Billets': das Merkmal ändert sich in mindestens einem Auftrag mit mehreren Billets. "
            "Median des Anteils der Mehrbillet-Aufträge, in denen es sich ändert: Stammdaten "
            f"{de(100 * zus['median_anteil_auftraege_billetebene']['Stammdaten'])} %, Prozessdaten "
            f"{de(100 * zus['median_anteil_auftraege_billetebene']['Prozessdaten'])} %.",
            f"Abgebrochene Schmelzen (letzte Programmphase nicht erreicht): {ab['abbrueche']} von {ab['mit_phasennummer']} Billets mit "
            f"Phasennummer; Median y {de(ab['y_median_abbruch'])} % gegen {de(ab['y_median_rest'])} %; "
            f"{ab['totalausfaelle_unter_abbruechen']} der {ab['totalausfaelle_gesamt']} Totalausfälle; "
            f"Anteil im oberen y-Dezil {de(100 * ab['anteil_abbrueche_im_oberen_dezil'])} %.",
        ])
        f1 = abb_schema(g, zus)
        pdf.savefig(f1)
        f2, artikel = abb_beispiel(df, y, zeit)
        pdf.savefig(f2)
        f3 = abb_zeit(y, zeit)
        pdf.savefig(f3)
        pdf.savefig(abb_verteilung(y))
        pdf.savefig(abb_fehlanteil(X))
        plt.close("all")

    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps({"gruppen": g, "merkmalsebenen": zus, "abbrueche": ab,
                                    "beispielartikel": [str(a) for a in artikel]},
                                   ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"gruppen": g, "tabelle": zus["tabelle"],
                      "median_anteil": zus["median_anteil_auftraege_billetebene"], "abbrueche": ab},
                     ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
