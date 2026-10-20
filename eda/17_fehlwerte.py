"""17 — Fehlende Werte.

In der Arbeit: Abschnitt 4.2.3 und 5.6: fehlende Werte gesamt, je Merkmal, je
Billet und je Datenquelle.

Umfang, gemeinsames Fehlen, Ebene und Zeit.
Ausgabe: reports/eda/17_fehlwerte.pdf.

    python eda/17_fehlwerte.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import mannwhitneyu

import _common as C
import data_prep as D
from data_prep import SP  # noqa: E402

N_PERM = 200

_AGGREGATE = (
    "_mean", "_std", "_min", "_max", "_last", "_range",
    "_n_samples", "_known_frac", "_frac_on_t", "_n_edges",
)

_SQL_GRUPPEN = {
    "": "Billetdatensatz",
    "prog": "Schmelzprogramm",
    "artikel": "Artikelstamm",
    "ingot": "Artikelstamm",
    "chargeA": "Charge Anode",
    "chargeK": "Charge Katode",
    "postenA": "Buchungsposten",
    "postenK": "Buchungsposten",
}
KURZ = {
    "Billetdatensatz": "Billet",
    "Schmelzprogramm": "Programm",
    "Artikelstamm": "Artikel",
    "Charge Anode": "Charge A",
    "Charge Katode": "Charge K",
    "Buchungsposten": "Posten",
    "Prozessdaten": "Prozess",
}
GRUPPEN_REIHE = [
    "Billetdatensatz", "Schmelzprogramm", "Artikelstamm",
    "Charge Anode", "Charge Katode", "Buchungsposten", "Prozessdaten",
]
FARBEN = dict(
    zip(
        GRUPPEN_REIHE,
        ["#4c72b0", "#dd8452", "#937860", "#55a868", "#8172b3", "#da8bc3", "#c44e52"],
    )
)


def gruppe(c: str, influx: set) -> str:
    if c in influx:
        return "Prozessdaten"
    return _SQL_GRUPPEN.get(c.split("__")[0] if "__" in c else "", "Billetdatensatz")


def kanal(c: str) -> str:
    for s in _AGGREGATE:
        if c.endswith(s):
            return c[: -len(s)]
    return c


def bh_q(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg, Regel 5."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return q.clip(0, 1)


def eta2_binaer(ind: np.ndarray, codes: np.ndarray, n_g: int) -> float:
    """Varianzanteil zwischen Gruppen fuer einen 0/1-Indikator, per bincount."""
    m = ind.mean()
    tot = ((ind - m) ** 2).sum()
    if tot == 0:
        return np.nan
    s = np.bincount(codes, weights=ind, minlength=n_g)
    n = np.bincount(codes, minlength=n_g)
    ok = n > 0
    return float((n[ok] * (s[ok] / n[ok] - m) ** 2).sum() / tot)


def ebene(ind_matrix: np.ndarray, gruppen: pd.Series, rng) -> pd.DataFrame:
    """eta² je Indikatorspalte gegen eine Gruppierung, mit Nullerwartung."""
    ok = gruppen.notna().to_numpy()
    codes, uniq = pd.factorize(gruppen[ok])
    n_g = len(uniq)
    zeilen = []
    for j in range(ind_matrix.shape[1]):
        ind = ind_matrix[ok, j]
        beob = eta2_binaer(ind, codes, n_g)
        null = [eta2_binaer(rng.permutation(ind), codes, n_g) for _ in range(N_PERM)]
        zeilen.append((beob, float(np.nanmean(null))))
    return pd.DataFrame(zeilen, columns=["beob", "null"])


def konstant_anteil(ind: np.ndarray, gruppen: pd.Series) -> float:
    """Anteil der Mehrbillet-Gruppen, in denen der Indikator bei ALLEN Billets gleich
    ist."""
    d = pd.DataFrame({"i": ind, "g": gruppen.to_numpy()}).dropna()
    gr = d.groupby("g").i
    n = gr.size()
    k = gr.nunique()[n >= 2]
    return float((k == 1).mean()) if len(k) else np.nan


def main():
    ds = C.lade(alle_arten=True)
    X, y, df = C.daten_teil(ds)
    fuss = C.kopfzeile(ds, alle_arten=True)
    pdf = C.pdf_oeffnen("17_fehlwerte.pdf")
    rng = np.random.default_rng(C.SEED)

    influx = set(D.influx_spalten(list(X.columns)))
    G = pd.Series({c: gruppe(c, influx) for c in X.columns})
    K = pd.Series({c: kanal(c) for c in X.columns if c in influx})
    N = X.isna()
    fehlt = N.mean()
    n, p = X.shape

    ts = pd.to_datetime(
        ds.melt_end_ts[C.eda_maske(ds)].reset_index(drop=True)
        if len(ds.melt_end_ts)
        else pd.Series(pd.NaT, index=X.index),
        errors="coerce",
    )

    teilweise = fehlt[(fehlt > 0.01) & (fehlt < 0.99)].index
    sql = [c for c in X.columns if c not in influx and c not in D._ABGELEITET]
    inf = [c for c in X.columns if c in influx]

    teil_mit_fehl = fehlt[(fehlt > 0) & (fehlt < 1)].index
    muster = N[teil_mit_fehl].T.apply(lambda s: hash(s.to_numpy().tobytes()), axis=1)
    bloecke = (
        pd.DataFrame(
            {
                "muster": muster,
                "gruppe": G[teil_mit_fehl],
                "fehlt": fehlt[teil_mit_fehl],
                "kanal": [K.get(c, "") for c in teil_mit_fehl],
            }
        )
        .groupby("muster")
        .agg(
            spalten=("fehlt", "size"),
            fehlt=("fehlt", "first"),
            gruppe=("gruppe", lambda s: " + ".join(sorted(set(s), key=GRUPPEN_REIHE.index))),
            kanaele=("kanal", lambda s: sorted(set(s) - {""})),
        )
        .sort_values("spalten", ascending=False)
        .reset_index(drop=True)
    )

    ind = N[teilweise].to_numpy(dtype=float)
    auftrag = df[SP["auftrag"]]
    artikel = df[SP["artikel"]]
    E_auf = ebene(ind, auftrag, rng)
    E_art = ebene(ind, artikel, rng)
    konst = np.array([konstant_anteil(ind[:, j], auftrag) for j in range(ind.shape[1])])
    konst_null = np.array(
        [konstant_anteil(rng.permutation(ind[:, j]), auftrag) for j in range(ind.shape[1])]
    )

    zeilen = []
    for c in teilweise:
        v = X[c].notna().to_numpy()
        if v.sum() < 30 or (~v).sum() < 30:
            continue
        u, pw = mannwhitneyu(y[v], y[~v])
        zeilen.append(
            {
                "merkmal": c,
                "gruppe": G[c],
                "fehlt": fehlt[c],
                "p": pw,
                "r": 2 * u / (v.sum() * (~v).sum()) - 1,
                "dmed": float(y[v].median() - y[~v].median()),
            }
        )
    M = pd.DataFrame(zeilen)
    M["q"] = bh_q(M.p.to_numpy())
    sig = M.q < 0.05

    je_billet_sql = N[sql].mean(axis=1)
    je_billet_inf = N[inf].mean(axis=1)
    je_billet = N.mean(axis=1)

    je_gruppe = (
        pd.DataFrame({"g": G, "f": fehlt})
        .groupby("g")
        .f.agg(["size", "mean", "median"])
        .reindex(GRUPPEN_REIHE)
        .dropna()
    )
    zeitlich = ts.notna()
    C.textseite(
        pdf,
        "17 — Fehlende Werte: Umfang und Muster",
        [
            f"{p} Merkmale, {C.umfang_text(n)}; davon {len(sql)} Stammdaten (SQL) und "
            f"{len(inf)} Prozessdaten (Influx). Fehlanteil gesamt "
            f"{100 * N.to_numpy().mean():.1f} % aller Zellen; vollständige Zeilen: "
            f"{int((~N.any(axis=1)).sum())} von {n}.",
            "Fehlanteil je Merkmalsgruppe (Spalten, Mittel, Median):",
            *[
                f"  {g}: {int(r['size'])} Spalten, Mittel {100 * r['mean']:.1f} %, "
                f"Median {100 * r['median']:.1f} %"
                for g, r in je_gruppe.iterrows()
            ],
            f"Je Billet fehlen im Median {100 * je_billet.median():.1f} % der Merkmale "
            f"(Stammdaten {100 * je_billet_sql.median():.1f} %, Prozessdaten "
            f"{100 * je_billet_inf.median():.1f} %).",
            f"Blöcke: Die {len(teil_mit_fehl)} Spalten mit mindestens einem und "
            f"weniger als 100 % Fehlwerten zeigen nur {len(bloecke)} verschiedene "
            f"Fehlmuster. Die fünf größten Blöcke umfassen "
            f"{int(bloecke.spalten.head(5).sum())} Spalten: "
            + "; ".join(
                f"{r.spalten} Spalten {r.gruppe} ({100 * r.fehlt:.1f} % fehlend)"
                for r in bloecke.head(5).itertuples()
            )
            + ". Spalten eines Blocks fehlen bei exakt denselben Billets; für das "
            "Fehlen gibt es damit viel weniger unabhängige Ursachen als Spalten.",
            f"Ebene (nur die {len(teilweise)} Spalten mit 1 bis 99 % Fehlanteil, "
            f"{N_PERM} Permutationen): Der Varianzanteil des Fehl-Indikators zwischen "
            f"Aufträgen beträgt im Median {E_auf.beob.median():.3f} bei einer "
            f"Nullerwartung von {E_auf.null.median():.3f}; zwischen Artikeln "
            f"{E_art.beob.median():.3f} gegen {E_art.null.median():.3f}. In "
            f"{100 * np.nanmedian(konst):.1f} % der Mehrbillet-Aufträge (Median über "
            f"die Spalten) fehlt ein Merkmal bei allen Billets oder bei keinem; bei "
            f"zufälliger Zuordnung wären es {100 * np.nanmedian(konst_null):.1f} %.",
            f"Fehlen und y: von {len(M)} testbaren Spalten (je mindestens 30 Billets "
            f"mit und ohne Wert) unterscheiden sich bei {int(sig.sum())} die y-Werte "
            f"(Mann-Whitney, FDR q < 0,05). Die Effektstärke ist die rang-biseriale "
            f"Korrelation r: +1 heißt, Billets mit Wert haben durchweg höheres y. "
            f"Median |r| der signifikanten Spalten {M.r[sig].abs().median():.2f}, "
            f"größte |r| {M.r.abs().max():.2f}.",
            f"Zeit: {int(zeitlich.sum())} Billets mit rekonstruiertem Schmelzende; "
            "Monate mit weniger als 20 Billets sind auf der Seite 'Fehlen über die Zeit' nicht gezeichnet.",
            "VORBEHALT: 'fehlt' heißt hier NaN in der Merkmalstabelle. Bei den "
            "Prozessdaten ist das selten; fehlende Abschnitte INNERHALB eines Kanals "
            "misst stattdessen *_known_frac (07_datenqualitaet.pdf). Kategoriale "
            "Stammdaten können auch einen leeren Text statt NaN tragen; der zählt "
            "hier als vorhanden.",
        ],
    )

    fig, ax = C.seite(pdf, f"Fehlanteil aller {p} Merkmale", 1, 1)
    sortiert = fehlt.sort_values(ascending=False)
    ax[0].bar(np.arange(p), sortiert.to_numpy(), width=1.0,
              color=[FARBEN[G[c]] for c in sortiert.index])
    ax[0].set_xlim(-0.5, p - 0.5)
    ax[0].set_ylim(0, 1.02)
    ax[0].yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f} %")
    ax[0].set_xlabel("Merkmale, absteigend nach Fehlanteil sortiert")
    ax[0].set_ylabel("Anteil der Billets ohne Wert")
    ax[0].grid(alpha=0.3, axis="y")
    for schwelle, text in ((1.0, "vollständig leer"), (0.0, "vollständig belegt")):
        k = int((sortiert == schwelle).sum())
        if k:
            lo = 0 if schwelle == 1.0 else p - k
            ax[0].annotate("", xy=(lo, 1.0 if schwelle == 1.0 else 0.06),
                           xytext=(lo + k, 1.0 if schwelle == 1.0 else 0.06),
                           arrowprops=dict(arrowstyle="<->", color="black", lw=0.8))
            if schwelle == 1.0:
                ax[0].text(lo + k + 8, 0.985, f"← {text}: {k} Merkmale", ha="left",
                           va="center", fontsize=9)
            else:
                ax[0].text(lo + k / 2, 0.09, f"{text}: {k} Merkmale", ha="center",
                           fontsize=9)
    ax[0].legend(
        handles=[Patch(color=FARBEN[g], label=f"{g} ({int((G == g).sum())})")
                 for g in GRUPPEN_REIHE],
        loc="upper right", fontsize=8.5,
    )
    C.schliesse(pdf, fig, fuss)

    klassen_fehlt = pd.cut(
        fehlt, [-np.inf, 0, 0.01, 0.5, 0.99, np.inf],
        labels=["vollständig", "≤ 1 % fehlt", "1–50 % fehlt", "50–99 % fehlt", "≥ 99 % fehlt"],
    )
    tab = pd.crosstab(G, klassen_fehlt).reindex(GRUPPEN_REIHE).fillna(0)
    farben_kl = ["#2e7d32", "#9ccc65", "#ffca28", "#fb8c00", "#c62828"]
    fig, ax = C.seite(pdf, "Wie vollständig sind die Merkmalsgruppen?", 1, 2)
    for a, werte, xlab in (
        (ax[0], tab, "Anzahl Merkmale"),
        (ax[1], tab.div(tab.sum(axis=1), axis=0), "Anteil der Merkmale der Gruppe"),
    ):
        links = np.zeros(len(werte))
        for kl, f in zip(tab.columns, farben_kl):
            a.barh(werte.index[::-1], werte[kl].to_numpy()[::-1], left=links[::-1],
                   color=f, label=kl, edgecolor="white", lw=0.5)
            links += werte[kl].to_numpy()
        a.set_xlabel(xlab)
        a.grid(alpha=0.3, axis="x")
    ax[1].set_yticklabels([])
    ax[1].xaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f} %")
    ax[0].set_title("absolut", fontsize=10)
    ax[1].set_title("relativ", fontsize=10)
    ax[1].legend(loc="upper center", bbox_to_anchor=(0, -0.08), ncol=5, fontsize=8.5)
    C.schliesse(pdf, fig, fuss)

    top_n = 60
    groesster = muster.value_counts().index[0]
    im_block = muster.index[muster == groesster]
    blockname = (f"[Block, {len(im_block)} Merkmale: "
                 + " + ".join(KURZ[g] for g in GRUPPEN_REIHE if (G[im_block] == g).any())
                 + "]")
    kandidaten = fehlt[(fehlt > 0) & (fehlt < 1)].drop(im_block)
    kandidaten[blockname] = fehlt[im_block[0]]
    benannt = kandidaten.sort_values(ascending=False).head(top_n)
    farbe_von = {c: FARBEN[G[c]] for c in benannt.index if c in G.index}
    farbe_von[blockname] = "#555555"
    fig, ax = C.seite(pdf, f"Die {len(benannt)} Merkmale mit dem höchsten Fehlanteil "
                      f"(ohne {int((fehlt == 1).sum())} vollständig leere)", 1, 1)
    ax[0].barh(np.arange(len(benannt))[::-1], benannt.to_numpy(),
               color=[farbe_von[c] for c in benannt.index])
    for i, v in zip(np.arange(len(benannt))[::-1], benannt.to_numpy()):
        ax[0].text(v + 0.005, i, f"{100 * v:.1f} %", va="center", fontsize=5.5)
    ax[0].set_yticks(np.arange(len(benannt))[::-1])
    ax[0].set_yticklabels([c if c == blockname else c[:45] for c in benannt.index],
                          fontsize=6)
    benannt = benannt.drop(blockname, errors="ignore")
    ax[0].set_ylim(-0.6, len(benannt) - 0.4)
    ax[0].set_xlim(0, 1)
    ax[0].xaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f} %")
    ax[0].set_xlabel("Anteil der Billets ohne Wert")
    ax[0].grid(alpha=0.3, axis="x")
    ax[0].legend(
        handles=[Patch(color=FARBEN[g], label=g) for g in GRUPPEN_REIHE
                 if (G[benannt.index] == g).any()],
        loc="lower right", fontsize=8,
    )
    C.schliesse(pdf, fig, fuss)

    spalten = sorted(
        X.columns, key=lambda c: (GRUPPEN_REIHE.index(G[c]), K.get(c, ""), -fehlt[c])
    )
    zeilen_zeit = np.argsort(ts.fillna(pd.Timestamp.max).to_numpy(), kind="stable")
    zeilen_menge = np.argsort(je_billet.to_numpy(), kind="stable")
    for titel, reihe, ylab in (
        ("Fehlwertmatrix, Billets nach Schmelzende sortiert", zeilen_zeit,
         "Billets (oben früh, unten spät; ohne Zeitstempel ganz unten)"),
        ("Fehlwertmatrix, Billets nach Fehlanteil sortiert", zeilen_menge,
         "Billets (oben wenige, unten viele Fehlwerte)"),
    ):
        fig, ax = C.seite(pdf, titel, 1, 1)
        a = ax[0]
        a.imshow(
            N[spalten].to_numpy()[reihe],
            aspect="auto",
            interpolation="nearest",
            cmap=ListedColormap(["#f2f2f2", "#303030"]),
        )
        grenzen = np.flatnonzero(
            np.r_[True, G[spalten].to_numpy()[1:] != G[spalten].to_numpy()[:-1]]
        )
        for i, g0 in enumerate(grenzen):
            g1 = grenzen[i + 1] if i + 1 < len(grenzen) else len(spalten)
            name = G[spalten[g0]]
            a.axvline(g0 - 0.5, color="white", lw=1.5)
            a.axvspan(g0 - 0.5, g1 - 0.5, ymin=1.0, ymax=1.025, color=FARBEN[name],
                      clip_on=False)
            if g1 - g0 >= 20:
                a.text((g0 + g1) / 2, -0.012 * n, str(g1 - g0), ha="center",
                       va="center", fontsize=7, color="white")
        a.set_xticks([])
        a.set_yticks([])
        a.set_xlabel("Merkmale, nach Gruppe, innerhalb der Prozessdaten nach Kanal sortiert "
                     "(dunkel = fehlt)")
        a.set_ylabel(ylab)
        a.legend(
            handles=[Patch(color=FARBEN[g], label=f"{g} ({int((G == g).sum())})")
                     for g in GRUPPEN_REIHE],
            loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=7, fontsize=7.5,
        )
        C.schliesse(pdf, fig, fuss)

    fig, ax = C.seite(pdf, "Wie viel fehlt — je Merkmal, je Gruppe, je Billet", 2, 2)

    bins = np.linspace(0, 1, 26)
    unten = np.zeros(len(bins) - 1)
    for g in GRUPPEN_REIHE:
        h, _ = np.histogram(fehlt[G == g], bins=bins)
        if h.sum():
            ax[0].bar(bins[:-1], h, width=np.diff(bins), bottom=unten, align="edge",
                      color=FARBEN[g], label=g, edgecolor="white", lw=0.3)
            unten += h
    ax[0].set_xlabel("Anteil fehlender Werte je Merkmal")
    ax[0].set_ylabel("Anzahl Merkmale")
    ax[0].set_title("Fehlanteil je Merkmal", fontsize=10)
    ax[0].legend(fontsize=7)
    ax[0].xaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f} %")

    jg = je_gruppe.iloc[::-1]
    ax[1].barh(jg.index, jg["mean"], color=[FARBEN[g] for g in jg.index])
    for i, (g, r) in enumerate(jg.iterrows()):
        ax[1].text(r["mean"] + 0.01, i, f"{100 * r['mean']:.1f} %  ({int(r['size'])} Sp.)",
                   va="center", fontsize=8)
    ax[1].set_xlim(0, 1)
    ax[1].xaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f} %")
    ax[1].set_xlabel("mittlerer Fehlanteil der Spalten")
    ax[1].set_title("Fehlanteil je Merkmalsgruppe", fontsize=10)
    ax[1].grid(alpha=0.3, axis="x")

    b2 = np.linspace(0, max(je_billet_sql.max(), je_billet_inf.max(), 0.01), 41)
    ax[2].hist(je_billet_sql, bins=b2, alpha=0.7, color=FARBEN["Billetdatensatz"],
               label="Stammdaten (SQL)")
    ax[2].hist(je_billet_inf, bins=b2, alpha=0.7, color=FARBEN["Prozessdaten"],
               label="Prozessdaten (Influx)")
    ax[2].set_yscale("log")
    ax[2].xaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f} %")
    ax[2].set_xlabel("Anteil fehlender Merkmale im Billet")
    ax[2].set_ylabel("Anzahl Billets (log)")
    ax[2].set_title("Fehlanteil je Billet", fontsize=10)
    ax[2].legend(fontsize=8)

    fk = (
        pd.DataFrame({"k": K, "f": fehlt[K.index]})
        .groupby("k").f.mean().sort_values()
    )
    ax[3].barh(fk.index, fk.to_numpy(), color=FARBEN["Prozessdaten"])
    ax[3].tick_params(axis="y", labelsize=6)
    ax[3].xaxis.set_major_formatter(lambda v, _: f"{100 * v:.1f} %")
    ax[3].set_xlabel("mittlerer Fehlanteil der Kennzahlen des Kanals")
    ax[3].set_title("Prozessdaten: Fehlanteil je Kanal", fontsize=10)
    ax[3].grid(alpha=0.3, axis="x")
    C.schliesse(pdf, fig, fuss)

    fig, ax = C.seite(pdf, "Fehlen Werte einzeln oder gemeinsam?", 1, 1)
    I = N[teilweise].astype(float)
    phi = I.corr().fillna(0).to_numpy()
    dist = np.clip(1 - np.abs(phi), 0, None)
    np.fill_diagonal(dist, 0)
    ordnung = leaves_list(linkage(squareform(dist, checks=False), "average"))
    bild = ax[0].imshow(phi[np.ix_(ordnung, ordnung)], cmap="RdBu_r", vmin=-1, vmax=1,
                        interpolation="nearest")
    for i, j in enumerate(ordnung):
        ax[0].add_patch(
            Rectangle((-0.5 - 0.04 * len(ordnung), i - 0.5), 0.03 * len(ordnung), 1,
                      color=FARBEN[G[teilweise[j]]], clip_on=False, lw=0)
        )
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title(f"Korrelation der Fehl-Indikatoren\n({len(teilweise)} Spalten mit "
                    "1–99 % Fehlanteil, geclustert; Randfarbe = Gruppe)", fontsize=9.5)
    fig.colorbar(bild, ax=ax[0], fraction=0.046, pad=0.02, label="φ")
    ax[0].legend(
        handles=[Patch(color=FARBEN[g], label=g) for g in GRUPPEN_REIHE
                 if (G[teilweise] == g).any()],
        loc="upper left", bbox_to_anchor=(1.12, 1), fontsize=8, title="Randfarbe",
    )
    C.schliesse(pdf, fig, fuss)

    fig, ax = C.seite(pdf, "Die größten Blöcke identischer Fehlmuster", 1, 1)
    ax = [None, ax[0]]

    def blockname(r) -> str:
        if r.gruppe == "Prozessdaten":
            k = r.kanaele
            return "Prozess: " + ", ".join(k[:3]) + (f" (+{len(k) - 3})" if len(k) > 3 else "")
        return " + ".join(KURZ[g] for g in r.gruppe.split(" + "))

    top = bloecke.head(15).iloc[::-1]
    ax[1].barh(range(len(top)), top.spalten,
               color=[FARBEN.get(g.split(" + ")[0], "grey") for g in top.gruppe])
    ax[1].set_yticks(range(len(top)))
    ax[1].set_yticklabels([f"{blockname(r)}  [{100 * r.fehlt:.1f} %]"
                           for r in top.itertuples()], fontsize=7.5)
    for i, v in enumerate(top.spalten):
        ax[1].text(v + 1, i, str(v), va="center", fontsize=7.5)
    ax[1].set_xlabel("Spalten mit exakt identischem Fehlmuster")
    ax[1].set_title(f"Die 15 größten Blöcke ({len(teil_mit_fehl)} Spalten mit Fehlwerten, "
                    f"{len(bloecke)} Muster)\n[ ] = Anteil der Billets, bei denen der "
                    "Block fehlt", fontsize=9.5)
    ax[1].grid(alpha=0.3, axis="x")
    C.schliesse(pdf, fig, fuss)

    fig, ax = C.seite(pdf, "Fehlen über die Zeit", 2, 1, groesse=(11.7, 8.3))
    monat = ts.dt.to_period("M")
    anz = monat.value_counts().sort_index()
    gueltig = anz[anz >= 20].index
    for g in GRUPPEN_REIHE:
        sp = G[G == g].index
        if not len(sp):
            continue
        reihe = N.loc[zeitlich, sp].mean(axis=1).groupby(monat[zeitlich]).mean()
        reihe = reihe.reindex(gueltig)
        anode = g == "Charge Anode"
        ax[0].plot(reihe.index.to_timestamp(), reihe.to_numpy(), marker="o", ms=3,
                   color=FARBEN[g], lw=4 if anode else 1.5, alpha=0.5 if anode else 1,
                   label=g + (" (deckungsgleich mit Katode)" if anode else ""))
    ax[0].yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f} %")
    ax[0].set_ylabel("mittlerer Fehlanteil je Billet")
    ax[0].set_title("Fehlanteil je Monat und Merkmalsgruppe (Monate mit ≥ 20 Billets)",
                    fontsize=10)
    ax[0].legend(fontsize=7.5, ncol=3)
    ax[0].grid(alpha=0.3)
    ax[1].bar(anz.index.to_timestamp(), anz.to_numpy(), width=20,
              color=["steelblue" if m in gueltig else "lightgrey" for m in anz.index])
    ax[1].set_ylabel("Billets je Monat")
    ax[1].set_title("Zum Vergleich: Zahl der Billets (grau: nicht gezeichnet)", fontsize=10)
    ax[1].grid(alpha=0.3, axis="y")
    ax[1].set_xlim(ax[0].get_xlim())
    C.schliesse(pdf, fig, fuss)

    fig, ax = C.seite(pdf, "Auf welcher Ebene entsteht das Fehlen?", 1, 3)
    b = np.linspace(0, 1, 26)
    for a, E, name in ((ax[0], E_auf, "Aufträgen"), (ax[1], E_art, "Artikeln")):
        a.hist(E.beob.dropna(), bins=b, color="steelblue", alpha=0.8, label="beobachtet")
        a.axvline(E.null.median(), color="grey", ls="--", lw=1.2,
                  label=f"Nullerwartung (Median {E.null.median():.2f})")
        a.axvline(E.beob.median(), color="crimson", lw=1.2,
                  label=f"Median beobachtet {E.beob.median():.2f}")
        a.set_xlabel(f"η² des Fehl-Indikators zwischen {name}")
        a.set_ylabel("Anzahl Merkmale")
        a.set_title(f"Varianzanteil zwischen {name}", fontsize=9.5)
        a.legend(fontsize=7.5, loc="upper left")
        a.grid(alpha=0.3, axis="y")
    b3 = np.linspace(0, 1, 26)
    ax[2].hist(konst_null, bins=b3, alpha=0.6, color="grey", label="zufällige Zuordnung")
    ax[2].hist(konst, bins=b3, alpha=0.7, color="steelblue", label="beobachtet")
    ax[2].xaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f} %")
    ax[2].set_xlabel("Anteil der Mehrbillet-Aufträge mit\neinheitlichem Fehlen")
    ax[2].set_ylabel("Anzahl Merkmale")
    ax[2].set_title("Fehlt das Merkmal bei allen Billets\neines Auftrags oder bei keinem?",
                    fontsize=9.5)
    ax[2].legend(fontsize=8)
    C.schliesse(pdf, fig, fuss)

    fig, ax = C.seite(pdf, "Hängt das Fehlen mit dem Ausschussgrad zusammen?", 1, 2)
    for g in GRUPPEN_REIHE:
        m = (M.gruppe == g).to_numpy()
        if m.any():
            ax[0].scatter(M.r[m], -np.log10(M.q[m].clip(lower=1e-300)), s=14, alpha=0.7,
                          color=FARBEN[g], label=g)
    ax[0].axhline(-np.log10(0.05), color="grey", ls="--", lw=1, label="q = 0,05")
    ax[0].axvline(0, color="grey", lw=0.8)
    ax[0].set_xlabel("rang-biseriale Korrelation r (+: Billets MIT Wert haben höheres y)")
    ax[0].set_ylabel("−log10 q (FDR-korrigiert)")
    ax[0].set_title(f"{int(sig.sum())} von {len(M)} Spalten signifikant", fontsize=10)
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)
    M["muster"] = muster.reindex(M.merkmal).to_numpy()
    staerkste = M[sig].sort_values("q").drop_duplicates("muster").head(8)
    ax[0].text(
        0.02, 0.62,
        "stärkste Spalten (je Block eine):\n"
        + "\n".join(f"{r.merkmal[:30]:<30}  r {r.r:+.2f}  {100 * r.fehlt:4.1f} % fehlt"
                     for r in staerkste.itertuples()),
        transform=ax[0].transAxes, fontsize=6.5, family="monospace", va="top",
        bbox=dict(facecolor="white", edgecolor="lightgrey"),
    )

    klassen = pd.qcut(je_billet.rank(method="first"), 5, labels=False)
    daten = [y[klassen == k].to_numpy() for k in range(5)]
    grenzen_k = [je_billet[klassen == k] for k in range(5)]
    bp = ax[1].boxplot(daten, showfliers=False, patch_artist=True)
    for k in bp["boxes"]:
        k.set_facecolor("lightsteelblue")
    ax[1].set_xticks(range(1, 6))
    ax[1].set_xticklabels(
        [f"{100 * g.min():.0f}–{100 * g.max():.0f} %\n(n = {len(g)})" for g in grenzen_k],
        fontsize=8,
    )
    ax[1].set_xlabel("Fehlanteil im Billet (Quintile)")
    ax[1].set_ylabel("y [%]")
    ax[1].set_title("Ausschussgrad nach Fehlanteil des Billets", fontsize=10)
    ax[1].grid(alpha=0.3, axis="y")
    C.schliesse(pdf, fig, fuss)

    pdf.close()
    print(f"geschrieben: {C.OUT_DIR / '17_fehlwerte.pdf'}")
    print(je_gruppe)
    print(bloecke.head(10))
    print("eta2 Auftrag", E_auf.median().to_dict(), "Artikel", E_art.median().to_dict())
    print("konstant", np.nanmedian(konst), np.nanmedian(konst_null), "sig", int(sig.sum()), len(M))


if __name__ == "__main__":
    main()
