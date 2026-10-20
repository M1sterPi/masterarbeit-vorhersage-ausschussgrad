"""07 — Datenqualität.

In der Arbeit: nicht berichtet; Vorarbeit zu Abschnitt 4.2.3 und 4.2.4 (dort
eda/17 und eda/18).

Fehlwerte, Aufzeichnungsanteile, Redundanz und Kanalkorrelation.
Ausgabe: reports/eda/07_datenqualitaet.pdf, reports/figures/kanalkorrelation.png.

    python eda/07_datenqualitaet.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

import _common as C

KANAL_SCHWELLE = 0.7


def kanalgruppen(paare: list[tuple[float, str, str]], kanaele: list[str]) -> list[list]:
    """Kanäle zu Gruppen zusammenfassen, sobald |rho| die Schwelle übersteigt."""
    eltern = {k: k for k in kanaele}

    def wurzel(a):
        while eltern[a] != a:
            eltern[a] = eltern[eltern[a]]
            a = eltern[a]
        return a

    for betrag, i, j in paare:
        if betrag > KANAL_SCHWELLE:
            eltern[wurzel(i)] = wurzel(j)
    gruppen: dict[str, list[str]] = {}
    for k in kanaele:
        gruppen.setdefault(wurzel(k), []).append(k)
    return list(gruppen.values())


def kanalkorrelation(pdf, X: pd.DataFrame, fuss: str) -> dict:
    """Eine Seite: Korrelationsmatrix der Kanäle und die stärksten Paare."""
    spalten = [c for c in X.columns if c.endswith("_mean")]
    M = pd.DataFrame({c: C.als_zahl(X[c]) for c in spalten})
    M = M.loc[:, M.notna().mean() >= 0.3]
    kanaele = [c[:-5] for c in M.columns]
    korr = M.corr(method="spearman")
    korr.index = korr.columns = kanaele

    paare = []
    for a in range(len(kanaele)):
        for b in range(a + 1, len(kanaele)):
            v = korr.iloc[a, b]
            if not np.isnan(v):
                paare.append((abs(v), kanaele[a], kanaele[b]))
    paare.sort(reverse=True)
    betraege = np.array([p[0] for p in paare])
    gruppen = kanalgruppen(paare, kanaele)
    mehrfach = [g for g in gruppen if len(g) > 1]

    reihenfolge = kanaele
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform

        d = 1 - korr.abs().fillna(0).to_numpy()
        np.fill_diagonal(d, 0)
        d = (d + d.T) / 2
        reihenfolge = [
            kanaele[i]
            for i in leaves_list(linkage(squareform(d, checks=False), method="average"))
        ]
    except Exception:
        pass
    K = korr.loc[reihenfolge, reihenfolge]
    stats = {
        "median": float(np.median(betraege)),
        "ueber_07": float((betraege > 0.7).mean()),
        "ueber_09": float((betraege > 0.9).mean()),
        "n_gruppen": len(gruppen),
        "n_kanaele": len(kanaele),
        "n_paare": len(paare),
        "gruppen": mehrfach,
    }
    if pdf is None:
        return stats

    fig, ax = C.seite(pdf, "Korrelation zwischen den Sensorkanälen", 1, 2)
    bild = ax[0].imshow(K.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax[0].set_xticks(range(len(reihenfolge)))
    ax[0].set_xticklabels(reihenfolge, rotation=90, fontsize=5.5)
    ax[0].set_yticks(range(len(reihenfolge)))
    ax[0].set_yticklabels(reihenfolge, fontsize=5.5)
    ax[0].set_title("Spearman ρ der Fenstermittelwerte je Kanal", fontsize=10)
    for rand in ax[0].spines.values():
        rand.set_visible(False)
    fig.colorbar(bild, ax=ax[0], fraction=0.046, shrink=0.8)

    top = paare[:12][::-1]
    werte = [korr.loc[i, j] for _, i, j in top]
    ax[1].barh(
        range(len(top)),
        werte,
        color=["#c0392b" if v < 0 else "#2471a3" for v in werte],
    )
    ax[1].set_yticks([])
    for k, (v, (_, i, j)) in enumerate(zip(werte, top)):
        ax[1].text(
            -0.03 if v > 0 else 0.03,
            k,
            f"{i} – {j}",
            ha="right" if v > 0 else "left",
            va="center",
            fontsize=6.5,
        )
    ax[1].axvline(0, color="black", lw=0.8)
    ax[1].set_xlim(-1.05, 1.05)
    ax[1].set_ylim(-0.6, len(top) - 0.4)
    ax[1].spines["left"].set_visible(False)
    ax[1].set_xlabel("Spearman ρ", fontsize=8)
    ax[1].set_title("Die zwölf stärksten Kanalpaare", fontsize=10)
    C.aufraeumen(fig)
    fig.savefig(
        C.ROOT / "reports" / "figures" / "kanalkorrelation.png",
        dpi=200,
        bbox_inches="tight",
    )
    C.schliesse(
        pdf,
        fig,
        fuss + " | je Kanal nur das Aggregat *_mean, damit die Matrix die "
        "Redundanz ZWISCHEN Kanälen misst und nicht die innerhalb eines Kanals",
    )
    return stats


def main():
    ds = C.lade(alle_arten=True)
    X, y, df = C.daten_teil(ds)
    fuss = C.kopfzeile(ds, alle_arten=True)
    pdf = C.pdf_oeffnen("07_datenqualitaet.pdf")

    fehlt = X.isna().mean().sort_values(ascending=False)
    teilweise = fehlt[(fehlt > 0.01) & (fehlt < 0.99)]

    zeilen = []
    for c in teilweise.index:
        vorhanden = X[c].notna()
        if vorhanden.sum() < 30 or (~vorhanden).sum() < 30:
            continue
        u, p = mannwhitneyu(y[vorhanden], y[~vorhanden])
        zeilen.append(
            {
                "merkmal": c,
                "anteil_fehlt": fehlt[c],
                "p": p,
                "delta_median": y[vorhanden].median() - y[~vorhanden].median(),
            }
        )
    M = pd.DataFrame(zeilen)
    if len(M):
        M = M.sort_values("p").reset_index(drop=True)
        M["q"] = np.minimum.accumulate((M.p * len(M) / (M.index + 1))[::-1])[::-1].clip(
            0, 1
        )
        n_mnar = int((M.q < 0.05).sum())
    else:
        n_mnar = 0

    kanal_stats = kanalkorrelation(None, X, fuss)
    h = (
        C.als_zahl(X["batch_duration_s"]) / 3600
        if "batch_duration_s" in X.columns
        else pd.Series(dtype=float)
    )
    am_deckel = float((h >= 8.99).mean()) if len(h) else float("nan")

    num = C.numerische_merkmale(X, min_belegt=0.6)
    Z = pd.DataFrame({c: C.als_zahl(X[c]) for c in num}).dropna(axis=1, how="all")
    korr = Z.corr(method="spearman").abs()
    paare = korr.where(np.triu(np.ones(korr.shape), 1).astype(bool)).stack()
    hoch = float((paare > 0.9).mean())

    C.textseite(
        pdf,
        "07 — Datenqualität und was sie einschränkt",
        [
            f"{X.shape[1]} Merkmale, {C.umfang_text(len(X))}.",
            f"  vollständig belegt: {int((fehlt == 0).sum())} Merkmale",
            f"  teilweise belegt (mehr als 1 % und weniger als 99 % fehlend): {len(teilweise)}",
            f"  fast leer (mindestens 99 % fehlend): {int((fehlt >= 0.99).sum())}",
            f"  sonst (höchstens 1 % fehlend): "
            f"{int(((fehlt > 0) & (fehlt <= 0.01)).sum())}",
            "Die Abgrenzung 'teilweise' ist hier enger als in 11_imputation.pdf, das "
            "jede Spalte mit mindestens einem Fehlwert mitzählt — daher die "
            "unterschiedlichen Zahlen.",
            f"Mechanismus der Fehlwerte: von {len(M)} testbaren teilweise belegten "
            f"Merkmalen (je mindestens 30 Zeilen mit und ohne Wert) zeigen {n_mnar} "
            "einen signifikanten y-Unterschied zwischen Zeilen mit und ohne Wert "
            "(Mann-Whitney, FDR-korrigiert, rot im Streudiagramm). Wo das zutrifft, "
            "fehlt der Wert NICHT zufällig — das Fehlen selbst trägt Information, und "
            "eine Imputation mit dem Mittelwert würde sie zerstören statt sie zu nutzen.",
            f"Das Prozessfenster ist GEDECKELT, nicht getrimmt. Es reicht von einem "
            f"Wechsel der Billetnummer bis zum nächsten, höchstens aber 9 Stunden; "
            f"{100 * am_deckel:.1f} % der Billets liegen exakt an dieser Grenze. Für "
            "sie ist die wahre Fensterlänge unbekannt (Rechtszensur), batch_duration_s "
            "misst dort die Grenze statt des Prozesses und ist trotzdem als Merkmal im "
            "Satz. Der Lichtbogen füllt nur einen Bruchteil des Fensters (in der "
            "Stichprobe aus den Rohkurven, 16_sensorverlaeufe.pdf, rund ein Fünftel) — "
            "jeder Mittelwert, jede Standardabweichung mittelt also überwiegend "
            "Stillstand mit, in einem Verhältnis, das von Billet zu Billet schwankt.",
            f"Redundanz: {len(num)} Merkmale mit mindestens 60 % Belegung, {len(paare)} "
            f"Paare. |ρ| > 0,9 bei {100 * hoch:.1f} %, |ρ| > 0,7 bei "
            f"{100 * float((paare > 0.7).mean()):.1f} % der Paare. Im Dendrogramm "
            "(1 − |ρ|, average linkage) sind Äste unterhalb der gestrichelten Linie "
            "(Distanz 0,3, also |ρ| > 0,7) farbig. Merkmale in einem Ast sind "
            "austauschbar; ein Baummodell verteilt die Importance beliebig auf sie — "
            "deshalb ist Importance kein Beitragsmass.",
            f"Kanäle: je Kanal nur das Aggregat *_mean, damit die Matrix die Redundanz "
            f"ZWISCHEN Kanälen misst und nicht die innerhalb eines Kanals. "
            f"{kanal_stats['n_kanaele']} Kanäle, {kanal_stats['n_paare']} Paare, "
            f"Median |ρ| {kanal_stats['median']:.2f}; |ρ| > 0,7 bei "
            f"{100 * kanal_stats['ueber_07']:.1f} %, |ρ| > 0,9 bei "
            f"{100 * kanal_stats['ueber_09']:.1f} %. Bei Schwelle {KANAL_SCHWELLE:g} "
            f"bleiben {kanal_stats['n_gruppen']} Gruppen, davon "
            f"{len(kanal_stats['gruppen'])} mit mehr als einem Kanal: "
            + "; ".join(", ".join(g) for g in kanal_stats["gruppen"])
            + ".",
        ],
    )

    fig, ax = C.seite(pdf, "Fehlwerte: Umfang und Mechanismus", 2, 2)

    ax[0].hist(fehlt, bins=40, color="steelblue")
    ax[0].set_xlabel("Anteil fehlender Werte je Merkmal")
    ax[0].set_ylabel("Anzahl Merkmale")
    ax[0].set_title("Fehlanteil je Merkmal", fontsize=10)

    if len(M):
        sig = M.q < 0.05
        for maske, farbe, name in ((sig, "crimson", "q < 0,05"), (~sig, "steelblue", "q ≥ 0,05")):
            ax[1].scatter(
                M.anteil_fehlt[maske],
                -np.log10(M.p[maske].clip(lower=1e-300)),
                s=10,
                alpha=0.6,
                color=farbe,
                label=name,
            )
        ax[1].axhline(-np.log10(0.05), color="grey", ls="--", lw=1, label="p = 0,05")
        ax[1].legend(fontsize=8, loc="upper center")
        ax[1].set_xlabel("Anteil fehlender Werte")
        ax[1].set_ylabel("−log10 p  (y vorhanden gegen fehlend)")
        ax[1].set_title("Hängt das Fehlen mit y zusammen?", fontsize=10)
        ax[1].grid(alpha=0.3)
    else:
        C.leer(ax[1])

    kf = [c for c in X.columns if c.endswith("_known_frac")]
    if kf:
        daten = [C.als_zahl(X[c]).dropna().to_numpy() for c in kf]
        namen = [c.replace("_known_frac", "")[:20] for c in kf]
        reihen = np.argsort([np.median(d) if len(d) else 0 for d in daten])
        bp = ax[2].boxplot(
            [daten[i] for i in reihen], vert=False, showfliers=False, patch_artist=True
        )
        for k in bp["boxes"]:
            k.set_facecolor("lightsteelblue")
        ax[2].set_yticklabels([namen[i] for i in reihen], fontsize=6.5)
        ax[2].set_xlabel("Anteil des Fensters, der beobachtet ist")
        ax[2].set_title("known_frac je Kanal", fontsize=10)
        ax[2].grid(alpha=0.3, axis="x")
    else:
        C.leer(ax[2])

    if len(h):
        ax[3].hist(h.dropna(), bins=60, color="steelblue")
        ax[3].axvline(9, color="crimson", ls="--", lw=1.2, label="9-h-Deckel")
        ax[3].set_xlabel("Fensterlänge [h]")
        ax[3].set_ylabel("Anzahl Billets")
        ax[3].set_title("Fensterlänge je Billet", fontsize=10)
        ax[3].legend(fontsize=8, loc="upper left")
    else:
        C.leer(ax[3])
    C.schliesse(pdf, fig, fuss)

    fig, ax = C.seite(pdf, "Redundanz: warum Importance-Ranglisten wandern", 1, 2)
    ax[0].hist(paare.dropna(), bins=50, color="steelblue")
    ax[0].set_xlabel("|Spearman ρ| zwischen zwei Merkmalen")
    ax[0].set_ylabel("Anzahl Paare")
    ax[0].set_title("Paarweise Korrelation zwischen Merkmalen", fontsize=10)

    try:
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform

        d = 1 - korr.fillna(0).to_numpy()
        np.fill_diagonal(d, 0)
        d = (d + d.T) / 2
        Zl = linkage(squareform(d, checks=False), method="average")
        dendrogram(Zl, ax=ax[1], no_labels=True, color_threshold=0.3)
        ax[1].axhline(0.3, color="grey", ls="--", lw=1)
        ax[1].set_title("Merkmalscluster", fontsize=10)
        ax[1].set_ylabel("Distanz 1 − |ρ|")
        ax[1].spines["bottom"].set_visible(False)
    except Exception as e:
        ax[1].text(
            0.5,
            0.5,
            f"Dendrogramm nicht berechenbar:\n{e}",
            ha="center",
            va="center",
            fontsize=8,
        )
        ax[1].axis("off")
    C.schliesse(pdf, fig, fuss)

    kanalkorrelation(pdf, X, fuss)

    pdf.close()
    print(f"geschrieben: {C.OUT_DIR / '07_datenqualitaet.pdf'}")
    print(
        f"  Kanalpaare: Median |rho| {kanal_stats['median']:.2f} | "
        f">0,7 {100 * kanal_stats['ueber_07']:.1f} % | "
        f"{kanal_stats['n_gruppen']} Gruppen aus {kanal_stats['n_kanaele']} Kanaelen"
    )
    print(
        f"  Merkmale vollstaendig {int((fehlt == 0).sum())} | teilweise {len(teilweise)}"
    )
    print(f"  nicht zufaellig fehlend (q<0.05): {n_mnar} von {len(M)}")
    print(f"  Merkmalspaare mit |rho|>0.9: {100 * hoch:.1f} %")


if __name__ == "__main__":
    main()
