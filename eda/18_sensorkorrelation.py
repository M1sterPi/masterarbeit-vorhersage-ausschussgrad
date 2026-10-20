"""18 — Korrelation zwischen den Prozesskanälen.

In der Arbeit: Abschnitt 4.2.4: Korrelation zwischen den Kanälen und innerhalb
eines Kanals, Kanalgruppen, Abbildung 7.

Gesamt, innerhalb der Aufträge und über alle Kennzahlen.
Ausgabe: reports/eda/18_sensorkorrelation.pdf, reports/figures/sensorkorrelation_matrix.png,
reports/figures/kapitel4/kanalpaare.png.

    python eda/18_sensorkorrelation.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _common as C
from data_prep import SP  # noqa: E402

AGGREGATE = ["mean", "std", "min", "max", "last", "range"]
MIN_BELEGT = 0.3
MAX_HAEUFIGSTER = 0.95
SCHWELLE = 0.7


def kanalmittel(X: pd.DataFrame) -> pd.DataFrame:
    """Je Kanal die Spalte *_mean, nur ausreichend belegte und nicht konstante."""
    M = pd.DataFrame(
        {c[: -len("_mean")]: C.als_zahl(X[c]) for c in X.columns if c.endswith("_mean")}
    )
    return M.loc[:, (M.notna().mean() >= MIN_BELEGT) & (M.nunique() > 2)]


def reihenfolge_nach_clustern(korr: pd.DataFrame) -> list[str]:
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    d = 1 - korr.abs().fillna(0).to_numpy()
    np.fill_diagonal(d, 0)
    d = (d + d.T) / 2
    idx = leaves_list(linkage(squareform(d, checks=False), method="average"))
    return [korr.columns[i] for i in idx]


def obere_dreiecke(korr: pd.DataFrame) -> pd.Series:
    maske = np.triu(np.ones(korr.shape, dtype=bool), 1)
    return korr.where(maske).stack()


def innerhalb_auftrag(M: pd.DataFrame, auftrag: pd.Series) -> pd.DataFrame:
    """Spearman innerhalb der Aufträge: Ränge minus Auftragsmittel der Ränge."""
    groesse = auftrag.map(auftrag.value_counts())
    R = M.rank().loc[groesse >= 2]
    a = auftrag.loc[groesse >= 2]
    return (R - R.groupby(a).transform("mean")).corr()


def main():
    ds = C.lade(alle_arten=True)
    X, _, df = C.daten_teil(ds)
    fuss = C.kopfzeile(ds, alle_arten=True)
    pdf = C.pdf_oeffnen("18_sensorkorrelation.pdf")

    M = kanalmittel(X)
    kanaele = list(M.columns)
    korr = M.corr(method="spearman")
    paare = obere_dreiecke(korr)
    betrag = paare.abs().sort_values(ascending=False)

    innen = []
    for k in kanaele:
        spalten = [f"{k}_{a}" for a in AGGREGATE if f"{k}_{a}" in X.columns]
        A = pd.DataFrame({c: C.als_zahl(X[c]) for c in spalten})
        A = A.loc[:, A.nunique() > 2]
        if A.shape[1] >= 2:
            innen.append(obere_dreiecke(A.corr(method="spearman")).abs())
    innen = pd.concat(innen) if innen else pd.Series(dtype=float)

    spalten_je_kanal = {}
    for k in kanaele:
        sp = [f"{k}_{a}" for a in AGGREGATE if f"{k}_{a}" in X.columns]
        sp = [c for c in sp if C.als_zahl(X[c]).notna().mean() >= MIN_BELEGT
              and C.als_zahl(X[c]).nunique() > 2
              and (c == f"{k}_mean"
                   or C.als_zahl(X[c]).value_counts(normalize=True).iloc[0] < MAX_HAEUFIGSTER)]
        if sp:
            spalten_je_kanal[k] = sp
    alle = [c for sp in spalten_je_kanal.values() for c in sp]
    K_alle = pd.DataFrame({c: C.als_zahl(X[c]) for c in alle}).corr(method="spearman").abs()
    stark_max, stark_kombi = {}, {}
    for i, j in paare.index:
        block = K_alle.loc[spalten_je_kanal[i], spalten_je_kanal[j]]
        a, b = block.stack().idxmax()
        stark_max[(i, j)] = block.loc[a, b]
        stark_kombi[(i, j)] = (a, b)
    stark_max = pd.Series(stark_max).reindex(paare.index)
    zuwachs = (stark_max - paare.abs()).sort_values(ascending=False)
    neu_stark = [ij for ij in paare.index
                 if stark_max[ij] > SCHWELLE and abs(paare[ij]) <= SCHWELLE]
    neu_last = [ij for ij in neu_stark if any(c.endswith("_last") for c in stark_kombi[ij])]

    auftrag = df[SP["auftrag"]].astype(str)
    korr_in = innerhalb_auftrag(M, auftrag)
    paare_in = obere_dreiecke(korr_in).reindex(paare.index)
    n_auftr = int((auftrag.map(auftrag.value_counts()) >= 2).sum())

    verlust = (paare.abs() - paare_in.abs()).sort_values(ascending=False)

    top = betrag.index[:5]
    C.textseite(
        pdf,
        "18 — Korrelationsstruktur zwischen den Sensorkanälen",
        [
            f"{len(kanaele)} Sensorkanäle (Fenstermittelwert *_mean, mindestens "
            f"{100 * MIN_BELEGT:.0f} % belegt), {C.umfang_text(len(X))}, "
            f"{len(paare)} Kanalpaare. Maß überall Spearman ρ.",
            "",
            "A — Welche Kanäle hängen zusammen?",
            f"  Median |ρ| über alle Paare: {betrag.median():.2f}. |ρ| > {SCHWELLE:g} bei "
            f"{int((betrag > SCHWELLE).sum())} von {len(betrag)} Paaren "
            f"({100 * (betrag > SCHWELLE).mean():.1f} %).",
            "  Die fünf stärksten Paare (Streudiagramme auf Seite 3):",
            *[f"    {i} – {j}: ρ = {korr.loc[i, j]:+.2f}" for i, j in top],
            "",
            "B — Zwischen Kanälen gegen innerhalb eines Kanals",
            f"  Aggregate desselben Kanals ({', '.join(AGGREGATE)}): Median |ρ| "
            f"{innen.median():.2f}, |ρ| > {SCHWELLE:g} bei {100 * (innen > SCHWELLE).mean():.1f} % "
            f"der {len(innen)} Paare.",
            f"  Verschiedene Kanäle (je *_mean): Median |ρ| {betrag.median():.2f}, "
            f"|ρ| > {SCHWELLE:g} bei {100 * (betrag > SCHWELLE).mean():.1f} %.",
            "  Die Redundanz des Merkmalssatzes sitzt damit überwiegend innerhalb der "
            "Kanäle, nicht zwischen ihnen.",
            "",
            "C — Zwischen Aufträgen gegen innerhalb eines Auftrags",
            f"  Innerhalb der Aufträge gerechnet ({n_auftr} Billets in Aufträgen mit "
            f"mindestens zwei Billets): Median |ρ| {paare_in.abs().median():.2f}, "
            f"|ρ| > {SCHWELLE:g} bei {100 * (paare_in.abs() > SCHWELLE).mean():.1f} %.",
            f"  Paare, deren |ρ| innerhalb der Aufträge um mehr als 0,3 kleiner ist: "
            f"{int(((paare.abs() - paare_in.abs()) > 0.3).sum())}. Deren Zusammenhang "
            "entsteht zwischen den Aufträgen (Rezept, Ofen, Maschine) und nicht im "
            "laufenden Prozess. Die größten Verluste:",
            *[
                f"    {i} – {j}: gesamt {paare[(i, j)]:+.2f}, im Auftrag {paare_in[(i, j)]:+.2f}"
                for i, j in verlust.index[:5]
            ],
            "",
            "D — Gegenprobe: stärkstes Aggregatpaar statt Mittelwert",
            f"  Je Kanalpaar die höchste |ρ| über alle Kombinationen aus "
            f"{', '.join(AGGREGATE)}: Median {stark_max.median():.2f} "
            f"(Mittelwerte: {betrag.median():.2f}), |ρ| > {SCHWELLE:g} bei "
            f"{int((stark_max > SCHWELLE).sum())} von {len(stark_max)} Paaren "
            f"(Mittelwerte: {int((betrag > SCHWELLE).sum())}).",
            f"  Paare, die erst über ein anderes Aggregat die Schwelle überschreiten: "
            f"{len(neu_stark)}, davon {len(neu_last)} über den letzten Wert (*_last) "
            "mindestens eines Kanals, also den Anlagenzustand am Fensterende. Ohne "
            f"diese bleiben {len(neu_stark) - len(neu_last)} neue Paare.",
            f"  Fast konstante Aggregate (ein Wert bei mindestens "
            f"{100 * MAX_HAEUFIGSTER:.0f} % der Billets) sind ausgeschlossen.",
            "  Die größten Zuwächse:",
            *[
                f"    {i} – {j}: Mittelwert {abs(paare[(i, j)]):.2f}, stärkstes Paar "
                f"{stark_max[(i, j)]:.2f} ({stark_kombi[(i, j)][0]} – {stark_kombi[(i, j)][1]})"
                for i, j in zuwachs.index[:5]
            ],
            "  Das Maximum über viele Kombinationen ist leicht nach oben verzerrt.",
            "",
            "Vorbehalte: Alle Kanäle sind über das gedeckelte Fenster gemittelt, das "
            "überwiegend Stillstand enthält (16_sensorverlaeufe.pdf). Ein Teil jeder "
            "Korrelation kann deshalb aus dem gemeinsamen Stillstandsanteil kommen. "
            "Bei binären Zustandskanälen ist *_mean ein Zeitanteil.",
        ],
    )

    reihe = reihenfolge_nach_clustern(korr)
    fig, ax = C.seite(pdf, "A — Spearman ρ zwischen den Kanälen", 1, 1)
    bild = ax[0].imshow(korr.loc[reihe, reihe].to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax[0].set_xticks(range(len(reihe)))
    ax[0].set_xticklabels(reihe, rotation=90, fontsize=7)
    ax[0].set_yticks(range(len(reihe)))
    ax[0].set_yticklabels(reihe, fontsize=7)
    for rand in ax[0].spines.values():
        rand.set_visible(False)
    fig.colorbar(bild, ax=ax[0], fraction=0.046, shrink=0.8, label="Spearman ρ")
    C.aufraeumen(fig)
    fig.savefig(
        C.ROOT / "reports" / "figures" / "sensorkorrelation_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )
    C.schliesse(pdf, fig, fuss)

    def streu_paare(achsen):
        for a, (i, j) in zip(achsen, betrag.index[:6]):
            ok = M[i].notna() & M[j].notna()
            a.scatter(M.loc[ok, i], M.loc[ok, j], s=4, alpha=0.3, color="steelblue", lw=0)
            a.set_xlim(C.robuste_grenzen(M.loc[ok, i]))
            a.set_ylim(C.robuste_grenzen(M.loc[ok, j]))
            a.set_xlabel(i, fontsize=8)
            a.set_ylabel(j, fontsize=8)
            a.set_title(f"ρ = {korr.loc[i, j]:+.2f}", fontsize=10)
            a.grid(True)

    fig, ax = C.seite(pdf, "A — Die sechs stärksten Kanalpaare", 2, 3)
    streu_paare(ax)
    C.schliesse(pdf, fig, fuss)
    fig, ax = plt.subplots(2, 3, figsize=(10.2, 6.9))
    streu_paare(ax.ravel())
    C.aufraeumen(fig)
    fig.tight_layout()
    ziel = C.ROOT / "reports" / "figures" / "kapitel4" / "kanalpaare.png"
    ziel.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ziel, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = C.seite(pdf, "B und C — Wo die Korrelation entsteht", 1, 2)
    bins = np.linspace(0, 1, 26)
    ax[0].hist(innen, bins=bins, density=True, histtype="step", lw=2,
               color="#e08214", label="Aggregate desselben Kanals")
    ax[0].hist(betrag, bins=bins, density=True, histtype="step", lw=2,
               color="#2166ac", label="verschiedene Kanäle (*_mean)")
    ax[0].axvline(SCHWELLE, color="grey", ls="--", lw=1)
    ax[0].set_xlabel("|Spearman ρ|")
    ax[0].set_ylabel("Dichte")
    ax[0].set_title("B — innerhalb gegen zwischen Kanälen", fontsize=10)
    ax[0].legend(fontsize=8)

    ax[1].scatter(paare, paare_in, s=14, color="#2166ac", alpha=0.7, lw=0)
    ax[1].plot([-1, 1], [-1, 1], color="grey", lw=0.8, ls="--")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].axvline(0, color="k", lw=0.5)
    ax[1].set_xlim(-1.05, 1.05)
    ax[1].set_ylim(-1.05, 1.05)
    ax[1].set_aspect("equal")
    ax[1].set_xlabel("ρ über alle Billets")
    ax[1].set_ylabel("ρ innerhalb der Aufträge")
    ax[1].set_title("C — je Kanalpaar: gesamt gegen innerhalb Auftrag", fontsize=10)
    ax[1].grid(True)
    C.schliesse(pdf, fig, fuss)

    fig, ax = C.seite(pdf, "D — Gegenprobe: Mittelwert gegen stärkstes Aggregatpaar", 1, 2)
    ax[0].scatter(paare.abs(), stark_max, s=14, color="#2166ac", alpha=0.7, lw=0)
    ax[0].plot([0, 1], [0, 1], color="grey", lw=0.8, ls="--")
    ax[0].axhline(SCHWELLE, color="grey", lw=0.8, ls=":")
    ax[0].axvline(SCHWELLE, color="grey", lw=0.8, ls=":")
    ax[0].set_xlim(-0.02, 1.02)
    ax[0].set_ylim(-0.02, 1.02)
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("|ρ| der Mittelwerte")
    ax[0].set_ylabel("höchste |ρ| über alle Aggregatkombinationen")
    ax[0].set_title("je Kanalpaar", fontsize=10)
    ax[0].grid(True)

    ax[1].hist(betrag, bins=bins, density=True, histtype="step", lw=2,
               color="#2166ac", label="nur Mittelwerte")
    ax[1].hist(stark_max, bins=bins, density=True, histtype="step", lw=2,
               color="#e08214", label="stärkstes Aggregatpaar")
    ax[1].axvline(SCHWELLE, color="grey", ls="--", lw=1)
    ax[1].set_xlabel("|Spearman ρ| zwischen zwei Kanälen")
    ax[1].set_ylabel("Dichte")
    ax[1].set_title("Verteilung über alle Kanalpaare", fontsize=10)
    ax[1].legend(fontsize=8)
    C.schliesse(pdf, fig, fuss)

    pdf.close()
    print(f"geschrieben: {C.OUT_DIR / '18_sensorkorrelation.pdf'}")
    print(f"  {len(kanaele)} Kanaele, {len(paare)} Paare")
    print(f"  zwischen Kanaelen: Median |rho| {betrag.median():.2f}, >0,7 {100 * (betrag > SCHWELLE).mean():.1f} %")
    print(f"  innerhalb Kanal:   Median |rho| {innen.median():.2f}, >0,7 {100 * (innen > SCHWELLE).mean():.1f} %")
    print(f"  innerhalb Auftrag: Median |rho| {paare_in.abs().median():.2f}")
    print(f"  Gegenprobe: Median max|rho| {stark_max.median():.2f}, >0,7 {int((stark_max > SCHWELLE).sum())} Paare, neu stark {len(neu_stark)} (davon _last {len(neu_last)})")
    for ij in neu_stark:
        if ij not in neu_last:
            print(f"    ohne last: {ij} {stark_kombi[ij]} {stark_max[ij]:.2f}")
    for i, j in zuwachs.index[:5]:
        print(f"    {i} - {j}: mean {abs(paare[(i, j)]):.2f} -> max {stark_max[(i, j)]:.2f} {stark_kombi[(i, j)]}")
    print("  staerkste Paare:")
    for i, j in top:
        print(f"    {i} - {j}: {korr.loc[i, j]:+.2f} (im Auftrag {korr_in.loc[i, j]:+.2f})")


if __name__ == "__main__":
    main()
