"""04 — Verteilung der wichtigsten Merkmale.

In der Arbeit: nicht berichtet; Streuung der wichtigsten Merkmale innerhalb und
zwischen Artikeln.

Streuung innerhalb und zwischen Artikeln (braucht xai/07).
Ausgaben unter reports/xai/, runs/xai/ und reports/xai/tabellen/.

    TEST_LEVEL=<billet|auftrag|artikel> python xai/04_verteilungen.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import _common as C
from data_prep import SP  # noqa: E402

TOP = 20
ETA_GRENZE = 0.70
RHO_GRENZE = 0.10


def kennzahlen(
    df: pd.DataFrame, y: pd.Series, gruppe: pd.Series, merkmale: list[str]
) -> pd.DataFrame:
    z = []
    for m in merkmale:
        x = pd.to_numeric(df[m], errors="coerce")
        ok = x.notna() & y.notna() & gruppe.notna()
        if ok.sum() < 30 or x[ok].nunique() < 2:
            z.append((m, np.nan, np.nan, np.nan, int(ok.sum()), x.nunique()))
            continue
        xx, yy, gg = x[ok], y[ok], gruppe[ok].astype(str)
        mu = xx.mean()
        gm = xx.groupby(gg).transform("mean")
        eta = float(((gm - mu) ** 2).sum() / ((xx - mu) ** 2).sum())
        roh = float(spearmanr(xx, yy).statistic)
        xi = xx - gm
        yi = yy - yy.groupby(gg).transform("mean")
        innen = (
            float(spearmanr(xi, yi).statistic)
            if xi.nunique() > 1 and yi.nunique() > 1
            else np.nan
        )
        z.append((m, eta, roh, innen, int(ok.sum()), int(xx.nunique())))
    return pd.DataFrame(
        z, columns=["Merkmal", "eta2_artikel", "rho_roh", "rho_innen", "n", "uniq"]
    )


def rangliste_07() -> list[str]:
    """Top-Merkmale nach kreuzvalidiertem |SHAP| aus 07 — wie in 08 bis 10."""
    p = C.TAB_DIR / C.mit_ebene("07_kreuzvalidierung_xgboost.csv")
    if not p.exists():
        raise SystemExit(
            f"{p.name} fehlt. Erst 07_kreuzvalidierung.py fuer TEST_LEVEL="
            f"{C.D.TEST_LEVEL} laufen lassen."
        )
    return pd.read_csv(p).Merkmal.head(TOP).tolist()


def seite_verteilungen(pdf, df, merkmale, k):
    fig, ax = C.seite(
        f"Verteilung der {TOP} wichtigsten Merkmale nach 07 (Ebene {C.D.TEST_LEVEL})  "
        "(η² = Anteil der Varianz zwischen den Artikeln)",
        4,
        5,
    )
    for a, m in zip(ax, merkmale):
        x = pd.to_numeric(df[m], errors="coerce").dropna()
        r = k.loc[k.Merkmal == m].iloc[0]
        wenige = r.uniq <= 12
        if wenige:
            v = x.value_counts().sort_index()
            a.bar(range(len(v)), v.values, color="#3182bd")
            a.set_xticks(range(len(v)))
            a.set_xticklabels([f"{i:g}" for i in v.index], fontsize=5.5)
        else:
            a.hist(x, bins=40, color="#3182bd")
        farbe = "#d62728" if r.eta2_artikel >= ETA_GRENZE else "#1a6b1a"
        a.set_title(f"{m[:26]}\nη² = {r.eta2_artikel:.2f}", fontsize=7, color=farbe)
        a.tick_params(labelsize=5.5)
        a.set_yticks([])
    for a in ax[len(merkmale) :]:
        a.axis("off")
    C.schliesse(
        pdf,
        fig,
        f"{C.D.CSV_PATH.name} | alle {len(df)} gefilterten "
        f"Billets | rot: η² ≥ {ETA_GRENZE:.0%} (artikelbestimmt) | "
        f"Auswahl nach 07, Ebene {C.D.TEST_LEVEL}",
    )


def seite_diagnose(pdf, k):
    fig, ax = C.seite("Artikelmerkmal oder echtes Prozesssignal?", 1, 2)

    a = ax[0]
    a.axvspan(ETA_GRENZE, 1.02, color="#d62728", alpha=0.08)
    a.axhspan(0, RHO_GRENZE, color="#999999", alpha=0.10)
    for _, r in k.iterrows():
        if not np.isfinite(r.eta2_artikel):
            continue
        farbe = "#e6550d" if C.herkunft(r.Merkmal) == "Prozess" else "#3182bd"
        ohne = not np.isfinite(r.rho_innen)
        yv = 0.0 if ohne else abs(r.rho_innen)
        a.scatter(
            r.eta2_artikel,
            yv,
            s=52 if ohne else 42,
            color="none" if ohne else farbe,
            edgecolors=farbe,
            linewidths=1.6 if ohne else 0,
            zorder=3,
        )
        a.annotate(
            r.Merkmal[:22] + (" *" if ohne else ""),
            (r.eta2_artikel, yv),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=6,
        )
    a.axvline(ETA_GRENZE, color="#d62728", ls="--", lw=1)
    a.axhline(RHO_GRENZE, color="#666666", ls="--", lw=1)
    a.set_xlim(-0.02, 1.05)
    a.set_xlabel("η²(Artikel) — wie stark der Artikel das Merkmal festlegt")
    a.set_ylabel("|ρ| mit y INNERHALB des Artikels")
    a.set_title(
        "A  Diagnose  (rechts = Artikelmerkmal, unten = kein " "Restsignal)",
        fontsize=10,
    )
    a.grid(alpha=0.3)
    a.annotate(
        "*  innerhalb des Artikels konstant —\n" "   keine Reststreuung vorhanden",
        (0.02, 0.93),
        xycoords="axes fraction",
        fontsize=6.5,
        color="#444444",
        va="top",
    )
    from matplotlib.patches import Patch

    a.legend(
        handles=[
            Patch(color="#3182bd", label="Stammdaten"),
            Patch(color="#e6550d", label="Prozess"),
        ],
        fontsize=7.5,
    )

    b = ax[1]
    t = k.dropna(subset=["rho_roh"]).sort_values("rho_roh", key=abs).tail(TOP)
    p = np.arange(len(t))
    b.barh(p - 0.2, t.rho_roh.abs(), 0.4, color="#9ecae1", label="roh")
    b.barh(
        p + 0.2, t.rho_innen.abs(), 0.4, color="#08519c", label="innerhalb des Artikels"
    )
    b.set_yticks(p)
    b.set_yticklabels([m[:26] for m in t.Merkmal], fontsize=6.5)
    b.set_xlabel("|Spearman| mit dem Ausschussgrad")
    b.set_title("B  Was von der Korrelation übrig bleibt", fontsize=10)
    b.legend(fontsize=7.5, loc="lower right")
    b.grid(alpha=0.3, axis="x")
    C.schliesse(
        pdf,
        fig,
        f"{C.D.CSV_PATH.name} | alle gefilterten Billets | "
        f"Artikelmittel bei Merkmal UND Ziel abgezogen",
    )


def main():
    print(f"Ebene: TEST_LEVEL={C.D.TEST_LEVEL}")
    merkmale = rangliste_07()

    ds = C.D.load_split(verbose=False)
    X_all, _, _ = C.D.process_features(ds.X_raw, ds.X_raw)
    y_all = ds.y.reset_index(drop=True)
    art = ds.df[SP["artikel"]].reset_index(drop=True)
    X_all = X_all.reset_index(drop=True)
    assert len(X_all) == len(y_all) == len(art), (len(X_all), len(y_all), len(art))
    n = len(X_all)

    k = kennzahlen(X_all, y_all, art, merkmale)

    pdf = C.pdf_oeffnen(C.mit_ebene("04_verteilungen.pdf"))
    seite_verteilungen(pdf, X_all, merkmale, k)
    seite_diagnose(pdf, k)

    artikelnah = k[k.eta2_artikel >= ETA_GRENZE]
    restsignal = k[k.rho_innen.abs() >= RHO_GRENZE]
    ohne_streuung = k[~np.isfinite(k.rho_innen)]
    C.textseite(
        pdf,
        "Lesehinweise und Vorbehalte",
        [
            "Die beiden Kennzahlen. η²(Artikel) ist der Anteil der Varianz eines "
            "Merkmals, der zwischen den Artikeln liegt; nahe eins heisst, dass der "
            "Artikel den Wert praktisch festlegt. ρ innerhalb misst die "
            "Rangkorrelation mit dem Ausschussgrad, nachdem bei Merkmal UND Ziel "
            "das jeweilige Artikelmittel abgezogen wurde — also genau den "
            "Zusammenhang, den ein Artikelbezeichner nicht erzeugen kann.",
            f"Auswahl. Gezeigt werden die {TOP} Merkmale mit dem groessten "
            f"kreuzvalidierten |SHAP| aus 07 (Ebene {C.D.TEST_LEVEL}), dieselbe "
            f"Rangliste wie in 08 bis 10. Die Kennzahlen selbst haengen nicht von "
            f"der Ebene ab, nur die Auswahl der Merkmale.",
            f"Befund. Von den {TOP} wichtigsten Merkmalen sind "
            f"{len(artikelnah)} weitgehend artikelbestimmt (η² ≥ "
            f"{ETA_GRENZE:.0%}); {len(restsignal)} behalten innerhalb des Artikels "
            f"eine Rangkorrelation von mindestens {RHO_GRENZE:.2f} mit dem "
            f"Ausschussgrad; bei {len(ohne_streuung)} existiert innerhalb des "
            f"Artikels ueberhaupt keine Streuung, sie sind dort konstant.",
            f"Warum η² allein nicht reicht. Bei {art.nunique()} Artikeln und "
            f"{n} Billets liegt die Nullerwartung von η² schon bei rund "
            f"{(art.nunique()-1)/(n-1):.3f}. Ein Wert knapp darüber ist kein "
            f"Beleg für Artikelbindung. Die Diagnose stützt sich deshalb auf die "
            f"Kombination beider Kennzahlen und nicht auf η² für sich.",
            "Was die Diagnose NICHT zeigt. Ein Merkmal ohne Restsignal innerhalb "
            "des Artikels kann trotzdem nützlich sein — es trägt dann Information "
            "über den Artikel, die dem Modell bei bekannten Erzeugnissen hilft. "
            "Unbrauchbar wird es erst bei der Vorhersage für ein unbekanntes "
            "Erzeugnis. Umgekehrt ist ein Restsignal kein Kausalnachweis, sondern "
            "eine Assoziation, die auch von einer dritten Größe stammen kann.",
            f"Bezugsmenge. Anders als 01 bis 03 rechnet diese Auswertung auf allen "
            f"{n} gefilterten Billets und nicht nur auf dem Test-Set: Die Frage "
            f"betrifft die Struktur der Daten, nicht das Verhalten des Modells.",
        ],
    )
    pdf.close()

    p = C.speichere(
        C.mit_ebene("04_verteilungen.json"),
        {
            "test_level": C.D.TEST_LEVEL,
            "auswahl": C.mit_ebene("07_kreuzvalidierung_xgboost.csv"),
            "eta_grenze": ETA_GRENZE,
            "rho_grenze": RHO_GRENZE,
            "n_zeilen": int(n),
            "n_artikel": int(art.nunique()),
            "eta_nullerwartung": float((art.nunique() - 1) / (n - 1)),
            "merkmale": k.to_dict("records"),
        },
    )
    print(f"geschrieben: {C.OUT_DIR / C.mit_ebene('04_verteilungen.pdf')}\n             {p}\n")
    print(f"{'Merkmal':<34}{'η²(Art.)':>10}{'ρ roh':>9}{'ρ innen':>10}{'Quelle':>12}")
    for _, r in k.iterrows():
        print(
            f"{r.Merkmal:<34}{r.eta2_artikel:>10.3f}{r.rho_roh:>+9.3f}"
            f"{r.rho_innen:>+10.3f}{C.herkunft(r.Merkmal):>12}"
        )


if __name__ == "__main__":
    main()
