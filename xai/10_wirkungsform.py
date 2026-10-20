"""10 — Wirkungsform.

In der Arbeit: nicht berichtet; Vorarbeit zu den Abhängigkeiten in Abschnitt
7.3 (dort xai/14, xai/15).

Abhängigkeit der SHAP-Werte vom Merkmalswert, roh und artikelbereinigt.
Ausgaben unter reports/xai/, runs/xai/ und reports/xai/tabellen/.

    TEST_LEVEL=<billet|auftrag|artikel> python xai/10_wirkungsform.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import _common as C
from data_prep import SP  # noqa: E402

TOP = 12
KAT_GRENZE = 12
FARBE_HERKUNFT = {"Stammdaten": "#3182bd", "Prozess": "#e6550d"}


def shap_rangliste() -> pd.DataFrame:
    p = C.TAB_DIR / C.mit_ebene("07_kreuzvalidierung_xgboost.csv")
    if not p.exists():
        raise SystemExit(
            f"{p.name} fehlt. Erst 07_kreuzvalidierung.py fuer TEST_LEVEL="
            f"{C.D.TEST_LEVEL} laufen lassen."
        )
    return pd.read_csv(p)


def kennzahlen_04() -> dict[str, dict]:
    """rho_innen (Merkmal gegen y) aus 04 derselben Ebene — nur fuer die Textseite."""
    p = C.ERG_DIR / C.mit_ebene("04_verteilungen.json")
    if not p.exists():
        print(f"  Hinweis: {p.name} fehlt, Vergleich mit 04 entfaellt.")
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    return {r["Merkmal"]: r for r in d.get("merkmale", [])}


def artikel_der_testzeilen(ds, n_test: int) -> pd.Series:
    """Artikelnummer je Zeile des Test-Sets, in derselben Reihenfolge."""
    art = ds.df[SP["artikel"]].reset_index(drop=True)
    maske = np.asarray(ds.test_mask)
    gewaehlt = art[maske] if maske.dtype == bool else art.iloc[maske]
    gewaehlt = gewaehlt.reset_index(drop=True)
    assert len(gewaehlt) == n_test, (len(gewaehlt), n_test)
    return gewaehlt


def bereinigen(werte: pd.Series, gruppe: pd.Series) -> pd.Series:
    """Gruppenmittel abziehen — bei Gruppen der Groesse 1 bleibt exakt 0."""
    return werte - werte.groupby(gruppe.values).transform("mean")


def trend(ax, x, y, farbe="#d62728"):
    """Gleitender Median ueber die sortierten x — robuster als eine Regression."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 20:
        return
    xs, ys = x[ok], y[ok]
    o = np.argsort(xs)
    xs, ys = xs[o], ys[o]
    fenster = max(15, len(xs) // 12)
    m = pd.Series(ys).rolling(fenster, center=True, min_periods=fenster // 2).median()
    ax.plot(xs, m.values, color=farbe, lw=1.4, zorder=4)


def feld_merkmal(ax, x: pd.Series, s: np.ndarray, name: str, kopf: str, kat: bool):
    farbe = FARBE_HERKUNFT[C.herkunft(name)]
    xv = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    if kat:
        stufen = sorted(pd.unique(xv[np.isfinite(xv)]))
        daten = [s[xv == v] for v in stufen]
        bp = ax.boxplot(
            daten,
            positions=range(len(stufen)),
            widths=0.6,
            showfliers=False,
            patch_artist=True,
        )
        for kasten in bp["boxes"]:
            kasten.set(facecolor=farbe, alpha=0.55, linewidth=0.6)
        for med in bp["medians"]:
            med.set(color="black", linewidth=1.0)
        ax.set_xticks(range(len(stufen)))
        ax.set_xticklabels([f"{v:g}" for v in stufen], fontsize=5.2, rotation=90)
    else:
        ax.scatter(xv, s, s=5, color=farbe, alpha=0.45, linewidths=0, zorder=2)
        trend(ax, xv, s)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_title(kopf, fontsize=6.8)
    ax.tick_params(labelsize=5.5)
    ax.grid(alpha=0.25)


def seite_diagramme(pdf, X, S, merkmale, spalten, zeile2, titel, fuss, bereinigt):
    """zeile2: Merkmal -> zweite Titelzeile, gerechnet auf den gezeigten Punkten."""
    fig, ax = C.seite(titel, 3, 4)
    for i, m in enumerate(merkmale):
        j = spalten.index(m)
        kopf = m[:24] + "\n" + zeile2.get(m, "")
        x = X[m]
        kat = x.nunique(dropna=True) <= KAT_GRENZE and not bereinigt
        if bereinigt and float(np.nanstd(pd.to_numeric(x, errors="coerce"))) < 1e-12:
            ax[i].text(
                0.5,
                0.5,
                "keine Varianz\ninnerhalb des Artikels",
                ha="center",
                va="center",
                fontsize=7,
                color="#d62728",
                transform=ax[i].transAxes,
            )
            ax[i].set_title(kopf, fontsize=6.8)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            continue
        feld_merkmal(ax[i], x, S[:, j], m, kopf, kat)
    for j in range(len(merkmale), len(ax)):
        ax[j].axis("off")
    fig.supxlabel(
        "Merkmalswert" + (" (artikelbereinigt)" if bereinigt else ""),
        fontsize=8,
        y=0.048,
    )
    fig.supylabel(
        "SHAP-Wert [PP]" + (" (artikelbereinigt)" if bereinigt else ""),
        fontsize=8,
        x=0.012,
    )
    C.schliesse(pdf, fig, fuss)


def rangkorrelation(X: pd.DataFrame, S: np.ndarray, merkmale, spalten, name: str):
    """Spearman zwischen Merkmal und SHAP-Wert je Merkmal — die Zahl zum Feld."""
    from scipy.stats import spearmanr

    aus = []
    for m in merkmale:
        j = spalten.index(m)
        x = pd.to_numeric(X[m], errors="coerce").to_numpy(dtype=float)
        y = S[:, j]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 20 or np.nanstd(x[ok]) < 1e-12:
            aus.append({"Merkmal": m, name: None, "n": int(ok.sum())})
            continue
        aus.append({"Merkmal": m, name: float(spearmanr(x[ok], y[ok])[0]), "n": int(ok.sum())})
    return pd.DataFrame(aus)


def eta2_artikel(X: pd.DataFrame, art: pd.Series, merkmale) -> dict[str, float]:
    """Anteil der Merkmalsvarianz zwischen den Artikeln, auf den gezeigten
    Testzeilen."""
    aus = {}
    for m in merkmale:
        x = pd.to_numeric(X[m], errors="coerce")
        ok = x.notna().to_numpy() & art.notna().to_numpy()
        xx, gg = x[ok], art[ok].astype(str).to_numpy()
        gesamt = float(((xx - xx.mean()) ** 2).sum())
        if ok.sum() < 20 or gesamt < 1e-12:
            aus[m] = np.nan
            continue
        gm = xx.groupby(gg).transform("mean")
        aus[m] = float(((gm - xx.mean()) ** 2).sum() / gesamt)
    return aus


def fmt(v, form: str) -> str:
    return "—" if v is None or not np.isfinite(v) else format(v, form)


def main():
    print(f"Ebene: TEST_LEVEL={C.D.TEST_LEVEL}")
    rang = shap_rangliste()
    kenn = kennzahlen_04()

    ds, X_tr, X_te = C.lade()
    m, quelle = C.xgb_modell(X_tr, ds.y_train)
    S, basis = C.shap_xgboost(m, X_te)
    C.pruefe_additivitaet(S, basis, m.predict(X_te), "XGBoost Test-Set")
    spalten = list(X_te.columns)

    merkmale = [x for x in rang.Merkmal if x in spalten][:TOP]
    print("  Merkmale:", ", ".join(x[:22] for x in merkmale))

    art = artikel_der_testzeilen(ds, len(X_te))
    X_te = X_te.reset_index(drop=True)
    Xb = X_te[merkmale].apply(lambda s: bereinigen(pd.to_numeric(s, errors="coerce"), art))
    Sb = np.column_stack(
        [bereinigen(pd.Series(S[:, spalten.index(x)]), art).to_numpy() for x in merkmale]
    )
    Sb_voll = np.zeros_like(S)
    for i, x in enumerate(merkmale):
        Sb_voll[:, spalten.index(x)] = Sb[:, i]

    fuss = (
        f"{C.D.CSV_PATH.name} | Test-Set {len(X_te)} Billets, {art.nunique()} Artikel "
        f"| y ∈ [{C.D.Y_MIN:g}; {C.D.Y_MAX:g}] | Test-Ebene {C.D.TEST_LEVEL} | "
        f"Reihenfolge nach 07 (kreuzvalidiert) | XGBoost {quelle}"
    )

    eta = eta2_artikel(X_te, art, merkmale)
    kr = rangkorrelation(X_te, S, merkmale, spalten, "rho_roh_shap")
    kb = rangkorrelation(Xb, Sb_voll, merkmale, spalten, "rho_innen_shap")
    rho_roh = dict(zip(kr.Merkmal, kr.rho_roh_shap))
    rho_inn = dict(zip(kb.Merkmal, kb.rho_innen_shap))
    zeile_roh = {
        m: f"η²={fmt(eta[m], '.2f')}  ρ(SHAP)={fmt(rho_roh[m], '+.2f')}" for m in merkmale
    }
    zeile_inn = {m: f"ρ_innen(SHAP)={fmt(rho_inn[m], '+.2f')}" for m in merkmale}

    pdf = C.pdf_oeffnen(C.mit_ebene("10_wirkungsform.pdf"))
    seite_diagramme(
        pdf,
        X_te,
        S,
        merkmale,
        spalten,
        zeile_roh,
        f"Wirkungsform der Top {TOP} — roh (Ebene {C.D.TEST_LEVEL})",
        fuss + " | η²: Varianzanteil zwischen Artikeln, ρ: Spearman Merkmal–SHAP",
        bereinigt=False,
    )
    seite_diagramme(
        pdf,
        Xb,
        Sb_voll,
        merkmale,
        spalten,
        zeile_inn,
        f"Dieselben Merkmale, Artikelmittel bei Merkmal UND SHAP-Wert abgezogen "
        f"(Ebene {C.D.TEST_LEVEL})",
        fuss + " | ρ_innen: Spearman bereinigtes Merkmal–bereinigter SHAP-Wert",
        bereinigt=True,
    )

    ohne = kb[kb.rho_innen_shap.isna()].Merkmal.tolist()
    schwach = kb[(kb.rho_innen_shap.notna()) & (kb.rho_innen_shap.abs() < 0.1)].Merkmal.tolist()
    stark = kb[(kb.rho_innen_shap.notna()) & (kb.rho_innen_shap.abs() >= 0.1)]
    vergleich = [
        f"{m} ({fmt(rho_inn[m], '+.2f')} / {fmt(kenn[m].get('rho_innen'), '+.2f')})"
        for m in merkmale
        if m in kenn
    ]

    C.textseite(
        pdf,
        "Lesehinweise und Vorbehalte",
        [
            f"Vorgehen. Gezeigt werden die {len(merkmale)} Merkmale mit dem groessten "
            f"kreuzvalidierten |SHAP| aus 07. Die Punkte stammen aus dem "
            f"eingefrorenen Test-Set ({len(X_te)} Billets, {art.nunique()} Artikel), "
            f"erklaert vom Test-Set-Modell wie in 01_global. Merkmale mit hoechstens "
            f"{KAT_GRENZE} Auspraegungen werden als Streifen je Auspraegung "
            f"gezeichnet, weil die ordinale Kodierung sonst eine Ordnung vortaeuscht, "
            f"die es nicht gibt.",
            "Zahlen in den Feldtiteln. Jede Zahl gehoert zu den Punkten ihres "
            "Feldes. Seite 1: η² ist der Anteil der Merkmalsvarianz zwischen den "
            "Artikeln, ρ(SHAP) die Rangkorrelation zwischen Merkmal und SHAP-Wert. "
            "Seite 2: ρ_innen(SHAP) ist dieselbe Rangkorrelation nach Abzug der "
            "Artikelmittel — das sind die Werte im Befund unten. Bei kategorial "
            "gezeichneten Merkmalen auf Seite 1 misst ρ nur die Monotonie ueber die "
            "Kodierung und ist entsprechend vorsichtig zu lesen.",
            "Bereinigung auf Seite 2. Von Merkmalswert UND SHAP-Wert wird das Mittel "
            "des jeweiligen Artikels abgezogen. Uebrig bleibt nur, was die Billets "
            "EINES Artikels voneinander unterscheidet. Das ist die Ebene, auf der "
            "eine prozessuale Handlungsempfehlung ueberhaupt formuliert werden "
            "koennte — zwischen Artikeln laesst sich nicht steuern, innerhalb eines "
            "Artikels schon.",
            f"Befund. Bei {len(ohne)} der {len(merkmale)} Merkmale bleibt nach der "
            f"Bereinigung keine Varianz uebrig, das Merkmal ist also vollstaendig "
            f"durch den Artikel festgelegt"
            + (f": {', '.join(ohne[:8])}." if ohne else ".")
            + f" Bei {len(schwach)} weiteren liegt die Rangkorrelation zwischen "
            f"bereinigtem Merkmal und bereinigtem SHAP-Wert unter 0,1. Ueber dieser "
            f"Schwelle bleiben {len(stark)} Merkmale"
            + (
                f": {', '.join(f'{r.Merkmal} ({r.rho_innen_shap:+.2f})' for r in stark.itertuples())}."
                if len(stark)
                else "."
            ),
            "Verhaeltnis zu 04. Dort wird derselbe Test auf der ZIELGROESSE "
            "gerechnet (rho_innen zwischen bereinigtem Merkmal und bereinigtem y, "
            "auf allen Billets), hier auf dem SHAP-Wert im Test-Set. Die beiden "
            "Zahlen beantworten verschiedene Fragen: 04 fragt, ob innerhalb eines "
            "Artikels ein Zusammenhang in den DATEN steht, 10 fragt, ob das MODELL "
            "einen nutzt. Ein grosser Wert hier bei kleinem Wert in 04 heisst: das "
            "Modell reagiert auf eine Streuung, die mit dem Ausschuss kaum "
            "zusammenhaengt (Wechselwirkung oder Ueberanpassung)."
            + (
                f" Gegenueberstellung ρ_innen(SHAP) / ρ_innen(y): {'; '.join(vergleich)}."
                if vergleich
                else f" ({C.mit_ebene('04_verteilungen.json')} fehlt, kein Zahlenvergleich.)"
            ),
            "Vorbehalt. Ein Abhaengigkeitsdiagramm zeigt die Wirkung des Merkmals "
            "einschliesslich aller Wechselwirkungen, in die es eingeht; die "
            "vertikale Streuung bei gleichem x ist genau dieser Anteil und nicht "
            "Rauschen. Die Kurve ist ausserdem keine Wirkungsfunktion des Prozesses, "
            "sondern die des Modells: sie beschreibt, wie das Modell auf den Wert "
            "reagiert, nicht wie die Anlage es tut.",
        ],
    )
    pdf.close()

    tab = (
        pd.DataFrame({"Merkmal": merkmale})
        .merge(rang[["Merkmal", "Mittel"]].rename(columns={"Mittel": "shap_mittel"}),
               on="Merkmal", how="left")
        .merge(kr[["Merkmal", "rho_roh_shap"]], on="Merkmal", how="left")
        .merge(kb, on="Merkmal", how="left")
    )
    tab["eta2_artikel_test"] = [eta[x] for x in tab.Merkmal]
    tab["rho_innen_y_04"] = [kenn.get(x, {}).get("rho_innen") for x in tab.Merkmal]
    pcsv = C.speichere_rangliste(C.mit_ebene("10_wirkungsform.csv"), tab, "shap_mittel")

    p = C.speichere(
        C.mit_ebene("10_wirkungsform.json"),
        {
            "test_level": C.D.TEST_LEVEL,
            "n_test": int(len(X_te)),
            "n_artikel": int(art.nunique()),
            "params_quelle": quelle,
            "merkmale": merkmale,
            "kennzahlen": tab.to_dict("records"),
            "ohne_varianz_innerhalb_artikel": ohne,
            "rho_unter_0_1": schwach,
            "vergleich_04": C.mit_ebene("04_verteilungen.json") if kenn else None,
            "rangliste_csv": pcsv.name,
        },
    )
    print(f"\ngeschrieben: {C.OUT_DIR / C.mit_ebene('10_wirkungsform.pdf')}\n             {p}"
          f"\n             {pcsv}\n")
    print(f"  {'Merkmal':<32}{'η²':>6}{'ρ roh':>8}{'ρ innen':>9}{'ρ(y) 04':>9}")
    for r in tab.itertuples():
        print(
            f"  {r.Merkmal:<32}{fmt(r.eta2_artikel_test, '.2f'):>6}"
            f"{fmt(r.rho_roh_shap, '+.2f'):>8}{fmt(r.rho_innen_shap, '+.2f'):>9}"
            f"{fmt(r.rho_innen_y_04, '+.2f'):>9}"
        )


if __name__ == "__main__":
    main()
