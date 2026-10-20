"""12 — Extreme Billets.

In der Arbeit: Abschnitt 7.4: Billets des oberen und unteren Ausschussdezils,
erkannte und verfehlte Billets (Abbildung E.1 und E.2, gezeichnet von xai/23).

Merkmale, die die Vorhersage bei sehr hohem und sehr niedrigem Ausschuss
treiben (Out-of-fold-SHAP).
Ausgaben unter reports/xai/, runs/xai/ und reports/xai/tabellen/.

    TEST_LEVEL=<billet|auftrag|artikel> python xai/12_extreme.py
"""

from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import trim_mean

import _common as C
from data_prep import SP  # noqa: E402

CV_SEED = 42
QUANTIL = 0.10
TOP = 20
FARBE_HERKUNFT = {"Stammdaten": "#3182bd", "Prozess": "#e6550d"}


def cv_ebene() -> str:
    return "random" if C.D.TEST_LEVEL == "billet" else C.D.TEST_LEVEL


def oof_shap(ds, falten):
    """SHAP und kodierte Merkmalswerte für alle Billets, je aus ihrer
    Validierungsfalte."""
    X_cv = ds.X_raw.reset_index(drop=True)
    y_cv = ds.y.reset_index(drop=True)
    n = len(X_cv)
    teile_s, teile_x = [], []
    vorher = np.full(n, np.nan)
    for k, (tr, va) in enumerate(falten, 1):
        Xtr, Xva, _ = C.D.process_features(X_cv.iloc[tr], X_cv.iloc[va])
        m, quelle = C.xgb_modell(Xtr, y_cv.iloc[tr])
        p = m.predict(Xva)
        b, basis = C.shap_xgboost(m, Xva)
        C.pruefe_additivitaet(b, basis, p, f"XGBoost Falte {k}")
        vorher[va] = p
        teile_s.append(pd.DataFrame(b, columns=Xva.columns, index=va))
        teile_x.append(pd.DataFrame(np.asarray(Xva, dtype=float), columns=Xva.columns, index=va))
        print(f"  Falte {k}: n_val {len(va)}, {Xva.shape[1]} Merkmale", flush=True)
    S = pd.concat(teile_s).sort_index().fillna(0.0)
    X = pd.concat(teile_x).sort_index()
    return S, X, vorher, quelle


def gruppen(y) -> dict[str, np.ndarray]:
    oben, unten = np.quantile(y, 1 - QUANTIL), np.quantile(y, QUANTIL)
    return {"oben": y > oben, "unten": y < unten, "Rest": (y >= unten) & (y <= oben)}


def _wert(roh: pd.Series) -> str:
    """Lesbarer Lagewert aus den ROHEN Merkmalswerten."""
    w = roh.dropna()
    if w.empty:
        return "—"
    if not pd.api.types.is_numeric_dtype(w):
        h = w.astype(str).value_counts(normalize=True)
        return f"{textwrap.shorten(h.index[0], 14, placeholder='…')} {h.iloc[0]:.0%}"
    if (w == 0).mean() > 0.5:
        return f"≠ 0: {(w != 0).mean():.0%}, Mittel {w[w != 0].mean():.3g}"
    return f"{w.median():.4g} / {trim_mean(w, 0.1):.4g}"


def kennzahlen(S, X_roh, maske) -> pd.DataFrame:
    rest = ~maske
    t = pd.DataFrame({"beitrag": S[maske].mean(), "betrag": S[maske].abs().mean()})
    zeigen = t.sort_values("betrag", ascending=False).head(TOP).index
    t["wert_gruppe"] = pd.Series({m: _wert(X_roh[m][maske]) if m in X_roh else "—" for m in zeigen})
    t["wert_rest"] = pd.Series({m: _wert(X_roh[m][rest]) if m in X_roh else "—" for m in zeigen})
    return t


def _balken(ax, t, titel):
    t = t.sort_values("betrag", ascending=False).head(TOP).iloc[::-1]
    farben = [FARBE_HERKUNFT[C.herkunft(m)] for m in t.index]
    ax.barh(range(len(t)), t.beitrag, color=farben)
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels(t.index, fontsize=6.5)
    for i, (b, a) in enumerate(zip(t.beitrag, t.betrag)):
        ax.annotate(
            f"{b:+.2f} (|{a:.2f}|)",
            xy=(b, i),
            xytext=(3 if b >= 0 else -3, 0),
            textcoords="offset points",
            ha="left" if b >= 0 else "right",
            va="center",
            fontsize=5.5,
        )
    ax.axvline(0, color="black", lw=0.8)
    lo, hi = min(t.beitrag.min(), 0), max(t.beitrag.max(), 0)
    rand = 0.35 * (hi - lo)
    ax.set_xlim(lo - rand, hi + rand)
    ax.set_xlabel("mittlerer SHAP-Beitrag in der Gruppe [Pp]  (in Klammern: mittlerer Betrag)")
    ax.set_title(titel, fontsize=10)
    ax.grid(alpha=0.3, axis="x")


def seite_gruppen(pdf, K, gr, fuss):
    fig, ax = C.seite(f"Was das Modell bei extremen Billets nutzt — Ebene {C.D.TEST_LEVEL}", 1, 2)
    _balken(ax[0], K["oben"], f"Extrem hoher Ausschuss (oberste 10 %, n = {gr['oben'].sum()})")
    _balken(ax[1], K["unten"], f"Extrem niedriger Ausschuss (unterste 10 %, n = {gr['unten'].sum()})")
    ax[1].legend(
        handles=[Patch(color=f, label=h) for h, f in FARBE_HERKUNFT.items()],
        fontsize=7,
        loc="lower right",
    )
    C.schliesse(pdf, fig, fuss)


def seite_trennung(pdf, K, S, y, vorher, gr, fuss):
    fig, ax = C.seite(f"Trennende Merkmale und erkannte Billets — Ebene {C.D.TEST_LEVEL}", 1, 2)

    a = ax[0]
    o, u = K["oben"].beitrag, K["unten"].beitrag
    farben = [FARBE_HERKUNFT[C.herkunft(m)] for m in o.index]
    a.scatter(o, u, s=10, c=farben, alpha=0.6, edgecolors="none")
    trenn = (o - u).sort_values(ascending=False)
    beschriften = list(trenn.head(10).index) + list(trenn.tail(3).index)
    for m in beschriften:
        a.annotate(m, (o[m], u[m]), fontsize=5.5, xytext=(3, 2), textcoords="offset points")
    a.axhline(0, color="#555555", lw=0.8)
    a.axvline(0, color="#555555", lw=0.8)
    a.set_xlabel("mittlerer Beitrag bei extrem hohem Ausschuss [Pp]")
    a.set_ylabel("mittlerer Beitrag bei extrem niedrigem Ausschuss [Pp]")
    a.set_title("A  Unten rechts: schiebt hohe hoch und niedrige runter", fontsize=10)
    a.grid(alpha=0.3)

    b = ax[1]
    treffer = gr["oben"] & (vorher >= np.quantile(vorher, 0.9))
    verfehlt = gr["oben"] & (vorher < np.median(vorher))
    d = (S[treffer].mean() - S[verfehlt].mean()).sort_values(key=np.abs, ascending=False)
    d = d.head(15).iloc[::-1]
    b.barh(range(len(d)), d.values, color=[FARBE_HERKUNFT[C.herkunft(m)] for m in d.index])
    b.set_yticks(range(len(d)))
    b.set_yticklabels(d.index, fontsize=6.5)
    for i, v in enumerate(d.values):
        b.annotate(
            f"{v:+.2f}",
            xy=(v, i),
            xytext=(3 if v >= 0 else -3, 0),
            textcoords="offset points",
            ha="left" if v >= 0 else "right",
            va="center",
            fontsize=6,
        )
    b.axvline(0, color="black", lw=0.8)
    lo, hi = min(d.min(), 0), max(d.max(), 0)
    b.set_xlim(lo - 0.25 * (hi - lo), hi + 0.25 * (hi - lo))
    b.set_xlabel("mittlerer Beitrag erkannt − verfehlt [Pp]")
    b.set_title(
        f"B  Hoher Ausschuss: erkannt (n = {treffer.sum()}) gegen verfehlt (n = {verfehlt.sum()})",
        fontsize=10,
    )
    b.grid(alpha=0.3, axis="x")
    C.schliesse(pdf, fig, fuss)
    return trenn, treffer, verfehlt


def seite_tabelle(pdf, K):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11.7, 8.3))
    fig.text(0.04, 0.95, f"Top 15 je Gruppe mit Merkmalswerten — Ebene {C.D.TEST_LEVEL}", fontsize=13, weight="bold", va="top")
    kopf = f"{'Merkmal':<40}{'Beitrag':>8}{'Betrag':>8}  {'Gruppe':<24}{'Rest':<24}"
    for spalte, (gruppe, titel) in enumerate([("oben", "Extrem hoher Ausschuss"), ("unten", "Extrem niedriger Ausschuss")]):
        t = K[gruppe].sort_values("betrag", ascending=False).head(15)
        zeilen = [titel, "", kopf, "-" * len(kopf)]
        for m, r in t.iterrows():
            zeilen.append(
                f"{textwrap.shorten(m, 39, placeholder='…'):<40}{r.beitrag:>+8.2f}{r.betrag:>8.2f}"
                f"  {r.wert_gruppe:<24}{r.wert_rest:<24}"
            )
        fig.text(0.04, 0.88 - spalte * 0.42, "\n".join(zeilen), fontsize=7.2, family="monospace", va="top")
    fig.text(
        0.04,
        0.03,
        "Werte: numerisch Median / 10-%-getrimmtes Mittel, kategorial häufigste Kategorie mit Anteil. "
        "Beitrag = mittlerer SHAP-Wert, Betrag = mittlerer |SHAP| in der Gruppe [Pp].",
        fontsize=7.5,
        style="italic",
    )
    pdf.savefig(fig)
    plt.close(fig)


def main():
    print(f"Ebene: TEST_LEVEL={C.D.TEST_LEVEL}  →  CV-Ebene {cv_ebene()}")
    ds = C.D.load_split(verbose=False)
    falten, ebene = C.D.make_cv(ds.df, level=cv_ebene(), seed=CV_SEED)
    S, X, vorher, quelle = oof_shap(ds, falten)
    y = ds.y.reset_index(drop=True).to_numpy()
    gr = gruppen(y)
    X_roh = ds.X_raw.reset_index(drop=True)
    K = {g: kennzahlen(S, X_roh, m) for g, m in gr.items()}
    fuss = (
        f"{C.D.CSV_PATH.name} | {len(y)} Billets, OOF-TreeSHAP aus {len(falten)} Falten "
        f"(Ebene {ebene!r}) | XGBoost {quelle}"
    )

    pdf = C.pdf_oeffnen(C.mit_ebene("12_extreme.pdf"))
    seite_gruppen(pdf, K, gr, fuss)
    trenn, treffer, verfehlt = seite_trennung(pdf, K, S, y, vorher, gr, fuss)
    seite_tabelle(pdf, K)

    diff = (S[treffer].mean() - S[verfehlt].mean()).sort_values(key=np.abs, ascending=False).head(15)
    abbruch = pd.to_numeric(X_roh[SP["phase"] + "_max"], errors="coerce").to_numpy(float) < SP["letzte_phase"]
    erkannt_vs_verfehlt = {
        "n_erkannt": int(treffer.sum()), "n_verfehlt": int(verfehlt.sum()),
        "y_mittel": {"erkannt": float(y[treffer].mean()), "verfehlt": float(y[verfehlt].mean())},
        "vorhersage_mittel": {"erkannt": float(vorher[treffer].mean()),
                              "verfehlt": float(vorher[verfehlt].mean())},
        "anteil_abbruch": {"erkannt": float(abbruch[treffer].mean()),
                           "verfehlt": float(abbruch[verfehlt].mean())},
        "merkmale": [{
            "merkmal": m, "shap_diff": float(diff[m]),
            "median_erkannt": (float(pd.to_numeric(X_roh[m], errors="coerce")[treffer].median())
                               if m in X_roh else None),
            "median_verfehlt": (float(pd.to_numeric(X_roh[m], errors="coerce")[verfehlt].median())
                                if m in X_roh else None),
        } for m in diff.index],
    }
    C.textseite(
        pdf,
        "Lesehinweise und Vorbehalte",
        [
            "SHAP erklärt das MODELL, nicht die Ursache im Prozess. Ein grosser Beitrag heisst: "
            "das Modell stützt seine Vorhersage für diese Billets auf das Merkmal.",
            "Die Gruppen sind nach dem tatsächlichen y gebildet. Ein Merkmal kann in der Gruppe "
            "'oben' einen grossen Beitrag haben, obwohl das Modell viele dieser Billets verfehlt "
            "— Seite 2 B trennt deshalb erkannte von verfehlten Billets.",
            "Korrelierte Merkmale teilen sich den Beitrag (siehe 09_redundanz). Ein einzelnes "
            "Merkmal weit oben steht oft für eine ganze Gruppe ähnlicher Spalten.",
            "Merkmale, die je Artikel konstant sind (Rezept, Sollmasse), können in der Gruppe "
            "'oben' nur deshalb auftauchen, weil bestimmte Artikel häufig viel Ausschuss haben. "
            "Auf Artikelebene (TEST_LEVEL=artikel) ist das am wenigsten der Fall.",
            "*_n_samples und *_n_edges zählen Messpunkte bzw. Schaltflanken im Billetfenster. "
            "Sie hängen an Fensterlänge und Anlage (*_n_samples sind weitgehend "
            "Maschinen-Fingerabdrücke) und sind deshalb kein direktes Prozessmass.",
            "Nur XGBoost; zur Übereinstimmung mit CatBoost siehe 07_kreuzvalidierung.",
        ],
    )
    pdf.close()

    C.speichere(
        C.mit_ebene("12_extreme.json"),
        {
            "ebene": C.D.TEST_LEVEL,
            "cv_ebene": ebene,
            "n": int(len(y)),
            "n_oben": int(gr["oben"].sum()),
            "n_unten": int(gr["unten"].sum()),
            "n_treffer": int(treffer.sum()),
            "n_verfehlt": int(verfehlt.sum()),
            "top_oben": K["oben"].sort_values("betrag", ascending=False).head(TOP).round(4).reset_index(names="merkmal").to_dict("records"),
            "top_unten": K["unten"].sort_values("betrag", ascending=False).head(TOP).round(4).reset_index(names="merkmal").to_dict("records"),
            "trenner": trenn.head(TOP).round(4).to_dict(),
            "erkannt_vs_verfehlt": erkannt_vs_verfehlt,
        },
    )
    print(f"geschrieben: {C.OUT_DIR / C.mit_ebene('12_extreme.pdf')}")
    for g in ("oben", "unten"):
        print(f"\n== {g}")
        print(K[g].sort_values("betrag", ascending=False).head(12).round(3).to_string())
    print("\n== Trenner (Beitrag oben − unten)")
    print(trenn.head(12).round(3).to_string())


if __name__ == "__main__":
    main()
