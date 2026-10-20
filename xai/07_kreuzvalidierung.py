"""07 — Wichtigkeit je Validierungsfalte.

In der Arbeit: Abschnitt 7.2: Abbildung 16 (Top-20 nach SHAP über die Falten);
kreuzvalidierte SHAP-Werte als Grundlage für xai/13–15 und xai/17.

SHAP-Wichtigkeit von XGBoost und CatBoost je Falte und Stabilität der
Rangliste. Grundlage für xai/04, 08–11 und 13.
Ausgaben unter reports/xai/, runs/xai/ und reports/xai/tabellen/.

    TEST_LEVEL=<billet|auftrag|artikel> python xai/07_kreuzvalidierung.py
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

import _common as C

TOP = 20
CV_SEED = 42
MODELLE = ("XGBoost", "CatBoost")
KURZ = {"XGBoost": "XGB", "CatBoost": "CB"}
N_LANDKARTE = 25
FARBEN_FALTEN = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
FARBE_HERKUNFT = {"Stammdaten": "#3182bd", "Prozess": "#e6550d"}


def cv_ebene() -> str:
    """CV-Ebene passend zur Test-Ebene; "billet" hat keine Gruppen."""
    return "random" if C.D.TEST_LEVEL == "billet" else C.D.TEST_LEVEL


def modell_fit(name: str, X_tr, y_tr):
    """Ein Modell mit den Parametern seines Trainingsskripts."""
    if name == "XGBoost":
        import xgboost as xgb

        params, quelle = C.xgb_parameter()
        _, fix = C._konstanten_aus_skript()
        m = xgb.XGBRegressor(**params, **fix, eval_metric="rmse")
        m.fit(X_tr, y_tr)
        return m, quelle

    from catboost import CatBoostRegressor

    params, quelle = C.catboost_parameter()
    _, fix = C._konstanten_aus_skript("catboost_regression.py")
    m = CatBoostRegressor(**params, **fix)
    m.fit(X_tr, y_tr, verbose=False)
    return m, quelle


def modell_shap(name: str, modell, X, y):
    """Exakter TreeSHAP des jeweiligen Pakets (siehe Kopf von _common.py)."""
    if name == "XGBoost":
        return C.shap_xgboost(modell, X)
    return C.shap_catboost(modell, X, y)


def falten_shap(ds, falten):
    """Je Falte und Modell: Wichtigkeitstabelle, Guete, Groesse."""
    X_cv = ds.X_raw.reset_index(drop=True)
    y_cv = ds.y.reset_index(drop=True)
    df_cv = ds.df.reset_index(drop=True)
    gruppen_spalte = C.D._TEST_LEVEL_GROUPS.get(C.D.TEST_LEVEL)

    ergebnisse = {n: [] for n in MODELLE}
    quellen: dict[str, str] = {}
    for k, (tr, va) in enumerate(falten, 1):
        Xtr, Xva, _ = C.D.process_features(X_cv.iloc[tr], X_cv.iloc[va])
        yva = y_cv.iloc[va]
        n_gruppen = (
            int(df_cv.iloc[va][gruppen_spalte].nunique()) if gruppen_spalte else None
        )
        for name in MODELLE:
            m, quellen[name] = modell_fit(name, Xtr, y_cv.iloc[tr])
            vorher = m.predict(Xva)
            b, basis = modell_shap(name, m, Xva, yva)
            C.pruefe_additivitaet(b, basis, vorher, f"{name} Falte {k}")
            ergebnisse[name].append(
                {
                    "falte": k,
                    "w": C.globale_wichtigkeit(b, Xva.columns),
                    "n_val": int(len(va)),
                    "n_gruppen": n_gruppen,
                    "r2": float(r2_score(yva, vorher)),
                    "mae": float(mean_absolute_error(yva, vorher)),
                    "mae_mittel": float(
                        mean_absolute_error(
                            yva, np.full(len(yva), y_cv.iloc[tr].mean())
                        )
                    ),
                    "basis": basis,
                }
            )
            e = ergebnisse[name][-1]
            print(
                f"  Falte {k} {name:<9} n_val {e['n_val']:>4}  R² {e['r2']:+.3f}  "
                f"MAE {e['mae']:.2f}",
                flush=True,
            )
    return ergebnisse, quellen


def test_set_wichtigkeit():
    """Referenz: das eingefrorene Test-Set wie in 01_global bzw. 03_vergleich."""
    ds, X_tr, X_te = C.lade()
    y_te = np.asarray(ds.y_test)
    w = {}
    for name in MODELLE:
        m, _ = modell_fit(name, X_tr, ds.y_train)
        b, basis = modell_shap(name, m, X_te, y_te)
        C.pruefe_additivitaet(b, basis, m.predict(X_te), f"{name} Test-Set")
        w[name] = C.globale_wichtigkeit(b, X_te.columns)
    return w, ds, X_te


def matrix(ergebnisse, w_test) -> pd.DataFrame:
    """Merkmal x Falte, mittlerer |SHAP|."""
    spalten = {
        f"Falte {e['falte']}": e["w"].set_index("Merkmal").mittlerer_betrag
        for e in ergebnisse
    }
    M = pd.DataFrame(spalten).fillna(0.0)
    M["Mittel"] = M[[c for c in M.columns if c.startswith("Falte")]].mean(axis=1)
    M["Test-Set"] = (
        w_test.set_index("Merkmal").mittlerer_betrag.reindex(M.index).fillna(0.0)
    )
    return M.sort_values("Mittel", ascending=False)


def matrizen(ergebnisse, w_test) -> dict[str, pd.DataFrame]:
    """Je Modell eine Matrix, auf DENSELBEN Merkmalsindex gebracht."""
    roh = {n: matrix(ergebnisse[n], w_test[n]) for n in MODELLE}
    alle = sorted(set().union(*(M.index for M in roh.values())))
    return {
        n: M.reindex(alle).fillna(0.0).sort_values("Mittel", ascending=False)
        for n, M in roh.items()
    }


def falten_spalten(M) -> list[str]:
    return [c for c in M.columns if c.startswith("Falte")]


def spalten_je_modell(M) -> list[str]:
    return falten_spalten(M) + ["Mittel", "Test-Set"]


def top(M, spalte, n=TOP) -> list[str]:
    return list(M[spalte].sort_values(ascending=False).head(n).index)


def raenge(M, spalte) -> pd.Series:
    return M[spalte].rank(ascending=False, method="min")


def feld_landkarte(fig, ax, Ms, n_zeilen=N_LANDKARTE):
    """Rang-Landkarte: Zeilen Merkmale, Spalten Falten und Modelle."""
    modelle = list(Ms)
    spalten: list[tuple[str, str, str]] = []
    for n in modelle:
        spalten += [(n, c, c.replace("Falte ", "F")) for c in falten_spalten(Ms[n])]
        spalten += [(n, "Mittel", "⌀"), (n, "Test-Set", "Test")]

    R = {(n, c): raenge(Ms[n], c) for n, c, _ in spalten}
    mittelrang = sum(R[(n, "Mittel")] for n in modelle) / len(modelle)
    zeilen = list(mittelrang.sort_values(kind="mergesort").head(n_zeilen).index)

    A = np.array([[R[(n, c)][m] for n, c, _ in spalten] for m in zeilen], dtype=float)
    ist_falte = np.array([c.startswith("Falte") for _, c, _ in spalten])

    grund = matplotlib.colormaps["Purples"]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "rang", grund(np.linspace(0.85, 0.10, 256))
    )
    cmap.set_bad("#f0f0f0")
    bild = ax.imshow(
        np.ma.masked_greater(A, TOP), aspect="auto", cmap=cmap, vmin=1, vmax=TOP
    )

    for i in range(len(zeilen)):
        for j in range(len(spalten)):
            r = int(A[i, j])
            drin = r <= TOP
            ax.text(
                j,
                i,
                str(r) if drin else "··",
                ha="center",
                va="center",
                fontsize=6.4 if drin else 5.5,
                color="white" if drin and r <= 6 else ("black" if drin else "#b0b0b0"),
            )

    start = 0
    bloecke = []
    for n in modelle:
        k = sum(1 for m, _, _ in spalten if m == n)
        bloecke.append((start, start + k - 1, n))
        start += k
    for a, b, n in bloecke:
        ax.axvline(b - 1.5, color="#777777", lw=0.8, ls=":")
        if (a, b, n) != bloecke[-1]:
            ax.axvline(b + 0.5, color="black", lw=1.5)
        ax.text(
            (a + b) / 2,
            -1.45,
            n,
            ha="center",
            va="center",
            fontsize=10.5,
            weight="bold",
        )

    ax.set_xticks(range(len(spalten)))
    ax.set_xticklabels([k for _, _, k in spalten], fontsize=7.5)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(range(len(zeilen)))
    ax.set_yticklabels([m[:38] for m in zeilen], fontsize=7)
    for t, m in zip(ax.get_yticklabels(), zeilen):
        t.set_color(FARBE_HERKUNFT[C.herkunft(m)])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    n_f = int(ist_falte.sum())
    treffer = (A[:, ist_falte] <= TOP).sum(axis=1)
    x0, breite = len(spalten) - 0.3, 3.6
    for i, t in enumerate(treffer):
        ax.barh(
            i,
            breite * t / n_f,
            left=x0,
            height=0.6,
            color="#31a354" if t == n_f else "#bdbdbd",
        )
        ax.text(x0 + breite + 0.15, i, f"{t}/{n_f}", fontsize=6.2, va="center")
    ax.text(
        x0 + breite / 2,
        -1.45,
        f"in Falten-Top {TOP}",
        ha="center",
        va="center",
        fontsize=8.5,
    )
    ax.set_xlim(-0.5, x0 + breite + 1.9)
    ax.set_ylim(len(zeilen) - 0.5, -2.0)

    cb = fig.colorbar(bild, ax=ax, fraction=0.022, pad=0.015)
    cb.set_label("Rang in dieser Spalte", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    cb.ax.invert_yaxis()
    ax.legend(
        handles=[
            Patch(color=FARBE_HERKUNFT["Stammdaten"], label="Merkmalsname: Stammdaten"),
            Patch(color=FARBE_HERKUNFT["Prozess"], label="Merkmalsname: Prozess"),
            Patch(color="#f0f0f0", label=f"·· = schlechter als Rang {TOP}"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.035),
        ncol=3,
        fontsize=8,
        frameon=False,
    )
    return zeilen


def seite_uebersicht(pdf, Ms, fuss):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11.7, 8.3))
    fig.suptitle(
        f"Übersicht — wo steht welches Merkmal, in jeder Falte und in beiden "
        f"Modellen (Ebene {C.D.TEST_LEVEL})",
        fontsize=13,
        y=0.985,
    )
    ax = fig.add_axes((0.215, 0.10, 0.68, 0.79))
    zeilen = feld_landkarte(fig, ax, Ms)
    fig.text(0.5, 0.015, fuss, ha="center", fontsize=7.5, style="italic")
    pdf.savefig(fig)
    plt.close(fig)
    return zeilen


def feld_falte(ax, e, mittel_top):
    t = e["w"].head(TOP).iloc[::-1]
    ax.barh(
        range(len(t)),
        t.mittlerer_betrag,
        color=[FARBE_HERKUNFT[C.herkunft(m)] for m in t.Merkmal],
        alpha=0.85,
    )
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels([m[:26] for m in t.Merkmal], fontsize=5.4)
    for tick, m in zip(ax.get_yticklabels(), t.Merkmal):
        if m not in mittel_top:
            tick.set_color("#999999")
    gr = f", {e['n_gruppen']} Gr." if e["n_gruppen"] else ""
    ax.set_title(
        f"Falte {e['falte']} — n={e['n_val']}{gr}, R² {e['r2']:+.3f}, "
        f"MAE {e['mae']:.2f}",
        fontsize=8,
    )
    ax.set_xlabel("mittlerer |SHAP| [PP]", fontsize=6.5)
    ax.tick_params(axis="x", labelsize=6)
    ax.grid(alpha=0.3, axis="x")


def feld_guete(ax, ergebnisse):
    r2 = [e["r2"] for e in ergebnisse]
    x = np.arange(len(r2))
    ax.bar(x, r2, color=["#31a354" if v > 0 else "#d62728" for v in r2], alpha=0.85)
    ax.axhline(float(np.mean(r2)), color="black", ls="--", lw=1)
    ax.axhline(0, color="black", lw=0.8)
    for i, v in enumerate(r2):
        ax.annotate(
            f"{v:+.3f}",
            (i, v),
            xytext=(0, 3 if v >= 0 else -9),
            textcoords="offset points",
            ha="center",
            fontsize=6.5,
            bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none"),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"F{e['falte']}\nMAE {e['mae']:.1f}" for e in ergebnisse], fontsize=6.5
    )
    ax.set_ylabel("R² der Falte", fontsize=7)
    ax.tick_params(axis="y", labelsize=6)
    ax.set_title(
        f"Güte der Falten (gestrichelt = Mittel {np.mean(r2):+.3f})", fontsize=8
    )
    ax.grid(alpha=0.3, axis="y")


def seite_falten(pdf, name, M, ergebnisse, fuss):
    n = len(ergebnisse)
    ncols = 3
    nrows = int(np.ceil((n + 1) / ncols))
    fig, ax = C.seite(
        f"Top {TOP} je Falte — {name}, Ebene {C.D.TEST_LEVEL} "
        f"(graue Namen: nicht in den gemittelten Top {TOP})",
        nrows,
        ncols,
    )
    mittel_top = set(top(M, "Mittel"))
    for i, e in enumerate(ergebnisse):
        feld_falte(ax[i], e, mittel_top)
    feld_guete(ax[n], ergebnisse)
    for j in range(n + 1, len(ax)):
        ax[j].axis("off")
    C.schliesse(pdf, fig, fuss)


def feld_a(ax, M, name):
    fs = falten_spalten(M)
    t = M.head(TOP).iloc[::-1]
    F = t[fs]
    n_top = [sum(m in top(M, f) for f in fs) for m in t.index]
    y = np.arange(len(t))
    ax.barh(
        y, t.Mittel, color=[FARBE_HERKUNFT[C.herkunft(m)] for m in t.index], alpha=0.85
    )
    ax.errorbar(
        t.Mittel,
        y,
        xerr=[t.Mittel - F.min(axis=1), F.max(axis=1) - t.Mittel],
        fmt="none",
        ecolor="black",
        lw=0.8,
        capsize=2,
    )
    for yi, (v, k) in enumerate(zip(F.max(axis=1), n_top)):
        ax.annotate(
            f"{k}/{len(fs)}",
            (v, yi),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=6.5,
            color="#d62728" if k < len(fs) else "black",
        )
    ax.set_yticks(y)
    ax.set_yticklabels([m[:34] for m in t.index], fontsize=7)
    ax.set_xlabel("mittlerer |SHAP|, gemittelt über die Falten [PP]")
    ax.set_xlim(right=float(F.max(axis=1).max()) * 1.14)
    ax.set_title(
        f"A  Gemittelte Top {TOP}, {name}\n"
        f"Balken = Spanne der Falten, Zahl = Falten mit Rang ≤ {TOP}",
        fontsize=9,
    )
    ax.grid(alpha=0.3, axis="x")


def feld_b(ax, M, name, kuerze=26):
    fs = falten_spalten(M)
    t = M.head(TOP)
    R = {f: raenge(M, f) for f in fs}
    y = np.arange(len(t))[::-1]
    for i, f in enumerate(fs):
        ax.scatter(
            [R[f][m] for m in t.index],
            y + (i - (len(fs) - 1) / 2) * 0.12,
            s=14,
            color=FARBEN_FALTEN[i % len(FARBEN_FALTEN)],
            label=f,
            zorder=3,
        )
    ax.axvline(TOP + 0.5, color="#999999", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([m[:kuerze] for m in t.index], fontsize=6.5)
    ax.set_xlabel("Rang in der Falte (log; gepunktet = Grenze Top 20)", fontsize=8)
    ax.set_title(
        f"B  Rang der gemittelten Top {TOP} in jeder Falte, {name}", fontsize=9.5
    )
    ax.legend(fontsize=6.5, loc="lower right")
    ax.grid(alpha=0.3, axis="x")
    return {f: {m: int(R[f][m]) for m in t.index} for f in fs}


def heatmap(ax, W, titel, fmt, vmin, vmax, cmap, fontsize=7):
    ax.imshow(W.values, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(W)))
    ax.set_yticks(range(len(W)))
    ax.set_xticklabels(W.columns, rotation=45, ha="right", fontsize=fontsize + 0.5)
    ax.set_yticklabels(W.index, fontsize=fontsize + 0.5)
    for i in range(len(W)):
        for j in range(len(W)):
            v = W.values[i, j]
            ax.text(
                j,
                i,
                format(v, fmt),
                ha="center",
                va="center",
                fontsize=fontsize,
                color="white" if (v - vmin) / (vmax - vmin) > 0.6 else "black",
            )
    ax.set_title(titel, fontsize=9.5)


def rho_matrix(reihen: dict) -> pd.DataFrame:
    namen = list(reihen)
    return pd.DataFrame(
        [[spearmanr(reihen[a], reihen[b])[0] for b in namen] for a in namen],
        index=namen,
        columns=namen,
    )


def ueberlappung_matrix(tops: dict) -> pd.DataFrame:
    namen = list(tops)
    return pd.DataFrame(
        [[len(tops[a] & tops[b]) for b in namen] for a in namen],
        index=namen,
        columns=namen,
    )


def seite_a_d(pdf, Ms, Us, fuss):
    fig, ax = C.seite(
        f"Gemittelte Rangliste und ihre Schnittmenge — Ebene {C.D.TEST_LEVEL}", 2, 2
    )
    for i, n in enumerate(MODELLE):
        feld_a(ax[i], Ms[n], n)
        heatmap(
            ax[2 + i],
            Us[n],
            f"D  Gemeinsame Merkmale in den Top {TOP}, {n}",
            "d",
            0,
            TOP,
            "Greens",
        )
    C.schliesse(pdf, fig, fuss)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    feld_a(ax, Ms["XGBoost"], "XGBoost")
    fig.tight_layout()
    ziel = C.ROOT / "reports" / "figures" / "kapitel7" / f"abb_7_top20_falten_{C.D.TEST_LEVEL}.png"
    ziel.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ziel, dpi=200, bbox_inches="tight")
    plt.close(fig)


def seite_beleg(pdf, Ms, R_alle, fuss):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11.7, 8.3))
    fig.suptitle(
        f"Beleg: Rangstreuung über die Falten und über die Modelle — "
        f"Ebene {C.D.TEST_LEVEL}",
        fontsize=13,
        y=0.985,
    )
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.05],
        left=0.145,
        right=0.975,
        top=0.90,
        bottom=0.10,
        hspace=0.45,
        wspace=0.34,
    )
    rg = {}
    for i, n in enumerate(MODELLE):
        rg[n] = feld_b(fig.add_subplot(gs[0, i]), Ms[n], n)
    neben = R_alle.values[~np.eye(len(R_alle), dtype=bool)]
    vmin = float(np.floor(np.nanmin(neben) * 20) / 20)
    heatmap(
        fig.add_subplot(gs[1, :]),
        R_alle,
        f"C  Spearman ρ über ALLE Merkmale (Skala ab {vmin:.2f}) — Blöcke auf der "
        f"Diagonale = Faltenstreuung eines Modells, Block dazwischen = Modellstreuung",
        ".2f",
        vmin,
        1,
        "Blues",
        fontsize=5.2,
    )
    fig.text(0.5, 0.015, fuss, ha="center", fontsize=7.5, style="italic")
    pdf.savefig(fig)
    plt.close(fig)
    return rg


def seite_tabelle(pdf, name, M, ergebnisse, fuss):
    import matplotlib.pyplot as plt

    fs = falten_spalten(M)
    fig = plt.figure(figsize=(11.7, 8.3))
    fig.suptitle(
        f"Top {TOP} je Falte, {name} — fett = auch in den gemittelten Top {TOP}, "
        f"grau = nur in dieser Falte",
        fontsize=12,
        y=0.975,
    )
    ax = fig.add_axes((0.01, 0.05, 0.98, 0.86))
    ax.axis("off")
    mittel_top = set(top(M, "Mittel"))
    listen = [(f, top(M, f)) for f in fs] + [
        ("Mittel", top(M, "Mittel")),
        ("Test-Set", top(M, "Test-Set")),
    ]
    kopf = {f"Falte {e['falte']}": e for e in ergebnisse}
    breite = 1.0 / len(listen)
    for s, (spalte, namen) in enumerate(listen):
        x = s * breite
        ax.text(x, 1.0, spalte, fontsize=9.5, weight="bold", transform=ax.transAxes)
        if spalte in kopf:
            e = kopf[spalte]
            gr = f", {e['n_gruppen']} Gr." if e["n_gruppen"] else ""
            unter = f"n={e['n_val']}{gr}\nR² {e['r2']:+.3f}  MAE {e['mae']:.2f}"
        elif spalte == "Mittel":
            r2 = np.mean([e["r2"] for e in ergebnisse])
            mae = np.mean([e["mae"] for e in ergebnisse])
            unter = f"über {len(ergebnisse)} Falten\nR² {r2:+.3f}  MAE {mae:.2f}"
        else:
            unter = "eingefrorenes Test-Set\n(Referenz, wie 01_global)"
        ax.text(x, 0.955, unter, fontsize=6.8, va="top", transform=ax.transAxes)
        for i, m in enumerate(namen):
            gemeinsam = m in mittel_top
            ax.text(
                x,
                0.88 - i * 0.044,
                f"{i+1:>2}. {m[:27]}",
                fontsize=6.3,
                va="top",
                transform=ax.transAxes,
                weight="bold" if gemeinsam else "normal",
                color="black" if gemeinsam else "#999999",
            )
    fig.text(0.5, 0.012, fuss, ha="center", fontsize=7.5, style="italic")
    pdf.savefig(fig)
    plt.close(fig)


def kennzahlen(M) -> dict:
    """Faltenstreuung eines Modells: paarweise ueber die Falten."""
    fs = falten_spalten(M)
    R = rho_matrix({f: M[f] for f in fs})
    U = ueberlappung_matrix({f: set(top(M, f)) for f in fs})
    paare = [(a, b) for i, a in enumerate(fs) for b in fs[i + 1 :]]
    return {
        "rho_paare": float(np.mean([R.loc[a, b] for a, b in paare])),
        "ueb_paare": float(np.mean([U.loc[a, b] for a, b in paare])),
        "stabil": [m for m in top(M, "Mittel") if all(m in top(M, f) for f in fs)],
        "irgendwo": sorted(set().union(*(top(M, f) for f in fs))),
    }


def modellvergleich(Ms) -> dict:
    """Modellstreuung — dieselbe Falte, anderes Verfahren."""
    a, b = MODELLE
    fs = falten_spalten(Ms[a])
    je_falte = {f: float(spearmanr(Ms[a][f], Ms[b][f])[0]) for f in fs}
    ueb_falte = {f: int(len(set(top(Ms[a], f)) & set(top(Ms[b], f)))) for f in fs}
    return {
        "rho_mittel": float(spearmanr(Ms[a]["Mittel"], Ms[b]["Mittel"])[0]),
        "ueb_mittel": int(len(set(top(Ms[a], "Mittel")) & set(top(Ms[b], "Mittel")))),
        "rho_je_falte": je_falte,
        "rho_je_falte_mittel": float(np.mean(list(je_falte.values()))),
        "ueb_je_falte": ueb_falte,
        "ueb_je_falte_mittel": float(np.mean(list(ueb_falte.values()))),
        "gemeinsam_mittel_top20": sorted(
            set(top(Ms[a], "Mittel")) & set(top(Ms[b], "Mittel"))
        ),
    }


def main():
    print(f"Ebene: TEST_LEVEL={C.D.TEST_LEVEL}  →  CV-Ebene {cv_ebene()}")
    w_test, ds, X_te = test_set_wichtigkeit()
    falten, ebene = C.D.make_cv(ds.df, level=cv_ebene(), seed=CV_SEED)
    ergebnisse, quellen = falten_shap(ds, falten)
    Ms = matrizen(ergebnisse, w_test)
    fs = falten_spalten(Ms[MODELLE[0]])

    K = {n: kennzahlen(Ms[n]) for n in MODELLE}
    Us = {
        n: ueberlappung_matrix(
            {s: set(top(Ms[n], s)) for s in spalten_je_modell(Ms[n])}
        )
        for n in MODELLE
    }
    R_je_modell = {
        n: rho_matrix({s: Ms[n][s] for s in spalten_je_modell(Ms[n])}) for n in MODELLE
    }
    R_alle = rho_matrix(
        {
            f"{KURZ[n]} {s.replace('Falte ', 'F')}": Ms[n][s]
            for n in MODELLE
            for s in spalten_je_modell(Ms[n])
        }
    )
    V = modellvergleich(Ms)

    quelle_txt = " | ".join(f"{n} {quellen[n]}" for n in MODELLE)
    fuss = (
        f"{C.D.CSV_PATH.name} | {len(ds.df)} Billets, {len(fs)} Falten, "
        f"CV-Ebene {ebene} | y ∈ [{C.D.Y_MIN:g}; {C.D.Y_MAX:g}] | Test-Ebene "
        f"{C.D.TEST_LEVEL} (Test-Set {X_te.shape[0]}) | {quelle_txt}"
    )

    pdf = C.pdf_oeffnen(C.mit_ebene("07_kreuzvalidierung.pdf"))
    zeilen = seite_uebersicht(pdf, Ms, fuss)
    for n in MODELLE:
        seite_falten(pdf, n, Ms[n], ergebnisse[n], fuss)
    seite_a_d(pdf, Ms, Us, fuss)
    rg = seite_beleg(pdf, Ms, R_alle, fuss)
    for n in MODELLE:
        seite_tabelle(pdf, n, Ms[n], ergebnisse[n], fuss)

    x, c = MODELLE
    C.textseite(
        pdf,
        "Lesehinweise und Vorbehalte",
        [
            f"Vorgehen. {len(fs)} Falten nach data_prep.make_cv() auf Ebene "
            f"{ebene!r} über alle {len(ds.df)} gefilterten Billets — dieselben "
            f"Falten wie die CV in den Trainingsskripten. Je Falte wird der "
            f"Encoder EINMAL gefittet, darauf werden {x} ({quellen[x]}) und {c} "
            f"({quellen[c]}) angepasst und TreeSHAP auf der Validierungsfalte "
            f"gerechnet. Jedes Billet wird so von jedem Verfahren genau einmal "
            f"erklärt, und zwar von einem Modell, das es nicht gesehen hat.",
            "Mittelung. Das gemittelte Ergebnis ist das ungewichtete Mittel des "
            "mittleren |SHAP| über die Falten; jede Falte zählt gleich, auch wenn "
            "gruppierte Falten verschieden groß sind. Ein Merkmal, das der Encoder "
            "in einer Falte verworfen hat, trägt dort 0 bei.",
            f"Faltenstreuung. Paarweise zwischen den Falten: {x} ρ "
            f"{K[x]['rho_paare']:+.3f} und {K[x]['ueb_paare']:.1f} von {TOP} "
            f"gemeinsamen Merkmalen, {c} ρ {K[c]['rho_paare']:+.3f} und "
            f"{K[c]['ueb_paare']:.1f} von {TOP}. In ALLEN Falten unter den Top "
            f"{TOP}: {len(K[x]['stabil'])} Merkmale bei {x}, "
            f"{len(K[c]['stabil'])} bei {c}.",
            f"Modellstreuung. Auf derselben Falte gerechnet: ρ im Mittel "
            f"{V['rho_je_falte_mittel']:+.3f}, gemeinsame Top {TOP} im Mittel "
            f"{V['ueb_je_falte_mittel']:.1f}. Zwischen den gemittelten Ranglisten "
            f"ρ {V['rho_mittel']:+.3f} mit {V['ueb_mittel']} gemeinsamen Merkmalen. "
            f"Der Vergleich mit der Faltenstreuung eine Zeile höher ist die "
            f"eigentliche Aussage dieser PDF: liegt die Modellstreuung deutlich "
            f"unter der Faltenstreuung, ist die Rangliste eher eine Eigenschaft "
            f"des Verfahrens als der Ziehung.",
            f"Güte der Falten. {x}: R² zwischen "
            f"{min(e['r2'] for e in ergebnisse[x]):+.3f} und "
            f"{max(e['r2'] for e in ergebnisse[x]):+.3f}. {c}: zwischen "
            f"{min(e['r2'] for e in ergebnisse[c]):+.3f} und "
            f"{max(e['r2'] for e in ergebnisse[c]):+.3f}. Eine Falte mit schwacher "
            f"Güte liefert eine Erklärung eines schwachen Modells; weicht ihre "
            f"Rangliste ab, ist das kein Widerspruch zu den übrigen.",
            f"Vergleich mit dem Test-Set. ρ zwischen Mittel und Test-Set: {x} "
            f"{R_je_modell[x].loc['Mittel', 'Test-Set']:+.3f} "
            f"({Us[x].loc['Mittel', 'Test-Set']} gemeinsame Top {TOP}), {c} "
            f"{R_je_modell[c].loc['Mittel', 'Test-Set']:+.3f} "
            f"({Us[c].loc['Mittel', 'Test-Set']}). Das Test-Set-Modell ist auf mehr "
            f"Daten angepasst als jedes Faltenmodell; kleine Abweichungen sind "
            f"erwartbar.",
            "Vorbehalt. Stabilität über Falten und über Modelle zeigt, dass die "
            "Rangliste nicht an einer Ziehung und nicht an einem Baumaufbau hängt "
            "— nicht, dass die Merkmale den Ausschuss verursachen. Korrelierte "
            "Merkmale können sich zwischen Falten gegenseitig vertreten; eine "
            "instabile Einzelposition heißt deshalb nicht zwingend, dass die "
            "dahinterliegende Größe unwichtig ist.",
        ],
    )
    pdf.close()

    csv_pfade = {}
    for n in MODELLE:
        M = Ms[n]
        tab = M.reset_index().rename(columns={"index": "Merkmal"})
        tab["std_falten"] = M[fs].std(axis=1).values
        tab["falten_in_top20"] = [int(sum(m in top(M, f) for f in fs)) for m in M.index]
        csv_pfade[n] = C.speichere_rangliste(
            C.mit_ebene(f"07_kreuzvalidierung_{n.lower()}.csv"), tab, "Mittel"
        )

    p = C.speichere(
        C.mit_ebene("07_kreuzvalidierung.json"),
        {
            "test_level": C.D.TEST_LEVEL,
            "cv_level": ebene,
            "n_billets": int(len(ds.df)),
            "n_test": int(X_te.shape[0]),
            "n_falten": len(fs),
            "modelle": {
                n: {
                    "params_quelle": quellen[n],
                    "falten": [
                        {
                            "falte": e["falte"],
                            "n_val": e["n_val"],
                            "n_gruppen": e["n_gruppen"],
                            "r2": e["r2"],
                            "mae": e["mae"],
                            "mae_baseline_mittel": e["mae_mittel"],
                            "basiswert": e["basis"],
                            "top20": e["w"].head(TOP).to_dict("records"),
                        }
                        for e in ergebnisse[n]
                    ],
                    "mittel_top50": [
                        {
                            "Merkmal": m,
                            "mittlerer_betrag": float(r.Mittel),
                            "std_falten": float(r[fs].std()),
                            "min_falten": float(r[fs].min()),
                            "max_falten": float(r[fs].max()),
                            "falten_in_top20": int(
                                sum(m in top(Ms[n], f) for f in fs)
                            ),
                            "test_set": float(r["Test-Set"]),
                        }
                        for m, r in Ms[n].head(50).iterrows()
                    ],
                    "raenge_der_mittel_top20": rg[n],
                    "spearman": R_je_modell[n].round(4).to_dict(),
                    "top20_ueberlappung": Us[n].to_dict(),
                    "spearman_falten_paarweise_mittel": K[n]["rho_paare"],
                    "top20_ueberlappung_falten_paarweise_mittel": K[n]["ueb_paare"],
                    "stabil_in_allen_falten": K[n]["stabil"],
                    "n_merkmale_in_irgendeiner_top20": len(K[n]["irgendwo"]),
                    "rangliste_csv": csv_pfade[n].name,
                }
                for n in MODELLE
            },
            "modellvergleich": V,
            "spearman_alle_spalten": R_alle.round(4).to_dict(),
            "uebersicht_zeilen": zeilen,
        },
    )
    print(
        f"\ngeschrieben: {C.OUT_DIR / C.mit_ebene('07_kreuzvalidierung.pdf')}"
        f"\n             {p}"
        + "".join(f"\n             {q}" for q in csv_pfade.values())
        + "\n"
    )
    for n in MODELLE:
        print(
            f"{n}: Falten paarweise ρ {K[n]['rho_paare']:+.3f}, "
            f"Top-{TOP}-Überlappung {K[n]['ueb_paare']:.1f}; "
            f"stabil in allen Falten: {len(K[n]['stabil'])}/{TOP}"
        )
    print(
        f"Modellvergleich: ρ je Falte {V['rho_je_falte_mittel']:+.3f}, "
        f"ρ der Mittel {V['rho_mittel']:+.3f}, "
        f"gemeinsame Top {TOP} der Mittel {V['ueb_mittel']}"
    )


if __name__ == "__main__":
    main()
