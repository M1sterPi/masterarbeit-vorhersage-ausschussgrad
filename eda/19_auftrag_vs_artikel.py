"""19 — Artikel oder Auftrag.

In der Arbeit: Tabelle 3 und Abschnitt 4.2.5 (Zerlegung auf drei Ebenen),
Tabelle 13 und Abschnitt 7.1 (Zerlegung der Güte, auch ohne abgebrochene
Schmelzen); TabPFN-Vorhersagen für xai/16. Abbildung 15 zeichnet xai/23.

Varianzanteile, Prognosen ohne Merkmale und Zerlegung des R² der Modelle
(XGBoost, TabPFN) nach Ebenen. Die TabPFN-Vorhersagen werden in
runs/eda/19_oof_tabpfn.npz zwischengespeichert.
Ausgabe: reports/eda/19_auftrag_vs_artikel.pdf, runs/eda/19_auftrag_vs_artikel.json.

    python eda/19_auftrag_vs_artikel.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold, KFold

import _common as C
from data_prep import SP  # noqa: E402

ART, AUF, BIL = "#2c5f8a", "#d95f02", "#9e9e9e"
GLOBAL, MODELL, TABPFN = "#c7c7c7", "#333333", "#7b3294"
N_PERM = 200
N_BOOT = 300
SCHEMATA = ["zufällig", "auftragsrein", "artikelrein"]
MIT_TABPFN = True
TABPFN_CACHE = C.ROOT / "runs" / "eda" / "19_oof_tabpfn.npz"


def gm(v, g) -> np.ndarray:
    return pd.Series(np.asarray(v, float)).groupby(np.asarray(g)).transform("mean").to_numpy()


def r2_von(y, p) -> float:
    ok = ~np.isnan(p)
    return C.r2(y[ok], p[ok])


def anteile(y, art, auf) -> tuple[float, float, float]:
    """Varianzanteile Artikel / Auftrag im Artikel / Billet im Auftrag (Summe 1)."""
    sst = ((y - y.mean()) ** 2).sum()
    m_art, m_auf = gm(y, art), gm(y, auf)
    return (
        ((m_art - y.mean()) ** 2).sum() / sst,
        ((m_auf - m_art) ** 2).sum() / sst,
        ((y - m_auf) ** 2).sum() / sst,
    )


def null_anteile(y, art, auf, rng) -> tuple[np.ndarray, np.ndarray]:
    """Nullverteilungen für die zwei Stufen, jeweils mit passender Permutation."""
    idx_je_art = [np.flatnonzero(art == a) for a in np.unique(art)]
    n_art, n_auf = np.empty(N_PERM), np.empty(N_PERM)
    for i in range(N_PERM):
        n_art[i] = anteile(rng.permutation(y), art, auf)[0]
        yp = y.copy()
        for ix in idx_je_art:
            yp[ix] = y[rng.permutation(ix)]
        n_auf[i] = anteile(yp, art, auf)[1]
    return n_art, n_auf


def artikel_bootstrap(art, statistik, n=N_BOOT, seed=C.SEED):
    """Cluster-Bootstrap über Artikel: statistik(idx) -> Zahl oder Vektor."""
    rng = np.random.default_rng(seed)
    arts = np.unique(art)
    idx_je = {a: np.flatnonzero(art == a) for a in arts}
    werte = []
    for _ in range(n):
        wahl = rng.choice(arts, len(arts), replace=True)
        werte.append(statistik(wahl, idx_je))
    return np.nanpercentile(np.array(werte, dtype=float), [2.5, 97.5], axis=0)


def zerlege(y, p, art, auf) -> dict:
    """Orthogonale R²-Zerlegung: R² = Σ_k s_k · R²_k (siehe Textseite)."""
    e = y - p
    teile_y = [gm(y, art) - y.mean(), gm(y, auf) - gm(y, art), y - gm(y, auf)]
    teile_e = [gm(e, art), gm(e, auf) - gm(e, art), e - gm(e, auf)]
    sst = ((y - y.mean()) ** 2).sum()
    out = {"gesamt": C.r2(y, p)}
    for name, ty, te in zip(["artikel", "auftrag", "billet"], teile_y, teile_e):
        s = (ty**2).sum() / sst
        r2k = 1 - (te**2).sum() / (ty**2).sum()
        out[name] = (s, r2k, s * r2k)
    return out


def zerlege_boot(y, p, art, auf):
    def stat(wahl, idx_je):
        idx, a_neu, f_neu = [], [], []
        for k, a in enumerate(wahl):
            i = idx_je[a]
            idx.append(i)
            a_neu.append(np.full(len(i), k))
            f_neu.append(np.char.add(f"{k}_", auf[i]))
        idx = np.concatenate(idx)
        z = zerlege(y[idx], p[idx], np.concatenate(a_neu), np.concatenate(f_neu))
        return [z["artikel"][2], z["auftrag"][2], z["billet"][2], z["gesamt"]]

    return artikel_bootstrap(art, stat)


def oof_gruppen(y, art, auf, folds, ebene: str, lam: float = 0.0) -> np.ndarray:
    """OOF-Gruppenmittel ohne jedes Merkmal."""
    ys, ga, gf = pd.Series(y), pd.Series(art), pd.Series(auf)
    pred = np.full(len(y), np.nan)
    for tr, va in folds:
        glob = ys.iloc[tr].mean()
        p = np.full(len(va), glob)
        if ebene in ("art", "auf"):
            m_art = ys.iloc[tr].groupby(ga.iloc[tr].to_numpy()).mean()
            p = ga.iloc[va].map(m_art).fillna(glob).to_numpy()
        if ebene == "auf":
            t = pd.DataFrame({"y": ys.iloc[tr].to_numpy(), "f": gf.iloc[tr].to_numpy()})
            t["d"] = t.y - ga.iloc[tr].map(m_art).to_numpy()
            s = t.groupby("f").d.agg(["mean", "size"])
            d = gf.iloc[va].map(s["mean"]).fillna(0).to_numpy()
            n = gf.iloc[va].map(s["size"]).fillna(0).to_numpy()
            p = p + n / (n + lam) * d if lam > 0 else p + d
        pred[va] = p
    return pred


def erklaerseite(pdf, fuss: str = ""):
    """Schaubild: wie die R²-Zerlegung rechnet, an einem Spielbeispiel."""
    import matplotlib.pyplot as plt

    art = np.repeat(["A", "B"], 6)
    auf = np.repeat(["A1", "A2", "B1", "B2"], 3)
    y = np.array([15, 25, 20, 33, 23, 28, 35, 47, 41, 30, 40, 35], dtype=float)
    p = np.repeat([25.0, 37.0], 6)
    e = y - p
    z = zerlege(y, p, art, auf)

    teile_y = [gm(y, art) - y.mean(), gm(y, auf) - gm(y, art), y - gm(y, auf)]
    teile_e = [gm(e, art), gm(e, auf) - gm(e, art), e - gm(e, auf)]

    x = np.array([0, 1, 2, 3.6, 4.6, 5.6, 8.2, 9.2, 10.2, 11.8, 12.8, 13.8])

    fig = plt.figure(figsize=(11.7, 8.3))
    fig.suptitle("Wie die R²-Zerlegung rechnet — ein Spielbeispiel", fontsize=13, y=0.985)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1], width_ratios=[1, 1, 1])

    a = fig.add_subplot(gs[0, :2])
    for g in ["A", "B"]:
        k = art == g
        a.hlines(y[k].mean(), x[k].min() - 0.4, x[k].max() + 0.4, color=ART, lw=2.5, zorder=1)
    for g in np.unique(auf):
        k = auf == g
        a.hlines(y[k].mean(), x[k].min() - 0.3, x[k].max() + 0.3, color=AUF, lw=1.6, ls="--", zorder=2)
        a.text(x[k].mean(), 12.5, f"Auftrag {g}", ha="center", fontsize=8, color="0.3")
    a.axhline(y.mean(), color=BIL, lw=1, ls=":", zorder=0)
    a.text(x.max() + 0.6, y.mean(), "Gesamtmittel ȳ", va="center", fontsize=8, color="0.3")
    a.vlines(x, np.minimum(y, p), np.maximum(y, p), color="crimson", lw=1, alpha=0.6, zorder=2)
    a.scatter(x, y, s=40, color="white", edgecolor="k", zorder=4, label="y (gemessen)")
    a.scatter(x, p, s=45, marker="x", color="crimson", zorder=5, label="Vorhersage p")
    for g, xm in [("A", x[:6].mean()), ("B", x[6:].mean())]:
        a.text(xm, 9.5, f"Artikel {g}", ha="center", fontsize=10, weight="bold", color=ART)
    a.plot([], [], color=ART, lw=2.5, label="Artikelmittel")
    a.plot([], [], color=AUF, lw=1.6, ls="--", label="Auftragsmittel")
    a.plot([], [], color="crimson", lw=1, label="Fehler e = y − p")
    a.set_ylim(8, 56)
    a.set_xlim(-0.8, x.max() + 3.2)
    a.set_xticks([])
    a.spines["bottom"].set_visible(False)
    a.set_ylabel("y [%]")
    a.legend(fontsize=8, loc="upper left", ncol=3)
    a.set_title("12 Billets, 4 Aufträge, 2 Artikel; das Modell kennt nur den Artikel", fontsize=10)

    t = fig.add_subplot(gs[0, 2])
    t.axis("off")
    schritte = [
        ("1  Fehler je Billet", r"$e = y - p$"),
        ("2  y und e in drei Teile zerlegen", ""),
        ("   Artikel:", r"$\bar{y}_{Art} - \bar{y}$     bzw.  $\bar{e}_{Art}$"),
        ("   Auftrag im Artikel:", r"$\bar{y}_{Auf} - \bar{y}_{Art}$  bzw.  $\bar{e}_{Auf} - \bar{e}_{Art}$"),
        ("   Billet im Auftrag:", r"$y - \bar{y}_{Auf}$      bzw.  $e - \bar{e}_{Auf}$"),
        ("3  Größe der Ebene", r"$s_k = \Sigma (y\mathrm{-Teil}_k)^2 \,/\, \Sigma (y - \bar{y})^2$"),
        ("4  Treffer auf der Ebene", r"$R^2_k = 1 - \Sigma (e\mathrm{-Teil}_k)^2 \,/\, \Sigma (y\mathrm{-Teil}_k)^2$"),
        ("5  Beitrag", r"$c_k = s_k \cdot R^2_k$"),
        ("", r"$R^2 = c_{Art} + c_{Auf} + c_{Bil}$   (exakt)"),
    ]
    yy = 0.98
    for kopf, formel in schritte:
        if kopf.strip():
            t.text(0.0, yy, kopf, fontsize=9, weight="bold" if not kopf.startswith("   ") else "normal",
                   va="top", transform=t.transAxes)
            yy -= 0.052
        if formel:
            t.text(0.06, yy, formel, fontsize=9.5, va="top", transform=t.transAxes)
            yy -= 0.068
    t.text(
        0.0, yy - 0.01,
        "Die Teile stehen senkrecht aufeinander, ihre\n"
        "Quadratsummen addieren sich ohne Rest.",
        fontsize=8, color="0.3", va="top", transform=t.transAxes,
    )
    ya, yf, yb = (tt[0] for tt in teile_y)
    ea, ef, eb = (tt[0] for tt in teile_e)
    t.text(
        0.0, yy - 0.12,
        f"Erstes Billet: y = {y[0]:.0f}, p = {p[0]:.0f}, ȳ = {y.mean():.0f}\n"
        f"  y − ȳ = {ya:+.0f} (Artikel) {yf:+.0f} (Auftrag) {yb:+.0f} (Billet) = {y[0] - y.mean():+.0f}\n"
        f"  e       = {ea:+.0f} (Artikel) {ef:+.0f} (Auftrag) {eb:+.0f} (Billet) = {e[0]:+.0f}\n"
        "Die nächste Seite rechnet dasselbe für XGBoost\nauf allen Billets.",
        fontsize=8, va="top", transform=t.transAxes,
    )

    ebenen = [("artikel", "Artikel", ART), ("auftrag", "Auftrag im Artikel", AUF), ("billet", "Billet im Auftrag", BIL)]
    lim = max(np.abs(np.r_[teile_y + teile_e]).max(), 1) * 1.15
    for j, ((key, name, f), ty, te) in enumerate(zip(ebenen, teile_y, teile_e)):
        b = fig.add_subplot(gs[1, j])
        b.bar(x - 0.2, ty, width=0.38, color=f, label="y-Teil (was es zu erklären gibt)")
        b.bar(x + 0.2, te, width=0.38, color="white", edgecolor="crimson", hatch="////", lw=0.8,
              label="Fehler-Teil (was übrig bleibt)")
        b.axhline(0, color="k", lw=0.8)
        b.set_ylim(-lim, lim)
        b.set_xticks([])
        b.spines["bottom"].set_visible(False)
        s, r2k, c = z[key]
        b.set_title(f"{name}\ns = {s:.2f},  R²_k = {r2k:+.2f},  Beitrag = {c:+.2f}", fontsize=9.5)
        if j == 0:
            b.set_ylabel("Abweichung [Pp]")
            b.legend(fontsize=7, loc="lower right")
    fig.text(
        0.5, 0.015,
        f"Summe der Beiträge: {z['artikel'][2]:+.2f} {z['auftrag'][2]:+.2f} {z['billet'][2]:+.2f} "
        f"= R² {z['gesamt']:+.2f}.  Lesart: Wo die schraffierten Balken so hoch sind wie die farbigen, "
        "hat das Modell auf dieser Ebene nichts getroffen (R²_k ≈ 0).",
        ha="center", fontsize=9,
    )
    C.aufraeumen(fig)
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    pdf.savefig(fig)
    plt.close(fig)


def tabpfn_parameter() -> dict:
    """DEFAULT_PARAMS/FIXED_PARAMS aus tabpfn_regression.py, per AST gelesen."""
    import ast

    quelle = (C.ROOT / "data_pipeline" / "tabpfn_regression.py").read_text(encoding="utf-8")
    gefunden = {}
    for k in ast.parse(quelle).body:
        if (
            isinstance(k, ast.Assign)
            and len(k.targets) == 1
            and getattr(k.targets[0], "id", "") in ("DEFAULT_PARAMS", "FIXED_PARAMS")
            and isinstance(k.value, ast.Call)
        ):
            gefunden[k.targets[0].id] = {kw.arg: ast.literal_eval(kw.value) for kw in k.value.keywords}
    if set(gefunden) != {"DEFAULT_PARAMS", "FIXED_PARAMS"}:
        raise RuntimeError("DEFAULT_PARAMS/FIXED_PARAMS in tabpfn_regression.py nicht gefunden")
    return {**gefunden["DEFAULT_PARAMS"], **gefunden["FIXED_PARAMS"]}


def oof_tabpfn(X, ys, folds, schema: str) -> np.ndarray:
    """TabPFN out-of-fold, Merkmalsverarbeitung je Falte nur auf Train."""
    import hashlib
    import time

    from tabpfn import TabPFNRegressor

    D = C.D
    params = tabpfn_parameter()
    cache = dict(np.load(TABPFN_CACHE)) if TABPFN_CACHE.exists() else {}
    pred = np.full(len(ys), np.nan)
    for i, (tr, va) in enumerate(folds):
        h = hashlib.sha1(ys.to_numpy(float).tobytes() + np.asarray(va).tobytes()).hexdigest()[:12]
        key = f"{schema}_{i}_{h}"
        if key in cache:
            pred[va] = cache[key]
            continue
        t0 = time.time()
        Xtr, Xva, _ = D.process_features(X.iloc[tr], X.iloc[va])
        m = TabPFNRegressor(**params)
        m.fit(Xtr, ys.iloc[tr])
        pred[va] = m.predict(Xva)
        cache[key] = pred[va]
        TABPFN_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(TABPFN_CACHE, **cache)
        print(f"  TabPFN {schema} Falte {i + 1}/{len(folds)}: {time.time() - t0:.0f} s", flush=True)
    return pred


def steigung(x, yv, w) -> float:
    """Gewichtete OLS-Steigung von yv auf x (Gewicht = Billets)."""
    xm, ym = np.average(x, weights=w), np.average(yv, weights=w)
    return float(np.sum(w * (x - xm) * (yv - ym)) / np.sum(w * (x - xm) ** 2))


def main():
    ds = C.lade(alle_arten=True)
    X, ys, df = C.daten_teil(ds)
    fuss = C.kopfzeile(ds, alle_arten=True)
    ts = ds.melt_end_ts[C.eda_maske(ds)].reset_index(drop=True)

    art = df[SP["artikel"]].astype(str).to_numpy(dtype=str)
    auf = df[SP["auftrag"]].astype(str).to_numpy(dtype=str)
    y = ys.to_numpy(float)
    N = len(y)
    rng = np.random.default_rng(C.SEED)

    n_auf_je_art = pd.Series(auf).groupby(art).nunique()
    n_bil_je_auf = pd.Series(auf).value_counts()
    n_auf = pd.Series(auf).map(n_bil_je_auf).to_numpy()
    n_art = pd.Series(art).map(pd.Series(art).value_counts()).to_numpy()
    art_mehr = pd.Series(art).map(n_auf_je_art).to_numpy() >= 2

    v_art, v_auf, v_bil = anteile(y, art, auf)
    null_art, null_auf = null_anteile(y, art, auf, rng)
    innen_auf = v_auf / (v_auf + v_bil)
    innen_null = null_auf.mean() / (1 - v_art)

    A = (
        pd.DataFrame({"y": y, "art": art, "auf": auf, "tag": (ts - ts.min()).dt.total_seconds() / 86400})
        .groupby("auf")
        .agg(art=("art", "first"), m=("y", "mean"), n=("y", "size"), tag=("tag", "min"))
    )
    art_summe = pd.Series(y).groupby(art).sum()
    A["art_andere"] = (A.art.map(art_summe) - A.m * A.n) / (A.art.map(pd.Series(art).value_counts()) - A.n)
    A = A.replace([np.inf, -np.inf], np.nan)
    Am = A.dropna(subset=["art_andere"])
    rho_auf_art, _ = spearmanr(Am.m, Am.art_andere)
    r2_auf_art = C.r2(Am.m, Am.art_andere)

    s_auf = pd.Series(y).groupby(auf).transform("sum").to_numpy()
    s_art = pd.Series(y).groupby(art).transform("sum").to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        p_global = (y.sum() - y) / (N - 1)
        p_auf = np.where(n_auf >= 2, (s_auf - y) / (n_auf - 1), np.nan)
        p_art_andere = np.where(n_art > n_auf, (s_art - s_auf) / (n_art - n_auf), np.nan)

    k_auf, k_art = len(n_bil_je_auf), len(n_auf_je_art)
    m_art_v, m_auf_v = gm(y, art), gm(y, auf)
    sig_w = ((y - m_auf_v) ** 2).sum() / (N - k_auf)
    n0 = (N - (n_bil_je_auf**2).sum() / N) / (k_auf - 1)
    sig_a = max((((m_auf_v - m_art_v) ** 2).sum() / (k_auf - k_art) - sig_w) / n0, 1e-6)
    lam = sig_w / sig_a
    m_sib = n_auf - 1
    w_sib = m_sib / (m_sib + lam)
    p_schrumpf = np.where(
        np.isnan(p_auf), p_art_andere, w_sib * np.nan_to_num(p_auf) + (1 - w_sib) * p_art_andere
    )

    S = ~np.isnan(p_auf) & ~np.isnan(p_art_andere)
    prognosen = [
        ("Gesamtmittel", p_global, GLOBAL),
        ("Artikelmittel\n(andere\nAufträge)", p_art_andere, ART),
        ("Geschwister\nim Auftrag", p_auf, AUF),
        ("Geschwister,\nzum Artikel\ngeschrumpft", p_schrumpf, "#8c510a"),
    ]
    yS = y[S]

    def prog_stat(wahl, idx_je):
        idx = np.concatenate([idx_je[a] for a in wahl])
        idx = idx[S[idx]]
        return [C.r2(y[idx], p[idx]) for _, p, _ in prognosen]

    prog_r2 = [C.r2(yS, p[S]) for _, p, _ in prognosen]
    prog_mae = [C.mae(yS, p[S]) for _, p, _ in prognosen]
    prog_ci = artikel_bootstrap(art, prog_stat)

    klassen = [(1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 5, "4–5"), (6, 9, "6–9"), (10, 99, "≥ 10")]
    geschw = []
    for lo, hi, name in klassen:
        k = S & (m_sib >= lo) & (m_sib <= hi)
        if k.sum() < 20:
            continue
        geschw.append(
            {
                "name": name,
                "n": int(k.sum()),
                **{lab: C.mae(y[k], p[k]) for lab, p, _ in prognosen[1:]},
            }
        )
    G = pd.DataFrame(geschw)

    faltungen = {
        "zufällig": list(KFold(5, shuffle=True, random_state=C.SEED).split(X)),
        "auftragsrein": list(GroupKFold(5, shuffle=True, random_state=C.SEED).split(X, groups=auf)),
        "artikelrein": list(GroupKFold(5, shuffle=True, random_state=C.SEED).split(X, groups=art)),
    }
    abbr = (C.als_zahl(X[SP["phase"] + "_max"]) < SP["letzte_phase"]).to_numpy()
    ok = ~abbr

    M = {}
    for sch, folds in faltungen.items():
        print(f"XGBoost-OOF {sch} …", flush=True)
        p_xgb = C.oof_xgb(X, ys, folds)
        vorher = {
            "Gesamtmittel": oof_gruppen(y, art, auf, folds, "global"),
            "Artikelmittel": oof_gruppen(y, art, auf, folds, "art"),
            "Auftragsmittel,\ngeschrumpft": oof_gruppen(y, art, auf, folds, "auf", lam),
            "XGBoost": p_xgb,
        }
        modelle = {"XGBoost": p_xgb}
        if MIT_TABPFN:
            print(f"TabPFN-OOF {sch} …", flush=True)
            modelle["TabPFN"] = vorher["TabPFN"] = oof_tabpfn(X, ys, folds, sch)
        je_modell = {
            name: {
                "zerlegung": zerlege(y, p, art, auf),
                "ci": zerlege_boot(y, p, art, auf),
                "ohne_abbr": zerlege(y[ok], p[ok], art[ok], auf[ok]),
            }
            for name, p in modelle.items()
        }
        M[sch] = {
            "p": p_xgb,
            "vergleich": {k: (C.r2(y, v), C.mae(y, v)) for k, v in vorher.items()},
            "zerlegung": je_modell["XGBoost"]["zerlegung"],
            "ci": je_modell["XGBoost"]["ci"],
            "modelle": je_modell,
        }

    for sch in SCHEMATA:
        p = M[sch]["p"]
        B = pd.DataFrame({"y": y, "p": p, "art": art, "auf": auf})
        Ar = B.groupby("art").agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
        Au = B.groupby("auf").agg(art=("art", "first"), y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
        Au = Au[Au.art.map(n_auf_je_art) >= 2]
        Au["dy"] = Au.y - Au.art.map(Ar.y)
        Au["dp"] = Au.p - Au.art.map(Ar.p)
        M[sch]["Ar"], M[sch]["Au"] = Ar, Au
        M[sch]["b_art"] = steigung(Ar.y.to_numpy(), Ar.p.to_numpy(), Ar.n.to_numpy())
        M[sch]["b_auf"] = steigung(Au.dy.to_numpy(), Au.dp.to_numpy(), Au.n.to_numpy())

    Am = Am.assign(dy=Am.m - Am.art.map(pd.Series(y).groupby(art).mean()))
    ts_ok = Am.tag.notna()
    Am["dts"] = Am.tag - Am.groupby("art").tag.transform("mean")
    rho_zeit, p_zeit = spearmanr(Am.dts[ts_ok], Am.dy[ts_ok])

    spalten = C.numerische_merkmale(X)
    Xn = pd.DataFrame({c: C.als_zahl(X[c]) for c in spalten})
    Xa = Xn.groupby(auf).mean().loc[Am.index]
    Xd = Xa - Xa.groupby(Am.art.to_numpy()).transform("mean")
    Xd = Xd.loc[:, (Xd.abs() > 1e-9).mean() > 0.2]
    T = C.spearman_tabelle(Xd.reset_index(drop=True), Am.dy.reset_index(drop=True), list(Xd.columns))
    n_sig = int((T.q < 0.05).sum())

    def _zerl(z):
        return {
            "gesamt": z["gesamt"],
            **{
                k: {"anteil": z[k][0], "r2_ebene": z[k][1], "beitrag": z[k][2]}
                for k in ("artikel", "auftrag", "billet")
            },
        }

    ergebnis = {
        "n": N,
        "n_artikel": k_art,
        "n_auftraege": k_auf,
        "varianzanteile": {
            "artikel": v_art,
            "auftrag": v_auf,
            "billet": v_bil,
            "null_artikel": float(null_art.mean()),
            "null_auftrag": float(null_auf.mean()),
            "innen_auftrag": innen_auf,
            "innen_null": float(innen_null),
        },
        "auftrag_vs_andere": {"n": len(Am), "rho": float(rho_auf_art), "r2": r2_auf_art},
        "schrumpf_lambda": lam,
        "n_abbrueche": int(abbr.sum()),
        "prognosen_ohne_merkmale": {
            lab.replace("\n", " "): {"r2": r, "mae": m, "ki": [prog_ci[0, i], prog_ci[1, i]]}
            for i, ((lab, _, _), r, m) in enumerate(zip(prognosen, prog_r2, prog_mae))
        },
        "schemata": {
            sch: {
                "steigung_artikel": M[sch]["b_art"],
                "steigung_auftrag": M[sch]["b_auf"],
                "gruppenmittel": {
                    k.replace("\n", " "): {"r2": r, "mae": m}
                    for k, (r, m) in M[sch]["vergleich"].items()
                },
                "modelle": {
                    name: {
                        "zerlegung": _zerl(v["zerlegung"]),
                        "ki_beitrag": {
                            k: [v["ci"][0, j], v["ci"][1, j]]
                            for j, k in enumerate(("artikel", "auftrag", "billet", "gesamt"))
                        },
                        "ohne_abbrueche": _zerl(v["ohne_abbr"]),
                    }
                    for name, v in M[sch]["modelle"].items()
                },
            }
            for sch in SCHEMATA
        },
        "zeit_rho": float(rho_zeit),
        "zeit_p": float(p_zeit),
        "n_merkmale_sig": n_sig,
    }
    C.speichere_json("19_auftrag_vs_artikel", ergebnis)

    pdf = C.pdf_oeffnen("19_auftrag_vs_artikel.pdf")
    modellnamen = list(M["zufällig"]["modelle"])
    n_abbr = int(abbr.sum())

    def zeile_zerlegung(sch, modell="XGBoost"):
        z, ci = M[sch]["modelle"][modell]["zerlegung"], M[sch]["modelle"][modell]["ci"]
        return (
            f"  {modell}, {sch}: R² {z['gesamt']:+.3f} = Artikel {z['artikel'][2]:+.3f} "
            f"[{ci[0, 0]:+.3f}; {ci[1, 0]:+.3f}] + Auftrag im Artikel {z['auftrag'][2]:+.3f} "
            f"[{ci[0, 1]:+.3f}; {ci[1, 1]:+.3f}] + Billet {z['billet'][2]:+.3f} "
            f"[{ci[0, 2]:+.3f}; {ci[1, 2]:+.3f}]; getroffen je Ebene R²_k "
            f"{z['artikel'][1]:+.2f} / {z['auftrag'][1]:+.2f} / {z['billet'][1]:+.2f}"
        )

    def zeile_abbr(sch, modell):
        z0 = M[sch]["modelle"][modell]["zerlegung"]
        z1 = M[sch]["modelle"][modell]["ohne_abbr"]
        return (
            f"  {modell}, {sch}: Billetbeitrag {z0['billet'][2]:+.3f} → {z1['billet'][2]:+.3f} "
            f"ohne Abbrüche (Artikel {z0['artikel'][2]:+.3f} → {z1['artikel'][2]:+.3f}, "
            f"Auftrag {z0['auftrag'][2]:+.3f} → {z1['auftrag'][2]:+.3f}, "
            f"R² {z0['gesamt']:+.3f} → {z1['gesamt']:+.3f})"
        )

    def zeile_vergleich(sch):
        v = M[sch]["vergleich"]
        return f"  {sch}: " + ", ".join(
            f"{k.replace(chr(10), ' ')} R² {r:+.3f} / MAE {m:.2f}" for k, (r, m) in v.items()
        )

    C.textseite(
        pdf,
        "19 — Auftrag oder Artikel?",
        [
            f"Datengrundlage: {C.umfang_text(N)}, {k_art} Artikel, {k_auf} Aufträge. Jeder "
            "Auftrag gehört zu genau einem Artikel. Die Frage lautet deshalb nicht 'Auftrag "
            "ODER Artikel', sondern: was trägt der Auftrag ÜBER den Artikel hinaus? "
            f"Median {n_auf_je_art.median():.0f} Aufträge je Artikel, {(n_auf_je_art >= 2).sum()} "
            f"Artikel haben mindestens zwei; median {n_bil_je_auf.median():.0f} Billets je "
            f"Auftrag, {(n_bil_je_auf == 1).sum()} Aufträge bestehen aus einem einzigen Billet.",
            "Frage 1 — Wie groß sind die Ebenen? Varianzanteile (Summe 1): Artikel "
            f"{v_art:.3f} (Null {null_art.mean():.3f}), Auftrag im Artikel {v_auf:.3f} "
            f"(Null {null_auf.mean():.3f} ± {null_auf.std():.3f}, y nur innerhalb der Artikel "
            f"gemischt), Billet im Auftrag {v_bil:.3f}. Der Überschuss beträgt beim Artikel "
            f"{v_art - null_art.mean():+.3f}, beim Auftrag {v_auf - null_auf.mean():+.3f}. "
            f"Innerhalb eines Artikels liegen {100 * innen_auf:.0f} % der Streuung zwischen "
            f"den Aufträgen (Null ≈ {100 * innen_null:.0f} %).",
            "Frage 2 — Sagt der Artikel den Auftrag vorher? Auftragsmittel gegen das Mittel "
            f"der ANDEREN Aufträge desselben Artikels ({len(Am)} Aufträge): Spearman ρ "
            f"{rho_auf_art:+.3f}, R² {r2_auf_art:+.3f}. Aufträge eines Artikels ähneln "
            "sich also, sind aber nicht austauschbar.",
            "Frage 3 — Was ist ohne Merkmale erreichbar? Leave-out-Prognose je Billet, "
            f"verglichen auf den {S.sum()} Billets, für die beides definiert ist (mind. ein "
            "Geschwister im Auftrag UND ein anderer Auftrag im Artikel). KI: Cluster-"
            "Bootstrap über Artikel.",
            *[
                f"  {lab.replace(chr(10), ' ')}: R² {r:+.3f} [{prog_ci[0, i]:+.3f}; "
                f"{prog_ci[1, i]:+.3f}], MAE {m:.2f}"
                for i, ((lab, _, _), r, m) in enumerate(zip(prognosen, prog_r2, prog_mae))
            ],
            f"Schrumpfgewicht λ = σ²_innen / σ²_Auftrag = {lam:.2f}: ab etwa {lam:.0f} "
            "bekannten Geschwistern zählt der Auftrag mehr als der Artikel. Die Seite "
            "'Was trägt der Auftrag über den Artikel hinaus' zeigt rechts, ab wie vielen Geschwistern das Geschwistermittel das Artikelmittel "
            "tatsächlich schlägt.",
            "Frage 4 — Was nutzen die Modelle? OOF-Vorhersagen von XGBoost (best_params_v3) "
            + ("und TabPFN (Parameter aus tabpfn_regression.py) " if MIT_TABPFN else "")
            + "auf denselben fünf Falten je Schema, der Fehler orthogonal auf die drei Ebenen "
            "projiziert. Beitrag = Varianzanteil × R²_k; die Summe ergibt exakt das Gesamt-R². "
            "KI: Bootstrap über Artikel. Der Rechenweg ist auf der Seite 'Wie die R²-Zerlegung "
            "rechnet' an einem Spielbeispiel mit zwölf Billets erklärt.",
            *[zeile_zerlegung(s, m) for m in modellnamen for s in SCHEMATA],
            f"Gegenprobe Abbrüche: {n_abbr} Billets erreichen die letzte Programmphase nicht. "
            "Dieselben OOF-Vorhersagen, ohne diese Billets zerlegt "
            "(Varianzanteile neu berechnet):",
            *[zeile_abbr(s, m) for m in modellnamen for s in SCHEMATA],
            "Zum Vergleich reine Gruppenmittel auf denselben Falten (ohne jedes Merkmal):",
            *[zeile_vergleich(s) for s in SCHEMATA],
            "Frage 5 — Steigung der Modellmittel gegen die y-Mittel (1 = Unterschied voll "
            "nachgebildet, 0 = gar nicht): Artikelebene "
            + ", ".join(f"{s} {M[s]['b_art']:+.2f}" for s in SCHEMATA)
            + "; Auftrag im Artikel "
            + ", ".join(f"{s} {M[s]['b_auf']:+.2f}" for s in SCHEMATA)
            + ".",
            f"Frage 6 — Woher kommt der Auftragseffekt im Artikel? Zeit: Abweichung vom "
            f"Artikelmittel gegen die Lage des Auftrags relativ zu den anderen Aufträgen "
            f"desselben Artikels, ρ {rho_zeit:+.3f} ({C.p_text(p_zeit)}). Merkmale: von "
            f"{Xd.shape[1]} Merkmalen, die zwischen Aufträgen desselben Artikels variieren, "
            f"hängen {n_sig} nach FDR-Korrektur (q < 0,05) mit der Auftragsabweichung "
            f"zusammen; stärkstes: {T.merkmal.iloc[0]} (ρ {T.rho.iloc[0]:+.3f}). Alle |ρ| "
            "sind klein, und die vorderen Plätze sind Influx-Zähler und -Streuungen, die sich "
            "über die Zeit verschieben. Weil die Aufträge desselben Artikels zeitlich "
            "nacheinander liegen und y fällt, ist ein Teil davon vermutlich Zeit, nicht "
            "Ursache (siehe 05_zeit.pdf).",
            "Lesart für die Modelle: Die Artikelebene trägt den Großteil der Güte, und zwar "
            "in jedem Schema. Der Auftragsbeitrag erscheint nur, wenn Geschwister desselben "
            "Auftrags im Training stehen (zufällige Falten) — dann ist er Wiedererkennen, "
            "nicht Erklären. Auf neuen Aufträgen fällt er auf null, obwohl die Auftragsebene "
            "in den Daten real und so groß wie die Artikelebene ist. Das ist die Lücke, die "
            "Merkmale schließen müssten, die zwischen Aufträgen desselben Artikels variieren.",
            "Dass das Modell auftragsrein das reine Artikelmittel deutlich schlägt "
            f"(R² {M['auftragsrein']['vergleich']['XGBoost'][0]:+.3f} gegen "
            f"{M['auftragsrein']['vergleich']['Artikelmittel'][0]:+.3f}), liegt fast ganz auf "
            "der Artikelebene: die Merkmale (Rezept, Sollmaße, Rohstoff) schätzen das "
            "Artikelniveau stabiler als der Mittelwert weniger Aufträge, und artikelrein "
            "übertragen sie es sogar auf unbekannte Artikel. Kurz: für die Modelle ist der "
            "ARTIKEL die wichtigere Ebene; der Auftrag wäre in den Daten gleich wichtig, ist "
            "aber aus den vorhandenen Merkmalen nicht erreichbar.",
        ],
    )

    fig, ax = C.seite(pdf, "Wie hängen Aufträge und Artikel zusammen?", 2, 2)

    bins = np.arange(0.5, n_auf_je_art.max() + 1.5)
    ax[0].hist(n_auf_je_art, bins=bins, color=ART, edgecolor="white", linewidth=0.8)
    ax[0].set_xlabel("Aufträge je Artikel")
    ax[0].set_ylabel("Artikel")
    ax[0].set_title(
        f"{k_art} Artikel, davon {(n_auf_je_art >= 2).sum()} mit ≥ 2 Aufträgen", fontsize=10
    )

    bins = np.arange(0.5, n_bil_je_auf.max() + 1.5)
    ax[1].hist(n_bil_je_auf, bins=bins, color=AUF, edgecolor="white", linewidth=0.8)
    ax[1].set_xlabel("Billets je Auftrag")
    ax[1].set_ylabel("Aufträge")
    ax[1].set_title(
        f"{k_auf} Aufträge, davon {(n_bil_je_auf == 1).sum()} mit nur einem Billet", fontsize=10
    )

    top = n_auf_je_art.sort_values(ascending=False).head(15).index
    for i, a in enumerate(top):
        k = art == a
        ax[2].scatter(np.full(k.sum(), i) + rng.uniform(-0.3, 0.3, k.sum()), y[k],
                      s=3, color=BIL, alpha=0.5, linewidths=0)
        am = A[A.art == a]
        ax[2].scatter(np.full(len(am), i), am.m, s=10 + 4 * am.n, color=AUF, alpha=0.75,
                      edgecolor="white", linewidths=0.5, zorder=3)
        ax[2].hlines(y[k].mean(), i - 0.42, i + 0.42, color=ART, lw=2.2, zorder=4)
    ax[2].set_xticks(range(len(top)))
    ax[2].set_xticklabels([f"{n_auf_je_art[a]}" for a in top], fontsize=8)
    ax[2].set_xlabel("Artikel (Beschriftung: Zahl der Aufträge)")
    ax[2].set_ylabel("y [%]")
    ax[2].set_title("15 Artikel mit den meisten Aufträgen", fontsize=10)
    ax[2].plot([], [], "o", color=AUF, label="Auftragsmittel (Größe ~ Billets)")
    ax[2].plot([], [], "-", color=ART, lw=2.2, label="Artikelmittel")
    ax[2].plot([], [], "o", color=BIL, ms=3, label="Billet")
    ax[2].legend(fontsize=7, loc="upper right")

    ax[3].scatter(Am.art_andere, Am.m, s=6 + 3 * Am.n, color=AUF, alpha=0.45,
                  edgecolor="white", linewidths=0.4)
    lim = [0, max(Am.m.max(), Am.art_andere.max()) + 2]
    ax[3].plot(lim, lim, color="k", lw=0.8, ls="--")
    ax[3].set_xlabel("Mittel der anderen Aufträge desselben Artikels [%]")
    ax[3].set_ylabel("Auftragsmittel [%]")
    ax[3].set_title(
        f"Sagt der Artikel den Auftrag vorher?  ρ {rho_auf_art:+.2f}, R² {r2_auf_art:+.2f}",
        fontsize=10,
    )
    C.schliesse(pdf, fig, fuss)

    fig, ax = C.seite(pdf, "Was trägt der Auftrag über den Artikel hinaus — ohne Modell?", 1, 3)

    namen = ["Artikel", "Auftrag\nim Artikel", "Billet\nim Auftrag"]
    werte = [v_art, v_auf, v_bil]
    nulls = [null_art, null_auf, None]
    for i, (w, nl, f) in enumerate(zip(werte, nulls, [ART, AUF, BIL])):
        ax[0].bar(i, w, color=f, width=0.6)
        ax[0].text(i, w + 0.01, f"{w:.3f}", ha="center", va="bottom", fontsize=9)
        if nl is not None:
            lo, hi = nl.mean() - 2 * nl.std(), nl.mean() + 2 * nl.std()
            ax[0].fill_between([i - 0.3, i + 0.3], lo, hi, color="crimson", alpha=0.3, lw=0)
            ax[0].hlines(nl.mean(), i - 0.3, i + 0.3, color="crimson", lw=1.6)
    ax[0].plot([], [], "-", color="crimson", lw=1.6, label="Nullerwartung ± 2σ")
    ax[0].set_xticks(range(3))
    ax[0].set_xticklabels(namen, fontsize=9)
    ax[0].set_ylim(0, v_bil * 1.12)
    ax[0].set_ylabel("Anteil der y-Varianz")
    ax[0].set_title("Varianzzerlegung mit Nullmodell", fontsize=10)
    ax[0].legend(fontsize=7, loc="upper left")

    pos = np.arange(len(prognosen))
    ax[1].bar(pos, prog_r2, color=[f for *_, f in prognosen], width=0.6)
    ax[1].errorbar(pos, prog_r2, yerr=[np.array(prog_r2) - prog_ci[0], prog_ci[1] - np.array(prog_r2)],
                   fmt="none", ecolor="k", lw=1, capsize=3)
    for i, r in enumerate(prog_r2):
        ax[1].text(i, prog_ci[1, i] + 0.005, f"{r:+.3f}", ha="center", va="bottom", fontsize=8)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xticks(pos)
    ax[1].set_xticklabels([lab for lab, *_ in prognosen], fontsize=8)
    ax[1].set_ylabel("R² (Leave-out)")
    ax[1].set_title(f"Prognose aus Gruppenmitteln (n = {S.sum()})", fontsize=10)

    xs = np.arange(len(G))
    for (lab, _, f), mk in zip(prognosen[1:], ["o", "s", "D"]):
        ax[2].plot(xs, G[lab], marker=mk, color=f, lw=2, ms=6, label=lab.replace("\n", " "))
    ax[2].set_xticks(xs)
    ax[2].set_xticklabels([f"{r['name']}\n(n={r['n']})" for _, r in G.iterrows()], fontsize=8)
    ax[2].set_xlabel("bekannte Geschwisterbillets im Auftrag")
    ax[2].set_ylabel("MAE [Prozentpunkte]")
    ax[2].set_title("Fehler je Zahl bekannter Geschwister", fontsize=10)
    ax[2].legend(fontsize=7, loc="upper right")
    ax[2].grid(alpha=0.3, axis="y")
    C.schliesse(pdf, fig, fuss)

    erklaerseite(pdf, fuss)

    fig, ax = C.seite(pdf, "Welche Ebene nutzt XGBoost?", 1, 3)

    pos = np.arange(len(SCHEMATA))
    for j, (ebene, f, lab) in enumerate(
        [("artikel", ART, "Artikel"), ("auftrag", AUF, "Auftrag im Artikel"), ("billet", BIL, "Billet im Auftrag")]
    ):
        c = np.array([M[s]["zerlegung"][ebene][2] for s in SCHEMATA])
        lo = np.array([M[s]["ci"][0, j] for s in SCHEMATA])
        hi = np.array([M[s]["ci"][1, j] for s in SCHEMATA])
        xpos = pos + (j - 1) * 0.26
        ax[0].bar(xpos, c, width=0.24, color=f, label=lab)
        ax[0].errorbar(xpos, c, yerr=[c - lo, hi - c], fmt="none", ecolor="k", lw=0.9, capsize=2)
    for i, s in enumerate(SCHEMATA):
        ax[0].text(i, -0.045, f"R² {M[s]['zerlegung']['gesamt']:+.3f}", ha="center", fontsize=8)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xticks(pos)
    ax[0].set_xticklabels(SCHEMATA)
    ax[0].set_ylim(-0.06, None)
    ax[0].set_ylabel("Beitrag zum Gesamt-R²")
    ax[0].set_title("R²-Zerlegung der OOF-Vorhersage", fontsize=10)
    ax[0].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)

    for j, (ebene, f, lab) in enumerate(
        [("artikel", ART, "Artikel"), ("auftrag", AUF, "Auftrag im Artikel"), ("billet", BIL, "Billet im Auftrag")]
    ):
        r = [M[s]["zerlegung"][ebene][1] for s in SCHEMATA]
        ax[1].bar(pos + (j - 1) * 0.26, r, width=0.24, color=f, label=lab)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xticks(pos)
    ax[1].set_xticklabels(SCHEMATA)
    ax[1].set_ylabel("R²_k innerhalb der Ebene")
    ax[1].set_title("Wie gut trifft das Modell jede Ebene?", fontsize=10)
    ax[1].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)

    verf = list(M["zufällig"]["vergleich"].keys())
    farben = [GLOBAL, ART, AUF, MODELL, TABPFN][: len(verf)]
    breite = 0.8 / len(verf)
    for j, (v, f) in enumerate(zip(verf, farben)):
        r = [M[s]["vergleich"][v][0] for s in SCHEMATA]
        ax[2].bar(pos + (j - (len(verf) - 1) / 2) * breite, r, width=breite * 0.9, color=f,
                  label=v.replace("\n", " "))
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_xticks(pos)
    ax[2].set_xticklabels(SCHEMATA)
    ax[2].set_ylabel("OOF-R²")
    ax[2].set_title("Modell gegen Gruppenmittel, gleiche Falten", fontsize=10)
    ax[2].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    C.schliesse(pdf, fig, fuss)

    ebenen = [("artikel", ART, "Artikel"), ("auftrag", AUF, "Auftrag im Artikel"), ("billet", BIL, "Billet im Auftrag")]
    fig, ax = C.seite(pdf, "Modelle im Vergleich und ohne abgebrochene Schmelzen", 1, 2)
    nm = len(modellnamen)
    breite = 0.8 / (3 * nm)
    for j, (ebene, f, lab) in enumerate(ebenen):
        for k, m in enumerate(modellnamen):
            c = np.array([M[s]["modelle"][m]["zerlegung"][ebene][2] for s in SCHEMATA])
            ci = np.array([M[s]["modelle"][m]["ci"][:, j] for s in SCHEMATA])
            xpos = pos + (j * nm + k - (3 * nm - 1) / 2) * breite
            ax[0].bar(xpos, c, width=breite * 0.9, color=f, hatch="////" if k else None,
                      edgecolor="white" if k else f, lw=0.5)
            ax[0].errorbar(xpos, c, yerr=[c - ci[:, 0], ci[:, 1] - c], fmt="none", ecolor="k",
                           lw=0.8, capsize=1.5)
    for i, s in enumerate(SCHEMATA):
        txt = "  ".join(f"{m} {M[s]['modelle'][m]['zerlegung']['gesamt']:+.3f}" for m in modellnamen)
        ax[0].text(i, -0.05, txt, ha="center", fontsize=7)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xticks(pos)
    ax[0].set_xticklabels(SCHEMATA)
    ax[0].set_ylim(-0.065, None)
    ax[0].set_ylabel("Beitrag zum Gesamt-R²")
    ax[0].set_title("R²-Zerlegung je Modell (schraffiert: " + ", ".join(modellnamen[1:]) + ")"
                    if nm > 1 else "R²-Zerlegung", fontsize=10)
    ax[0].legend(
        handles=[Patch(color=f, label=lab) for _, f, lab in ebenen]
        + [Patch(facecolor="white", edgecolor="k", hatch="////", label=m) for m in modellnamen[1:]],
        fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4,
    )

    breite = 0.8 / (2 * nm)
    for k, m in enumerate(modellnamen):
        for h, (schl, alpha, zusatz) in enumerate([("zerlegung", 1.0, "alle"), ("ohne_abbr", 0.45, "ohne Abbrüche")]):
            c = [M[s]["modelle"][m][schl]["billet"][2] for s in SCHEMATA]
            xpos = pos + (2 * k + h - (2 * nm - 1) / 2) * breite
            ax[1].bar(xpos, c, width=breite * 0.9, color=[MODELL, TABPFN][k], alpha=alpha,
                      label=f"{m}, {zusatz}")
            for xx, cc in zip(xpos, c):
                ax[1].text(xx, cc + 0.002, f"{cc:+.3f}", ha="center", va="bottom", fontsize=7)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xticks(pos)
    ax[1].set_xticklabels(SCHEMATA)
    ax[1].set_ylabel("Beitrag der Billetebene")
    ax[1].set_title(f"Billetbeitrag mit und ohne die {n_abbr} Abbrüche", fontsize=10)
    ax[1].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    C.schliesse(pdf, fig, fuss)

    fig, ax = C.seite(pdf, "Bildet das Modell Artikel- und Auftragsunterschiede nach?", 2, 3)
    for i, s in enumerate(SCHEMATA):
        Ar, Au = M[s]["Ar"], M[s]["Au"]
        a0 = ax[i]
        a0.scatter(Ar.y, Ar.p, s=4 + 1.5 * Ar.n, color=ART, alpha=0.5, edgecolor="white", linewidths=0.4)
        lim = [0, max(Ar.y.max(), Ar.p.max()) + 2]
        a0.plot(lim, lim, color="k", lw=0.8, ls="--")
        a0.set_title(f"Artikel — {s}: Steigung {M[s]['b_art']:+.2f}", fontsize=10)
        a0.set_xlabel("Artikelmittel y [%]")
        a0.set_ylabel("Artikelmittel Vorhersage [%]")

        a1 = ax[3 + i]
        a1.scatter(Au.dy, Au.dp, s=4 + 3 * Au.n, color=AUF, alpha=0.45, edgecolor="white", linewidths=0.4)
        g = C.robuste_grenzen(np.r_[Au.dy, Au.dp])
        a1.plot(g, g, color="k", lw=0.8, ls="--")
        a1.axhline(0, color=BIL, lw=0.6)
        a1.axvline(0, color=BIL, lw=0.6)
        a1.set_xlim(g)
        a1.set_ylim(g)
        a1.set_title(f"Auftrag im Artikel — {s}: Steigung {M[s]['b_auf']:+.2f}", fontsize=10)
        a1.set_xlabel("Auftragsmittel − Artikelmittel, y [Pp]")
        a1.set_ylabel("… der Vorhersage [Pp]")
    C.schliesse(pdf, fig, fuss)

    fig, ax = C.seite(pdf, "Was unterscheidet Aufträge desselben Artikels?", 1, 2)
    ax[0].scatter(Am.dts[ts_ok], Am.dy[ts_ok], s=6 + 3 * Am.n[ts_ok], color=AUF, alpha=0.45,
                  edgecolor="white", linewidths=0.4)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xlabel("Auftragsbeginn relativ zum Mittel der Aufträge des Artikels [Tage]")
    ax[0].set_ylabel("Auftragsmittel − Artikelmittel [Pp]")
    ax[0].set_title(f"Zeit im Artikel: ρ {rho_zeit:+.3f} ({C.p_text(p_zeit)})", fontsize=10)

    t = T.head(15).iloc[::-1]
    farbe = [AUF if q < 0.05 else GLOBAL for q in t.q]
    ax[1].barh(np.arange(len(t)), t.rho, color=farbe)
    ax[1].set_yticks(np.arange(len(t)))
    ax[1].set_yticklabels(t.merkmal, fontsize=7)
    ax[1].axvline(0, color="k", lw=0.8)
    ax[1].set_xlabel("Spearman ρ mit Auftragsmittel − Artikelmittel")
    ax[1].set_title(f"Merkmale im Artikel ({n_sig} von {Xd.shape[1]} mit q < 0,05)", fontsize=10)
    ax[1].legend(
        handles=[Patch(color=AUF, label="q < 0,05 (FDR)"), Patch(color=GLOBAL, label="nicht signifikant")],
        fontsize=7,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=2,
    )
    C.schliesse(pdf, fig, fuss)

    pdf.close()
    print(f"geschrieben: {C.OUT_DIR / '19_auftrag_vs_artikel.pdf'}")
    print(f"Anteile: Artikel {v_art:.3f} (Null {null_art.mean():.3f}) | Auftrag {v_auf:.3f} "
          f"(Null {null_auf.mean():.3f}) | Billet {v_bil:.3f}; λ {lam:.2f}")
    for (lab, *_), r, m in zip(prognosen, prog_r2, prog_mae):
        print(f"  {lab.replace(chr(10), ' ')}: R² {r:+.3f}, MAE {m:.2f}")
    for s in SCHEMATA:
        for m in modellnamen:
            print(zeile_zerlegung(s, m))
            print(zeile_abbr(s, m))
        print(zeile_vergleich(s))
        print(f"  Steigung Artikel {M[s]['b_art']:+.2f}, Auftrag {M[s]['b_auf']:+.2f}")
    print(f"Zeit ρ {rho_zeit:+.3f}; Merkmale q<0,05: {n_sig}/{Xd.shape[1]}")
    print(T.head(10).to_string())


if __name__ == "__main__":
    main()
