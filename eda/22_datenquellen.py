"""22 — Beitrag der Datenquellen.

In der Arbeit: Abschnitt 6.4: Abbildung 14 (nur Stammdaten bzw. nur
Prozessdaten) mit den Intervallen aus dem gepaarten Cluster-Bootstrap.

Stammdaten, Prozessdaten oder beide, vier Modelle, drei Ebenen, Gruppen-Bootstrap.
Optionen --modelle und --nur-plot. Ausgabe: runs/eda/22_*,
reports/figures/datenquellen_ebenen.png/.pdf.

    python eda/22_datenquellen.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

import _common as C
from data_prep import SP  # noqa: E402

D = C.D

MODELLE = ["XGBoost", "CatBoost", "TabPFN", "MiniRocket"]
EBENEN = [
    ("Billet", "random", SP["auftrag"]),
    ("Auftrag", "auftrag", SP["auftrag"]),
    ("Artikel", "artikel", SP["artikel"]),
]
QUELLEN = {"beides": "beide Quellen", "sql": "nur Stammdaten", "influx": "nur Prozesskanäle"}

N_BOOT = 2000

STIL = {
    "sql": dict(label="nur Stammdaten", farbe="#2a78d6"),
    "influx": dict(label="nur Prozesskanäle", farbe="#eb6834"),
}
BREITE = 0.36
VERSATZ = {"sql": -BREITE / 2, "influx": BREITE / 2}
TEXT = "#52514e"
FIG_BREITE = 16 / 2.54
FIG_HOEHE = 3.6

CACHE = C.ROOT / "runs" / "eda"
ERGEBNIS = CACHE / "22_datenquellen.json"
OUT = C.ROOT / "reports" / "figures" / "datenquellen_ebenen"


def _xai_common():
    spec = importlib.util.spec_from_file_location("xai_common", C.ROOT / "xai" / "_common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


XC = _xai_common()


def mache_xgb():
    import xgboost as xgb

    params, _ = XC.xgb_parameter()
    _, fix = XC._konstanten_aus_skript("xgboost_regression.py")
    return xgb.XGBRegressor(**params, **fix)


def mache_catboost():
    from catboost import CatBoostRegressor

    params, _ = XC.catboost_parameter()
    _, fix = XC._konstanten_aus_skript("catboost_regression.py")
    return CatBoostRegressor(**params, **fix, verbose=0)


def mache_tabpfn():
    from tabpfn import TabPFNRegressor

    default, fix = XC._konstanten_aus_skript("tabpfn_regression.py")
    return TabPFNRegressor(**default, **fix)


def lade_quelle(quelle: str):
    D.FEATURE_SOURCE = quelle
    ds = C.lade(alle_arten=True)
    return ds


def tabellarisch_oof(mache, X_raw, y, folds):
    pred = np.full(len(y), np.nan)
    n_feat = None
    for tr, va in folds:
        Xtr, Xva, _ = D.process_features(X_raw.iloc[tr], X_raw.iloc[va])
        n_feat = n_feat or Xtr.shape[1]
        m = mache()
        m.fit(Xtr, y.iloc[tr])
        pred[va] = m.predict(Xva)
    return pred, n_feat


def rocket_oof(RK, X3d, X_raw, y, folds, mit_kurven: bool):
    """Wie rocket_regression._fit_predict (Hybrid), Kurven optional."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    pred = np.full(len(y), np.nan)
    n_feat = None
    for tr, va in folds:
        Tt, Tv, _ = D.process_features(X_raw.iloc[tr], X_raw.iloc[va])
        Ztr, Zva = Tt.fillna(0.0).to_numpy(), Tv.fillna(0.0).to_numpy()
        if mit_kurven:
            tf = RK._make_transformer()
            tf.fit(X3d[tr])
            Rtr = np.nan_to_num(np.asarray(tf.transform(X3d[tr]), dtype=np.float64),
                                nan=0.0, posinf=0.0, neginf=0.0)
            Rva = np.nan_to_num(np.asarray(tf.transform(X3d[va]), dtype=np.float64),
                                nan=0.0, posinf=0.0, neginf=0.0)
            Ztr, Zva = np.hstack([Rtr, Ztr]), np.hstack([Rva, Zva])
        n_feat = n_feat or Ztr.shape[1]
        sc = StandardScaler().fit(Ztr)
        ytr = y.iloc[tr].to_numpy()
        ridge = RidgeCV(alphas=RK.RIDGE_ALPHAS).fit(sc.transform(Ztr), ytr)
        p = ridge.predict(sc.transform(Zva))
        if RK.CLIP_TO_TRAIN_RANGE:
            p = np.clip(p, ytr.min(), ytr.max())
        pred[va] = p
    return pred, n_feat


def cache_pfad(modell, cv, quelle):
    return CACHE / f"22_oof_{modell}_{cv}_{quelle}.npz"


def rechne(modelle):
    daten = {q: lade_quelle(q) for q in QUELLEN}
    basis = daten["beides"]
    for q, ds in daten.items():
        if not np.array_equal(ds.batch_ids.to_numpy(), basis.batch_ids.to_numpy()):
            raise AssertionError(f"Quelle {q}: andere Zeilen als 'beides'")
        print(f"Quelle {q:7s}: {ds.X_raw.shape[1]} Rohspalten, {len(ds.y)} Billets")
    y = basis.y.reset_index(drop=True)

    RK, X3d = None, None
    for modell in modelle:
        for _, cv, _ in EBENEN:
            folds, _ = D.make_cv(basis.df, level=cv, seed=42)
            for q, ds in daten.items():
                pfad = cache_pfad(modell, cv, q)
                if pfad.exists():
                    continue
                X_raw = ds.X_raw.reset_index(drop=True)
                t0 = time.time()
                if modell == "MiniRocket":
                    if RK is None:
                        import rocket_regression as RK  # noqa: N811
                        X3d = RK.build_or_load_tensor(basis.batch_ids)
                    pred, n_feat = rocket_oof(RK, X3d, X_raw, y, folds, mit_kurven=(q != "sql"))
                else:
                    mache = {"XGBoost": mache_xgb, "CatBoost": mache_catboost,
                             "TabPFN": mache_tabpfn}[modell]
                    pred, n_feat = tabellarisch_oof(mache, X_raw, y, folds)
                mae_f = [C.mae(y.iloc[va], pred[va]) for _, va in folds]
                r2_f = [C.r2(y.iloc[va], pred[va]) for _, va in folds]
                CACHE.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    pfad, oof=pred, batch_ids=basis.batch_ids.to_numpy().astype(str),
                    mae_folds=mae_f, r2_folds=r2_f, n_feat=n_feat,
                    fold_id=_fold_id(folds, len(y)),
                )
                print(f"  {modell:10s} {cv:8s} {q:7s}  {n_feat:5d} Merkmale  "
                      f"MAE {np.mean(mae_f):6.3f}  R² {np.mean(r2_f):+.3f}  "
                      f"({time.time() - t0:5.0f} s)", flush=True)


def _fold_id(folds, n):
    f = np.full(n, -1)
    for i, (_, va) in enumerate(folds):
        f[va] = i
    return f


def gruppen_summen(werte, gidx, k):
    return np.bincount(gidx, weights=werte, minlength=k)


def auswerten():
    D.FEATURE_SOURCE = "beides"
    ds = C.lade(alle_arten=True)
    df = ds.df.reset_index(drop=True)
    y = ds.y.reset_index(drop=True).to_numpy(float)
    bids = ds.batch_ids.to_numpy().astype(str)
    rng = np.random.default_rng(C.SEED)

    erg = {}
    for name, cv, gcol in EBENEN:
        folds, _ = D.make_cv(ds.df, level=cv, seed=42)
        med = np.full(len(y), np.nan)
        for tr, va in folds:
            med[va] = np.median(y[tr])
        med_mae = float(np.mean([C.mae(y[va], med[va]) for _, va in folds]))

        codes, uniq = pd.factorize(df[gcol].astype(str))
        k = len(uniq)
        n_g = np.bincount(codes, minlength=k).astype(float)
        sy = gruppen_summen(y, codes, k)
        syy = gruppen_summen(y**2, codes, k)
        W = np.stack([np.bincount(rng.integers(0, k, k), minlength=k) for _ in range(N_BOOT)])
        Wn = W @ n_g
        sstot = W @ syy - (W @ sy) ** 2 / Wn

        erg[name] = {"cv_level": cv, "bootstrap_gruppe": gcol, "n_gruppen": k,
                     "median_mae": med_mae, "modelle": {}}
        for modell in MODELLE:
            pf = {q: cache_pfad(modell, cv, q) for q in QUELLEN}
            if not all(p.exists() for p in pf.values()):
                continue
            z = {q: np.load(p) for q, p in pf.items()}
            for q in QUELLEN:
                if not np.array_equal(z[q]["batch_ids"], bids):
                    raise AssertionError(f"{pf[q].name}: Zeilen passen nicht")
            ae = {q: gruppen_summen(np.abs(y - z[q]["oof"]), codes, k) for q in QUELLEN}
            se = {q: gruppen_summen((y - z[q]["oof"]) ** 2, codes, k) for q in QUELLEN}
            mae_b = {q: (W @ ae[q]) / Wn for q in QUELLEN}
            r2_b = {q: 1 - (W @ se[q]) / sstot for q in QUELLEN}
            zeilen = {}
            for q in QUELLEN:
                e = {
                    "merkmale": int(z[q]["n_feat"]),
                    "cv_mae": float(np.mean(z[q]["mae_folds"])),
                    "cv_r2": float(np.mean(z[q]["r2_folds"])),
                    "oof_mae": float(ae[q].sum() / n_g.sum()),
                }
                if q != "beides":
                    d_mae = mae_b["beides"] - mae_b[q]
                    d_r2 = r2_b[q] - r2_b["beides"]
                    e |= {
                        "d_mae": float((ae["beides"].sum() - ae[q].sum()) / n_g.sum()),
                        "d_mae_lo": float(np.percentile(d_mae, 2.5)),
                        "d_mae_hi": float(np.percentile(d_mae, 97.5)),
                        "d_r2": e["cv_r2"] - float(np.mean(z["beides"]["r2_folds"])),
                        "d_r2_lo": float(np.percentile(d_r2, 2.5)),
                        "d_r2_hi": float(np.percentile(d_r2, 97.5)),
                    }
                zeilen[q] = e
            erg[name]["modelle"][modell] = zeilen

    ERGEBNIS.write_text(json.dumps(erg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\ngeschrieben: {ERGEBNIS.relative_to(C.ROOT)}")
    return erg


def drucke(erg):
    for name, e in erg.items():
        print(f"\n=== {name} (CV {e['cv_level']}, Bootstrap über {e['n_gruppen']} "
              f"{e['bootstrap_gruppe']}, {N_BOOT} Ziehungen; Median-MAE {e['median_mae']:.2f}) ===")
        print(f"{'Modell':11s} {'Quelle':19s} {'Merkm.':>6s} {'CV-MAE':>7s} {'CV-R²':>7s}"
              f"  {'ΔMAE (95-%-Bereich)':>26s}  {'ΔR² (95-%-Bereich)':>26s}")
        for modell, z in e["modelle"].items():
            for q, v in z.items():
                dm = (f"{v['d_mae']:+.2f} ({v['d_mae_lo']:+.2f} bis {v['d_mae_hi']:+.2f})"
                      if "d_mae" in v else "—")
                dr = (f"{v['d_r2']:+.3f} ({v['d_r2_lo']:+.3f} bis {v['d_r2_hi']:+.3f})"
                      if "d_r2" in v else "—")
                print(f"{modell:11s} {QUELLEN[q]:19s} {v['merkmale']:6d} {v['cv_mae']:7.2f} "
                      f"{v['cv_r2']:+7.3f}  {dm:>26s}  {dr:>26s}")


def plot(erg):
    modelle = [m for m in MODELLE if any(m in e["modelle"] for e in erg.values())]
    if not modelle:
        print("Kein vollständiger Block im Cache — keine Abbildung.")
        return
    fig, axes = plt.subplots(1, len(EBENEN), figsize=(FIG_BREITE, FIG_HOEHE), sharey=True)
    xs = np.arange(len(modelle))
    for ax, (name, _, _) in zip(axes, EBENEN):
        e = erg[name]["modelle"]
        for x, modell in zip(xs, modelle):
            if modell not in e:
                continue
            for q, stil in STIL.items():
                v = e[modell][q]
                ax.bar(x + VERSATZ[q], v["d_mae"], width=BREITE, color=stil["farbe"],
                       edgecolor="white", linewidth=1.0, zorder=2)
                ax.errorbar(x + VERSATZ[q], v["d_mae"],
                            yerr=[[v["d_mae"] - v["d_mae_lo"]], [v["d_mae_hi"] - v["d_mae"]]],
                            fmt="none", ecolor="#0b0b0b", elinewidth=0.9, capsize=2, zorder=3)
        ax.axhline(0, color="#0b0b0b", lw=0.8, zorder=3)
        ax.set_title(name, fontsize=10)
        ax.set_xticks(xs, modelle, rotation=35, ha="right", fontsize=8)
        ax.set_xlim(-0.6, len(xs) - 0.4)
        ax.grid(axis="y")
        ax.tick_params(colors=TEXT)
    axes[0].set_ylabel("Δ CV-MAE ggü. beiden Quellen\nin Prozentpunkten")
    legende = [Patch(color=s["farbe"], label=s["label"]) for s in STIL.values()]
    fig.legend(handles=legende, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.09))
    fig.tight_layout()
    C.aufraeumen(fig)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for endung in ("png", "pdf"):
        fig.savefig(OUT.with_suffix(f".{endung}"), dpi=300, bbox_inches="tight")
    print(f"Gespeichert: {OUT.with_suffix('.png')} (+ .pdf)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelle", nargs="+", default=MODELLE, choices=MODELLE)
    ap.add_argument("--nur-plot", action="store_true")
    a = ap.parse_args()
    if not a.nur_plot:
        rechne(a.modelle)
    erg = auswerten()
    drucke(erg)
    plot(erg)


if __name__ == "__main__":
    main()
