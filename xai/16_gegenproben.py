"""16 — Modell-Gegenprobe.

In der Arbeit: Abschnitt 7.2 und 7.3: Tabelle 15 (Modell-Gegenprobe mit drei
Modellen, Tabelle über xai/20).

Reagieren XGBoost, CatBoost und TabPFN auf dieselben Schwellen? Unterschied
kritischer Bereich minus Rest, roh und gegenüber dem Mittel des eigenen Artikels
bzw. Auftrags. Abgebrochene Schmelzen mit fester Schwelle wie in xai/14, die
übrigen Kennzahlen mit der Schwelle aus xai/14. Braucht
runs/eda/19_oof_tabpfn.npz (eda/19) und xai/14.

    python xai/16_gegenproben.py
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import _common as C
from data_prep import SP  # noqa: E402

S14 = importlib.import_module("14_sensorwerte")
_spec = importlib.util.spec_from_file_location("eda_common", C.ROOT / "eda" / "_common.py")
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

TABPFN_CACHE = C.ROOT / "runs" / "eda" / "19_oof_tabpfn.npz"
KENNZAHLEN = [SP["kanal_a"] + "_n_samples", SP["kanal_b"] + "_std", SP["phase"] + "_max",
              SP["kanal_o"] + "_known_frac"]
# Kennzahlen ohne Schwelle aus 14: (Schwelle, Richtung); "unter" heißt v < Schwelle
FESTE_SCHWELLEN = {SP["phase"] + "_max": (S14.PHASE_REGULAER, "unter")}


def tabpfn_oof(ys, folds) -> np.ndarray:
    """TabPFN-Vorhersagen aus dem Cache von eda/19 (gleicher Schluessel)."""
    cache = dict(np.load(TABPFN_CACHE))
    pred = np.full(len(ys), np.nan)
    for i, (_, va) in enumerate(folds):
        h = hashlib.sha1(ys.to_numpy(float).tobytes() + np.asarray(va).tobytes()).hexdigest()[:12]
        key = f"artikelrein_{i}_{h}"
        if key not in cache:
            raise SystemExit(f"{key} fehlt in {TABPFN_CACHE.name} — erst eda/19 laufen lassen.")
        pred[va] = cache[key]
    return pred


def baum_oof(X, ys, folds, art: str) -> np.ndarray:
    import xgboost as xgb
    from catboost import CatBoostRegressor

    pred = np.full(len(ys), np.nan)
    for tr, va in folds:
        Xtr, Xva, _ = C.D.process_features(X.iloc[tr], X.iloc[va])
        if art == "XGBoost":
            params, _ = C.xgb_parameter()
            _, fix = C._konstanten_aus_skript()
            m = xgb.XGBRegressor(**params, **fix)
        else:
            params, _ = C.catboost_parameter()
            _, fix = C._konstanten_aus_skript("catboost_regression.py")
            m = CatBoostRegressor(**params, **fix)
        m.fit(Xtr, ys.iloc[tr])
        pred[va] = m.predict(Xva)
    return pred


def zentriert(v, g) -> np.ndarray:
    return v - pd.Series(v).groupby(g).transform("mean").to_numpy()


def main():
    ds = E.lade(alle_arten=True)
    X, ys, df = E.daten_teil(ds)
    art = df[SP["artikel"]].astype(str).to_numpy(dtype=str)
    auf = df[SP["auftrag"]].astype(str).to_numpy(dtype=str)
    y = ys.to_numpy(float)
    folds = list(GroupKFold(5, shuffle=True, random_state=E.SEED).split(X, groups=art))
    print(f"{len(y)} Billets, {len(np.unique(art))} Artikel, artikelreine Falten wie eda/19")

    vorhersage = {"XGBoost": baum_oof(X, ys, folds, "XGBoost"),
                  "CatBoost": baum_oof(X, ys, folds, "CatBoost"),
                  "TabPFN": tabpfn_oof(ys, folds)}
    guete = {m: {"mae": float(np.mean(np.abs(y - p))), "mae_median": float(np.mean(np.abs(y - np.median(y))))}
             for m, p in vorhersage.items()}
    for m, g in guete.items():
        print(f"  {m:<8} MAE {g['mae']:.3f} (Median {g['mae_median']:.3f})")

    schwellen = pd.read_csv(C.TAB_DIR / "14_sensorwerte_schwellen_artikel.csv").set_index("Merkmal")
    rng = np.random.default_rng(C.SEED)
    zeilen = []
    for k in KENNZAHLEN:
        if k in FESTE_SCHWELLEN:
            schwelle, richtung = FESTE_SCHWELLEN[k]
        else:
            schwelle, richtung = schwellen.loc[k, "schwelle"], schwellen.loc[k, "richtung"]
        v = pd.to_numeric(X[k], errors="coerce").to_numpy(float)
        ok = ~np.isnan(v)
        krit = {"niedriger": v <= schwelle, "unter": v < schwelle}.get(richtung, v > schwelle)
        reihen = {"beobachteter Ausschuss": y, **vorhersage}
        for quelle, werte in reihen.items():
            for bezug, wz in (("roh", werte - np.nanmean(werte)), ("im Artikel", zentriert(werte, art)),
                              ("im Auftrag", zentriert(werte, auf))):
                d = wz[ok]
                a = krit[ok]
                w = d[a].mean() - d[~a].mean()
                lo, hi = boot_diff(d, a, art[ok], rng)
                zeilen.append({"Merkmal": k, "schwelle": schwelle, "richtung": richtung,
                               "anteil_kritisch": float(a.mean()), "quelle": quelle, "bezug": bezug,
                               "unterschied": float(w), "ki_lo": lo, "ki_hi": hi})
                print(f"  {k:<26} {quelle:<23} {bezug:<10} {w:+6.2f} [{lo:+.2f}; {hi:+.2f}]")
    T = pd.DataFrame(zeilen)
    pc = C.speichere_rangliste(C.mit_ebene("16_gegenproben.csv"), T, "_reihenfolge_behalten")
    p = C.speichere(C.mit_ebene("16_gegenproben.json"), {
        "test_level": "artikel", "falten": "GroupKFold(5, shuffle, 42) ueber Artikel wie eda/19",
        "n": int(len(y)), "guete": guete, "ergebnisse": T.to_dict("records"), "csv": pc.name,
    })
    print("geschrieben:", p)


def boot_diff(d, krit, art, rng, n=1000):
    """95-%-KI fuer mean(d | krit) − mean(d | ¬krit), gezogen werden ganze Artikel."""
    codes, inv = np.unique(art, return_inverse=True)
    s1 = np.bincount(inv, weights=d * krit, minlength=len(codes))
    c1 = np.bincount(inv, weights=krit.astype(float), minlength=len(codes))
    s0 = np.bincount(inv, weights=d * ~krit, minlength=len(codes))
    c0 = np.bincount(inv, weights=(~krit).astype(float), minlength=len(codes))
    z = rng.integers(0, len(codes), (n, len(codes)))
    w = s1[z].sum(1) / np.maximum(c1[z].sum(1), 1) - s0[z].sum(1) / np.maximum(c0[z].sum(1), 1)
    return float(np.percentile(w, 2.5)), float(np.percentile(w, 97.5))


if __name__ == "__main__":
    main()
