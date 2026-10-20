"""Gepaarter Bootstrap über Fertigungsaufträge auf den gespeicherten
Out-of-fold-Vorhersagen der finalen Läufe (Auswahl: runs_io.waehle_finale_laeufe).

In der Arbeit: nicht berichtet; gepaarte Intervalle der Modellvergleiche als
Gegenprobe zu Abschnitt 6.4; die Intervalle in Abbildung 14 rechnet eda/22.

Ergebnis: runs/sweeps/bootstrap_ci.json mit MAE und R² samt 95-%-Intervall und
den gepaarten Differenzen zu Median und XGBoost. Die Punktschätzer sind gepoolt.

    python data_pipeline/bootstrap_ci.py
"""
import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, pandas as pd, data_prep as dp
from runs_io import waehle_finale_laeufe
from data_prep import SP  # noqa: E402

N_BOOT = 2000
SEED = 42
EBENEN = {"billet": "random", "auftrag": "auftrag", "artikel": "artikel"}
KONFIG = {"XGBoost": "XGBoost", "CatBoost": "CatBoost", "TabPFN": "TabPFN",
          "ROCKET": "ROCKET (hybrid, ohne z-Norm)", "HYDRA": "HYDRA",
          "FreshPRINCE": "FreshPRINCE"}

s = dp.load_split(verbose=False)
df = s.df.reset_index(drop=True)
gruppe = dict(zip(df["batch_id"].astype(str), df[SP["auftrag"]].astype(str)))

gewaehlt = waehle_finale_laeufe(dp.RUNS_DIR / "runs_summary.csv", dp.RUN_ARTIFACTS_DIR,
                                konfigs=list(KONFIG.values()))

def waehle(modell, ebene):
    t = gewaehlt[(gewaehlt.konfig == KONFIG[modell]) & (gewaehlt.ebene == ebene)]
    return None if t.empty else t.run_id.iloc[0]

def lade_oof(run_id):
    p = dp.RUN_ARTIFACTS_DIR / f"run_{run_id}.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    pr = d["predictions"]
    return pd.DataFrame({"bid": [str(b) for b in pr["batch_ids"]],
                         "y": pr["y_true"], "p": pr["oof_pred"]}).dropna()

def mae(y, p): return float(np.mean(np.abs(y - p)))
def r2(y, p):  return float(1 - ((y-p)**2).sum() / ((y-y.mean())**2).sum())

rng = np.random.default_rng(SEED)
ergebnis = {}
for ebene in EBENEN:
    frames, ids = {}, {}
    for m in KONFIG:
        rid = waehle(m, ebene)
        if rid is None:
            continue
        ids[m] = rid
        frames[m] = lade_oof(rid)
    if not frames:
        continue
    gemein = set.intersection(*[set(f.bid) for f in frames.values()])
    basis = next(iter(frames.values()))
    basis = basis[basis.bid.isin(gemein)].sort_values("bid").reset_index(drop=True)
    y = basis.y.to_numpy()
    g = np.array([gruppe.get(b, b) for b in basis.bid])
    preds = {}
    for m, f in frames.items():
        f = f[f.bid.isin(gemein)].sort_values("bid").reset_index(drop=True)
        assert np.allclose(f.y.to_numpy(), y)
        preds[m] = f.p.to_numpy()
    preds["Median"] = np.full(len(y), np.median(y))
    preds["Mittel"] = np.full(len(y), np.mean(y))

    gu = pd.Series(range(len(g))).groupby(g).apply(lambda s: s.to_numpy())
    gkeys = list(gu.index)
    zieh = [np.concatenate([gu[k] for k in rng.choice(gkeys, len(gkeys), replace=True)])
            for _ in range(N_BOOT)]

    tab = {}
    for m, p in preds.items():
        maes = np.array([mae(y[i], p[i]) for i in zieh])
        r2s  = np.array([r2(y[i], p[i]) for i in zieh])
        d_med = np.array([mae(y[i], preds["Median"][i]) - mae(y[i], p[i]) for i in zieh])
        tab[m] = dict(
            n=len(y), run=ids.get(m),
            mae=mae(y, p), mae_lo=float(np.percentile(maes, 2.5)), mae_hi=float(np.percentile(maes, 97.5)),
            r2=r2(y, p), r2_lo=float(np.percentile(r2s, 2.5)), r2_hi=float(np.percentile(r2s, 97.5)),
            d_median=mae(y, preds["Median"]) - mae(y, p),
            d_median_lo=float(np.percentile(d_med, 2.5)), d_median_hi=float(np.percentile(d_med, 97.5)),
        )
    if "XGBoost" in preds:
        px = preds["XGBoost"]
        for m, p in preds.items():
            if m == "XGBoost":
                continue
            d = np.array([mae(y[i], px[i]) - mae(y[i], p[i]) for i in zieh])
            tab[m]["d_xgb"] = mae(y, px) - mae(y, p)
            tab[m]["d_xgb_lo"] = float(np.percentile(d, 2.5))
            tab[m]["d_xgb_hi"] = float(np.percentile(d, 97.5))
    ergebnis[ebene] = tab
    print(f"\n=== Ebene {ebene} (n={len(y)}, {len(gkeys)} Auftraege, {N_BOOT} Ziehungen) ===")
    print(f"{'Modell':13s} {'MAE':>7s} {'95%-KI':>17s} {'R2':>7s} {'95%-KI':>17s} {'vs Median':>9s} {'KI':>17s} {'vs XGB':>8s}")
    for m, v in sorted(tab.items(), key=lambda kv: kv[1]['mae']):
        dx = f"{v.get('d_xgb', 0):+.3f}" if 'd_xgb' in v else "   —"
        kix = f"[{v['d_xgb_lo']:+.2f};{v['d_xgb_hi']:+.2f}]" if 'd_xgb_lo' in v else ""
        print(f"{m:13s} {v['mae']:7.3f} [{v['mae_lo']:6.3f};{v['mae_hi']:6.3f}] {v['r2']:+7.3f} "
              f"[{v['r2_lo']:+6.3f};{v['r2_hi']:+6.3f}] {v['d_median']:+9.3f} "
              f"[{v['d_median_lo']:+6.3f};{v['d_median_hi']:+6.3f}] {dx} {kix}")

ziel = dp.SWEEPS_DIR / "bootstrap_ci.json"
ziel.parent.mkdir(parents=True, exist_ok=True)
ziel.write_text(json.dumps(ergebnis, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\ngeschrieben: {ziel.relative_to(dp.ROOT)}")
