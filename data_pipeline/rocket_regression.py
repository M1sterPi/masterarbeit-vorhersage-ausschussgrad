"""MiniRocket-Regression auf den Rohkurven, wahlweise zusammen mit der
Merkmalstabelle (HYBRID).

In der Arbeit: Rasterung des Kurvenzweigs (Abschnitt 5.2), Grenzen je Kanal
(5.6), z-Normierung mit und ohne (5.7); MiniRocket auf Kurven und hybrid für
Abschnitt 6.3/6.4 (Tabelle 12, Abbildungen 12 und 13).

Die Kurven je Kanal werden auf ein festes Raster gebracht und als Tensor in
data/cache/ abgelegt; darauf folgt eine Ridge-Regression. Schalter HYBRID und
ZNORM, Ebene über TEST_LEVEL und CV_LEVEL. Schreibt runs/single/,
runs/runs_summary.csv und reports/figures/rocket_results.png.

    python data_pipeline/rocket_regression.py
"""

import json
import os
import hashlib
import datetime
import pathlib

import numpy as np
import pandas as pd

from runs_io import lies_summary
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor

from aeon.transformations.collection.convolution_based import (
    MultiRocket,
    MiniRocket,
    Rocket,
)

import data_prep
from data_prep import (
    CSV_PATH,
    RUNS_DIR,
    RUN_ARTIFACTS_DIR,
    ONLY_CATEGORY,
    ONLY_GLASTYP,
    ONLY_MACHINE,
    Y_MIN,
    Y_MAX,
    SPLIT_SEED,
    TEST_SIZE,
    DROP_ID_FEATURES,
    TOP_FEATURES,
    process_features,
    load_split,
    _sha256,
    _git_state,
)
from data_prep import SP  # noqa: E402

MODEL_NAME = "ROCKET"

BILLET_DIR = data_prep.KURVEN_DIR
MAX_FEHLENDE_DATEIEN = 0.02

_PROC_COLS_JSON = (
    data_prep.ROOT / "data" / "cache" / (CSV_PATH.stem + ".process_cols.json")
)


def _lade_kanaele():
    """(kanäle, binäre) aus der Klassifikation des aktiven Datenstands."""
    if not _PROC_COLS_JSON.exists():
        raise SystemExit(
            f"Klassifikation fehlt: {_PROC_COLS_JSON}\n"
            "Erst data_pipeline/prepare_data.py laufen lassen — ROCKET leitet seine "
            "Kanäle daraus ab, damit beide Pfade nicht auseinanderlaufen."
        )
    d = json.loads(_PROC_COLS_JSON.read_text(encoding="utf-8"))
    proc = d["process_cols"]
    binaer = set(d.get("binary_cols", []))
    kanaele = [
        c
        for c in proc
        if not any(c.startswith(pref) for pref in data_prep._TIME_PROXY_PREFIXES)
    ]
    return kanaele, [c for c in kanaele if c in binaer]


CHANNELS, BINARY_CHANNELS = _lade_kanaele()

# Plausibilitätsgrenze je Kanal (Betrag), aus data/spalten.json
GRENZEN = dict(SP["kanalgrenzen"])
MAX_PLAUSIBLE_ABS = 1e6

SERIES_LEN = 512

TRANSFORMER = "MiniRocket"
N_KERNELS = 6250 if TRANSFORMER == "MultiRocket" else 1000
N_JOBS = -1

RIDGE_ALPHAS = np.logspace(-1, 6, 15)

# HYBRID=1: Kurven zusammen mit der Merkmalstabelle; ZNORM=1: z-Normierung je Kurve
HYBRID = os.environ.get("HYBRID", "1") != "0"

ZNORM = os.environ.get("ZNORM", "1") != "0"

CLIP_TO_TRAIN_RANGE = True

CV_SEED = 42


def _make_transformer():
    if TRANSFORMER == "MultiRocket":
        return MultiRocket(n_kernels=N_KERNELS, n_jobs=N_JOBS, random_state=42)
    if TRANSFORMER == "MiniRocket":
        return MiniRocket(n_kernels=N_KERNELS, n_jobs=N_JOBS, random_state=42)
    return Rocket(n_kernels=N_KERNELS, normalise=False, n_jobs=N_JOBS, random_state=42)


def _to_num_any(s: pd.Series) -> pd.Series:
    """Numerisch, unabhängig vom Speicherformat."""
    txt = s.astype(str).str.strip()
    mapped = txt.replace(
        {
            "True": "1",
            "False": "0",
            "true": "1",
            "false": "0",
            "TRUE": "1",
            "FALSE": "0",
        }
    )
    return pd.to_numeric(mapped.str.replace(",", ".", regex=False), errors="coerce")


def _zoh_resample(t, v, kanten):
    """Zero-Order-Hold-Resampling auf ein Zeitraster (zeitgewichtetes Zellmittel)."""
    if len(t) == 1:
        return np.full(len(kanten) - 1, v[0], dtype=np.float64)
    if kanten[0] < t[0]:
        t = np.concatenate([[kanten[0]], t])
        v = np.concatenate([[v[0]], v])
    if kanten[-1] > t[-1]:
        t = np.concatenate([t, [kanten[-1]]])
        v = np.concatenate([v, [v[-1]]])
    F = np.concatenate([[0.0], np.cumsum(v[:-1] * np.diff(t))])
    F_kanten = np.interp(kanten, t, F)
    breite = np.diff(kanten)
    return np.diff(F_kanten) / np.where(breite > 0, breite, 1.0)


def _read_billet_series(path: pathlib.Path):
    """Eine Billet-CSV → (C × SERIES_LEN)-Array, z-normiert je Kanal."""
    out = np.zeros((len(CHANNELS), SERIES_LEN), dtype=np.float32)

    header = pd.read_csv(path, sep=";", nrows=0, encoding="utf-8").columns
    present = [c for c in CHANNELS if c in header]
    if "_time" not in header or not present:
        return out, list(CHANNELS)

    df = pd.read_csv(
        path, sep=";", usecols=["_time"] + present, low_memory=False, encoding="utf-8"
    )
    t = pd.to_datetime(df["_time"], errors="coerce", utc=True)
    sec = (t - t.min()).dt.total_seconds().to_numpy()

    valid_t = sec[np.isfinite(sec)]
    if valid_t.size == 0:
        return out, list(CHANNELS)

    kanten = np.linspace(valid_t.min(), valid_t.max(), SERIES_LEN + 1)
    missing = [c for c in CHANNELS if c not in present]
    for i, ch in enumerate(CHANNELS):
        if ch not in present:
            continue
        vals = _to_num_any(df[ch]).to_numpy()
        grenze = GRENZEN.get(ch, MAX_PLAUSIBLE_ABS)
        m = np.isfinite(sec) & np.isfinite(vals) & (np.abs(vals) <= grenze)
        if m.sum() == 0:
            missing.append(ch)
            continue
        s = pd.Series(vals[m], index=sec[m]).groupby(level=0).mean().sort_index()
        series = _zoh_resample(s.index.to_numpy(), s.to_numpy(), kanten)
        if ZNORM:
            sd = series.std()
            series = (series - series.mean()) / sd if sd > 1e-8 else series * 0.0
        out[i] = series
    return out, missing


def _cache_path():
    key = hashlib.sha256(
        (
            "|".join(CHANNELS)
            + f"|L={SERIES_LEN}|znorm={ZNORM}|dir={BILLET_DIR.name}"
            + f"|resample=zoh|clip={MAX_PLAUSIBLE_ABS:g}|grenzen={sorted(GRENZEN.items())}"
        ).encode()
    ).hexdigest()[:12]
    cache_dir = data_prep.ROOT / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"_rocket_series_{key}.npz"


def build_or_load_tensor(batch_ids: pd.Series) -> np.ndarray:
    """3D-Tensor (n × C × SERIES_LEN) in der Reihenfolge von batch_ids."""
    ids = batch_ids.astype(str).tolist()
    cache = _cache_path()
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["batch_ids"]) == ids and int(z["series_len"]) == SERIES_LEN:
            print(f"  Tensor aus Cache geladen: {cache.name}  shape={z['X'].shape}")
            return z["X"]
        print("  Cache passt nicht (andere Billets/Länge) → neu bauen.")

    print(f"  Baue Tensor aus {len(ids)} Roh-CSVs in {BILLET_DIR} …")
    X = np.zeros((len(ids), len(CHANNELS), SERIES_LEN), dtype=np.float32)
    n_missing_file, missing_channel_counts = 0, {c: 0 for c in CHANNELS}
    for k, bid in enumerate(ids):
        f = BILLET_DIR / f"{bid}.csv"
        if not f.exists():
            n_missing_file += 1
            continue
        arr, miss = _read_billet_series(f)
        X[k] = arr
        for c in miss:
            missing_channel_counts[c] += 1
        if (k + 1) % 50 == 0:
            print(f"    … {k + 1}/{len(ids)} Billets eingelesen")

    print(f"  Fertig. Fehlende Rohdateien: {n_missing_file}")
    print("  Kanäle ohne Messpunkte (Anteil Billets):")
    for c in CHANNELS:
        print(f"    {c:<28}{missing_channel_counts[c]:>4}/{len(ids)}")

    anteil = n_missing_file / max(len(ids), 1)
    if anteil > MAX_FEHLENDE_DATEIEN:
        raise SystemExit(
            f"{n_missing_file} von {len(ids)} Rohdateien fehlen in {BILLET_DIR} "
            f"({100 * anteil:.1f} %, Grenze {100 * MAX_FEHLENDE_DATEIEN:.0f} %).\n"
            "Der Lauf wuerde fuer diese Billets Nullkurven modellieren und "
            "trotzdem eine Zahl liefern. BILLET_DIR pruefen."
        )
    tot = [c for c in CHANNELS if missing_channel_counts[c] == len(ids)]
    if tot:
        raise SystemExit(
            f"Diese Kanaele haben in KEINEM der {len(ids)} Billets Messpunkte: "
            f"{', '.join(tot)}.\nEntweder ist BILLET_DIR falsch oder die "
            "Klassifikation in prepare_data.py passt nicht zu diesen Rohdaten."
        )

    np.savez_compressed(
        cache,
        X=X,
        batch_ids=np.array(ids),
        series_len=SERIES_LEN,
        channels=np.array(CHANNELS),
    )
    print(f"  Tensor gecacht: {cache.relative_to(data_prep.ROOT)}  shape={X.shape}")
    return X


def _fit_predict(Xtr3d, ytr, Xva3d, tab_tr=None, tab_va=None):
    """ROCKET-Transform (fit nur auf Train) → optional Hybrid-Concat → StandardScaler
    → RidgeCV."""
    tf = _make_transformer()
    tf.fit(Xtr3d)
    Ztr = np.asarray(tf.transform(Xtr3d), dtype=np.float64)
    Zva = np.asarray(tf.transform(Xva3d), dtype=np.float64)
    Ztr = np.nan_to_num(Ztr, nan=0.0, posinf=0.0, neginf=0.0)
    Zva = np.nan_to_num(Zva, nan=0.0, posinf=0.0, neginf=0.0)

    if HYBRID and tab_tr is not None:
        Tt, Tv, _ = process_features(tab_tr, tab_va)
        Tt = Tt.fillna(0.0).to_numpy()
        Tv = Tv.fillna(0.0).to_numpy()
        Ztr = np.hstack([Ztr, Tt])
        Zva = np.hstack([Zva, Tv])

    scaler = StandardScaler().fit(Ztr) 
    ridge = RidgeCV(alphas=RIDGE_ALPHAS).fit(scaler.transform(Ztr), ytr)
    pred = ridge.predict(scaler.transform(Zva))
    if CLIP_TO_TRAIN_RANGE:
        pred = np.clip(pred, np.min(ytr), np.max(ytr))
    return pred, Ztr.shape[1]


def main():
    ds = load_split(verbose=True)
    if not BILLET_DIR.exists():
        raise SystemExit(
            f"BILLET_DIR nicht gefunden: {BILLET_DIR}\n"
            f"Passe BILLET_DIR in rocket_regression.py an das Rohdaten-Verzeichnis an."
        )

    print("═" * 64)
    print(f" ROHKURVEN → TENSOR  ({len(CHANNELS)} Kanäle × {SERIES_LEN} Punkte)")
    print("═" * 64)
    X3d = build_or_load_tensor(ds.batch_ids)
    y = ds.y.reset_index(drop=True)
    X_raw = ds.X_raw.reset_index(drop=True)
    test_mask = ds.test_mask
    batch_ids = ds.batch_ids

    print("\n" + "═" * 64)
    print(
        f" MODELL: {MODEL_NAME}  ({TRANSFORMER}, n_kernels={N_KERNELS}, "
        f"hybrid={HYBRID})"
    )
    print("═" * 64)

    tr = np.where(~test_mask)[0]
    te = np.where(test_mask)[0]
    y_pred, n_feat = _fit_predict(
        X3d[tr],
        y.iloc[tr],
        X3d[te],
        tab_tr=X_raw.iloc[tr] if HYBRID else None,
        tab_va=X_raw.iloc[te] if HYBRID else None,
    )
    y_test = y.iloc[te].to_numpy()
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    base = DummyRegressor(strategy="mean").fit(X3d[tr].reshape(len(tr), -1), y.iloc[tr])
    y_base = base.predict(X3d[te].reshape(len(te), -1))
    base_mae = mean_absolute_error(y_test, y_base)
    base_rmse = np.sqrt(mean_squared_error(y_test, y_base))
    base_r2 = r2_score(y_test, y_base)
    base_med = DummyRegressor(strategy="median").fit(
        X3d[tr].reshape(len(tr), -1), y.iloc[tr]
    )
    y_base_med = base_med.predict(X3d[te].reshape(len(te), -1))
    base_med_mae = mean_absolute_error(y_test, y_base_med)
    base_med_rmse = np.sqrt(mean_squared_error(y_test, y_base_med))
    base_med_r2 = r2_score(y_test, y_base_med)

    print(f"  ROCKET-Features: {n_feat}")
    print("\n── Testergebnisse (eingefrorener Split) ────────────────────")
    print(f"  MAE  : {mae:.4f}   RMSE : {rmse:.4f}   R² : {r2:.4f}")
    print(f"  Baseline(mean):  MAE {base_mae:.4f}   R² {base_r2:.4f}")
    print(
        f"  Baseline(median):MAE {base_med_mae:.4f}   R² {base_med_r2:.4f}"
        f"   → Gewinn {(base_med_mae - mae) / base_med_mae * 100:+.1f} %"
    )
    print("────────────────────────────────────────────────────────────\n")

    cv_folds, cv_level = data_prep.make_cv(ds.df, seed=CV_SEED)
    scores = {
        MODEL_NAME: {"r2": [], "mae": []},
        "Baseline (Mittel)": {"r2": [], "mae": []},
        "Baseline (Median)": {"r2": [], "mae": []},
    }
    oof_pred = np.full(len(y), np.nan)

    for fold, (tr_idx, va_idx) in enumerate(cv_folds, start=1):
        ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
        pred, _ = _fit_predict(
            X3d[tr_idx],
            ytr,
            X3d[va_idx],
            tab_tr=X_raw.iloc[tr_idx] if HYBRID else None,
            tab_va=X_raw.iloc[va_idx] if HYBRID else None,
        )
        oof_pred[va_idx] = pred
        scores[MODEL_NAME]["r2"].append(r2_score(yva, pred))
        scores[MODEL_NAME]["mae"].append(mean_absolute_error(yva, pred))
        for name, strat in [
            ("Baseline (Mittel)", "mean"),
            ("Baseline (Median)", "median"),
        ]:
            b = DummyRegressor(strategy=strat).fit(np.zeros((len(tr_idx), 1)), ytr)
            bp = b.predict(np.zeros((len(va_idx), 1)))
            scores[name]["r2"].append(r2_score(yva, bp))
            scores[name]["mae"].append(mean_absolute_error(yva, bp))
        print(
            f"  Fold {fold}/5:  R² = {scores[MODEL_NAME]['r2'][-1]:+.4f}   "
            f"MAE = {scores[MODEL_NAME]['mae'][-1]:.4f}"
        )

    print(f"\n── 5-Fold CV: {MODEL_NAME} vs. Baselines ────────────────────")
    print(f"  {'Modell':<20}{'R²':>16}{'MAE':>16}")
    print("  " + "─" * 50)
    for name, s in scores.items():
        r2a, maea = np.array(s["r2"]), np.array(s["mae"])
        print(
            f"  {name:<20}{r2a.mean():>8.4f} ± {r2a.std():.4f}"
            f"{maea.mean():>8.4f} ± {maea.std():.4f}"
        )
    print("────────────────────────────────────────────────────────────\n")

    oof_r2 = r2_score(y, oof_pred)
    oof_residuals = y.to_numpy() - oof_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"{MODEL_NAME} ({TRANSFORMER}) – y aus Rohkurven (5-Fold CV)", fontsize=14
    )
    axes[0].scatter(y, oof_pred, alpha=0.5, edgecolors="none", s=30)
    lims = [min(y.min(), np.nanmin(oof_pred)), max(y.max(), np.nanmax(oof_pred))]
    axes[0].plot(lims, lims, "r--", linewidth=1)
    axes[0].set_xlabel("Tatsächlich")
    axes[0].set_ylabel("Vorhergesagt (out-of-fold)")
    axes[0].set_title(f"Predicted vs. Actual  (OOF R²={oof_r2:.3f})")
    axes[1].scatter(oof_pred, oof_residuals, alpha=0.5, edgecolors="none", s=30)
    axes[1].axhline(0, color="r", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Vorhergesagt (out-of-fold)")
    axes[1].set_ylabel("Residuum")
    axes[1].set_title("Residuen (out-of-fold)")
    plt.tight_layout()
    out_png = data_prep.ROOT / "reports" / "figures" / "rocket_results.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Plot gespeichert: {out_png}")

    RUNS_DIR.mkdir(exist_ok=True)
    RUN_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    train_ids = batch_ids[~test_mask].tolist()
    test_ids_present = batch_ids[test_mask].tolist()
    params_source = f"{TRANSFORMER} n_kernels={N_KERNELS} + RidgeCV (kein Tuning)"

    cv_summary = {
        name: {
            "r2_mean": float(np.mean(s["r2"])),
            "r2_std": float(np.std(s["r2"])),
            "mae_mean": float(np.mean(s["mae"])),
            "mae_std": float(np.std(s["mae"])),
            "r2_folds": [float(v) for v in s["r2"]],
            "mae_folds": [float(v) for v in s["mae"]],
        }
        for name, s in scores.items()
    }

    record = {
        "run_id": run_id,
        "model": MODEL_NAME,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "git": _git_state(),
        "data": {
            "csv": str(CSV_PATH.relative_to(data_prep.ROOT)),
            "sha256": _sha256(CSV_PATH),
            "n_rows_after_filter": int(len(ds.df)),
            "billet_dir": str(BILLET_DIR),
        },
        "config": {
            "only_category": ONLY_CATEGORY,
            "only_glastyp": ONLY_GLASTYP,
            "only_machine": ONLY_MACHINE,
            "y_min": Y_MIN,
            "y_max": Y_MAX,
            "split_seed": SPLIT_SEED,
            "test_size": TEST_SIZE,
            "test_level": data_prep.TEST_LEVEL,
            "feature_source": data_prep.FEATURE_SOURCE,
            "categorical_encoding": data_prep.CATEGORICAL_ENCODING,
            "max_cardinality": data_prep.MAX_CARDINALITY,
            "drop_id_features": DROP_ID_FEATURES,
            "top_features": list(TOP_FEATURES) if TOP_FEATURES else None,
            "params_source": params_source,
        },
        "model_params": {
            "transformer": TRANSFORMER,
            "n_kernels": N_KERNELS,
            "channels": CHANNELS,
            "series_len": SERIES_LEN,
            "ridge_alphas": list(RIDGE_ALPHAS),
            "hybrid": HYBRID,
            "znorm": ZNORM,
            "clip_to_train_range": CLIP_TO_TRAIN_RANGE,
            "n_rocket_features": int(n_feat),
        },
        "split": {
            "n_train": int(len(train_ids)),
            "n_test": int(len(test_ids_present)),
            "train_ids": train_ids,
            "test_ids": test_ids_present,
        },
        "features": {
            "n_used": int(n_feat),
            "channels": CHANNELS,
            "series_len": SERIES_LEN,
        },
        "metrics": {
            "test": {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)},
            "baseline_mean": {
                "mae": float(base_mae),
                "rmse": float(base_rmse),
                "r2": float(base_r2),
            },
            "baseline_median": {
                "mae": float(base_med_mae),
                "rmse": float(base_med_rmse),
                "r2": float(base_med_r2),
            },
            "cv": cv_summary,
            "cv_level": cv_level,
            "oof_r2": float(oof_r2),
        },
        "predictions": {
            "batch_ids": [str(b) for b in batch_ids],
            "y_true": [round(float(v), 6) for v in y],
            "oof_pred": [round(float(v), 6) for v in oof_pred],
            "test_batch_ids": [str(b) for b in test_ids_present],
            "test_y_true": [round(float(v), 6) for v in y_test],
            "test_pred": [round(float(v), 6) for v in y_pred],
        },
    }

    run_json = RUN_ARTIFACTS_DIR / f"run_{run_id}.json"
    run_json.write_text(
        json.dumps(record, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    fig.savefig(RUN_ARTIFACTS_DIR / f"run_{run_id}.png", dpi=150, bbox_inches="tight")

    summary_row = {
        "run_id": run_id,
        "model": MODEL_NAME,
        "timestamp": record["timestamp"],
        "git_commit": (record["git"]["commit"] or "")[:8],
        "git_dirty": record["git"]["dirty"],
        "data_sha256": record["data"]["sha256"][:12],
        "only_category": ONLY_CATEGORY,
        "only_glastyp": ONLY_GLASTYP,
        "only_machine": ONLY_MACHINE,
        "test_level": data_prep.TEST_LEVEL,
        "cv_level": cv_level,
        "feature_source": data_prep.FEATURE_SOURCE,
        "categorical_encoding": data_prep.CATEGORICAL_ENCODING,
        "max_cardinality": data_prep.MAX_CARDINALITY,
        "ziel": data_prep.ZIEL,
        "y_min": Y_MIN,
        "y_max": Y_MAX,
        "n_rows": record["data"]["n_rows_after_filter"],
        "n_train": record["split"]["n_train"],
        "n_test": record["split"]["n_test"],
        "n_features": int(n_feat),
        "params_source": params_source,
        "best_iteration": N_KERNELS,
        "test_mae": round(float(mae), 4),
        "test_rmse": round(float(rmse), 4),
        "test_r2": round(float(r2), 4),
        "baseline_mittel_mae": round(float(base_mae), 4),
        "baseline_mittel_rmse": round(float(base_rmse), 4),
        "baseline_mittel_r2": round(float(base_r2), 4),
        "baseline_median_mae": round(float(base_med_mae), 4),
        "baseline_median_rmse": round(float(base_med_rmse), 4),
        "baseline_median_r2": round(float(base_med_r2), 4),
        "cv_ausgefuehrt": True,
        "cv_r2_mean": round(cv_summary[MODEL_NAME]["r2_mean"], 4),
        "cv_r2_std": round(cv_summary[MODEL_NAME]["r2_std"], 4),
        "cv_mae_mean": round(cv_summary[MODEL_NAME]["mae_mean"], 4),
        "cv_mae_std": round(cv_summary[MODEL_NAME]["mae_std"], 4),
        "cv_baseline_mittel_r2_mean": round(
            cv_summary["Baseline (Mittel)"]["r2_mean"], 4
        ),
        "cv_baseline_mittel_mae_mean": round(
            cv_summary["Baseline (Mittel)"]["mae_mean"], 4
        ),
        "cv_baseline_median_r2_mean": round(
            cv_summary["Baseline (Median)"]["r2_mean"], 4
        ),
        "cv_baseline_median_mae_mean": round(
            cv_summary["Baseline (Median)"]["mae_mean"], 4
        ),
        "oof_r2": round(float(oof_r2), 4),
    }
    summary_path = RUNS_DIR / "runs_summary.csv"
    new_row = pd.DataFrame([summary_row])
    if summary_path.exists():
        prev = lies_summary(summary_path)
        summary_df = pd.concat([prev, new_row], ignore_index=True)
        cols = list(summary_row.keys()) + [
            c for c in summary_df.columns if c not in summary_row
        ]
        summary_df = summary_df[cols]
    else:
        summary_df = new_row
    summary_df.to_csv(summary_path, index=False)

    print(f"Run protokolliert: {run_json.relative_to(data_prep.ROOT)}")
    print(f"Übersicht aktualisiert: {summary_path.relative_to(data_prep.ROOT)}")


if __name__ == "__main__":
    main()
