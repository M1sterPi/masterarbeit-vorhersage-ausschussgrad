"""Gemeinsamer Ablauf für HYDRA und FreshPRINCE auf dem Kurventensor aus
rocket_regression.py: Kreuzvalidierung, Test, Protokoll in runs/.

In der Arbeit: gemeinsamer Ablauf für die HYDRA- und FreshPRINCE-Läufe aus
Abschnitt 6.3/6.4.
"""

from __future__ import annotations

import datetime
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.dummy import DummyRegressor  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import data_prep  # noqa: E402
import rocket_regression as RK  # noqa: E402
from data_prep import (  # noqa: E402
    CSV_PATH,
    DROP_ID_FEATURES,
    ONLY_CATEGORY,
    ONLY_GLASTYP,
    ONLY_MACHINE,
    RUN_ARTIFACTS_DIR,
    RUNS_DIR,
    SPLIT_SEED,
    TEST_SIZE,
    TOP_FEATURES,
    Y_MAX,
    Y_MIN,
    _git_state,
    _sha256,
    load_split,
)
from runs_io import lies_summary  # noqa: E402

CV_SEED = 42


def _kennzahlen(y_true, y_pred):
    return (
        mean_absolute_error(y_true, y_pred),
        float(np.sqrt(mean_squared_error(y_true, y_pred))),
        r2_score(y_true, y_pred),
    )


def lauf(model_name: str, mache_modell, params_source: str, modell_params: dict):
    """Ein vollstaendiger Lauf: Test-Set, CV, Plot, Run-JSON, Uebersichtszeile."""
    ds = load_split(verbose=True)
    y = ds.y.reset_index(drop=True)
    batch_ids = ds.batch_ids
    test_mask = ds.test_mask

    print("═" * 64)
    print(f" ROHKURVEN → TENSOR  ({len(RK.CHANNELS)} Kanäle × {RK.SERIES_LEN} Punkte)")
    print("═" * 64)
    X3d = RK.build_or_load_tensor(batch_ids)

    print("\n" + "═" * 64)
    print(f" MODELL: {model_name}   |   {params_source}")
    print("═" * 64)

    tr, te = np.where(~test_mask)[0], np.where(test_mask)[0]
    y_test = y.iloc[te].to_numpy()
    modell = mache_modell().fit(X3d[tr], y.iloc[tr].to_numpy())
    y_pred = modell.predict(X3d[te])
    mae, rmse, r2 = _kennzahlen(y_test, y_pred)

    flach_tr = X3d[tr].reshape(len(tr), -1)
    flach_te = X3d[te].reshape(len(te), -1)
    base = DummyRegressor(strategy="mean").fit(flach_tr, y.iloc[tr])
    base_mae, base_rmse, base_r2 = _kennzahlen(y_test, base.predict(flach_te))
    base_med = DummyRegressor(strategy="median").fit(flach_tr, y.iloc[tr])
    base_med_mae, base_med_rmse, base_med_r2 = _kennzahlen(
        y_test, base_med.predict(flach_te)
    )

    print("\n── Testergebnisse (eingefrorener Split) ────────────────────")
    print(f"  MAE  : {mae:.4f}   RMSE : {rmse:.4f}   R² : {r2:.4f}")
    print(f"  Baseline(median): MAE {base_med_mae:.4f}   R² {base_med_r2:.4f}")
    print("────────────────────────────────────────────────────────────\n")

    cv_folds, cv_level = data_prep.make_cv(ds.df, seed=CV_SEED)
    scores = {
        model_name: {"r2": [], "mae": []},
        "Baseline (Mittel)": {"r2": [], "mae": []},
        "Baseline (Median)": {"r2": [], "mae": []},
    }
    oof = np.full(len(y), np.nan)
    for i, (itr, iva) in enumerate(cv_folds, 1):
        ytr, yva = y.iloc[itr].to_numpy(), y.iloc[iva].to_numpy()
        pred = mache_modell().fit(X3d[itr], ytr).predict(X3d[iva])
        oof[iva] = pred
        scores[model_name]["r2"].append(r2_score(yva, pred))
        scores[model_name]["mae"].append(mean_absolute_error(yva, pred))
        for nm, strat in (
            ("Baseline (Mittel)", "mean"),
            ("Baseline (Median)", "median"),
        ):
            b = DummyRegressor(strategy=strat).fit(np.zeros((len(itr), 1)), ytr)
            bp = b.predict(np.zeros((len(iva), 1)))
            scores[nm]["r2"].append(r2_score(yva, bp))
            scores[nm]["mae"].append(mean_absolute_error(yva, bp))
        print(
            f"  Falte {i}/{len(cv_folds)}:  R² = {scores[model_name]['r2'][-1]:+.4f}   "
            f"MAE = {scores[model_name]['mae'][-1]:.4f}"
        )

    cv = {
        nm: {
            "r2_mean": float(np.mean(s["r2"])),
            "r2_std": float(np.std(s["r2"])),
            "mae_mean": float(np.mean(s["mae"])),
            "mae_std": float(np.std(s["mae"])),
            "r2_folds": [float(v) for v in s["r2"]],
            "mae_folds": [float(v) for v in s["mae"]],
        }
        for nm, s in scores.items()
    }
    oof_r2 = float(r2_score(y, oof))
    oof_mae = float(mean_absolute_error(y, oof))
    print(
        f"\n  CV ({cv_level}): MAE {cv[model_name]['mae_mean']:.4f} "
        f"± {cv[model_name]['mae_std']:.4f}   R² {cv[model_name]['r2_mean']:+.4f}"
    )
    print(f"  Baseline (Median): MAE {cv['Baseline (Median)']['mae_mean']:.4f}\n")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{model_name} – y aus Rohkurven (OOF, {cv_level})", fontsize=14)
    ax[0].scatter(y, oof, alpha=0.5, edgecolors="none", s=30)
    lims = [min(y.min(), np.nanmin(oof)), max(y.max(), np.nanmax(oof))]
    ax[0].plot(lims, lims, "r--", lw=1)
    ax[0].set_xlabel("Tatsächlich")
    ax[0].set_ylabel("Vorhergesagt (out-of-fold)")
    ax[0].set_title(f"Predicted vs. Actual  (OOF R²={oof_r2:.3f})")
    ax[1].scatter(oof, y.to_numpy() - oof, alpha=0.5, edgecolors="none", s=30)
    ax[1].axhline(0, color="r", ls="--", lw=1)
    ax[1].set_xlabel("Vorhergesagt (out-of-fold)")
    ax[1].set_ylabel("Residuum")
    ax[1].set_title("Residuen (out-of-fold)")
    plt.tight_layout()
    png = data_prep.ROOT / "reports" / "figures" / f"{model_name.lower()}_results.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"Plot gespeichert: {png}")

    RUNS_DIR.mkdir(exist_ok=True)
    RUN_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    train_ids = batch_ids[~test_mask].tolist()
    test_ids = batch_ids[test_mask].tolist()
    n_feat = len(RK.CHANNELS) * RK.SERIES_LEN

    record = {
        "run_id": run_id,
        "model": model_name,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "git": _git_state(),
        "data": {
            "csv": str(CSV_PATH.relative_to(data_prep.ROOT)),
            "sha256": _sha256(CSV_PATH),
            "n_rows_after_filter": int(len(ds.df)),
            "billet_dir": str(RK.BILLET_DIR),
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
            "channels": RK.CHANNELS,
            "series_len": RK.SERIES_LEN,
            "znorm": RK.ZNORM,
        },
        "model_params": modell_params,
        "split": {
            "n_train": len(train_ids),
            "n_test": len(test_ids),
            "train_ids": train_ids,
            "test_ids": test_ids,
        },
        "metrics": {
            "test": {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)},
            "baseline_mean": {"mae": base_mae, "rmse": base_rmse, "r2": base_r2},
            "baseline_median": {
                "mae": base_med_mae,
                "rmse": base_med_rmse,
                "r2": base_med_r2,
            },
            "cv": cv,
            "cv_level": cv_level,
            "oof_r2": oof_r2,
            "oof_mae": oof_mae,
        },
        "predictions": {
            "batch_ids": [str(b) for b in batch_ids],
            "y_true": [round(float(v), 6) for v in y],
            "oof_pred": [round(float(v), 6) for v in oof],
            "test_batch_ids": [str(b) for b in test_ids],
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
        "model": model_name,
        "timestamp": record["timestamp"],
        "git_commit": (record["git"]["commit"] or "")[:8],
        "git_dirty": record["git"]["dirty"],
        "data_sha256": record["data"]["sha256"][:12],
        "n_rows": record["data"]["n_rows_after_filter"],
        "only_category": ONLY_CATEGORY,
        "only_glastyp": ONLY_GLASTYP,
        "only_machine": ONLY_MACHINE,
        "test_level": data_prep.TEST_LEVEL,
        "cv_level": cv_level,
        "cv_ausgefuehrt": True,
        "feature_source": data_prep.FEATURE_SOURCE,
        "categorical_encoding": data_prep.CATEGORICAL_ENCODING,
        "max_cardinality": data_prep.MAX_CARDINALITY,
        "ziel": data_prep.ZIEL,
        "y_min": Y_MIN,
        "y_max": Y_MAX,
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "n_features": n_feat,
        "params_source": params_source,
        "best_iteration": None,
        "test_mae": round(float(mae), 4),
        "test_rmse": round(float(rmse), 4),
        "test_r2": round(float(r2), 4),
        "baseline_mittel_mae": round(base_mae, 4),
        "baseline_mittel_rmse": round(base_rmse, 4),
        "baseline_mittel_r2": round(base_r2, 4),
        "baseline_median_mae": round(base_med_mae, 4),
        "baseline_median_rmse": round(base_med_rmse, 4),
        "baseline_median_r2": round(base_med_r2, 4),
        "cv_r2_mean": round(cv[model_name]["r2_mean"], 4),
        "cv_r2_std": round(cv[model_name]["r2_std"], 4),
        "cv_mae_mean": round(cv[model_name]["mae_mean"], 4),
        "cv_mae_std": round(cv[model_name]["mae_std"], 4),
        "cv_baseline_mittel_r2_mean": round(cv["Baseline (Mittel)"]["r2_mean"], 4),
        "cv_baseline_mittel_mae_mean": round(cv["Baseline (Mittel)"]["mae_mean"], 4),
        "cv_baseline_median_r2_mean": round(cv["Baseline (Median)"]["r2_mean"], 4),
        "cv_baseline_median_mae_mean": round(cv["Baseline (Median)"]["mae_mean"], 4),
        "oof_r2": round(oof_r2, 4),
    }
    pfad = RUNS_DIR / "runs_summary.csv"
    neu = pd.DataFrame([summary_row])
    if pfad.exists():
        vorher = lies_summary(pfad)
        df = pd.concat([vorher, neu], ignore_index=True)
        cols = list(summary_row) + [c for c in df.columns if c not in summary_row]
        df = df[cols]
    else:
        df = neu
    df.to_csv(pfad, index=False)
    plt.close(fig)
    print(f"Run protokolliert: {run_json.relative_to(data_prep.ROOT)}")
    print(f"Übersicht aktualisiert: {pfad.relative_to(data_prep.ROOT)}")
    return record
