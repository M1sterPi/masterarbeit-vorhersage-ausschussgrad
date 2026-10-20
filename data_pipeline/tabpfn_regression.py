"""TabPFN-Regression auf der Merkmalstabelle (vortrainiert, ohne Tuning).

In der Arbeit: TabPFN-Läufe für Abschnitt 6.4 (Abbildungen 12 und 13) und die
Gegenprobe in Kapitel 7.

Ebene über TEST_LEVEL und CV_LEVEL, nutzt die GPU, wenn vorhanden. Schreibt
runs/single/, runs/runs_summary.csv und reports/figures/tabpfn_results.png.

    python data_pipeline/tabpfn_regression.py
"""

import json
import datetime
import numpy as np
import pandas as pd

from runs_io import lies_summary
import matplotlib.pyplot as plt
from tabpfn import TabPFNRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor

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
    _fmt_cols,
    _sha256,
    _git_state,
)

MODEL_NAME = "TabPFN"


# Wird von anderen Skripten per ast gelesen: als literales dict(...) belassen.
DEFAULT_PARAMS = dict(
    n_estimators=8,
    softmax_temperature=0.9,
)
FIXED_PARAMS = dict(
    device="auto",
    random_state=42,
    ignore_pretraining_limits=False,
)


ds = load_split(verbose=True)
df = ds.df
X_raw = ds.X_raw
y = ds.y
batch_ids = ds.batch_ids
test_mask = ds.test_mask
X_train_raw = ds.X_train_raw
X_test_raw = ds.X_test_raw
y_train = ds.y_train
y_test = ds.y_test
leakage_cols = ds.leakage_cols
id_cols = ds.id_cols

print("═" * 64)
print(" FEATURE-VERARBEITUNG  (Typ-Entscheidung & Fit nur auf Train)")
print("═" * 64)
X_train, X_test, feat_info = process_features(X_train_raw, X_test_raw, verbose=True)
missing_frac = X_train.isnull().mean()
high_missing = missing_frac[missing_frac > 0.5].index.tolist()
print(f"\n  Hinweis – Spalten mit >50% fehlenden Werten: {len(high_missing)}")
if high_missing:
    print(_fmt_cols(high_missing))
print()

n_features = X_train.shape[1]

model_params = dict(DEFAULT_PARAMS)
params_source = "default (TabPFN ist vortrainiert; kein klassisches Tuning)"
print("═" * 64)
print(f" MODELL: {MODEL_NAME}   |   HYPERPARAMETER: {params_source}")
print("═" * 64)
for k, v in {**model_params, **FIXED_PARAMS}.items():
    print(f"  {k:<26}{v}")
print()

model = TabPFNRegressor(**model_params, **FIXED_PARAMS)
model.fit(X_train, y_train)

tabpfn_limits = {
    "max_features": int(model.inference_config_.MAX_NUMBER_OF_FEATURES),
    "max_samples": int(model.inference_config_.MAX_NUMBER_OF_SAMPLES),
    "n_features_verwendet": int(n_features),
    "n_train_verwendet": int(len(X_train)),
}
print(
    f"  TabPFN-Grenzen: {tabpfn_limits['n_features_verwendet']} von "
    f"{tabpfn_limits['max_features']} Merkmalen, "
    f"{tabpfn_limits['n_train_verwendet']} von "
    f"{tabpfn_limits['max_samples']} Zeilen\n"
)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n── Testergebnisse ──────────────────────────────────────────")
print(f"  MAE  : {mae:.4f}")
print(f"  RMSE : {rmse:.4f}")
print(f"  R²   : {r2:.4f}")
print("────────────────────────────────────────────────────────────\n")

baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)
y_base = baseline.predict(X_test)

base_mae = mean_absolute_error(y_test, y_base)
base_rmse = np.sqrt(mean_squared_error(y_test, y_base))
base_r2 = r2_score(y_test, y_base)
base_med = DummyRegressor(strategy="median").fit(X_train, y_train)
y_base_med = base_med.predict(X_test)
base_med_mae = mean_absolute_error(y_test, y_base_med)
base_med_rmse = np.sqrt(mean_squared_error(y_test, y_base_med))
base_med_r2 = r2_score(y_test, y_base_med)

print("── Baseline (Mittelwert-Vorhersage) ─────────────────────────")
print(
    f"  Vorhergesagter Wert (mean y_train): {float(np.ravel(baseline.constant_)[0]):.4f}"
)
print(f"  MAE  : {base_mae:.4f}")
print(f"  RMSE : {base_rmse:.4f}")
print(f"  R²   : {base_r2:.4f}")
print("────────────────────────────────────────────────────────────")
print(f"── Verbesserung {MODEL_NAME} ggü. Baseline ───────────────────")
print(f"  MAE  : {(1 - mae / base_mae) * 100:5.1f} %  geringer")
print(f"  RMSE : {(1 - rmse / base_rmse) * 100:5.1f} %  geringer")
print("────────────────────────────────────────────────────────────\n")

CV_SEED = 42

RUN_CV = True

if not RUN_CV:
    print("═" * 64)
    print(" SCHNELLMODUS: Kreuzvalidierung ABGESCHALTET (RUN_CV = False)")
    print(" Nur Test-Set-Auswertung. Diese Zahlen sind NICHT berichtbar —")
    print(" eine Einzelziehung streut zu stark (siehe Testkonzept 20.08.2026).")
    print("═" * 64 + "\n")

if RUN_CV:
    cv_folds, cv_level = data_prep.make_cv(df, seed=CV_SEED)
    cv_folds_random = (
        data_prep.make_cv(df, level="random", seed=CV_SEED)[0]
        if data_prep.CV_ALSO_RANDOM and cv_level != "random"
        else None
    )
else:
    cv_folds, cv_folds_random = [], None
    cv_level = data_prep.CV_LEVEL

X_cv = X_raw.reset_index(drop=True)
y_cv = y.reset_index(drop=True)

scores = {
    MODEL_NAME: {"r2": [], "mae": []},
    "Baseline (Mittel)": {"r2": [], "mae": []},
    "Baseline (Median)": {"r2": [], "mae": []},
}

oof_pred = np.full(len(y_cv), np.nan)

for tr_idx, va_idx in cv_folds:
    Xtr_raw, Xva_raw = X_cv.iloc[tr_idx], X_cv.iloc[va_idx]
    ytr, yva = y_cv.iloc[tr_idx], y_cv.iloc[va_idx]
    Xtr, Xva, _ = process_features(Xtr_raw, Xva_raw)

    fold_models = {
        MODEL_NAME: TabPFNRegressor(**model_params, **FIXED_PARAMS),
        "Baseline (Mittel)": DummyRegressor(strategy="mean"),
        "Baseline (Median)": DummyRegressor(strategy="median"),
    }
    for name, est in fold_models.items():
        est.fit(Xtr, ytr)
        pred = est.predict(Xva)
        scores[name]["r2"].append(r2_score(yva, pred))
        scores[name]["mae"].append(mean_absolute_error(yva, pred))
        if name == MODEL_NAME:
            oof_pred[va_idx] = pred

if RUN_CV:
    print(
        f"── 5-Fold CV: {MODEL_NAME} vs. Baselines (leakage-frei, identische Folds) ─"
    )
    print(f"  {'Modell':<20}{'R²':>16}{'MAE':>16}")
    print("  " + "─" * 50)
    for name, s in scores.items():
        r2a = np.array(s["r2"])
        maea = np.array(s["mae"])
        print(
            f"  {name:<20}"
            f"{r2a.mean():>8.4f} ± {r2a.std():.4f}"
            f"{maea.mean():>8.4f} ± {maea.std():.4f}"
        )
    print("────────────────────────────────────────────────────────────\n")

if RUN_CV:
    oof_r2 = r2_score(y_cv, oof_pred)
    oof_residuals = y_cv - oof_pred
    p_wahr, p_vorher, p_resid = y_cv, oof_pred, oof_residuals
    p_quelle, p_r2 = "out-of-fold", oof_r2
    titel = f"{MODEL_NAME} Regression – Vorhersage von y (5-Fold CV)"
else:
    oof_r2 = oof_pred = oof_residuals = None
    p_wahr, p_vorher = y_test, y_pred
    p_resid, p_quelle, p_r2 = y_test - y_pred, "Test-Set", r2
    titel = (
        f"{MODEL_NAME} Regression – Vorhersage von y "
        "(SCHNELLMODUS: nur Test-Set, keine CV)"
    )

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(titel, fontsize=14)

axes[0].scatter(p_wahr, p_vorher, alpha=0.5, edgecolors="none", s=30)
lims = [min(min(p_wahr), min(p_vorher)), max(max(p_wahr), max(p_vorher))]
axes[0].plot(lims, lims, "r--", linewidth=1)
axes[0].set_xlabel("Tatsächlich")
axes[0].set_ylabel(f"Vorhergesagt ({p_quelle})")
axes[0].set_title(f"Predicted vs. Actual  ({p_quelle} R²={p_r2:.3f})")

axes[1].scatter(p_vorher, p_resid, alpha=0.5, edgecolors="none", s=30)
axes[1].axhline(0, color="r", linestyle="--", linewidth=1)
axes[1].set_xlabel(f"Vorhergesagt ({p_quelle})")
axes[1].set_ylabel("Residuum")
axes[1].set_title(f"Residuen ({p_quelle})")

plt.tight_layout()
out_path = data_prep.ROOT / "reports" / "figures" / "tabpfn_results.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Plot gespeichert: {out_path}")

RUNS_DIR.mkdir(exist_ok=True)
RUN_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

train_ids = batch_ids[~test_mask].tolist()
test_ids_present = batch_ids[test_mask].tolist()
ensemble_size = int(model_params["n_estimators"])

cv_summary = (
    {
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
    if RUN_CV
    else None
)


def _cv(name, feld, stellen=4):
    """Kennzahl aus cv_summary, oder None im Schnellmodus."""
    return None if cv_summary is None else round(cv_summary[name][feld], stellen)


record = {
    "run_id": run_id,
    "model": MODEL_NAME,
    "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    "git": _git_state(),
    "data": {
        "csv": str(CSV_PATH.relative_to(data_prep.ROOT)),
        "sha256": _sha256(CSV_PATH),
        "n_rows_after_filter": int(len(df)),
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
        "id_features_dropped": id_cols if DROP_ID_FEATURES else [],
        "top_features": list(TOP_FEATURES) if TOP_FEATURES else None,
        "leakage_cols_dropped": leakage_cols,
        "params_source": params_source,
    },
    "model_params": {**model.get_params(), "ensemble_size": ensemble_size},
    "tabpfn_limits": tabpfn_limits,
    "split": {
        "n_train": int(len(train_ids)),
        "n_test": int(len(test_ids_present)),
        "train_ids": train_ids,
        "test_ids": test_ids_present,
    },
    "features": {
        "n_used": int(X_train.shape[1]),
        "used": X_train.columns.tolist(),
        "numeric": feat_info["numeric_cols"],
        "converted_de": feat_info["converted_de_cols"],
        "encoded": feat_info["encoded_cols"],
        "dropped_highcard": feat_info["dropped_highcard_cols"],
        "dropped_all_nan": feat_info["all_nan_cols"],
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
        "cv_ausgefuehrt": bool(RUN_CV),
        "oof_r2": float(oof_r2) if RUN_CV else None,
    },
    "predictions": {
        "batch_ids": [str(b) for b in batch_ids],
        "y_true": [round(float(v), 6) for v in y_cv],
        "oof_pred": ([round(float(v), 6) for v in oof_pred] if RUN_CV else None),
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
    "n_features": record["features"]["n_used"],
    "params_source": params_source,
    "best_iteration": ensemble_size,
    "test_mae": round(float(mae), 4),
    "test_rmse": round(float(rmse), 4),
    "test_r2": round(float(r2), 4),
    "baseline_mittel_mae": round(float(base_mae), 4),
    "baseline_mittel_rmse": round(float(base_rmse), 4),
    "baseline_mittel_r2": round(float(base_r2), 4),
    "baseline_median_mae": round(float(base_med_mae), 4),
    "baseline_median_rmse": round(float(base_med_rmse), 4),
    "baseline_median_r2": round(float(base_med_r2), 4),
    "cv_ausgefuehrt": bool(RUN_CV),
    "cv_r2_mean": _cv(MODEL_NAME, "r2_mean"),
    "cv_r2_std": _cv(MODEL_NAME, "r2_std"),
    "cv_mae_mean": _cv(MODEL_NAME, "mae_mean"),
    "cv_mae_std": _cv(MODEL_NAME, "mae_std"),
    "cv_baseline_mittel_r2_mean": _cv("Baseline (Mittel)", "r2_mean"),
    "cv_baseline_mittel_mae_mean": _cv("Baseline (Mittel)", "mae_mean"),
    "cv_baseline_median_r2_mean": _cv("Baseline (Median)", "r2_mean"),
    "cv_baseline_median_mae_mean": _cv("Baseline (Median)", "mae_mean"),
    "oof_r2": round(float(oof_r2), 4) if RUN_CV else None,
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

plt.show()
