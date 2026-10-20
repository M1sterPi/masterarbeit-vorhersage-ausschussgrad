"""CatBoost-Regression auf der Merkmalstabelle.

In der Arbeit: CatBoost-Läufe für Abschnitt 6.4 (Abbildungen 12 und 13) und die
Gegenprobe in Kapitel 7.

Daten, Testmenge und Merkmale kommen aus data_prep, die Parameter aus
data/catboost_best_params.json (tune.py). Ebene über TEST_LEVEL und CV_LEVEL.
Schreibt runs/single/run_<id>.json, eine Zeile in runs/runs_summary.csv und
reports/figures/catboost_results.png.

    python data_pipeline/catboost_regression.py
"""

import json
import datetime
import numpy as np
import pandas as pd

from runs_io import lies_summary
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
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

MODEL_NAME = "CatBoost"

# Wird von xai/_common.py per ast gelesen: als literale Ausdrücke belassen.
CATBOOST_BEST_PARAMS_PATH = data_prep.ROOT / "data" / "catboost_best_params.json"

DEFAULT_PARAMS = dict(
    iterations=1000,
    learning_rate=0.05,
    depth=4,
    l2_leaf_reg=3.0,
    subsample=0.8,
    rsm=0.8,
    min_data_in_leaf=5,
)
FIXED_PARAMS = dict(
    random_seed=42,
    loss_function="RMSE",
    allow_writing_files=False,
)


def load_model_params():
    """DEFAULT_PARAMS oder, falls vorhanden, die getunten CatBoost-Parameter."""
    if CATBOOST_BEST_PARAMS_PATH.exists():
        rec = json.loads(CATBOOST_BEST_PARAMS_PATH.read_text(encoding="utf-8"))
        tuned = rec.get("params", rec)
        params = {**DEFAULT_PARAMS, **tuned}
        for k in FIXED_PARAMS:
            params.pop(k, None)
        return params, f"getunt ({CATBOOST_BEST_PARAMS_PATH.name})"
    return dict(DEFAULT_PARAMS), "default (kein Tuning-Ergebnis)"


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

model_params, params_source = load_model_params()
print("═" * 64)
print(f" MODELL: {MODEL_NAME}   |   HYPERPARAMETER: {params_source}")
print("═" * 64)
for k, v in model_params.items():
    print(f"  {k:<18}{v}")
print()

model = CatBoostRegressor(**model_params, **FIXED_PARAMS)
model.fit(X_train, y_train, verbose=False)

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

cv_folds, cv_level = data_prep.make_cv(df, seed=CV_SEED)
cv_folds_random = (
    data_prep.make_cv(df, level="random", seed=CV_SEED)[0]
    if data_prep.CV_ALSO_RANDOM and cv_level != "random"
    else None
)
n_best = int(model.tree_count_)

cv_params = {**model_params, **FIXED_PARAMS}

X_cv = X_raw.reset_index(drop=True)
y_cv = y.reset_index(drop=True)

scores = {
    MODEL_NAME: {"r2": [], "mae": []},
    "Baseline (Mittel)": {"r2": [], "mae": []},
    "Baseline (Median)": {"r2": [], "mae": []},
}

oof_pred = np.full(len(y_cv), np.nan)
importances_per_fold = []

for tr_idx, va_idx in cv_folds:
    Xtr_raw, Xva_raw = X_cv.iloc[tr_idx], X_cv.iloc[va_idx]
    ytr, yva = y_cv.iloc[tr_idx], y_cv.iloc[va_idx]
    Xtr, Xva, _ = process_features(Xtr_raw, Xva_raw)

    fold_models = {
        MODEL_NAME: CatBoostRegressor(**cv_params, verbose=0),
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
            importances_per_fold.append(
                pd.Series(est.feature_importances_, index=Xtr.columns)
            )

print(f"── 5-Fold CV: {MODEL_NAME} vs. Baselines (leakage-frei, identische Folds) ─")
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

oof_r2 = r2_score(y_cv, oof_pred)
oof_residuals = y_cv - oof_pred

imp_df = pd.concat(importances_per_fold, axis=1).fillna(0.0)
imp_mean = imp_df.mean(axis=1)
imp_std = imp_df.std(axis=1)
top_mean = imp_mean.sort_values(ascending=True).tail(20)
top_std = imp_std.reindex(top_mean.index)

imp_full = imp_mean.sort_values(ascending=False)
std_full = imp_std.reindex(imp_full.index)
print("── Feature Importance: vollständige Rangliste (CV-Mittel ± std) ──────")
print(f"  {'#':>3}  {'Feature':<45}{'Importance':>12}{'± std':>12}")
print("  " + "─" * 72)
for rank, (feat, val) in enumerate(imp_full.items(), start=1):
    print(f"  {rank:>3}  {feat:<45}{val:>12.5f}{std_full[feat]:>12.5f}")
print("────────────────────────────────────────────────────────────────────\n")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f"{MODEL_NAME} Regression – Vorhersage von y (5-Fold CV)", fontsize=14)

axes[0].scatter(y_cv, oof_pred, alpha=0.5, edgecolors="none", s=30)
lims = [min(y_cv.min(), oof_pred.min()), max(y_cv.max(), oof_pred.max())]
axes[0].plot(lims, lims, "r--", linewidth=1)
axes[0].set_xlabel("Tatsächlich")
axes[0].set_ylabel("Vorhergesagt (out-of-fold)")
axes[0].set_title(f"Predicted vs. Actual  (OOF R²={oof_r2:.3f})")

axes[1].scatter(oof_pred, oof_residuals, alpha=0.5, edgecolors="none", s=30)
axes[1].axhline(0, color="r", linestyle="--", linewidth=1)
axes[1].set_xlabel("Vorhergesagt (out-of-fold)")
axes[1].set_ylabel("Residuum")
axes[1].set_title("Residuen (out-of-fold)")

top_mean.plot(kind="barh", ax=axes[2], xerr=top_std, capsize=2)
axes[2].set_title("Feature Importance (CV-Mittel ± std, Top 20)")
axes[2].set_xlabel("Importance Score")

plt.tight_layout()
out_path = data_prep.ROOT / "reports" / "figures" / "catboost_results.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Plot gespeichert: {out_path}")

RUNS_DIR.mkdir(exist_ok=True)
RUN_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

train_ids = batch_ids[~test_mask].tolist()
test_ids_present = batch_ids[test_mask].tolist()

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
        "n_rows_after_filter": int(len(df)),
    },
    "config": {
        "only_category": ONLY_CATEGORY,
        "only_glastyp": ONLY_GLASTYP,
        "only_machine": ONLY_MACHINE,
        "ziel": data_prep.ZIEL,
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
    "model_params": {**model.get_params(), "best_iteration": int(n_best)},
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
        "cv_ausgefuehrt": True,
        "oof_r2": float(oof_r2),
    },
    "predictions": {
        "batch_ids": [str(b) for b in batch_ids],
        "y_true": [round(float(v), 6) for v in y_cv],
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
    "cv_ausgefuehrt": True,
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
    "best_iteration": int(n_best),
    "test_mae": round(float(mae), 4),
    "test_rmse": round(float(rmse), 4),
    "test_r2": round(float(r2), 4),
    "baseline_mittel_mae": round(float(base_mae), 4),
    "baseline_mittel_rmse": round(float(base_rmse), 4),
    "baseline_mittel_r2": round(float(base_r2), 4),
    "baseline_median_mae": round(float(base_med_mae), 4),
    "baseline_median_rmse": round(float(base_med_rmse), 4),
    "baseline_median_r2": round(float(base_med_r2), 4),
    "cv_r2_mean": round(cv_summary[MODEL_NAME]["r2_mean"], 4),
    "cv_r2_std": round(cv_summary[MODEL_NAME]["r2_std"], 4),
    "cv_mae_mean": round(cv_summary[MODEL_NAME]["mae_mean"], 4),
    "cv_mae_std": round(cv_summary[MODEL_NAME]["mae_std"], 4),
    "cv_baseline_mittel_r2_mean": round(cv_summary["Baseline (Mittel)"]["r2_mean"], 4),
    "cv_baseline_mittel_mae_mean": round(
        cv_summary["Baseline (Mittel)"]["mae_mean"], 4
    ),
    "cv_baseline_median_r2_mean": round(cv_summary["Baseline (Median)"]["r2_mean"], 4),
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

plt.show()
