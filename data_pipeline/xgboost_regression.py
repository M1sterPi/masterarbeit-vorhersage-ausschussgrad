"""XGBoost-Regression auf der Merkmalstabelle.

In der Arbeit: XGBoost-Läufe für Abschnitt 6.4 (Abbildungen 12 und 13),
Testmenge in Abschnitt 8.1; mit XGB_PARAMS=default die Voreinstellung in
Abbildung 11.

Parameter aus data/best_params_v3.json (tune.py), mit XGB_PARAMS=default die
Bibliotheks-Voreinstellungen. Ebene über TEST_LEVEL und CV_LEVEL. Schreibt
runs/single/run_<id>.json, runs/runs_summary.csv und
reports/figures/xgboost_results.png.

    python data_pipeline/xgboost_regression.py
"""

import json
import os
import datetime
import numpy as np
import pandas as pd

from runs_io import lies_summary
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor

import data_prep
from data_prep import (
    CSV_PATH,
    BEST_PARAMS_PATH,
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

# Wird von anderen Skripten per ast gelesen: als literales dict(...) belassen.
DEFAULT_PARAMS = dict(
)
FIXED_PARAMS = dict(random_state=42, n_jobs=-1)


def load_model_params():
    """DEFAULT_PARAMS oder, falls vorhanden, die getunten Parameter aus tune.py."""
    if os.environ.get("XGB_PARAMS", "getunt") == "default":
        return dict(DEFAULT_PARAMS), "default (kein Tuning-Ergebnis)"
    if BEST_PARAMS_PATH.exists():
        rec = json.loads(BEST_PARAMS_PATH.read_text(encoding="utf-8"))
        tuned = rec.get("params", rec)
        params = {**DEFAULT_PARAMS, **tuned}
        for k in FIXED_PARAMS:
            params.pop(k, None)
        return params, f"getunt ({BEST_PARAMS_PATH.name})"
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
print(f" HYPERPARAMETER: {params_source}")
print("═" * 64)
for k, v in model_params.items():
    print(f"  {k:<18}{v}")
print()

model = xgb.XGBRegressor(**model_params, **FIXED_PARAMS, eval_metric="rmse")
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
print(f"  Vorhergesagter Wert (mean y_train): {baseline.constant_[0][0]:.4f}")
print(f"  MAE  : {base_mae:.4f}")
print(f"  RMSE : {base_rmse:.4f}")
print(f"  R²   : {base_r2:.4f}")
print("────────────────────────────────────────────────────────────")
print("── Verbesserung XGBoost ggü. Baseline ───────────────────────")
print(f"  MAE  : {(1 - mae / base_mae) * 100:5.1f} %  geringer")
print(f"  RMSE : {(1 - rmse / base_rmse) * 100:5.1f} %  geringer")
print("────────────────────────────────────────────────────────────\n")

RUN_NESTED_CV = False
NESTED_TRIALS = 30
NESTED_INNER_SPLITS = 3
CV_SEED = 42

MODEL_NAME = "XGBoost"

RUN_CV = True

if not RUN_CV:
    print("═" * 64)
    print(" SCHNELLMODUS: Kreuzvalidierung ABGESCHALTET (RUN_CV = False)")
    print(" Nur Test-Set-Auswertung. Diese Zahlen sind NICHT berichtbar —")
    print(" eine Einzelziehung streut zu stark (siehe Testkonzept 20.08.2026).")
    if RUN_NESTED_CV:
        print(" Die verschachtelte CV wird mit abgeschaltet, sie baut eigene Falten.")
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

cv_params = {**model_params, **FIXED_PARAMS}

n_best = int(model.get_booster().num_boosted_rounds())

X_cv = X_raw.reset_index(drop=True)
y_cv = y.reset_index(drop=True)

scores = {
    "XGBoost": {"r2": [], "mae": []},
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
        "XGBoost": xgb.XGBRegressor(**cv_params),
        "Baseline (Mittel)": DummyRegressor(strategy="mean"),
        "Baseline (Median)": DummyRegressor(strategy="median"),
    }
    for name, est in fold_models.items():
        est.fit(Xtr, ytr)
        pred = est.predict(Xva)
        scores[name]["r2"].append(r2_score(yva, pred))
        scores[name]["mae"].append(mean_absolute_error(yva, pred))
        if name == "XGBoost":
            oof_pred[va_idx] = pred
            importances_per_fold.append(
                pd.Series(est.feature_importances_, index=Xtr.columns)
            )

if RUN_CV:
    print("── 5-Fold CV: XGBoost vs. Baselines (leakage-frei, identische Folds) ─")
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
    oof_mae = mean_absolute_error(y_cv, oof_pred)
    oof_residuals = y_cv - oof_pred
else:
    oof_r2 = oof_mae = None
    oof_pred = oof_residuals = None

nested_r2 = nested_mae = None
nested_params_je_falte = None
if RUN_NESTED_CV and RUN_CV:
    import optuna
    from sklearn.model_selection import GroupKFold as _GKF

    from tune import suchraum as _suchraum

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("\n── Verschachtelte CV (Tuning je aeusserer Falte neu) ───────")
    print(
        f"   {NESTED_TRIALS} Trials x {NESTED_INNER_SPLITS} innere Falten "
        f"x 5 aeussere = {5 * NESTED_TRIALS * NESTED_INNER_SPLITS + 5} Anpassungen"
    )
    _gcol = data_prep._TEST_LEVEL_GROUPS.get(data_prep.CV_LEVEL)
    _groups = (
        df[_gcol].astype(str).reset_index(drop=True)
        if _gcol
        else pd.Series(range(len(df)), dtype=str)
    )
    _Xall = X_raw.reset_index(drop=True)
    _outer, _ = data_prep.make_cv(df, seed=CV_SEED)

    _oof = np.full(len(y_cv), np.nan)
    nested_params_je_falte = []
    for _k, (_otr, _ova) in enumerate(_outer, 1):
        _Xi = _Xall.iloc[_otr].reset_index(drop=True)
        _yi = y_cv.iloc[_otr].reset_index(drop=True)
        _gi = _groups.iloc[_otr].reset_index(drop=True)
        _inner_folds = list(
            _GKF(NESTED_INNER_SPLITS, shuffle=True, random_state=CV_SEED).split(
                _Xi, _yi, _gi
            )
        )

        def _ziel(trial, _Xi=_Xi, _yi=_yi, _inner_folds=_inner_folds):
            p = _suchraum(trial)
            fehler = []
            for _itr, _iva in _inner_folds:
                _a, _b, _ = process_features(_Xi.iloc[_itr], _Xi.iloc[_iva])
                m = xgb.XGBRegressor(**p).fit(_a, _yi.iloc[_itr])
                fehler.append(mean_absolute_error(_yi.iloc[_iva], m.predict(_b)))
            return float(np.mean(fehler))

        _study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=CV_SEED),
        )
        _study.optimize(_ziel, n_trials=NESTED_TRIALS, show_progress_bar=False)
        _p_k = {**_study.best_params, **FIXED_PARAMS}
        nested_params_je_falte.append(
            {"falte": _k, "inner_mae": round(float(_study.best_value), 4), **_p_k}
        )

        _a, _b, _ = process_features(_Xall.iloc[_otr], _Xall.iloc[_ova])
        _oof[_ova] = xgb.XGBRegressor(**_p_k).fit(_a, y_cv.iloc[_otr]).predict(_b)
        print(
            f"   Falte {_k}: innere MAE {_study.best_value:.4f} | "
            f"max_depth={_p_k['max_depth']} lr={_p_k['learning_rate']:.4f} "
            f"n_est={_p_k['n_estimators']}"
        )

    nested_r2 = float(r2_score(y_cv, _oof))
    nested_mae = float(mean_absolute_error(y_cv, _oof))
    print(f"   → verschachtelt R² {nested_r2:+.4f}  MAE {nested_mae:.4f}")
    print(
        f"   → flach (feste Parameter) R² {oof_r2:+.4f}  "
        f"→ Selektions-Verzerrung {oof_r2 - nested_r2:+.4f}"
    )

if RUN_CV:
    imp_df = pd.concat(importances_per_fold, axis=1).fillna(0.0)
    imp_mean = imp_df.mean(axis=1)
    imp_std = imp_df.std(axis=1)
else:
    imp_mean = pd.Series(model.feature_importances_, index=X_train.columns)
    imp_std = pd.Series(0.0, index=imp_mean.index)
top_mean = imp_mean.sort_values(ascending=True).tail(20)
top_std = imp_std.reindex(top_mean.index)

imp_full = imp_mean.sort_values(ascending=False)
std_full = imp_std.reindex(imp_full.index)
print(
    "── Feature Importance: vollständige Rangliste "
    + ("(CV-Mittel ± std) ──────" if RUN_CV else "(finales Modell, keine CV) ──")
)
print(f"  {'#':>3}  {'Feature':<45}{'Importance':>12}{'± std':>12}")
print("  " + "─" * 72)
for rank, (feat, val) in enumerate(imp_full.items(), start=1):
    print(f"  {rank:>3}  {feat:<45}{val:>12.5f}{std_full[feat]:>12.5f}")
print("────────────────────────────────────────────────────────────────────\n")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

if RUN_CV:
    fig.suptitle("XGBoost Regression – Vorhersage von y (5-Fold CV)", fontsize=14)
    p_wahr, p_vorher, p_resid = y_cv, oof_pred, oof_residuals
    p_quelle, p_r2 = "out-of-fold", oof_r2
else:
    fig.suptitle(
        "XGBoost Regression – Vorhersage von y (SCHNELLMODUS: nur Test-Set, keine CV)",
        fontsize=14,
    )
    p_wahr, p_vorher = y_test, y_pred
    p_resid, p_quelle, p_r2 = y_test - y_pred, "Test-Set", r2

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

top_mean.plot(kind="barh", ax=axes[2], xerr=top_std, capsize=2)
axes[2].set_title(
    "Feature Importance (CV-Mittel ± std, Top 20)"
    if RUN_CV
    else "Feature Importance (finales Modell, Top 20)"
)
axes[2].set_xlabel("Importance Score")

plt.tight_layout()
out_path = data_prep.ROOT / "reports" / "figures" / "xgboost_results.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Plot gespeichert: {out_path}")

RUNS_DIR.mkdir(exist_ok=True)
RUN_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

train_ids = batch_ids[~test_mask].tolist()
test_ids_present = batch_ids[test_mask].tolist()

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


def _cv(pfad, stellen=4):
    """Kennzahl aus cv_summary, oder None im Schnellmodus."""
    if cv_summary is None:
        return None
    name, feld = pfad
    return round(cv_summary[name][feld], stellen)


record = {
    "run_id": run_id,
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
    "model": {**model.get_params(), "best_iteration": int(n_best)},
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
        "oof_mae": float(oof_mae) if RUN_CV else None,
        "nested_cv_r2": nested_r2,
        "nested_cv_mae": nested_mae,
        "nested_params_je_falte": nested_params_je_falte,
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
    "cv_ausgefuehrt": bool(RUN_CV),
    "cv_r2_mean": _cv((MODEL_NAME, "r2_mean")),
    "cv_r2_std": _cv((MODEL_NAME, "r2_std")),
    "cv_mae_mean": _cv((MODEL_NAME, "mae_mean")),
    "cv_mae_std": _cv((MODEL_NAME, "mae_std")),
    "cv_baseline_mittel_r2_mean": _cv(("Baseline (Mittel)", "r2_mean")),
    "cv_baseline_mittel_mae_mean": _cv(("Baseline (Mittel)", "mae_mean")),
    "cv_baseline_median_r2_mean": _cv(("Baseline (Median)", "r2_mean")),
    "cv_baseline_median_mae_mean": _cv(("Baseline (Median)", "mae_mean")),
    "oof_r2": round(float(oof_r2), 4) if RUN_CV else None,
    "nested_cv_r2": round(nested_r2, 4) if nested_r2 is not None else None,
    "nested_cv_mae": round(nested_mae, 4) if nested_mae is not None else None,
    "selection_bias_r2": (
        round(float(oof_r2) - nested_r2, 4)
        if (nested_r2 is not None and RUN_CV)
        else None
    ),
    "selection_bias_mae": (
        round(nested_mae - float(oof_mae), 4)
        if (nested_mae is not None and RUN_CV)
        else None
    ),
    "n_top_features": len(TOP_FEATURES) if TOP_FEATURES else None,
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
