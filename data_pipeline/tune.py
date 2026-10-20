"""Hyperparameter-Suche für XGBoost und CatBoost mit Optuna (TPE).

In der Arbeit: Hyperparameter-Suche aus Abschnitt 6.2: Ergebnis in Tabelle 11
(XGBoost) und Tabelle A.1 (CatBoost).

Gesucht wird nur auf dem Trainingsteil, mit gruppenreinen Falten der Ebene
CV_LEVEL. Ergebnis: data/best_params_v3.json bzw. data/catboost_best_params.json,
Suchverlauf in runs/tuning/.

    python data_pipeline/tune.py --model {xgboost,catboost} [--trials N] [--ziel {mae,r2}]
"""

from __future__ import annotations

import argparse
import datetime
import json

import numpy as np
import optuna
from sklearn.metrics import mean_absolute_error, r2_score

import data_prep
from data_prep import (
    BEST_PARAMS_PATH,
    CSV_PATH,
    SPLIT_SEED,
    _git_state,
    _sha256,
    load_split,
    process_features,
)

FIXED_PARAMS = dict(random_state=42, n_jobs=-1)
CATBOOST_FIXED = dict(random_seed=42, loss_function="RMSE", allow_writing_files=False)

ZIELDATEI = {
    "xgboost": BEST_PARAMS_PATH,
    "catboost": data_prep.ROOT / "data" / "catboost_best_params.json",
}

N_SPLITS = 5
TUNE_SEED = 42

STUDY_DIR = data_prep.ROOT / "runs" / "tuning"


def suchraum(trial, modell: str = "xgboost") -> dict:
    """Der Suchraum, EINE Definition fuer alle Aufrufer."""
    if modell == "xgboost":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 800),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            max_depth=trial.suggest_int("max_depth", 2, 6),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            **FIXED_PARAMS,
        )
    return dict(
        iterations=trial.suggest_int("iterations", 200, 800),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        depth=trial.suggest_int("depth", 2, 6),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-1, 30.0, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        rsm=trial.suggest_float("rsm", 0.6, 1.0),
        min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 1, 20),
        **CATBOOST_FIXED,
    )


STARTPUNKT = {
    "xgboost": dict(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
    ),
    "catboost": dict(
        iterations=800,
        learning_rate=0.05,
        depth=4,
        l2_leaf_reg=3.0,
        subsample=0.8,
        rsm=0.8,
        min_data_in_leaf=5,
    ),
}


def baue_modell(modell: str, params: dict):
    if modell == "xgboost":
        import xgboost as xgb

        return xgb.XGBRegressor(**params)
    from catboost import CatBoostRegressor

    return CatBoostRegressor(**params, verbose=0)


def make_objective(modell, X_train_raw, y_train, folds, ziel):
    """Optuna-Zielfunktion: mittlere Kennzahl ueber die gruppenweisen Falten."""

    def objective(trial: optuna.Trial) -> float:
        params = suchraum(trial, modell)
        werte = []
        for tr_idx, va_idx in folds:
            Xtr_raw = X_train_raw.iloc[tr_idx]
            Xva_raw = X_train_raw.iloc[va_idx]
            ytr, yva = y_train.iloc[tr_idx], y_train.iloc[va_idx]
            Xtr, Xva, _ = process_features(Xtr_raw, Xva_raw)
            p = baue_modell(modell, params).fit(Xtr, ytr)
            v = p.predict(Xva)
            werte.append(
                mean_absolute_error(yva, v) if ziel == "mae" else r2_score(yva, v)
            )
        return float(np.mean(werte))

    return objective


def train_folds(ds):
    """Gruppenweise Falten AUF DEN TRAININGSZEILEN."""
    df_tr = ds.df[~ds.test_mask].reset_index(drop=True)
    folds, level = data_prep.make_cv(df_tr, n_splits=N_SPLITS, seed=TUNE_SEED)
    return folds, level


def main():
    p = argparse.ArgumentParser(description="Hyperparameter-Tuning (Optuna)")
    p.add_argument("--model", choices=sorted(ZIELDATEI), default="xgboost")
    p.add_argument("--trials", type=int, default=100, help="Anzahl Optuna-Trials")
    p.add_argument(
        "--ziel",
        choices=["mae", "r2"],
        default="mae",
        help="Zielgroesse; mae entspricht der Hauptkennzahl der Arbeit",
    )
    p.add_argument("--out", help="abweichender Ausgabepfad (Voreinstellung je Modell)")
    p.add_argument(
        "--study-name",
        help="Name der Optuna-Study in der SQLite-Datei; gleicher Name setzt "
        "einen abgebrochenen Lauf fort (Voreinstellung: modell_ziel_cvebene)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="vorhandene Ergebnisdatei ueberschreiben (sonst Abbruch)",
    )
    args = p.parse_args()

    ziel_pfad = data_prep.ROOT / args.out if args.out else ZIELDATEI[args.model]
    if ziel_pfad.exists() and not args.overwrite:
        stamm = ziel_pfad.stem
        n = 2
        if "_v" in stamm and stamm.rsplit("_v", 1)[1].isdigit():
            stamm, n = stamm.rsplit("_v", 1)[0], int(stamm.rsplit("_v", 1)[1]) + 1
        raise SystemExit(
            f"{ziel_pfad.name} existiert bereits und wird nicht überschrieben.\n"
            f"  Neues Ergebnis danebenlegen:\n"
            f"    --out data/{stamm}_v{n}.json\n"
            f"  Oder bewusst ersetzen:\n"
            f"    --overwrite"
        )

    ds = load_split(verbose=True)
    X_train_raw = ds.X_train_raw.reset_index(drop=True)
    y_train = ds.y_train.reset_index(drop=True)
    folds, cv_level = train_folds(ds)

    richtung = "minimize" if args.ziel == "mae" else "maximize"
    print("═" * 64)
    print(f" HYPERPARAMETER-TUNING {args.model.upper()} ({args.trials} Trials)")
    print(
        f" Optimiert: mittlere {args.ziel.upper()} über {len(folds)} gruppenweise "
        f"Falten (Ebene {cv_level!r})"
    )
    print(f" NUR auf Train ({len(X_train_raw)} Billets), Test-Set unangetastet.")
    print("═" * 64)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    study_name = args.study_name or f"{args.model}_{args.ziel}_{cv_level}"
    db = STUDY_DIR / f"tune_{args.model}.db"
    study = optuna.create_study(
        direction=richtung,
        sampler=optuna.samplers.TPESampler(seed=TUNE_SEED),
        storage=f"sqlite:///{db.as_posix()}",
        study_name=study_name,
        load_if_exists=True,
    )
    schon = len(study.trials)
    if schon:
        print(f" Fortsetzung: {schon} Trials liegen bereits in {db.name} vor.")
    else:
        study.enqueue_trial(STARTPUNKT[args.model])

    def _fortschritt(study, trial):
        print(
            f"  Trial {trial.number:>3}: {args.ziel.upper()}={trial.value:+.4f}   "
            f"(bisher bestes: {study.best_value:+.4f})",
            flush=True,
        )

    offen = max(args.trials - schon, 0)
    if not offen:
        print(f" Nichts zu tun: {schon} Trials vorhanden, --trials {args.trials}.")
    else:
        study.optimize(
            make_objective(args.model, X_train_raw, y_train, folds, args.ziel),
            n_trials=offen,
            callbacks=[_fortschritt],
        )

    print("\n── Bestes Ergebnis ─────────────────────────────────────────")
    print(
        f"  CV-{args.ziel.upper()} (Mittel über {len(folds)} Falten, nur Train): "
        f"{study.best_value:+.4f}"
    )
    for k, v in study.best_params.items():
        print(f"    {k:<20}{v}")
    print("────────────────────────────────────────────────────────────")

    fix = FIXED_PARAMS if args.model == "xgboost" else CATBOOST_FIXED
    record = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "git": _git_state(),
        "data": {"csv": str(CSV_PATH.name), "sha256": _sha256(CSV_PATH)},
        "tuning": {
            "n_trials": args.trials,
            "n_splits": len(folds),
            "cv_level": cv_level,
            "tune_seed": TUNE_SEED,
            "split_seed": SPLIT_SEED,
            "n_train": int(len(X_train_raw)),
            "n_features_raw": int(X_train_raw.shape[1]),
            "y_min": data_prep.Y_MIN,
            "y_max": data_prep.Y_MAX,
            "objective": f"mean_cv_{args.ziel}_train_only_group_{cv_level}",
            "best_value": float(study.best_value),
            "n_trials_vorhanden": len(study.trials),
            "study": f"{db.name}::{study_name}",
        },
        "params": {**study.best_params, **fix},
    }
    ziel = ziel_pfad
    ziel.parent.mkdir(parents=True, exist_ok=True)
    ziel.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nGespeichert: {ziel}")
    print(
        f"→ {args.model}_regression.py verwendet diese Parameter ab dem nächsten Lauf."
    )


if __name__ == "__main__":
    main()
