"""Zweistufiges Modell für y: Stufe A schätzt die Wahrscheinlichkeit eines
Totalausfalls, Stufe B den Ausschuss der übrigen Billets; kombiniert
E[y] = p · 100 + (1 − p) · ŷ.

In der Arbeit: nicht berichtet; Vorarbeit zur Frage, ob ein eigenes Modell für
Totalausfälle hilft.

Rechnet nur die Kreuzvalidierung (Falten über CV_LEVEL). Schreibt
runs/single/run_<id>.json, runs/runs_summary.csv und
reports/figures/hurdle_results.png.

    python data_pipeline/hurdle_regression.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runs_io import lies_summary  # noqa: E402
import data_prep  # noqa: E402
from data_prep import (  # noqa: E402
    CSV_PATH,
    RUN_ARTIFACTS_DIR,
    RUNS_DIR,
    _git_state,
    _sha256,
    load_split,
    make_cv,
    process_features,
)

TOTAL_SCHWELLE = 99.999

SEED = 42

PARAMS_A = dict(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
)


def params_b() -> tuple[dict, str]:
    """Getunte Parameter für die Regression, wie in xgboost_regression.py."""
    p = data_prep.BEST_PARAMS_PATH
    if p.exists():
        rec = json.loads(p.read_text(encoding="utf-8"))
        tuned = rec.get("params", rec)
        sauber = {k: v for k, v in tuned.items() if k not in ("random_state", "n_jobs")}
        return sauber, f"getunt ({p.name})"
    return (
        dict(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
        ),
        "default (kein Tuning-Ergebnis)",
    )


def kennzahlen(wahr, vorher) -> dict:
    return {
        "r2": float(r2_score(wahr, vorher)),
        "mae": float(mean_absolute_error(wahr, vorher)),
    }


def main():
    t0 = time.time()
    ds = load_split()
    X = ds.X_raw.reset_index(drop=True)
    y = ds.y.reset_index(drop=True)
    df = ds.df.reset_index(drop=True)
    total = (y >= TOTAL_SCHWELLE).astype(int)

    folds, cv_level = make_cv(df, seed=SEED)
    pb, pb_quelle = params_b()

    print("═" * 68)
    print(" ZWEISTUFIGES MODELL (HURDLE)")
    print("═" * 68)
    print(f"  Zeilen                {len(y)}")
    print(f"  Totalausfälle         {int(total.sum())}  ({100 * total.mean():.2f} %)")
    print(f"  Merkmale (roh)        {X.shape[1]}")
    print(f"  CV-Ebene              {cv_level!r}, {len(folds)} Falten")
    print(f"  y-Filter              [{data_prep.Y_MIN}, {data_prep.Y_MAX}]")
    print(f"  Parameter Stufe B     {pb_quelle}")
    print()

    oof_p = np.full(len(y), np.nan)
    oof_b = np.full(len(y), np.nan)
    oof_h = np.full(len(y), np.nan)
    oof_e = np.full(len(y), np.nan)
    auc_je_falte, ap_je_falte, npos = [], [], []

    for k, (tr, va) in enumerate(folds, 1):
        Xtr, Xva, _ = process_features(X.iloc[tr], X.iloc[va])
        ytr = y.iloc[tr]
        ttr, tva = total.iloc[tr], total.iloc[va]

        klf = xgb.XGBClassifier(
            **PARAMS_A, random_state=SEED, n_jobs=-1, verbosity=0
        ).fit(Xtr, ttr)
        p = klf.predict_proba(Xva)[:, 1]
        oof_p[va] = p
        npos.append(int(tva.sum()))
        if tva.nunique() > 1:
            auc_je_falte.append(float(roc_auc_score(tva, p)))
            ap_je_falte.append(float(average_precision_score(tva, p)))

        ok = (ttr == 0).to_numpy()
        reg = xgb.XGBRegressor(**pb, random_state=SEED, n_jobs=-1, verbosity=0).fit(
            Xtr[ok], ytr[ok]
        )
        b = reg.predict(Xva)
        oof_b[va] = b
        oof_h[va] = p * 100.0 + (1.0 - p) * b

        einzeln = xgb.XGBRegressor(**pb, random_state=SEED, n_jobs=-1, verbosity=0).fit(
            Xtr, ytr
        )
        oof_e[va] = einzeln.predict(Xva)

        print(
            f"  Falte {k}: {int(tva.sum()):3d} Totalausfälle in der Validierung | "
            f"AUC {auc_je_falte[-1] if auc_je_falte else float('nan'):.3f}"
        )

    if np.isnan(oof_h).any():
        raise RuntimeError(
            "Nicht jede Zeile hat eine Vorhersage — Falten unvollständig"
        )

    nt = (total == 0).to_numpy()
    med = float(np.median(y))
    konst = np.full(len(y), med)

    alle = {
        "Hurdle (zweistufig)": kennzahlen(y, oof_h),
        "Einzelregression": kennzahlen(y, oof_e),
        "Median-Baseline": kennzahlen(y, konst),
    }
    ohne = {
        "Hurdle (zweistufig)": kennzahlen(y[nt], oof_h[nt]),
        "bedingte Regression": kennzahlen(y[nt], oof_b[nt]),
        "Einzelregression": kennzahlen(y[nt], oof_e[nt]),
        "Median-Baseline": kennzahlen(y[nt], konst[nt]),
    }

    def tabelle(titel, d):
        print(f"\n── {titel} " + "─" * max(0, 52 - len(titel)))
        print(f"  {'Modell':<26}{'R²':>10}{'MAE':>10}")
        for name, m in d.items():
            print(f"  {name:<26}{m['r2']:>+10.4f}{m['mae']:>10.3f}")

    tabelle(f"auf ALLEN {len(y)} Validierungszeilen", alle)
    tabelle(f"nur auf den {int(nt.sum())} Nicht-Totalausfällen", ohne)

    kal = float(oof_p.mean()) / float(total.mean()) if total.mean() > 0 else np.nan
    print("\n── Stufe A " + "─" * 52)
    print(
        f"  ROC-AUC               {np.mean(auc_je_falte):.3f} ± {np.std(auc_je_falte):.3f}"
    )
    print(
        f"  Average Precision     {np.mean(ap_je_falte):.3f} ± {np.std(ap_je_falte):.3f}"
        f"   (Zufall = {total.mean():.3f})"
    )
    print(
        f"  Kalibrierung          mittleres p {oof_p.mean():.4f} gegen Rate "
        f"{total.mean():.4f}  (Faktor {kal:.2f})"
    )
    print(f"  Positive je Falte     {npos}")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ax[0].scatter(
        oof_e[nt], y[nt], s=4, alpha=0.25, color="grey", label="ohne Totalausfall"
    )
    ax[0].scatter(
        oof_e[~nt], y[~nt], s=14, alpha=0.7, color="crimson", label="Totalausfall"
    )
    ax[0].set_title(
        f"Einzelregression  R² {alle['Einzelregression']['r2']:+.3f}", fontsize=10
    )
    ax[1].scatter(oof_h[nt], y[nt], s=4, alpha=0.25, color="grey")
    ax[1].scatter(oof_h[~nt], y[~nt], s=14, alpha=0.7, color="crimson")
    ax[1].set_title(f"Hurdle  R² {alle['Hurdle (zweistufig)']['r2']:+.3f}", fontsize=10)
    for a in ax[:2]:
        lim = [0, max(float(y.max()), 100)]
        a.plot(lim, lim, ls="--", lw=1, color="black")
        a.set_xlabel("Vorhersage [%]")
        a.set_ylabel("y [%]")
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    ax[2].hist(oof_p[nt], bins=40, alpha=0.6, color="grey", label="kein Totalausfall")
    ax[2].hist(oof_p[~nt], bins=40, alpha=0.8, color="crimson", label="Totalausfall")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("vorhergesagte Wahrscheinlichkeit p")
    ax[2].set_title(f"Stufe A  AUC {np.mean(auc_je_falte):.3f}", fontsize=10)
    ax[2].legend(fontsize=8)
    fig.tight_layout()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": "hurdle (XGBClassifier + XGBRegressor)",
        "git": _git_state(),
        "config": {
            "total_schwelle": TOTAL_SCHWELLE,
            "y_min": data_prep.Y_MIN,
            "ziel": data_prep.ZIEL,
            "y_max": data_prep.Y_MAX,
            "cv_level": cv_level,
            "test_level": data_prep.TEST_LEVEL,
            "feature_source": data_prep.FEATURE_SOURCE,
            "categorical_encoding": data_prep.CATEGORICAL_ENCODING,
            "max_cardinality": data_prep.MAX_CARDINALITY,
            "params_a": PARAMS_A,
            "params_b": pb,
            "params_source": pb_quelle,
        },
        "data": {
            "sha256": _sha256(CSV_PATH),
            "n_rows": int(len(y)),
            "n_total": int(total.sum()),
            "n_features": int(X.shape[1]),
        },
        "metrics": {
            "alle_zeilen": alle,
            "ohne_totalausfaelle": ohne,
            "stufe_a": {
                "auc_mean": float(np.mean(auc_je_falte)),
                "auc_std": float(np.std(auc_je_falte)),
                "ap_mean": float(np.mean(ap_je_falte)),
                "ap_std": float(np.std(ap_je_falte)),
                "grundrate": float(total.mean()),
                "kalibrierung_faktor": kal,
                "positive_je_falte": npos,
            },
        },
        "laufzeit_s": round(time.time() - t0, 1),
    }
    (RUN_ARTIFACTS_DIR / f"run_{run_id}.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    fig.savefig(RUN_ARTIFACTS_DIR / f"run_{run_id}.png", dpi=150, bbox_inches="tight")
    fig.savefig(
        data_prep.ROOT / "reports" / "figures" / "hurdle_results.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    summary_row = {
        "run_id": run_id,
        "timestamp": record["timestamp"],
        "model": "hurdle",
        "git_commit": (record["git"]["commit"] or "")[:8],
        "git_dirty": record["git"]["dirty"],
        "data_sha256": record["data"]["sha256"][:12],
        "params_source": pb_quelle,
        "cv_level": cv_level,
        "test_level": data_prep.TEST_LEVEL,
        "ziel": data_prep.ZIEL,
        "y_min": data_prep.Y_MIN,
        "y_max": data_prep.Y_MAX,
        "max_cardinality": data_prep.MAX_CARDINALITY,
        "n_rows": len(y),
        "n_features": int(X.shape[1]),
        "hurdle_r2_alle": alle["Hurdle (zweistufig)"]["r2"],
        "hurdle_mae_alle": alle["Hurdle (zweistufig)"]["mae"],
        "einzel_r2_alle": alle["Einzelregression"]["r2"],
        "einzel_mae_alle": alle["Einzelregression"]["mae"],
        "hurdle_r2_ohne": ohne["Hurdle (zweistufig)"]["r2"],
        "hurdle_mae_ohne": ohne["Hurdle (zweistufig)"]["mae"],
        "bedingt_r2_ohne": ohne["bedingte Regression"]["r2"],
        "bedingt_mae_ohne": ohne["bedingte Regression"]["mae"],
        "baseline_median_mae": alle["Median-Baseline"]["mae"],
        "stufe_a_auc": float(np.mean(auc_je_falte)),
        "stufe_a_ap": float(np.mean(ap_je_falte)),
    }
    summary_path = RUNS_DIR / "runs_summary.csv"
    neu = pd.DataFrame([summary_row])
    if summary_path.exists():
        prev = lies_summary(summary_path)
        zusammen = pd.concat([prev, neu], ignore_index=True)
        cols = list(prev.columns) + [c for c in summary_row if c not in prev.columns]
        zusammen = zusammen[cols]
    else:
        zusammen = neu
    zusammen.to_csv(summary_path, index=False)

    print(f"\nRun protokolliert: runs/single/run_{run_id}.json")
    print("Plot: reports/figures/hurdle_results.png")
    print(f"Laufzeit: {record['laufzeit_s']:.0f} s")


if __name__ == "__main__":
    main()
