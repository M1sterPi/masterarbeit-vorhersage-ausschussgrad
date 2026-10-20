"""FreshPRINCE-Regression auf den Rohkurven (Ablauf in sequenz_runner.py).

In der Arbeit: FreshPRINCE-Läufe für Abschnitt 6.3/6.4 (Tabelle 12, Abbildungen
12 und 13).

Ebene über TEST_LEVEL und CV_LEVEL, z-Normierung über ZNORM.

    python data_pipeline/freshprince_regression.py
"""

from aeon.regression.feature_based import FreshPRINCERegressor

import rocket_regression as RK
import sequenz_runner

MODEL_NAME = "FreshPRINCE"

FC_PARAMETER = "minimal"
KANAELE = None
N_ESTIMATORS = 200
N_JOBS = -1
SEED = 42


def mache_modell():
    return FreshPRINCERegressor(
        default_fc_parameters=FC_PARAMETER,
        n_estimators=N_ESTIMATORS,
        n_jobs=N_JOBS,
        random_state=SEED,
    )


if __name__ == "__main__":
    if KANAELE is not None:
        fehlend = [k for k in KANAELE if k not in RK.CHANNELS]
        if fehlend:
            raise SystemExit(f"Unbekannte Kanäle: {fehlend}")
        raise SystemExit(
            "Kanalauswahl ist vorbereitet, aber noch nicht verdrahtet — der "
            "Tensor kommt aus rocket_regression und enthaelt alle Kanäle. Vor "
            "dem ersten Teilmengen-Lauf hier die Auswahl auf den Tensor "
            "anwenden, sonst weicht das Protokoll vom Gerechneten ab."
        )
    sequenz_runner.lauf(
        model_name=MODEL_NAME,
        mache_modell=mache_modell,
        params_source=(
            f"default (tsfresh {FC_PARAMETER}, RotationForest "
            f"n_estimators={N_ESTIMATORS}, alle {len(RK.CHANNELS)} Kanäle)"
        ),
        modell_params={
            "default_fc_parameters": FC_PARAMETER,
            "n_estimators": N_ESTIMATORS,
            "n_kanaele": len(RK.CHANNELS),
            "random_state": SEED,
        },
    )
