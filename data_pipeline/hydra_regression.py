"""HYDRA-Regression auf den Rohkurven (Ablauf in sequenz_runner.py).

In der Arbeit: HYDRA-Läufe für Abschnitt 6.3/6.4 (Tabelle 12, Abbildungen 12
und 13).

Ebene über TEST_LEVEL und CV_LEVEL, z-Normierung über ZNORM.

    python data_pipeline/hydra_regression.py
"""

from aeon.regression.convolution_based import (
    HydraRegressor,
    MultiRocketHydraRegressor,
)

import sequenz_runner

MODEL_NAME = "HYDRA"

VARIANTE = "hydra"

N_KERNELS = 8
N_GROUPS = 64
N_JOBS = -1
SEED = 42


def mache_modell():
    kls = HydraRegressor if VARIANTE == "hydra" else MultiRocketHydraRegressor
    return kls(n_kernels=N_KERNELS, n_groups=N_GROUPS, n_jobs=N_JOBS, random_state=SEED)


if __name__ == "__main__":
    sequenz_runner.lauf(
        model_name=MODEL_NAME if VARIANTE == "hydra" else "MultiRocketHYDRA",
        mache_modell=mache_modell,
        params_source=(
            f"default ({VARIANTE}, n_kernels={N_KERNELS}, n_groups={N_GROUPS}; "
            "Ridge-Kopf wird intern per CV regularisiert)"
        ),
        modell_params={
            "variante": VARIANTE,
            "n_kernels": N_KERNELS,
            "n_groups": N_GROUPS,
            "random_state": SEED,
        },
    )
