"""20 — Hyperparameter.

In der Arbeit: Abschnitt 6.2: Tabelle 11, Tabelle A.1 und Abbildung 11
(Voreinstellung gegen optimierte Parameter je Validierungsschema).

XGBoost optimiert gegen Voreinstellung je Ebene und Tabellen der
Hyperparameter von XGBoost und CatBoost.
Ausgabe: reports/figures/hyperparameter_ebenen.png/.pdf, reports/eda/20_hyperparameter_*.

    python eda/20_hyperparameter.py
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import _common as C
from runs_io import waehle_finale_laeufe

EBENEN = [
    ("Billet", "billet", "random"),
    ("Auftrag", "auftrag", "auftrag"),
    ("Artikel", "artikel", "artikel"),
]
KONFIG = {"optimiert": "XGBoost", "voreinstellung": "XGBoost (Standard)"}

SAETZE = {
    "optimiert": dict(label="optimiert", farbe="#2a78d6"),
    "voreinstellung": dict(label="Voreinstellung", farbe="#eb6834"),
}
BREITE = 0.36
VERSATZ = {"optimiert": -BREITE / 2, "voreinstellung": BREITE / 2}
TEXT = "#52514e"
REFERENZ = "#8a8985"

FIG_BREITE = 16 / 2.54
FIG_HOEHE = 3.4

OUT = C.ROOT / "reports" / "figures" / "hyperparameter_ebenen"

VOREINSTELLUNG = {
    "xgboost": dict(
        n_estimators=100,
        learning_rate=0.3,
        max_depth=6,
        subsample=1,
        colsample_bytree=1,
        min_child_weight=1,
        reg_alpha=0,
        reg_lambda=1,
    ),
    "catboost": dict(
        iterations=1000,
        learning_rate=None,
        depth=6,
        l2_leaf_reg=3,
        subsample=0.8,
        rsm=1,
        min_data_in_leaf=1,
    ),
}
WIRKUNG = {
    "n_estimators": "Anzahl der Bäume",
    "iterations": "Zahl der Bäume",
    "learning_rate": "Schrittweite je Baum",
    "max_depth": "Baumtiefe, also bis zu welcher Ordnung Wechselwirkungen gelernt werden",
    "depth": "Baumtiefe; CatBoost baut symmetrische Bäume, Tiefe d ergibt genau 2^d Blätter",
    "subsample": "Anteil der Zeilen je Baum",
    "colsample_bytree": "Anteil der Merkmale je Baum",
    "rsm": "Anteil der Merkmale, die je Teilung geprüft werden",
    "min_child_weight": "Mindestbelegung je Blatt (bei quadratischem Fehler etwa die Zeilenzahl)",
    "min_data_in_leaf": "Mindestbelegung je Blatt; bei symmetrischen Bäumen ohne Wirkung",
    "reg_alpha": "L1-Strafe auf die Blattgewichte",
    "reg_lambda": "L2-Strafe auf die Blattgewichte",
    "l2_leaf_reg": "L2-Strafe auf die Blattgewichte",
}
PARAMDATEI = {
    "xgboost": C.ROOT / "data" / "best_params_v3.json",
    "catboost": C.ROOT / "data" / "catboost_best_params.json",
}


def lade_lauf(run_id: str, cv_level: str, satz: str) -> dict:
    pfad = C.ROOT / "runs" / "single" / f"run_{run_id}.json"
    lauf = json.loads(pfad.read_text(encoding="utf-8"))
    m = lauf["metrics"]
    quelle = lauf["config"]["params_source"]
    if m["cv_level"] != cv_level:
        raise ValueError(f"{run_id}: cv_level {m['cv_level']!r}, erwartet {cv_level!r}")
    if (satz == "optimiert") != quelle.startswith("getunt"):
        raise ValueError(f"{run_id}: params_source {quelle!r} passt nicht zu {satz!r}")
    cv = m["cv"]
    return {
        "run_id": run_id,
        "mae": np.array(cv["XGBoost"]["mae_folds"]),
        "r2": np.array(cv["XGBoost"]["r2_folds"]),
        "mae_median": np.array(cv["Baseline (Median)"]["mae_folds"]),
    }


def lade_alle() -> dict:
    gewaehlt = waehle_finale_laeufe(
        C.ROOT / "runs" / "runs_summary.csv",
        C.ROOT / "runs" / "single",
        konfigs=list(KONFIG.values()),
    )
    daten = {}
    for name, ebene, level in EBENEN:
        daten[name] = {}
        for satz, konfig in KONFIG.items():
            t = gewaehlt[(gewaehlt.ebene == ebene) & (gewaehlt.konfig == konfig)]
            if t.empty:
                raise SystemExit(
                    f"Kein Lauf für {konfig!r} auf Ebene {ebene!r} in runs_summary.csv "
                    "(reproduzieren.py --nur modelle)"
                )
            daten[name][satz] = lade_lauf(t.run_id.iloc[0], level, satz)
    return daten


def balken(ax, x, wert, stil):
    """Ein Balken mit dem CV-Mittel."""
    ax.bar(x, wert, width=BREITE, color=stil["farbe"], edgecolor="white", linewidth=1.5,
           zorder=2)
    ax.annotate(
        f"{wert:.2f}", (x, wert), xytext=(0, -16), textcoords="offset points",
        ha="center", va="top", fontsize=9, color="white", zorder=4,
    )


def referenzstrich(ax, x, wert):
    ax.hlines(wert, x - 0.42, x + 0.42, colors=REFERENZ, linestyles="--", lw=1.5, zorder=3)


class _Suchraum:
    """Ersatz für einen Optuna-Trial: merkt sich die Grenzen je Parameter."""

    def __init__(self):
        self.grenzen = {}

    def suggest_int(self, name, lo, hi, **_):
        self.grenzen[name] = (lo, hi)
        return lo

    def suggest_float(self, name, lo, hi, **_):
        self.grenzen[name] = (lo, hi)
        return lo


def catboost_auto_lernrate() -> float:
    """Lernrate, die CatBoost mit Voreinstellungen selbst wählt (artikelreiner
    Trainingsteil, wie beim Tuning)."""
    from catboost import CatBoostRegressor

    s = C.D.load_split(verbose=False)
    X_tr, _, _ = C.D.process_features(s.X_train_raw, s.X_test_raw, verbose=False)
    m = CatBoostRegressor(random_seed=42, allow_writing_files=False, verbose=0)
    m.fit(X_tr, s.y_train)
    return float(m.get_all_params()["learning_rate"])


def hyperparameter_tabellen():
    import tune

    zeilen_md = {}
    for modell in ("xgboost", "catboost"):
        rec = json.loads(PARAMDATEI[modell].read_text(encoding="utf-8"))
        optimiert = rec["params"]
        raum = _Suchraum()
        tune.suchraum(raum, modell)
        vor = dict(VOREINSTELLUNG[modell])
        if modell == "catboost":
            lr = catboost_auto_lernrate()
            print(f"CatBoost: automatisch gewählte Lernrate {lr:.4f}")
            vor["learning_rate"] = f"automatisch (hier {lr:.3f})"
        zeilen = []
        for p, (lo, hi) in raum.grenzen.items():
            zeilen.append(
                {
                    "Parameter": p,
                    "Wirkung": WIRKUNG.get(p, ""),
                    "Suchraum": f"{lo:g} – {hi:g}",
                    "Bibliotheks-Voreinstellung": vor.get(p),
                    "Optimierte Parameter": optimiert[p],
                }
            )
        t = pd.DataFrame(zeilen)
        ziel = C.OUT_DIR / f"20_hyperparameter_{modell}"
        C.OUT_DIR.mkdir(parents=True, exist_ok=True)
        t.to_csv(ziel.with_suffix(".csv"), index=False, encoding="utf-8")
        md = t.copy()
        md["Optimierte Parameter"] = [
            f"{v:.3g}" if isinstance(v, float) else str(v) for v in md["Optimierte Parameter"]
        ]
        kopf = "| " + " | ".join(md.columns) + " |"
        trenn = "|" + "---|" * len(md.columns)
        koerper = ["| " + " | ".join(str(x) for x in r) + " |" for r in md.itertuples(index=False)]
        ziel.with_suffix(".md").write_text(
            "\n".join([kopf, trenn, *koerper]) + "\n", encoding="utf-8"
        )
        zeilen_md[modell] = ziel
        print(f"Gespeichert: {ziel.with_suffix('.csv')} (+ .md)")
        print(
            f"  Tuning: {rec['tuning']['n_trials']} Trials, Ebene {rec['tuning']['cv_level']}, "
            f"Ziel {rec['tuning']['objective']}, bester Wert {rec['tuning']['best_value']:.3f}"
        )
    return zeilen_md


def main():
    daten = lade_alle()
    namen = [name for name, _, _ in EBENEN]
    xs = np.arange(len(namen))

    fig, ax = plt.subplots(figsize=(FIG_BREITE, FIG_HOEHE))

    for x, name in zip(xs, namen):
        referenzstrich(ax, x, daten[name]["optimiert"]["mae_median"].mean())
        for satz, stil in SAETZE.items():
            balken(ax, x + VERSATZ[satz], daten[name][satz]["mae"].mean(), stil)

    ax.axhline(0, color="#0b0b0b", lw=0.8, zorder=3)
    ax.set_ylabel("CV-MAE in Prozentpunkten")
    ax.set_xticks(xs, namen)
    ax.set_xlim(-0.5, len(xs) - 0.5)
    ax.set_xlabel("Bewertungsebene")
    ax.grid(axis="y")
    ax.tick_params(colors=TEXT)

    legende = [Patch(color=s["farbe"], label=s["label"]) for s in SAETZE.values()] + [
        Line2D([], [], color=REFERENZ, ls="--", lw=1.5, label="Median-Vorhersage"),
    ]
    ax.legend(handles=legende, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    C.aufraeumen(fig)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for endung in ("png", "pdf"):
        fig.savefig(OUT.with_suffix(f".{endung}"), dpi=300, bbox_inches="tight")
    print(f"Gespeichert: {OUT.with_suffix('.png')} (+ .pdf)")

    print("\nEbene     Satz            Lauf              CV-MAE          CV-R²            Gewinn ggü. Median")
    for name in namen:
        for satz in SAETZE:
            d = daten[name][satz]
            g = d["mae_median"] - d["mae"]
            print(
                f"{name:<9} {satz:<15} {d['run_id']}  {d['mae'].mean():6.2f} ± {d['mae'].std():4.2f}   "
                f"{d['r2'].mean():+.3f} ± {d['r2'].std():.3f}   {g.mean():+.2f} ± {g.std():.2f}"
            )
    print()
    hyperparameter_tabellen()


if __name__ == "__main__":
    main()
