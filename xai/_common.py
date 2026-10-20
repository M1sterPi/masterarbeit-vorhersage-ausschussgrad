"""Gemeinsame Hilfen der XAI-Auswertungen: Daten und Testmenge je TEST_LEVEL,
Modelle mit den Parametern der Modellskripte, SHAP für XGBoost und CatBoost,
Ablage der Ergebnisse.

In der Arbeit: gemeinsame Hilfen der xai-Skripte, erzeugt selbst nichts für die
Arbeit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data_pipeline"))

import data_prep as D  # noqa: E402

OUT_DIR = ROOT / "reports" / "xai"
ERG_DIR = ROOT / "runs" / "xai"
SEED = 42
for _ordner in (OUT_DIR / "tabellen", ERG_DIR, ROOT / "reports" / "figures"):
    _ordner.mkdir(parents=True, exist_ok=True)


def lade():
    """Split und kodierte Matrizen — exakt wie in den Trainingsskripten."""
    ds = D.load_split(verbose=False)
    X_tr, X_te, _ = D.process_features(ds.X_train_raw, ds.X_test_raw)
    return ds, X_tr, X_te


def xgb_parameter() -> tuple[dict, str]:
    """Getunte Parameter — Wort fuer Wort wie in xgboost_regression.py."""
    DEFAULT_PARAMS, FIXED_PARAMS = _konstanten_aus_skript()
    p = D.BEST_PARAMS_PATH
    if not p.exists():
        return dict(DEFAULT_PARAMS), "default (kein Tuning-Ergebnis)"
    rec = json.loads(p.read_text(encoding="utf-8"))
    params = {**DEFAULT_PARAMS, **rec.get("params", rec)}
    for k in FIXED_PARAMS:
        params.pop(k, None)
    return params, f"getunt ({p.name})"


def _konstanten_aus_skript(datei: str = "xgboost_regression.py") -> tuple[dict, dict]:
    """DEFAULT_PARAMS und FIXED_PARAMS aus einem Trainingsskript auslesen."""
    import ast

    quelle = (ROOT / "data_pipeline" / datei).read_text(encoding="utf-8")
    baum = ast.parse(quelle)
    gefunden: dict[str, dict] = {}
    for k in baum.body:
        if not isinstance(k, ast.Assign) or len(k.targets) != 1:
            continue
        ziel = k.targets[0]
        if not isinstance(ziel, ast.Name) or ziel.id not in (
            "DEFAULT_PARAMS",
            "FIXED_PARAMS",
        ):
            continue
        aufruf = k.value
        if isinstance(aufruf, ast.Call) and getattr(aufruf.func, "id", "") == "dict":
            gefunden[ziel.id] = {
                kw.arg: ast.literal_eval(kw.value) for kw in aufruf.keywords
            }
        else:
            gefunden[ziel.id] = ast.literal_eval(aufruf)
    fehlt = {"DEFAULT_PARAMS", "FIXED_PARAMS"} - set(gefunden)
    if fehlt:
        raise RuntimeError(
            f"{fehlt} in {datei} nicht gefunden — die Definition "
            "hat sich geaendert, bitte _konstanten_aus_skript() nachziehen."
        )
    return gefunden["DEFAULT_PARAMS"], gefunden["FIXED_PARAMS"]


def _strings_in_reihenfolge(knoten) -> list[str]:
    """String-Literale eines Pfad-Ausdrucks von links nach rechts."""
    import ast

    if isinstance(knoten, ast.BinOp):
        return _strings_in_reihenfolge(knoten.left) + _strings_in_reihenfolge(
            knoten.right
        )
    if isinstance(knoten, ast.Constant) and isinstance(knoten.value, str):
        return [knoten.value]
    return []


def _pfad_aus_skript(datei: str, name: str) -> Path:
    """Eine Pfad-Konstante der Form ROOT / "a" / "b.json" aus einem Skript lesen."""
    import ast

    baum = ast.parse((ROOT / "data_pipeline" / datei).read_text(encoding="utf-8"))
    for k in baum.body:
        if (
            isinstance(k, ast.Assign)
            and len(k.targets) == 1
            and getattr(k.targets[0], "id", "") == name
        ):
            teile = _strings_in_reihenfolge(k.value)
            if teile:
                return ROOT.joinpath(*teile)
    raise RuntimeError(
        f"{name} in {datei} nicht gefunden oder nicht als Pfad-Ausdruck "
        "geschrieben — bitte _pfad_aus_skript() nachziehen."
    )


def xgb_modell(X_tr, y_tr):
    import xgboost as xgb

    params, quelle = xgb_parameter()
    _, fix = _konstanten_aus_skript()
    m = xgb.XGBRegressor(**params, **fix, eval_metric="rmse")
    m.fit(X_tr, y_tr)
    return m, quelle


def shap_xgboost(modell, X) -> tuple[np.ndarray, float]:
    """Exakte TreeSHAP-Beitraege."""
    import xgboost as xgb

    c = modell.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    return c[:, :-1], float(c[0, -1])


def catboost_parameter() -> tuple[dict, str]:
    """Getunte CatBoost-Parameter — gespiegelte load_model_params()."""
    DEFAULT_PARAMS, FIXED_PARAMS = _konstanten_aus_skript("catboost_regression.py")
    p = _pfad_aus_skript("catboost_regression.py", "CATBOOST_BEST_PARAMS_PATH")
    if not p.exists():
        return dict(DEFAULT_PARAMS), "default (kein Tuning-Ergebnis)"
    rec = json.loads(p.read_text(encoding="utf-8"))
    params = {**DEFAULT_PARAMS, **rec.get("params", rec)}
    for k in FIXED_PARAMS:
        params.pop(k, None)
    return params, f"getunt ({p.name})"


def catboost_modell(X_tr, y_tr):
    """CatBoost mit denselben Parametern wie catboost_regression.py."""
    from catboost import CatBoostRegressor

    params, quelle = catboost_parameter()
    _, fix = _konstanten_aus_skript("catboost_regression.py")
    m = CatBoostRegressor(**params, **fix)
    m.fit(X_tr, y_tr, verbose=False)
    return m, quelle


def shap_catboost(modell, X, y) -> tuple[np.ndarray, float]:
    from catboost import Pool

    sv = modell.get_feature_importance(Pool(X, y), type="ShapValues")
    return sv[:, :-1], float(sv[0, -1])


def pruefe_additivitaet(beitraege, basis, vorhersage, name: str, grenze=1e-3):
    """Summe der Beitraege + Basis muss die Vorhersage ergeben."""
    ab = np.abs(beitraege.sum(axis=1) + basis - np.asarray(vorhersage))
    schlimm = float(ab.max())
    print(f"  Additivitaet {name}: max |Summe + Basis − Vorhersage| = {schlimm:.2e}")
    if schlimm > grenze:
        raise AssertionError(
            f"{name}: Additivitaet verletzt ({schlimm:.3e} > {grenze:.0e}) — "
            "die SHAP-Werte gehoeren nicht zu diesem Modell."
        )
    return schlimm


def globale_wichtigkeit(beitraege, spalten) -> pd.DataFrame:
    """Mittlerer Betrag je Merkmal — die uebliche globale Aggregation."""
    return (
        pd.DataFrame(
            {
                "Merkmal": list(spalten),
                "mittlerer_betrag": np.abs(beitraege).mean(axis=0),
                "mittlerer_beitrag": beitraege.mean(axis=0),
                "std": beitraege.std(axis=0),
            }
        )
        .sort_values("mittlerer_betrag", ascending=False)
        .reset_index(drop=True)
    )


def herkunft(name: str) -> str:
    """Grobe Zuordnung fuer die Auswertung nach Datenquelle (FF2)."""
    suffixe = (
        "_mean",
        "_std",
        "_min",
        "_max",
        "_last",
        "_range",
        "_n_samples",
        "_frac_on",
        "_n_edges",
        "_tmean",
        "_integral",
        "_known_frac",
        "_frac_on_t",
        "_n_edges_t",
    )
    return "Prozess" if name.endswith(suffixe) else "Stammdaten"


def mit_ebene(name: str) -> str:
    """Dateiname mit angehaengter Test-Ebene: 01_global.pdf -> 01_global_auftrag.pdf."""
    stamm, _, endung = name.rpartition(".")
    return f"{stamm}_{D.TEST_LEVEL}.{endung}"


TAB_DIR = OUT_DIR / "tabellen"


def speichere_rangliste(name: str, tabelle, wert_spalte: str = "mittlerer_betrag"):
    """Vollstaendige Rangliste als CSV — eine Zeile je Merkmal, MIT Zahl."""
    df = tabelle.copy()
    if wert_spalte in df.columns:
        df = df.sort_values(wert_spalte, ascending=False)
    df = df.reset_index(drop=True)
    df.insert(0, "rang", np.arange(1, len(df) + 1))
    if "Merkmal" in df.columns and "herkunft" not in df.columns:
        df.insert(2, "herkunft", [herkunft(m) for m in df.Merkmal])
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    p = TAB_DIR / name
    df.to_csv(p, index=False, encoding="utf-8", float_format="%.6g")
    return p


def pdf_oeffnen(name: str) -> PdfPages:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return PdfPages(OUT_DIR / name)


def seite(titel: str, zeilen: int, spalten: int, groesse=(11.7, 8.3)):
    fig, axes = plt.subplots(zeilen, spalten, figsize=groesse)
    fig.suptitle(titel, fontsize=13, y=0.985)
    return fig, np.atleast_1d(axes).ravel()


def schliesse(pdf: PdfPages, fig, unten: str = ""):
    if unten:
        fig.text(0.5, 0.012, unten, ha="center", fontsize=7.5, style="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    pdf.savefig(fig)
    plt.close(fig)


def textseite(pdf: PdfPages, titel: str, absaetze: list[str]):
    fig = plt.figure(figsize=(11.7, 8.3))
    fig.text(0.06, 0.93, titel, fontsize=15, weight="bold", va="top")
    y = 0.85
    for a in absaetze:
        fig.text(0.06, y, a, fontsize=10, va="top", wrap=True)
        y -= 0.035 + 0.016 * (len(a) // 110)
    pdf.savefig(fig)
    plt.close(fig)


def fusszeile(ds, X, quelle: str) -> str:
    return (
        f"{D.CSV_PATH.name} | Test-Set {X.shape[0]} Billets × {X.shape[1]} "
        f"Merkmale | y ∈ [{D.Y_MIN:g}; {D.Y_MAX:g}] | Test-Ebene "
        f"{D.TEST_LEVEL} | MAX_CARDINALITY {D.MAX_CARDINALITY} | {quelle}"
    )


def speichere(name: str, obj: dict):
    ERG_DIR.mkdir(parents=True, exist_ok=True)
    p = ERG_DIR / name
    p.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return p
