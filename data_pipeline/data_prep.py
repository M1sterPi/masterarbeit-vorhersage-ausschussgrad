"""Gemeinsame Daten- und Merkmalslogik aller Skripte.

In der Arbeit: Filter der Zielgröße (Abschnitt 4.1.3), gesperrte Spalten und
Datumserkennung (5.4), Kodierung und Kardinalitätsschwelle (5.5),
Plausibilitätsgrenze im tabellarischen Zweig (5.6), eingefrorene Testmengen und
Falten der drei Validierungsschemata (6.1).

Laden der Merkmalstabelle, Filter auf y, Entfernen von Leckage-, Kennungs-,
Datums- und Zeitspalten, eingefrorene Testmengen je Bewertungsebene,
Kreuzvalidierungsfalten (make_cv) und Merkmalsverarbeitung (process_features,
nur auf dem Trainingsteil angepasst).

    python data_pipeline/data_prep.py   # fehlende Testmenge für TEST_LEVEL ziehen
"""

import os
import pathlib
import json
import hashlib
import subprocess
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

KURVEN_DIR = pathlib.Path(os.environ.get("KURVEN_DIR") or "D:/billet_kurven")

# Längstes Zeitfenster je Billet beim Export aus der InfluxDB
MAX_BILLET_DURATION_H = 9

BESCHRIFTUNGEN_PATH = ROOT / "data" / "beschriftungen.json"


def beschriftungen(abschnitt: str) -> dict:
    """Lesbare Bezeichnungen für Abbildungen und Tabellen aus data/beschriftungen.json
    (nicht im Repo); ohne die Datei bleiben die Spaltennamen stehen."""
    if not BESCHRIFTUNGEN_PATH.exists():
        return {}
    return json.loads(BESCHRIFTUNGEN_PATH.read_text(encoding="utf-8")).get(abschnitt, {})


SPALTEN_PATH = ROOT / "data" / "spalten.json"


class _Spalten(dict):
    """Spaltennamen und Kenngrößen der Quelldaten aus data/spalten.json (nicht im
    Repo). Der Code spricht Spalten nur über ihre Rolle an, z. B. SP["auftrag"]."""

    def __missing__(self, rolle):
        raise SystemExit(
            f"Rolle {rolle!r} unbekannt: {SPALTEN_PATH} fehlt oder enthält sie nicht "
            "(siehe data/README.md)."
        )


SP = _Spalten(
    json.loads(SPALTEN_PATH.read_text(encoding="utf-8")) if SPALTEN_PATH.exists() else {}
)

CSV_PATH = ROOT / "data" / "featurized_batches_trimmed_v6.csv"

# Eingefrorene Testmengen (Listen von batch_id), je Bewertungsebene eine Datei
TEST_IDS_PATH = ROOT / "data" / "test_billets_v3.json"
TEST_SIZE = 0.2
SPLIT_SEED = 42

# Bewertungsebene der Testmenge: billet, auftrag oder artikel
TEST_LEVEL = os.environ.get("TEST_LEVEL", "artikel")

_TEST_LEVEL_GROUPS = {
    "billet": None,
    "auftrag": SP["auftrag"],
    "artikel": SP["artikel"],
}
TEST_IDS_PATHS = {
    "billet": TEST_IDS_PATH,
    "auftrag": ROOT / "data" / "test_billets_auftrag_v3.json",
    "artikel": ROOT / "data" / "test_billets_artikel_v3.json",
}


def test_ids_path(level: str = None) -> pathlib.Path:
    """Pfad des eingefrorenen Test-Sets für die gewählte Ebene."""
    level = level or TEST_LEVEL
    if level not in TEST_IDS_PATHS:
        raise ValueError(
            f"TEST_LEVEL={level!r} unbekannt — erlaubt: {sorted(TEST_IDS_PATHS)}"
        )
    return TEST_IDS_PATHS[level]


# Falten der Kreuzvalidierung: random, auftrag oder artikel (gruppenrein)
CV_LEVEL = os.environ.get("CV_LEVEL", "artikel")
CV_ALSO_RANDOM = True
CV_SPLITS = 5


def make_cv(df: pd.DataFrame, level: str = None, n_splits: int = None, seed: int = 42):
    """Faltenliste [(train_idx, val_idx), ...] plus Beschriftung der Ebene."""
    from sklearn.model_selection import KFold, GroupKFold

    level = level or CV_LEVEL
    n_splits = n_splits or CV_SPLITS
    if level in (None, "random"):
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(cv.split(df)), "random"

    if level not in _TEST_LEVEL_GROUPS:
        raise ValueError(
            f"CV_LEVEL={level!r} unbekannt — erlaubt: 'random', "
            + ", ".join(repr(k) for k in _TEST_LEVEL_GROUPS if k != "billet")
        )
    col = _TEST_LEVEL_GROUPS[level]
    if col is None:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(cv.split(df)), "random"
    if col not in df.columns:
        raise KeyError(f"Gruppenspalte {col!r} fehlt im Datenrahmen")

    groups = df[col].astype(str).reset_index(drop=True)
    cv = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(df, groups=groups)), level


# Ergebnis von tune.py für XGBoost
BEST_PARAMS_PATH = ROOT / "data" / "best_params_v3.json"

RUNS_DIR = ROOT / "runs"
RUN_ARTIFACTS_DIR = RUNS_DIR / "single"
SWEEPS_DIR = RUNS_DIR / "sweeps"
LOGS_DIR = RUNS_DIR / "logs"

# Textspalten mit mehr Ausprägungen kennzeichnen Artikel/Aufträge und entfallen
MAX_CARDINALITY = 90

# Beträge darüber gelten als Fehlwert der Steuerung
MAX_PLAUSIBLE_ABS = 1e12

CATEGORICAL_ENCODING = "ordinal"

ONEHOT_MAX_CARDINALITY = None

# Merkmalsquellen: beides, sql (Stammdaten) oder influx (Prozesskanäle)
FEATURE_SOURCE = os.environ.get("FEATURE_SOURCE", "beides")

_AGG_SUFFIXES = (
    "_mean",
    "_std",
    "_min",
    "_max",
    "_last",
    "_range",
    "_n_samples",
    "_known_frac",
    "_frac_on_t",
    "_n_edges",
)
_EXTRA_INFLUX = {"batch_duration_s"}

MASTER_CSV = ROOT / "data" / "billets_master.csv"
_SQL_UMBENANNT = {SP["energie"]: "Energie_prog"}
_ABGELEITET = {"melt_weekday", "melt_hour"}


def influx_spalten(cols) -> list:
    """Spalten, die aus Influx stammen: Kanal-Aggregate UND Influx-Metadaten."""
    sql = {
        _SQL_UMBENANNT.get(c, c)
        for c in pd.read_csv(MASTER_CSV, nrows=0).columns
    }
    pfad = ROOT / "data" / "cache" / (CSV_PATH.stem + ".process_cols.json")
    if not pfad.exists():
        raise SystemExit(
            f"FEATURE_SOURCE={FEATURE_SOURCE!r} braucht die Kanal-Klassifikation, "
            f"aber {pfad} fehlt.\nEntweder prepare_data.py für diesen Datenstand "
            "laufen lassen oder FEATURE_SOURCE = 'beides' setzen."
        )
    pc = json.loads(pfad.read_text(encoding="utf-8"))["process_cols"]
    return [
        c
        for c in cols
        if c in _EXTRA_INFLUX
        or any(c.startswith(k) and c[len(k) :] in _AGG_SUFFIXES for k in pc)
        or (c not in sql and c not in _ABGELEITET)
    ]


# Optionale Teilmengen der Daten; für die Arbeit alle ausgeschaltet
TOP_FEATURES = []

ONLY_CATEGORY = None

ONLY_GLASTYP = None

ONLY_MACHINE = None

# Zielgröße: Masseverlust in Prozent der Sollmasse
ZIEL = "masse"

Y_MIN, Y_MAX = 1, 100.0

# Kennungen, Datumsfelder und versteckte Zeitstempel sind keine Merkmale
DROP_ID_FEATURES = True
_ID_FEATURES = [
    SP["billetnummer"],
]

DROP_DATE_COLUMNS = True
_DATE_PATTERN = r"^\s*(\d{1,2}\.\d{1,2}\.\d{4}|\d{4}-\d{1,2}-\d{1,2})"
_DATE_MIN_SHARE = 0.8


def _looks_like_date(s: pd.Series) -> bool:
    v = s.dropna()
    if len(v) < 20:
        return False
    return bool(v.astype(str).str.match(_DATE_PATTERN).mean() >= _DATE_MIN_SHARE)


DROP_TIME_PROXIES = True
_TIME_PROXY_FEATURES = [
    SP["billetnr_kanal"],
    SP["billetnr_8"],
    SP["kontrolle_am"],
    SP["kalenderwoche"],
    SP["monat"],
    SP["aussenschliff_am"],
]

_TIME_PROXY_PREFIXES = [
    SP["schmelzende_zeit"],
    SP["extraktion_zeit"],
]

# Wochentag und Stunde des Schmelzendes als Merkmale ergänzen
ADD_CYCLIC_FEATURES = True

# Spalten (Teilstrings), die erst nach der Schmelze feststehen und deshalb nicht
# als Merkmal dienen dürfen. Die Liste ist in diesem Repository nicht enthalten.
_LEAKAGE_KEYWORDS: list[str] = []


def _is_leakage(col: str) -> bool:
    return any(kw in col for kw in _LEAKAGE_KEYWORDS)


def _to_num_de(s: pd.Series) -> pd.Series:
    """Deutsches Zahlenformat ('1.234,5') → float."""
    return pd.to_numeric(
        s.astype(str)
        .str.strip()
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False),
        errors="coerce",
    )


def _fmt_cols(cols, annot=None, indent="      ") -> str:
    """Spaltenliste untereinander, optional mit Annotation je Spalte."""
    if not cols:
        return f"{indent}(keine)"
    lines = []
    for c in cols:
        suffix = f"  ({annot[c]})" if annot and c in annot else ""
        lines.append(f"{indent}- {c}{suffix}")
    return "\n".join(lines)


def process_features(X_fit: pd.DataFrame, X_apply: pd.DataFrame, verbose: bool = False):
    """Feature-Processing mit Fit ausschließlich auf X_fit."""
    X_fit = X_fit.copy()
    X_apply = X_apply.copy()
    n_input = X_fit.shape[1]
    dropped_cols, encoded_cols, numeric_cols, converted_de_cols = [], [], [], []
    encoded_nunique, highcard_nunique = {}, {}
    onehot_fit, onehot_apply, onehot_ersetzt = {}, {}, []
    onehot_cols, ordinal_cols = [], []

    for col in list(X_fit.columns):
        if X_fit[col].dtype == object:
            conv = _to_num_de(X_fit[col])
            if conv.notna().sum() >= 0.7 * X_fit[col].notna().sum():
                X_fit[col] = conv
                X_apply[col] = _to_num_de(X_apply[col])
                numeric_cols.append(col)
                converted_de_cols.append(col)
                continue

        if X_fit[col].dtype == object:
            n_unique = int(X_fit[col].nunique(dropna=True))
            if n_unique <= MAX_CARDINALITY:
                cats = sorted(X_fit[col].dropna().astype(str).unique())
                onehot_hier = CATEGORICAL_ENCODING == "onehot" and (
                    ONEHOT_MAX_CARDINALITY is None or n_unique <= ONEHOT_MAX_CARDINALITY
                )
                if onehot_hier:
                    if n_unique > 1:
                        sf = X_fit[col].astype(str)
                        sa = X_apply[col].astype(str)
                        for kat in cats:
                            neu = f"{col}__{kat}"
                            onehot_fit[neu] = (sf == kat).astype(float)
                            onehot_apply[neu] = (sa == kat).astype(float)
                        onehot_ersetzt.append(col)
                        encoded_cols.append(col)
                        onehot_cols.append(col)
                        encoded_nunique[col] = n_unique
                    else:
                        onehot_ersetzt.append(col)
                else:
                    mapping = {c: i for i, c in enumerate(cats)}

                    def _encode(v, _m=mapping):
                        if pd.isna(v):
                            return np.nan
                        return _m.get(str(v), -1)

                    X_fit[col] = pd.to_numeric(X_fit[col].map(_encode), errors="coerce")
                    X_apply[col] = pd.to_numeric(
                        X_apply[col].map(_encode), errors="coerce"
                    )
                    encoded_cols.append(col)
                    ordinal_cols.append(col)
                    encoded_nunique[col] = n_unique
            else:
                dropped_cols.append(col)
                highcard_nunique[col] = n_unique
        else:
            numeric_cols.append(col)

    X_fit = X_fit.drop(columns=dropped_cols)
    X_apply = X_apply.drop(columns=dropped_cols)

    if onehot_ersetzt:
        X_fit = X_fit.drop(columns=onehot_ersetzt)
        X_apply = X_apply.drop(columns=onehot_ersetzt)
        if onehot_fit:
            namen = list(onehot_fit)
            X_fit = pd.concat(
                [X_fit, pd.DataFrame(onehot_fit, index=X_fit.index)[namen]], axis=1
            )
            X_apply = pd.concat(
                [X_apply, pd.DataFrame(onehot_apply, index=X_apply.index)[namen]],
                axis=1,
            )

    all_nan = X_fit.columns[X_fit.isna().all()].tolist()
    if all_nan:
        X_fit = X_fit.drop(columns=all_nan)
        X_apply = X_apply.drop(columns=all_nan)

    if verbose:
        n_plain_num = len(numeric_cols) - len(converted_de_cols)
        bar = "  " + "─" * 56
        print(f"  Eingangsspalten:               {n_input}")
        print(bar)
        print(f"  → numerisch (direkt):          {n_plain_num}")
        print(f"  → aus DE-Zahlenformat gewandelt:{len(converted_de_cols):>3}")
        if CATEGORICAL_ENCODING == "onehot":
            print(
                f"  → one-hot ({len(onehot_cols)} Spalten):        "
                f"{len(onehot_fit)} Merkmale"
            )
            n_konst = len(onehot_ersetzt) - len(onehot_cols)
            if n_konst:
                print(f"  → konstant, uebersprungen:     {n_konst}")
            if ordinal_cols:
                print(
                    f"  → label-encodiert (> {ONEHOT_MAX_CARDINALITY} Kat.): "
                    f"{len(ordinal_cols)}"
                )
        else:
            print(f"  → label-encodiert:             {len(encoded_cols)}")
        print(f"  → verworfen (hochkardinal):    {len(dropped_cols)}")
        print(f"  → verworfen (komplett leer):   {len(all_nan)}")
        print(bar)
        print(f"  Features gesamt:               {X_fit.shape[1]}")

        if converted_de_cols:
            print("\n  Aus DE-Zahlenformat gewandelt (object → float):")
            print(_fmt_cols(converted_de_cols))
        if encoded_cols:
            print("\n  Label-encodiert (Kategorie → Integer):")
            print(
                _fmt_cols(
                    encoded_cols,
                    {c: f"{encoded_nunique[c]} Kategorien" for c in encoded_cols},
                )
            )
        if dropped_cols:
            print(f"\n  Verworfen – hochkardinal (> {MAX_CARDINALITY} unique):")
            print(
                _fmt_cols(
                    dropped_cols,
                    {c: f"{highcard_nunique[c]} unique" for c in dropped_cols},
                )
            )
        if all_nan:
            print("\n  Verworfen – komplett leer (auf Train):")
            print(_fmt_cols(all_nan))

    X_apply = X_apply[X_fit.columns]

    n_ersetzt = 0
    for M in (X_fit, X_apply):
        num_cols = M.select_dtypes(include=[np.number]).columns
        if not len(num_cols):
            continue
        block = M[num_cols].to_numpy(dtype="float64", copy=True)
        schlecht = ~np.isfinite(block) | (np.abs(block) > MAX_PLAUSIBLE_ABS)
        n_ersetzt += int((schlecht & ~M[num_cols].isna().to_numpy()).sum())
        if schlecht.any():
            block[schlecht] = np.nan
            M[num_cols] = block
    if n_ersetzt:
        print(
            f"\n  Nicht-physikalische Werte auf NaN gesetzt: {n_ersetzt} Zellen "
            f"(Grenze |x| > {MAX_PLAUSIBLE_ABS:g})"
        )

    info = {
        "numeric_cols": numeric_cols,
        "converted_de_cols": converted_de_cols,
        "encoded_cols": encoded_cols,
        "onehot_cols": onehot_cols,
        "ordinal_cols": ordinal_cols,
        "dropped_highcard_cols": dropped_cols,
        "all_nan_cols": all_nan,
        "implausible_cells": n_ersetzt,
    }
    return X_fit, X_apply, info


def _sha256(path: pathlib.Path) -> str:
    """SHA-256 des Dateiinhalts — belegt, auf welchem Datenstand der Lauf basiert."""
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _git_state() -> dict:
    """Aktueller Git-Commit + ob das Arbeitsverzeichnis uncommittete Änderungen hat."""

    def _run(args):
        return (
            subprocess.check_output(args, cwd=ROOT, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )

    try:
        commit = _run(["git", "rev-parse", "HEAD"])
        dirty = bool(_run(["git", "status", "--porcelain"]))
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def _draw_group_ids(df_ok, group_col: str, test_size: float, seed: int) -> list:
    """Ganze Gruppen zurückhalten, geschichtet nach Gruppengrösse."""
    sizes = df_ok.groupby(group_col).size().sort_values()
    rng = np.random.RandomState(seed)
    ranks = pd.Series(range(len(sizes)), index=sizes.index)
    n_bands = min(10, len(sizes))
    bands = pd.cut(ranks, bins=n_bands, labels=False)

    picked = []
    for b in range(n_bands):
        members = list(sizes.index[bands == b])
        if not members:
            continue
        n_take = int(round(test_size * len(members)))
        if n_take:
            picked.extend(rng.choice(members, size=n_take, replace=False).tolist())
    return sorted(df_ok.loc[df_ok[group_col].isin(picked), "batch_id"].tolist())


def draw_test_ids(
    csv_path: pathlib.Path = None,
    out_path: pathlib.Path = None,
    test_size: float = TEST_SIZE,
    seed: int = SPLIT_SEED,
    overwrite: bool = False,
    level: str = None,
) -> list:
    """Test-Set über den GESAMTEN Datensatz ziehen und einfrieren."""
    level = level or TEST_LEVEL
    csv_path = pathlib.Path(csv_path or CSV_PATH)
    out_path = pathlib.Path(out_path or test_ids_path(level))

    if out_path.exists() and not overwrite:
        ids = json.loads(out_path.read_text(encoding="utf-8"))
        print(f"{out_path.name} existiert bereits ({len(ids)} Billets) — unverändert.")
        return ids

    df = _setze_ziel(pd.read_csv(csv_path, low_memory=False))
    ok = df["y"].notna() & df["y"].between(Y_MIN, Y_MAX)
    df_ok = df.loc[ok]

    group_col = _TEST_LEVEL_GROUPS[level]

    if group_col is not None:
        if ONLY_CATEGORY is not None and SP["kategorie"] in df_ok.columns:
            df_ok = df_ok[df_ok[SP["kategorie"]] == ONLY_CATEGORY]
        if ONLY_GLASTYP is not None and "Glastyp" in df_ok.columns:
            df_ok = df_ok[df_ok["Glastyp"] == ONLY_GLASTYP]

    all_ids = df_ok["batch_id"].tolist()
    if group_col is None:
        rng = np.random.RandomState(seed)
        n_test = int(round(test_size * len(all_ids)))
        test_ids = sorted(rng.choice(all_ids, size=n_test, replace=False).tolist())
    else:
        if group_col not in df_ok.columns:
            raise KeyError(f"Spalte {group_col!r} fehlt in {csv_path.name}")
        test_ids = _draw_group_ids(df_ok, group_col, test_size, seed)

    out_path.write_text(json.dumps(test_ids, indent=2), encoding="utf-8")

    print(
        f"Test-Set gezogen aus {csv_path.name} (Ebene {level!r}): {len(test_ids)} von "
        f"{len(all_ids)} Chargen mit plausiblem y ({len(df)} gesamt), Seed {seed} "
        f"→ {out_path.name}"
    )
    if group_col is not None:
        te = df_ok[df_ok.batch_id.isin(set(test_ids))]
        tr = df_ok[~df_ok.batch_id.isin(set(test_ids))]
        print(
            f"   Gruppen ({group_col}): {te[group_col].nunique()} von "
            f"{df_ok[group_col].nunique()} zurueckgehalten | Testanteil "
            f"{len(te) / len(df_ok) * 100:.1f} %"
        )
        print(
            f"   y Test  Mittel {te.y.mean():5.2f} Median {te.y.median():5.2f} "
            f"Std {te.y.std():5.2f}"
        )
        print(
            f"   y Train Mittel {tr.y.mean():5.2f} Median {tr.y.median():5.2f} "
            f"Std {tr.y.std():5.2f}"
        )
        for cname, ccol in (
            ("Auftrag", SP["auftrag"]),
            ("Artikel", SP["artikel"]),
        ):
            if ccol in df_ok.columns:
                ov = te[ccol].astype(str).isin(set(tr[ccol].astype(str))).mean()
                print(
                    f"   Test-Billets mit {cname:8s} auch im Train: {ov * 100:5.1f} %"
                )
    return test_ids


@dataclass
class DataSplit:
    """Ergebnis von load_split(): Rohdaten, eingefrorener Split, Metadaten."""

    df: pd.DataFrame
    X_raw: pd.DataFrame
    y: pd.Series
    batch_ids: pd.Series
    test_mask: np.ndarray
    X_train_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    leakage_cols: list = field(default_factory=list)
    id_cols: list = field(default_factory=list)
    melt_end_ts: pd.Series = field(default_factory=pd.Series)


def _setze_ziel(df: "pd.DataFrame") -> "pd.DataFrame":
    """y gemaess ZIEL neu bilden."""
    if ZIEL == "masse":
        return df
    if ZIEL != "laenge":
        raise ValueError(f"ZIEL muss 'masse' oder 'laenge' sein, nicht {ZIEL!r}")
    fehlt = {SP["laenge_soll"], SP["laenge_ist"]} - set(df.columns)
    if fehlt:
        raise KeyError(f"ZIEL='laenge' braucht {fehlt} in {CSV_PATH.name}")
    soll = pd.to_numeric(df[SP["laenge_soll"]], errors="coerce")
    ist = pd.to_numeric(df[SP["laenge_ist"]], errors="coerce")
    df = df.copy()
    df["y"] = (soll - ist) / soll.where(soll != 0) * 100
    return df


def load_split(verbose: bool = True) -> DataSplit:
    """Laden, Filtern, Spalten-Entfernung und eingefrorener Train/Test-Split."""

    def _say(*a, **k):
        if verbose:
            print(*a, **k)

    _say("═" * 64)
    _say(" ROHDATEN-AUFBEREITUNG")
    _say("═" * 64)

    df = _setze_ziel(pd.read_csv(CSV_PATH, low_memory=False))
    n_cols_raw = df.shape[1]
    _say(
        f"Geladen:                {df.shape[0]} Zeilen × {n_cols_raw} Spalten  ({CSV_PATH.name})"
    )

    if ONLY_CATEGORY is not None:
        before = len(df)
        df = df[df[SP["kategorie"]] == ONLY_CATEGORY].reset_index(drop=True)
        _say(
            f"Kategorie-Filter:       '{ONLY_CATEGORY}'  →  {before} → {len(df)} Zeilen "
            f"({before - len(df)} entfernt)"
        )

    if ONLY_GLASTYP is not None:
        before = len(df)
        df = df[df["Glastyp"] == ONLY_GLASTYP].reset_index(drop=True)
        _say(
            f"Glastyp-Filter:         '{ONLY_GLASTYP}'  →  {before} → {len(df)} Zeilen "
            f"({before - len(df)} entfernt)"
        )

    if ONLY_MACHINE is not None:
        before = len(df)
        df = df[df[SP["ofen_kennung"]] == ONLY_MACHINE].reset_index(drop=True)
        _say(
            f"Maschinen-Filter:       '{ONLY_MACHINE}'  →  {before} → {len(df)} Zeilen "
            f"({before - len(df)} entfernt)"
        )

    na_mask = df["y"].notna()
    range_mask = na_mask & df["y"].between(Y_MIN, Y_MAX)
    n_na = int((~na_mask).sum())
    n_oob = int((na_mask & ~range_mask).sum())
    _say(
        f"y-Bereinigung:          {n_na} ohne y + {n_oob} außerhalb "
        f"[{Y_MIN:.0f}, {Y_MAX:.0f}]  →  {int(range_mask.sum())} verbleiben"
    )
    df = df[range_mask].reset_index(drop=True)
    _say(
        f"y-Verteilung (final):   mean={df['y'].mean():.2f}  std={df['y'].std():.2f}  "
        f"min={df['y'].min():.2f}  max={df['y'].max():.2f}"
    )
    y0 = df[df["y"] == 0]["batch_id"].tolist()
    y100 = df[df["y"] == 100]["batch_id"].tolist()
    _say(f"  Randwerte:            {len(y0)}× y=0,  {len(y100)}× y=100")

    y = df["y"].copy()
    batch_ids = df["batch_id"].copy()
    X_raw = df.drop(columns=["y", "batch_id"], errors="ignore").copy()

    _say(
        f"\nSpalten entfernt — Ziel/ID:  y, batch_id  "
        f"→  {X_raw.shape[1] + 2} → {X_raw.shape[1]}"
    )

    leakage_cols = [c for c in X_raw.columns if _is_leakage(c)]
    X_raw = X_raw.drop(columns=leakage_cols)
    _say(f"Spalten entfernt — Leakage:  {len(leakage_cols)}")
    if leakage_cols:
        _say(_fmt_cols(leakage_cols))

    drop_names = (list(_ID_FEATURES) if DROP_ID_FEATURES else []) + (
        list(_TIME_PROXY_FEATURES) if DROP_TIME_PROXIES else []
    )
    id_cols = [c for c in X_raw.columns if c in drop_names]
    if DROP_TIME_PROXIES:
        id_cols += [
            c
            for c in X_raw.columns
            if c not in id_cols and any(c.startswith(p) for p in _TIME_PROXY_PREFIXES)
        ]
    if DROP_DATE_COLUMNS:
        date_cols = [
            c
            for c in X_raw.columns
            if c not in id_cols
            and X_raw[c].dtype == object
            and _looks_like_date(X_raw[c])
        ]
        if date_cols:
            _say(
                f"Spalten entfernt — Datumsfelder (generisch erkannt): {len(date_cols)}"
            )
            _say(_fmt_cols(date_cols))
        id_cols += date_cols
    if id_cols:
        X_raw = X_raw.drop(columns=id_cols)
    _say(
        f"Spalten entfernt — ID-Features/Zeit-Proxies: {len(id_cols)}  "
        f"(DROP_ID_FEATURES={DROP_ID_FEATURES}, DROP_TIME_PROXIES={DROP_TIME_PROXIES})"
    )
    if id_cols:
        _say(_fmt_cols(id_cols))

    if FEATURE_SOURCE not in ("sql", "influx", "beides"):
        raise SystemExit(f"FEATURE_SOURCE={FEATURE_SOURCE!r} unbekannt")
    if FEATURE_SOURCE != "beides":
        influx = influx_spalten(X_raw.columns)
        weg = (
            influx
            if FEATURE_SOURCE == "sql"
            else [c for c in X_raw.columns if c not in influx]
        )
        X_raw = X_raw.drop(columns=weg)
        _say(
            f"Feature-Quelle '{FEATURE_SOURCE}': {len(weg)} Spalten entfernt "
            f"({len(influx)} Influx-Spalten insgesamt erkannt)  →  {X_raw.shape[1]}"
        )

    melt_end_ts = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    if SP["schmelzende_zeit"] + "_max" in df.columns:
        raw_ms = pd.to_numeric(df[SP["schmelzende_zeit"] + "_max"], errors="coerce")
        melt_end_ts = pd.to_datetime(
            raw_ms.where(raw_ms > 1.6e12), unit="ms", errors="coerce"
        )
        n_ok = int(melt_end_ts.notna().sum())
        _say(
            f"\nSchmelzende-Zeitstempel rekonstruiert: {n_ok}/{len(df)} Chargen "
            f"({melt_end_ts.min()} .. {melt_end_ts.max()})"
        )

    if TOP_FEATURES:
        present = [c for c in TOP_FEATURES if c in X_raw.columns]
        missing = [c for c in TOP_FEATURES if c not in X_raw.columns]
        X_raw = X_raw[present]
        _say(
            f"Feature-Selektion (TOP_FEATURES): {len(present)} von {len(TOP_FEATURES)} "
            f"angeforderten Features behalten"
        )
        if missing:
            _say(f"  Nicht im Datensatz gefunden ({len(missing)}):")
            _say(_fmt_cols(missing))

    if ADD_CYCLIC_FEATURES and melt_end_ts.notna().any():
        X_raw = X_raw.assign(
            melt_weekday=melt_end_ts.dt.dayofweek,
            melt_hour=melt_end_ts.dt.hour,
        )
        _say("Zyklische Features ergänzt: melt_weekday, melt_hour")

    _say(
        f"\nVerbleibende Roh-Features: {X_raw.shape[1]}  (Eingang fürs Feature-Processing)"
    )

    ids_path = test_ids_path()
    all_ids = batch_ids.tolist()
    if ids_path.exists():
        test_ids = set(json.loads(ids_path.read_text(encoding="utf-8")))
        missing = test_ids - set(all_ids)
        _say(
            f"Test-Set geladen aus {ids_path.name} (Ebene {TEST_LEVEL!r}): "
            f"{len(test_ids)} Billets"
            + (f"  ({len(missing)} davon nicht mehr im Datensatz)" if missing else "")
        )
    else:
        group_col = _TEST_LEVEL_GROUPS[TEST_LEVEL]
        if group_col is None:
            rng = np.random.RandomState(SPLIT_SEED)
            n_test = int(round(TEST_SIZE * len(all_ids)))
            test_ids = set(rng.choice(all_ids, size=n_test, replace=False).tolist())
        else:
            frame = df[["batch_id", group_col]].copy()
            test_ids = set(_draw_group_ids(frame, group_col, TEST_SIZE, SPLIT_SEED))
        ids_path.write_text(json.dumps(sorted(test_ids), indent=2), encoding="utf-8")
        _say(
            f"Test-Set NEU gezogen (Ebene {TEST_LEVEL!r}, Seed {SPLIT_SEED}) und "
            f"gespeichert: {len(test_ids)} Billets → {ids_path.name}"
        )

    test_mask = batch_ids.isin(test_ids).to_numpy()
    X_train_raw = X_raw[~test_mask].reset_index(drop=True)
    X_test_raw = X_raw[test_mask].reset_index(drop=True)
    y_train = y[~test_mask].reset_index(drop=True)
    y_test = y[test_mask].reset_index(drop=True)
    _say(f"Train: {len(X_train_raw)} | Test: {len(X_test_raw)}\n")

    return DataSplit(
        df=df,
        X_raw=X_raw,
        y=y,
        batch_ids=batch_ids,
        test_mask=test_mask,
        X_train_raw=X_train_raw,
        X_test_raw=X_test_raw,
        y_train=y_train,
        y_test=y_test,
        leakage_cols=leakage_cols,
        id_cols=id_cols,
        melt_end_ts=melt_end_ts,
    )


if __name__ == "__main__":
    draw_test_ids()
    load_split(verbose=True)
