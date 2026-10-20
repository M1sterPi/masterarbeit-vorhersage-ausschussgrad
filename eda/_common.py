"""Gemeinsame Hilfen der EDA: Laden der Daten, Nullmodelle, Effektgrößen,
Kreuzvalidierung, Plot- und PDF-Helfer.

In der Arbeit: gemeinsame Hilfen der eda-Skripte, erzeugt selbst nichts für die
Arbeit.

NUR_TRAIN schaltet alle Auswertungen auf den Trainingsteil um (Standard: alle
Billets).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import re  # noqa: E402
import textwrap  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.3,
        "legend.frameon": False,
        "axes.titleweight": "normal",
    }
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data_pipeline"))

import data_prep as D  # noqa: E402
from data_prep import SP  # noqa: E402

OUT_DIR = ROOT / "reports" / "eda"
(ROOT / "reports" / "figures").mkdir(parents=True, exist_ok=True)

CSV_PATH = ROOT / "data" / "featurized_batches_trimmed_v6.csv"

Y_MIN, Y_MAX = D.Y_MIN, D.Y_MAX


def y_grenzen_text() -> str:
    """'[1; 100]' — fuer Fliesstext in den Textseiten."""

    def f(v):
        return f"{v:g}".replace(".", ",")

    return f"[{f(Y_MIN)}; {f(Y_MAX)}]"


GRUPPIERUNGEN = [
    (SP["kategorie"], "Kategorie"),
    ("Glastyp", "Glastyp"),
    ("Ofennummer", "Ofen"),
    (SP["ofen_kennung"], "Maschine"),
    (SP["artikel"], "Artikel"),
    (SP["auftrag"], "Auftrag"),
    (SP["rohstoff_charge_anode"], "Charge Anode"),
]

SEED = 42


def lade(alle_arten: bool = True):
    """load_split() auf v6, standardmäßig OHNE Kategorie-/Glastyp-Filter."""
    D.CSV_PATH = CSV_PATH
    if alle_arten:
        D.ONLY_CATEGORY = None
        D.ONLY_GLASTYP = None
        D.ONLY_MACHINE = None
    return D.load_split()


NUR_TRAIN = False


def eda_maske(ds) -> np.ndarray:
    """Boolesche Zeilenmaske über ds.df/ds.X_raw — für Tensoren und Zeitstempel."""
    if NUR_TRAIN:
        return ~ds.test_mask
    return np.ones(len(ds.test_mask), dtype=bool)


def umfang_text(n: int) -> str:
    """'alle n Billets' bzw. 'Train (n Billets)' für Titel und Fußnoten."""
    return f"Train ({n} Billets)" if NUR_TRAIN else f"alle {n} Billets"


def daten_teil(ds):
    """(X, y, df) für die Zeilen, auf denen die EDA rechnet — Regel 2."""
    m = eda_maske(ds)
    return (
        ds.X_raw.loc[m].reset_index(drop=True),
        ds.y.loc[m].reset_index(drop=True),
        ds.df.loc[m].reset_index(drop=True),
    )


def numerische_merkmale(X: pd.DataFrame, min_belegt: float = 0.3) -> list[str]:
    """Spalten, die sich sinnvoll gegen y plotten lassen."""
    out = []
    for c in X.columns:
        s = X[c]
        if s.dtype == object:
            s = D._to_num_de(s)
            if s.notna().mean() < 0.7:
                continue
        if s.notna().mean() >= min_belegt and s.nunique(dropna=True) > 2:
            out.append(c)
    return out


def als_zahl(s: pd.Series) -> pd.Series:
    return D._to_num_de(s) if s.dtype == object else s


def var_anteil(y: np.ndarray, gruppe: np.ndarray) -> float:
    """Anteil der y-Varianz zwischen den Gruppen (eta²)."""
    df = pd.DataFrame({"y": y, "g": gruppe}).dropna()
    if len(df) < 3 or df.g.nunique() < 2:
        return np.nan
    gm = df.groupby("g").y.transform("mean")
    tot = ((df.y - df.y.mean()) ** 2).sum()
    return float(((gm - df.y.mean()) ** 2).sum() / tot) if tot > 0 else np.nan


def var_anteil_mit_null(y, gruppe, n_perm: int = 200, seed: int = SEED) -> dict:
    """Beobachteter Varianzanteil, Nullverteilung und Überschuss — Regel 3."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    g = np.asarray(gruppe)
    ok = ~pd.isna(y) & ~pd.isna(g)
    y, g = y[ok], g[ok]
    beob = var_anteil(y, g)
    null = np.array([var_anteil(rng.permutation(y), g) for _ in range(n_perm)])
    return {
        "beobachtet": beob,
        "null_mittel": float(np.nanmean(null)),
        "null_std": float(np.nanstd(null)),
        "ueberschuss": beob - float(np.nanmean(null)),
        "n": int(len(y)),
        "gruppen": int(pd.Series(g).nunique()),
    }


def omega2(y, gruppe) -> float:
    """Erwartungstreue Effektstärke einer Gruppierung (omega²)."""
    g = pd.Series(np.asarray(gruppe, dtype=object))
    d = pd.DataFrame({"y": np.asarray(y, dtype=float), "g": g.to_numpy()}).dropna()
    d["g"] = d.g.astype(str)
    k, n = d.g.nunique(), len(d)
    if k < 2 or n <= k:
        return np.nan
    gm = d.groupby("g").y.transform("mean")
    ssb = ((gm - d.y.mean()) ** 2).sum()
    ssw = ((d.y - gm) ** 2).sum()
    msw = ssw / (n - k)
    return float((ssb - (k - 1) * msw) / (ssb + ssw + msw))


def bootstrap_ci(
    werte, statistik, n_boot: int = 1000, alpha: float = 0.05, seed: int = SEED
) -> tuple[float, float]:
    """Perzentil-Bootstrap-CI für eine beliebige Statistik — Regel 4."""
    rng = np.random.default_rng(seed)
    werte = np.asarray(werte)
    n = len(werte)
    if n < 10:
        return (np.nan, np.nan)
    schaetzer = [statistik(werte[rng.integers(0, n, n)]) for _ in range(n_boot)]
    return tuple(np.nanpercentile(schaetzer, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def spearman_tabelle(
    X: pd.DataFrame, ziel: pd.Series, spalten: list[str]
) -> pd.DataFrame:
    """Spearman rho je Merkmal gegen ein Ziel, FDR-korrigiert — Regeln 5 und 2."""
    from scipy.stats import spearmanr

    zeilen = []
    for c in spalten:
        s = als_zahl(X[c])
        ok = s.notna() & ziel.notna()
        if ok.sum() < 30 or s[ok].nunique() < 3:
            continue
        r, p = spearmanr(s[ok], ziel[ok])
        if np.isnan(r):
            continue
        zeilen.append({"merkmal": c, "rho": r, "p": p, "n": int(ok.sum())})
    if not zeilen:
        return pd.DataFrame(columns=["merkmal", "rho", "p", "q", "n"])
    t = pd.DataFrame(zeilen).sort_values("p").reset_index(drop=True)
    m = len(t)
    t["q"] = np.minimum.accumulate((t.p * m / (t.index + 1))[::-1])[::-1].clip(0, 1)
    return t.sort_values("rho", key=np.abs, ascending=False).reset_index(drop=True)


def within_tabelle(
    X: pd.DataFrame, y: pd.Series, gruppe: pd.Series, spalten: list[str]
) -> pd.DataFrame:
    """Rang-Korrelation INNERHALB der Gruppen, FDR-korrigiert."""
    from scipy.stats import t as t_vert

    g = gruppe.astype(str)
    zeilen = []
    for c in spalten:
        s = als_zahl(X[c])
        ok = s.notna() & y.notna()
        if ok.sum() < 30 or s[ok].nunique() < 3:
            continue
        d = pd.DataFrame({"x": s[ok].rank(), "y": y[ok].rank(), "g": g[ok]})
        d = d[d.g.map(d.g.value_counts()) >= 2]
        k = d.g.nunique()
        n = len(d)
        if n - k - 1 < 10:
            continue
        xc = d.x - d.groupby("g").x.transform("mean")
        yc = d.y - d.groupby("g").y.transform("mean")
        sxx, syy = float((xc**2).sum()), float((yc**2).sum())
        if sxx < 1e-9 * max(float(((d.x - d.x.mean()) ** 2).sum()), 1.0):
            zeilen.append({"merkmal": c, "rho": np.nan, "p": np.nan, "n": n,
                           "k": k, "in_konstant": True})
            continue
        r = float((xc * yc).sum() / np.sqrt(sxx * syy))
        df_ = n - k - 1
        tt = r * np.sqrt(df_ / max(1 - r**2, 1e-12))
        p = float(2 * t_vert.sf(abs(tt), df_))
        zeilen.append({"merkmal": c, "rho": r, "p": p, "n": n, "k": k,
                       "in_konstant": False})
    t = pd.DataFrame(zeilen)
    if t.empty:
        return pd.DataFrame(columns=["merkmal", "rho", "p", "q", "n", "k", "in_konstant"])
    getestet = t.p.notna()
    t["q"] = np.nan
    tp = t[getestet].sort_values("p")
    m = len(tp)
    q = np.minimum.accumulate((tp.p.to_numpy() * m / np.arange(1, m + 1))[::-1])[::-1]
    t.loc[tp.index, "q"] = np.clip(q, 0, 1)
    return t


def auftragsabweichung(y: pd.Series, auftrag: pd.Series) -> pd.Series:
    """y minus Auftragsmittelwert."""
    return y - y.groupby(auftrag).transform("mean")


_DEZIMAL = re.compile(r"(?<![\w.])(\d+)\.(\d+)(?!\d|\.\d)")


def de(text: str) -> str:
    """Dezimalpunkt -> Dezimalkomma, fuer beliebigen Beschriftungstext."""
    return _DEZIMAL.sub(r"\1,\2", text) if isinstance(text, str) else text


def p_text(p: float) -> str:
    """p-Wert fuer Text inkl. 'p': numerisch 0 wird zu 'p < 1e-16' statt 'p = 0'."""
    if p is None or np.isnan(p):
        return "n. b."
    return "p < 1e-16" if p < 1e-16 else f"p = {p:.2g}"


def zahl(x: float, nachkomma: int = 0) -> str:
    """Deutsches Zahlenformat mit schmalem Tausenderabstand."""
    s = f"{x:,.{nachkomma}f}"
    return s.replace(",", " ").replace(".", ",")


def _de_formatter(alt):
    if getattr(alt, "_de", False):
        return alt
    neu = FuncFormatter(lambda v, pos=None: de(alt(v, pos)))
    neu._de = True
    neu.get_offset = lambda: de(alt.get_offset()) if hasattr(alt, "get_offset") else ""
    neu.set_locs = alt.set_locs
    return neu


def aufraeumen(fig):
    """Letzter Schritt vor dem Speichern: Zahlen ins deutsche Format."""
    for ax in fig.axes:
        for achse in (ax.xaxis, ax.yaxis):
            achse.set_major_formatter(_de_formatter(achse.get_major_formatter()))
            achse.set_minor_formatter(_de_formatter(achse.get_minor_formatter()))
    for t in fig.findobj(Text):
        t.set_text(de(t.get_text()))


def robuste_grenzen(werte, faktor: float = 3.0) -> tuple[float, float]:
    """Achsengrenzen, die Einzelausreisser und Platzhalterwerte ausblenden."""
    w = pd.Series(np.asarray(werte, dtype=float)).replace([np.inf, -np.inf], np.nan).dropna()
    if w.empty:
        return (0.0, 1.0)
    q1, q3 = w.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        iqr = (w.quantile(0.99) - w.quantile(0.01)) or 1.0
    lo, hi = max(w.min(), q1 - faktor * iqr), min(w.max(), q3 + faktor * iqr)
    rand = (hi - lo) * 0.04 or 0.5
    return (lo - rand, hi + rand)


def ausserhalb(werte, grenzen) -> int:
    w = np.asarray(werte, dtype=float)
    return int(((w < grenzen[0]) | (w > grenzen[1])).sum())


def seite(pdf: PdfPages, titel: str, zeilen: int, spalten: int, groesse=(11.7, 8.3)):
    """Eine Querformat-A4-Seite mit Raster."""
    fig, axes = plt.subplots(zeilen, spalten, figsize=groesse)
    fig.suptitle(titel, fontsize=13, y=0.985)
    return fig, np.atleast_1d(axes).ravel()


def schliesse(pdf: PdfPages, fig, unten: str = ""):
    """Seite speichern und schliessen."""
    aufraeumen(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig)
    plt.close(fig)


def leer(ax):
    """Nicht genutztes Rasterfeld unsichtbar machen."""
    ax.axis("off")


_TEXT_BREITE = 140
_ZEILE = 0.0245
_OBEN, _UNTEN = 0.86, 0.06


def textseite(pdf: PdfPages, titel: str, absaetze: list[str]):
    """Reine Textseite — für Methodenhinweise und Vorbehalte direkt in der PDF."""
    zeilen = []
    absaetze = [de(a) for a in absaetze]
    for i, a in enumerate(absaetze):
        if not a.strip():
            if zeilen and zeilen[-1] != "":
                zeilen.append("")
            continue
        einzug = " " * (len(a) - len(a.lstrip(" ")))
        zeilen += textwrap.wrap(
            a.strip(),
            _TEXT_BREITE - len(einzug),
            initial_indent=einzug,
            subsequent_indent=einzug,
        ) or [""]
        folgt = absaetze[i + 1] if i + 1 < len(absaetze) else ""
        if not (einzug and folgt.startswith(" ") and folgt.strip()):
            zeilen.append("")
    while zeilen and not zeilen[-1].strip():
        zeilen.pop()

    pro_seite = int((_OBEN - _UNTEN) / _ZEILE)
    for i in range(0, max(len(zeilen), 1), pro_seite):
        fig = plt.figure(figsize=(11.7, 8.3))
        t = titel if i == 0 else f"{titel} (Fortsetzung)"
        fig.text(0.06, 0.93, t, fontsize=15, weight="bold", va="top")
        y = _OBEN
        for z in zeilen[i : i + pro_seite]:
            fig.text(0.06, y, z, fontsize=10, va="top")
            y -= _ZEILE
        pdf.savefig(fig)
        plt.close(fig)


def pdf_oeffnen(name: str) -> PdfPages:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return PdfPages(OUT_DIR / name)


ERG_DIR = ROOT / "runs" / "eda"


def _json_wert(x):
    """numpy-/pandas-Werte für json.dumps umwandeln."""
    if isinstance(x, dict):
        return {str(k): _json_wert(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_wert(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_json_wert(v) for v in x.tolist()]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return None if np.isnan(x) else x
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    return x


def speichere_json(name: str, daten: dict) -> Path:
    """Kennzahlen eines Skripts nach runs/eda/<name>.json (für Tabellen der Arbeit)."""
    import json

    ERG_DIR.mkdir(parents=True, exist_ok=True)
    ziel = ERG_DIR / f"{name}.json"
    ziel.write_text(
        json.dumps(_json_wert(daten), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"geschrieben: {ziel.relative_to(ROOT)}")
    return ziel


def kopfzeile(ds, alle_arten: bool) -> str:
    """Einheitliche Fußnote: worauf beruhen die Bilder dieser Seite."""
    n = int(eda_maske(ds).sum())
    return (
        f"{CSV_PATH.name} | {umfang_text(n)} | "
        f"{'alle Kategorien und Glastypen' if alle_arten else 'gefilterte Teilmenge'} | "
        f"Leckage-Spalten ausgeschlossen ({len(ds.leakage_cols)})"
    )


def folds_final(df: pd.DataFrame, n_splits: int = 5, seed: int = SEED):
    """Die Falten, auf denen die Arbeit berichtet: GroupKFold nach Artikel."""
    from sklearn.model_selection import GroupKFold

    gruppen = df[SP["artikel"]].astype(str).to_numpy()
    gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(gkf.split(df, groups=gruppen))


def getunte_params() -> dict:
    """Die Hyperparameter aus data/best_params_v3.json — dieselben wie im Lauf."""
    import json

    p = json.loads((D.BEST_PARAMS_PATH).read_text(encoding="utf-8"))["params"]
    return {**p, "n_jobs": -1, "verbosity": 0}


def r2(wahr, vorher) -> float:
    wahr, vorher = np.asarray(wahr, float), np.asarray(vorher, float)
    ss_res = ((wahr - vorher) ** 2).sum()
    ss_tot = ((wahr - wahr.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def mae(wahr, vorher) -> float:
    return float(np.mean(np.abs(np.asarray(wahr, float) - np.asarray(vorher, float))))


def oof_xgb(X, y, folds, nachbearbeiten=None, params: dict = None) -> np.ndarray:
    """XGBoost out-of-fold; process_features wird je Falte NUR auf Train gefittet."""
    import xgboost as xgb

    p = params or getunte_params()
    pred = np.full(len(y), np.nan)
    for tr, va in folds:
        Xtr, Xva, _ = D.process_features(X.iloc[tr], X.iloc[va])
        if nachbearbeiten is not None:
            Xtr, Xva = nachbearbeiten(Xtr, Xva)
        m = xgb.XGBRegressor(**p, eval_metric="rmse")
        m.fit(Xtr, y.iloc[tr])
        pred[va] = m.predict(Xva)
    return pred


def falten_streuung(y, pred, folds) -> tuple[float, float]:
    """(Mittel, Standardabweichung) der MAE ueber die Falten."""
    w = [mae(np.asarray(y)[va], np.asarray(pred)[va]) for _, va in folds]
    return float(np.mean(w)), float(np.std(w))


def vergleichstabelle(ax, zeilen, basis: str, spalte: str = "MAE", titel: str = ""):
    """Balken 'Differenz zur Basisvariante' mit Faltenstreuung als Fehlerbalken."""
    namen = [z["name"] for z in zeilen]
    werte = np.array([z[spalte.lower()] for z in zeilen], dtype=float)
    b = next(z for z in zeilen if z["name"] == basis)
    diff = werte - b[spalte.lower()]
    farben = [
        "#7f7f7f" if n == basis else ("#d62728" if d > 0 else "#2ca02c")
        for n, d in zip(namen, diff)
    ]
    pos = np.arange(len(namen))
    ax.barh(pos, diff, color=farben)
    ax.axvspan(
        -b["sd"],
        b["sd"],
        color="0.85",
        zorder=0,
        label="Faltenstreuung der Basis",
    )
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(pos)
    ax.set_yticklabels(namen, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f"Δ {spalte} (links = besser)")
    if titel:
        ax.set_title(titel, fontsize=10)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16))


def zahlenzeilen(zeilen, basis: str) -> list[str]:
    """Die Werte, die vergleichstabelle() zeichnet, als Zeilen für die Textseite."""
    b = next(z for z in zeilen if z["name"] == basis)
    return [
        f"  {z['name']}: MAE {z['mae']:.3f} (Δ {z['mae'] - b['mae']:+.3f}), "
        f"Faltenstreuung ± {z['sd']:.3f}, R² {z['r2']:+.4f}"
        for z in zeilen
    ]
