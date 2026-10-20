"""33 — Belege für die Abgrenzung des Vorhersagezeitpunkts.

In der Arbeit: Abschnitt 5.4 (Datumsfelder gegen das Schmelzende, Soll- und
Ist-Masse, versteckte Zeitstempel) und 5.6 (größter Wert der
Lichtbogenspannung unterhalb der Kanalgrenze des Kurvenzweigs).

  A  je gesperrtem Datumsfeld: Rangkorrelation mit dem Schmelzende und
     mittlerer Abstand in Tagen
  B  XGBoost (artikelreine Falten) mit Soll- und Ist-Masse als Merkmalen
  C  XGBoost mit und ohne die Kanäle, die verdeckt eine Uhrzeit enthalten
     (data_prep._TIME_PROXY_PREFIXES), auf dem aktuellen Datenstand
  D  Lichtbogenspannung: größter Wert unter der Grenze aus rocket_regression

Ausgabe: runs/eda/33_vorhersagezeitpunkt.json.

    python eda/33_vorhersagezeitpunkt.py
"""

from __future__ import annotations

import sys

import pandas as pd
from scipy.stats import spearmanr

import _common as C

D = C.D
sys.path.insert(0, str(C.ROOT / "data_pipeline"))
import rocket_regression as R  # noqa: E402
from data_prep import SP  # noqa: E402

MASSEN = [SP["masse_soll"], SP["masse_ist"]]


def datumsfelder(ds, df, ende):
    namen = set(D._ID_FEATURES if D.DROP_ID_FEATURES else []) | set(
        D._TIME_PROXY_FEATURES if D.DROP_TIME_PROXIES else []
    )
    datum = [
        c for c in ds.id_cols
        if c not in namen and not any(c.startswith(p) for p in D._TIME_PROXY_PREFIXES)
    ]
    zeilen = []
    for c in dict.fromkeys(datum):
        s = df[c].astype("string").str.strip()
        de = s.str.match(r"^\d{1,2}\.\d{1,2}\.\d{4}", na=False)
        t = pd.to_datetime(s.where(~de), errors="coerce", format="ISO8601")
        t = t.fillna(pd.to_datetime(s.where(de), errors="coerce", format="%d.%m.%Y"))
        ok = t.notna() & ende.notna()
        if ok.sum() < 20:
            continue
        rho = spearmanr(t[ok].astype("int64"), ende[ok].astype("int64")).statistic
        tage = (t[ok] - ende[ok]).dt.total_seconds() / 86400
        zeilen.append(
            {"spalte": c, "n": int(ok.sum()), "rho": float(rho),
             "abstand_tage_mittel": float(tage.mean()), "abstand_tage_median": float(tage.median())}
        )
    return sorted(zeilen, key=lambda z: -abs(z["rho"]))


def guete(X, y, folds, params: dict = None) -> dict:
    p = C.oof_xgb(X, y, folds, params=params)
    return {"oof_mae": C.mae(y, p), "oof_r2": C.r2(y, p), "merkmale": int(X.shape[1])}


def main():
    ds = C.lade(alle_arten=True)
    X, y, df = C.daten_teil(ds)
    ende = pd.Series(pd.to_datetime(ds.melt_end_ts[C.eda_maske(ds)].to_numpy()))
    folds = C.folds_final(df)

    erg = {"A_datumsfelder": datumsfelder(ds, df, ende)}
    for z in erg["A_datumsfelder"]:
        print(f"  {z['spalte']:28s} rho {z['rho']:+.3f}  Abstand {z['abstand_tage_mittel']:+.1f} Tage (n={z['n']})")

    print("rechne: ohne Zusatzspalten ...")
    erg["basis"] = guete(X, y, folds)

    print("rechne: mit Soll- und Ist-Masse ...")
    X_m = X.drop(columns=[c for c in MASSEN if c in X.columns]).join(
        df[MASSEN].apply(C.als_zahl)
    )
    erg["B_mit_massen"] = guete(X_m, y, folds)
    erg["B_mit_massen_voreinstellung"] = guete(X_m, y, folds, params={"random_state": C.SEED})

    print("rechne: mit versteckten Zeitstempeln ...")
    zeit = [c for c in df.columns if any(c.startswith(p) for p in D._TIME_PROXY_PREFIXES)]
    X_z = X.drop(columns=[c for c in zeit if c in X.columns]).join(df[zeit].apply(C.als_zahl))
    erg["C_mit_zeitstempeln"] = {**guete(X_z, y, folds), "spalten": zeit}

    grenze = R.GRENZEN[SP["lichtbogen_spannung"]]
    v = C.als_zahl(df[SP["lichtbogen_spannung"] + "_max"])
    erg["D_lichtbogenspannung"] = {
        "grenze": grenze,
        "max_unter_grenze": float(v[v.abs() <= grenze].max()),
        "q999_unter_grenze": float(v[v.abs() <= grenze].quantile(0.999)),
        "billets_ueber_grenze": int((v.abs() > grenze).sum()),
    }

    for k in ("basis", "B_mit_massen", "B_mit_massen_voreinstellung", "C_mit_zeitstempeln"):
        print(f"  {k:28s} MAE {erg[k]['oof_mae']:.3f}  R² {erg[k]['oof_r2']:+.3f}  ({erg[k]['merkmale']} Merkmale)")
    print(f"  Lichtbogenspannung: {erg['D_lichtbogenspannung']}")
    C.speichere_json("33_vorhersagezeitpunkt", erg)


if __name__ == "__main__":
    main()
