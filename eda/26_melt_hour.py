"""26 — Stunde des Schmelzendes.

In der Arbeit: nicht berichtet; Gegenprobe zu den abgeleiteten Merkmalen
Wochentag und Stunde (Abschnitt 5.4).

Prüft, ob das abgeleitete Merkmal melt_hour das Ende der eigenen Schmelze trifft.
Braucht data/_arc_end_scan.csv und KURVEN_DIR.
Ausgabe: reports/xai/tabellen/pruefe_melt_hour.csv.

    python eda/26_melt_hour.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data_pipeline"))
import data_prep as D  # noqa: E402
from data_prep import SP  # noqa: E402

KURVEN = D.KURVEN_DIR
ZIEL = ROOT / "reports" / "xai" / "tabellen" / "pruefe_melt_hour.csv"
TOLERANZ_H = 0.25


def fensterbeginn(nr: str):
    p = KURVEN / f"billet_{nr}.csv"
    if not p.exists():
        return pd.NaT
    with open(p, encoding="utf-8", errors="replace") as f:
        kopf = f.readline().rstrip("\n").split(";")
        zeile = f.readline().rstrip("\n").split(";")
    return pd.to_datetime(zeile[kopf.index("_time")], utc=True, format="mixed")


def boot_diff(d, a, maske, rng, n=1000):
    """Mittelwertunterschied von d (maske vs."""
    arts = np.unique(a)
    idx = {x: np.where(a == x)[0] for x in arts}
    w = d[maske].mean() - d[~maske].mean()
    bs = []
    for _ in range(n):
        i = np.concatenate([idx[x] for x in rng.choice(arts, len(arts))])
        m = maske[i]
        if m.any() and (~m).any():
            bs.append(d[i][m].mean() - d[i][~m].mean())
    return w, *np.percentile(bs, [2.5, 97.5])


def main():
    df = pd.read_csv(D.CSV_PATH, low_memory=False, usecols=lambda c: c in (
        SP["billetnummer"], SP["artikel"], SP["schmelzende_zeit"] + "_max"))
    nr_sp = SP["billetnummer"]
    scan = pd.read_csv(ROOT / "data" / "_arc_end_scan.csv", dtype={"billet": str})
    df[nr_sp] = df[nr_sp].astype(str).str.replace(r"\.0$", "", regex=True)
    d = df.merge(scan[["billet", f"a_{SP['lichtbogen_an']}_end_h"]], left_on=nr_sp, right_on="billet", how="left")
    d["start"] = pd.to_datetime([fensterbeginn(n) for n in d[nr_sp]], utc=True)
    tte = pd.to_numeric(d[SP["schmelzende_zeit"] + "_max"], errors="coerce")
    d["tte"] = pd.to_datetime(tte.where(tte > 1.6e12), unit="ms", utc=True)
    d["arc_ende"] = d.start + pd.to_timedelta(d[f"a_{SP['lichtbogen_an']}_end_h"], unit="h")
    d["abstand_h"] = (d.tte - d.arc_ende).dt.total_seconds() / 3600
    d["trifft"] = d.abstand_h.abs() <= TOLERANZ_H
    d["stunde_alt"] = d.tte.dt.hour
    d["stunde_neu"] = d.arc_ende.dt.hour
    ZIEL.parent.mkdir(parents=True, exist_ok=True)
    d.drop(columns=["billet"], errors="ignore").to_csv(ZIEL, index=False)

    ok = d.abstand_h.notna()
    print(f"Billets mit beiden Zeitpunkten: {ok.sum()} von {len(d)}")
    print(f"melt_hour trifft das Lichtbogenende (±{TOLERANZ_H * 60:.0f} min): "
          f"{d.trifft[ok].mean():.0%}")
    print("Abstand TTE_max − Lichtbogenende [h], Quantile 5/25/50/75/95 %:",
          d.abstand_h[ok].quantile([.05, .25, .5, .75, .95]).round(2).tolist())
    print(f"gleiche Stunde alt/neu: {(d.stunde_alt == d.stunde_neu)[ok].mean():.0%}")

    ds = D.load_split(verbose=False)
    m = ds.df.reset_index(drop=True)
    m[nr_sp] = m[nr_sp].astype(str).str.replace(r"\.0$", "", regex=True)
    y = pd.Series(np.asarray(ds.y), name="y") if len(np.asarray(ds.y)) == len(m) else None
    if y is None:
        print("y nicht zeilengleich verfügbar — Effektvergleich entfällt.")
        return
    m = m[[nr_sp, SP["artikel"]]].assign(y=y).merge(
        d[[nr_sp, "stunde_alt", "stunde_neu"]], on=nr_sp, how="left")
    a = m[SP["artikel"]].astype(str).to_numpy()
    y_art = (m.y - m.y.groupby(a).transform("mean")).to_numpy()
    rng = np.random.default_rng(42)
    for sp in ("stunde_alt", "stunde_neu"):
        ok = m[sp].notna().to_numpy()
        w, lo, hi = boot_diff(y_art[ok], a[ok], (m[sp][ok] <= 2).to_numpy(), rng)
        print(f"{sp}: 0–2 Uhr UTC n={int((m[sp][ok] <= 2).sum())}, y − Artikelmittel "
              f"gegen Rest {w:+.2f} [{lo:+.2f}; {hi:+.2f}]")
    print("geschrieben:", ZIEL)


if __name__ == "__main__":
    main()
