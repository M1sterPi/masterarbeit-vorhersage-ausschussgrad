"""Lichtbogenende je Billet, in den exportierten Kurven (A) und im ungekappten
Influx-Fenster (B).

In der Arbeit: Lichtbogenende je Billet für Abschnitt 5.1 (ausgewertet in
eda/25, genutzt von eda/26).

Braucht KURVEN_DIR und die InfluxDB (Zugang über .env). Ergebnis:
data/_arc_end_scan.csv, ausgewertet in eda/25.

    python data_pipeline/arc_end_scan.py
"""

from __future__ import annotations

import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data_pipeline"))
import merge_billetv2 as mb  # noqa: E402
from data_prep import SP  # noqa: E402

CURVE_DIR = mb.OUTPUT_DIR
OUT = ROOT / "data" / "_arc_end_scan.csv"
CAP_H = float(mb.D.MAX_BILLET_DURATION_H)
TAIL_MIN = 5.0
SCHWELLE = {SP["lichtbogen_an"]: 0.5, SP["lichtbogen_leistung"]: 1.0, SP["lichtbogen_strom"]: 1.0}


def scan_csv(path: str) -> dict:
    bid = os.path.basename(path)[len("billet_") : -len(".csv")]
    rec = {"billet": bid}
    try:
        df = pd.read_csv(
            path,
            sep=";",
            decimal=",",
            usecols=lambda c: c in ("_time", *SCHWELLE),
            low_memory=False,
        )
    except Exception as e:  # noqa: BLE001
        rec["a_fehler"] = str(e)[:80]
        return rec
    t = pd.to_datetime(df["_time"], utc=True, format="ISO8601", errors="coerce")
    df = df.assign(_t=t).dropna(subset=["_t"]).sort_values("_t")
    if df.empty:
        rec["a_fehler"] = "leer"
        return rec
    t0, t1 = df["_t"].iloc[0], df["_t"].iloc[-1]
    rec["a_span_h"] = (t1 - t0).total_seconds() / 3600
    for col, thr in SCHWELLE.items():
        if col not in df.columns:
            continue
        v = pd.to_numeric(df[col], errors="coerce")
        an = df["_t"][v > thr]
        if len(an):
            rec[f"a_{col}_start_h"] = (an.iloc[0] - t0).total_seconds() / 3600
            rec[f"a_{col}_end_h"] = (an.iloc[-1] - t0).total_seconds() / 3600
            rec[f"a_{col}_tail_on"] = bool(
                (t1 - an.iloc[-1]).total_seconds() / 60 <= TAIL_MIN
            )
        last = v.dropna()
        rec[f"a_{col}_last"] = float(last.iloc[-1]) if len(last) else np.nan
    return rec


def influx_arc(client, bid: int, start, stop, machine) -> dict:
    s = start.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    e = stop.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    rec = {"billet": str(bid), "b_window_h": (stop - start).total_seconds() / 3600}
    for col, thr in SCHWELLE.items():
        base = f"""
        from(bucket: "{mb.INFLUX_BUCKET}")
          |> range(start: {s}, stop: {e})
          |> filter(fn: (r) => r.{SP['ofen_kennung']} == "{machine}" and r._field == "{col}")
          |> toFloat()
        """
        q_on = base + f"|> filter(fn: (r) => r._value > {thr})"
        tabs = client.query_api().query_data_frame(
            q_on + '|> group() |> reduce(identity: {lo: 0, hi: 0, n: 0},'
            ' fn: (r, accumulator) => ({lo: if accumulator.n == 0 then int(v: r._time)'
            ' else accumulator.lo, hi: int(v: r._time), n: accumulator.n + 1}))'
        )
        if isinstance(tabs, list):
            tabs = pd.concat(tabs) if tabs else pd.DataFrame()
        if len(tabs) and tabs["n"].iloc[0] > 0:
            lo = pd.Timestamp(int(tabs["lo"].iloc[0]), tz="UTC")
            hi = pd.Timestamp(int(tabs["hi"].iloc[0]), tz="UTC")
            rec[f"b_{col}_start_h"] = (lo - start).total_seconds() / 3600
            rec[f"b_{col}_end_h"] = (hi - start).total_seconds() / 3600
            rec[f"b_{col}_n"] = int(tabs["n"].iloc[0])
    return rec


def main() -> None:
    files = sorted(glob.glob(os.path.join(CURVE_DIR, "billet_*.csv")))
    ids = [os.path.basename(f)[len("billet_") : -len(".csv")] for f in files]
    print(f"{len(files)} Kurven-CSVs", flush=True)

    if not (mb.INFLUX_TOKEN and mb.INFLUX_ORG and mb.INFLUX_BUCKET):
        raise SystemExit("INFLUX_TOKEN/ORG/BUCKET fehlen: in .env eintragen (Vorlage .env.example).")
    mb.MAX_BILLET_DURATION = None
    from influxdb_client import InfluxDBClient

    b_rows = []
    with InfluxDBClient(
        url=mb.INFLUX_URL, token=mb.INFLUX_TOKEN, org=mb.INFLUX_ORG, timeout=600_000
    ) as client:
        bounds = mb.load_all_billet_boundaries(client)
        todo = [(int(i), bounds[int(i)]) for i in ids if int(i) in bounds]
        print(f"B: {len(todo)} Billets mit Influx-Grenzen", flush=True)
        with ThreadPoolExecutor(6) as ex:
            futs = [
                ex.submit(influx_arc, client, b, d["start"], d["stop"], d[SP["ofen_kennung"]])
                for b, d in todo
            ]
            for k, f in enumerate(as_completed(futs), 1):
                b_rows.append(f.result())
                if k % 250 == 0:
                    print(f"  B {k}/{len(todo)}", flush=True)
    b = pd.DataFrame(b_rows)

    a_rows = []
    with ProcessPoolExecutor(8) as ex:
        for k, r in enumerate(ex.map(scan_csv, files, chunksize=4), 1):
            a_rows.append(r)
            if k % 250 == 0:
                print(f"  A {k}/{len(files)}", flush=True)
    a = pd.DataFrame(a_rows)

    res = a.merge(b, on="billet", how="outer")
    res.to_csv(OUT, index=False)
    print(f"→ {OUT}", flush=True)


if __name__ == "__main__":
    main()
