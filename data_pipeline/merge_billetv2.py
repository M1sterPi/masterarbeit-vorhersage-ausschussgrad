"""Export der Prozesskurven je Billet aus der InfluxDB.

In der Arbeit: Prozesskurven je Billet (Abschnitt 4.1.1) mit dem Zeitfenster
und seiner Neun-Stunden-Grenze aus Abschnitt 5.1.

Das Zeitfenster eines Billets reicht vom Wechsel der Billetnummer bis zum
nächsten Wechsel auf derselben Maschine, höchstens MAX_BILLET_DURATION_H
Stunden. Die Billetliste kommt aus data/billets_master.csv, geschrieben wird je
Billet KURVEN_DIR/billet_<nr>.csv; vorhandene Dateien werden übersprungen.
Zugang über .env.

    python data_pipeline/merge_billetv2.py
"""

from __future__ import annotations
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path

import pandas as pd
from influxdb_client import InfluxDBClient
from influxdb_client.client.warnings import MissingPivotFunction

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_prep as D  # noqa: E402
from data_prep import SP  # noqa: E402

warnings.simplefilter("ignore", MissingPivotFunction)

INFLUX_URL = os.environ.get("INFLUX_URL", "http://localhost:8086")
INFLUX_TOKEN = os.environ.get("INFLUX_TOKEN", "")
INFLUX_ORG = os.environ.get("INFLUX_ORG", "")
INFLUX_BUCKET = os.environ.get("INFLUX_BUCKET", "")

ROOT = Path(__file__).resolve().parent.parent
CSV_FILE = ROOT / "data" / "billets_master.csv"
CSV_SEPARATOR = ","
CSV_BILLET_COL = SP["billetnummer"]

OUTPUT_DIR = str(D.KURVEN_DIR)

LOOKBACK = "-900d"

MAX_WORKERS = 6

DEFAULT_DURATION = timedelta(hours=24)

MAX_BILLET_DURATION: timedelta | None = timedelta(hours=D.MAX_BILLET_DURATION_H)

KEEP_FIELDS: list[str] = []

TAG_COLS = [SP["ofen_kennung"], SP["ofen_typ"]]


def load_all_billet_boundaries(client: InfluxDBClient) -> dict[int, dict]:
    """EIN Query holt alle Datenpunkte des Billetnummer-Kanals."""
    query = f"""
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: {LOOKBACK})
      |> filter(fn: (r) => r._measurement == "{SP['billetnr_kanal']}")
      |> filter(fn: (r) => r._field == "{SP['billetnr_kanal']}")
      |> keep(columns: ["_time", "_value", "{SP['ofen_kennung']}"])
    """
    df = client.query_api().query_data_frame(query)
    if isinstance(df, list):
        df = pd.concat(df, ignore_index=True) if df else pd.DataFrame()
    if df.empty:
        return {}

    df = df.rename(columns={"_value": "billet_id", "_time": "time"})
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["billet_id"] = pd.to_numeric(df["billet_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["billet_id"])

    df = df.sort_values([SP["ofen_kennung"], "time"]).reset_index(drop=True)

    df["prev_billet"] = df.groupby(SP["ofen_kennung"])["billet_id"].shift()
    is_start = (df["billet_id"] != df["prev_billet"]).fillna(True)
    starts = df[is_start].copy()

    starts["stop"] = starts.groupby(SP["ofen_kennung"])["time"].shift(-1)
    starts = starts.rename(columns={"time": "start"})

    starts["stop"] = starts["stop"].fillna(starts["start"] + DEFAULT_DURATION)

    if MAX_BILLET_DURATION is not None:
        cap_limit = starts["start"] + MAX_BILLET_DURATION
        n_capped = int((starts["stop"] > cap_limit).sum())
        starts["stop"] = starts["stop"].clip(upper=cap_limit)
        print(f"  → {n_capped} Fenster auf max. {MAX_BILLET_DURATION} begrenzt")

    result = (
        starts[["billet_id", SP["ofen_kennung"], "start", "stop"]]
        .drop_duplicates(subset=["billet_id"], keep="first")
        .set_index("billet_id")
    )
    return result.to_dict("index")


def fetch_influx_data(
    client: InfluxDBClient,
    billet_id: int,
    start: pd.Timestamp,
    stop: pd.Timestamp,
    machine: str,
) -> pd.DataFrame:
    """Lädt Prozessdaten als Wide-DataFrame (ein Pivot pro Billet)."""
    start_str = start.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    stop_str = stop.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    field_filter = ""
    if KEEP_FIELDS:
        joined = " or ".join(f'r._field == "{f}"' for f in KEEP_FIELDS)
        field_filter = f"|> filter(fn: (r) => {joined})"

    query = f"""
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: {start_str}, stop: {stop_str})
      |> filter(fn: (r) => r.{SP['ofen_kennung']} == "{machine}")
      {field_filter}
      |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
    """
    tables = client.query_api().query_data_frame(query)
    if isinstance(tables, list):
        if not tables:
            return pd.DataFrame()
        df = pd.concat(tables, ignore_index=True)
    else:
        df = tables

    if df.empty:
        return pd.DataFrame()

    drop_cols = [
        c
        for c in ["result", "table", "_start", "_stop", "_measurement"]
        if c in df.columns
    ]
    df = df.drop(columns=drop_cols)
    df["_time"] = pd.to_datetime(df["_time"], utc=True)

    if SP["billetnr_kanal"] not in df.columns:
        return pd.DataFrame()

    billet_numeric = pd.to_numeric(df[SP["billetnr_kanal"]], errors="coerce")
    if not (billet_numeric == float(billet_id)).any():
        return pd.DataFrame()

    tag_cols = [c for c in TAG_COLS if c in df.columns]
    group_cols = ["_time"] + tag_cols
    if df.duplicated(subset=["_time"]).any():
        df = (
            df.sort_values(group_cols)
            .groupby(group_cols, as_index=False, sort=False)
            .first()
        )

    return df.sort_values("_time").reset_index(drop=True)


def load_csv_data() -> pd.DataFrame:
    df = pd.read_csv(
        CSV_FILE, sep=CSV_SEPARATOR, dtype={CSV_BILLET_COL: str}, encoding="utf-8-sig"
    )
    df = df.dropna(subset=[CSV_BILLET_COL])
    df = df[df[CSV_BILLET_COL].str.strip() != ""]
    df[CSV_BILLET_COL] = df[CSV_BILLET_COL].str.strip()
    return df


def process_one_billet(
    client: InfluxDBClient,
    billet_id_str: str,
    boundaries: dict[int, dict],
) -> tuple[str, str]:
    """Verarbeitet ein Billet."""
    filename = os.path.join(OUTPUT_DIR, f"billet_{billet_id_str}.csv")
    if os.path.exists(filename):
        return ("exists", filename)

    try:
        billet_int = int(billet_id_str)
    except ValueError:
        return ("no_influx", f"Billet-ID '{billet_id_str}' ist nicht numerisch")

    info = boundaries.get(billet_int)
    if info is None:
        return ("no_influx", "nicht in Influx gefunden")

    start, stop, machine = info["start"], info["stop"], info[SP["ofen_kennung"]]
    influx_rows = fetch_influx_data(client, billet_int, start, stop, machine)

    if influx_rows.empty:
        return ("no_data", "keine Messdaten")

    influx_rows.to_csv(filename, index=False, sep=";", decimal=",")
    return ("ok", f"{len(influx_rows):,} Zeilen → {filename}")


def export_csvs(client: InfluxDBClient, csv_df: pd.DataFrame):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    billet_ids = csv_df[CSV_BILLET_COL].unique().tolist()
    total = len(billet_ids)
    print(f"Gefundene Billets in CSV: {total}")

    print("Lade Billet-Boundaries aus Influx (1 Query)...")
    boundaries = load_all_billet_boundaries(client)
    print(f"  → {len(boundaries)} Billet-Zeitfenster ermittelt\n")

    stats = {"ok": 0, "exists": 0, "no_influx": 0, "no_data": 0, "error": 0}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {
            ex.submit(process_one_billet, client, bid, boundaries): bid
            for bid in billet_ids
        }
        for i, fut in enumerate(as_completed(futures), 1):
            bid = futures[fut]
            try:
                status, msg = fut.result()
            except Exception as e:
                status, msg = "error", f"{type(e).__name__}: {e}"
            stats[status] = stats.get(status, 0) + 1
            tag = {
                "ok": "OK",
                "exists": "SKIP",
                "no_influx": "NO_INFLUX",
                "no_data": "NO_DATA",
                "error": "ERROR",
            }.get(status, status.upper())
            print(f"[{i}/{total}] {bid}: [{tag}] {msg}")

    print(
        f"\nFertig: {stats['ok']} exportiert | "
        f"{stats['exists']} bereits vorhanden | "
        f"{stats['no_influx']} nicht in Influx | "
        f"{stats['no_data']} keine Daten | "
        f"{stats['error']} Fehler"
    )


def main():
    if not (INFLUX_TOKEN and INFLUX_ORG and INFLUX_BUCKET):
        raise SystemExit("INFLUX_TOKEN/ORG/BUCKET fehlen: in .env eintragen (Vorlage .env.example).")
    print("Lade CSV-Daten...")
    csv_df = load_csv_data()
    print(f"  → {len(csv_df)} Zeilen geladen\n")

    with InfluxDBClient(
        url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG, timeout=500_000
    ) as client:
        export_csvs(client, csv_df)


if __name__ == "__main__":
    main()
