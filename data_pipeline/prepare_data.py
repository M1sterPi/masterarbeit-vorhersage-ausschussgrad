"""Merkmalstabelle aus den Rohkurven: je Billet die Zielgröße y und Kennzahlen
jedes Prozesskanals, verbunden mit den Stammdaten.

In der Arbeit: Merkmalstabelle aller Kapitel: Kennzahlenkatalog je Kanal
(Abschnitt 5.3, Tabelle B.2) aus den tatsächlich geschriebenen Werten.

Liest KURVEN_DIR und data/billets_master.csv, schreibt
data/featurized_batches_trimmed_v6.csv und die Kanalklassifikation
data/cache/<name>.process_cols.json. Ein abgebrochener Lauf setzt über
data/cache/<name>.progress.jsonl fort.

    python data_pipeline/prepare_data.py [--limit N] [--workers N] [--assemble-only]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_prep as D  # noqa: E402
from data_prep import SP  # noqa: E402

ROOT = Path(__file__).parent.parent

BATCH_DIR = D.KURVEN_DIR

OUT_CSV = ROOT / "data" / "featurized_batches_trimmed_v6.csv"

MASTER_CSV = ROOT / "data" / "billets_master.csv"

MAX_WORKERS = 4


def to_num_de(s):
    return pd.to_numeric(
        s.astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )


CLASSIFY_SAMPLE = 25

FORCE_META_COLS = {SP["masse_kanal"], SP["energie"], SP["kanal_m"], SP["kanal_n"]}


def to_num_any(s):
    """Numerisch, unabhängig vom Speicherformat."""
    txt = s.astype(str).str.strip()
    mapped = txt.replace(
        {
            "True": "1",
            "False": "0",
            "true": "1",
            "false": "0",
            "TRUE": "1",
            "FALSE": "0",
        }
    )
    return pd.to_numeric(mapped.str.replace(",", ".", regex=False), errors="coerce")


def _process_cols_einer_datei(path):
    """Kandidaten aus GENAU EINER CSV: (variierende Spalten, binäre, alle Spalten)."""
    df = pd.read_csv(path, sep=";", encoding="utf-8", low_memory=False)
    known_skip = {"_time", SP["masse_soll"], SP["masse_ist"]}
    variierend, binaer = set(), set()
    for c in [c for c in df.columns if c not in known_skip]:
        v = to_num_any(df[c]).dropna()
        if len(v) > 1 and v.nunique() > 1:
            variierend.add(c)
            if set(np.unique(v.to_numpy())) <= {0.0, 1.0}:
                binaer.add(c)
    return variierend, binaer, set(df.columns)


def detect_columns(paths, verbose=True):
    """PROCESS_COLS über eine STICHPROBE von CSVs statt aus einer einzigen."""
    paths = list(paths)
    pro_datei, bin_je_datei, spalten_je_datei = [], [], []
    for p in paths:
        pc, bn, cols = _process_cols_einer_datei(p)
        pro_datei.append(pc)
        bin_je_datei.append(bn)
        spalten_je_datei.append(cols)
        if verbose:
            print(
                f"  {Path(p).stem:24} {len(cols):>4} Spalten, {len(pc):>3} Kandidaten"
            )

    kandidaten = sorted(set().union(*pro_datei)) if pro_datei else []
    process_cols = [c for c in kandidaten if c not in FORCE_META_COLS]
    geblockt = [c for c in kandidaten if c in FORCE_META_COLS]
    binary_cols = [
        c
        for c in process_cols
        if all(c in bn for pc, bn in zip(pro_datei, bin_je_datei) if c in pc)
    ]

    if verbose:
        print(f"\n=== Spalten-Klassifikation über {len(paths)} CSVs ===")
        print(f"PROCESS_COLS: {len(process_cols)}")
        for c in process_cols:
            n_var = sum(1 for pc in pro_datei if c in pc)
            n_da = sum(1 for cols in spalten_je_datei if c in cols)
            print(f"    {c:32} variiert in {n_var:>2}/{n_da:<2}")
        if geblockt:
            print(
                f"\n  per FORCE_META_COLS als Metadatum festgeschrieben ({len(geblockt)}):"
            )
            for c in geblockt:
                n_var = sum(1 for pc in pro_datei if c in pc)
                n_da = sum(1 for cols in spalten_je_datei if c in cols)
                print(f"    {c:32} variiert in {n_var:>2}/{n_da:<2}")
        print(f"\n  davon binär (nur 0/1 in allen Dateien): {len(binary_cols)}")
        print(f"    {', '.join(binary_cols)}")
        print("=" * 48)

    return process_cols, binary_cols


def featurize_batch(path, process_cols, binary_cols=(), master_row=None):
    print("READING", path)
    df = pd.read_csv(path, sep=";", encoding="utf-8")
    df["_time"] = pd.to_datetime(
        df["_time"], errors="coerce", utc=True, format="ISO8601"
    )
    df = df.sort_values("_time").reset_index(drop=True)

    for c in process_cols:
        if c in df.columns:
            df[c] = to_num_any(df[c])

    masse_soll = master_row.get(SP["masse_soll"])
    masse_ist = master_row.get(SP["masse_ist"])
    if pd.isna(masse_soll) or masse_soll == 0 or pd.isna(masse_ist):
        return None
    y = (masse_soll - masse_ist) / masse_soll * 100

    influx_meta_cols = [
        c for c in df.columns if c not in set(process_cols) and c != "_time"
    ]
    meta = {
        c: df[c].dropna().iloc[0] if df[c].notna().any() else np.nan
        for c in influx_meta_cols
    }

    sql_meta = {c: v for c, v in master_row.items() if c != "batch_id"}

    if "Energie" in meta and "Energie" in sql_meta:
        sql_meta["Energie_prog"] = sql_meta.pop(SP["energie"])
    rest = set(meta) & set(sql_meta)
    if rest:
        raise ValueError(
            f"Unerwartete Namenskollision Influx/SQL: {sorted(rest)} — "
            f"bitte in prepare_data.featurize_batch eine Umbenennung festlegen"
        )
    meta.update(sql_meta)

    agg = {}

    for c in process_cols:
        if c not in df or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        s = df[c].dropna()
        if len(s) == 0:
            continue
        agg[f"{c}_mean"] = s.mean()
        agg[f"{c}_std"] = s.std()
        agg[f"{c}_min"] = s.min()
        agg[f"{c}_max"] = s.max()
        agg[f"{c}_last"] = s.iloc[-1]
        agg[f"{c}_range"] = s.max() - s.min()
        agg[f"{c}_n_samples"] = len(s)

    agg["batch_duration_s"] = (df["_time"].max() - df["_time"].min()).total_seconds()

    t_all = df["_time"]
    fenster = (t_all.max() - t_all.min()).total_seconds()
    for c in process_cols:
        if c not in df.columns:
            continue
        m = df[c].notna()
        if m.sum() < 2:
            continue
        tc = t_all[m].to_numpy(dtype="datetime64[ns]").astype("int64") / 1e9
        vc = df.loc[m, c].to_numpy()
        dt = np.diff(tc)
        beobachtet = float(dt.sum())

        agg[f"{c}_known_frac"] = beobachtet / fenster if fenster > 0 else np.nan

        if c in binary_cols and beobachtet > 0:
            agg[f"{c}_frac_on_t"] = float(dt[vc[:-1] == 1].sum() / beobachtet)
            agg[f"{c}_n_edges"] = int((np.diff(vc) != 0).sum())

    return {"batch_id": Path(path).stem, "y": y, **meta, **agg}


def _worker(args):
    """featurize_batch in einem eigenen Prozess; None → als _skip vermerken."""
    path, process_cols, binary_cols, master_row = args
    batch_id = Path(path).stem
    if master_row is None:
        return {"batch_id": batch_id, "_skip": "keine Zeile in billets_master.csv"}
    try:
        rec = featurize_batch(path, process_cols, binary_cols, master_row)
    except Exception as exc:
        return {"batch_id": batch_id, "_error": f"{type(exc).__name__}: {exc}"}
    if rec is None:
        return {"batch_id": batch_id, "_skip": "Soll- oder Ist-Masse fehlt oder 0"}
    return rec


def _json_default(o):
    """Nur Typ-Umwandlung für die JSONL-Zwischenablage, keine Wertänderung."""
    if isinstance(o, np.generic):
        return o.item()
    if o is pd.NaT or o is pd.NA:
        return None
    if isinstance(o, (pd.Timestamp, pd.Timedelta)):
        return str(o)
    return str(o)


def load_progress(jsonl_path):
    """Bereits featurisierte Chargen einlesen (batch_id -> Zeile)."""
    rows = {}
    if not jsonl_path.exists():
        return rows
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "batch_id" in rec:
                rows[rec["batch_id"]] = rec
    return rows


def assemble(jsonl_path, out_csv):
    """JSONL → CSV."""
    rows = load_progress(jsonl_path)
    ok = [r for r in rows.values() if "_skip" not in r and "_error" not in r]
    skipped = [r for r in rows.values() if "_skip" in r]
    failed = [r for r in rows.values() if "_error" in r]

    if not ok:
        print("Keine featurisierten Chargen in der JSONL — nichts zu schreiben.")
        return None

    ok.sort(key=lambda r: r["batch_id"])
    table = pd.DataFrame(ok)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)
    print(f"\n{len(table)} Chargen × {table.shape[1]} Features → {out_csv}")
    if skipped:
        print(f"{len(skipped)} übersprungen (kein Target) — wie bisher via return None")
    if failed:
        print(
            f"{len(failed)} FEHLER — für einen Retry diese Zeilen aus {jsonl_path.name} löschen:"
        )
        for r in failed[:10]:
            print(f"  {r['batch_id']}: {r['_error']}")
        if len(failed) > 10:
            print(f"  ... und {len(failed) - 10} weitere")
    return table


def main():
    ap = argparse.ArgumentParser(description="Billet-Roh-CSVs → featurized_batches CSV")
    ap.add_argument("--batch-dir", type=Path, default=BATCH_DIR)
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    ap.add_argument("--master", type=Path, default=MASTER_CSV)
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument(
        "--limit", type=int, default=0, help="nur die ersten N CSVs (Testlauf)"
    )
    ap.add_argument(
        "--assemble-only", action="store_true", help="nur CSV aus JSONL bauen"
    )
    args = ap.parse_args()

    cache_dir = args.out.parent / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = cache_dir / (args.out.stem + ".progress.jsonl")
    cols_path = cache_dir / (args.out.stem + ".process_cols.json")

    if args.assemble_only:
        assemble(jsonl_path, args.out)
        return

    csv_files = sorted(args.batch_dir.glob("billet_*.csv"))
    if not csv_files:
        raise SystemExit(f"Keine billet_*.csv in {args.batch_dir}")
    if args.limit:
        csv_files = csv_files[: args.limit]

    if cols_path.exists():
        saved = json.loads(cols_path.read_text(encoding="utf-8"))
        process_cols = saved["process_cols"]
        binary_cols = saved.get("binary_cols", [])
        print(
            f"Klassifikation aus {cols_path.name} übernommen "
            f"({len(process_cols)} PROCESS_COLS, davon {len(binary_cols)} binär)"
        )
    else:
        idx = np.linspace(0, len(csv_files) - 1, min(CLASSIFY_SAMPLE, len(csv_files)))
        sample = [csv_files[int(i)] for i in dict.fromkeys(idx.astype(int))]
        process_cols, binary_cols = detect_columns(sample)
        cols_path.parent.mkdir(parents=True, exist_ok=True)
        cols_path.write_text(
            json.dumps(
                {
                    "batch_dir": str(args.batch_dir),
                    "classified_from": [p.name for p in sample],
                    "n_classify_sample": len(sample),
                    "process_cols": process_cols,
                    "binary_cols": binary_cols,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    done = load_progress(jsonl_path)
    todo = [p for p in csv_files if p.stem not in done]
    print(
        f"\n{len(csv_files)} CSVs in {args.batch_dir} | "
        f"{len(done)} bereits erledigt | {len(todo)} offen | {args.workers} Worker"
    )
    if not todo:
        assemble(jsonl_path, args.out)
        return

    master = pd.read_csv(args.master, low_memory=False)
    master["batch_id"] = master["batch_id"].astype(str)
    master_rows = {r["batch_id"]: r for r in master.to_dict("records")}
    fehlend = [p.stem for p in todo if p.stem not in master_rows]
    print(
        f"Stammdaten: {len(master_rows)} Billets aus {args.master.name}"
        + (f" | {len(fehlend)} Kurvendateien ohne Stammzeile" if fehlend else "")
    )

    t_start = time.time()
    payload = [
        (str(p), process_cols, binary_cols, master_rows.get(p.stem)) for p in todo
    ]
    with open(jsonl_path, "a", encoding="utf-8") as fh:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(_worker, item) for item in payload]
            for i, fut in enumerate(as_completed(futures), start=1):
                rec = fut.result()
                fh.write(
                    json.dumps(rec, default=_json_default, ensure_ascii=False) + "\n"
                )
                fh.flush()

                elapsed = time.time() - t_start
                eta = elapsed / i * (len(todo) - i)
                status = rec.get("_error") or rec.get("_skip") or "ok"
                print(
                    f"[{i}/{len(todo)}] {rec['batch_id']} {status} | "
                    f"{elapsed / 60:.1f} min | ETA {eta / 60:.0f} min",
                    flush=True,
                )

    assemble(jsonl_path, args.out)


if __name__ == "__main__":
    main()
