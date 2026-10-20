"""Stammdaten je Billet aus den ERP-Exporten der Fertigungsdatenbank.

In der Arbeit: Stammdaten je Billet (Abschnitt 4.1.1), Eingang aller weiteren
Schritte.

Liest einen oder mehrere Exportordner (mit index.xml, neuester zuerst), führt die
Tabellen je Primärschlüssel zusammen und verknüpft jedes Billet mit Programm,
Rohstoffchargen und Artikeln. Ergebnis: data/billets_master.csv und das
Datenverzeichnis data/billets_master_columns.csv.

    python data_pipeline/convert_sql_db.py --export <neuester Export> [<älterer Export> ...]
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from data_prep import SP  # noqa: E402

SOURCES: list[Path] = []
ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = ROOT / "data" / "billets_master.csv"
OUT_DICT = ROOT / "data" / "billets_master_columns.csv"

ENCODING = "latin-1"
BILLET_KEY = SP["billetnummer"]


def read_schema(
    source: Path,
) -> dict[str, tuple[str, list[tuple[str, str]], list[str]]]:
    """index.xml -> {table: (filename, [(column, type), ...], [primary key, ...])};
    duplicate column names get a .1 suffix the way pandas would."""
    root = ET.parse(source / "index.xml").getroot()
    schema = {}
    for table in root.iter("Table"):
        cols, keys, seen = [], [], collections.Counter()
        for child in table.find("VariableLength"):
            name = child.findtext("Name")
            seen[name] += 1
            unique = name if seen[name] == 1 else f"{name}.{seen[name] - 1}"
            kind = (
                "date"
                if child.find("Date") is not None
                else "num"
                if child.find("Numeric") is not None
                else "str"
            )
            cols.append((unique, kind))
            if child.tag == "VariablePrimaryKey":
                keys.append(unique)
        schema[table.findtext("Name")] = (table.findtext("URL"), cols, keys)
    return schema


def to_number(s: pd.Series) -> pd.Series:
    """German formatting: '1.800,00' -> 1800.0, '-36,00' -> -36.0."""
    txt = s.str.strip()
    txt = txt.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(txt, errors="coerce")


def load_table(name: str, schema: dict, source: Path) -> pd.DataFrame:
    filename, cols, _ = schema[name]
    df = pd.read_csv(
        source / filename,
        sep=";",
        header=None,
        names=[c[0] for c in cols],
        dtype=str,
        quotechar='"',
        encoding=ENCODING,
        engine="python",
        keep_default_na=False,
    )
    for col, kind in cols:
        if kind == "num":
            df[col] = to_number(df[col])
        elif kind == "date":
            df[col] = pd.to_datetime(
                df[col].str.strip(), format="%d.%m.%Y", errors="coerce"
            ).dt.date
        else:
            df[col] = df[col].str.strip().replace({"": np.nan})
    return df


def load_merged(name: str, schemas: dict[Path, dict]) -> pd.DataFrame:
    """One table, read from every export in SOURCES and merged on its primary key."""
    frames, seen = [], set()
    for source in SOURCES:
        schema = schemas[source]
        if name not in schema:
            print(f"    {source.parent.name}: table {name} absent, skipped")
            continue
        if schema[name][1] != schemas[SOURCES[0]][name][1]:
            raise ValueError(
                f"{name}: {source} defines a different schema than "
                f"{SOURCES[0]}; merging would misalign columns"
            )

        df = load_table(name, schema, source)
        keys = schema[name][2]
        ident = (
            pd.MultiIndex.from_frame(df[keys])
            if len(keys) > 1
            else pd.Index(df[keys[0]])
        )
        fresh = df[~np.asarray(ident.isin(seen))]
        seen.update(ident)
        if source is not SOURCES[0]:
            print(
                f"    {source.parent.name}: +{len(fresh)} rows not in a newer "
                f"export ({len(df) - len(fresh)} superseded)"
            )
        frames.append(fresh)
    return pd.concat(frames, ignore_index=True)


def aggregate_charges(charges: pd.DataFrame) -> pd.DataFrame:
    """One row per raw-material lot, keyed by (article, lot number)."""
    key = [SP["endartikel"], SP["charge_nr"]]
    missing = [c for c in key if c not in charges.columns]
    if missing:
        raise KeyError(f"{SP['tabelle_chargen']} is missing {missing}")

    grouped = charges.groupby(key, dropna=True)

    spread = int((grouped.nunique(dropna=False) > 1).any(axis=1).sum())
    if spread:
        print(
            f"  WARNING: {spread} (article, lot) pairs carry differing "
            f"values across their rows and are averaged"
        )

    numeric = [c for c in charges.columns if charges[c].dtype.kind == "f"]
    text = [c for c in charges.columns if c not in numeric and c not in key]
    out = grouped[numeric].mean(numeric_only=True)
    if text:
        out = out.join(grouped[text].first())
    out["n_rows"] = grouped.size()
    return out


def aggregate_postings(postings: pd.DataFrame) -> pd.DataFrame:
    """Item-ledger rollup per raw-material lot: how much was booked and when."""
    key = [SP["endartikel"], SP["charge_nr"]]
    if any(c not in postings.columns for c in key):
        return pd.DataFrame()
    grouped = postings.dropna(subset=key).groupby(key)
    out = pd.DataFrame({"n_postings": grouped.size()})
    if SP["posten_menge"] in postings.columns:
        out["menge_sum"] = grouped[SP["posten_menge"]].sum()
        out["menge_negative"] = grouped[SP["posten_menge"]].apply(lambda s: s[s < 0].sum())
    if SP["posten_datum"] in postings.columns:
        out["erste_buchung"] = grouped[SP["posten_datum"]].min()
        out["letzte_buchung"] = grouped[SP["posten_datum"]].max()
    return out


def prefixed(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return df.add_prefix(prefix)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    aus_env = [Path(p) for p in os.environ.get("SQL_EXPORTS", "").split(";") if p.strip()]
    ap.add_argument(
        "--export",
        type=Path,
        nargs="+",
        default=aus_env or None,
        required=not aus_env,
        help="ERP export folder(s) with index.xml, newest first "
        "(default: SQL_EXPORTS from the environment)",
    )
    ap.add_argument(
        "--drop-empty",
        action="store_true",
        help="omit columns that are empty for every billet",
    )
    args = ap.parse_args()
    SOURCES[:] = args.export

    schemas = {source: read_schema(source) for source in SOURCES}
    schema = schemas[SOURCES[0]]
    print(
        f"schema: {len(schema)} tables from index.xml, "
        f"{len(SOURCES)} export(s): {', '.join(s.parent.name for s in SOURCES)}"
    )
    tables = {}
    for name in schema:
        print(f"  {name}")
        tables[name] = load_merged(name, schemas)
    for name, df in tables.items():
        print(f"  {name:22s} {df.shape[0]:6d} rows x {df.shape[1]:3d} cols")

    billets = tables[SP["tabelle_billets"]].copy()
    billets = billets[billets[BILLET_KEY].notna()]
    billets = billets.sort_values(BILLET_KEY, kind="stable")
    billets.insert(0, "batch_id", "billet_" + billets[BILLET_KEY].astype(str))
    master = billets.set_index(BILLET_KEY, drop=False)
    sources = {c: SP["tabelle_billets"] for c in master.columns}

    prog = tables[SP["tabelle_programme"]].drop_duplicates(SP["programm_schluessel"]).set_index(SP["programm_schluessel"])
    joined = prefixed(prog, "prog__")
    master = master.join(joined, on=SP["programmnummer"])
    sources.update({c: SP["tabelle_programme"] for c in joined.columns})
    hit = master["prog__" + SP["abmessung"]].notna().mean()
    print(
        f"\nprogram table joined on program number: {hit:.1%} of billets matched"
    )

    charges = aggregate_charges(tables[SP["tabelle_chargen"]])
    print(f"lot table collapsed to {len(charges)} distinct lots")
    for side, article_col, charge_col, prefix in (
        ("anode", SP["rohstoff_artikel_anode"], SP["rohstoff_charge_anode"], "chargeA__"),
        ("cathode", SP["rohstoff_artikel_kathode"], SP["rohstoff_charge_kathode"], "chargeK__"),
    ):
        if article_col not in master.columns or charge_col not in master.columns:
            continue
        joined = prefixed(charges, prefix)
        master = master.join(joined, on=[article_col, charge_col])
        sources.update({c: f"{SP['tabelle_chargen']} ({side})" for c in joined.columns})
        matched = master[f"{prefix}n_rows"].notna().mean()
        chem = master[prefix + SP["chemie_marker"]].notna().mean() if prefix + SP["chemie_marker"] in master else 0
        print(
            f"  {side:8s} lot matched for {matched:.1%} of billets; "
            f"chemistry present for {chem:.1%}"
        )

    artikel = tables[SP["tabelle_artikel"]].drop_duplicates(SP["artikel_schluessel"]).set_index(SP["artikel_schluessel"])
    for column, prefix in ((SP["endartikel"], "artikel__"), (SP["artikel"], "ingot__")):
        if column not in master.columns:
            continue
        joined = prefixed(artikel, prefix)
        master = master.join(joined, on=column)
        sources.update({c: f"Artikel ({column})" for c in joined.columns})

    postings = aggregate_postings(tables[SP["tabelle_posten"]])
    if len(postings):
        for article_col, charge_col, prefix in (
            (SP["rohstoff_artikel_anode"], SP["rohstoff_charge_anode"], "postenA__"),
            (SP["rohstoff_artikel_kathode"], SP["rohstoff_charge_kathode"], "postenK__"),
        ):
            if article_col not in master.columns or charge_col not in master.columns:
                continue
            joined = prefixed(postings, prefix)
            master = master.join(joined, on=[article_col, charge_col])
            sources.update({c: f"{SP['tabelle_posten']} ({charge_col})" for c in joined.columns})
        print(f"ledger table rolled up to {len(postings)} lots")

    for prefix in ("chargeA__", "chargeK__"):
        if prefix + SP["chemie_marker"] in master.columns:
            master[f"{prefix}has_chemistry"] = master[prefix + SP["chemie_marker"]].fillna(0) > 0
            sources[f"{prefix}has_chemistry"] = "derived"
        if prefix + SP["korngroesse"] in master.columns:
            master[f"{prefix}has_grainsize"] = master[prefix + SP["korngroesse"]].fillna(0) > 0
            sources[f"{prefix}has_grainsize"] = "derived"

    master = master.reset_index(drop=True)

    dictionary = pd.DataFrame(
        {
            "column": master.columns,
            "source_table": [sources.get(c, "derived") for c in master.columns],
            "dtype": [str(master[c].dtype) for c in master.columns],
            "fill_rate": [float(master[c].notna().mean()) for c in master.columns],
            "n_unique": [int(master[c].nunique(dropna=True)) for c in master.columns],
        }
    )

    empty = dictionary.loc[dictionary["fill_rate"] == 0, "column"].tolist()
    if args.drop_empty and empty:
        master = master.drop(columns=empty)
        dictionary = dictionary[dictionary["fill_rate"] > 0]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(OUT_CSV, index=False, encoding="utf-8")
    dictionary.to_csv(OUT_DICT, index=False, encoding="utf-8")

    print()
    print(f"billets       : {len(master)}")
    print(
        f"columns       : {master.shape[1]}  "
        + ", ".join(
            f"{k}={v}"
            for k, v in collections.Counter(dictionary["source_table"]).most_common()
        )
    )
    print(
        f"always empty  : {len(empty)}"
        + (" (dropped)" if args.drop_empty else " (kept)")
    )
    print(f"wrote {OUT_CSV}  and  {OUT_DICT}")


if __name__ == "__main__":
    sys.exit(main())
