"""28 — Spaltenbilanz bis zu den Rohmerkmalen.

In der Arbeit: Abschnitt 5.4: Tabelle 7 (Spaltenbilanz); Aufteilung der
Merkmale nach Quelle für Tabelle 1 und Abschnitt 5.3/5.8.

Ausgabe: reports/eda/28_spaltenbilanz.csv, runs/eda/28_spaltenbilanz.json.

    python eda/28_spaltenbilanz.py
"""

from __future__ import annotations

import pandas as pd

import _common as C

D = C.D


def main():
    kopf = D._setze_ziel(pd.read_csv(D.CSV_PATH, nrows=5, low_memory=False))
    n_roh = kopf.shape[1]
    ds = C.lade(alle_arten=True)

    namen = set(D._ID_FEATURES if D.DROP_ID_FEATURES else []) | set(
        D._TIME_PROXY_FEATURES if D.DROP_TIME_PROXIES else []
    )
    kennungen = [
        c
        for c in ds.id_cols
        if c in namen or any(c.startswith(p) for p in D._TIME_PROXY_PREFIXES)
    ]
    datum = [c for c in ds.id_cols if c not in kennungen]
    abgeleitet = [c for c in ds.X_raw.columns if c in D._ABGELEITET]
    influx = D.influx_spalten(ds.X_raw.columns)
    sql = [c for c in ds.X_raw.columns if c not in influx and c not in abgeleitet]

    schritte = [
        ("aufbereiteter Bestand", n_roh, []),
        ("− Zielgröße und Chargenkennung", n_roh - 2, ["y", "batch_id"]),
        ("− Ergebnisse der Nachbearbeitung und Prüfung", None, ds.leakage_cols),
        ("− Datumsfelder", None, datum),
        ("− Kennungen und versteckte Zeitstempel", None, kennungen),
        ("+ Wochentag und Stunde des Schmelzendes", None, abgeleitet),
    ]
    stand = n_roh - 2
    zeilen = []
    for text, n, spalten in schritte:
        if n is None:
            stand = stand + len(spalten) if text.startswith("+") else stand - len(spalten)
            n = stand
        zeilen.append({"Schritt": text, "Spalten": n, "Änderung": len(spalten), "spalten": spalten})
    assert stand == ds.X_raw.shape[1], (stand, ds.X_raw.shape[1])
    zeilen.append(
        {
            "Schritt": "davon Stammdaten / Prozessdaten / abgeleitet",
            "Spalten": f"{len(sql)} / {len(influx)} / {len(abgeleitet)}",
            "Änderung": "",
            "spalten": [],
        }
    )

    # Stammdatenspalten im Rohbestand, Kanalkennzahlen und Kopfangaben (Abschnitt 4.1.1)
    roh_influx = set(D.influx_spalten(list(kopf.columns)))
    sql_roh = [c for c in kopf.columns if c not in roh_influx]
    kennzahlen = [c for c in influx if c.endswith(tuple(D._AGG_SUFFIXES))]
    kopfangaben = [c for c in influx if c not in kennzahlen]
    quellen = {
        "stammdaten_roh": len(sql_roh),
        "stammdaten_entfernt": {
            z["Schritt"]: len([c for c in z["spalten"] if c in sql_roh]) for z in zeilen if z["spalten"]
        },
        "kanalkennzahlen": len(kennzahlen),
        "kopfangaben": kopfangaben,
    }

    t = pd.DataFrame(zeilen)
    print(t[["Schritt", "Spalten", "Änderung"]].to_string(index=False))
    print(f"\nStammdatenspalten im Rohbestand: {len(sql_roh)}, entfernt: {quellen['stammdaten_entfernt']}")
    print(f"Prozessdaten: {len(kennzahlen)} Kanalkennzahlen, {len(kopfangaben)} Kopfangaben")
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    t[["Schritt", "Spalten", "Änderung"]].to_csv(
        C.OUT_DIR / "28_spaltenbilanz.csv", index=False, encoding="utf-8"
    )
    C.speichere_json(
        "28_spaltenbilanz",
        {
            "tabelle": [{k: z[k] for k in ("Schritt", "Spalten", "Änderung")} for z in zeilen],
            "entfernt": {z["Schritt"]: z["spalten"] for z in zeilen if z["spalten"]},
            "stammdaten": sql,
            "prozessdaten": influx,
            "abgeleitet": abgeleitet,
            "quellen": quellen,
        },
    )


if __name__ == "__main__":
    main()
