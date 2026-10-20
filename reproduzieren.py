"""Alle Rechenschritte der Masterarbeit in der richtigen Reihenfolge.

In der Arbeit: startet alle Skripte, die Zahlen, Tabellen und Abbildungen der
Arbeit erzeugen, mit den Einstellungen, mit denen sie gerechnet wurden.

Jeder Schritt ist ein Aufruf eines Skripts aus data_pipeline/, eda/ oder xai/
mit den Umgebungsvariablen, mit denen er für die Arbeit lief. Damit ist an
einer Stelle festgehalten, was vorher nur durch Editieren von Konstanten oder
von Hand gesetzte Variablen eingestellt wurde (Bewertungsebene, z-Normierung,
Standard- statt getunter Parameter).

Gruppen, in dieser Reihenfolge:
  daten       SQL-Export → billets_master.csv, Influx → Rohkurven, Rohkurven →
              Merkmalstabelle v6, Lichtbogen-Scan, Testmengen     (nur mit --mit-daten)
  tuning      Optuna-Suche für XGBoost und CatBoost                (nur mit --mit-tuning)
  modelle     alle Modellläufe je Bewertungsebene (billet/random, auftrag/auftrag,
              artikel/artikel) → runs/runs_summary.csv, runs/single/
  auswertung  Bootstrap-Intervalle, Ergebnistabelle
  eda         eda/01 … eda/33
  xai         xai/01 … xai/25, je Ebene bzw. nur artikelrein

Aufrufe:
  python reproduzieren.py --liste                  # alle Schritte mit Bedarf und Dauer
  python reproduzieren.py --dry-run                # nur anzeigen, was liefe
  python reproduzieren.py --nur eda                # eine Gruppe
  python reproduzieren.py --nur eda/19 xai/16      # einzelne Schritte (Namensanfang)
  python reproduzieren.py --ab xai                 # ab einem Schritt bzw. einer Gruppe
  python reproduzieren.py --ohne-gpu --ohne-kurven # was ohne GPU und Rohkurven geht

Bedarf je Schritt: daten = Merkmalstabelle und Testmengen in data/ (immer),
kurven = Rohkurven in KURVEN_DIR oder der Kurven-Cache in data/cache/,
influx = laufende InfluxDB, sql = ERP-Export, gpu = CUDA-Grafikkarte
(TabPFN läuft auch auf der CPU, dann aber um ein Vielfaches langsamer).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Bewertungsebene → (TEST_LEVEL, CV_LEVEL), die "Diagonale" der Arbeit
EBENEN = {"billet": ("billet", "random"), "auftrag": ("auftrag", "auftrag"),
          "artikel": ("artikel", "artikel")}


@dataclass
class Schritt:
    gruppe: str
    name: str
    befehl: list[str]
    env: dict = field(default_factory=dict)
    bedarf: tuple = ()
    dauer: str = ""

    def zeile(self) -> str:
        env = " ".join(f"{k}={v}" for k, v in self.env.items())
        return f"{env + ' ' if env else ''}python {' '.join(self.befehl)}"


def schritte() -> list[Schritt]:
    s: list[Schritt] = []

    # ── daten ────────────────────────────────────────────────────────────────
    s += [
        Schritt("daten", "daten/sql_export", ["data_pipeline/convert_sql_db.py"],
                bedarf=("sql",)),
        Schritt("daten", "daten/influx_export", ["data_pipeline/merge_billetv2.py"],
                bedarf=("influx",)),
        Schritt("daten", "daten/merkmale", ["data_pipeline/prepare_data.py"],
                bedarf=("kurven",), dauer="ca. 3 h"),
        Schritt("daten", "daten/lichtbogen_scan", ["data_pipeline/arc_end_scan.py"],
                bedarf=("kurven", "influx")),
    ]
    # Zieht eine Testmenge nur, wenn ihre Datei fehlt; neu gezogene weichen von
    # denen der Arbeit ab.
    for ebene in EBENEN:
        s.append(Schritt("daten", f"daten/testmenge_{ebene}", ["data_pipeline/data_prep.py"],
                         env={"TEST_LEVEL": ebene}))

    # ── tuning ───────────────────────────────────────────────────────────────
    for modell in ("xgboost", "catboost"):
        s.append(Schritt("tuning", f"tuning/{modell}",
                         ["data_pipeline/tune.py", "--model", modell, "--trials", "100",
                          "--ziel", "mae"],
                         env={"TEST_LEVEL": "artikel", "CV_LEVEL": "artikel"}))

    # ── modelle ──────────────────────────────────────────────────────────────
    # HYDRA und FreshPRINCE: z-Normierung je Ebene wie in den Läufen der Arbeit
    znorm_sequenz = {"billet": "0", "auftrag": "0", "artikel": "1"}
    for ebene, (tl, cl) in EBENEN.items():
        e = {"TEST_LEVEL": tl, "CV_LEVEL": cl}
        s += [
            Schritt("modelle", f"modelle/xgboost_{ebene}", ["data_pipeline/xgboost_regression.py"],
                    env=e),
            Schritt("modelle", f"modelle/xgboost_standard_{ebene}",
                    ["data_pipeline/xgboost_regression.py"],
                    env={**e, "XGB_PARAMS": "default"}),
            Schritt("modelle", f"modelle/catboost_{ebene}", ["data_pipeline/catboost_regression.py"],
                    env=e),
            Schritt("modelle", f"modelle/tabpfn_{ebene}", ["data_pipeline/tabpfn_regression.py"],
                    env=e, bedarf=("gpu",)),
            Schritt("modelle", f"modelle/minirocket_hybrid_{ebene}",
                    ["data_pipeline/rocket_regression.py"],
                    env={**e, "HYBRID": "1", "ZNORM": "1"}, bedarf=("kurven",)),
            Schritt("modelle", f"modelle/minirocket_hybrid_ohne_znorm_{ebene}",
                    ["data_pipeline/rocket_regression.py"],
                    env={**e, "HYBRID": "1", "ZNORM": "0"}, bedarf=("kurven",)),
            Schritt("modelle", f"modelle/minirocket_kurven_{ebene}",
                    ["data_pipeline/rocket_regression.py"],
                    env={**e, "HYBRID": "0", "ZNORM": "1"}, bedarf=("kurven",)),
            Schritt("modelle", f"modelle/hydra_{ebene}", ["data_pipeline/hydra_regression.py"],
                    env={**e, "ZNORM": znorm_sequenz[ebene]}, bedarf=("kurven",)),
            Schritt("modelle", f"modelle/freshprince_{ebene}",
                    ["data_pipeline/freshprince_regression.py"],
                    env={**e, "ZNORM": znorm_sequenz[ebene]}, bedarf=("kurven",), dauer="ca. 2,5 h"),
            # hurdle rechnet nur die CV; die Testebene blieb in der Arbeit auf artikel.
            Schritt("modelle", f"modelle/hurdle_{ebene}", ["data_pipeline/hurdle_regression.py"],
                    env={"TEST_LEVEL": "artikel", "CV_LEVEL": cl}),
        ]

    # ── auswertung ───────────────────────────────────────────────────────────
    s += [
        Schritt("auswertung", "auswertung/bootstrap", ["data_pipeline/bootstrap_ci.py"]),
        Schritt("auswertung", "auswertung/ergebnistabelle", ["data_pipeline/ergebnistabelle.py"]),
    ]

    # ── eda ──────────────────────────────────────────────────────────────────
    # Laufzeiten, soweit in den Skripten vermerkt (gemessen auf einer Quadro T1000)
    eda_bedarf = {
        "11": ((), "4 min"), "12": ((), "7 min"), "13": (("kurven",), "8 min"),
        "14": (("kurven",), "2 min"), "15": ((), "8 min"), "16": (("kurven",), ""),
        "17": ((), "1 min"), "19": (("gpu",), "ca. 1 h"), "22": (("gpu", "kurven"), "ca. 2,5 h"),
        "26": (("kurven",), ""), "32": (("kurven",), ""),
    }
    for p in sorted((ROOT / "eda").glob("[0-9][0-9]_*.py")):
        nr = p.name[:2]
        bedarf, dauer = eda_bedarf.get(nr, ((), ""))
        s.append(Schritt("eda", f"eda/{p.stem}", [f"eda/{p.name}"], bedarf=bedarf, dauer=dauer))

    # ── xai ──────────────────────────────────────────────────────────────────
    def x(name, skript, ebene="artikel", bedarf=(), dauer="", args=(), **env):
        return Schritt("xai", f"xai/{name}", [f"xai/{skript}", *args],
                       env={"TEST_LEVEL": ebene, **env}, bedarf=bedarf, dauer=dauer)

    for ebene in EBENEN:  # Grundauswertung je Ebene; 07 vor 04/08/09/10/13
        s += [
            x(f"01_global_{ebene}", "01_global.py", ebene),
            x(f"02_lokal_{ebene}", "02_lokal.py", ebene),
            x(f"03_vergleich_{ebene}", "03_vergleich.py", ebene),
            x(f"07_kreuzvalidierung_{ebene}", "07_kreuzvalidierung.py", ebene),
            x(f"12_extreme_{ebene}", "12_extreme.py", ebene),
            x(f"04_verteilungen_{ebene}", "04_verteilungen.py", ebene),
            x(f"08_nutzen_{ebene}", "08_nutzen.py", ebene),
            x(f"09_redundanz_{ebene}", "09_redundanz.py", ebene),
            x(f"10_wirkungsform_{ebene}", "10_wirkungsform.py", ebene),
            x(f"13_sensoren_{ebene}", "13_sensoren.py", ebene),
        ]
    s += [  # nur artikelrein, wie in der Arbeit
        x("05_tabpfn_artikel", "05_tabpfn.py", bedarf=("gpu",), dauer="ca. 8 h"),
        x("06_tabpfn_vs_xgboost_artikel", "06_tabpfn_vs_xgboost.py"),
        x("11_pdp_ice_artikel", "11_pdp_ice.py", bedarf=("gpu",), dauer="ca. 6 min je Merkmal"),
    ]
    for ebene in EBENEN:  # 14/15 brauchen 13 für artikel UND auftrag
        s += [
            x(f"14_sensorwerte_{ebene}", "14_sensorwerte.py", ebene),
            x(f"15_stammdatenwerte_{ebene}", "15_stammdatenwerte.py", ebene),
        ]
    s += [
        x("16_gegenproben_artikel", "16_gegenproben.py"),
        x("17_herkunftsanteil", "17_herkunftsanteil.py"),
        x("18_tabelle_sensornutzen", "18_tabelle_sensornutzen.py"),
        x("19_tabellen_sensor_stammdatenwerte", "19_tabellen_sensor_stammdatenwerte.py"),
        x("20_tabelle_gegenproben", "20_tabelle_gegenproben.py"),
        x("21_anhang_gruppen", "21_anhang_gruppen.py"),
    ]
    for ebene in EBENEN:
        s.append(x(f"22_anhang_kennzahlauswahl_{ebene}", "22_anhang_kennzahlauswahl.py", ebene,
                   args=(ebene,)))
    s += [
        x("23_abbildungen_kapitel7", "23_abbildungen_kapitel7.py"),
        x("24_abbildungen_wertebereiche", "24_abbildungen_wertebereiche.py"),
        x("25_gestoerte_schmelzen", "25_gestoerte_schmelzen.py"),
    ]
    return s


def lade_env():
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--liste", action="store_true", help="Schritte anzeigen und beenden")
    ap.add_argument("--dry-run", action="store_true", help="Befehle zeigen, nicht ausführen")
    ap.add_argument("--nur", nargs="+", metavar="NAME", help="Gruppen oder Schritte (Namensanfang)")
    ap.add_argument("--ab", metavar="NAME", help="ab diesem Schritt bzw. dieser Gruppe")
    ap.add_argument("--mit-daten", action="store_true", help="Gruppe daten einschließen")
    ap.add_argument("--mit-tuning", action="store_true", help="Gruppe tuning einschließen")
    ap.add_argument("--ohne-gpu", action="store_true")
    ap.add_argument("--ohne-kurven", action="store_true")
    ap.add_argument("--ohne-influx", action="store_true")
    ap.add_argument("--weiter-bei-fehler", action="store_true")
    a = ap.parse_args()

    alle = schritte()
    wahl = alle
    if a.nur:
        wahl = [x for x in wahl if any(x.name.startswith(n) or x.gruppe == n for n in a.nur)]
    else:
        wahl = [x for x in wahl if (x.gruppe != "daten" or a.mit_daten)
                and (x.gruppe != "tuning" or a.mit_tuning)]
    if a.ab:
        start = next((i for i, x in enumerate(wahl)
                      if x.name.startswith(a.ab) or x.gruppe == a.ab), None)
        if start is None:
            raise SystemExit(f"Kein Schritt beginnt mit {a.ab!r}")
        wahl = wahl[start:]
    aus = {b for b, f in (("gpu", a.ohne_gpu), ("kurven", a.ohne_kurven),
                          ("influx", a.ohne_influx)) if f}
    uebersprungen = [x for x in wahl if aus & set(x.bedarf)]
    wahl = [x for x in wahl if not aus & set(x.bedarf)]

    if a.liste:
        for i, x in enumerate(alle, 1):
            bedarf = ", ".join(x.bedarf) or "–"
            print(f"{i:3d}  {x.name:45s} {bedarf:14s} {x.dauer:16s} {x.zeile()}")
        return

    lade_env()
    env_basis = {**os.environ, "MPLBACKEND": "Agg", "PYTHONUTF8": "1"}
    protokoll = []
    for i, x in enumerate(wahl, 1):
        print(f"\n[{i}/{len(wahl)}] {x.name}  ({x.dauer})\n    {x.zeile()}", flush=True)
        if a.dry_run:
            continue
        t0 = time.time()
        r = subprocess.run([sys.executable, *x.befehl], cwd=ROOT, env={**env_basis, **x.env})
        protokoll.append((x.name, r.returncode, time.time() - t0))
        if r.returncode != 0 and not a.weiter_bei_fehler:
            print(f"\nAbbruch: {x.name} endete mit Code {r.returncode}. "
                  f"Fortsetzen mit: python reproduzieren.py --ab {x.name}")
            break

    if uebersprungen:
        print(f"\nÜbersprungen wegen --ohne-*: {', '.join(x.name for x in uebersprungen)}")
    if protokoll:
        print("\nZusammenfassung:")
        for name, code, dt in protokoll:
            print(f"  {'ok ' if code == 0 else 'FEHLER'} {name:45s} {dt / 60:6.1f} min")
        if any(code for _, code, _ in protokoll):
            sys.exit(1)


if __name__ == "__main__":
    main()
