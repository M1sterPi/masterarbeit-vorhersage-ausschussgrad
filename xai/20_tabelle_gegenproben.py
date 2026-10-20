"""20 — Tabelle zur Modell-Gegenprobe.

In der Arbeit: Tabelle 15 (Anhang C, Modell-Gegenprobe).

Aus xai/16. Ausgabe: reports/xai/tabellen/tabelle_16_gegenproben.md.

    python xai/20_tabelle_gegenproben.py
"""
import importlib
from pathlib import Path

import pandas as pd

_t19 = importlib.import_module("19_tabellen_sensor_stammdatenwerte")
vz, D = _t19.vz, _t19.D
SP = D.SP


def _zahl(x):
    """Schwelle lesbar: Tausenderpunkt, Dezimalkomma, keine überflüssigen Nullen."""
    nk = 0 if abs(x) >= 100 else (2 if abs(x) >= 1 else 3)
    s = f"{x:,.{nk}f}".replace(",", "§").replace(".", ",").replace("§", ".")
    return s.rstrip("0").rstrip(",") if "," in s else s


ROOT = Path(__file__).resolve().parent.parent
TAB = ROOT / "reports" / "xai" / "tabellen"
MERKMALE = [SP["kanal_a"] + "_n_samples", SP["phase"] + "_max", SP["kanal_o"] + "_known_frac", SP["kanal_b"] + "_std"]
# Vorlage ohne Eintrag in beschriftungen.json, je Richtung aus xai/16
ZEICHEN = {"hoeher": " > {schwelle}", "niedriger": " ≤ {schwelle}", "unter": " < {schwelle}"}
QUELLEN = ["beobachteter Ausschuss", "XGBoost", "CatBoost", "TabPFN"]
TITEL = ("Modell-Gegenprobe, artikelrein: Unterschied der kreuzvalidierten Vorhersage zwischen "
         "kritischem Bereich und Rest, über alle Billets (roh) und gegenüber dem Mittel des "
         "eigenen Artikels (im Artikel) bzw. Auftrags (im Auftrag), in Prozentpunkten mit 95-%-KI über Artikel; zum "
         "Vergleich derselbe Unterschied im beobachteten Ausschuss. Fett: KI schließt null aus. "
         "Erzeugt mit xai/16_gegenproben.py.")


def zelle(r):
    t = f"{vz(r.unterschied, 1)} [{vz(r.ki_lo, 1)}; {vz(r.ki_hi, 1)}]"
    return f"**{t}**" if r.ki_lo > 0 or r.ki_hi < 0 else t


def main():
    T = pd.read_csv(TAB / "16_gegenproben_artikel.csv")
    S = T.drop_duplicates("Merkmal").set_index("Merkmal")
    vorlage = D.beschriftungen("xai20_namen")
    NAME = {m: vorlage.get(m, m + ZEICHEN[S.loc[m, "richtung"]]).format(schwelle=_zahl(S.loc[m, "schwelle"]))
            for m in MERKMALE}
    kopf = ["Kennzahl: kritischer Bereich", "Quelle", "Unterschied roh", "Unterschied im Artikel",
            "Unterschied im Auftrag"]
    zeilen = []
    for k, name in NAME.items():
        for i, q in enumerate(QUELLEN):
            t = T[(T.Merkmal == k) & (T.quelle == q)].set_index("bezug")
            zeilen.append([name if i == 0 else "", q, zelle(t.loc["roh"]), zelle(t.loc["im Artikel"]),
                           zelle(t.loc["im Auftrag"])])
    md = "\n".join(["| " + " | ".join(kopf) + " |", "|" + "---|" * len(kopf)]
                   + ["| " + " | ".join(z) + " |" for z in zeilen])
    (TAB / "tabelle_16_gegenproben.md").write_text(md + "\n", encoding="utf-8")
    print(TITEL, "\n")
    print(md)


if __name__ == "__main__":
    main()
