"""22 — Anhangtabellen zur Kennzahlauswahl.

In der Arbeit: Abschnitt 7.3: Auswahl der untersuchten Kennzahlen je Kanal und
Block (Grundlage für Abbildung D.1).

Welche Kennzahlen je Kanal bzw. Block xai/14 und xai/15 untersucht haben.
Ausgabe: reports/xai/tabellen/anhang_kennzahlauswahl_<ebene>.csv/.md.

    python xai/22_anhang_kennzahlauswahl.py [artikel|auftrag|billet]
"""
import importlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TAB = ROOT / "reports" / "xai" / "tabellen"
sys.path[:0] = [str(ROOT / "xai")]

_gruppen = importlib.import_module("21_anhang_gruppen")
_tabellen = importlib.import_module("19_tabellen_sensor_stammdatenwerte")
BLOCK, DEUTSCH = _gruppen.BLOCK, _gruppen.DEUTSCH
de, vz = _tabellen.de, _tabellen.vz

EBENE_TEXT = {"artikel": "artikelrein", "auftrag": "auftragsrein", "billet": "billetweise"}


def nutzen(r):
    """Einzelpermutation mit KI; fett, wenn das KI null ausschliesst."""
    t = f"{vz(r.delta_mae_einzeln, 2)} [{vz(r.ki_lo, 2)}; {vz(r.ki_hi, 2)}]"
    return f"**{t}**" if r.ki_lo > 0 else t


def anteil(x):
    return de(100 * x, 0) + " %"


def tabelle(df, gruppe, anteil_sp, auswahl, name_gruppe, name_kennzahl, extra=None):
    """Eine Zeile je untersuchter Kennzahl; die nicht untersuchten stehen
    zusammengefasst in der letzten Spalte (Zahl, Summe ihrer Anteile, groesste
    davon)."""
    gesamt = df.groupby(gruppe, sort=False).shap_betrag.sum().sort_values(ascending=False)
    zeilen, csv = [], []
    for g in gesamt.index:
        t = df[df[gruppe] == g].sort_values("shap_betrag", ascending=False)
        gewaehlt, rest = t[t.Merkmal.isin(auswahl)], t[~t.Merkmal.isin(auswahl)]
        uebrig = "–"
        if len(rest):
            r0 = rest.iloc[0]
            uebrig = (f"{len(rest)} mit {anteil(rest[anteil_sp].sum())}; größte: "
                      f"{name_kennzahl(r0.Merkmal, g)} {anteil(r0[anteil_sp])}")
        for i, (_, r) in enumerate(gewaehlt.iterrows()):
            z = [name_gruppe(g) if i == 0 else "", name_kennzahl(r.Merkmal, g)]
            if extra:
                z.append(extra(r))
            z += [anteil(r[anteil_sp]), nutzen(r), uebrig if i == 0 else ""]
            zeilen.append(z)
        for _, r in t.iterrows():
            csv.append({"Gruppe": g, "Merkmal": r.Merkmal, "shap_betrag": r.shap_betrag,
                        "anteil": r[anteil_sp], "delta_mae_einzeln": r.delta_mae_einzeln,
                        "ki_lo": r.ki_lo, "ki_hi": r.ki_hi, "untersucht": r.Merkmal in auswahl})
    return zeilen, csv


def main(ebene="artikel"):
    S14 = importlib.import_module("14_sensorwerte")
    S15 = importlib.import_module("15_stammdatenwerte")
    A = pd.read_csv(TAB / f"14_sensorwerte_aggregate_{ebene}.csv")
    B = pd.read_csv(TAB / f"15_stammdatenwerte_merkmale_{ebene}.csv")
    wahl_a = {m for _, m in S14.fuehrende(A)}
    wahl_b = {m for _, m in S15.fuehrende(B)}

    for wahl, datei in ((wahl_a, "14_sensorwerte"), (wahl_b, "15_stammdatenwerte")):
        ist = set(pd.read_csv(TAB / f"{datei}_schwellen_{ebene}.csv").Merkmal)
        assert wahl == ist, f"{datei}: Auswahl weicht ab: {sorted(wahl ^ ist)}"

    za, ca = tabelle(A, "Kanal", "shap_anteil_kanal", wahl_a, lambda k: DEUTSCH.get(k, k),
                     lambda m, k: m[len(k) + 1:] if m.startswith(k + "_") else m)
    zb, cb = tabelle(B, "Block", "shap_anteil_block", wahl_b, lambda b: BLOCK.get(b, b),
                     lambda m, b: m,
                     extra=lambda r: "Artikelmerkmal" if r.artikelmerkmal else "variiert")
    kopf_a = ["Kanal", "Kennzahl", "Anteil", "ΔMAE einzeln [KI]", "nicht untersucht"]
    kopf_b = ["Block", "Merkmal", "Art", "Anteil", "ΔMAE einzeln [KI]", "nicht untersucht"]
    et = EBENE_TEXT[ebene]
    titel_a = (f"Auswahl der Sensorkennzahlen für Abschnitt 7.5, {et} ({len(wahl_a)} Kennzahlen "
               f"aus {A.Kanal.nunique()} Kanälen). Anteil: Anteil am mittleren SHAP-Betrag des "
               f"Kanals; untersucht wird die führende Kennzahl, die zweite ab "
               f"{de(100 * S14.ZWEITES_AB, 0)} %. ΔMAE einzeln: Anstieg des MAE, wenn nur diese "
               f"Kennzahl permutiert wird (PP, 95-%-KI über Artikel; fett: KI schließt null aus). "
               f"Nicht untersucht: Zahl der übrigen Kennzahlen des Kanals, ihr gemeinsamer Anteil "
               f"und die größte von ihnen.")
    titel_b = (f"Auswahl der Stammdatenmerkmale für Abschnitt 7.5, {et} ({len(wahl_b)} Merkmale "
               f"aus {B.Block.nunique()} Blöcken). Untersucht "
               f"werden je Block bis zu {S15.MAX_JE_BLOCK} Merkmale: das führende und weitere ab "
               f"{de(100 * S15.AB_ANTEIL, 0)} % Anteil. Art: Artikelmerkmal = im Artikel praktisch "
               f"konstant. Übrige Angaben wie in der vorigen Tabelle.")

    pd.DataFrame([{"Tabelle": "Sensoren", **c} for c in ca] + [{"Tabelle": "Stammdaten", **c} for c in cb]
                 ).to_csv(TAB / f"anhang_kennzahlauswahl_{ebene}.csv", index=False)
    md = []
    for titel, kopf, zeilen in ((titel_a, kopf_a, za), (titel_b, kopf_b, zb)):
        md += [f"*{titel}*", "", "| " + " | ".join(kopf) + " |", "|" + "---|" * len(kopf)]
        md += ["| " + " | ".join(z) + " |" for z in zeilen] + [""]
    (TAB / f"anhang_kennzahlauswahl_{ebene}.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"Sensoren: {len(wahl_a)} untersucht von {len(A)} | Stammdaten: {len(wahl_b)} von {len(B)}")
    print("geschrieben:", (TAB / f"anhang_kennzahlauswahl_{ebene}.md").relative_to(ROOT))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "artikel")
