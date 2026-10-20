"""21 — Anhangtabellen zur Gruppenbildung.

In der Arbeit: Tabellen B.1 und B.3 (Anhang B, Gruppenbildung für die
Nutzenmessung).

Welche Spalten in xai/13 zu welcher Gruppe gehören.
Ausgabe: reports/xai/tabellen/anhang_gruppen_7_4.csv/.md.

    python xai/21_anhang_gruppen.py
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data_pipeline"))
import data_prep as D  # noqa: E402

TAB = ROOT / "reports" / "xai" / "tabellen"
EBENEN = ("artikel", "auftrag", "billet")
SUFFIXE = ["_mean", "_std", "_min", "_max", "_last", "_range", "_n_samples",
           "_known_frac", "_frac_on_t", "_n_edges"]

DEUTSCH = D.beschriftungen("xai21_deutsch")
BLOCK = {
    "Rezept/Programm": "Rezept/Programm", "Sollmasse": "Sollmaße", "Glas/Material": "Glas/Material",
    "Rohstoff/Charge": "Rohstoff/Charge",
    "uebrige Stammdaten": "übrige Stammdaten", "Influx-Kopf": "Kopfangaben Kurvendatei",
}
KENNZAHLEN = [
    ("_mean", "Mittelwert der geschriebenen Werte im Billet-Fenster", "alle"),
    ("_std", "Standardabweichung", "alle"),
    ("_min", "Minimum", "alle"),
    ("_max", "Maximum", "alle"),
    ("_last", "letzter geschriebener Wert im Fenster", "alle"),
    ("_range", "Spannweite (Maximum − Minimum)", "alle"),
    ("_n_samples", "Zahl der geschriebenen Werte", "alle"),
    ("_known_frac", "Anteil des Fensters, der durch geschriebene Werte abgedeckt ist", "alle"),
    ("_frac_on_t", "zeitgewichteter Anteil im Zustand 1 (Wert gilt bis zum nächsten Schreibvorgang)", "nur Schaltsignale (0/1)"),
    ("_n_edges", "Zahl der Zustandswechsel", "nur Schaltsignale (0/1)"),
]


def laden():
    """Vereinigung der Gruppen ueber alle Ebenen."""
    t = {e: pd.read_csv(TAB / f"13_sensoren_{e}.csv").set_index("Gruppe") for e in EBENEN}
    if not all(set(d.index) == set(t["artikel"].index) for d in t.values()):
        sys.exit("Die Gruppennamen unterscheiden sich zwischen den Ebenen.")
    zeilen = []
    for g in t["artikel"].index:
        je = {e: set(t[e].loc[g, "spalten"].split()) for e in EBENEN}
        alle = sorted(set.union(*je.values()), key=lambda c: (next(
            (i for e in EBENEN for i, x in enumerate(t[e].loc[g, "spalten"].split()) if x == c), 0), c))
        zeilen.append({"Gruppe": g, "art": t["artikel"].loc[g, "art"], "spalten": alle,
                       **{f"n_{e}": len(je[e]) for e in EBENEN},
                       "nicht_ueberall": sorted(c for c in alle if any(c not in je[e] for e in EBENEN)),
                       "je": je})
    return zeilen


def anzahl(z):
    n = [z[f"n_{e}"] for e in EBENEN]
    return str(n[0]) if len(set(n)) == 1 else " / ".join(map(str, n))


def kanal_basis(spalten):
    """Influx-Namen (ohne Kennzahl-Endung) und Endungen eines Kanals."""
    namen, endungen = set(), set()
    for c in spalten:
        s = next(s for s in sorted(SUFFIXE, key=len, reverse=True) if c.endswith(s))
        namen.add(c[: -len(s)]); endungen.add(s)
    return sorted(namen), [s for s in SUFFIXE if s in endungen]


def kompakt(spalten):
    """Paare chargeA__X/chargeK__X und postenA__X/postenK__X zusammenfassen."""
    rest, paare = [], {}
    for c in spalten:
        m = re.match(r"^(charge|posten)([AK])__(.+)$", c)
        if m:
            paare.setdefault(m.group(1), {}).setdefault(m.group(3), set()).add(m.group(2))
        else:
            rest.append(c)
    teile = [", ".join(rest)] if rest else []
    for p, felder in paare.items():
        beide = [f for f, s in felder.items() if s == {"A", "K"}]
        einzeln = [f"{p}{''.join(sorted(s))}__{f}" for f, s in felder.items() if s != {"A", "K"}]
        teile.append(f"{p}A__ und {p}K__ (je für Anode und Kathode): " + ", ".join(beide))
        if einzeln:
            teile.append(", ".join(einzeln))
    return "; ".join(teile)


def main():
    t = laden()
    zuordnung = [(c, z["Gruppe"], z["art"], *[c in z["je"][e] for e in EBENEN])
                 for z in t for c in z["spalten"]]
    pd.DataFrame(zuordnung, columns=["Spalte", "Gruppe", "Art", *[f"in_{e}" for e in EBENEN]]).to_csv(
        TAB / "anhang_gruppen_7_4.csv", index=False, encoding="utf-8")

    kan = sorted((z for z in t if z["art"] == "Kanal"), key=lambda z: z["Gruppe"].lower())
    a1 = [["Influx-Kanal", "Lesehilfe", "Spalten", "Kennzahlen"]]
    for z in kan:
        namen, end = kanal_basis(z["spalten"])
        satz = "Standardsatz (8)" if len(end) == 8 else "Standardsatz + _frac_on_t, _n_edges (10)"
        if len(namen) > 1:
            satz += f"; zwei Schreibweisen zusammengelegt ({', '.join(namen)})"
        a1.append([z["Gruppe"], DEUTSCH.get(z["Gruppe"], "–"), anzahl(z), satz])
    a2 = [["Endung", "Bedeutung", "gebildet für"]] + [[s, b, f] for s, b, f in KENNZAHLEN]
    st = [z for z in t if z["art"] != "Kanal"]
    a3 = [["Block", "Spalten", "Merkmale"]]
    for z in st:
        txt = kompakt(z["spalten"])
        if z["nicht_ueberall"]:
            fehlt = [e for e in EBENEN if z[f"n_{e}"] < len(z["spalten"])]
            txt += (f". Nur auftragsrein und billetweise im Modell (artikelrein in der ersten "
                    f"Trainingsfalte konstant): {', '.join(z['nicht_ueberall'])}") if fehlt == ["artikel"]                 else f". Nicht auf allen Ebenen im Modell: {', '.join(z['nicht_ueberall'])}"
        a3.append([BLOCK.get(z["Gruppe"], z["Gruppe"]), anzahl(z), txt])

    hinweis = " Spaltenzahl bei Abweichung je Ebene: artikelrein / auftragsrein / billetweise."
    tabellen = [
        (f"Tabelle A.1: Sensorgruppen in Abschnitt 7.4 ({len(kan)} Influx-Kanäle im Modell). "
         "Je Kanal werden alle Kennzahlen gemeinsam durchmischt.", a1),
        ("Tabelle A.2: Kennzahlen je Influx-Kanal, berechnet über das Zeitfenster eines Billets "
         "(data_pipeline/prepare_data.py).", a2),
        ("Tabelle A.3: Stammdatenblöcke und Kopfangaben der Kurvendateien in Abschnitt 7.4." + hinweis, a3),
    ]
    md = []
    for titel, rows in tabellen:
        md += [f"*{titel}*", "", "| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
        md += ["| " + " | ".join(z) + " |" for z in rows[1:]] + [""]
    (TAB / "anhang_gruppen_7_4.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print("geschrieben:", (TAB / "anhang_gruppen_7_4.md").relative_to(ROOT))
    for e in EBENEN:
        print(f"Spalten {e}: {sum(z[f'n_{e}'] for z in t)} "
              f"(Kanäle {sum(z[f'n_{e}'] for z in kan)}, Stammdaten/Kopf {sum(z[f'n_{e}'] for z in st)})")


if __name__ == "__main__":
    main()
