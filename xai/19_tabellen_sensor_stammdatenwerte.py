"""19 — Tabellen der Schwellen.

In der Arbeit: Tabelle 16 (Anhang D, Kennwerte der Kanäle) und die
entsprechende Tabelle der Stammdaten.

Prozesskanäle und Stammdaten, artikelrein, aus xai/14 und xai/15.
Ausgabe: reports/xai/tabellen/tabelle_7_4_*.csv/.md, tabelle_7_5_*.csv/.md.

    python xai/19_tabellen_sensor_stammdatenwerte.py
"""
import contextlib
from decimal import ROUND_HALF_UP, Decimal
import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TAB = ROOT / "reports" / "xai" / "tabellen"
sys.path[:0] = [str(ROOT / "xai"), str(ROOT / "data_pipeline")]
import data_prep as D  # noqa: E402
from data_prep import SP  # noqa: E402

SENSOREN = {m: D.beschriftungen("xai19_sensoren").get(m, m) for m in [
    SP["kanal_a"] + "_n_samples", SP["kanal_q"] + "_std", SP["kanal_f"] + "_mean",
    SP["kanal_o"] + "_known_frac", SP["phase"] + "_std", SP["kanal_h"] + "_mean",
    SP["kanal_d"] + "_last", SP["lichtbogen_leistung"] + "_std", SP["kanal_c"] + "_min", SP["kanal_b"] + "_std",
]}
BLOCK = {"Rezept/Programm": "Rezept", "Sollmasse": "Sollmaße", "Glas/Material": "Glas/Material",
         "Rohstoff/Charge": "Rohstoff/Charge", "uebrige Stammdaten": "übrige"}
BEREICH = {}
AUSGESCHLOSSEN = {"melt_hour"}
KURZ_KAT = D.beschriftungen("xai19_kurz_kat")


def de(x, nk=None):
    """Zahl im Stil der Arbeit: Tausenderpunkt, Dezimalkomma, Stellen nach Groesse."""
    if not np.isfinite(x):
        return "–"
    a = abs(x)
    if nk is None:
        nk = 0 if a >= 100 else (1 if a >= 10 else (2 if a >= 1 else 3))
    q = Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nk if a < 10000 else 1),
                                          rounding=ROUND_HALF_UP)
    s = f"{float(q):,.{nk}f}".replace(",", "§").replace(".", ",").replace("§", ".")
    return s.replace("-", "−")


def vz(x, nk=1):
    return ("+" if x >= 0 else "") + de(x, nk) if np.isfinite(x) else "–"


def diff(s, k, kritisch_unten, nk=1, fett=True):
    w, lo, hi = s[f"d_{k}"], s[f"d_{k}_lo"], s[f"d_{k}_hi"]
    if kritisch_unten:
        w, lo, hi = -w, -hi, -lo
    txt = f"{vz(w, nk)} [{vz(lo, nk)}; {vz(hi, nk)}]"
    return f"**{txt}**" if fett and (lo > 0 or hi < 0) else txt


def bereich(s, kat=None):
    if s.name in BEREICH:
        return BEREICH[s.name]
    unten = s["richtung"] == "niedriger"
    if kat is not None:
        return kat
    t, a, b = s["schwelle"], s["schwelle_boot_p05"], s["schwelle_boot_p95"]
    band = "" if de(a) == de(b) else f" [{de(a)}; {de(b)}]"
    return f"{'≤' if unten else '>'} {de(t)}{band}"


def anteil(s):
    a = s["anteil_oberhalb"]
    return 1 - a if s["richtung"] == "niedriger" else a


def md_tabelle(kopf, zeilen):
    return "\n".join(["| " + " | ".join(kopf) + " |", "|" + "---|" * len(kopf)]
                     + ["| " + " | ".join(z) + " |" for z in zeilen])


def tabelle_74():
    S = pd.read_csv(TAB / "14_sensorwerte_schwellen_artikel.csv").set_index("Merkmal")
    fehlt = [m for m in SENSOREN if m not in S.index]
    if fehlt:
        sys.exit(f"In 14_sensorwerte_schwellen_artikel.csv fehlen: {fehlt}")
    kopf = ["Sensor: Kennzahl", "kritischer Bereich [Band der Schwelle]", "Anteil Billets",
            "Modell ΔSHAP", "Ausschuss artikel­bereinigt", "Ausschuss auftrags­bereinigt",
            "artikel­bereinigt ohne Abbrüche"]
    zeilen, csv = [], []
    for m, txt in SENSOREN.items():
        s = S.loc[m]
        u = s["richtung"] == "niedriger"
        zeilen.append([txt, bereich(s), f"{anteil(s):.0%}".replace("%", " %"),
                       vz(abs(s["d_shap"]), 2), diff(s, "y_artikel", u), diff(s, "y_auftrag", u),
                       diff(s, "y_artikel_ohne_abbruch", u)])
        csv.append({"Merkmal": m, "Bezeichnung": txt, **s.to_dict()})
    return kopf, zeilen, pd.DataFrame(csv)


def kritisch_zaehlen(S):
    """Billets und Artikel im kritischen Bereich (aus den Rohdaten)."""
    os.environ.setdefault("TEST_LEVEL", "artikel")
    with contextlib.redirect_stdout(io.StringIO()):
        import _common as C
        import importlib
        S15 = importlib.import_module("15_stammdatenwerte")
        ds = C.D.load_split(verbose=False)
    X = ds.X_raw.reset_index(drop=True)
    art = ds.df[SP["artikel"]].astype(str).reset_index(drop=True).to_numpy()
    aus = {}
    for m, s in S.iterrows():
        roh = X[m]
        if s["kategorial"]:
            k = roh.astype(str).where(roh.notna(), "(fehlt)")
            anteil_k = k.value_counts(normalize=True)
            k = k.where(~k.isin(anteil_k[anteil_k < S15.MIN_KAT].index), "übrige").to_numpy()
            krit = set(str(s["kategorien_oberhalb"]).split(", "))
            mk = np.isin(k, list(krit))
            if s["richtung"] == "niedriger":
                mk = ~mk
        else:
            v = pd.to_numeric(roh, errors="coerce").to_numpy(float)
            mk = (v <= s["schwelle"]) if s["richtung"] == "niedriger" else (v > s["schwelle"])
        aus[m] = (int(mk.sum()), int(len(np.unique(art[mk]))), len(art))
    return aus


def tabelle_75():
    S = pd.read_csv(TAB / "15_stammdatenwerte_schwellen_artikel.csv").set_index("Merkmal")
    S = S.drop(index=[m for m in AUSGESCHLOSSEN if m in S.index])
    zahl = kritisch_zaehlen(S)
    S["_ord"] = [(not a, -abs(d)) for a, d in zip(S.artikelmerkmal, S.d_y_roh)]
    S = S.sort_values("_ord")
    kopf = ["Block: Merkmal", "Streuung im Artikel", "kritischer Bereich [Band der Schwelle]",
            "Billets (Artikel)", "Modell ΔSHAP", "Ausschuss roh (zwischen Artikeln)",
            "Ausschuss artikel­bereinigt", "artikel­bereinigt ohne Abbrüche"]
    zeilen, csv = [], []
    for m, s in S.iterrows():
        u = s["richtung"] == "niedriger"
        kat = None
        if s["kategorial"]:
            kat = KURZ_KAT.get(m, s["kategorien_oberhalb"])
        n_b, n_a, n_ges = zahl[m]
        innen = f"{s['anteil_innen']:.0%}".replace("%", " %")
        if s["artikelmerkmal"]:
            ab, ohne = "– (Artikelmerkmal)", "–"
        else:
            ab, ohne = diff(s, "y_artikel", u), diff(s, "y_artikel_ohne_abbruch", u)
        zeilen.append([f"{BLOCK.get(s['Block'], s['Block'])}: {m}", innen, bereich(s, kat),
                       f"{n_b / n_ges:.0%} ({n_a})".replace("%", " %"),
                       vz(abs(s["d_shap"]), 2), diff(s, "y_roh", u), ab, ohne])
        csv.append({"Merkmal": m, "billets_kritisch": n_b, "artikel_kritisch": n_a, **s.to_dict()})
    return kopf, zeilen, pd.DataFrame(csv).drop(columns="_ord")


TITEL_74 = ("Schwellen der Sensorkennzahlen (XGBoost, artikelrein, kreuzvalidiert). Unterschied = "
            "kritischer Bereich minus Rest in Prozentpunkten, 95-%-KI über Artikel. Fett: KI "
            "schließt null aus. Rohwerte in der Einheit der Aufzeichnung.")
TITEL_75 = ("Schwellen der wichtigsten Stammdatenmerkmale je Block (XGBoost, artikelrein, "
            "kreuzvalidiert). Streuung im Artikel = Anteil der Streuung des Merkmals, der "
            "innerhalb der Artikel liegt (kategorial: Anteil der Billets, die vom häufigsten "
            "Wert ihres Artikels abweichen). Unterschied = kritischer Bereich minus Rest in "
            "Prozentpunkten, 95-%-KI über Artikel; fett: KI schließt null aus. Bei "
            "Artikelmerkmalen ist nur der Vergleich zwischen Artikeln möglich. Die Uhrzeit "
            "des Schmelzendes (melt_hour) fehlt, weil ihre Ableitung fehlerhaft ist.")


def main():
    k4, z4, c4 = tabelle_74()
    k5, z5, c5 = tabelle_75()
    md4, md5 = md_tabelle(k4, z4), md_tabelle(k5, z5)
    for name, md, c in (("tabelle_7_4_sensorwerte", md4, c4), ("tabelle_7_5_stammdatenwerte", md5, c5)):
        (TAB / f"{name}.md").write_text(md + "\n", encoding="utf-8")
        c.to_csv(TAB / f"{name}.csv", index=False, encoding="utf-8", float_format="%.6g")
    print(TITEL_74, "\n")
    print(md4, "\n")
    print(TITEL_75, "\n")
    print(md5)
    print(f"\ngeschrieben: {TAB.relative_to(ROOT)}/tabelle_7_4_sensorwerte.* und "
          "tabelle_7_5_stammdatenwerte.*")


if __name__ == "__main__":
    main()
