"""25 — Eigenständigkeit der kritischen Bereiche aus xai/14.

In der Arbeit: Abschnitt 7.2, 7.3 und 7.5: Überlappung der kritischen Bereiche,
Unterschiede ohne abgebrochene Schmelzen, Beitrag gegen Nutzen je Kanal.

Mehrere kritische Bereiche markieren dieselben Billets. Hier wird getrennt, was
eigenständig ist:

  A  Markierungen: abgebrochene Schmelzen (feste Schwelle wie in xai/14) und
     die kritischen Bereiche aller Kennzahlen der Schwellentabelle von xai/14.
  B  Überlappung der Markierungen (Billets je Paar).
  C  Unterschied im Ausschuss, kritischer Bereich minus Rest, roh, im Artikel
     und im Auftrag, 95-%-KI im Cluster-Bootstrap über Artikel, in fünf
     Teilmengen: alle, ohne Abbrüche, ohne den kritischen Bereich von BEZUG,
     ohne beide, nur Fenster an der Fenstergrenze (MAX_BILLET_DURATION_H).
  D  Nutzen gegen SHAP-Beitrag je Kanal aus xai/13 (artikelrein).

Nur die Datenseite; ob die Modelle auf dieselben Bereiche reagieren, rechnet
xai/16. Braucht xai/13 und xai/14 (artikelrein).

    python xai/25_gestoerte_schmelzen.py
"""

from __future__ import annotations

import importlib
import importlib.util
import json

import numpy as np
import pandas as pd

import _common as C
from data_prep import SP  # noqa: E402

S14 = importlib.import_module("14_sensorwerte")
S16 = importlib.import_module("16_gegenproben")
_spec = importlib.util.spec_from_file_location("eda_common", C.ROOT / "eda" / "_common.py")
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

BEZUG = SP["kanal_a"] + "_n_samples"  # zweite Gegenprobe neben den Abbrüchen
ABBRUCH = "Abbruch"
# Zeilen der Markdown-Tabelle; die CSV enthält alle Kennzahlen aus xai/14
TABELLE = [ABBRUCH, BEZUG, SP["kanal_q"] + "_std", SP["kanal_o"] + "_known_frac",
           SP["kanal_f"] + "_mean", SP["phase"] + "_std"]
TEILMENGEN = ["alle", "ohne Abbruch", "ohne Bezug", "ohne beide", "Fenstergrenze"]


def markierungen(X, schwellen) -> dict[str, np.ndarray]:
    """Kritischer Bereich je Kennzahl als Bool-Array, fehlende Werte gelten als nicht kritisch."""
    phase = pd.to_numeric(X[SP["phase"] + "_max"], errors="coerce").to_numpy(float)
    m = {ABBRUCH: np.nan_to_num(phase, nan=S14.PHASE_REGULAER) < S14.PHASE_REGULAER}
    for k, s in schwellen.iterrows():
        v = pd.to_numeric(X[k], errors="coerce").to_numpy(float)
        krit = (v <= s.schwelle) if s.richtung == "niedriger" else (v > s.schwelle)
        m[k] = krit & ~np.isnan(v)
    return m


def gueltig(k, X) -> np.ndarray:
    spalte = SP["phase"] + "_max" if k == ABBRUCH else k
    return pd.to_numeric(X[spalte], errors="coerce").notna().to_numpy()


def main():
    ds = E.lade(alle_arten=True)
    X, ys, df = E.daten_teil(ds)
    y = ys.to_numpy(float)
    art = df[SP["artikel"]].astype(str).to_numpy(dtype=str)
    auf = df[SP["auftrag"]].astype(str).to_numpy(dtype=str)
    dauer = pd.to_numeric(X["batch_duration_s"], errors="coerce").to_numpy(float)
    gekappt = dauer >= C.D.MAX_BILLET_DURATION_H * 3600 * 0.995

    schwellen = pd.read_csv(C.TAB_DIR / "14_sensorwerte_schwellen_artikel.csv").set_index("Merkmal")
    M = markierungen(X, schwellen)
    abbruch, bezugsbereich = M[ABBRUCH], M[BEZUG]

    namen = list(M)
    ueber = pd.DataFrame([[int((M[a] & M[b]).sum()) for b in namen] for a in namen], index=namen, columns=namen)
    print("Überlappung (Billets):")
    print(ueber.loc[TABELLE, TABELLE].to_string())

    # Gruppenmittel auf allen Billets, damit "im Artikel" in jeder Teilmenge dasselbe heißt
    bezug = {"roh": y - y.mean(), "im Artikel": S16.zentriert(y, art), "im Auftrag": S16.zentriert(y, auf)}
    maske = {"alle": np.ones(len(y), bool), "ohne Abbruch": ~abbruch, "ohne Bezug": ~bezugsbereich,
             "ohne beide": ~abbruch & ~bezugsbereich, "Fenstergrenze": gekappt}
    rng = np.random.default_rng(C.SEED)
    zeilen = []
    for k in namen:
        ok = gueltig(k, X)
        for tm in TEILMENGEN:
            sel = ok & maske[tm]
            a = M[k][sel]
            if a.sum() < 10 or (~a).sum() < 10:  # Teilmenge schließt die Markierung aus
                continue
            for b, werte in bezug.items():
                d = werte[sel]
                lo, hi = S16.boot_diff(d, a, art[sel], rng)
                zeilen.append({"Merkmal": k, "teilmenge": tm, "bezug": b, "n": int(sel.sum()),
                               "n_kritisch": int(a.sum()), "anteil_kritisch": float(a.mean()),
                               "unterschied": float(d[a].mean() - d[~a].mean()), "ki_lo": lo, "ki_hi": hi,
                               "gesichert": bool(lo > 0 or hi < 0)})
    T = pd.DataFrame(zeilen)

    g13 = json.loads((C.ERG_DIR / "13_sensoren_artikel.json").read_text(encoding="utf-8"))["gruppen"]
    nutzen = pd.DataFrame([{"Gruppe": g["Gruppe"], "delta_mae": g["delta_mae"], "ki_lo": g["ki_lo"],
                            "ki_hi": g["ki_hi"], "shap_summe": g["shap_summe"],
                            "nutzen_je_shap": g["delta_mae"] / g["shap_summe"] if g["shap_summe"] else np.nan}
                           for g in g13 if g["art"] == "Kanal"]).sort_values("delta_mae", ascending=False)

    pc = C.speichere_rangliste("25_gestoerte_schmelzen.csv", T, "_reihenfolge_behalten")
    md = tabelle_md(T)
    (C.TAB_DIR / "tabelle_25_gestoerte_schmelzen.md").write_text(md + "\n", encoding="utf-8")
    p = C.speichere("25_gestoerte_schmelzen.json", {
        "n": int(len(y)), "n_gekappt": int(gekappt.sum()),
        "anzahl_markiert": {k: int(v.sum()) for k, v in M.items()},
        "ueberlappung": ueber.to_dict(), "ergebnisse": T.to_dict("records"),
        "nutzen_je_shap": nutzen.to_dict("records"), "csv": pc.name,
    })
    print("\n" + md)
    print("geschrieben:", p)


def tabelle_md(T: pd.DataFrame) -> str:
    """im Artikel / im Auftrag über alle Billets und im Artikel je Gegenprobe; fett = KI ohne null."""
    vz = lambda x: f"{x:+.1f}".replace(".", ",").replace("-", "−")
    name = C.D.beschriftungen("xai25_namen")

    def zelle(k, tm, b):
        t = T[(T.Merkmal == k) & (T.teilmenge == tm) & (T.bezug == b)]
        if t.empty:
            return "–"
        r = t.iloc[0]
        s = f"{vz(r.unterschied)} [{vz(r.ki_lo)}; {vz(r.ki_hi)}]"
        return f"**{s}**" if r.gesichert else s

    kopf = ["Kennzahl: kritischer Bereich", "Billets", "im Artikel", "im Auftrag", "im Artikel ohne Abbrüche",
            "im Artikel ohne " + name.get(BEZUG, BEZUG), "im Artikel ohne beide", "im Artikel, Fenstergrenze"]
    zeilen = []
    for k in TABELLE:
        n = T[(T.Merkmal == k) & (T.teilmenge == "alle")].n_kritisch.iloc[0]
        zeilen.append([name.get(k, k), str(n), zelle(k, "alle", "im Artikel"), zelle(k, "alle", "im Auftrag"),
                       zelle(k, "ohne Abbruch", "im Artikel"), zelle(k, "ohne Bezug", "im Artikel"),
                       zelle(k, "ohne beide", "im Artikel"), zelle(k, "Fenstergrenze", "im Artikel")])
    return "\n".join(["| " + " | ".join(kopf) + " |", "|" + "---|" * len(kopf)]
                     + ["| " + " | ".join(z) + " |" for z in zeilen])


if __name__ == "__main__":
    main()
