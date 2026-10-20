"""24 — Abbildungen und Tabelle zu Wertebereichen und Schwellen.

In der Arbeit: Abschnitt 7.3: Abbildungen 20 und 21 und die Schwellentabelle.

Aus xai/14 und xai/15, artikelrein.
Ausgabe: reports/figures/kapitel7/abb_7_3_*.png, reports/xai/tabellen/kapitel7_3_klassen.csv,
tabelle_7_3_schwellen.csv/.md.

    python xai/24_abbildungen_wertebereiche.py
"""
from __future__ import annotations

import contextlib
import importlib
import io
import json
import locale
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import NullFormatter, ScalarFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ERG = ROOT / "runs" / "xai"
TAB = ROOT / "reports" / "xai" / "tabellen"
ZIEL = ROOT / "reports" / "figures" / "kapitel7"
os.environ["TEST_LEVEL"] = "artikel"
sys.path[:0] = [str(ROOT / "xai"), str(ROOT / "data_pipeline")]
import data_prep as D  # noqa: E402
from data_prep import SP  # noqa: E402

try:
    locale.setlocale(locale.LC_NUMERIC, "de_DE")
    plt.rcParams["axes.formatter.use_locale"] = True
except locale.Error:
    pass
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False})

ORANGE, BLAU, GRAU = "#e6550d", "#3182bd", "#4d4d4d"
L_SHAP, L_GES, L_ART, L_AUF = ("SHAP (Modell)", "y − Gesamtmittel (Daten)",
                               "y − Artikelmittel (Daten)", "y − Auftragsmittel (Daten)")

SENSOREN = [(m, D.beschriftungen("xai24_sensoren").get(m, m), skala, einheit, log)
            for m, skala, einheit, log in [
    (SP["kanal_a"] + "_n_samples", 1, "", True),
    (SP["kanal_c"] + "_min", 1, "", False),
    (SP["kanal_b"] + "_std", 1, "", False),
    (SP["kanal_f"] + "_mean", 1e3, ", in Tausend", False),
]]
STAMMDATEN = [
    (SP["wand_soll"], f"Sollwandstärke ({SP['wand_soll']})"),
    (SP["laenge_erwartet"], f"Erwartete Länge ({SP['laenge_erwartet']})"),
    (SP["granulatdichte_kathode"], SP["granulatdichte_kathode"]),
    ("Glastyp", "Glastyp"),
]


def laden():
    """Klassen der gezeigten Merkmale mit nachgetragenem y − Auftragsmittel."""
    with contextlib.redirect_stdout(io.StringIO()):
        S14 = importlib.import_module("14_sensorwerte")
        S15 = importlib.import_module("15_stammdatenwerte")
    j14 = json.loads((ERG / "14_sensorwerte_artikel.json").read_text(encoding="utf-8"))
    j15 = json.loads((ERG / "15_stammdatenwerte_artikel.json").read_text(encoding="utf-8"))
    assert j14["cv_level"] == j15["cv_level"] == "artikel", (j14["cv_level"], j15["cv_level"])

    K14 = pd.DataFrame(j14["klassen"])
    K14 = K14[K14.Merkmal.isin([m for m, *_ in SENSOREN])]
    K15 = pd.DataFrame(j15["klassen"])
    K15 = K15[K15.Merkmal.isin([m for m, _ in STAMMDATEN])]
    SW15 = pd.DataFrame(j15["schwellen"])
    print("trage y − Auftragsmittel aus den Daten nach (Prüfung gegen n und y − Artikelmittel) …", flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        if "y_auftragsbereinigt" not in K14.columns:
            K14 = S14.auftragsreihe_nachtragen(K14)
        if "y_auftragsbereinigt" not in K15.columns:
            K15 = S15.auftragsreihe_nachtragen(K15, SW15)
    return K14, K15, SW15.set_index("Merkmal")


def stil(ax, ylabel):
    ax.axhline(0, color="#999999", lw=0.6, zorder=0)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)
    if ylabel:
        ax.set_ylabel("Prozentpunkte", fontsize=7.5)


def linien(ax, x, K, gesamt=None, artikelmerkmal=False):
    ax.plot(x, K.shap_mittel, "o-", color=ORANGE, ms=3.2, lw=1.3, label=L_SHAP, zorder=4)
    if gesamt is not None:
        ax.plot(x, K.y_mittel - gesamt, "D-.", color="black", ms=2.8, lw=1.0, label=L_GES, zorder=3)
    im_auftrag = not np.allclose(K.y_auftragsbereinigt, 0, atol=1e-9)
    if not artikelmerkmal:
        ax.plot(x, K.y_artikelbereinigt, "s-", color=BLAU, ms=3, lw=1.2, label=L_ART, zorder=3)
        if im_auftrag:
            ax.plot(x, K.y_auftragsbereinigt, "^--", color=GRAU, ms=3.4, lw=1.2, label=L_AUF, zorder=2)
    return im_auftrag


def legende(fig, axe, reihenfolge):
    eintraege = {}
    for ax in axe.ravel():
        for h, l in zip(*ax.get_legend_handles_labels()):
            eintraege.setdefault(l, h)
    lab = [l for l in reihenfolge if l in eintraege]
    fig.legend([eintraege[l] for l in lab], lab, loc="lower center", ncol=3 if len(lab) == 3 else 2,
               fontsize=7.5, frameon=False, columnspacing=2.5)


def abb_sensoren(K14):
    fig, axe = plt.subplots(2, 2, figsize=(6.6, 6.0), squeeze=False)
    for i, (ax, (m, titel, skala, einheit, logx)) in enumerate(zip(axe.ravel(), SENSOREN)):
        k = K14[K14.Merkmal == m].sort_values("klasse")
        x = k.wert_median.to_numpy() / skala
        linien(ax, x, k)
        if logx:
            ax.set_xscale("log")
            ax.set_xticks([t for t in (20, 50, 100, 200, 300, 500) if x.min() * 0.9 <= t <= x.max() * 1.1])
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(titel, fontsize=8.5)
        ax.set_xlabel("Merkmalswert (Klassenmedian" + einheit + (", log. Skala)" if logx else ")"),
                      fontsize=7.5)
        stil(ax, i % 2 == 0)
    legende(fig, axe, [L_SHAP, L_ART, L_AUF])
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=1.6)
    return fig


def abb_stammdaten(K15, SW15):
    fig, axe = plt.subplots(2, 2, figsize=(6.6, 6.4), squeeze=False)
    for i, (ax, (m, titel)) in enumerate(zip(axe.ravel(), STAMMDATEN)):
        k = K15[K15.Merkmal == m].sort_values("klasse")
        s = SW15.loc[m]
        kat = bool(s.kategorial)
        x = np.arange(len(k)) if kat else k.wert_median.to_numpy()
        im_auftrag = linien(ax, x, k, gesamt=s.y_gesamt, artikelmerkmal=bool(s.artikelmerkmal))
        art = ("Artikelmerkmal" if s.artikelmerkmal
               else "variiert im Artikel" + ("" if im_auftrag else ", nicht im Auftrag"))
        ax.set_title(f"{titel}\n({art})", fontsize=8.5)
        if kat:
            ax.set_xticks(x, [c[:16] for c in k.kategorien], rotation=30, ha="right", fontsize=7)
            ax.set_xlabel("Kategorie (nach mittlerem SHAP geordnet)", fontsize=7.5)
        else:
            ax.set_xlabel("Merkmalswert (Klassenmedian)", fontsize=7.5)
        stil(ax, i % 2 == 0)
    legende(fig, axe, [L_SHAP, L_GES, L_ART, L_AUF])
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=1.6)
    return fig


def speichern(fig, name):
    ZIEL.mkdir(parents=True, exist_ok=True)
    p = ZIEL / name
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("geschrieben:", p.relative_to(ROOT))


ZEILEN = [(m, None) if m.startswith("**") else (m, D.beschriftungen("xai24_zeilen").get(m, m))
          for m in ["**Sensoren**", SP["kanal_a"] + "_n_samples", SP["kanal_c"] + "_min", SP["kanal_b"] + "_std",
                    SP["kanal_f"] + "_mean", "**Stammdaten**", SP["wand_soll"], SP["laenge_erwartet"],
                    SP["granulatdichte_kathode"], "Glastyp"]]


def zahl(x):
    """Schwelle lesbar: Tausenderpunkt, Dezimalkomma, keine überflüssigen Nullen."""
    a = abs(x)
    nk = 0 if a >= 100 else (2 if a >= 1 else 3)
    s = f"{x:,.{nk}f}".replace(",", "§").replace(".", ",").replace("§", ".")
    return s.rstrip("0").rstrip(",") if "," in s else s


def schwellentabelle():
    T75 = importlib.import_module("19_tabellen_sensor_stammdatenwerte")

    def unterschied(s, k, kritisch_unten):
        """Kritischer Bereich minus Rest mit KI, fett wenn das KI null ausschließt."""
        w, lo, hi = s[f"d_{k}"], s[f"d_{k}_lo"], s[f"d_{k}_hi"]
        if kritisch_unten:
            w, lo, hi = -w, -hi, -lo

        def vz(x):
            return "0,0" if T75.de(abs(x), 1) == "0,0" else T75.vz(x, 1)

        txt = f"{vz(w)} [{vz(lo)}; {vz(hi)}]"
        return f"**{txt}**" if (lo > 0 or hi < 0) else txt

    S = pd.concat([pd.read_csv(TAB / "14_sensorwerte_schwellen_artikel.csv"),
                   pd.read_csv(TAB / "15_stammdatenwerte_schwellen_artikel.csv")],
                  ignore_index=True).set_index("Merkmal")
    with contextlib.redirect_stdout(io.StringIO()):
        import _common as C
        S15 = importlib.import_module("15_stammdatenwerte")
        ds = C.D.load_split(verbose=False)
    X = ds.X_raw.reset_index(drop=True)
    art = ds.df[SP["artikel"]].astype(str).reset_index(drop=True).to_numpy()

    kopf = ["Merkmal", "kritischer Bereich", "Billets (Artikel)", "Modell", "Ausschuss alle Billets",
            "Ausschuss im Artikel", "Ausschuss im Auftrag"]
    zeilen, csv = [], []
    for m, text in ZEILEN:
        if text is None:
            zeilen.append([m] + [""] * 6)
            continue
        s = S.loc[m]
        unten = s["richtung"] == "niedriger"
        kat = str(s.get("kategorial")) == "True"
        if kat:
            k = S15.kategorien(X[m])
            krit = np.isin(k, str(s["kategorien_oberhalb"]).split(", "))
            mit_wert = np.ones(len(k), dtype=bool)
            praefix = D.beschriftungen("xai24_texte").get("kategorie_praefix", "")
            bereich = str(s["kategorien_oberhalb"])
            if praefix:
                bereich = bereich.replace(", " + praefix, ", ")
        else:
            v = pd.to_numeric(X[m], errors="coerce").to_numpy(float)
            v[np.abs(v) > C.D.MAX_PLAUSIBLE_ABS] = np.nan
            mit_wert = ~np.isnan(v)
            krit = (v <= s["schwelle"]) if unten else (v > s["schwelle"])
            bereich = f"{'≤' if unten else '>'} {zahl(s['schwelle'])}"
        if kat and unten:
            krit = ~krit
        anteil = krit.sum() / mit_wert.sum()
        assert abs(anteil - T75.anteil(s)) < 0.005, (m, anteil, T75.anteil(s))
        artikelmerkmal = str(s.get("artikelmerkmal")) == "True"
        fest_im_auftrag = artikelmerkmal or np.isclose(s["d_y_auftrag_lo"], 0) and np.isclose(s["d_y_auftrag_hi"], 0)
        n_art = len(np.unique(art[krit]))
        zeilen.append([
            text, bereich,
            f"{anteil:.0%} ({n_art})".replace("%", " %"),
            T75.vz(abs(s["d_shap"]), 1),
            unterschied(s, "y_roh", unten),
            "–" if artikelmerkmal else unterschied(s, "y_artikel", unten),
            "–" if fest_im_auftrag else unterschied(s, "y_auftrag", unten),
        ])
        csv.append({"Merkmal": m, "Bezeichnung": text, "kritischer_bereich": bereich,
                    "anteil_billets": anteil, "billets_kritisch": int(krit.sum()),
                    "artikel_kritisch": n_art, **s.to_dict()})
    md = "\n".join(["| " + " | ".join(kopf) + " |", "|" + "---|" * len(kopf)]
                   + ["| " + " | ".join(z) + " |" for z in zeilen])
    (TAB / "tabelle_7_3_schwellen.md").write_text(md + "\n", encoding="utf-8")
    pd.DataFrame(csv).to_csv(TAB / "tabelle_7_3_schwellen.csv", index=False, encoding="utf-8",
                             float_format="%.6g")
    print(md)
    print("geschrieben:", (TAB / "tabelle_7_3_schwellen.md").relative_to(ROOT), "(+ .csv)")


def main():
    K14, K15, SW15 = laden()
    speichern(abb_sensoren(K14), "abb_7_3_sensoren.png")
    speichern(abb_stammdaten(K15, SW15), "abb_7_3_stammdaten.png")
    schwellentabelle()

    spalten = ["Merkmal", "klasse", "wert_von", "wert_bis", "wert_median", "n", "n_artikel", "shap_mittel",
               "y_mittel", "y_artikelbereinigt", "y_auftragsbereinigt"]
    K15 = K15.assign(y_gesamtbereinigt=K15.y_mittel - K15.Merkmal.map(SW15.y_gesamt))
    alle = pd.concat([K14[spalten].assign(art="Sensor"),
                      K15[spalten + ["y_gesamtbereinigt", "kategorien"]].assign(art="Stammdaten")],
                     ignore_index=True)
    p = TAB / "kapitel7_3_klassen.csv"
    alle.to_csv(p, index=False, encoding="utf-8", float_format="%.6g")
    print("geschrieben:", p.relative_to(ROOT))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(alle.drop(columns=["wert_von", "wert_bis", "kategorien"]).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
