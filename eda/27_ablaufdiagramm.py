"""27 — Schema der Datenvorverarbeitung.

In der Arbeit: Abbildung 10 (Ablauf der Datenvorverarbeitung, Kapitel 5).

Die Zahlen im Bild werden aus den Daten bestimmt.
Ausgabe: reports/figures/kapitel5/ablauf_vorverarbeitung.png.

    python eda/27_ablaufdiagramm.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

import _common as C  # noqa: E402
import rocket_regression as R  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ZIEL = ROOT / "reports" / "figures" / "kapitel5" / "ablauf_vorverarbeitung.png"

BLAU = "#1f77b4"
HELLBLAU = "#a6cee3"
ORANGE = "#d95f02"
HELLORANGE = "#fdd0a2"
GRAU = "#666666"
HELLGRAU = "#e6e6e6"


def kasten(ax, x0, x1, y, h, titel, text, fill, rand, titelfarbe="black", textfarbe="black"):
    """Kasten mit fettem Titel (Abschnittsnummer) und normalem Text darunter."""
    ax.add_patch(FancyBboxPatch((x0, y - h / 2), x1 - x0, h, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=fill, ec=rand, lw=1.0))
    xm = (x0 + x1) / 2
    if text:
        zweizeilig = "\n" in text
        ax.text(xm, y + h / 2 - 1.9, titel, ha="center", va="center", fontsize=7.6, fontweight="bold",
                color=titelfarbe)
        ax.text(xm, y - (1.3 if zweizeilig else 0.9), text, ha="center", va="center", fontsize=6.8,
                color=textfarbe, linespacing=1.2)
    else:
        ax.text(xm, y, titel, ha="center", va="center", fontsize=7.6, fontweight="bold", color=titelfarbe,
                linespacing=1.2)


def pfeil(ax, x0, y0, x1, y1, gestrichelt=False):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=1.0,
                                 color=GRAU, linestyle=(0, (3, 2)) if gestrichelt else "-",
                                 shrinkA=0, shrinkB=0))


def kennzahlen() -> dict:
    """Spalten- und Merkmalszahlen der Vorverarbeitung."""
    D = C.D
    ds = D.load_split(verbose=False)
    influx = D.influx_spalten(ds.X_raw.columns)
    pfad = D.ROOT / "data" / "cache" / (D.CSV_PATH.stem + ".process_cols.json")
    kanaele = json.loads(pfad.read_text(encoding="utf-8"))["process_cols"]
    agg = [c for c in influx if c in D._EXTRA_INFLUX
           or any(c.startswith(k) and c[len(k):] in D._AGG_SUFFIXES for k in kanaele)]
    X_tr, _, _ = D.process_features(ds.X_train_raw, ds.X_test_raw)
    return {
        "kanaele": len(R.CHANNELS),
        "raster": R.SERIES_LEN,
        "sql": pd.read_csv(D.MASTER_CSV, nrows=0).shape[1],
        "agg": len(agg),
        "kopf": len(influx) - len(agg),
        "merkmale": ds.X_raw.shape[1],
        "kodiert": X_tr.shape[1],
        "kard": D.MAX_CARDINALITY,
        "fenster": D.MAX_BILLET_DURATION_H,
    }


def main():
    k = kennzahlen()
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(6.3, 5.6), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(-1, 79)
    ax.axis("off")

    kasten(ax, 2, 64, 74, 7, "Prozessdaten (InfluxDB)",
           f"{k['kanaele']} Kanäle, Wert nur bei Änderung oder Takt",
           BLAU, BLAU, "white", textfarbe="white")
    kasten(ax, 68, 98, 74, 7, "Stammdaten (SQL)", f"{k['sql']} Spalten je Billet", BLAU, BLAU, "white",
           textfarbe="white")

    kasten(ax, 2, 64, 63.5, 7, "5.1  Zeitfenster je Billet",
           f"von Wechsel zu Wechsel der Billetnummer, höchstens {k['fenster']:g} h", HELLGRAU, GRAU)
    pfeil(ax, 33, 70.5, 33, 67)

    ax.text(19, 55.5, "Kurvenzweig", ha="left", va="center", fontsize=8, fontweight="bold", color=ORANGE)
    ax.text(52, 55.5, "Tabellarischer Zweig", ha="left", va="center", fontsize=8, fontweight="bold",
            color=BLAU)

    kasten(ax, 2, 31, 45.5, 9, "5.2  Raster", f"Zero-Order-Hold,\n{k['raster']} Punkte je Kanal",
           HELLORANGE, ORANGE)
    pfeil(ax, 17, 60, 17, 50)
    kasten(ax, 2, 31, 16.5, 9, "5.6 / 5.7", "fehlende Kanäle = 0,\nz-Normierung je Kurve", HELLORANGE, ORANGE)
    pfeil(ax, 17, 41, 17, 21)
    kasten(ax, 2, 31, 3.5, 8, "MiniRocket, HYDRA,\nFreshPRINCE", None, ORANGE, ORANGE, "white")
    pfeil(ax, 17, 12, 17, 7.5)

    kasten(ax, 35, 64, 45.5, 9, "5.3  Kennzahlen",
           f"{k['agg']} aus den Kanälen,\ndazu {k['kopf']} Kopfangaben", HELLBLAU, BLAU)
    pfeil(ax, 50, 60, 50, 50)
    kasten(ax, 35, 98, 34.5, 7, "5.4  Abgrenzung des Vorhersagezeitpunkts",
           f"Prüfergebnisse und Zeitstempel gesperrt  →  {k['merkmale']} Merkmale", HELLBLAU, BLAU)
    pfeil(ax, 50, 41, 50, 38)
    pfeil(ax, 83, 70.5, 83, 38)
    kasten(ax, 35, 98, 25.5, 7, "5.5  Kodierung",
           f"ordinal, höchstens {k['kard']} Ausprägungen  →  {k['kodiert']} Merkmale",
           HELLBLAU, BLAU)
    pfeil(ax, 66.5, 31, 66.5, 29)
    kasten(ax, 35, 98, 16.5, 7, "5.6 / 5.7", "|x| > 10¹² → fehlend, keine Imputation, keine Skalierung",
           HELLBLAU, BLAU)
    pfeil(ax, 66.5, 22, 66.5, 20)
    kasten(ax, 35, 98, 3.5, 8, "XGBoost, CatBoost, TabPFN", "eine Zeile je Billet", BLAU, BLAU, "white",
           textfarbe="white")
    pfeil(ax, 66.5, 13, 66.5, 7.5)

    pfeil(ax, 38, 13, 31.5, 7.5, gestrichelt=True)
    ax.text(40, 10.2, "hybrid: fehlend → 0,\nstandardisiert", fontsize=6.6, color=GRAU, ha="left",
            va="center", style="italic", linespacing=1.15)

    ZIEL.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ZIEL, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    print("geschrieben:", ZIEL)


if __name__ == "__main__":
    main()
