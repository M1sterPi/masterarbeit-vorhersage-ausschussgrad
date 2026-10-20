"""32 — Beispiel eines Schmelzverlaufs.

In der Arbeit: Abbildung 6 (Beispiel eines Schmelzverlaufs, Abschnitt 4.2.2).

Ausgewählte Kanäle eines Billets, je Kurve normiert. Billetnummer über
BEISPIEL_BILLET (.env), Rohkurve aus KURVEN_DIR.
Ausgabe: reports/figures/kapitel4/sensorverlauf_beispiel.png.

    python eda/32_schmelzverlauf.py
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _common as C
from data_prep import SP  # noqa: E402

BILLET = os.environ.get("BEISPIEL_BILLET", "")
KANAELE = [SP["lichtbogen_spannung"], SP["kanal_d"], SP["kanal_b"], SP["kanal_c"], SP["kanal_j"]]
FARBEN = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2"]
ZIEL = C.ROOT / "reports" / "figures" / "kapitel4" / "sensorverlauf_beispiel.png"


def lade(pfad):
    kopf = pd.read_csv(pfad, sep=";", nrows=0, encoding="utf-8").columns
    spalten = ["_time"] + [k for k in KANAELE if k in kopf]
    df = pd.read_csv(pfad, sep=";", usecols=spalten, encoding="utf-8", low_memory=False)
    t = pd.to_datetime(df["_time"], errors="coerce", utc=True, format="ISO8601")
    df, t = df[t.notna()], t[t.notna()]
    stunden = ((t - t.min()).dt.total_seconds() / 3600).to_numpy()
    reihen = {}
    for k in spalten[1:]:
        v = C.D._to_num_de(df[k])
        m = v.notna().to_numpy()
        if m.any():
            reihen[k] = (stunden[m], v.to_numpy()[m])
    return reihen


def main():
    if not BILLET:
        raise SystemExit("BEISPIEL_BILLET fehlt: in .env eintragen (Vorlage .env.example).")
    pfad = C.D.KURVEN_DIR / f"billet_{BILLET}.csv"
    if not pfad.exists():
        raise SystemExit(f"Rohkurve fehlt: {pfad} (KURVEN_DIR in .env setzen)")
    reihen = lade(pfad)

    fig, ax = plt.subplots(figsize=(16 / 2.54, 5.2 / 2.54 * 1.6))
    for (k, (t, v)), farbe in zip(reihen.items(), FARBEN):
        lo, hi = np.nanmin(v), np.nanmax(v)
        vn = np.zeros_like(v) if hi == lo else (v - lo) / (hi - lo)
        ordnung = np.argsort(t, kind="stable")
        ax.plot(t[ordnung], vn[ordnung], lw=0.9, color=farbe, label=k)
    ax.set_title(f"Billet {BILLET}", fontsize=10)
    ax.set_xlabel("Stunden ab Fensterstart")
    ax.set_ylabel("normiert (Min-Max je Kurve)")
    ax.set_xlim(0, max(t.max() for t, _ in reihen.values()))
    ax.set_ylim(-0.03, 1.06)
    ax.grid(axis="y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=len(reihen), fontsize=7,
              frameon=False)
    C.aufraeumen(fig)
    ZIEL.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ZIEL, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"geschrieben: {ZIEL.relative_to(C.ROOT)}")
    for k, (t, v) in reihen.items():
        print(f"  {k:28s} {len(t):6d} Messpunkte, {t.min():.2f}–{t.max():.2f} h")


if __name__ == "__main__":
    main()
