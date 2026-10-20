"""21 — Extreme Billets.

In der Arbeit: Abschnitt 7.4: Herkunft des Gewinns gegenüber der Median-
Vorhersage nach Ausschussdezilen.

Modellgüte bei sehr hohem und sehr niedrigem Ausschuss (Läufe wie eda/10).
Ausgabe: reports/eda/21_extreme_billets.pdf.

    python eda/21_extreme_billets.py
"""

from __future__ import annotations

import importlib
import json

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score

import _common as C

MV = importlib.import_module("10_modellvergleich")

DEZILE = np.arange(1, 11)
QUANTIL = 0.10


def _oof(run_id) -> tuple[np.ndarray, np.ndarray]:
    f = C.ROOT / "runs" / "single" / f"run_{run_id}.json"
    p = json.loads(f.read_text(encoding="utf-8"))["predictions"]
    y = np.asarray(p["y_true"], dtype=float)
    pred = np.asarray(p["oof_pred"], dtype=float)
    ok = ~np.isnan(pred)
    return y[ok], pred[ok]


def _dezil(werte) -> np.ndarray:
    return pd.qcut(pd.Series(werte).rank(method="first"), 10, labels=DEZILE).astype(int).to_numpy()


def kennzahlen(y, pred) -> dict:
    """Alle Zahlen eines Laufs."""
    med = np.median(y)
    oben, unten = np.quantile(y, 1 - QUANTIL), np.quantile(y, QUANTIL)
    dy, dp = _dezil(y), _dezil(pred)
    f_mod, f_med = np.abs(y - pred), np.abs(y - med)
    q_gewinn = (y - med) ** 2 - (y - pred) ** 2
    ist_oben, ist_unten = y > oben, y < unten
    return {
        "gewinn_je_ydezil": np.array([f_med[dy == k].mean() - f_mod[dy == k].mean() for k in DEZILE]),
        "top_je_pdezil": np.array([ist_oben[dp == k].mean() * 100 for k in DEZILE]),
        "bottom_je_pdezil": np.array([ist_unten[dp == k].mean() * 100 for k in DEZILE]),
        "auc_oben": roc_auc_score(ist_oben, pred),
        "auc_unten": roc_auc_score(ist_unten, -pred),
        "r2_top_dezil": q_gewinn[dy == 10].sum() / ((y - y.mean()) ** 2).sum(),
        "r2_rest": q_gewinn[dy != 10].sum() / ((y - y.mean()) ** 2).sum(),
        "r2": 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum(),
        "mae": f_mod.mean(),
        "mae_median": f_med.mean(),
    }


def lade() -> pd.DataFrame:
    e = MV.lade()
    zeilen = []
    for _, r in e.iterrows():
        y, pred = _oof(r.run_id)
        zeilen.append({"modell": r.modell, "ebene": r.ebene, "run_id": r.run_id, "n": len(y), **kennzahlen(y, pred)})
    return pd.DataFrame(zeilen)


def _linien(ax, t, spalte, modelle):
    for m in modelle:
        s = t[t.modell == m]
        if s.empty:
            continue
        ax.plot(
            DEZILE,
            s[spalte].iloc[0],
            marker="o",
            ms=3.5,
            lw=1.4,
            color=MV.FARBEN.get(m, "#888888"),
            label=m,
        )
    ax.set_xticks(DEZILE)
    ax.grid(alpha=0.3)


def seite_ebene(pdf, k, ebene, modelle):
    t = k[k.ebene == ebene]
    fig, ax = C.seite(pdf, f"Extreme Billets — Ebene {ebene}", 2, 2)

    _linien(ax[0], t, "gewinn_je_ydezil", modelle)
    ax[0].axhline(0, color="black", lw=1)
    ax[0].set_xlabel("Dezil des tatsächlichen Ausschusses (1 = wenig, 10 = viel)")
    ax[0].set_ylabel("MAE Median − MAE Modell [Pp]\n> 0: Modell besser")
    ax[0].set_title("A  Wo das Modell gegenüber dem Median gewinnt", fontsize=10)
    ax[0].legend(fontsize=7, loc="upper left")

    _linien(ax[1], t, "top_je_pdezil", modelle)
    ax[1].axhline(QUANTIL * 100, color="#555555", ls="--", lw=1)
    ax[1].set_xlabel("Dezil der Vorhersage (1 = Modell sagt wenig, 10 = viel)")
    ax[1].set_ylabel("Anteil echter Top-10-%-Billets [%]")
    ax[1].set_title("B  Findet das Modell die Billets mit viel Ausschuss?", fontsize=10)
    ax[1].legend(
        handles=[Line2D([], [], color="#555555", ls="--", lw=1, label="Zufall (10 %)")],
        fontsize=7,
        loc="upper left",
    )

    _linien(ax[2], t, "bottom_je_pdezil", modelle)
    ax[2].axhline(QUANTIL * 100, color="#555555", ls="--", lw=1)
    ax[2].set_xlabel("Dezil der Vorhersage (1 = Modell sagt wenig, 10 = viel)")
    ax[2].set_ylabel("Anteil echter Bottom-10-%-Billets [%]")
    ax[2].set_title("C  Findet das Modell die Billets mit wenig Ausschuss?", fontsize=10)

    _auc_balken(ax[3], t, modelle)
    ax[3].set_title("D  Trennschärfe für die oberen und unteren 10 %", fontsize=10)

    hi = max(ax[1].get_ylim()[1], ax[2].get_ylim()[1])
    ax[1].set_ylim(0, hi)
    ax[2].set_ylim(0, hi)
    C.schliesse(pdf, fig)


def _auc_balken(ax, t, modelle):
    x = np.arange(len(modelle))
    breite = 0.38
    for i, (spalte, name, farbe) in enumerate(
        [("auc_oben", "obere 10 %", "#d62728"), ("auc_unten", "untere 10 %", "#3182bd")]
    ):
        werte = [t.set_index("modell")[spalte].get(m, np.nan) for m in modelle]
        balken = ax.bar(x + (i - 0.5) * breite, werte, breite, color=farbe, label=name, alpha=0.85)
        MV._beschrifte(ax, balken, werte, "{:.3f}")
    ax.axhline(0.5, color="#555555", ls="--", lw=1, label="Zufall (0,5)")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" (", "\n(") for m in modelle], fontsize=7.5)
    ax.set_ylabel("AUC")
    ax.set_ylim(0.4, 1.0)
    ax.legend(fontsize=7, loc="upper right", ncol=3)
    ax.grid(alpha=0.3, axis="y")


def seite_uebersicht(pdf, k, modelle):
    fig, ax = C.seite(pdf, "Extreme Billets — Übersicht über die Ebenen", 1, 2, groesse=(11.7, 5.2))
    ebenen = [n for n, _, _ in MV.EBENEN]
    x = np.arange(len(modelle))
    breite = 0.26
    for a, spalte, titel in [
        (ax[0], "auc_oben", "AUC: Billet gehört zu den oberen 10 %"),
        (ax[1], "auc_unten", "AUC: Billet gehört zu den unteren 10 %"),
    ]:
        for i, eb in enumerate(ebenen):
            s = k[k.ebene == eb].set_index("modell")[spalte]
            werte = [s.get(m, np.nan) for m in modelle]
            balken = a.bar(x + (i - 1) * breite, werte, breite, color=MV.plt_farbe(i), label=eb)
            MV._beschrifte(a, balken, werte, "{:.2f}")
        a.axhline(0.5, color="#555555", ls="--", lw=1)
        a.set_xticks(x)
        a.set_xticklabels([m.replace(" (", "\n(") for m in modelle], fontsize=7.5)
        a.set_ylim(0.4, 0.95)
        a.set_ylabel("AUC (0,5 = Zufall)")
        a.set_title(titel, fontsize=10)
        a.legend(fontsize=7.5, loc="upper right", ncol=3)
        a.grid(alpha=0.3, axis="y")
    C.schliesse(pdf, fig)


def main():
    k = lade()
    if k.empty:
        raise SystemExit("Keine passenden Läufe gefunden.")
    e = MV.lade()
    modelle = MV._reihenfolge(e)

    pdf = C.pdf_oeffnen("21_extreme_billets.pdf")
    for eb, _, _ in MV.EBENEN:
        seite_ebene(pdf, k, eb, modelle)
    seite_uebersicht(pdf, k, modelle)

    zeilen = []
    for m in modelle:
        for eb, _, _ in MV.EBENEN:
            t = k[(k.modell == m) & (k.ebene == eb)]
            if t.empty:
                continue
            r = t.iloc[0]
            zeilen.append(
                f"  {m}, {eb}: R² {r.r2:+.3f}, MAE {r.mae:.2f} (Median {r.mae_median:.2f}), "
                f"Gewinn ggü. Median in R²-Punkten: oberstes y-Dezil {r.r2_top_dezil:+.3f}, "
                f"übrige Dezile {r.r2_rest:+.3f}; "
                f"im obersten Vorhersage-Dezil {r.top_je_pdezil[-1]:.0f} % Top-10-%-Billets, "
                f"AUC oben {r.auc_oben:.3f} / unten {r.auc_unten:.3f}"
            )
    C.textseite(
        pdf,
        "Zahlen und Vorbehalte",
        [
            "Grundlage: Out-of-fold-Vorhersagen derselben Läufe wie in 10_modellvergleich.py, "
            f"je {int(k.n.iloc[0])} Billets. Extrem heisst: oberhalb des 90-%- bzw. unterhalb "
            "des 10-%-Quantils von y.",
            "Gewinn in R²-Punkten: Summe von (y − Median)² − (y − Vorhersage)² über die "
            "Billets des obersten y-Dezils bzw. der übrigen neun Dezile, geteilt durch die "
            "Gesamtstreuung Σ(y − ȳ)². Ein negativer Wert bei den übrigen Dezilen heisst, dass "
            "das Modell dort quadriert schlechter ist als der Median.",
            *zeilen,
            "Vorbehalt zu Feld A. Wer nach dem tatsächlichen y einteilt, sieht bei jedem Modell, "
            "dessen Vorhersagen überhaupt streuen, Verluste in der Mitte und Gewinne an den "
            "Rändern — der Median liegt in der Mitte definitionsgemäss fast richtig. Feld A "
            "zeigt deshalb, WO der Gewinn entsteht, belegt aber keine Trennfähigkeit. Diese "
            "zeigen die Felder B bis D, die nach der Vorhersage einteilen.",
            "Vorbehalt zur Basis. Die Median-Vorhersage ist hier der Median über alle Billets, "
            "nicht der Median der jeweiligen Trainingsfalte wie in 10_modellvergleich.py. Die "
            "MAE der Basis weicht dadurch in der dritten Nachkommastelle ab.",
            "Vorbehalt zur Unabhängigkeit. Billets desselben Auftrags sind nicht unabhängig; "
            "auf Billetebene liegen Geschwister in Training und Bewertung. Die AUC auf "
            "Billetebene ist deshalb die optimistischste der drei.",
        ],
    )
    pdf.close()
    print(f"geschrieben: {C.OUT_DIR / '21_extreme_billets.pdf'}")
    print(k[["modell", "ebene", "auc_oben", "auc_unten", "r2_top_dezil", "r2_rest"]].round(3).to_string())


if __name__ == "__main__":
    main()
