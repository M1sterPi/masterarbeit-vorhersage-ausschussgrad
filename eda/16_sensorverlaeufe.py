"""16 — Aufzeichnung der Rohkanäle.

In der Arbeit: Abschnitt 4.2.2 und 5.2: Messpunkte je Kanal und Billet,
Brenndauer des Lichtbogens.

Abtastung, Fensterlänge und Lichtbogenanteil an einer Stichprobe aus KURVEN_DIR.
Ausgabe: reports/eda/16_sensorverlaeufe.pdf, reports/figures/sensorverlaeufe.png,
sensorabtastung.png.

    python eda/16_sensorverlaeufe.py
"""

from __future__ import annotations


import numpy as np
import pandas as pd

import _common as C
from data_prep import SP  # noqa: E402

BILLET_DIR = C.D.KURVEN_DIR

STICHPROBE = 150
DECKEL_H = 9.0

KANAELE = [(k, C.D.beschriftungen("eda16_kanaele").get(k, k), farbe) for k, farbe in [
    (SP["lichtbogen_leistung"], "#d62728"),
    (SP["kanal_j"], "#1f77b4"),
    (SP["kanal_g"], "#2ca02c"),
    (SP["phase"], "#7b3294"),
]]


def _lies(bid: str) -> pd.DataFrame | None:
    f = BILLET_DIR / f"{bid}.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f, sep=";", decimal=",", low_memory=False)
    d["_t"] = pd.to_datetime(d["_time"], format="ISO8601", utc=True, errors="coerce")
    return d.dropna(subset=["_t"])


def stichprobe(ids: list[str], n: int, seed: int = C.SEED) -> pd.DataFrame:
    """Je Billet: Fensterlänge, Bogendauer und Messpunkte je Kanal."""
    rng = np.random.default_rng(seed)
    wahl = rng.choice(ids, size=min(n, len(ids)), replace=False)
    zeilen, dichte = [], {}
    for bid in wahl:
        d = _lies(str(bid))
        if d is None or d.empty:
            continue
        t = d["_t"]
        fenster = (t.max() - t.min()).total_seconds() / 3600
        arc = pd.to_numeric(d.get(SP["lichtbogen_an"]), errors="coerce")
        bogen = np.nan
        if arc is not None and arc.notna().any():
            an = t[arc.fillna(0) > 0]
            bogen = (an.max() - an.min()).total_seconds() / 3600 if len(an) else 0.0
        zeilen.append({"batch_id": str(bid), "fenster_h": fenster, "bogen_h": bogen})
        for c in d.columns:
            if c in ("_time", "_t", SP["ofen_kennung"], SP["ofen_typ"]):
                continue
            dichte.setdefault(c, []).append(
                int(pd.to_numeric(d[c], errors="coerce").notna().sum())
            )
    S = pd.DataFrame(zeilen)
    S.attrs["dichte"] = pd.Series(
        {k: float(np.median(v)) for k, v in dichte.items()}
    ).sort_values(ascending=False)
    return S


def seite_verlaeufe(pdf, beispiele: list[tuple[str, float]]):
    fig, ax = C.seite(pdf, "Typische Sensorverläufe — drei Billets, vier Kanäle", 4, 3)
    punkte = {}
    for spalte, (bid, yv) in enumerate(beispiele):
        d = _lies(bid)
        if d is None:
            continue
        t0 = d["_t"].min()
        std = (d["_t"] - t0).dt.total_seconds() / 3600
        fenster = float(std.max())
        for zeile, (kanal, label, farbe) in enumerate(KANAELE):
            a = ax[zeile * 3 + spalte]
            if kanal not in d.columns:
                C.leer(a)
                continue
            v = pd.to_numeric(d[kanal], errors="coerce")
            m = v.notna()
            a.plot(std[m], v[m], lw=0.6, color=farbe, drawstyle="steps-post")
            punkte[(bid, kanal)] = int(m.sum())
            a.set_xlim(0, fenster)
            a.set_ylabel(label, fontsize=7)
            a.tick_params(labelsize=7)
            if zeile == 0:
                a.set_title(f"{bid}  ·  y = {yv:.1f} %", fontsize=9)
            if zeile == len(KANAELE) - 1:
                a.set_xlabel("Stunden seit Fensterbeginn", fontsize=8)
    C.schliesse(
        pdf,
        fig,
        "Rohaufzeichnung aus BILLET_DIR, nicht die Merkmalstabelle | "
        "je Feld die Zahl der tatsächlich geschriebenen Messpunkte | "
        "gleiche Zeitachse je Spalte, eigene Werteachse je Kanal",
    )
    fig.punkte = punkte
    return fig


def seite_kennzahlen(pdf, S: pd.DataFrame):
    fig_a, ax_a = C.seite(pdf, "Abtastung je Kanal", 1, 1)
    dichte = S.attrs["dichte"]
    top = dichte[dichte > 0][::-1]
    ax_a[0].barh(range(len(top)), top.values, color="#4292c6")
    ax_a[0].set_yticks(range(len(top)))
    ax_a[0].set_yticklabels(top.index, fontsize=8)
    ax_a[0].spines["left"].set_visible(False)
    ax_a[0].tick_params(axis="y", length=0)
    ax_a[0].set_xscale("log")
    ax_a[0].set_xlabel("Messpunkte je Billet (Median)", fontsize=9)
    ax_a[0].set_title("A  Messpunkte je Kanal", fontsize=10)
    C.schliesse(pdf, fig_a)

    fig, ax = C.seite(
        pdf, "Fensterlänge und Bogenanteil", 1, 3, groesse=(11.7, 5.2)
    )
    ax = [None, *ax]

    ax[1].hist(S.fenster_h.dropna(), bins=40, color="#4292c6")
    ax[1].axvline(DECKEL_H, color="crimson", ls="--", lw=1.2, label=f"{DECKEL_H:g}-h-Deckel")
    ax[1].legend(fontsize=7, loc="upper left")
    anteil = float((S.fenster_h >= DECKEL_H - 0.1).mean())
    ax[1].set_xlabel("Fensterlänge [h]", fontsize=8)
    ax[1].set_ylabel("Anzahl Billets", fontsize=8)
    ax[1].set_title("B  Fensterlänge", fontsize=10)

    q = (S.bogen_h / S.fenster_h).dropna()
    ax[2].hist(q, bins=30, color="#fd8d3c")
    ax[2].axvline(q.median(), color="black", ls="--", lw=1.2, label="Median")
    ax[2].legend(fontsize=7, loc="upper right")
    ax[2].set_xlabel("Anteil des Fensters mit Lichtbogen", fontsize=8)
    ax[2].set_ylabel("Anzahl Billets", fontsize=8)
    ax[2].set_title("C  Anteil des Fensters mit Lichtbogen", fontsize=10)

    ax[3].scatter(S.fenster_h, S.bogen_h, s=12, alpha=0.6, color="#756bb1")
    ax[3].set_xlabel("Fensterlänge [h]", fontsize=8)
    ax[3].set_ylabel("Bogendauer [h]", fontsize=8)
    rho = S[["fenster_h", "bogen_h"]].corr(method="spearman").iloc[0, 1]
    ax[3].set_title("D  Bogendauer gegen Fensterlänge", fontsize=10)
    C.schliesse(
        pdf,
        fig,
        f"Stichprobe von {len(S)} Billets aus der Rohaufzeichnung | "
        f"Deckel {DECKEL_H:g} h aus der Featurisierung",
    )
    return fig, anteil, q.median(), rho


def main():
    ds = C.lade(alle_arten=True)
    X, y, tr = C.daten_teil(ds)
    y = pd.to_numeric(y, errors="coerce").reset_index(drop=True)
    ids = tr["batch_id"].astype(str).tolist()

    S = stichprobe(ids, STICHPROBE)

    beispiele = []
    for qq in (0.1, 0.5, 0.9):
        ziel = y.quantile(qq)
        kand = (y - ziel).abs().sort_values().index
        for i in kand[:50]:
            bid = str(tr["batch_id"].iloc[i])
            if (BILLET_DIR / f"{bid}.csv").exists():
                beispiele.append((bid, float(y.iloc[i])))
                break

    pdf = C.pdf_oeffnen("16_sensorverlaeufe.pdf")
    fig1 = seite_verlaeufe(pdf, beispiele)
    C.aufraeumen(fig1)
    fig1.savefig(
        C.ROOT / "reports" / "figures" / "sensorverlaeufe.png",
        dpi=200,
        bbox_inches="tight",
    )
    fig2, anteil, bogen_q, rho = seite_kennzahlen(pdf, S)
    C.aufraeumen(fig2)
    fig2.savefig(
        C.ROOT / "reports" / "figures" / "sensorabtastung.png",
        dpi=200,
        bbox_inches="tight",
    )

    d = S.attrs["dichte"]
    C.textseite(
        pdf,
        "Befunde und Vorbehalte",
        [
            f"Die Kanäle sind nicht gemeinsam abgetastet. Über {len(S)} Billets "
            f"liegt der dichteste Kanal ({d.index[0]}) bei {d.iloc[0]:.0f} "
            f"Messpunkten je Billet, {SP['phase']} bei "
            f"{d.get(SP['phase'], float('nan')):.0f} und {SP['kanal_e']} bei "
            f"{d.get(SP['kanal_e'], float('nan')):.0f}. Die dünnen Kanäle werden "
            "ereignisbasiert geschrieben, also nur bei Zustandswechsel, die "
            "dichten zyklisch. Ein Mittelwert 'über das Fenster' bedeutet "
            "deshalb je Kanal etwas anderes, und ein gleichgewichteter "
            "Mittelwert über die Messpunkte gewichtet implizit nach "
            "Schreibhäufigkeit. Messpunkte je Billet (Median der Stichprobe): "
            + ", ".join(f"{k} {v:.0f}" for k, v in d[d > 0].items())
            + ".",
            "Beispielbillets auf Seite 1 (10-, 50- und 90-%-Quantil von y), "
            "geschriebene Messpunkte je Kanal: "
            + "; ".join(
                f"{bid} (y = {yv:.1f} %): "
                + ", ".join(
                    f"{lab} {fig1.punkte.get((bid, kan), 0)}"
                    for kan, lab, _ in KANAELE
                )
                for bid, yv in beispiele
            )
            + ". Die Kurven sind als Treppe gezeichnet, weil jeder Kanal seinen "
            "Wert bis zum nächsten Schreibvorgang hält.",
            f"Das Fenster ist gedeckelt. {100 * anteil:.0f} Prozent der "
            f"Stichprobe liegen am {DECKEL_H:g}-Stunden-Deckel; dort ist die "
            "wahre Fensterlänge unbekannt und batch_duration_s rechtszensiert. "
            "Die Ursache ist die Fensterdefinition: von einem Wechsel der Billetnummer "
            "bis zum nächsten, was bei Reinigungs- und Stillstandszeiten "
            "beliebig lang wird.",
            f"Der Lichtbogen füllt das Fenster nicht. Median der Bogendauer "
            f"{S.bogen_h.median():.1f} Stunden gegenüber "
            f"{S.fenster_h.median():.1f} Stunden Fenster, also ein Anteil von "
            f"{bogen_q:.2f}. Bogendauer heisst hier: Zeit vom ersten bis zum letzten "
            f"{SP['lichtbogen_an']}-Eintrag > 0 in der Rohaufzeichnung. Fensterlänge und Bogendauer "
            f"hängen dabei nur "
            f"schwach zusammen (Spearman {rho:+.2f}). Jeder Fenstermittelwert mittelt "
            "also überwiegend Stillstand mit, in einem Verhältnis, das von "
            "Billet zu Billet schwankt — und nicht von y abhängen muss.",
            "Vorbehalt zur Stichprobe. Die Kennzahlen beruhen auf "
            f"{len(S)} zufällig gezogenen Billets, nicht auf allen "
            f"{len(y)}. Für Verteilungsaussagen dieser Grobheit reicht das; einzelne "
            "seltene Kanäle können in der Stichprobe fehlen. Der Anteil am "
            "Deckel ist über den vollständigen Datensatz in "
            "07_datenqualitaet.pdf nachgerechnet.",
            "Vorbehalt zur Auswahl der drei Beispielbillets. Sie sind nach dem "
            "10-, 50- und 90-Prozent-Quantil von y gewählt und illustrieren die "
            "Aufzeichnungsform, nicht einen Zusammenhang mit dem Ausschussgrad. "
            "Aus drei Verläufen lässt sich kein Unterschied zwischen guten und "
            "schlechten Schmelzen ablesen; dafür stehen die Skripte 03 und 06.",
        ],
    )
    pdf.close()
    print(f"geschrieben: {C.OUT_DIR / '16_sensorverlaeufe.pdf'}")
    print(f"  Stichprobe {len(S)} Billets, {100 * anteil:.1f} % am Deckel")
    print(
        f"  Fenster Median {S.fenster_h.median():.2f} h | "
        f"Bogen Median {S.bogen_h.median():.2f} h | Anteil {bogen_q:.2f}"
    )
    print(f"  Messpunkte je Billet: {d.iloc[0]:.0f} bis {d[d > 0].iloc[-1]:.0f}")


if __name__ == "__main__":
    main()
