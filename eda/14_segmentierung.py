"""14 — Zeitfenster je Billet.

In der Arbeit: Abschnitt 4.2.2 und 8.2: Anteil der Billetfenster an der Neun-
Stunden-Grenze.

Fensterlänge, Lichtbogenanteil und Ausschnittsstrategien (MiniRocket).
Braucht den Kurventensor. Ausgabe: reports/eda/14_segmentierung.pdf.

    python eda/14_segmentierung.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import _common as C

sys.path.insert(0, str(C.ROOT / "data_pipeline"))
import rocket_regression as R  # noqa: E402
from data_prep import SP  # noqa: E402

ARC_SCHWELLE = 1.0
ARC_KANAL = SP["lichtbogen_leistung"]

PREFIX_H = 2.0

SEED = 42


def _strecken(zeile: np.ndarray, von: int, bis: int, ziel: int) -> np.ndarray:
    """Zellen [von, bis) einer Kurve auf ziel Punkte umtasten (linear)."""
    if bis - von < 2:
        return np.zeros(ziel, dtype=np.float32)
    q = np.linspace(0, bis - von - 1, ziel)
    return np.interp(q, np.arange(bis - von), zeile[von:bis]).astype(np.float32)


def arc_maske(T: np.ndarray) -> np.ndarray:
    """(n × SERIES_LEN) Boolean: brennt in dieser Rasterzelle der Lichtbogen?"""
    k = R.CHANNELS.index(ARC_KANAL)
    return T[:, k, :] > ARC_SCHWELLE


def auf_arc_ausrichten(T: np.ndarray) -> np.ndarray:
    """Jede Kurve auf ihr Lichtbogenfenster beschneiden und wieder auf 512 dehnen."""
    A = arc_maske(T)
    out = np.zeros_like(T)
    for i in range(len(T)):
        idx = np.flatnonzero(A[i])
        if idx.size < 2:
            out[i] = T[i]
            continue
        von, bis = int(idx[0]), int(idx[-1]) + 1
        for c in range(T.shape[1]):
            out[i, c] = _strecken(T[i, c], von, bis, T.shape[2])
    return out


def fester_vorlauf(T: np.ndarray, dauer_h: np.ndarray, stunden: float) -> np.ndarray:
    """Die ersten `stunden` jedes Fensters, auf 512 Punkte gedehnt."""
    out = np.zeros_like(T)
    L = T.shape[2]
    for i in range(len(T)):
        if not np.isfinite(dauer_h[i]) or dauer_h[i] <= 0:
            out[i] = T[i]
            continue
        bis = int(np.clip(round(L * stunden / dauer_h[i]), 2, L))
        for c in range(T.shape[1]):
            out[i, c] = _strecken(T[i, c], 0, bis, L)
    return out


def oof_rocket(X3d, y, folds):
    """Identisch zu 13_glaettung.oof_rocket — MiniRocket + Scaler + RidgeCV."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from aeon.transformations.collection.convolution_based import MiniRocket

    pred = np.full(len(y), np.nan)
    for tr, va in folds:
        tf = MiniRocket(n_kernels=1000, n_jobs=-1, random_state=SEED).fit(X3d[tr])
        Ztr = np.nan_to_num(np.asarray(tf.transform(X3d[tr]), dtype=np.float64))
        Zva = np.nan_to_num(np.asarray(tf.transform(X3d[va]), dtype=np.float64))
        sc = StandardScaler().fit(Ztr)
        r = RidgeCV(alphas=np.logspace(-1, 6, 15)).fit(sc.transform(Ztr), y.iloc[tr])
        p = r.predict(sc.transform(Zva))
        pred[va] = np.clip(p, y.iloc[tr].min(), y.iloc[tr].max())
    return pred


def main():
    ds = C.lade(alle_arten=True)
    X, y, df = C.daten_teil(ds)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    df = df.reset_index(drop=True)
    fuss = C.kopfzeile(ds, alle_arten=True)
    pdf = C.pdf_oeffnen("14_segmentierung.pdf")

    print("  lade Kurventensor ...")
    T = np.ascontiguousarray(R.build_or_load_tensor(ds.batch_ids)[C.eda_maske(ds)])
    dauer_h = (C.als_zahl(X["batch_duration_s"]) / 3600).to_numpy()
    cap_h = float(C.D.MAX_BILLET_DURATION_H)
    am_cap = dauer_h >= cap_h - 0.01
    A = arc_maske(T)
    arc_anteil = A.mean(axis=1)
    arc_h = arc_anteil * dauer_h
    arc_start = np.array(
        [np.flatnonzero(a)[0] / len(a) if a.any() else np.nan for a in A]
    )
    zelle0 = A[:, 0]
    arcon_aus_in_zelle0 = np.nan
    if SP["lichtbogen_an"] in R.CHANNELS and zelle0.any():
        k_on = R.CHANNELS.index(SP["lichtbogen_an"])
        arcon_aus_in_zelle0 = float((T[zelle0, k_on, 0] < 0.5).mean())
    maske_verzerrt = zelle0.mean() > 0.05
    arc_normiert = float(np.nanmin(T[:, R.CHANNELS.index(ARC_KANAL), :])) < 0

    fig, ax = C.seite(
        pdf, "A/B — Fensterlänge und Lichtbogen", 2, 2
    )

    ax[0].hist(dauer_h, bins=90, color="steelblue")
    ax[0].axvline(cap_h, color="crimson", ls="--", lw=1.5, label=f"Deckel {cap_h:g} h")
    ax[0].set_xlabel("Fensterdauer [h]")
    ax[0].set_ylabel("Anzahl Billets")
    ax[0].set_title("A1  Fensterdauer je Billet", fontsize=10)
    ax[0].legend(fontsize=7, loc="upper left")

    ax[1].hist(arc_h[np.isfinite(arc_h)], bins=80, color="seagreen")
    ax[1].set_xlabel("Lichtbogendauer [h]")
    ax[1].set_ylabel("Anzahl Billets")
    ax[1].set_title("A2  Lichtbogendauer je Billet (Maske im Raster)", fontsize=10)
    ax[1].axvline(
        PREFIX_H,
        color="crimson",
        ls="--",
        lw=1.2,
        label=f"{PREFIX_H:g}-h-Vorlauf",
    )
    ax[1].legend(fontsize=7, loc="upper right")

    ax[2].scatter(dauer_h, 100 * arc_anteil, s=5, alpha=0.25, color="#3182bd")
    ax[2].set_xlabel("Fensterdauer [h]")
    ax[2].set_ylabel("Anteil Lichtbogen am Fenster [%]")
    ax[2].set_title("B1  Lichtbogenanteil gegen Fensterdauer", fontsize=10)
    ax[2].axvline(cap_h, color="crimson", ls="--", lw=1)
    ax[2].grid(alpha=0.3)

    ax[3].hist(arc_start[np.isfinite(arc_start)] * 100, bins=60, color="darkorange")
    ax[3].set_xlabel("Beginn des Lichtbogens, in % des Fensters")
    ax[3].set_ylabel("Anzahl Billets")
    ax[3].set_title("B2  Beginn des Lichtbogens im Fenster", fontsize=10)
    C.schliesse(
        pdf,
        fig,
        fuss + f" | Fenster = Wechsel der Billetnummer bis zum nächsten, gedeckelt "
        f"auf {cap_h:g} h (data_prep.MAX_BILLET_DURATION_H) | "
        f"Lichtbogen = {ARC_KANAL} > {ARC_SCHWELLE:g}",
    )

    fig, ax = C.seite(
        pdf, "B3 — Ist die Fensterdauer ein Prozessmerkmal?", 1, 3, groesse=(11.7, 5.2)
    )

    abw = C.auftragsabweichung(y, df[SP["auftrag"]].astype(str))
    ok = np.isfinite(dauer_h)
    r_y, p_y = spearmanr(dauer_h[ok], y[ok])
    r_a, p_a = spearmanr(dauer_h[ok], abw[ok])
    r_arc, _ = spearmanr(arc_h[ok], y[ok])
    r_arc_a, _ = spearmanr(arc_h[ok], abw[ok])

    for i, (werte, lab) in enumerate(
        [(dauer_h, "Fensterdauer [h]"), (arc_h, "Lichtbogendauer [h]")]
    ):
        a = ax[i]
        a.scatter(werte[ok], y[ok], s=5, alpha=0.2, color="#3182bd")
        q = pd.qcut(pd.Series(werte[ok]), 10, duplicates="drop")
        gm = pd.Series(y[ok].to_numpy()).groupby(q.to_numpy(), observed=True).median()
        zentren = [iv.mid for iv in gm.index]
        a.plot(
            zentren, gm.to_numpy(), "o-", color="crimson", lw=1.5, label="Dezilmedian"
        )
        a.set_xlabel(lab)
        a.set_ylabel("y [%]")
        a.set_title(f"B3.{i + 1}  {lab.split(' [')[0]} gegen y", fontsize=10)
        a.legend(fontsize=7, loc="upper right")
        a.grid(alpha=0.3)

    ax[2].bar(
        ["Dauer→y", "Dauer→Abw.", "Lichtb.→y", "Lichtb.→Abw."],
        [r_y, r_a, r_arc, r_arc_a],
        color=["#3182bd", "#9ecae1", "#2ca02c", "#a1d99b"],
    )
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_ylabel("Spearman ρ")
    ax[2].set_title("B3.3  Spearman ρ", fontsize=10)
    ax[2].tick_params(axis="x", labelsize=8)
    ax[2].grid(alpha=0.3, axis="y")

    va_d = C.var_anteil_mit_null(
        dauer_h, df[SP["auftrag"]].astype(str).to_numpy(), n_perm=200
    )
    va_a = C.var_anteil_mit_null(
        arc_h, df[SP["auftrag"]].astype(str).to_numpy(), n_perm=200
    )
    C.schliesse(pdf, fig, fuss)


    folds = C.folds_final(df)
    strategien = [
        ("längennormiert (Stand)", lambda: T),
        ("auf Lichtbogen ausgerichtet", lambda: auf_arc_ausrichten(T)),
        (
            f"feste {PREFIX_H:g} h ab Fensterbeginn",
            lambda: fester_vorlauf(T, dauer_h, PREFIX_H),
        ),
    ]
    erg = []
    for name, bau in strategien:
        print(f"  MiniRocket / {name} ...", flush=True)
        G = np.ascontiguousarray(bau())
        pred = oof_rocket(G, y, folds)
        erg.append(
            {
                "name": name,
                "mae": C.mae(y, pred),
                "r2": C.r2(y, pred),
                "sd": C.falten_streuung(y, pred, folds)[1],
            }
        )
        print(f"     MAE {erg[-1]['mae']:.3f}  R² {erg[-1]['r2']:+.4f}")
        if G is not T:
            del G

    fig, ax = C.seite(
        pdf, "C — Drei Wege zu einheitlichen Batch-Längen", 1, 3, groesse=(11.7, 5.2)
    )

    C.vergleichstabelle(
        ax[0], erg, erg[0]["name"], "MAE", "C1  OOF-MAE, MiniRocket + RidgeCV"
    )

    k = R.CHANNELS.index(ARC_KANAL)
    bsp = int(np.argsort(-arc_anteil)[len(arc_anteil) // 2])
    kurven = [
        ("längennormiert", T[bsp, k]),
        ("arc-ausgerichtet", auf_arc_ausrichten(T[bsp : bsp + 1])[0, k]),
        (
            f"feste {PREFIX_H:g} h",
            fester_vorlauf(T[bsp : bsp + 1], dauer_h[bsp : bsp + 1], PREFIX_H)[0, k],
        ),
    ]
    for nm, kv in kurven:
        ax[1].plot(kv, lw=1.0, label=nm)
    ax[1].set_xlabel("Rasterpunkt (0 … 511)")
    ax[1].set_ylabel(ARC_KANAL)
    ax[1].set_title("C2  Dieselbe Schmelze, drei Fenster", fontsize=10)
    ax[1].legend(fontsize=7, loc="upper right")
    ax[1].grid(alpha=0.3)

    zellen = arc_anteil * T.shape[2]
    ax[2].hist(zellen[np.isfinite(zellen)], bins=60, color="indianred")
    ax[2].set_xlabel(f"Rasterzellen mit Lichtbogen (von {T.shape[2]})")
    ax[2].set_ylabel("Anzahl Billets")
    ax[2].set_title("C3  Rasterzellen mit Lichtbogen", fontsize=10)
    ax[2].axvline(np.nanmedian(zellen), color="k", ls="--", lw=1, label="Median")
    ax[2].legend(fontsize=7, loc="upper right")

    C.schliesse(
        pdf,
        fig,
        f"{len(folds)} artikelreine Falten | MiniRocket 1000 Kernel | "
        "alle drei Strategien auf demselben Tensor, damit nur die "
        "Fensterwahl variiert",
    )

    sd0 = erg[0]["sd"]

    def lage(e):
        d = e["mae"] - erg[0]["mae"]
        if abs(d) <= sd0:
            return f"innerhalb der Faltenstreuung (Δ MAE {d:+.3f}, ± {sd0:.3f})"
        return (
            f"{'SCHLECHTER' if d > 0 else 'BESSER'} als der Stand, ausserhalb der "
            f"Faltenstreuung (Δ MAE {d:+.3f}, ± {sd0:.3f})"
        )

    arc, fest = erg[1], erg[2]
    arc_klar_schlechter = arc["mae"] - erg[0]["mae"] > sd0
    arc_klar_besser = erg[0]["mae"] - arc["mae"] > sd0

    C.textseite(
        pdf,
        "Batch-Segmentierung und Zeitfenster — Entscheidung und Begründung",
        [
            f"ENTSCHEIDUNG: Ein Batch ist das Zeitfenster zwischen zwei "
            f"Wechseln der Billetnummer auf derselben Maschine, gedeckelt auf {cap_h:g} "
            f"Stunden. Für die tabellarischen Merkmale wird über dieses Fenster "
            f"aggregiert, für den Kurvenzweig wird es auf {R.SERIES_LEN} Punkte "
            f"längennormiert.",
            "",
            f"Befund A — die Längen sind uneinheitlich, und der Deckel ist "
            f"sichtbar. Die Fensterdauer reicht von {np.nanmin(dauer_h):.3f} bis "
            f"{cap_h:g} h bei einem Median von {np.nanmedian(dauer_h):.2f} h; "
            f"{int(am_cap.sum())} Billets ({100 * am_cap.mean():.1f} %) liegen "
            f"EXAKT am Deckel. Bei diesen ist die Fensterlänge keine Prozess"
            f"grösse mehr, sondern eine Konstante der Datenerhebung. Das ist der "
            f"Preis für eine Entscheidung, die notwendig war: ohne Deckel laufen "
            f"mehrtägige Reinigungsstillstände in die Rohdatei, die dadurch auf "
            f"1-2 GB wächst. Die Wahl von 9 h ist begründet — wann der Lichtbogen "
            f"über alle Billets endet, zeigt eda/25_lichtbogenende.py —, aber sie "
            f"hat diese Nebenwirkung, und die gehört berichtet.",
            "",
            f"Befund B — das Fenster ist überwiegend Stillstand. Laut Maske im "
            f"Raster ({ARC_KANAL} > {ARC_SCHWELLE:g}) brennt der Lichtbogen im Median "
            f"{np.nanmedian(arc_h):.2f} h, also {100 * np.nanmedian(arc_anteil):.1f} % "
            f"des Fensters; sein Beginn liegt im Median bei "
            f"{100 * np.nanmedian(arc_start):.1f} % des Fensters (Quartile "
            f"{100 * np.nanpercentile(arc_start, 25):.1f} bis "
            f"{100 * np.nanpercentile(arc_start, 75):.1f} %). Lichtbogen ≤ "
            f"{PREFIX_H:g} h: {100 * np.nanmean(arc_h <= PREFIX_H):.1f} % der Billets.",
            (
                f"VORBEHALT ZUR MASKE — diese Zahlen sind nach oben verzerrt. Bei "
                f"{100 * zelle0.mean():.0f} % der Billets gilt die Maske schon in der "
                f"ersten Rasterzelle als aktiv"
                + (
                    f", und in {100 * arcon_aus_in_zelle0:.1f} % dieser Fälle meldet der "
                    f"Zustandskanal des Lichtbogens dort 'aus'"
                    if np.isfinite(arcon_aus_in_zelle0)
                    else ""
                )
                + ". Ursache ist das ZOH-Resampling: es hält den ersten geschriebenen "
                "Wert rückwärts bis zum Fensterbeginn. A2, B1 und B2 enthalten diese "
                "Scheinzellen; auf den Rohkurven (16_sensorverlaeufe.pdf) ist der Bogen "
                "kürzer."
                if maske_verzerrt
                else f"Prüfung der Maske: nur bei {100 * zelle0.mean():.1f} % der Billets "
                "ist sie schon in der ersten Rasterzelle aktiv — die frühere Verzerrung "
                "durch das rückwärts gehaltene ZOH-Startsegment ist im aktuellen "
                "Tensor nicht mehr zu sehen."
            )
            + (
                f" Der Tensor ist im aktuellen Stand normiert ({ARC_KANAL} nimmt Werte "
                f"unter 0 an); die Schwelle {ARC_SCHWELLE:g} ist damit keine "
                "physikalische Leistung, sondern eine Schwelle in normierten Einheiten, "
                "und die Lichtbogendauer hängt an ihr."
                if arc_normiert
                else ""
            ),
            "Daraus folgt "
            f"unmittelbar eine Einschränkung für JEDES Aggregat der Arbeit: "
            f"'{ARC_KANAL}_mean' ist kein Mittel über den Lichtbogen, sondern "
            f"ein Mittel inklusive Leerlauf — und weil der Leerlauf"
            f"anteil je Billet zwischen "
            f"{100 * np.nanpercentile(arc_anteil, 5):.0f} und "
            f"{100 * np.nanpercentile(arc_anteil, 95):.0f} % schwankt, ist der "
            f"Mischungsanteil selbst eine Variable. Ein Aggregat, das NUR über "
            f"das Lichtbogenfenster gebildet wird, wäre die sachlich richtigere "
            f"Grösse; dass es sie nicht gibt, ist eine offene Baustelle und keine "
            f"Entscheidung.",
            "",
            f"Befund B3 — die Dauer selbst ist ein Auftragsmerkmal. Die "
            f"Fensterdauer korreliert mit y zu rho = {r_y:+.3f}, mit der "
            f"Abweichung vom Auftragsmittel dagegen nur zu {r_a:+.3f}; für die "
            f"Lichtbogendauer stehen {r_arc:+.3f} gegen {r_arc_a:+.3f}. Beide "
            f"folgen damit demselben Muster wie praktisch alle starken Merkmale "
            f"dieser Arbeit (03_merkmale.pdf): sie trennen Aufträge, nicht "
            f"Billets innerhalb eines Auftrags. Ein Varianzanteil von "
            f"{va_d['beobachtet']:.3f} zwischen den Aufträgen bei einer "
            f"Nullerwartung von {va_d['null_mittel']:.3f} sagt dasselbe "
            f"(Überschuss {va_d['ueberschuss']:+.3f}; Lichtbogendauer: "
            f"{va_a['beobachtet']:.3f} gegen {va_a['null_mittel']:.3f}, Überschuss "
            f"{va_a['ueberschuss']:+.3f}; {va_d['gruppen']} Aufträge auf "
            f"{va_d['n']} Billets).",
            "",
            f"Befund C, die Zahlen (MiniRocket 1000 Kernel + StandardScaler + "
            f"RidgeCV, {len(folds)} artikelreine Falten, alle drei Strategien auf "
            "demselben Tensor, damit nur die Fensterwahl variiert):",
            *C.zahlenzeilen(erg, erg[0]["name"]),
            f"Befund C. Der feste {PREFIX_H:g}-Stunden-Vorlauf liegt {lage(fest)}. "
            f"Die Ausrichtung am Lichtbogen liegt {lage(arc)}. "
            + (
                "Beschneidet man das Fenster auf den Teil, in dem der "
                "Schmelzprozess stattfindet, verliert der Kurvenzweig also an "
                "Vorhersagekraft."
                if arc_klar_schlechter
                else (
                    "Die Ausrichtung am Lichtbogen hilft damit messbar."
                    if arc_klar_besser
                    else "Keine der drei Fensterstrategien ist von den anderen "
                    "unterscheidbar."
                )
            ),
            (
                "Das ist ein unbequemer Befund: was der Kurvenzweig an y vorhersagt, "
                "steht dann nicht im Lichtbogen, sondern in dem, was ihn umgibt — in "
                "der Fensterlänge, im Leerlaufanteil, in der Lage des Prozesses im "
                "Fenster, also in Grössen der Fertigungsorganisation. Es fügt sich "
                "in die Kernthese der Arbeit ein: das Modell beschreibt den Auftrag, "
                "nicht das einzelne Billet."
                if arc_klar_schlechter
                else "Die Ausrichtung am Lichtbogen ist nicht erkennbar schlechter als "
                "der Stand (innerhalb der Faltenstreuung): Die Fensterwahl wirkt sich im "
                "Kurvenzweig nicht deutlich aus, was zu Befund D in 13_glaettung.pdf passt."
            ),
            "Die 'arc-ausgerichtete' Strategie hängt an der Maske aus Befund B; eine "
            "andere Definition (etwa über einen anderen Lichtbogenkanal) könnte das "
            "Ergebnis verschieben. Die Zahlen dieser Seite haben sich mit Änderungen "
            "am Kurven-Tensor bereits mehrfach bewegt und sind nur zusammen mit dem "
            "Stand des Tensors zu zitieren.",
            "Einschränkung, die den Befund C begrenzt. Alle drei Strategien "
            "wurden auf dem bereits gerasterten Tensor umgesetzt. Das korrigiert "
            "die Ausrichtung, stellt aber keine Auflösung her: der Lichtbogen "
            f"belegt im Median nur {np.nanmedian(zellen):.0f} der "
            f"{R.SERIES_LEN} Rasterzellen, und mehr wird er durch Strecken nicht. "
            "Eine Strategie, die den Tensor je Variante aus den Roh-CSVs neu baut "
            "— also 512 Punkte NUR über das Lichtbogenfenster abtastet —, wäre "
            "der vollständige Test. Er ist nicht gerechnet worden; angesichts "
            "des Befunds D aus 13_glaettung.pdf ist allerdings wenig davon zu "
            "erwarten.",
            "",
            "Konsequenz für die Arbeit. Die Fensterdefinition ist für die "
            "berichteten Ergebnisse unkritisch, weil diese aus den Stammdaten "
            "stammen und nicht aus dem Kurvenverlauf. Der Deckel von "
            f"{cap_h:g} Stunden bleibt und ist begründet. Was sich ändert, ist "
            "die Formulierung in der Arbeit: die Längennormierung auf "
            f"{R.SERIES_LEN} Punkte darf nicht als 'Vereinheitlichung "
            "unterschiedlich langer Batches' beschrieben werden, als wäre die "
            "Länge eine Störgrösse. Die Normierung wirft die absolute Zeitachse "
            "weg; batch_duration_s gleicht das als separates Merkmal teilweise aus.",
        ],
    )
    pdf.close()
    print(f"geschrieben: {C.OUT_DIR / '14_segmentierung.pdf'}")
    for e in erg:
        print(f"  {e['name']:<34} MAE {e['mae']:.3f}  R² {e['r2']:+.4f}")


if __name__ == "__main__":
    main()
