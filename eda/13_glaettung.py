"""13 — Glättung der Kurven.

In der Arbeit: Abschnitt 5.2: Tabelle 6 (zusätzliche Glättung und Rauschtest)
und Messpunkte je Rasterpunkt.

Zusätzliche Glättung gegen das Resampling allein (MiniRocket).
Braucht den Kurventensor bzw. KURVEN_DIR. Ausgabe: reports/eda/13_glaettung.pdf.

    python eda/13_glaettung.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import _common as C

sys.path.insert(0, str(C.ROOT / "data_pipeline"))
import rocket_regression as R  # noqa: E402

FENSTER = [1, 3, 9, 33, 65, 129]

RAUSCHEN = 0.5

SEED = 42


def gleitendes_mittel(X: np.ndarray, w: int) -> np.ndarray:
    """Gleitendes Mittel entlang der Zeitachse, Raender gespiegelt."""
    if w <= 1:
        return X
    pad = w // 2
    kern = np.ones(w, dtype=np.float32) / w
    out = np.empty_like(X)
    BLOCK = 256
    for a in range(0, len(X), BLOCK):
        b = min(a + BLOCK, len(X))
        Xp = np.pad(X[a:b], ((0, 0), (0, 0), (pad, pad)), mode="reflect")
        flach = Xp.reshape(-1, Xp.shape[-1])
        ziel = out[a:b].reshape(-1, X.shape[-1])
        for i in range(flach.shape[0]):
            ziel[i] = np.convolve(flach[i], kern, mode="valid")
        del Xp, flach
    return out


def gleitender_median(X: np.ndarray, w: int) -> np.ndarray:
    """Gleitender Median — robust gegen einzelne Ausreisser, erhaelt Sprungkanten."""
    if w <= 1:
        return X
    from scipy.ndimage import median_filter

    out = np.empty_like(X)
    BLOCK = 256
    for a in range(0, len(X), BLOCK):
        b = min(a + BLOCK, len(X))
        out[a:b] = median_filter(X[a:b], size=(1, 1, w), mode="reflect")
    return out


def oof_rocket(X3d, y, folds, tab=None):
    """MiniRocket + StandardScaler + RidgeCV out-of-fold — wie rocket_regression."""
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


def _beispielkurven(bid: str):
    """Rohe Messpunkte eines Billets, je Kanal — fuer die Bilder in A."""
    f = R.BILLET_DIR / f"{bid}.csv"
    if not f.exists():
        return None
    header = pd.read_csv(f, sep=";", nrows=0, encoding="utf-8").columns
    da = [c for c in R.CHANNELS if c in header]
    d = pd.read_csv(
        f, sep=";", usecols=["_time"] + da, low_memory=False, encoding="utf-8"
    )
    t = pd.to_datetime(d["_time"], errors="coerce", utc=True)
    d["_sec"] = (t - t.min()).dt.total_seconds()
    return d


def main():
    ds = C.lade(alle_arten=True)
    X, y, df = C.daten_teil(ds)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    df = df.reset_index(drop=True)
    fuss = C.kopfzeile(ds, alle_arten=True)
    pdf = C.pdf_oeffnen("13_glaettung.pdf")

    print("  lade Kurventensor ...")
    X3d = np.ascontiguousarray(R.build_or_load_tensor(ds.batch_ids)[C.eda_maske(ds)])
    print(f"  EDA-Tensor {X3d.shape}")

    fig, ax = C.seite(
        pdf, "A — Abtastraten und Resampling", 1, 3, groesse=(11.7, 5.2)
    )

    ns = {}
    for ch in R.CHANNELS:
        c = f"{ch}_n_samples"
        if c in X.columns:
            v = C.als_zahl(X[c]).dropna()
            if len(v) > 100 and v.median() > 0:
                ns[ch] = v
    ordnung = sorted(ns, key=lambda k: ns[k].median())
    ax[0].boxplot(
        [ns[k] for k in ordnung],
        vert=False,
        showfliers=False,
        widths=0.6,
    )
    ax[0].set_yticklabels([k[:24] for k in ordnung], fontsize=6)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("Messpunkte je Schmelze (logarithmisch)")
    ax[0].axvline(
        R.SERIES_LEN,
        color="crimson",
        ls="--",
        lw=1.2,
        label=f"Rasterpunkte ({R.SERIES_LEN})",
    )
    ax[0].set_title("A1  Messpunkte je Kanal", fontsize=10)
    ax[0].legend(fontsize=7, loc="lower right")

    bid = str(ds.batch_ids[C.eda_maske(ds)].iloc[0])
    beispiel_punkte = {}
    zwischen = None
    roh = _beispielkurven(bid)
    if roh is not None:
        dicht = max(ns, key=lambda k: ns[k].median() if k in roh.columns else -1)
        belegt = [
            k for k in ns
            if k in roh.columns and R._to_num_any(roh[k]).notna().sum() >= 5 and k != dicht
        ]
        duenn = min(belegt, key=lambda k: ns[k].median())
        for i, ch in enumerate([dicht, duenn]):
            a = ax[1 + i]
            v = R._to_num_any(roh[ch]).to_numpy()
            sec = roh["_sec"].to_numpy()
            gr = R.GRENZEN.get(ch, R.MAX_PLAUSIBLE_ABS)
            m = np.isfinite(sec) & np.isfinite(v) & (np.abs(v) <= gr)
            if m.sum() < 5:
                C.leer(a)
                continue
            s = pd.Series(v[m], index=sec[m]).groupby(level=0).mean().sort_index()
            tt, vv = s.index.to_numpy(), s.to_numpy()
            kanten = np.linspace(tt.min(), tt.max(), R.SERIES_LEN + 1)
            mitte = (kanten[:-1] + kanten[1:]) / 2
            zoh = R._zoh_resample(tt, vv, kanten)
            lin = np.interp(mitte, tt, vv)

            a.plot(tt / 60, vv, lw=0.4, color="0.7", label="Rohwerte")
            a.plot(mitte / 60, zoh, lw=1.0, color="#2ca02c", label="ZOH-Zellmittel")
            a.plot(
                mitte / 60,
                lin,
                lw=1.0,
                color="#d62728",
                ls="--",
                label="lineare Interpolation",
            )
            a.set_title(f"A{2 + i}  {ch}", fontsize=10)
            beispiel_punkte[ch] = len(tt)
            a.set_xlabel("Minuten seit Fensterbeginn")
            a.legend(fontsize=6.5)
            if i == 1 and vv.max() <= 1.001:
                zwischen = int(((lin > 0.02) & (lin < 0.98)).sum())
    else:
        C.leer(ax[1])
        C.leer(ax[2])
    kern = {k: ns[k].median() / R.SERIES_LEN for k in ordnung}

    C.schliesse(pdf, fig, fuss + f" | Beispielschmelze {bid}")

    fig, ax = C.seite(pdf, "B — Was Glättung mit den Aggregaten macht", 2, 2)

    rng = np.random.default_rng(SEED)
    stich = rng.choice(len(X3d), size=min(400, len(X3d)), replace=False)
    Xs = X3d[stich]

    agg = {"mean": [], "std": [], "max": [], "range": []}
    mean_abw = []
    for w in FENSTER:
        G = gleitendes_mittel(Xs, w)
        agg["mean"].append(G.mean(axis=2))
        agg["std"].append(G.std(axis=2))
        agg["max"].append(G.max(axis=2))
        agg["range"].append(G.max(axis=2) - G.min(axis=2))

    for i, name in enumerate(["mean", "std", "max", "range"]):
        a = ax[i]
        basis = agg[name][0]
        for k, ch in enumerate(R.CHANNELS):
            b = basis[:, k]
            if not np.isfinite(b).all() or np.abs(b).max() < 1e-9:
                continue
            rel = [
                float(
                    np.nanmedian(
                        np.abs(agg[name][j][:, k]) / np.maximum(np.abs(b), 1e-9)
                    )
                )
                for j in range(len(FENSTER))
            ]
            a.plot(FENSTER, rel, lw=0.8, alpha=0.55)
            if name == "mean":
                nenner = float(np.nanmedian(np.abs(b)))
                if nenner > 1e-3:
                    mean_abw.append(
                        (float(np.nanmedian(np.abs(agg[name][-1][:, k] - b))) / nenner, ch)
                    )
        a.set_xscale("log")
        a.set_xlabel("Fensterbreite [Rasterpunkte]")
        a.set_ylabel(f"{name} relativ zu ungeglättet")
        a.axhline(1.0, color="k", lw=1)
        a.set_title(f"B{i + 1}  {name}", fontsize=10)
        a.grid(alpha=0.3)
        a.set_ylim(0, 1.4)
    C.schliesse(
        pdf,
        fig,
        f"je Linie ein Kanal, {len(stich)} zufällige Billets | "
        "gleitendes Mittel, Ränder gespiegelt | "
        "mean MUSS bei 1 bleiben — weicht es ab, ist die Randbehandlung schuld",
    )

    folds = C.folds_final(df)
    erg = []
    for w in FENSTER:
        for art, fn in [("Mittel", gleitendes_mittel), ("Median", gleitender_median)]:
            if w == 1 and art == "Median":
                continue
            name = "ungeglättet (Stand)" if w == 1 else f"{art} w={w}"
            print(f"  MiniRocket / {name} ...", flush=True)
            G = np.ascontiguousarray(fn(X3d, w))
            pred = oof_rocket(G, y, folds)
            erg.append(
                {
                    "name": name,
                    "w": w,
                    "art": art,
                    "mae": C.mae(y, pred),
                    "r2": C.r2(y, pred),
                    "sd": C.falten_streuung(y, pred, folds)[1],
                }
            )
            print(f"     MAE {erg[-1]['mae']:.3f}  R² {erg[-1]['r2']:+.4f}")
            del G

    fig, ax = C.seite(
        pdf, "C/D — Ändert Glättung die Modellgüte?", 1, 3, groesse=(11.7, 5.2)
    )

    basis = erg[0]
    C.vergleichstabelle(
        ax[0], erg, basis["name"], "MAE", "C1  MiniRocket + RidgeCV, OOF-MAE"
    )

    for art, farbe in [("Mittel", "#3182bd"), ("Median", "#ff7f0e")]:
        pkt = [e for e in erg if e["art"] == art or e["w"] == 1]
        pkt = sorted(pkt, key=lambda e: e["w"])
        ax[1].plot(
            [e["w"] for e in pkt], [e["r2"] for e in pkt], "o-", color=farbe, label=art
        )
    ax[1].set_xscale("log")
    ax[1].set_xlabel("Fensterbreite [Rasterpunkte]")
    ax[1].set_ylabel("OOF-R²")
    ax[1].axhline(basis["r2"], color="k", ls="--", lw=1, label="ungeglättet")
    ax[1].axhline(0, color="0.6", lw=0.8)
    ax[1].set_title("C2  Dieselbe Frage über die Fensterbreite", fontsize=10)
    ax[1].legend(fontsize=7, loc="upper left")
    ax[1].grid(alpha=0.3)

    print("  Gegentest: künstlich verrauschte Kurven ...", flush=True)
    kanal_sd = X3d.std(axis=(0, 2), keepdims=True).astype(np.float32)
    rng2 = np.random.default_rng(SEED)
    Xn = np.empty_like(X3d)
    BLOCK = 256
    for a in range(0, len(X3d), BLOCK):
        b = min(a + BLOCK, len(X3d))
        rausch = rng2.standard_normal(X3d[a:b].shape, dtype=np.float32)
        rausch *= RAUSCHEN * kanal_sd
        np.add(X3d[a:b], rausch, out=Xn[a:b])
        del rausch
    gegen = []
    for w in [1, 9, 33]:
        name = "verrauscht, roh" if w == 1 else f"verrauscht, Mittel w={w}"
        print(f"    {name} ...", flush=True)
        G = np.ascontiguousarray(gleitendes_mittel(Xn, w))
        pred = oof_rocket(G, y, folds)
        gegen.append(
            {
                "name": name,
                "w": w,
                "mae": C.mae(y, pred),
                "r2": C.r2(y, pred),
                "sd": C.falten_streuung(y, pred, folds)[1],
            }
        )
        print(f"       MAE {gegen[-1]['mae']:.3f}  R² {gegen[-1]['r2']:+.4f}")
        del G
    del Xn

    C.vergleichstabelle(
        ax[2],
        gegen,
        gegen[0]["name"],
        "MAE",
        f"D1  Gegentest mit Rauschen {RAUSCHEN:g}·SD",
    )

    C.schliesse(
        pdf,
        fig,
        f"MiniRocket 1000 Kernel + StandardScaler + RidgeCV | {len(folds)} "
        "artikelreine Falten | Vorhersagen auf den y-Bereich der Trainingsfalte "
        "begrenzt (wie im berichteten Lauf)",
    )

    beste = min(erg, key=lambda e: e["mae"])
    spanne = max(e["mae"] for e in erg) - min(e["mae"] for e in erg)
    gegen_hilft = gegen[0]["mae"] - min(g["mae"] for g in gegen[1:])
    mean_abw.sort(reverse=True)
    C.textseite(
        pdf,
        "Glättung gegen Rohdaten — Entscheidung und Begründung",
        [
            "ENTSCHEIDUNG: keine zusätzliche Glättung. Die Kurven gehen so in die "
            "Modelle, wie das ZOH-Resampling sie liefert.",
            "",
            "Wichtiger als die Entscheidung ist die Feststellung, dass sie schon "
            "einmal getroffen wurde. 'Rohdaten' gibt es in dieser Pipeline nicht: "
            "ereignisgesteuerte Kanäle MUESSEN auf ein gemeinsames Raster, und "
            "jedes Verfahren dafür ist eine Filterentscheidung. Die Arbeit benutzt "
            "Zero-Order-Hold mit zeitgewichtetem Zellmittel "
            "statt linearer Interpolation. Das ist die eigentliche Antwort auf "
            "'Glättung vs. Rohdaten', und sie ist nicht kosmetisch: bei den "
            f"dichtesten Kanälen mittelt das Verfahren über rund "
            f"{max(kern.values()):.0f} Messpunkte je "
            "Rasterzelle (Antialiasing), bei den dünnsten hält es den letzten "
            "Wert, statt Zwischenzustände zu erfinden, die es physikalisch nicht "
            "gibt.",
            "",
            f"Befund A — die Abtastraten liegen um Faktor "
            f"{ns[ordnung[-1]].median() / max(ns[ordnung[0]].median(), 1):.0f} "
            f"auseinander. Ein EINHEITLICHES Glättungsfenster ist deshalb gar "
            f"nicht definierbar: dieselben 9 Rasterpunkte entsprechen bei "
            f"den dichtesten Kanälen mehreren hundert Messpunkten und bei einem Zustandskanal "
            f"mehr Punkten, als der Kanal überhaupt hat. Nur das ZOH-Zellmittel "
            f"ist ein Verfahren für beide Regime.",
            f"Messpunkte je Schmelze (Median) und effektive Kernbreite N/{R.SERIES_LEN} "
            "des Resamplings, dichteste Kanäle zuerst (Kern < 1: der Kanal hat weniger "
            "Punkte als Rasterzellen, ZOH HÄLT dann Werte, statt zu mitteln):",
            *[
                f"  {k}: {ns[k].median():.0f} Messpunkte, Kern {kern[k]:.2f}"
                for k in ordnung[::-1]
            ],
            f"Feld A2 und A3 zeigen die Beispielschmelze {bid} "
            + ", ".join(f"({k}: {n} Rohpunkte)" for k, n in beispiel_punkte.items())
            + ". Grün ist das ZOH-Zellmittel, rot gestrichelt die lineare Interpolation"
            + (
                f"; beim Zustandskanal erzeugt sie {zwischen} Rasterpunkte mit "
                "Zwischenzuständen, die es physikalisch nicht gibt."
                if zwischen is not None
                else "."
            ),
            "",
            f"Befund B — die Aggregate sind nicht glättungsneutral (je Linie ein "
            f"Kanal, Median über {len(stich)} zufällige Billets; gleitendes Mittel "
            f"mit gespiegelten Rändern). mean sollte als Kontrolle bei 1 bleiben. "
            + (
                f"Es tut das nicht überall: beim breitesten Fenster weicht es bei "
                f"{mean_abw[0][1]} um {100 * mean_abw[0][0]:.0f} % ab"
                + (
                    f", bei {mean_abw[1][1]} um {100 * mean_abw[1][0]:.0f} %"
                    if len(mean_abw) > 1 and mean_abw[1][0] > 0.02
                    else ""
                )
                + ". Ein gleitendes Mittel verändert den Mittelwert nur über die "
                "Ränder; die Abweichung heisst also, dass diese Kanäle am Fensterrand "
                "viel Signal tragen und die Spiegelung es dort verdoppelt. Für die "
                "Aussage zu std, max und range ändert das nichts, die Randbehandlung "
                "ist aber bei diesen Kanälen nicht neutral. "
                if mean_abw and mean_abw[0][0] > 0.02
                else "Das tut es (Abweichung unter 2 %). "
            )
            + "std, max und range schrumpfen dagegen monoton mit der Fensterbreite. "
            "Daraus folgt "
            "eine Lesevorschrift für die Arbeit: ein Merkmal wie '<Kanal>_std' "
            "misst NICHT die Prozessschwankung allein, sondern die Schwankung "
            "NACH einer kanalabhängig breiten Mittelung. Zwei Kanäle mit "
            "verschiedener Schreibrate sind in ihren std-Werten deshalb nicht "
            "direkt vergleichbar. Das erklärt mit, warum '*_n_samples' in den "
            "Importance-Ranglisten so weit oben steht: die Abtastrate ist ein "
            "Maschinen- und Zustandsfingerabdruck und steckt indirekt auch in den "
            "anderen Aggregaten desselben Kanals.",
            "",
            f"Befund C, die Zahlen (MiniRocket 1000 Kernel + StandardScaler + "
            f"RidgeCV, {len(folds)} artikelreine Falten, Vorhersagen auf den "
            "y-Bereich der Trainingsfalte begrenzt):",
            *C.zahlenzeilen(erg, basis["name"]),
            "Gegentest:",
            *C.zahlenzeilen(gegen, gegen[0]["name"]),
            f"Befund C — zusätzliche Glättung bringt nichts. Über Fensterbreiten "
            f"von 1 bis {max(FENSTER)} Rasterpunkten und beide Filterarten liegen "
            f"die Ergebnisse {spanne:.3f} MAE-Punkte auseinander, bei einer "
            f"Faltenstreuung von ±{basis['sd']:.3f}. Bestes Ergebnis: "
            f"{beste['name']} mit MAE {beste['mae']:.3f} gegen "
            f"{basis['mae']:.3f} ungeglättet. Auch die breitesten Fenster "
            f"verschlechtern kaum — was für sich genommen schon aussagt, dass die "
            f"feine Struktur der Kurven nicht viel zur Vorhersage beiträgt.",
            "",
            f"Befund D — der Gegentest fällt deutlicher aus als Befund C und "
            f"verändert dessen Lesart. Auf jeden Kanal wurde Rauschen in Höhe "
            f"von {RAUSCHEN:g}·SD aufaddiert, also eine massive Störung. Das "
            f"Ergebnis: MAE {gegen[0]['mae']:.3f} gegen {basis['mae']:.3f} "
            f"ungestört — die verrauschten Kurven sind "
            + (
                "sogar minimal BESSER"
                if gegen[0]["mae"] < basis["mae"]
                else "gleich gut"
            )
            + f". Glättung holt davon {gegen_hilft:+.3f} MAE-Punkte zurück, "
            f"also nichts.",
            "",
            "Was daraus folgt, ist kein Ergebnis über Glättung, sondern eines "
            "über die Kurven. Ein Modell, dessen Güte sich nicht ändert, wenn "
            "man die Eingangssignale halb zerstört, liest aus diesen Signalen "
            "kaum etwas ab. Das passt zum Gesamtbild der Arbeit: MiniRocket auf "
            f"reinen Kurven erreicht hier OOF-R² {basis['r2']:+.4f}, während "
            "XGBoost auf den Stammdaten deutlich höher liegt, und die "
            "Feature-Source-Messung in 15_featureextraktion.pdf (Befund D) zeigt, "
            "dass die Prozesskanäle zu den Stammdaten nur wenig hinzufügen. "
            "Befund C ('Glättung ändert nichts') ist damit richtig, aber "
            "schwach belegt: er ist nicht der Nachweis einer bereits optimalen "
            "Aufbereitung, sondern die Folge davon, dass in diesem Zweig zu wenig "
            "Signal steckt, um Unterschiede der Vorverarbeitung sichtbar zu "
            "machen. So und nicht anders gehört er zitiert.",
            "",
            "Offen und als Limitation zu nennen. Die ZOH-Randbehandlung hält den "
            "ersten Wert rückwärts und den letzten vorwärts. Schreibt ein Kanal "
            "nur während der Schmelze und schweigt danach, wird sein letzter Wert "
            "über den Rest des Fensters gehalten — bei einem 9-Stunden-Fenster mit "
            "ein bis zwei Stunden Lichtbogen ist das der grössere Teil. Sauberer "
            "wäre ein zweiter Maskenkanal je Signal ('bekannt/unbekannt'), was die "
            "Kanalzahl von 29 auf 58 verdoppelte. Wie gross das Fensterproblem ist, "
            "steht in 14_segmentierung.pdf.",
        ],
    )
    pdf.close()
    print(f"geschrieben: {C.OUT_DIR / '13_glaettung.pdf'}")
    for e in erg + gegen:
        print(f"  {e['name']:<26} MAE {e['mae']:.3f}  R² {e['r2']:+.4f}")


if __name__ == "__main__":
    main()
