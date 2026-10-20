"""05 — Zeit und Drift.

In der Arbeit: Abschnitt 4.2.7: Tabelle 5 (Ausschussgrad je Quartal) und die
Zahl der Billets mit rekonstruiertem Schmelzende.

Trendtests, Quartale, Trend innerhalb der Artikel, Drift der Merkmale.
Ausgabe: reports/eda/05_zeit.pdf, runs/eda/05_zeit.json.

    python eda/05_zeit.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import _common as C
from data_prep import SP  # noqa: E402


def mann_kendall(x: np.ndarray) -> tuple[float, float]:
    """Verteilungsfreier Trendtest (normalapproximiert), als Gegenprobe zu Spearman."""
    n = len(x)
    if n < 10:
        return (np.nan, np.nan)
    r = pd.Series(x).rank().to_numpy()
    s = 0.0
    for i in range(n - 1):
        s += np.sign(r[i + 1 :] - r[i]).sum()
    var = n * (n - 1) * (2 * n + 5) / 18
    z = (s - np.sign(s)) / np.sqrt(var) if var > 0 else 0.0
    from scipy.stats import norm

    return float(z), float(2 * (1 - norm.cdf(abs(z))))


N_BOOT_MIX = 500


def _steigung(d: pd.DataFrame, fe: str | None = None) -> float:
    """OLS-Steigung von y auf die Zeit in pp/Monat, optional innerhalb von fe."""
    x = d.tage.to_numpy() / 30.44
    yy = d.y.to_numpy()
    if fe is None:
        x, yy = x - x.mean(), yy - yy.mean()
    else:
        x = x - d.groupby(fe).tage.transform("mean").to_numpy() / 30.44
        yy = yy - d.groupby(fe).y.transform("mean").to_numpy()
    return float((x * yy).sum() / (x * x).sum())


def _quartalseffekte(d: pd.DataFrame, fe: tuple[str, ...] = ()) -> np.ndarray:
    """Quartalseffekte relativ zum ersten Quartal in d, per Dummy-OLS mit
    Fixeffekten."""
    Z = pd.get_dummies(d.q, dtype=float)
    Z = Z.drop(columns=Z.columns[0])
    k = Z.shape[1]
    for f in fe:
        Z = Z.join(pd.get_dummies(d[f], prefix=f, drop_first=True, dtype=float))
    Z.insert(0, "c", 1.0)
    b, *_ = np.linalg.lstsq(Z.to_numpy(), d.y.to_numpy(), rcond=None)
    return b[1 : 1 + k]


def _boot_auftrag(d: pd.DataFrame, fn, seed: int = C.SEED) -> np.ndarray:
    """95-%-Intervall per Bootstrap über ganze Aufträge (Billets eines Auftrags sind
    nicht unabhängig, siehe 02_ebenen.pdf)."""
    rng = np.random.default_rng(seed)
    auf = d.auf.unique()
    idx = d.groupby("auf").indices
    werte = []
    for _ in range(N_BOOT_MIX):
        zeilen = np.concatenate([idx[a] for a in rng.choice(auf, len(auf))])
        werte.append(np.atleast_1d(fn(d.iloc[zeilen].reset_index(drop=True))))
    return np.percentile(np.array(werte), [2.5, 97.5], axis=0)


def mix_seiten(pdf, T: pd.DataFrame, df: pd.DataFrame, fuss: str):
    M = T.assign(
        art=df.loc[T.index, SP["artikel"]],
        auf=df.loc[T.index, SP["auftrag"]],
        masch=df.loc[T.index, SP["ofen_kennung"]],
        glas=df.loc[T.index, "Glastyp"],
    ).dropna(subset=["art"])
    M["tage"] = (M.t - M.t.min()).dt.total_seconds() / 86400
    M["q"] = M.t.dt.to_period("Q").astype(str)
    M = M.reset_index(drop=True)

    steig = []
    for name, d in (("alle Billets", M), ("nur 2026", M[M.t >= "2026-01-01"])):
        d = d.reset_index(drop=True)
        for art, fe in (("roh", None), ("innerhalb der Artikel", "art")):
            wert = _steigung(d, fe)
            lo, hi = _boot_auftrag(d, lambda z, fe=fe: _steigung(z, fe))[:, 0]
            steig.append({"menge": name, "art": art, "wert": wert, "lo": lo, "hi": hi})
    S = pd.DataFrame(steig)

    M26 = M[M.t >= "2026-01-01"].reset_index(drop=True)
    quartale = sorted(M26.q.unique())
    varianten = (
        ("roh", ()),
        ("innerhalb der Artikel", ("art",)),
        ("innerhalb Artikel + Maschine", ("art", "masch")),
    )
    QE = []
    for name, fe in varianten:
        wert = _quartalseffekte(M26, fe)
        ci = _boot_auftrag(M26, lambda z, fe=fe: _quartalseffekte(z, fe))
        for j, q in enumerate(quartale[1:]):
            QE.append({"variante": name, "q": q, "wert": wert[j], "lo": ci[0, j], "hi": ci[1, j]})
    QE = pd.DataFrame(QE)

    r_roh, _ = spearmanr(M.tage, M.y)
    r_in, p_in = spearmanr(M.tage, M.y - M.groupby("art").y.transform("mean"))
    beide = M26[M26.q.isin(quartale[:2])].groupby("art").q.nunique()
    n_beide_art = int((beide == 2).sum())
    n_beide_bil = int(M26[M26.art.isin(beide[beide == 2].index) & M26.q.isin(quartale[:2])].shape[0])

    def zeile(r):
        return f"{r.wert:+.2f} [{r.lo:+.2f}; {r.hi:+.2f}]"

    q2 = QE[QE.q == quartale[1]].set_index("variante")
    anteil_q2 = q2.loc["innerhalb der Artikel", "wert"] / q2.loc["roh", "wert"]
    C.textseite(
        pdf,
        "Artikelmix oder Veränderung innerhalb der Artikel?",
        [
            "Frage: Sinkt y, weil später andere Artikel gefertigt werden (Mix), oder weil "
            "derselbe Artikel später weniger Ausschuss hat? Methode: Fixeffekte. Ein Modell, "
            "das Artikel und Zeit gleichzeitig enthält, vergleicht nur Billets desselben "
            "Artikels miteinander; was dann vom Trend übrig bleibt, liegt innerhalb der "
            "Artikel, die Differenz zum rohen Trend ist der Mix. Intervalle: 95 %, "
            f"Bootstrap über ganze Aufträge ({N_BOOT_MIX} Ziehungen).",
            f"Rangkorrelation mit der Zeit: roh ρ = {r_roh:+.3f}, nach Abzug des "
            f"Artikelmittels ρ = {r_in:+.3f} ({C.p_text(p_in)}).",
            "Linearer Trend in Prozentpunkten je Monat:",
            *[
                f"  {r.menge}, {r.art}: {zeile(r)}"
                for r in S.itertuples()
            ],
            f"Quartalseffekte 2026 relativ zu {quartale[0]} (Prozentpunkte):",
            *[f"  {r.q}, {r.variante}: {zeile(r)}" for r in QE.itertuples()],
            f"Befund: Der Rückgang {quartale[0]} → {quartale[1]} bleibt zu "
            f"{100 * anteil_q2:.0f} % bestehen, wenn nur Billets desselben Artikels "
            "verglichen werden; auch die Maschinenverteilung erklärt ihn nicht. Über den "
            "ganzen Zeitraum liegt der Anteil innerhalb der Artikel bei "
            f"{100 * S.wert[1] / S.wert[0]:.0f} %. Der Artikelmix trägt also zum kleineren "
            f"Teil bei. Grundlage des Vergleichs sind {n_beide_art} Artikel mit "
            f"{n_beide_bil} Billets, die in {quartale[0]} und {quartale[1]} gefertigt wurden.",
            f"VORBEHALT 1: {quartale[-1]} ist klein; sein Intervall schliesst null ein. "
            "2025 ist für eine eigene Aussage zu dünn und fehlt in den Quartalseffekten.",
            "VORBEHALT 2: 'Innerhalb der Artikel' heisst nicht 'Prozess wird besser'. Gezeigt "
            "ist nur, dass dasselbe Produkt später weniger Ausschuss hat. Rohstoff, "
            "Auftragslage oder die Erfassung der Istmasse können sich ebenso über die "
            "Zeit geändert haben.",
            "VORBEHALT 3 (Methode): Die naheliegende Zerlegung 'Quartalsmittel minus "
            "Mix-Erwartung aus den Artikelmitteln' überschätzt den Mix, weil die "
            "Artikelmittel über den ganzen Zeitraum den Trend selbst enthalten: ein "
            "Artikel, der vor allem spät gefertigt wird, bekommt ein niedrigeres Mittel. "
            "Die Fixeffekt-Schätzung hat dieses Problem nicht.",
        ],
    )

    fig, ax = C.seite(pdf, "Artikelmix oder Veränderung innerhalb der Artikel?", 2, 2)

    farben = {"roh": "lightgrey", "innerhalb der Artikel": "steelblue",
              "innerhalb Artikel + Maschine": "navy"}
    mengen = list(dict.fromkeys(S.menge))
    for j, art in enumerate(("roh", "innerhalb der Artikel")):
        d = S[S.art == art]
        xs = np.arange(len(mengen)) + (j - 0.5) * 0.35
        ax[0].bar(xs, d.wert, width=0.35, color=farben[art], label=art,
                  yerr=[d.wert - d.lo, d.hi - d.wert], capsize=3)
    ax[0].set_xticks(np.arange(len(mengen)))
    ax[0].set_xticklabels(mengen)
    ax[0].axhline(0, color="black", lw=0.8)
    ax[0].set_ylabel("Trend [Prozentpunkte je Monat]")
    ax[0].set_title("A  Linearer Trend: roh gegen innerhalb der Artikel", fontsize=10)
    ax[0].legend(fontsize=8, loc="lower right")
    ax[0].grid(alpha=0.3, axis="y")

    qs = quartale[1:]
    for j, (name, _) in enumerate(varianten):
        d = QE[QE.variante == name]
        xs = np.arange(len(qs)) + (j - 1) * 0.26
        ax[1].bar(xs, d.wert, width=0.26, color=farben[name], label=name,
                  yerr=[d.wert - d.lo, d.hi - d.wert], capsize=3)
    ax[1].set_xticks(np.arange(len(qs)))
    ax[1].set_xticklabels([f"{q}\n(n = {int((M26.q == q).sum())})" for q in qs])
    ax[1].axhline(0, color="black", lw=0.8)
    ax[1].set_ylabel(f"Unterschied zu {quartale[0]} [Prozentpunkte]")
    ax[1].set_title(f"B  Quartale 2026 gegen {quartale[0]} "
                    f"(n = {int((M26.q == quartale[0]).sum())})", fontsize=10)
    ax[1].legend(fontsize=8, loc="upper left")
    ax[1].grid(alpha=0.3, axis="y")

    monat = M.t.dt.to_period("M")
    anz = monat.value_counts().sort_index()
    gueltig = anz[anz >= 20].index
    roh_m = (M.y - M.y.mean()).groupby(monat).mean().reindex(gueltig)
    in_m = (M.y - M.groupby("art").y.transform("mean")).groupby(monat).mean().reindex(gueltig)
    x_m = gueltig.to_timestamp()
    ax[2].plot(x_m, roh_m, "o-", color="grey", ms=3, label="y minus Gesamtmittel")
    ax[2].plot(x_m, in_m, "o-", color="steelblue", ms=3, label="y minus Artikelmittel")
    ax[2].axhline(0, color="black", lw=0.8)
    ax[2].set_ylabel("Monatsmittel [Prozentpunkte]")
    ax[2].set_title("C  Monatsmittel, roh und innerhalb der Artikel (Monate ≥ 20 Billets)",
                    fontsize=10)
    ax[2].legend(fontsize=8)
    ax[2].tick_params(axis="x", labelsize=7, rotation=25)
    ax[2].grid(alpha=0.3)

    anteile = pd.crosstab(M.q, M.glas.fillna("(kein Wert)"), normalize="index")
    haupt = anteile.columns[anteile.max() >= 0.05]
    anteile = anteile[haupt].assign(**{"übrige": 1 - anteile[haupt].sum(axis=1)})
    unten = np.zeros(len(anteile))
    for spalte in anteile.columns:
        ax[3].bar(range(len(anteile)), anteile[spalte], bottom=unten, label=spalte)
        unten += anteile[spalte].to_numpy()
    n_q = M.q.value_counts()
    ax[3].set_xticks(range(len(anteile)))
    ax[3].set_xticklabels([f"{q}\n(n = {n_q[q]})" for q in anteile.index], fontsize=7)
    ax[3].yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f} %")
    ax[3].set_ylabel("Anteil der Billets")
    ax[3].set_title("D  Wie sich der Mix verschiebt: Glastyp je Quartal", fontsize=10)
    ax[3].legend(fontsize=7, loc="lower left")
    C.schliesse(pdf, fig, fuss)


def main():
    ds = C.lade(alle_arten=True)
    X, y, df = C.daten_teil(ds)
    fuss = C.kopfzeile(ds, alle_arten=True)
    pdf = C.pdf_oeffnen("05_zeit.pdf")

    ts = (
        ds.melt_end_ts[C.eda_maske(ds)].reset_index(drop=True)
        if len(ds.melt_end_ts)
        else pd.Series(index=df.index)
    )
    ts = pd.to_datetime(ts, errors="coerce")
    ok = ts.notna()
    T = pd.DataFrame({"t": ts[ok], "y": y[ok]})
    Ts = T.sort_values("t")

    rho, p_rho = spearmanr(T.t.astype("int64"), T.y)
    z_mk, p_mk = mann_kendall(Ts.y.to_numpy())
    quartal = Ts.set_index("t").y.resample("QE").agg(["median", "mean", "count"])
    vor_2026 = int((T.t < "2026-01-01").sum())

    q_zeilen = []
    for stichtag, teil in Ts.set_index("t").y.resample("QE"):
        name = f"{stichtag.year}-Q{(stichtag.month - 1) // 3 + 1}"
        if len(teil) < 5:
            q_zeilen.append({"name": name, "n": len(teil), "m": np.nan, "lo": np.nan, "hi": np.nan,
                             "mittel": teil.mean()})
            continue
        lo, hi = C.bootstrap_ci(teil.to_numpy(), np.median, n_boot=500)
        q_zeilen.append({"name": name, "n": len(teil), "m": teil.median(), "lo": lo, "hi": hi,
                         "mittel": teil.mean()})
    Q = pd.DataFrame(q_zeilen)
    C.speichere_json(
        "05_zeit",
        {
            "n_mit_zeitstempel": int(len(T)),
            "n_ohne_zeitstempel": int((~ok).sum()),
            "spearman_zeit_y": float(rho),
            "mann_kendall_z": float(z_mk),
            "quartale": [
                {"quartal": z["name"], "n": int(z["n"]), "median": z["m"], "mittel": z["mittel"],
                 "median_ki": [z["lo"], z["hi"]]}
                for _, z in Q.iterrows()
            ],
        },
    )

    num = C.numerische_merkmale(X)
    tab = C.spearman_tabelle(X.loc[ok], y[ok], num).head(12)
    drift = []
    for _, r in tab.iterrows():
        s = C.als_zahl(X[r.merkmal]).loc[ok]
        gut = s.notna()
        rr, pp = spearmanr(T.t[gut].astype("int64"), s[gut])
        drift.append({"merkmal": r.merkmal, "rho": rr, "p": pp, "s": s[gut]})

    C.textseite(
        pdf,
        "05 — Zeit und Drift",
        [
            f"{len(T)} Billets mit rekonstruiertem Schmelzende, "
            f"{T.t.min():%d.%m.%Y} bis {T.t.max():%d.%m.%Y} "
            f"({int((~ok).sum())} ohne Zeitstempel).",
            f"  Spearman(Zeit, y) = {rho:+.3f}, {C.p_text(p_rho)}",
            f"  Mann-Kendall z = {z_mk:+.2f}, {C.p_text(p_mk)}",
            "Beide Tests sind rangbasiert; sie stimmen in der Richtung überein.",
            f"VORBEHALT: nur {vor_2026} von {len(T)} Billets liegen vor 2026 "
            f"({100 * vor_2026 / len(T):.1f} %). Der Trend vergleicht im Kern eine "
            "kleine frühe gegen eine grosse späte Stichprobe. Die Richtung ist "
            "belastbar, die Steigung nicht. Der rollende Median ist deshalb nur dort "
            "gezeichnet, wo das 30-Tage-Fenster mindestens 20 Billets enthält.",
            "Quartale (Median mit 95-%-Bootstrap-Intervall, n):",
            *[
                f"  {z['name']}: n {z['n']}"
                + (
                    f", Median {z['m']:.1f} % [{z['lo']:.1f}; {z['hi']:.1f}]"
                    if not np.isnan(z["m"])
                    else " (zu wenige für ein Intervall)"
                )
                for _, z in Q.iterrows()
            ],
            "Drift der zwölf am stärksten mit y korrelierten Merkmale, Spearman ρ gegen "
            "die Zeit (die Hochachsen der Streudiagramme sind auf den Bereich ohne "
            "Ausreisser begrenzt):",
            *[
                f"  {d['merkmal']}: ρ(Zeit) {d['rho']:+.2f} ({C.p_text(d['p'])})"
                for d in drift
            ],
            "Driften Eingang und Ziel gemeinsam, kann ein Modell die Epoche statt den "
            "Prozess lernen.",
            "Methodische Folge, und sie wiegt schwerer als der Prozessbefund: bei "
            "driftendem y ist eine geshuffelte CV optimistisch — Nachbarbillets landen "
            "in Train UND Validierung, und das Modell kann die Epoche als Stellvertreter "
            "nutzen.",
            "Für die Arbeit: neben der gruppenweisen auch eine zeitliche Auswertung "
            "berichten (früh trainieren, spät testen). Das ist die Zahl, die der Frage "
            "'hilft das im Betrieb' entspricht.",
        ],
    )

    fig, ax = C.seite(pdf, "Entwicklung des Ausschusses über die Zeit", 2, 2)

    ax[0].scatter(Ts.t, Ts.y, s=3, alpha=0.2, color="steelblue")
    reihe = Ts.set_index("t").y
    roll = reihe.rolling("30D").median()
    anzahl = reihe.rolling("30D").count()
    roll[anzahl < 20] = np.nan
    ax[0].plot(roll.index, roll.values, color="crimson", lw=1.6, label="rollender Median (30 Tage)")
    ax[0].set_ylabel("y [%]")
    ax[0].set_title("y je Billet", fontsize=10)
    ax[0].legend(fontsize=8, loc="lower left")
    ax[0].tick_params(axis="x", labelsize=7, rotation=25)

    xs = np.arange(len(Q))
    ax[1].bar(xs, Q.n, color="lightsteelblue")
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels(Q.name, fontsize=7, rotation=30)
    ax[1].set_ylabel("Anzahl Billets")
    ax[1].set_title("Billets je Quartal", fontsize=10)

    ax[2].errorbar(
        xs,
        Q.m,
        yerr=[Q.m - Q.lo, Q.hi - Q.m],
        fmt="o-",
        color="steelblue",
        capsize=3,
    )
    ax[2].set_xticks(xs)
    ax[2].set_xticklabels(Q.name, fontsize=7, rotation=30)
    ax[2].set_ylabel("Median y [%]")
    ax[2].set_title("Quartalsmediane mit 95-%-Bootstrap-Intervall", fontsize=10)
    ax[2].grid(alpha=0.3)

    wt = T.t.dt.dayofweek
    daten = [T.y[wt == d].to_numpy() for d in range(7)]
    daten = [d if len(d) else np.array([np.nan]) for d in daten]
    bp = ax[3].boxplot(daten, showfliers=False, patch_artist=True)
    for k in bp["boxes"]:
        k.set_facecolor("lightsteelblue")
    ax[3].set_xticklabels(["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"], fontsize=8)
    ax[3].set_ylabel("y [%]")
    ax[3].set_title("y nach Wochentag des Schmelzendes", fontsize=10)
    ax[3].grid(alpha=0.3, axis="y")
    C.schliesse(pdf, fig, fuss)

    fig, ax = C.seite(pdf, "Driften auch die Eingangsgrössen?", 3, 4)
    t_lo, t_hi = T.t.min(), T.t.max()
    for a, d in zip(ax, drift):
        s = d["s"]
        a.scatter(T.t.loc[s.index], s, s=3, alpha=0.25, color="darkorange")
        a.set_ylim(*C.robuste_grenzen(s))
        a.set_xlim(t_lo, t_hi)
        a.set_title(f"{d['merkmal'][:28]}", fontsize=7.5)
        a.tick_params(labelsize=6)
        a.tick_params(axis="x", rotation=30)
    for a in ax[len(drift) :]:
        C.leer(a)
    C.schliesse(pdf, fig, fuss)

    mix_seiten(pdf, T, df, fuss)

    pdf.close()
    print(f"geschrieben: {C.OUT_DIR / '05_zeit.pdf'}")
    print(
        f"  Spearman(Zeit,y) = {rho:+.3f} (p={p_rho:.2g}) | "
        f"Mann-Kendall z={z_mk:+.2f} (p={p_mk:.2g})"
    )


if __name__ == "__main__":
    main()
