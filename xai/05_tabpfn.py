"""05 — SHAP für TabPFN.

In der Arbeit: Abschnitt 7.2: Shapley-Werte von TabPFN auf der Stichprobe der
Testmenge.

Shapley-Werte über shapiq für eine Stichprobe des Test-Sets; setzt nach einem
Abbruch fort. Option --nur-plot. GPU empfohlen.

    python xai/05_tabpfn.py
"""

from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
from scipy.stats import spearmanr

import _common as C

BUDGET = 4096
N_STICHPROBE = 125

N_ZWEITZIEHUNG = 4
SEED_ZIEHUNG_1 = C.SEED
SEED_ZIEHUNG_2 = 4242

ERG = C.ERG_DIR / "05_tabpfn.json"
FARBE = {"Stammdaten": "#3182bd", "Prozess": "#e6550d"}


def modellparameter() -> dict:
    """Parameter wie in tabpfn_regression.py, plus den Cache."""
    default, fix = C._konstanten_aus_skript("tabpfn_regression.py")
    return {**default, **fix, "fit_mode": "fit_with_cache"}


def basisvektor(X_tr) -> np.ndarray:
    """Spaltenweiser Median als Bezugspunkt — siehe Kopfkommentar."""
    A = X_tr.to_numpy(dtype=float)
    b = np.nanmedian(A, axis=0)
    leer = int(np.isnan(b).sum())
    if leer:
        print(f"  Hinweis: {leer} Spalten sind vollstaendig leer, "
              f"ihr Basiswert bleibt NaN")
    return b


def lade_stand() -> dict:
    leer = {
        "budget": BUDGET,
        "fit_mode": "fit_with_cache",
        "basis": "nanmedian",
        "beitraege": {},
        "sekunden": {},
        "zweitziehung": {},
    }
    if not ERG.exists():
        return leer
    d = json.loads(ERG.read_text(encoding="utf-8"))
    passt = (
        d.get("budget") == BUDGET
        and d.get("fit_mode") == "fit_with_cache"
        and d.get("basis") == "nanmedian"
    )
    if passt:
        d.setdefault("zweitziehung", {})
        return d
    print(
        f"  Zwischenstand passt nicht zur aktuellen Einstellung (budget "
        f"{d.get('budget')} statt {BUDGET}, fit_mode {d.get('fit_mode')}, basis "
        f"{d.get('basis')}) — wird verworfen. Der alte Lauf liegt als "
        f"05_tabpfn_alt_budget2048_493merkmale.json daneben."
    )
    return leer


def waehle_billets(n_test: int) -> list[int]:
    """Zufallsstichprobe plus die vier Faelle aus 02_lokal — so werden die lokalen
    Erklaerungen zwischen TreeSHAP und shapiq direkt vergleichbar."""
    rng = np.random.default_rng(C.SEED)
    idx = set(
        rng.choice(n_test, size=min(N_STICHPROBE, n_test), replace=False).tolist()
    )
    p = C.ERG_DIR / C.mit_ebene("02_lokal.json")
    if p.exists():
        idx |= {f["index"] for f in json.loads(p.read_text(encoding="utf-8"))["faelle"]}
    return sorted(idx)


def _erklaerer(modell, X_tr, basis, seed):
    from tabpfn_extensions.interpretability.shapiq import (
        get_tabpfn_imputation_explainer,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expl = get_tabpfn_imputation_explainer(
            model=modell,
            data=X_tr,
            index="SV",
            max_order=1,
            imputer="baseline",
            random_state=seed,
        )
        expl.imputer.init_background(basis)
        expl.imputer.calc_empty_prediction()
    return expl


def _erklaere(expl, X_te, i) -> list[float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sv = expl.explain(X_te.iloc[i : i + 1].values, budget=BUDGET)
    v = sv.get_n_order_values(1) if hasattr(sv, "get_n_order_values") else sv.values
    return [round(float(x), 6) for x in np.asarray(v).ravel()]


def rechne(ds, X_tr, X_te):
    from tabpfn import TabPFNRegressor

    stand = lade_stand()
    ziel = waehle_billets(len(X_te))
    zweit_ziel = ziel[:N_ZWEITZIEHUNG]
    offen = [i for i in ziel if str(i) not in stand["beitraege"]]
    offen_zweit = [i for i in zweit_ziel if str(i) not in stand["zweitziehung"]]
    print(
        f"{len(ziel)} Billets geplant, {len(ziel)-len(offen)} bereits gerechnet, "
        f"{len(offen)} offen; Zweitziehung {len(offen_zweit)} offen"
    )
    if not offen and not offen_zweit:
        return stand

    params = modellparameter()
    print(f"  Modell: {params}")
    t0 = time.time()
    modell = TabPFNRegressor(**params)
    modell.fit(X_tr, ds.y_train)
    print(f"  Fit mit KV-Cache: {time.time()-t0:.1f} s (einmalig)")

    basis = basisvektor(X_tr)
    t0 = time.time()
    expl1 = _erklaerer(modell, X_tr, basis, SEED_ZIEHUNG_1)
    print(
        f"  Erklaerer: {time.time()-t0:.1f} s, "
        f"Leervorhersage f(Basis) = {expl1.imputer.empty_prediction:.2f}\n"
    )

    def sichere():
        C.speichere(
            ERG.name,
            {
                **stand,
                "spalten": list(X_te.columns),
                "n_estimators": params.get("n_estimators"),
                "leervorhersage": float(expl1.imputer.empty_prediction),
            },
        )

    if offen_zweit:
        expl2 = _erklaerer(modell, X_tr, basis, SEED_ZIEHUNG_2)
        for k, i in enumerate(offen_zweit, 1):
            t0 = time.time()
            stand["zweitziehung"][str(i)] = _erklaere(expl2, X_te, i)
            sichere()
            print(
                f"  Zweitziehung {k}/{len(offen_zweit)}  Billet {i:>4}  "
                f"{time.time()-t0:6.1f} s",
                flush=True,
            )
        del expl2

    for k, i in enumerate(offen, 1):
        t0 = time.time()
        stand["beitraege"][str(i)] = _erklaere(expl1, X_te, i)
        dt = time.time() - t0
        if k == 1:
            phi = np.array(stand["beitraege"][str(i)])
            vorher = float(
                modell.predict(X_te.iloc[i : i + 1].to_numpy(dtype=float))[0]
            )
            ab = abs(float(phi.sum()) + expl1.imputer.empty_prediction - vorher)
            print(f"  Additivitaet: |Summe + Basis − Vorhersage| = {ab:.3f} pp")
            if ab > 1.0:
                raise AssertionError(
                    f"Additivitaet verletzt ({ab:.3f} pp) — die Shapley-Werte "
                    "gehoeren nicht zu diesem Modell."
                )
        stand["sekunden"][str(i)] = round(dt, 1)
        sichere()
        rest = (len(offen) - k) * dt / 3600
        print(
            f"  {k:>3}/{len(offen)}  Billet {i:>4}  {dt:6.1f} s  noch ~{rest:5.2f} h",
            flush=True,
        )
    return stand


def _reproduzierbarkeit(stand) -> tuple[list, float, float]:
    """Erst- gegen Zweitziehung: (Paare, mittleres rho, mittlere Top-20-Deckung)."""
    paare, rhos, deck = [], [], []
    for k, zweit in stand.get("zweitziehung", {}).items():
        if k not in stand["beitraege"]:
            continue
        a = np.abs(np.array(stand["beitraege"][k]))
        b = np.abs(np.array(zweit))
        paare.append((a, b))
        rhos.append(spearmanr(a, b).statistic)
        deck.append(len(set(np.argsort(-a)[:20]) & set(np.argsort(-b)[:20])))
    if not paare:
        return [], float("nan"), float("nan")
    return paare, float(np.mean(rhos)), float(np.mean(deck))


def plotte(ds, X_tr, X_te, stand):
    sp = stand.get("spalten") or list(X_te.columns)
    B = np.array([stand["beitraege"][k] for k in sorted(stand["beitraege"], key=int)])
    if B.size == 0:
        print("  noch keine Ergebnisse — nichts zu plotten")
        return
    w = C.globale_wichtigkeit(B, sp)
    n = B.shape[0]
    paare, rho_wied, deck_wied = _reproduzierbarkeit(stand)
    v = w.mittlerer_betrag.values
    med = float(np.median(v))
    q05, q95 = float(np.quantile(v, 0.05)), float(np.quantile(v, 0.95))
    ueber = int((v > 1.5 * med).sum())

    pdf = C.pdf_oeffnen("05_tabpfn.pdf")
    fig, ax = C.seite(
        f"TabPFN-Erklärung über shapiq  ({n} Billets, budget = {BUDGET}, "
        f"KV-Cache, Basis = Spaltenmedian)",
        2,
        3,
    )

    t = w.head(20).iloc[::-1]
    ax[0].barh(
        range(len(t)),
        t.mittlerer_betrag,
        color=[FARBE[C.herkunft(m)] for m in t.Merkmal],
    )
    ax[0].set_yticks(range(len(t)))
    ax[0].set_yticklabels([m[:30] for m in t.Merkmal], fontsize=6.5)
    ax[0].set_xlabel("mittlerer |Shapley| [Prozentpunkte]")
    ax[0].set_title("A  Die 20 wichtigsten Merkmale (TabPFN)", fontsize=10)
    ax[0].grid(alpha=0.3, axis="x")

    p = C.ERG_DIR / C.mit_ebene("01_global.json")
    rho = float("nan")
    if p.exists():
        tree = {
            r["Merkmal"]: r["mittlerer_betrag"]
            for r in json.loads(p.read_text(encoding="utf-8"))["top50"]
        }
        gem = [m for m in w.Merkmal if m in tree]
        a = w.set_index("Merkmal").loc[gem, "mittlerer_betrag"].to_numpy()
        b = np.array([tree[m] for m in gem])
        if len(gem) > 3:
            rho = float(spearmanr(a, b).statistic)
            ax[1].scatter(b, a, s=22, color="#7b3294")
            for m, x, yv in zip(gem, b, a):
                ax[1].annotate(
                    m[:16],
                    (x, yv),
                    fontsize=5,
                    textcoords="offset points",
                    xytext=(4, 3),
                )
    ax[1].set_xlabel("mittlerer |SHAP| — XGBoost (TreeSHAP, exakt)")
    ax[1].set_ylabel("mittlerer |Shapley| — TabPFN (shapiq)")
    ax[1].set_title(
        f"B  Stimmen die Verfahren überein?  ρ = {rho:+.3f}"
        if np.isfinite(rho)
        else "B  Vergleich (01_global.json fehlt)",
        fontsize=10,
    )
    ax[1].grid(alpha=0.3)

    k = np.cumsum(w.mittlerer_betrag.values)
    k = k / k[-1]
    ax[2].plot(np.arange(1, len(k) + 1), k, color="black", lw=1.6)
    for a_ in (0.5, 0.8):
        nn = int(np.searchsorted(k, a_) + 1)
        ax[2].axhline(a_, color="#999999", ls=":", lw=1)
        ax[2].annotate(
            f"{a_:.0%} bei {nn}",
            (nn, a_),
            fontsize=8,
            textcoords="offset points",
            xytext=(6, -12),
            color="#d62728",
        )
    ax[2].set_xscale("log")
    ax[2].set_xlabel("Merkmale, nach Wichtigkeit sortiert (log)")
    ax[2].set_ylabel("kumulierter Anteil")
    ax[2].set_title("C  Konzentration der Wirkung", fontsize=10)
    ax[2].grid(alpha=0.3)

    ax[3].hist(v, bins=60, color="#969696")
    hoch = ax[3].get_ylim()[1]
    for f, farbe, rel in ((1.5, "#d62728", 0.92), (2.0, "#7b3294", 0.78)):
        ax[3].axvline(f * med, color=farbe, ls="--", lw=1.2)
        ax[3].annotate(
            f"{f:g}× Median: {(v > f*med).sum()} Merkmale",
            (f * med, hoch * rel),
            fontsize=7,
            color=farbe,
            textcoords="offset points",
            xytext=(5, 0),
        )
    ax[3].set_xlabel("mittlerer |Shapley| je Merkmal")
    ax[3].set_ylabel("Merkmale")
    ax[3].set_title(
        f"D  Rauschboden  (Median {med:.3f}, 5–95 % {q05:.3f}–{q95:.3f})",
        fontsize=10,
    )
    ax[3].grid(alpha=0.3, axis="y")

    if paare:
        for a_, b_ in paare:
            ax[4].scatter(a_, b_, s=6, alpha=0.35, color="#7b3294")
        lim = max(max(a_.max(), b_.max()) for a_, b_ in paare)
        ax[4].plot([0, lim], [0, lim], color="black", lw=0.8, ls=":")
        ax[4].set_xlabel("|Shapley| — Ziehung 1")
        ax[4].set_ylabel("|Shapley| — Ziehung 2")
        ax[4].set_title(
            f"E  Zwei unabhängige Ziehungen  ρ = {rho_wied:+.3f}, "
            f"Top-20 {deck_wied:.1f}/20",
            fontsize=10,
        )
    else:
        ax[4].text(
            0.5, 0.5, "keine Zweitziehung im Zwischenstand",
            ha="center", va="center", fontsize=9,
        )
        ax[4].set_title("E  Reproduzierbarkeit", fontsize=10)
    ax[4].grid(alpha=0.3)

    sek = np.array(list(stand["sekunden"].values()), dtype=float)
    if sek.size == 0:
        sek = np.array([float("nan")])
    ax[5].hist(sek[np.isfinite(sek)], bins=20, color="#e6550d")
    ax[5].set_xlabel("Sekunden je Billet")
    ax[5].set_ylabel("Billets")
    ax[5].set_title(
        f"F  Rechenaufwand  (Median {np.nanmedian(sek):.0f} s, "
        f"gesamt {np.nansum(sek)/3600:.1f} h)",
        fontsize=10,
    )
    ax[5].grid(alpha=0.3, axis="y")

    leerv = stand.get("leervorhersage", float("nan"))
    C.schliesse(
        pdf,
        fig,
        f"{C.D.CSV_PATH.name} | TabPFN n_estimators="
        f"{stand.get('n_estimators','?')}, Kontext {len(ds.y_train)} Zeilen, "
        f"fit_with_cache | shapiq index=SV, max_order=1, imputer=baseline "
        f"(Spaltenmedian), f(Basis) = {leerv:.2f}",
    )

    C.textseite(
        pdf,
        "Lesehinweise und Vorbehalte",
        [
            "Verfahren. shapiq mit imputationsbasiertem Erklärer, wie in der "
            "Prior-Labs-Dokumentation vorgesehen. Anders als das TreeSHAP in 01 "
            f"bis 03 sind diese Werte APPROXIMIERT: exakte Shapley-Werte "
            f"erforderten 2^{len(sp)} Koalitionen.",
            f"Budget: {BUDGET} Koalitionen je Billet.",
            (
                f"Reproduzierbarkeit. Für {len(paare)} Billets wurde eine zweite, "
                f"unabhängige Ziehung gerechnet: ρ = {rho_wied:+.3f} über alle "
                f"{len(sp)} Merkmale, {deck_wied:.1f} von 20 Spitzenmerkmalen "
                f"gemeinsam. Die Rangfolge über ALLE Merkmale ist damit nicht "
                f"berichtbar, die Spitzengruppe schon."
                if paare
                else "Reproduzierbarkeit. Noch keine Zweitziehung gerechnet."
            ),
            f"Rauschboden. Der mittlere |Shapley| liegt für 90 % der Merkmale "
            f"zwischen {q05:.2f} und {q95:.2f} Prozentpunkten. Über dem Boden "
            f"stehen {ueber} Merkmale (mehr als 1,5× Median). Alles darunter ist "
            f"von Koalitionsrauschen nicht zu unterscheiden — die Rangplätze im "
            f"Mittelfeld sind bedeutungslos.",
            f"Bezugspunkt. Der Basisvektor ist der spaltenweise Median; f(Basis) = "
            f"{leerv:.2f}. Der Vorgabe-Basisvektor von shapiq (Spaltenmittel) ist bei "
            f"leeren Spalten NaN und deshalb nicht geeignet.",
            f"Rechenaufwand. {np.nanmedian(sek):.0f} s je Billet im Median (mit "
            f"KV-Cache, der den Trainingskontext nur einmal kodiert).",
            "Vergleich mit XGBoost. Rangkorrelation der gemeinsamen Merkmale: "
            + (f"ρ = {rho:+.3f}. " if np.isfinite(rho) else "nicht verfügbar. ")
            + "Übereinstimmung spräche dafür, dass die Rangfolge in den Daten "
            "liegt und nicht im Modell; Abweichung wäre kein Fehler, sondern ein "
            "Hinweis darauf, dass die beiden Modellklassen verschiedene Merkmale "
            "nutzen.",
        ],
    )
    pdf.close()

    tab = w.copy()
    tab["konstant_im_train"] = [X_tr[m].nunique(dropna=False) <= 1 for m in tab.Merkmal]
    decke = tab.loc[tab.konstant_im_train, "mittlerer_betrag"].max()
    tab["ueber_rauschdecke"] = tab.mittlerer_betrag > decke
    pcsv = C.speichere_rangliste(C.mit_ebene("05_tabpfn.csv"), tab)
    print(f"\ngeschrieben: {C.OUT_DIR / '05_tabpfn.pdf'}\n             {pcsv}")
    print(f"\nTop 12 (TabPFN, {n} Billets):")
    for i, r in w.head(12).iterrows():
        print(f"  {i+1:>2}. {r.Merkmal:<40}{r.mittlerer_betrag:>9.4f}")
    if paare:
        print(
            f"\nReproduzierbarkeit ({len(paare)} Billets, zwei Ziehungen): "
            f"rho {rho_wied:+.3f}, Top-20 {deck_wied:.1f}/20"
        )
    if np.isfinite(rho):
        print(f"Rangkorrelation zu XGBoost-TreeSHAP: {rho:+.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nur-plot", action="store_true")
    args = ap.parse_args()

    ds, X_tr, X_te = C.lade()
    stand = lade_stand() if args.nur_plot else rechne(ds, X_tr, X_te)
    plotte(ds, X_tr, X_te, stand)


if __name__ == "__main__":
    main()
