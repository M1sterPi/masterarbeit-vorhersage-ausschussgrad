"""08 — Beitrag gegen Nutzen.

In der Arbeit: nicht berichtet; verlustbasierte Gegenprobe zu den SHAP-Werten
je Merkmal.

Verlustbasierte Wichtigkeit (Permutation, Weglassen) gegen SHAP.
Ausgaben unter reports/xai/, runs/xai/ und reports/xai/tabellen/.

    TEST_LEVEL=<billet|auftrag|artikel> python xai/08_nutzen.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

import _common as C

CV_SEED = 42
TOP_PERM = 30
TOP_LOCO = 10
N_WDH = 5
N_KONTROLLE = 15
KONTROLLE_AB_RANG = 100
FARBE_HERKUNFT = {"Stammdaten": "#3182bd", "Prozess": "#e6550d"}
FARBE_KONTROLLE = "#999999"


def cv_ebene() -> str:
    return "random" if C.D.TEST_LEVEL == "billet" else C.D.TEST_LEVEL


def shap_rangliste() -> pd.DataFrame:
    """Kreuzvalidierte |SHAP|-Rangliste aus 07 — dieselben Falten wie hier."""
    p = C.TAB_DIR / C.mit_ebene("07_kreuzvalidierung_xgboost.csv")
    if not p.exists():
        raise SystemExit(
            f"{p.name} fehlt. Erst 07_kreuzvalidierung.py fuer TEST_LEVEL="
            f"{C.D.TEST_LEVEL} laufen lassen — 08 vergleicht gegen dessen Rangliste."
        )
    return pd.read_csv(p)


def kandidaten(rang: pd.DataFrame, seed=C.SEED):
    """Top-Merkmale plus Kontrollgruppe von weit hinten."""
    top = rang.Merkmal.head(TOP_PERM).tolist()
    hinten = rang.Merkmal.iloc[KONTROLLE_AB_RANG:].tolist()
    rng = np.random.default_rng(seed)
    kontrolle = (
        rng.choice(hinten, size=min(N_KONTROLLE, len(hinten)), replace=False).tolist()
        if hinten
        else []
    )
    return top, kontrolle


def permutation(modell, Xva, yva, merkmale, seed) -> dict[str, float]:
    """MAE-Anstieg beim Durchmischen einer Spalte, gemittelt ueber N_WDH Zuege."""
    rng = np.random.default_rng(seed)
    basis = mean_absolute_error(yva, modell.predict(Xva))
    aus = {}
    for m in merkmale:
        if m not in Xva.columns:
            continue
        spalte = Xva[m].to_numpy(copy=True)
        werte = []
        for _ in range(N_WDH):
            X2 = Xva.copy()
            X2[m] = rng.permutation(spalte)
            werte.append(mean_absolute_error(yva, modell.predict(X2)) - basis)
        aus[m] = float(np.mean(werte))
    return aus


def falten_rechnen(ds, falten, top, kontrolle):
    """Je Falte: Permutation fuer alle Kandidaten, LOCO fuer die Top 10."""
    import xgboost as xgb

    params, quelle = C.xgb_parameter()
    _, fix = C._konstanten_aus_skript()
    X_cv = ds.X_raw.reset_index(drop=True)
    y_cv = ds.y.reset_index(drop=True)

    perm, loco, guete = [], [], []
    for k, (tr, va) in enumerate(falten, 1):
        Xtr, Xva, _ = C.D.process_features(X_cv.iloc[tr], X_cv.iloc[va])
        ytr, yva = y_cv.iloc[tr], y_cv.iloc[va]
        m = xgb.XGBRegressor(**params, **fix, eval_metric="rmse")
        m.fit(Xtr, ytr)
        mae = mean_absolute_error(yva, m.predict(Xva))
        mae_basis = mean_absolute_error(yva, np.full(len(yva), ytr.mean()))
        guete.append({"falte": k, "mae": float(mae), "mae_baseline": float(mae_basis)})
        print(f"  Falte {k}: MAE {mae:.3f} (Baseline Mittelwert {mae_basis:.3f})", flush=True)

        perm.append(permutation(m, Xva, yva, top + kontrolle, CV_SEED + k))

        werte = {}
        for j, f in enumerate(top[:TOP_LOCO], 1):
            if f not in Xtr.columns:
                continue
            mo = xgb.XGBRegressor(**params, **fix, eval_metric="rmse")
            mo.fit(Xtr.drop(columns=[f]), ytr)
            werte[f] = float(
                mean_absolute_error(yva, mo.predict(Xva.drop(columns=[f]))) - mae
            )
            print(f"    LOCO {j:>2}/{TOP_LOCO} {f[:30]:<32}ΔMAE {werte[f]:+.4f}", flush=True)
        loco.append(werte)
    return perm, loco, guete, quelle


def zusammenfassen(werte: list[dict], merkmale) -> pd.DataFrame:
    """Je Merkmal Mittel, Minimum und Maximum ueber die Falten."""
    M = pd.DataFrame(werte).reindex(columns=merkmale)
    return pd.DataFrame(
        {
            "Merkmal": merkmale,
            "delta_mae": M.mean(axis=0).values,
            "min_falten": M.min(axis=0).values,
            "max_falten": M.max(axis=0).values,
            "n_falten_positiv": (M > 0).sum(axis=0).values,
        }
    )


def feld_a(ax, P, decke, n_falten):
    t = P.sort_values("delta_mae", ascending=False).head(TOP_PERM).iloc[::-1]
    y = np.arange(len(t))
    ax.barh(
        y,
        t.delta_mae,
        color=[FARBE_HERKUNFT[C.herkunft(m)] for m in t.Merkmal],
        alpha=0.85,
    )
    ax.errorbar(
        t.delta_mae,
        y,
        xerr=[
            np.maximum(t.delta_mae - t.min_falten, 0),
            np.maximum(t.max_falten - t.delta_mae, 0),
        ],
        fmt="none",
        ecolor="black",
        lw=0.8,
        capsize=2,
    )
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline(decke, color="#d62728", ls="--", lw=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels([m[:32] for m in t.Merkmal], fontsize=6.5)
    ax.set_xlabel("ΔMAE bei Permutation [PP], Mittel über die Falten")
    ax.set_title(
        f"A  Verlustbasierte Wichtigkeit — rot gestrichelt = Rauschdecke der "
        f"Kontrollgruppe ({decke:+.3f})",
        fontsize=9.5,
    )
    ax.grid(alpha=0.3, axis="x")


def feld_b(ax, P, rang, decke):
    t = P.merge(rang[["Merkmal", "Mittel"]], on="Merkmal", how="left").dropna(
        subset=["Mittel"]
    )
    t = t[t.Mittel > 0]
    rho = spearmanr(t.Mittel, t.delta_mae)[0] if len(t) > 2 else float("nan")
    ax.scatter(
        t.Mittel,
        t.delta_mae,
        s=22,
        c=[FARBE_HERKUNFT[C.herkunft(m)] for m in t.Merkmal],
        alpha=0.85,
        zorder=3,
    )
    for _, r in t.nlargest(6, "Mittel").iterrows():
        ax.annotate(
            r.Merkmal[:22],
            (r.Mittel, r.delta_mae),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=6,
        )
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(decke, color="#d62728", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlim(right=float(t.Mittel.max()) * 1.9)
    ax.set_xlabel("mittlerer |SHAP| [PP] (aus 07, log)")
    ax.set_ylabel("ΔMAE bei Permutation [PP]")
    ax.set_title(
        f"B  Beitrag gegen Nutzen — Spearman ρ = {rho:+.3f}\n"
        f"rechts unten: viel benutzt, kauft keine Genauigkeit",
        fontsize=9.5,
    )
    ax.grid(alpha=0.3)
    return float(rho), t


def feld_c(ax, P, L):
    t = L.merge(
        P[["Merkmal", "delta_mae"]].rename(columns={"delta_mae": "perm"}),
        on="Merkmal",
        how="left",
    ).sort_values("perm", ascending=False)
    y = np.arange(len(t))[::-1]
    ax.barh(y + 0.19, t.perm, height=0.36, color="#9ecae1", label="Permutation")
    ax.barh(y - 0.19, t.delta_mae, height=0.36, color="#3182bd", label="LOCO (neu trainiert)")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([m[:30] for m in t.Merkmal], fontsize=6.5)
    ax.set_xlabel("ΔMAE [PP]")
    ax.set_title(
        f"C  Ersetzbarkeit: Top {TOP_LOCO} — LOCO deutlich kleiner als Permutation "
        f"heisst, das Modell findet Ersatz",
        fontsize=9.5,
    )
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3, axis="x")
    return t


def feld_d(ax, P, K, guete):
    daten = [P.delta_mae.values, K.delta_mae.values]
    bp = ax.boxplot(daten, tick_labels=[f"Top {TOP_PERM}", f"Kontrolle (n={len(K)})"], widths=0.5)
    for med in bp["medians"]:
        med.set_color("black")
    ax.axhline(0, color="black", lw=0.8)
    mae = np.mean([g["mae"] for g in guete])
    basis = np.mean([g["mae_baseline"] for g in guete])
    spanne = basis - mae
    ax.axhline(spanne, color="#31a354", ls="--", lw=1.2)
    ax.annotate(
        f"gesamter Vorsprung des Modells\nvor der Mittelwert-Baseline: {spanne:.2f} PP",
        (1.5, spanne),
        xytext=(0, -22),
        textcoords="offset points",
        ha="center",
        fontsize=7,
        color="#31a354",
    )
    ax.set_ylabel("ΔMAE [PP]")
    ax.set_title(
        "D  Einordnung — wie gross ist ein ΔMAE überhaupt?", fontsize=9.5
    )
    ax.grid(alpha=0.3, axis="y")
    return float(mae), float(basis)


def main():
    print(f"Ebene: TEST_LEVEL={C.D.TEST_LEVEL}  →  CV-Ebene {cv_ebene()}")
    rang = shap_rangliste()
    top, kontrolle = kandidaten(rang)
    print(f"  {len(top)} Top-Merkmale, {len(kontrolle)} Kontrollmerkmale "
          f"(ab Rang {KONTROLLE_AB_RANG})")

    ds = C.D.load_split(verbose=False)
    falten, ebene = C.D.make_cv(ds.df, level=cv_ebene(), seed=CV_SEED)
    perm, loco, guete, quelle = falten_rechnen(ds, falten, top, kontrolle)

    P = zusammenfassen(perm, top)
    K = zusammenfassen(perm, kontrolle)
    L = zusammenfassen(loco, top[:TOP_LOCO]).dropna(subset=["delta_mae"])
    decke = float(np.nanmax(K.delta_mae)) if len(K) else 0.0
    ueber = P[P.delta_mae > decke]

    fuss = (
        f"{C.D.CSV_PATH.name} | {len(ds.df)} Billets, {len(falten)} Falten, CV-Ebene "
        f"{ebene} | y ∈ [{C.D.Y_MIN:g}; {C.D.Y_MAX:g}] | Test-Ebene {C.D.TEST_LEVEL} "
        f"| {N_WDH} Permutationen je Merkmal und Falte | XGBoost {quelle}"
    )

    pdf = C.pdf_oeffnen(C.mit_ebene("08_nutzen.pdf"))
    fig, ax = C.seite(
        f"Beitrag oder Nutzen? Verlustbasierte Wichtigkeit — Ebene {C.D.TEST_LEVEL}",
        2,
        2,
    )
    feld_a(ax[0], P, decke, len(falten))
    rho, punkte = feld_b(ax[1], P, rang, decke)
    tc = feld_c(ax[2], P, L)
    mae, basis = feld_d(ax[3], P, K, guete)
    C.schliesse(pdf, fig, fuss)

    ersetzbar = [
        r.Merkmal
        for _, r in tc.iterrows()
        if np.isfinite(r.perm) and r.perm > decke and r.delta_mae < decke
    ]
    C.textseite(
        pdf,
        "Lesehinweise und Vorbehalte",
        [
            f"Vorgehen. Beide Masse werden AUSSERHALB der Trainingsfalte gerechnet, "
            f"auf denselben {len(falten)} Falten wie 07. Permutation: die Spalte in "
            f"der Validierungsfalte {N_WDH}-mal durchmischen und den MAE-Anstieg "
            f"mitteln. LOCO: die Spalte entfernen und das Modell auf den "
            f"Trainingsfalten neu anpassen. Kandidaten sind die Top {TOP_PERM} der "
            f"kreuzvalidierten SHAP-Rangliste aus 07, LOCO nur fuer die Top "
            f"{TOP_LOCO}.",
            f"Rauschdecke. {len(K)} zufaellig gezogene Merkmale ab Rang "
            f"{KONTROLLE_AB_RANG} bilden die Kontrollgruppe; ihr wahrer Beitrag ist "
            f"nahe null. Gemessen: Median {np.median(K.delta_mae):+.4f}, Maximum "
            f"{decke:+.4f} PP. Nur was darueber liegt, ist als Nutzen lesbar — das "
            f"trifft auf {len(ueber)} der {len(P)} geprueften Merkmale zu.",
            f"Kernbefund. Die Rangkorrelation zwischen mittlerem |SHAP| und ΔMAE "
            f"betraegt ρ = {rho:+.3f}. Beide Masse beantworten also verschiedene "
            f"Fragen: SHAP misst, wie stark ein Merkmal die Vorhersage bewegt, ΔMAE "
            f"misst, ob es die Vorhersage besser macht. Ein Merkmal mit hohem |SHAP| "
            f"und ΔMAE unter der Rauschdecke bewegt das Modell, ohne Information "
            f"beizusteuern, die nicht auch anderswo steht.",
            f"Ersetzbarkeit. {len(ersetzbar)} der Top {TOP_LOCO} verlieren beim "
            f"Neutraining ihren Effekt (Permutation ueber, LOCO unter der "
            f"Rauschdecke): {', '.join(ersetzbar[:6]) if ersetzbar else '—'}. Das ist "
            f"kein Widerspruch, sondern die erwartete Folge korrelierter Merkmale — "
            f"das Modell findet ohne das Merkmal einen Stellvertreter. Welche "
            f"Merkmale sich gegenseitig vertreten koennen, zeigt 09_redundanz.py.",
            f"Massstab. Das Modell liegt im Mittel der Falten bei MAE {mae:.2f} PP, "
            f"die Mittelwert-Baseline bei {basis:.2f} PP; der gesamte Vorsprung "
            f"betraegt also {basis - mae:.2f} PP. Jedes einzelne ΔMAE ist an dieser "
            f"Spanne zu messen, nicht an der Zielgroesse.",
            "Vorbehalt. Die Permutation zerstoert die Abhaengigkeit zwischen dem "
            "Merkmal und allen uebrigen und wertet das Modell damit an "
            "Merkmalskombinationen aus, die so nicht vorkommen. Bei stark "
            "korrelierten Merkmalen ist der gemessene ΔMAE deshalb eher eine obere "
            "Schranke. LOCO hat dieses Problem nicht, misst dafuer eine andere "
            "Groesse: nicht den Beitrag DIESES Modells, sondern den Verlust an "
            "erreichbarer Guete.",
        ],
    )
    pdf.close()

    tab = pd.concat(
        [P.assign(gruppe="top"), K.assign(gruppe="kontrolle")], ignore_index=True
    ).merge(rang[["Merkmal", "Mittel"]].rename(columns={"Mittel": "shap_mittel"}),
            on="Merkmal", how="left")
    tab = tab.merge(
        L[["Merkmal", "delta_mae"]].rename(columns={"delta_mae": "loco_delta_mae"}),
        on="Merkmal",
        how="left",
    )
    tab["ueber_rauschdecke"] = tab.delta_mae > decke
    pcsv = C.speichere_rangliste(C.mit_ebene("08_nutzen.csv"), tab, "delta_mae")

    p = C.speichere(
        C.mit_ebene("08_nutzen.json"),
        {
            "test_level": C.D.TEST_LEVEL,
            "cv_level": ebene,
            "n_falten": len(falten),
            "n_wiederholungen": N_WDH,
            "params_quelle": quelle,
            "guete_falten": guete,
            "mae_mittel": mae,
            "mae_baseline_mittel": basis,
            "rauschdecke": decke,
            "rauschdecke_median": float(np.median(K.delta_mae)) if len(K) else None,
            "kontrollgruppe": K.to_dict("records"),
            "spearman_shap_vs_delta_mae": rho,
            "n_ueber_rauschdecke": int(len(ueber)),
            "permutation": P.to_dict("records"),
            "loco": L.to_dict("records"),
            "ersetzbar": ersetzbar,
            "rangliste_csv": pcsv.name,
        },
    )
    print(f"\ngeschrieben: {C.OUT_DIR / C.mit_ebene('08_nutzen.pdf')}\n             {p}"
          f"\n             {pcsv}\n")
    print(f"ρ(|SHAP|, ΔMAE) = {rho:+.3f} | Rauschdecke {decke:+.4f} PP | "
          f"{len(ueber)}/{len(P)} darueber | ersetzbar: {len(ersetzbar)}/{len(L)}")
    for _, r in P.sort_values("delta_mae", ascending=False).head(10).iterrows():
        print(f"  {r.Merkmal:<36}{r.delta_mae:>+9.4f}")


if __name__ == "__main__":
    main()
