"""13 — Nutzen je Prozesskanal und Stammdatenblock.

In der Arbeit: Abschnitt 7.2: Nutzen je Kanal und Stammdatenblock mit
Konfidenzintervall (Abbildungen 18 und 19, Tabelle 14) und die Summe der
Nutzenwerte.

Gruppenpermutation mit Konfidenzintervall über Artikel.
Ausgaben unter reports/xai/, runs/xai/ und reports/xai/tabellen/.

    TEST_LEVEL=<billet|auftrag|artikel> python xai/13_sensoren.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score

import _common as C
from data_prep import SP  # noqa: E402

CV_SEED = 42
N_WDH = 5
N_KONTROLLE = 15
KONTROLLE_AB_RANG = 100
KONTROLLE_GROESSE = 10
N_BOOT = 1000
OBEN = 0.90

FARBE = {"Kanal": "#e6550d", "Influx-Kopf": "#fdae6b", "Stammdaten": "#3182bd"}


def cv_ebene() -> str:
    return "random" if C.D.TEST_LEVEL == "billet" else C.D.TEST_LEVEL


def name(datei: str) -> str:
    return C.mit_ebene(datei)


def kanalliste() -> list[str]:
    p = C.ROOT / "data" / "cache" / (C.D.CSV_PATH.stem + ".process_cols.json")
    return json.loads(p.read_text(encoding="utf-8"))["process_cols"]


def stammblock(c: str) -> str:
    if c.startswith("prog__") or c in {
        SP["programmnummer"], SP["rezept_a"], SP["rezept_b"], SP["rezept_c"], SP["rezept_d"], "Energie_prog",
    }:
        return "Rezept/Programm"
    if c in {
        SP["wand_soll"], SP["laenge_soll"], SP["aussen_soll"], SP["innen_soll"],
        SP["laenge_erwartet"], SP["geometrie_a"], SP["geometrie_b"], SP["laenge_billet"],
    }:
        return "Sollmasse"
    if c in {"Glastyp", SP["glasvariante"], "Material", SP["kategorie"]}:
        return "Glas/Material"
    if c.startswith(("charge", "posten", SP["rohstoff_praefix"], SP["granulatdichte_praefix"])):
        return "Rohstoff/Charge"
    return "uebrige Stammdaten"


def gruppen(spalten) -> dict[str, tuple[str, list[str]]]:
    """Name -> (Art, Spalten)."""
    influx = set(C.D.influx_spalten(spalten))
    kanaele = sorted(kanalliste(), key=len, reverse=True)
    g: dict[str, tuple[str, list[str]]] = {}
    for c in spalten:
        if c in influx:
            k = next(
                (k for k in kanaele
                 if c.startswith(k) and c[len(k):] in C.D._AGG_SUFFIXES),
                None,
            )
            key, art = (k, "Kanal") if k else ("Influx-Kopf", "Influx-Kopf")
        else:
            key, art = stammblock(c), "Stammdaten"
        g.setdefault(key, (art, []))[1].append(c)
    if SP["kanal_l"] in g and SP["kanal_k"] in g:
        g[SP["kanal_k"]][1].extend(g.pop(SP["kanal_l"])[1])
    return g


def kontrollgruppen(spalten, seed=C.SEED) -> dict[str, list[str]]:
    p = C.TAB_DIR / C.mit_ebene("07_kreuzvalidierung_xgboost.csv")
    if not p.exists():
        raise SystemExit(f"{p.name} fehlt — erst 07_kreuzvalidierung.py laufen lassen.")
    rang = pd.read_csv(p)
    hinten = [m for m in rang.Merkmal.iloc[KONTROLLE_AB_RANG:] if m in spalten]
    rng = np.random.default_rng(seed)
    return {
        f"Kontrolle {i + 1}": rng.choice(hinten, KONTROLLE_GROESSE, replace=False).tolist()
        for i in range(N_KONTROLLE)
    }


def rechnen(ds, falten):
    import xgboost as xgb

    params, quelle = C.xgb_parameter()
    _, fix = C._konstanten_aus_skript()
    X = ds.X_raw.reset_index(drop=True)
    y = ds.y.reset_index(drop=True).to_numpy()

    n = len(y)
    pred = np.full(n, np.nan)
    G = K = None
    perm_pred: dict[str, np.ndarray] = {}
    perm_abs: dict[str, np.ndarray] = {}
    falte_dmae: dict[str, list[float]] = {}
    guete = []
    for f, (tr, va) in enumerate(falten, 1):
        Xtr, Xva, _ = C.D.process_features(X.iloc[tr], X.iloc[va])
        if G is None:
            G = gruppen(list(Xtr.columns))
            K = kontrollgruppen(list(Xtr.columns))
            alle = {**{k: v[1] for k, v in G.items()}, **K}
            for k in alle:
                perm_pred[k] = np.full(n, np.nan)
                perm_abs[k] = np.full(n, np.nan)
                falte_dmae[k] = []
        m = xgb.XGBRegressor(**params, **fix)
        m.fit(Xtr, y[tr])
        p = m.predict(Xva)
        pred[va] = p
        mae = mean_absolute_error(y[va], p)
        mae_med = mean_absolute_error(y[va], np.full(len(va), np.median(y[tr])))
        guete.append({"falte": f, "mae": float(mae), "mae_median": float(mae_med)})
        print(f"  Falte {f}: MAE {mae:.3f} (Median {mae_med:.3f})", flush=True)

        rng = np.random.default_rng(CV_SEED + f)
        for k, cols in alle.items():
            cols = [c for c in cols if c in Xva.columns]
            ps, ab = np.zeros(len(va)), np.zeros(len(va))
            for _ in range(N_WDH):
                X2 = Xva.copy()
                idx = rng.permutation(len(va))
                for c in cols:
                    X2[c] = Xva[c].to_numpy()[idx]
                q = m.predict(X2)
                ps += q / N_WDH
                ab += np.abs(y[va] - q) / N_WDH
            perm_pred[k][va] = ps
            perm_abs[k][va] = ab
            falte_dmae[k].append(float(ab.mean() - mae))
    return pred, perm_pred, perm_abs, falte_dmae, G, K, guete, quelle, y


def bootstrap(d: np.ndarray, cluster: np.ndarray, seed=C.SEED) -> tuple[float, float]:
    """95-%-KI des Mittels von d, gezogen werden ganze Artikel."""
    rng = np.random.default_rng(seed)
    codes, inv = np.unique(cluster, return_inverse=True)
    s = np.bincount(inv, weights=d)
    c = np.bincount(inv)
    werte = []
    for _ in range(N_BOOT):
        z = rng.integers(0, len(codes), len(codes))
        werte.append(s[z].sum() / c[z].sum())
    return float(np.percentile(werte, 2.5)), float(np.percentile(werte, 97.5))


def tabelle(pred, perm_pred, perm_abs, falte_dmae, G, K, y, cluster):
    oben = y >= np.quantile(y, OBEN)
    e0 = np.abs(y - pred)
    auc0 = roc_auc_score(oben, pred)
    zeilen = []
    for k in list(G) + list(K):
        d = perm_abs[k] - e0
        lo, hi = bootstrap(d, cluster)
        art, cols = G[k] if k in G else ("Kontrolle", K[k])
        zeilen.append({
            "Gruppe": k,
            "art": art,
            "n_spalten": len(cols),
            "delta_mae": float(np.mean(falte_dmae[k])),
            "min_falten": float(np.min(falte_dmae[k])),
            "max_falten": float(np.max(falte_dmae[k])),
            "n_falten_positiv": int(np.sum(np.array(falte_dmae[k]) > 0)),
            "ki_lo": lo,
            "ki_hi": hi,
            "delta_auc": float(auc0 - roc_auc_score(oben, perm_pred[k])),
            "spalten": " ".join(cols),
        })
    T = pd.DataFrame(zeilen)
    return T, float(auc0)


def shap_je_gruppe(G) -> pd.DataFrame:
    """Σ mittlerer |SHAP| je Gruppe aus 07 — obere Schranke des Gruppenbetrags."""
    p = C.TAB_DIR / C.mit_ebene("07_kreuzvalidierung_xgboost.csv")
    r = pd.read_csv(p).set_index("Merkmal")["Mittel"]
    return pd.DataFrame(
        {"Gruppe": k, "shap_summe": float(r.reindex(cols).fillna(0).sum())}
        for k, (_, cols) in G.items()
    )


def feld_a(ax, T, decke):
    t = T[T.art != "Kontrolle"].sort_values("delta_mae").tail(25)
    yy = np.arange(len(t))
    ax.barh(yy, t.delta_mae, color=[FARBE[a] for a in t.art], alpha=0.85)
    ax.errorbar(t.delta_mae, yy,
                xerr=[np.maximum(t.delta_mae - t.ki_lo, 0), np.maximum(t.ki_hi - t.delta_mae, 0)],
                fmt="none", ecolor="black", lw=0.8, capsize=2)
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline(decke, color="#d62728", ls="--", lw=1.2)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{g} ({n})" for g, n in zip(t.Gruppe, t.n_spalten)], fontsize=7)
    ax.set_xlabel("ΔMAE bei gemeinsamer Permutation [PP] (Balken: Faltenmittel, "
                  "Fehler: 95-%-KI über Artikel)")
    ax.set_title(f"A  Nutzen je Gruppe — rot = Rauschdecke ({decke:+.3f})", fontsize=9.5)
    ax.grid(alpha=0.3, axis="x")


def feld_b(ax, T):
    t = T[T.art != "Kontrolle"].sort_values("delta_auc").tail(25)
    yy = np.arange(len(t))
    ax.barh(yy, t.delta_auc, color=[FARBE[a] for a in t.art], alpha=0.85)
    k = T[T.art == "Kontrolle"].delta_auc
    ax.axvspan(k.min(), k.max(), color="#999999", alpha=0.25, label="Spanne Kontrolle")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(yy)
    ax.set_yticklabels(t.Gruppe, fontsize=7)
    ax.set_xlabel(f"ΔAUC für y ≥ {int(OBEN * 100)}. Perzentil")
    ax.set_title("B  Erkennung hohen Ausschusses ohne die Gruppe", fontsize=9.5)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="x")


def feld_c(ax, T, S, decke):
    t = T.merge(S, on="Gruppe")
    for a, d in t.groupby("art"):
        ax.scatter(d.shap_summe, d.delta_mae, color=FARBE[a], label=a, s=22)
    for _, r in t.nlargest(8, "delta_mae").iterrows():
        ax.annotate(r.Gruppe[:22], (r.shap_summe, r.delta_mae), fontsize=6.5)
    ax.axhline(decke, color="#d62728", ls="--", lw=1)
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_xlabel("Σ mittlerer |SHAP| der Gruppe aus 07 [PP]")
    ax.set_ylabel("ΔMAE [PP]")
    ax.set_title("C  Beitrag gegen Nutzen je Gruppe", fontsize=9.5)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)


def feld_d(ax, S, G):
    art = {k: v[0] for k, v in G.items()}
    s = S.assign(art=S.Gruppe.map(art)).groupby("art").shap_summe.sum()
    s = s / s.sum() * 100
    ax.bar(s.index, s.values, color=[FARBE[a] for a in s.index])
    for i, v in enumerate(s.values):
        ax.text(i, v + 1, f"{v:.1f} %", ha="center", fontsize=8)
    ax.set_ylabel("Anteil an Σ |SHAP| [%]")
    ax.set_title("D  Quellenanteil (Influx nach data_prep.influx_spalten)", fontsize=9.5)
    ax.grid(alpha=0.3, axis="y")
    return s


def main():
    print(f"Ebene: TEST_LEVEL={C.D.TEST_LEVEL} → CV-Ebene {cv_ebene()}")
    ds = C.D.load_split(verbose=False)
    falten, ebene = C.D.make_cv(ds.df, level=cv_ebene(), seed=CV_SEED)
    pred, perm_pred, perm_abs, falte_dmae, G, K, guete, quelle, y = rechnen(ds, falten)
    cluster = ds.df[SP["artikel"]].astype(str).to_numpy()

    T, auc0 = tabelle(pred, perm_pred, perm_abs, falte_dmae, G, K, y, cluster)
    S = shap_je_gruppe(G)
    kontrolle = T[T.art == "Kontrolle"]
    decke = float(kontrolle.delta_mae.max())
    echt = T[T.art != "Kontrolle"].merge(S, on="Gruppe", how="left")
    echt["ueber_decke"] = echt.delta_mae > decke
    echt["ki_ueber_null"] = echt.ki_lo > 0
    echt = echt.sort_values("delta_mae", ascending=False).reset_index(drop=True)

    mae = float(np.mean([g["mae"] for g in guete]))
    mae_med = float(np.mean([g["mae_median"] for g in guete]))
    fuss = (f"{C.D.CSV_PATH.name} | {len(y)} Billets, {len(falten)} Falten, CV-Ebene "
            f"{ebene} | {N_WDH} Permutationen je Gruppe und Falte | XGBoost {quelle}")

    pdf = C.pdf_oeffnen(name("13_sensoren.pdf"))
    fig, ax = C.seite(f"Welche Sensoren tragen? Gruppenpermutation — Ebene {C.D.TEST_LEVEL}", 2, 2)
    feld_a(ax[0], T, decke)
    feld_b(ax[1], T)
    feld_c(ax[2], T, S, decke)
    anteil = feld_d(ax[3], S, G)
    C.schliesse(pdf, fig, fuss)

    kan = echt[echt.art == "Kanal"]
    C.textseite(pdf, "Lesehinweise und Vorbehalte", [
        f"Modell MAE {mae:.3f} PP gegen Median-Vorhersage {mae_med:.3f} PP; der "
        f"gesamte Vorsprung beträgt {mae_med - mae:.3f} PP. Daran ist jedes ΔMAE zu messen.",
        f"Rauschdecke (Maximum von {N_KONTROLLE} Zufallsgruppen à {KONTROLLE_GROESSE} "
        f"Spalten ab Rang {KONTROLLE_AB_RANG}): {decke:+.4f} PP. Darüber liegen "
        f"{int(kan.ueber_decke.sum())} von {len(kan)} Kanälen; ein KI über null haben "
        f"{int(kan.ki_ueber_null.sum())}.",
        "Kanäle über der Decke: " + ", ".join(
            f"{r.Gruppe} {r.delta_mae:+.3f} [{r.ki_lo:+.3f}; {r.ki_hi:+.3f}]"
            for _, r in kan[kan.ueber_decke].iterrows()) + ".",
        f"AUC für das obere Dezil mit allen Merkmalen: {auc0:.3f}.",
        "Vorbehalt: Die gemeinsame Permutation misst den Nutzen DIESES Modells. "
        "Kann ein anderer Kanal oder ein Stammdatum dieselbe Information liefern, "
        "zeigt ein Neutraining ohne den Kanal weniger Verlust (vgl. LOCO in 08). "
        "Ein Kanal über der Decke ist ein Hinweis auf Zusammenhang, keine Ursache.",
    ])
    pdf.close()

    pcsv = C.speichere_rangliste(name("13_sensoren.csv"), echt, "delta_mae")
    p = C.speichere(name("13_sensoren.json"), {
        "test_level": C.D.TEST_LEVEL,
        "cv_level": ebene,
        "n_falten": len(falten),
        "n_wiederholungen": N_WDH,
        "params_quelle": quelle,
        "guete_falten": guete,
        "mae_mittel": mae,
        "mae_median_mittel": mae_med,
        "auc_oben_basis": auc0,
        "rauschdecke": decke,
        "kontrollgruppen": kontrolle.drop(columns="spalten").to_dict("records"),
        "quellenanteil_shap_prozent": anteil.round(2).to_dict(),
        "gruppen": echt.to_dict("records"),
        "csv": pcsv.name,
    })
    print(f"\ngeschrieben: {C.OUT_DIR / name('13_sensoren.pdf')}\n             {p}")
    print(f"MAE {mae:.3f} | Median {mae_med:.3f} | Decke {decke:+.4f} | AUC oben {auc0:.3f}")
    print(f"Quellenanteil Σ|SHAP|: {anteil.round(1).to_dict()}")
    for _, r in echt.head(20).iterrows():
        print(f"  {r.art:<11}{r.Gruppe:<28}{r.n_spalten:>3}  ΔMAE {r.delta_mae:+.4f} "
              f"[{r.ki_lo:+.3f}; {r.ki_hi:+.3f}] {r.n_falten_positiv}/5  ΔAUC {r.delta_auc:+.4f}")


if __name__ == "__main__":
    main()
