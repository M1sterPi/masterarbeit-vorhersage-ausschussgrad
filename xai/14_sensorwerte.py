"""14 — Wertebereiche und Schwellen der Prozesskanäle.

In der Arbeit: Abschnitt 7.3 und 7.5: Wertebereiche und kritische Bereiche der
Kanäle, Abbildung 20, Tabelle 16 (über xai/19 und xai/24).

Braucht xai/13 für artikel und auftrag. Option --nur-pdf.
Ausgaben unter reports/xai/, runs/xai/ und reports/xai/tabellen/.

    TEST_LEVEL=<billet|auftrag|artikel> python xai/14_sensorwerte.py
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

import _common as C
from data_prep import SP  # noqa: E402

S13 = importlib.import_module("13_sensoren")

CV_SEED = S13.CV_SEED
N_WDH = S13.N_WDH
N_BOOT = 1000
N_KLASSEN = 10
MIN_BLATT = 0.05
ZWEITES_AB = 0.25
PHASE_REGULAER = SP["letzte_phase"]


def name(datei: str) -> str:
    return S13.name(datei)


def gesicherte_kanaele() -> list[str]:
    """Kanäle mit KI über null in 13, artikel- oder auftragsrein; im billetweisen
    Lauf zusätzlich die nur billetweise gesicherten."""
    kan: list[str] = []
    ebenen = ("artikel", "auftrag") + (("billet",) if C.D.TEST_LEVEL == "billet" else ())
    for ebene in ebenen:
        p = C.TAB_DIR / f"13_sensoren_{ebene}.csv"
        if not p.exists():
            raise SystemExit(f"{p.name} fehlt — erst 13_sensoren.py laufen lassen.")
        t = pd.read_csv(p)
        for g in t[(t.art == "Kanal") & t.ki_ueber_null].Gruppe:
            if g not in kan:
                kan.append(g)
    return kan


def rechnen(ds, falten, kanaele):
    import xgboost as xgb

    params, quelle = C.xgb_parameter()
    _, fix = C._konstanten_aus_skript()
    X = ds.X_raw.reset_index(drop=True)
    y = ds.y.reset_index(drop=True).to_numpy()

    n = len(y)
    pred = np.full(n, np.nan)
    G = None
    shap: dict[str, np.ndarray] = {}
    wert: dict[str, np.ndarray] = {}
    perm_abs: dict[str, np.ndarray] = {}
    falte_dmae: dict[str, list[float]] = {}
    guete, additiv = [], []
    for f, (tr, va) in enumerate(falten, 1):
        Xtr, Xva, _ = C.D.process_features(X.iloc[tr], X.iloc[va])
        if G is None:
            G = S13.gruppen(list(Xtr.columns))
            fehlt = [k for k in kanaele if k not in G]
            if fehlt:
                raise SystemExit(f"Kanaele nicht in den Merkmalen: {fehlt}")
            for k in kanaele:
                for c in G[k][1]:
                    shap[c] = np.full(n, np.nan)
                    wert[c] = np.full(n, np.nan)
                    perm_abs[c] = np.full(n, np.nan)
                    falte_dmae[c] = []
        m = xgb.XGBRegressor(**params, **fix)
        m.fit(Xtr, y[tr])
        p = m.predict(Xva)
        pred[va] = p
        sv, basis = C.shap_xgboost(m, Xva)
        additiv.append(C.pruefe_additivitaet(sv, basis, p, f"Falte {f}"))
        spalten = list(Xva.columns)
        mae = mean_absolute_error(y[va], p)
        mae_med = mean_absolute_error(y[va], np.full(len(va), np.median(y[tr])))
        guete.append({"falte": f, "mae": float(mae), "mae_median": float(mae_med)})
        print(f"  Falte {f}: MAE {mae:.3f} (Median {mae_med:.3f})", flush=True)

        rng = np.random.default_rng(CV_SEED + 100 + f)
        for k in kanaele:
            for c in G[k][1]:
                if c not in spalten:
                    continue
                shap[c][va] = sv[:, spalten.index(c)]
                wert[c][va] = pd.to_numeric(Xva[c], errors="coerce").to_numpy(float)
                ab = np.zeros(len(va))
                for _ in range(N_WDH):
                    X2 = Xva.copy()
                    X2[c] = Xva[c].to_numpy()[rng.permutation(len(va))]
                    ab += np.abs(y[va] - m.predict(X2)) / N_WDH
                perm_abs[c][va] = ab
                falte_dmae[c].append(float(ab.mean() - mae))
    return dict(pred=pred, shap=shap, wert=wert, perm_abs=perm_abs,
                falte_dmae=falte_dmae, G=G, guete=guete, quelle=quelle, y=y,
                additiv=max(additiv), X=X)


def boot_mittel(d, cluster, rng) -> tuple[float, float]:
    """95-%-KI des Mittels von d; gezogen werden ganze Artikel."""
    codes, inv = np.unique(cluster, return_inverse=True)
    s = np.bincount(inv, weights=d, minlength=len(codes))
    c = np.bincount(inv, minlength=len(codes))
    z = rng.integers(0, len(codes), (N_BOOT, len(codes)))
    w = s[z].sum(1) / np.maximum(c[z].sum(1), 1)
    return float(np.percentile(w, 2.5)), float(np.percentile(w, 97.5))


def boot_differenz(d, oben, cluster, rng) -> tuple[float, float]:
    """95-%-KI von mean(d | oben) − mean(d | unten), Ziehung ueber Artikel."""
    codes, inv = np.unique(cluster, return_inverse=True)
    k = len(codes)
    so = np.bincount(inv, weights=d * oben, minlength=k)
    co = np.bincount(inv, weights=oben.astype(float), minlength=k)
    su = np.bincount(inv, weights=d * ~oben, minlength=k)
    cu = np.bincount(inv, weights=(~oben).astype(float), minlength=k)
    z = rng.integers(0, k, (N_BOOT, k))
    w = so[z].sum(1) / np.maximum(co[z].sum(1), 1) - su[z].sum(1) / np.maximum(cu[z].sum(1), 1)
    return float(np.percentile(w, 2.5)), float(np.percentile(w, 97.5))


def beste_schwelle(v, s, min_blatt=MIN_BLATT) -> float:
    """Trennwert eines einstufigen Regressionsbaums von s auf v (SSE-minimal)."""
    o = np.argsort(v, kind="stable")
    v, s = v[o], s[o]
    n = len(v)
    cs, cq = np.cumsum(s), np.cumsum(s * s)
    i = np.arange(1, n)
    links = cq[i - 1] - cs[i - 1] ** 2 / i
    rechts = (cq[-1] - cq[i - 1]) - (cs[-1] - cs[i - 1]) ** 2 / (n - i)
    sse = links + rechts
    gueltig = (v[i] > v[i - 1]) & (i >= min_blatt * n) & (n - i >= min_blatt * n)
    if not gueltig.any():
        return float("nan")
    j = i[gueltig][np.argmin(sse[gueltig])]
    return float((v[j - 1] + v[j]) / 2)


def gruppenzentriert(y, gruppe) -> np.ndarray:
    return y - pd.Series(y).groupby(gruppe).transform("mean").to_numpy()


def aggregate_tabelle(R, kanaele, cluster, e0, rng) -> pd.DataFrame:
    zeilen = []
    for k in kanaele:
        cols = R["G"][k][1]
        betrag = {c: float(np.nanmean(np.abs(R["shap"][c]))) for c in cols}
        summe = sum(betrag.values()) or 1.0
        for c in cols:
            d = R["perm_abs"][c] - e0
            ok = ~np.isnan(d)
            lo, hi = boot_mittel(d[ok], cluster[ok], rng)
            fd = R["falte_dmae"][c]
            zeilen.append({
                "Kanal": k,
                "Aggregat": c[len(k):].lstrip("_"),
                "Merkmal": c,
                "shap_betrag": betrag[c],
                "shap_anteil_kanal": betrag[c] / summe,
                "delta_mae_einzeln": float(np.mean(fd)),
                "n_falten_positiv": int(np.sum(np.array(fd) > 0)),
                "ki_lo": lo,
                "ki_hi": hi,
                "anteil_fehlend": float(np.mean(np.isnan(R["wert"][c]))),
            })
    return pd.DataFrame(zeilen)


def fuehrende(A: pd.DataFrame) -> list[tuple[str, str]]:
    aus = []
    for k, t in A.groupby("Kanal", sort=False):
        t = t.sort_values("shap_betrag", ascending=False)
        aus.append((k, t.Merkmal.iloc[0]))
        if len(t) > 1 and t.shap_anteil_kanal.iloc[1] >= ZWEITES_AB:
            aus.append((k, t.Merkmal.iloc[1]))
    return aus


def klassen(v: np.ndarray) -> np.ndarray:
    """Quantilklassen; Bindungen (Zaehlgroessen) werden zusammengelegt."""
    u = np.unique(v)
    if len(u) <= N_KLASSEN:
        return np.searchsorted(u, v)
    return pd.qcut(v, N_KLASSEN, labels=False, duplicates="drop").astype(int)


def werte_analyse(R, kanal, c, artikel, auftrag, abbruch, X, rng):
    v_all, s_all, y_all = R["wert"][c], R["shap"][c], R["y"]
    ok = ~np.isnan(v_all) & ~np.isnan(s_all)
    v, s, y = v_all[ok], s_all[ok], y_all[ok]
    art, auf = artikel[ok], auftrag[ok]
    y_art = gruppenzentriert(y_all, artikel)[ok]
    y_auf = gruppenzentriert(y_all, auftrag)[ok]
    s_art = gruppenzentriert(np.where(ok, s_all, 0.0), artikel)[ok]

    kl = klassen(v)
    kz = []
    for i in np.unique(kl):
        mk = kl == i
        lo_s, hi_s = boot_mittel(s[mk], art[mk], rng)
        lo_y, hi_y = boot_mittel(y_art[mk], art[mk], rng)
        kz.append({
            "Kanal": kanal, "Merkmal": c, "klasse": int(i) + 1,
            "wert_von": float(v[mk].min()), "wert_bis": float(v[mk].max()),
            "wert_median": float(np.median(v[mk])), "n": int(mk.sum()),
            "n_artikel": int(len(np.unique(art[mk]))),
            "shap_mittel": float(s[mk].mean()), "shap_ki_lo": lo_s, "shap_ki_hi": hi_s,
            "shap_artikelbereinigt": float(s_art[mk].mean()),
            "y_mittel": float(y[mk].mean()), "y_median": float(np.median(y[mk])),
            "y_artikelbereinigt": float(y_art[mk].mean()),
            "y_art_ki_lo": lo_y, "y_art_ki_hi": hi_y,
            "y_auftragsbereinigt": float(y_auf[mk].mean()),
        })

    t = beste_schwelle(v, s)
    codes, inv = np.unique(art, return_inverse=True)
    idx_je = [np.flatnonzero(inv == j) for j in range(len(codes))]
    ts = []
    for _ in range(200):
        z = rng.integers(0, len(codes), len(codes))
        ii = np.concatenate([idx_je[j] for j in z])
        ts.append(beste_schwelle(v[ii], s[ii]))
    ts = np.array(ts)
    oben = v > t

    def effekt(d, maske=None):
        mm = np.ones_like(oben) if maske is None else maske
        dd, oo, aa = d[mm], oben[mm], art[mm]
        if oo.all() or (~oo).all():
            return float("nan"), float("nan"), float("nan")
        lo, hi = boot_differenz(dd, oo, aa, rng)
        return float(dd[oo].mean() - dd[~oo].mean()), lo, hi

    regulaer = ~abbruch[ok]
    eff = {
        "shap": effekt(s), "y_roh": effekt(y), "y_artikel": effekt(y_art),
        "y_auftrag": effekt(y_auf),
        "shap_ohne_abbruch": effekt(s, regulaer), "y_artikel_ohne_abbruch": effekt(y_art, regulaer),
    }
    stoer = {}
    for ref in ("batch_duration_s", SP["phase"] + "_max", SP["phase"] + "_last"):
        if ref in X.columns and ref != c:
            r = pd.to_numeric(X[ref], errors="coerce").to_numpy(float)[ok]
            mm = ~np.isnan(r)
            stoer[ref] = float(spearmanr(v[mm], r[mm]).statistic) if mm.sum() > 10 else float("nan")
    s_zeile = {
        "Kanal": kanal, "Merkmal": c, "n": int(ok.sum()),
        "schwelle": t,
        "schwelle_boot_p05": float(np.nanpercentile(ts, 5)),
        "schwelle_boot_median": float(np.nanmedian(ts)),
        "schwelle_boot_p95": float(np.nanpercentile(ts, 95)),
        "anteil_oberhalb": float(oben.mean()),
        "artikel_oberhalb": int(len(np.unique(art[oben]))),
        "richtung": "hoeher" if eff["shap"][0] > 0 else "niedriger",
        "rho_wert_shap": float(spearmanr(v, s).statistic),
        "rho_innen_wert_y": float(spearmanr(gruppenzentriert(v, art), y_art).statistic),
        "n_abbruch": int((~regulaer).sum()),
        "anteil_abbruch_oberhalb": float((~regulaer)[oben].mean()) if oben.any() else float("nan"),
        "anteil_abbruch_unterhalb": float((~regulaer)[~oben].mean()) if (~oben).any() else float("nan"),
    }
    for kk, (w, lo, hi) in eff.items():
        s_zeile[f"d_{kk}"], s_zeile[f"d_{kk}_lo"], s_zeile[f"d_{kk}_hi"] = w, lo, hi
    for kk, w in stoer.items():
        s_zeile[f"rho_{kk}"] = w
    return pd.DataFrame(kz), s_zeile


def feld(ax, K, s):
    x = K.wert_median.to_numpy()
    ax.axhline(0, color="#999999", lw=0.6)
    ax.plot(x, K.shap_mittel, "o-", color="#e6550d", ms=3, lw=1.2, label="SHAP (Modell)")
    ax.plot(x, K.y_artikelbereinigt, "s-", color="#3182bd", ms=3, lw=1.2,
            label="y − Artikelmittel (Daten)")
    ax.plot(x, K.y_auftragsbereinigt, "^--", color="#4d4d4d", ms=3.5, lw=1.2,
            label="y − Auftragsmittel (Daten)")
    ax.set_title(s["Merkmal"], fontsize=8.5)
    ax.set_xlabel("Merkmalswert (Klassenmedian)", fontsize=7.5)
    ax.set_ylabel("PP", fontsize=7.5)
    ax.tick_params(labelsize=7)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 4))
    ax.grid(alpha=0.3)


def main():
    print(f"Ebene: TEST_LEVEL={C.D.TEST_LEVEL} → CV-Ebene {S13.cv_ebene()}")
    kanaele = gesicherte_kanaele()
    print("Kanaele:", ", ".join(kanaele))
    ds = C.D.load_split(verbose=False)
    falten, ebene = C.D.make_cv(ds.df, level=S13.cv_ebene(), seed=CV_SEED)
    R = rechnen(ds, falten, kanaele)

    df = ds.df.reset_index(drop=True)
    artikel = df[SP["artikel"]].astype(str).to_numpy()
    auftrag = df[SP["auftrag"]].astype(str).to_numpy()
    phase = pd.to_numeric(R["X"][SP["phase"] + "_max"], errors="coerce").to_numpy(float)
    abbruch = np.nan_to_num(phase, nan=PHASE_REGULAER) < PHASE_REGULAER
    e0 = np.abs(R["y"] - R["pred"])
    rng = np.random.default_rng(C.SEED)

    A = aggregate_tabelle(R, kanaele, artikel, e0, rng)
    KL, SW = [], []
    for kanal, c in fuehrende(A):
        K, s = werte_analyse(R, kanal, c, artikel, auftrag, abbruch, R["X"], rng)
        KL.append(K)
        SW.append(s)
        print(f"  {c:<34} Schwelle {s['schwelle']:>11.4g} [{s['schwelle_boot_p05']:.4g}; "
              f"{s['schwelle_boot_p95']:.4g}]  oberhalb {s['anteil_oberhalb']:.0%}  "
              f"ΔSHAP {s['d_shap']:+.2f}  Δy_art {s['d_y_artikel']:+.2f} "
              f"[{s['d_y_artikel_lo']:+.2f}; {s['d_y_artikel_hi']:+.2f}]")
    KL = pd.concat(KL, ignore_index=True)
    SW = pd.DataFrame(SW)

    mae = float(np.mean([g["mae"] for g in R["guete"]]))
    mae_med = float(np.mean([g["mae_median"] for g in R["guete"]]))
    fuss = (f"{C.D.CSV_PATH.name} | {len(R['y'])} Billets, {len(falten)} Falten, CV-Ebene "
            f"{ebene} | OOF-TreeSHAP XGBoost {R['quelle']} | KI: Cluster-Bootstrap über "
            f"Artikel ({N_BOOT})")

    pdf_schreiben(SW, KL, mae, mae_med, fuss)

    pa = C.speichere_rangliste(name("14_sensorwerte_aggregate.csv"), A, "shap_betrag")
    pk = C.speichere_rangliste(name("14_sensorwerte.csv"), KL, "_reihenfolge_behalten")
    ps = C.speichere_rangliste(name("14_sensorwerte_schwellen.csv"), SW, "d_shap")
    p = C.speichere(name("14_sensorwerte.json"), {
        "test_level": C.D.TEST_LEVEL, "cv_level": ebene,
        "n_falten": len(falten), "params_quelle": R["quelle"],
        "additivitaet_max": R["additiv"],
        "guete_falten": R["guete"], "mae_mittel": mae, "mae_median_mittel": mae_med,
        "kanaele": kanaele, "n_abbruch": int(abbruch.sum()), "fuss": fuss,
        "aggregate": A.to_dict("records"),
        "schwellen": SW.to_dict("records"),
        "klassen": KL.to_dict("records"),
        "csv": [pa.name, pk.name, ps.name],
    })
    print(f"\ngeschrieben: {C.OUT_DIR / name('14_sensorwerte.pdf')}\n             {p}")
    print(f"MAE {mae:.3f} | Median {mae_med:.3f} | Abbruchschmelzen {int(abbruch.sum())}")


def pdf_schreiben(SW, KL, mae, mae_med, fuss):
    pdf = C.pdf_oeffnen(name("14_sensorwerte.pdf"))
    for start in range(0, len(SW), 6):
        fig, ax = C.seite(f"Sensorwerte — Ebene {C.D.TEST_LEVEL}", 2, 3)
        for a, (_, s) in zip(ax, SW.iloc[start:start + 6].iterrows()):
            feld(a, KL[KL.Merkmal == s.Merkmal], s)
        for a in ax[len(SW.iloc[start:start + 6]):]:
            a.axis("off")
        ax[0].legend(fontsize=6.5)
        C.schliesse(pdf, fig, fuss)
    C.textseite(pdf, "Lesehinweise", [
        f"Modell MAE {mae:.3f} PP gegen Median-Vorhersage {mae_med:.3f} PP.",
        "Orange: mittlerer OOF-SHAP-Wert je Wertklasse (Modellsicht). Blau: mittlere "
        "Abweichung des Ausschusses vom Mittel des eigenen Artikels — enthält noch die "
        "Unterschiede zwischen Aufträgen desselben Artikels. Grau gestrichelt: Abweichung "
        "vom Mittel des eigenen Auftrags — verglichen werden nur Billets desselben Auftrags.",
        "Fällt Blau mit dem Wert, Grau aber nicht, stammt der Zusammenhang aus Unterschieden "
        "zwischen Aufträgen, nicht zwischen Billets eines Auftrags.",
        "Die Ebene im Dateinamen betrifft nur die Faltenbildung des Modells (Orange); Blau und "
        "Grau sind reine Datenauswertungen und auf allen Ebenen gleich.",
        "Konfidenzintervalle (Cluster-Bootstrap über Artikel) und Schwellen stehen in "
        "14_sensorwerte_<ebene>.csv und 14_sensorwerte_schwellen_<ebene>.csv.",
    ])
    pdf.close()


def auftragsreihe_nachtragen(KL: pd.DataFrame) -> pd.DataFrame:
    """y − Auftragsmittel je Wertklasse fuer JSON aus Laeufen vor dieser Spalte."""
    ds = C.D.load_split(verbose=False)
    df = ds.df.reset_index(drop=True)
    X = ds.X_raw.reset_index(drop=True)
    y = ds.y.reset_index(drop=True).to_numpy(float)
    y_art = gruppenzentriert(y, df[SP["artikel"]].astype(str).to_numpy())
    y_auf = gruppenzentriert(y, df[SP["auftrag"]].astype(str).to_numpy())
    KL = KL.copy()
    KL["y_auftragsbereinigt"] = np.nan
    for c, idx in KL.groupby("Merkmal").groups.items():
        v = pd.to_numeric(X[c], errors="coerce").to_numpy(float)
        v[np.abs(v) > C.D.MAX_PLAUSIBLE_ABS] = np.nan
        ok = ~np.isnan(v)
        kl = klassen(v[ok])
        for i in idx:
            mk = kl == KL.at[i, "klasse"] - 1
            if mk.sum() != KL.at[i, "n"] or not np.isclose(
                    y_art[ok][mk].mean(), KL.at[i, "y_artikelbereinigt"]):
                raise SystemExit(f"{c} Klasse {KL.at[i, 'klasse']}: passt nicht zum "
                                 "gespeicherten Lauf — Skript ohne --nur-pdf neu rechnen")
            KL.at[i, "y_auftragsbereinigt"] = float(y_auf[ok][mk].mean())
    return KL


def nur_pdf():
    """PDF aus dem gespeicherten JSON neu zeichnen, ohne Neuberechnung."""
    import json
    j = json.loads((C.ERG_DIR / name("14_sensorwerte.json")).read_text(encoding="utf-8"))
    fuss = j.get("fuss") or (f"{C.D.CSV_PATH.name} | {j['n_falten']} Falten, CV-Ebene "
                             f"{j['cv_level']} | OOF-TreeSHAP XGBoost {j['params_quelle']} | KI: "
                             f"Cluster-Bootstrap über Artikel ({N_BOOT})")
    KL = pd.DataFrame(j["klassen"])
    if "y_auftragsbereinigt" not in KL.columns:
        KL = auftragsreihe_nachtragen(KL)
    pdf_schreiben(pd.DataFrame(j["schwellen"]), KL,
                  j["mae_mittel"], j["mae_median_mittel"], fuss)
    print("geschrieben:", C.OUT_DIR / name("14_sensorwerte.pdf"))


if __name__ == "__main__":
    import sys
    nur_pdf() if "--nur-pdf" in sys.argv else main()
