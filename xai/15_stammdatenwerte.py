"""15 — Wertebereiche und Schwellen der Stammdaten.

In der Arbeit: Abschnitt 7.3: Wertebereiche der Stammdatenmerkmale, Abbildung
21 (über xai/24) und die Einteilung in Artikelmerkmale.

Braucht xai/13 für artikel und auftrag. Option --nur-pdf.
Ausgaben unter reports/xai/, runs/xai/ und reports/xai/tabellen/.

    TEST_LEVEL=<billet|auftrag|artikel> python xai/15_stammdatenwerte.py
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

import _common as C
from data_prep import SP  # noqa: E402

S13 = importlib.import_module("13_sensoren")
S14 = importlib.import_module("14_sensorwerte")

CV_SEED = S13.CV_SEED
N_WDH = S13.N_WDH
MAX_JE_BLOCK = 3
AB_ANTEIL = 0.10
KONSTANT_AB = 0.95
INNEN_BIS = 0.05
MIN_KAT = 0.01


def name(datei: str) -> str:
    return S13.name(datei)


def gesicherte_bloecke() -> list[str]:
    bl: list[str] = []
    for ebene in ("artikel", "auftrag"):
        t = pd.read_csv(C.TAB_DIR / f"13_sensoren_{ebene}.csv")
        for g in t[(t.art == "Stammdaten") & t.ki_ueber_null].Gruppe:
            if g not in bl:
                bl.append(g)
    return bl


def rechnen(ds, falten, bloecke):
    import xgboost as xgb

    params, quelle = C.xgb_parameter()
    _, fix = C._konstanten_aus_skript()
    X = ds.X_raw.reset_index(drop=True)
    y = ds.y.reset_index(drop=True).to_numpy()
    n = len(y)
    pred = np.full(n, np.nan)
    G, spalten_bl = None, {}
    shap, code, perm_abs, falte_dmae = {}, {}, {}, {}
    guete, additiv = [], []
    for f, (tr, va) in enumerate(falten, 1):
        Xtr, Xva, _ = C.D.process_features(X.iloc[tr], X.iloc[va])
        if G is None:
            G = S13.gruppen(list(Xtr.columns))
            for b in bloecke:
                spalten_bl[b] = list(G[b][1])
                for c in spalten_bl[b]:
                    shap[c] = np.full(n, np.nan)
                    code[c] = np.full(n, np.nan)
                    perm_abs[c] = np.full(n, np.nan)
                    falte_dmae[c] = []
        m = xgb.XGBRegressor(**params, **fix)
        m.fit(Xtr, y[tr])
        p = m.predict(Xva)
        pred[va] = p
        sv, basis = C.shap_xgboost(m, Xva)
        additiv.append(C.pruefe_additivitaet(sv, basis, p, f"Falte {f}"))
        sp = list(Xva.columns)
        mae = mean_absolute_error(y[va], p)
        mae_med = mean_absolute_error(y[va], np.full(len(va), np.median(y[tr])))
        guete.append({"falte": f, "mae": float(mae), "mae_median": float(mae_med)})
        print(f"  Falte {f}: MAE {mae:.3f} (Median {mae_med:.3f})", flush=True)
        rng = np.random.default_rng(CV_SEED + 200 + f)
        for b in bloecke:
            for c in spalten_bl[b]:
                if c not in sp:
                    continue
                shap[c][va] = sv[:, sp.index(c)]
                code[c][va] = pd.to_numeric(Xva[c], errors="coerce").to_numpy(float)
                ab = np.zeros(len(va))
                for _ in range(N_WDH):
                    X2 = Xva.copy()
                    X2[c] = Xva[c].to_numpy()[rng.permutation(len(va))]
                    ab += np.abs(y[va] - m.predict(X2)) / N_WDH
                perm_abs[c][va] = ab
                falte_dmae[c].append(float(ab.mean() - mae))
    return dict(pred=pred, shap=shap, code=code, perm_abs=perm_abs, falte_dmae=falte_dmae,
                G=G, spalten_bl=spalten_bl, guete=guete, quelle=quelle, y=y,
                additiv=max(additiv), X=X)


def ist_kategorial(roh: pd.Series) -> bool:
    if roh.dtype != object:
        return False
    conv = C.D._to_num_de(roh)
    return conv.notna().sum() < 0.7 * roh.notna().sum()


def konstanz(roh: pd.Series, artikel: np.ndarray) -> dict:
    g = roh.groupby(artikel)
    konst = float((g.nunique(dropna=False) <= 1).mean())
    num = pd.to_numeric(roh, errors="coerce") if not ist_kategorial(roh) else None
    if num is not None and num.notna().mean() > 0.5 and num.var() > 0:
        innen = float((num - num.groupby(artikel).transform("mean")).var() / num.var())
    else:
        modus = roh.astype(str).groupby(artikel).transform(lambda s: s.value_counts().index[0])
        innen = float((roh.astype(str) != modus).mean())
    return {"anteil_artikel_konstant": konst, "anteil_innen": innen,
            "artikelmerkmal": konst >= KONSTANT_AB or innen < INNEN_BIS}


def kategorien(roh: pd.Series) -> np.ndarray:
    """Kategorie je Billet; fehlende als '(fehlt)', seltene zu 'uebrige'."""
    k = roh.astype(str).where(roh.notna(), "(fehlt)").to_numpy()
    anteil = pd.Series(k).value_counts(normalize=True)
    selten = set(anteil[anteil < MIN_KAT].index)
    return np.where(np.isin(k, list(selten)), "übrige", k)


def kategorie_werte(roh: pd.Series, s: np.ndarray):
    """Kategorien nach mittlerem OOF-SHAP ordnen; seltene zu 'uebrige'."""
    k = kategorien(roh)
    ok = ~np.isnan(s)
    mittel = pd.Series(s[ok]).groupby(k[ok]).mean().sort_values()
    rang = {kat: i for i, kat in enumerate(mittel.index)}
    v = np.array([rang.get(x, np.nan) for x in k], dtype=float)
    return v, list(mittel.index)


def block_tabelle(R, bloecke, artikel, e0, rng) -> pd.DataFrame:
    zeilen = []
    for b in bloecke:
        cols = R["spalten_bl"][b]
        betrag = {c: float(np.nanmean(np.abs(R["shap"][c]))) for c in cols}
        summe = sum(betrag.values()) or 1.0
        for c in cols:
            d = R["perm_abs"][c] - e0
            ok = ~np.isnan(d)
            lo, hi = S14.boot_mittel(d[ok], artikel[ok], rng)
            fd = R["falte_dmae"][c]
            zeilen.append({
                "Block": b, "Merkmal": c, "shap_betrag": betrag[c],
                "shap_anteil_block": betrag[c] / summe,
                "delta_mae_einzeln": float(np.mean(fd)),
                "n_falten_positiv": int(np.sum(np.array(fd) > 0)),
                "ki_lo": lo, "ki_hi": hi,
                **konstanz(R["X"][c], artikel),
                "kategorial": ist_kategorial(R["X"][c]),
            })
    return pd.DataFrame(zeilen)


KLASSEN_ORIGINAL = S14.klassen


def fuehrende(A: pd.DataFrame) -> list[tuple[str, str]]:
    aus = []
    for b, t in A.groupby("Block", sort=False):
        t = t.sort_values("shap_betrag", ascending=False)
        for i, (_, r) in enumerate(t.head(MAX_JE_BLOCK).iterrows()):
            if i == 0 or r.shap_anteil_block >= AB_ANTEIL:
                aus.append((b, r.Merkmal))
    return aus


def feld(ax, K, s, kats):
    x = np.arange(len(K)) if kats else K.wert_median.to_numpy()
    ax.axhline(0, color="#999999", lw=0.6)
    ax.plot(x, K.shap_mittel, "o-", color="#e6550d", ms=3, lw=1.2, label="SHAP (Modell)")
    ax.plot(x, K.y_mittel - s["y_gesamt"], "D-.", color="black", ms=3, lw=1.0,
            label="y − Gesamtmittel (Daten)")
    im_auftrag = not np.allclose(K.y_auftragsbereinigt, 0, atol=1e-9)
    if not s["artikelmerkmal"]:
        ax.plot(x, K.y_artikelbereinigt, "s-", color="#3182bd", ms=3, lw=1.2,
                label="y − Artikelmittel (Daten)")
        if im_auftrag:
            ax.plot(x, K.y_auftragsbereinigt, "^--", color="#4d4d4d", ms=3.5, lw=1.2,
                    label="y − Auftragsmittel (Daten)")
    art = ("Artikelmerkmal" if s["artikelmerkmal"] else "variiert im Artikel"
           + ("" if im_auftrag else ", nicht im Auftrag"))
    ax.set_title(f"{s['Merkmal']} ({art})", fontsize=8.5)
    ax.set_ylabel("PP", fontsize=7.5)
    ax.tick_params(labelsize=7)
    if kats:
        ax.set_xticks(x, [k[:16] for k in K.kategorien], rotation=40, ha="right", fontsize=6.5)
        ax.set_xlabel("Kategorie (nach mittlerem SHAP geordnet)", fontsize=7.5)
    else:
        ax.set_xlabel("Merkmalswert (Klassenmedian)", fontsize=7.5)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 4))
    ax.grid(alpha=0.3)


def main():
    print(f"Ebene: TEST_LEVEL={C.D.TEST_LEVEL} → CV-Ebene {S13.cv_ebene()}")
    bloecke = gesicherte_bloecke()
    print("Bloecke:", ", ".join(bloecke))
    ds = C.D.load_split(verbose=False)
    falten, ebene = C.D.make_cv(ds.df, level=S13.cv_ebene(), seed=CV_SEED)
    R = rechnen(ds, falten, bloecke)

    df = ds.df.reset_index(drop=True)
    artikel = df[SP["artikel"]].astype(str).to_numpy()
    auftrag = df[SP["auftrag"]].astype(str).to_numpy()
    phase = pd.to_numeric(R["X"][SP["phase"] + "_max"], errors="coerce").to_numpy(float)
    abbruch = np.nan_to_num(phase, nan=S14.PHASE_REGULAER) < S14.PHASE_REGULAER
    e0 = np.abs(R["y"] - R["pred"])
    rng = np.random.default_rng(C.SEED)
    y_gesamt = float(R["y"].mean())

    A = block_tabelle(R, bloecke, artikel, e0, rng)
    KL, SW = [], []
    for b, c in fuehrende(A):
        a = A[A.Merkmal == c].iloc[0]
        kats = None
        if a.kategorial:
            v, kats = kategorie_werte(R["X"][c], R["shap"][c])
        else:
            v = pd.to_numeric(R["X"][c], errors="coerce").to_numpy(float)
        R14 = {"wert": {c: v}, "shap": R["shap"], "y": R["y"]}
        K, s = S14.werte_analyse(R14, b, c, artikel, auftrag, abbruch, R["X"], rng)
        K = K.rename(columns={"Kanal": "Block"})
        s["Block"] = s.pop("Kanal")
        lo_hi = []
        for _, k in K.iterrows():
            mk = (v >= k.wert_von) & (v <= k.wert_bis) & ~np.isnan(R["shap"][c])
            lo_hi.append(S14.boot_mittel(R["y"][mk], artikel[mk], rng))
        K["y_roh_ki_lo"], K["y_roh_ki_hi"] = zip(*lo_hi)
        if kats is not None:
            K["kategorien"] = [", ".join(kats[int(i)] for i in range(int(k.wert_von), int(k.wert_bis) + 1))
                               for _, k in K.iterrows()]
            s["kategorien_oberhalb"] = ", ".join(kats[i] for i in range(len(kats)) if i > s["schwelle"])
        s.update({k: a[k] for k in ("anteil_artikel_konstant", "anteil_innen", "artikelmerkmal",
                                    "kategorial", "shap_anteil_block", "delta_mae_einzeln",
                                    "ki_lo", "ki_hi")})
        s["y_gesamt"] = y_gesamt
        KL.append(K)
        SW.append(s)
        print(f"  {c:<26} {'Artikelm.' if a.artikelmerkmal else 'variiert':9s} "
              f"Schwelle {s['schwelle']:>10.4g}  jenseits {s['anteil_oberhalb']:.0%} "
              f"({s['artikel_oberhalb']} Art.)  ΔSHAP {s['d_shap']:+.2f}  Δy_roh {s['d_y_roh']:+.2f} "
              f"[{s['d_y_roh_lo']:+.2f}; {s['d_y_roh_hi']:+.2f}]  Δy_art {s['d_y_artikel']:+.2f} "
              f"[{s['d_y_artikel_lo']:+.2f}; {s['d_y_artikel_hi']:+.2f}]"
              + (f"  oben: {s['kategorien_oberhalb']}" if kats else ""))
    KL = pd.concat(KL, ignore_index=True)
    SW = pd.DataFrame(SW)

    mae = float(np.mean([g["mae"] for g in R["guete"]]))
    mae_med = float(np.mean([g["mae_median"] for g in R["guete"]]))
    fuss = (f"{C.D.CSV_PATH.name} | {len(R['y'])} Billets, {len(falten)} Falten, CV-Ebene "
            f"{ebene} | OOF-TreeSHAP XGBoost {R['quelle']} | KI: Cluster-Bootstrap über "
            f"Artikel ({S14.N_BOOT})")
    pdf_schreiben(SW, KL, mae, mae_med, fuss)

    pa = C.speichere_rangliste(name("15_stammdatenwerte_merkmale.csv"), A, "shap_betrag")
    pk = C.speichere_rangliste(name("15_stammdatenwerte.csv"), KL, "_reihenfolge_behalten")
    ps = C.speichere_rangliste(name("15_stammdatenwerte_schwellen.csv"), SW, "d_shap")
    p = C.speichere(name("15_stammdatenwerte.json"), {
        "test_level": C.D.TEST_LEVEL, "cv_level": ebene, "n_falten": len(falten),
        "params_quelle": R["quelle"], "additivitaet_max": R["additiv"],
        "guete_falten": R["guete"], "mae_mittel": mae, "mae_median_mittel": mae_med,
        "bloecke": bloecke, "fuss": fuss,
        "merkmale": A.to_dict("records"), "schwellen": SW.to_dict("records"),
        "klassen": KL.to_dict("records"), "csv": [pa.name, pk.name, ps.name],
    })
    print(f"\ngeschrieben: {C.OUT_DIR / name('15_stammdatenwerte.pdf')}\n             {p}")
    print(f"MAE {mae:.3f} | Median {mae_med:.3f}")


def pdf_schreiben(SW, KL, mae, mae_med, fuss):
    pdf = C.pdf_oeffnen(name("15_stammdatenwerte.pdf"))
    for start in range(0, len(SW), 6):
        fig, ax = C.seite(f"Stammdatenwerte — Ebene {C.D.TEST_LEVEL}", 2, 3)
        for a_, (_, s) in zip(ax, SW.iloc[start:start + 6].iterrows()):
            feld(a_, KL[KL.Merkmal == s.Merkmal], s, bool(s.kategorial))
        for a_ in ax[len(SW.iloc[start:start + 6]):]:
            a_.axis("off")
        eintraege = {}
        for a_ in ax:
            for h, l in zip(*a_.get_legend_handles_labels()):
                eintraege.setdefault(l, h)
        ax[0].legend(eintraege.values(), eintraege.keys(), fontsize=6.5)
        C.schliesse(pdf, fig, fuss)
    C.textseite(pdf, "Lesehinweise", [
        f"Modell MAE {mae:.3f} PP gegen Median-Vorhersage {mae_med:.3f} PP.",
        "Orange: mittlerer OOF-SHAP-Wert je Wertklasse (Modellsicht). Schwarz "
        "(Strich-Punkt): beobachteter Ausschuss je Klasse minus Gesamtmittel — vergleicht "
        "auch verschiedene Artikel miteinander. Blau: Abweichung vom Mittel des eigenen "
        "Artikels. Grau gestrichelt: Abweichung vom Mittel des eigenen Auftrags — "
        "verglichen werden nur Billets desselben Auftrags.",
        "Blau und Grau fehlen bei Artikelmerkmalen: Jede Wertklasse enthält dort ganze "
        "Artikel, beide Reihen wären null. Ihr Effekt ist nur zwischen Artikeln "
        "beobachtbar, und die Zahl der Artikel im kritischen Bereich ist die eigentliche "
        "Stichprobengröße; ein Effekt aus wenigen Artikeln ist ein Artikelunterschied, "
        "keine Wirkung des Merkmals. Aus demselben Grund fehlt Grau, wenn ein Merkmal "
        "innerhalb jedes Auftrags fest ist (Titelzusatz „nicht im Auftrag“).",
        "Fällt Schwarz mit dem Wert, Blau aber nicht, stammt der Zusammenhang aus "
        "Unterschieden zwischen Artikeln; fällt Blau, Grau aber nicht, aus Unterschieden "
        "zwischen Aufträgen desselben Artikels.",
        "Die Ebene im Dateinamen betrifft nur die Faltenbildung des Modells (Orange); "
        "Schwarz, Blau und Grau sind reine Datenauswertungen und auf allen Ebenen gleich.",
        "Konfidenzintervalle (Cluster-Bootstrap über Artikel) und Schwellen stehen in "
        "15_stammdatenwerte_<ebene>.csv und 15_stammdatenwerte_schwellen_<ebene>.csv.",
    ])
    pdf.close()


def auftragsreihe_nachtragen(KL: pd.DataFrame, SW: pd.DataFrame) -> pd.DataFrame:
    """y − Auftragsmittel je Wertklasse fuer JSON aus Laeufen vor dieser Spalte (wie
    14)."""
    ds = C.D.load_split(verbose=False)
    df = ds.df.reset_index(drop=True)
    X = ds.X_raw.reset_index(drop=True)
    y = ds.y.reset_index(drop=True).to_numpy(float)
    y_art = S14.gruppenzentriert(y, df[SP["artikel"]].astype(str).to_numpy())
    y_auf = S14.gruppenzentriert(y, df[SP["auftrag"]].astype(str).to_numpy())
    kategorial = SW.set_index("Merkmal").kategorial
    KL = KL.copy()
    KL["y_auftragsbereinigt"] = np.nan
    for c, idx in KL.groupby("Merkmal").groups.items():
        if kategorial[c]:
            k = kategorien(X[c])
            masken = {i: np.isin(k, KL.at[i, "kategorien"].split(", ")) for i in idx}
        else:
            v = pd.to_numeric(X[c], errors="coerce").to_numpy(float)
            ok = ~np.isnan(v)
            kl = S14.klassen(v[ok])
            masken = {}
            for i in idx:
                m = np.zeros(len(v), dtype=bool)
                m[ok] = kl == KL.at[i, "klasse"] - 1
                masken[i] = m
        for i, mk in masken.items():
            if mk.sum() != KL.at[i, "n"] or not np.isclose(
                    y_art[mk].mean(), KL.at[i, "y_artikelbereinigt"]):
                raise SystemExit(f"{c} Klasse {KL.at[i, 'klasse']}: passt nicht zum "
                                 "gespeicherten Lauf — Skript ohne --nur-pdf neu rechnen")
            KL.at[i, "y_auftragsbereinigt"] = float(y_auf[mk].mean())
    return KL


def nur_pdf():
    """PDF aus dem gespeicherten JSON neu zeichnen, ohne Neuberechnung."""
    import json
    j = json.loads((C.ERG_DIR / name("15_stammdatenwerte.json")).read_text(encoding="utf-8"))
    fuss = j.get("fuss") or (f"{C.D.CSV_PATH.name} | {j['n_falten']} Falten, CV-Ebene "
                             f"{j['cv_level']} | OOF-TreeSHAP XGBoost {j['params_quelle']} | KI: "
                             f"Cluster-Bootstrap über Artikel ({S14.N_BOOT})")
    SW, KL = pd.DataFrame(j["schwellen"]), pd.DataFrame(j["klassen"])
    if "y_auftragsbereinigt" not in KL.columns:
        KL = auftragsreihe_nachtragen(KL, SW)
    pdf_schreiben(SW, KL, j["mae_mittel"], j["mae_median_mittel"], fuss)
    print("geschrieben:", C.OUT_DIR / name("15_stammdatenwerte.pdf"))


if __name__ == "__main__":
    import sys
    nur_pdf() if "--nur-pdf" in sys.argv else main()
