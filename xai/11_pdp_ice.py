"""11 — Partial Dependence, ICE und ALE.

In der Arbeit: nicht berichtet; Wirkungsform bei TabPFN und XGBoost.

TabPFN gegen XGBoost. Optionen --nur-plot und --merkmale.

    python xai/11_pdp_ice.py
"""

from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import _common as C

N_STICHPROBE = 300
RASTER = 20
QUANTILE = (0.05, 0.95)
KAT_GRENZE = 12
TOP_JE_MODELL = 6
N_ICE_LINIEN = 60
BLOCK = 4096
MAX_FEHLEND = 0.95
MIN_UEBERLAPPUNG = 100

FARBE = {"TabPFN": "#7b3294", "XGBoost": "#e66101"}


def erg_pfad():
    return C.ERG_DIR / C.mit_ebene("11_pdp_ice.json")


def modellparameter() -> dict:
    """Wie 05: Parameter aus tabpfn_regression.py plus Cache."""
    default, fix = C._konstanten_aus_skript("tabpfn_regression.py")
    return {**default, **fix, "fit_mode": "fit_with_cache"}


def waehle_merkmale(X_tr: pd.DataFrame) -> list[str]:
    """Top-k aus der XGBoost-CV-Rangliste (07) vereinigt mit Top-k aus 05."""
    spalten = list(X_tr.columns)
    fehlend = X_tr.isna().mean()
    gewaehlt: list[str] = []
    for datei in ("07_kreuzvalidierung_xgboost.csv", "05_tabpfn.csv"):
        p = C.TAB_DIR / C.mit_ebene(datei)
        if not p.exists():
            print(f"  Hinweis: {p.name} fehlt, Rangliste entfaellt.")
            continue
        df = pd.read_csv(p)
        if "konstant_im_train" in df.columns:
            df = df[~df.konstant_im_train.astype(bool)]
        leer = [m for m in df.Merkmal[: 3 * TOP_JE_MODELL]
                if m in spalten and fehlend[m] > MAX_FEHLEND]
        if leer:
            print(f"  {p.name}: uebersprungen wegen > {MAX_FEHLEND:.0%} Fehlwerten "
                  f"im Training: {leer}")
        top = [m for m in df.Merkmal
               if m in spalten and fehlend[m] <= MAX_FEHLEND][:TOP_JE_MODELL]
        gewaehlt += [m for m in top if m not in gewaehlt]
    if not gewaehlt:
        raise SystemExit(
            "Keine Rangliste gefunden — erst 07 (und moeglichst 05) laufen "
            "lassen oder --merkmale angeben."
        )
    return gewaehlt


def raster(x_tr: pd.Series) -> tuple[np.ndarray, bool]:
    """(Rasterwerte, kategorial?) aus der Trainingsverteilung."""
    v = pd.to_numeric(x_tr, errors="coerce").dropna().to_numpy(dtype=float)
    stufen = np.unique(v)
    if len(stufen) <= KAT_GRENZE:
        return stufen, True
    q = np.quantile(v, np.linspace(*QUANTILE, RASTER))
    return np.unique(q), False


def ale_grenzen(x_tr: pd.Series) -> np.ndarray:
    """Quantilgrenzen fuer ALE auf DEMSELBEN Bereich wie der PDP (QUANTILE)."""
    v = pd.to_numeric(x_tr, errors="coerce").dropna().to_numpy(dtype=float)
    return np.unique(np.quantile(v, np.linspace(*QUANTILE, RASTER + 1)))


def baue_zeilen(Xs: np.ndarray, j: int, gitter: np.ndarray, grenzen):
    """Alle Abfragezeilen eines Merkmals in EINER Matrix."""
    n = len(Xs)
    teile = []
    for z in gitter:
        B = Xs.copy()
        B[:, j] = z
        teile.append(B)
    ale = None
    if grenzen is not None:
        x = Xs[:, j]
        ok = np.flatnonzero(np.isfinite(x) & (x >= grenzen[0]) & (x <= grenzen[-1]))
        k = np.clip(np.searchsorted(grenzen, x[ok], side="left"), 1, len(grenzen) - 1)
        unten, oben = Xs[ok].copy(), Xs[ok].copy()
        unten[:, j] = grenzen[k - 1]
        oben[:, j] = grenzen[k]
        teile += [unten, oben]
        ale = (ok, k)
    return np.vstack(teile), n, ale


def zerlege(pred: np.ndarray, n: int, G: int, ale, grenzen) -> dict:
    ice = pred[: G * n].reshape(G, n)
    out = {"ice": ice}
    if ale is not None:
        ok, k = ale
        m = len(ok)
        d = pred[G * n + m : G * n + 2 * m] - pred[G * n : G * n + m]
        K = len(grenzen) - 1
        summe = np.bincount(k - 1, weights=d, minlength=K)
        anzahl = np.bincount(k - 1, minlength=K)
        schritt = np.divide(summe, anzahl, out=np.zeros(K), where=anzahl > 0)
        kurve = np.concatenate([[0.0], np.cumsum(schritt)])
        mitte = 0.5 * (kurve[1:] + kurve[:-1])
        kurve -= np.sum(mitte * anzahl) / max(anzahl.sum(), 1)
        out["ale"] = kurve
        out["ale_n"] = anzahl
    return out


def vorhersage_blockweise(modell, A: np.ndarray) -> np.ndarray:
    teile = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for s in range(0, len(A), BLOCK):
            teile.append(np.asarray(modell.predict(A[s : s + BLOCK]), dtype=float))
    return np.concatenate(teile)


def lade_stand() -> dict:
    leer = {
        "n_stichprobe": N_STICHPROBE,
        "raster": RASTER,
        "quantile": list(QUANTILE),
        "fit_mode": "fit_with_cache",
        "merkmale": {},
    }
    p = erg_pfad()
    if not p.exists():
        return leer
    d = json.loads(p.read_text(encoding="utf-8"))
    passt = (
        d.get("n_stichprobe") == N_STICHPROBE
        and d.get("raster") == RASTER
        and d.get("quantile") == list(QUANTILE)
    )
    if passt:
        return d
    print("  Zwischenstand passt nicht zur Einstellung — wird verworfen.")
    return leer


def r(a, nk=4):
    return np.round(np.asarray(a, dtype=float), nk).tolist()


def rechne(ds, X_tr, X_te, merkmale) -> dict:
    stand = lade_stand()
    offen = [m for m in merkmale if m not in stand["merkmale"]]
    print(f"{len(merkmale)} Merkmale geplant, {len(offen)} offen: {offen}")

    rng = np.random.default_rng(C.SEED)
    idx = np.sort(rng.choice(len(X_te), size=min(N_STICHPROBE, len(X_te)), replace=False))
    stand["stichprobe"] = idx.tolist()
    if not offen:
        return stand
    Xs = X_te.iloc[idx].to_numpy(dtype=float)

    xgb_m, quelle = C.xgb_modell(X_tr, ds.y_train)
    stand["xgb_quelle"] = quelle

    from tabpfn import TabPFNRegressor

    params = modellparameter()
    print(f"  TabPFN: {params}")
    t0 = time.time()
    tab_m = TabPFNRegressor(**params)
    tab_m.fit(X_tr, ds.y_train)
    print(f"  Fit mit KV-Cache: {time.time()-t0:.1f} s (einmalig)")
    stand["n_estimators"] = params.get("n_estimators")

    stand["basis_tabpfn"] = float(vorhersage_blockweise(tab_m, Xs).mean())
    stand["basis_xgboost"] = float(xgb_m.predict(Xs).mean())

    spalten = list(X_tr.columns)
    for k, name in enumerate(offen, 1):
        t0 = time.time()
        j = spalten.index(name)
        gitter, kat = raster(X_tr[name])
        grenzen = None if kat else ale_grenzen(X_tr[name])
        A, n, ale = baue_zeilen(Xs, j, gitter, grenzen)
        G = len(gitter)
        eintrag = {
            "kategorial": kat,
            "gitter": r(gitter, 6),
            "grenzen": None if grenzen is None else r(grenzen, 6),
            "anteil_fehlend_stichprobe": float(np.isnan(Xs[:, j]).mean()),
        }
        for modell_name, m in (("XGBoost", xgb_m), ("TabPFN", tab_m)):
            pred = (
                vorhersage_blockweise(m, A)
                if modell_name == "TabPFN"
                else np.asarray(m.predict(A), dtype=float)
            )
            z = zerlege(pred, n, G, ale, grenzen)
            eintrag[modell_name] = {
                "ice": r(z["ice"], 3),
                "ale": r(z["ale"]) if "ale" in z else None,
                "ale_n": z["ale_n"].tolist() if "ale_n" in z else None,
            }
        stand["merkmale"][name] = eintrag
        C.speichere(erg_pfad().name, stand)
        dt = time.time() - t0
        print(
            f"  {k:>2}/{len(offen)}  {name:<32} {len(A):>6} Zeilen  {dt:6.1f} s  "
            f"noch ~{(len(offen)-k)*dt/60:5.1f} min",
            flush=True,
        )
    return stand


def max_korrelation(X_tr: pd.DataFrame, merkmale) -> dict[str, tuple[float, str]]:
    """Staerkster Spearman-Partner je Merkmal — Warnsignal fuer den PDP."""
    R = X_tr.rank(na_option="keep")
    R = R.loc[:, R.std() > 0]
    bekannt = R.notna().astype(float)
    out = {}
    for m in merkmale:
        if m not in R.columns:
            out[m] = (float("nan"), "")
            continue
        ueber = bekannt.T @ bekannt[m]
        with np.errstate(invalid="ignore", divide="ignore"):
            c = R.corrwith(R[m]).drop(m)
        c = c[ueber.drop(m) >= MIN_UEBERLAPPUNG].abs().dropna()
        out[m] = (float(c.max()), str(c.idxmax())) if len(c) else (float("nan"), "")
    return out


def kennzahlen(stand, korr) -> pd.DataFrame:
    warnings.filterwarnings("ignore", message="An input array is constant")
    zeilen = []
    for name, e in stand["merkmale"].items():
        z = {"Merkmal": name, "herkunft": C.herkunft(name), "kategorial": e["kategorial"]}
        pdps = {}
        for mn in ("TabPFN", "XGBoost"):
            ice = np.asarray(e[mn]["ice"])
            pdp = ice.mean(axis=1)
            pdps[mn] = pdp
            spann = float(pdp.max() - pdp.min())
            z[f"pdp_spanne_{mn}"] = spann
            c = ice - ice[0]
            z[f"ice_streuung_{mn}"] = float(c[-1].std() / spann) if spann > 0 else np.nan
            if e[mn]["ale"] is not None:
                a = np.asarray(e[mn]["ale"])
                z[f"ale_spanne_{mn}"] = float(a.max() - a.min())
        if len(pdps["TabPFN"]) >= 3:
            z["rho_pdp"] = float(spearmanr(pdps["TabPFN"], pdps["XGBoost"])[0])
        if e["TabPFN"]["ale"] is not None:
            z["rho_ale"] = float(
                spearmanr(e["TabPFN"]["ale"], e["XGBoost"]["ale"])[0]
            )
            for mn in ("TabPFN", "XGBoost"):
                g, gr = np.asarray(e["gitter"]), np.asarray(e["grenzen"])
                im = (gr >= g.min()) & (gr <= g.max())
                if im.sum() >= 3:
                    p = np.interp(gr[im], g, pdps[mn])
                    a = np.asarray(e[mn]["ale"])[im]
                    sp = a.max() - a.min()
                    z[f"pdp_vs_ale_{mn}"] = (
                        float(np.abs((p - p.mean()) - (a - a.mean())).mean() / sp)
                        if sp > 0
                        else np.nan
                    )
        z["anteil_fehlend"] = e["anteil_fehlend_stichprobe"]
        z["max_rho"], z["partner"] = korr.get(name, (np.nan, ""))
        zeilen.append(z)
    return pd.DataFrame(zeilen)


def _achse_merkmal(ax, e, x_tr):
    if e["kategorial"]:
        g = e["gitter"]
        ax.set_xticks(g)
        ax.set_xticklabels([f"{v:g}" for v in g], fontsize=5, rotation=90)
    else:
        v = pd.to_numeric(x_tr, errors="coerce").dropna().to_numpy(dtype=float)
        lo, hi = e["gitter"][0], e["gitter"][-1]
        v = v[(v >= lo) & (v <= hi)]
        ax.plot(v, np.zeros_like(v), "|", color="0.5", ms=4, alpha=0.25,
                transform=ax.get_xaxis_transform())
    ax.tick_params(labelsize=5.5)
    ax.grid(alpha=0.25)
    ax.axhline(0, color="black", lw=0.6)


def _raster_seite(n):
    spalten = 4 if n > 9 else 3
    return int(np.ceil(n / spalten)), spalten


def seite_pdp(pdf, stand, X_tr, kz, fuss):
    namen = list(stand["merkmale"])
    fig, axes = C.seite(
        "PDP: mittlerer Effekt, TabPFN gegen XGBoost (je Modell auf Mittel zentriert)",
        *_raster_seite(len(namen)),
    )
    for ax, name in zip(axes, namen):
        e = stand["merkmale"][name]
        g = np.asarray(e["gitter"])
        for mn in ("TabPFN", "XGBoost"):
            pdp = np.asarray(e[mn]["ice"]).mean(axis=1)
            stil = dict(marker="o", ms=3, ls="-" if not e["kategorial"] else ":")
            ax.plot(g, pdp - pdp.mean(), color=FARBE[mn], lw=1.4, label=mn, **stil)
        _achse_merkmal(ax, e, X_tr[name])
        zk = kz.set_index("Merkmal").loc[name]
        rho = zk.get("rho_pdp", np.nan)
        ax.set_title(
            f"{name} ({C.herkunft(name)})\nρ(PDP) {rho:+.2f} | Spanne T "
            f"{zk.pdp_spanne_TabPFN:.2f} / X {zk.pdp_spanne_XGBoost:.2f} pp",
            fontsize=6.8,
        )
    axes[0].legend(fontsize=6)
    for ax in axes[len(namen):]:
        ax.axis("off")
    C.schliesse(pdf, fig, fuss)


def seite_ice(pdf, stand, X_tr, mn, fuss):
    namen = list(stand["merkmale"])
    fig, axes = C.seite(
        f"ICE {mn}: eine Kurve je Billet, am ersten Rasterpunkt zentriert "
        f"({N_ICE_LINIEN} von {N_STICHPROBE} gezeigt), schwarz = PDP",
        *_raster_seite(len(namen)),
    )
    rng = np.random.default_rng(C.SEED)
    for ax, name in zip(axes, namen):
        e = stand["merkmale"][name]
        g = np.asarray(e["gitter"])
        ice = np.asarray(e[mn]["ice"])
        c = ice - ice[0]
        wahl = rng.choice(c.shape[1], size=min(N_ICE_LINIEN, c.shape[1]), replace=False)
        ax.plot(g, c[:, wahl], color=FARBE[mn], lw=0.5, alpha=0.3)
        ax.plot(g, c.mean(axis=1), color="black", lw=1.6)
        _achse_merkmal(ax, e, X_tr[name])
        ax.set_title(name, fontsize=6.8)
    for ax in axes[len(namen):]:
        ax.axis("off")
    C.schliesse(pdf, fig, fuss)


def seite_ale(pdf, stand, X_tr, kz, fuss):
    namen = [n for n, e in stand["merkmale"].items() if not e["kategorial"]]
    if not namen:
        return
    fig, axes = C.seite(
        "ALE: lokaler Effekt ohne Extrapolation, TabPFN gegen XGBoost",
        *_raster_seite(len(namen)),
    )
    for ax, name in zip(axes, namen):
        e = stand["merkmale"][name]
        gr = np.asarray(e["grenzen"])
        for mn in ("TabPFN", "XGBoost"):
            ax.plot(gr, e[mn]["ale"], color=FARBE[mn], lw=1.4, marker=".", ms=3,
                    label=mn)
        _achse_merkmal(ax, {**e, "gitter": [gr[0], gr[-1]]}, X_tr[name])
        zk = kz.set_index("Merkmal").loc[name]
        ax.set_title(
            f"{name}\nρ(ALE) {zk.get('rho_ale', np.nan):+.2f} | PDP≠ALE T "
            f"{zk.get('pdp_vs_ale_TabPFN', np.nan):.2f} / X "
            f"{zk.get('pdp_vs_ale_XGBoost', np.nan):.2f}",
            fontsize=6.8,
        )
    axes[0].legend(fontsize=6)
    for ax in axes[len(namen):]:
        ax.axis("off")
    C.schliesse(pdf, fig, fuss)


def seite_tabelle(pdf, kz, fuss):
    import matplotlib.pyplot as plt

    spalten = [
        ("Merkmal", "Merkmal", "{}"),
        ("max_rho", "max |ρ|", "{:.2f}"),
        ("partner", "Partner", "{}"),
        ("anteil_fehlend", "fehlt", "{:.0%}"),
        ("pdp_spanne_TabPFN", "PDP T", "{:.2f}"),
        ("pdp_spanne_XGBoost", "PDP X", "{:.2f}"),
        ("rho_pdp", "ρ PDP", "{:+.2f}"),
        ("ale_spanne_TabPFN", "ALE T", "{:.2f}"),
        ("ale_spanne_XGBoost", "ALE X", "{:.2f}"),
        ("rho_ale", "ρ ALE", "{:+.2f}"),
        ("pdp_vs_ale_TabPFN", "PDP≠ALE T", "{:.2f}"),
        ("pdp_vs_ale_XGBoost", "PDP≠ALE X", "{:.2f}"),
        ("ice_streuung_TabPFN", "ICE T", "{:.2f}"),
        ("ice_streuung_XGBoost", "ICE X", "{:.2f}"),
    ]

    def f(v, form):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "–"
        return form.format(v)

    zellen = [[f(z.get(k), form) for k, _, form in spalten] for _, z in kz.iterrows()]
    fig = plt.figure(figsize=(11.7, 8.3))
    fig.suptitle("Kennzahlen je Merkmal (Spannen in Prozentpunkten)", fontsize=13)
    ax = fig.add_axes((0.02, 0.1, 0.96, 0.8))
    ax.axis("off")
    breiten = [0.17 if k in ("Merkmal", "partner") else 0.055 for k, _, _ in spalten]
    t = ax.table(cellText=zellen, colLabels=[s[1] for s in spalten], loc="upper center",
                 cellLoc="center", colWidths=breiten)
    t.auto_set_font_size(False)
    t.set_fontsize(6.5)
    t.scale(1, 1.4)
    fig.text(0.5, 0.012, fuss, ha="center", fontsize=7.5, style="italic")
    pdf.savefig(fig)
    plt.close(fig)


def plotte(ds, X_tr, X_te, stand):
    if not stand["merkmale"]:
        raise SystemExit("Zwischenstand leer — erst ohne --nur-plot rechnen.")
    korr = max_korrelation(X_tr, list(stand["merkmale"]))
    kz = kennzahlen(stand, korr)
    C.speichere_rangliste(C.mit_ebene("11_pdp_ice.csv"), kz, "pdp_spanne_TabPFN")

    fuss = (
        C.fusszeile(ds, X_te, stand.get("xgb_quelle", "?"))
        + f" | Stichprobe {len(stand.get('stichprobe', []))} | TabPFN n_estimators="
        f"{stand.get('n_estimators', '?')}, fit_with_cache"
    )
    with C.pdf_oeffnen(C.mit_ebene("11_pdp_ice.pdf")) as pdf:
        seite_pdp(pdf, stand, X_tr, kz, fuss)
        seite_ice(pdf, stand, X_tr, "TabPFN", fuss)
        seite_ice(pdf, stand, X_tr, "XGBoost", fuss)
        seite_ale(pdf, stand, X_tr, kz, fuss)
        seite_tabelle(pdf, kz, fuss)
        C.textseite(
            pdf,
            "Lesehinweise",
            [
                "PDP: Merkmal bei allen Stichproben-Billets auf den Rasterwert gesetzt, "
                "Vorhersagen gemittelt. Beide Modelle sind auf ihr eigenes Mittel "
                "zentriert, verglichen wird die FORM, nicht das Niveau.",
                "ρ(PDP)/ρ(ALE): Rangkorrelation der beiden Kurven ueber das Raster. "
                "+1 = gleiche Richtung und Reihenfolge; bei flachen Kurven (kleine "
                "Spanne) ist ρ bedeutungslos.",
                "ICE: zentrierte Einzelkurven. Parallele Kurven = additiver Effekt; "
                "faechernde oder kreuzende Kurven = Interaktion. Kennzahl 'ICE' = "
                "Streuung am Rasterende relativ zur PDP-Spanne.",
                "ALE: nur lokale Differenzen innerhalb von Quantilintervallen, daher "
                "ohne unrealistische Merkmalskombinationen. 'PDP≠ALE' = mittlere "
                "Abweichung der zentrierten Kurven relativ zur ALE-Spanne; Werte "
                "ueber ~0,3 heissen: der PDP extrapoliert, ALE ist massgeblich.",
                "max |ρ|/Partner: staerkste Spearman-Korrelation zu irgendeinem "
                "anderen Merkmal (Trainingsmenge). Hohe Werte erklaeren PDP≠ALE.",
                "Kategoriale Merkmale (≤ 12 Auspraegungen, ordinal kodiert): Punkte "
                "je Auspraegung, Verbindungslinie nur zur Orientierung; kein ALE, "
                "weil die Kodierung keine Ordnung traegt.",
                "Was das NICHT ist: keine Wichtigkeitsrangliste ueber alle Merkmale "
                "(dafuer 05/07/08) und kein Kausalnachweis — die Kurven beschreiben "
                "das Modell, nicht den Prozess.",
            ],
        )
    print(kz.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nur-plot", action="store_true")
    ap.add_argument("--merkmale", help="Kommagetrennt, ersetzt die Auswahl aus 05/07")
    args = ap.parse_args()

    ds, X_tr, X_te = C.lade()
    if args.merkmale:
        merkmale = [m.strip() for m in args.merkmale.split(",") if m.strip()]
        fehlt = [m for m in merkmale if m not in X_tr.columns]
        if fehlt:
            raise SystemExit(f"Unbekannte Merkmale: {fehlt}")
    else:
        merkmale = waehle_merkmale(X_tr)
    stand = lade_stand() if args.nur_plot else rechne(ds, X_tr, X_te, merkmale)
    if not args.nur_plot:
        stand = {**stand, "merkmale": {m: stand["merkmale"][m] for m in merkmale
                                       if m in stand["merkmale"]}}
    plotte(ds, X_tr, X_te, stand)


if __name__ == "__main__":
    main()
