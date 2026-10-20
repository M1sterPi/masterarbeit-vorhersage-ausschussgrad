"""24 — Merkmale gegen y auf drei Ebenen.

In der Arbeit: Abschnitt 4.2.6: Tabelle 4 und die Zahl der Merkmale mit Bezug
zu y auf drei Ebenen, Rohstoffcharge und -analyse innerhalb der Gruppen.

Zusammenhang mit y gesamt, innerhalb der Artikel und innerhalb der Aufträge.
Ausgabe: runs/eda/24_merkmalsebenen.json, reports/eda/24_merkmalsebenen.csv,
reports/figures/merkmale_drei_ebenen.png.

    python eda/24_merkmalsebenen.py
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

import _common as C
from data_prep import SP  # noqa: E402

D = C.D

JSON_OUT = C.ROOT / "runs" / "eda" / "24_merkmalsebenen.json"
CSV_OUT = C.ROOT / "reports" / "eda" / "24_merkmalsebenen.csv"
FIG_OUT = C.ROOT / "reports" / "figures" / "merkmale_drei_ebenen.png"
ALPHA = 0.05
TOP = 10
GLEICH = 0.99
ELEMENTE = ["Al", "Na", "K", "Li", "Ca", "Ti", "Fe", "Zr"]


def klasse(r) -> str:
    if r.q_y >= ALPHA:
        return "nicht signifikant"
    if r.q_auf < ALPHA:
        return "bis Billet" if np.sign(r.rho_auf) == np.sign(r.rho_y) else "kehrt sich um"
    if r.q_art < ALPHA:
        return "bis Auftrag" if np.sign(r.rho_art) == np.sign(r.rho_y) else "kehrt sich um"
    return "verschwindet"


def merkmalstabelle(X, y, df) -> pd.DataFrame:
    auftrag = df[SP["auftrag"]]
    artikel = df[SP["artikel"]].astype(str)
    num = C.numerische_merkmale(X)
    t_y = C.spearman_tabelle(X, y, num).set_index("merkmal")
    w_art = C.within_tabelle(X, y, artikel, num).set_index("merkmal")
    w_auf = C.within_tabelle(X, y, auftrag, num).set_index("merkmal")
    influx = set(D.influx_spalten(X.columns))
    T = pd.DataFrame({
        "quelle": ["Prozess" if c in influx else "Stamm" for c in t_y.index],
        "rho_y": t_y.rho, "q_y": t_y.q, "n": t_y.n,
        "rho_art": w_art.rho.reindex(t_y.index), "q_art": w_art.q.reindex(t_y.index),
        "konst_art": w_art.in_konstant.reindex(t_y.index).fillna(True).astype(bool),
        "rho_auf": w_auf.rho.reindex(t_y.index), "q_auf": w_auf.q.reindex(t_y.index),
        "konst_auf": w_auf.in_konstant.reindex(t_y.index).fillna(True).astype(bool),
    }, index=t_y.index)
    T["klasse"] = T.apply(klasse, axis=1)
    T["klasse_03"] = np.select(
        [T.q_y >= ALPHA, T.q_auf < ALPHA, T.q_art < ALPHA],
        ["nicht signifikant", "Billet", "Auftrag"], "Artikel")
    T = T.reindex(T.rho_y.abs().sort_values(ascending=False).index)
    T["rang_y"] = np.arange(1, len(T) + 1)
    return T


def charge_und_chemie(X, y, df) -> dict:
    auftrag = df[SP["auftrag"]]
    artikel = df[SP["artikel"]].astype(str)
    dev = C.auftragsabweichung(y, auftrag)
    charge = df[SP["rohstoff_charge_anode"]].astype(str).replace("nan", np.nan)
    groesse = df.groupby(SP["auftrag"]).size()
    mehr = groesse[groesse > 1].index
    je_auftrag = df.groupby(SP["auftrag"])[SP["rohstoff_charge_anode"]].nunique()
    mit_charge = df[charge.notna()].groupby(SP["auftrag"]).size()
    mehr_mit = mit_charge[mit_charge > 1].index
    ok = charge.notna() & auftrag.isin(mehr)
    r_dev = C.var_anteil_mit_null(dev[ok], charge[ok], n_perm=200)

    spalten = [f"chargeA__{el}" for el in ELEMENTE if f"chargeA__{el}" in X.columns]
    al = C.als_zahl(X["chargeA__Al"])
    gut = al.notna() & (al > 0)
    Xc, yc = X.loc[gut, spalten], y[gut]
    w_art = C.within_tabelle(Xc, yc, artikel[gut], spalten).set_index("merkmal")
    w_auf = C.within_tabelle(Xc, yc, auftrag[gut], spalten).set_index("merkmal")
    return {
        "auftraege_mehrere_billets_mit_charge": int(len(mehr_mit)),
        "anteil_mehrere_chargen": float((je_auftrag[mehr_mit] > 1).mean()),
        "charge_an_auftragsabweichung": r_dev,
        "chemie_billets": int(gut.sum()),
        "chemie_artikel_billets_in_gruppen": int(w_art.n.max()),
        "chemie_artikel_gruppen": int(w_art.k.max()),
        "chemie_innerhalb_artikel": w_art[["rho", "q", "n", "k"]].round(4).to_dict("index"),
        "chemie_innerhalb_auftrag": w_auf[["rho", "q", "n", "k"]].round(4).to_dict("index"),
    }


def abbrueche(X, y) -> dict:
    pm = C.als_zahl(X[SP["phase"] + "_max"])
    ab = pm < SP["letzte_phase"]
    voll = y >= 100
    return {
        "mit_phase": int(pm.notna().sum()),
        "abbruch": int(ab.sum()),
        "y_median_abbruch": float(y[ab].median()),
        "y_median_rest": float(y[pm.notna() & ~ab].median()),
        "totalausfaelle": int(voll.sum()),
        "totalausfaelle_abbruch": int((voll & ab).sum()),
    }


def tabelle_ohne_abbrueche(X, y, df) -> pd.DataFrame:
    """Gegenprobe: Einteilung ohne die abgebrochenen Schmelzen (letzte Phase nicht erreicht)."""
    ok = ~(C.als_zahl(X[SP["phase"] + "_max"]) < SP["letzte_phase"])
    return merkmalstabelle(X[ok], y[ok], df[ok]).assign(billets=int(ok.sum()))


def ohne_abbrueche(T: pd.DataFrame, billet_merkmale: list[str]) -> dict:
    sig = T[T.klasse != "nicht signifikant"]
    return {
        "billets": int(T.billets.iloc[0]),
        "signifikant": int(len(sig)),
        "klassen": sig.klasse.value_counts().to_dict(),
        "bis_billet": sig[sig.klasse == "bis Billet"].index.tolist(),
        "vorherige_billet_merkmale": T.loc[billet_merkmale, ["rho_y", "rho_art", "q_art", "rho_auf", "q_auf",
                                                              "klasse"]].round(4).to_dict("index"),
    }


def billetlaenge(X, df) -> dict:
    lb = C.als_zahl(X[SP["laenge_billet"]])
    soll = C.als_zahl(df[SP["laenge_soll"]]) * 1000
    ok = lb.notna() & soll.notna()
    variiert = df[ok].assign(lb=lb[ok]).groupby(SP["auftrag"]).lb.nunique()
    return {
        "billets": int(ok.sum()),
        "gleich_soll": float((lb[ok] == soll[ok]).mean()),
        "kuerzer": float((lb[ok] < soll[ok]).mean()),
        "laenger": float((lb[ok] > soll[ok]).mean()),
        "auftraege_mit_wechsel": float((variiert > 1).mean()),
    }


def kanal_a(T: pd.DataFrame, X) -> dict:
    out = {}
    for c in (SP["kanal_a"] + "_n_samples", SP["kanal_a"] + "_n_edges"):
        if c in T.index:
            s = C.als_zahl(X[c])
            out[c] = {**T.loc[c, ["rho_y", "q_y", "rho_art", "q_art", "rho_auf", "q_auf", "klasse"]].to_dict(),
                      "p95": float(s.quantile(0.95))}
    return out


EBENEN = {
    "y": ("über alle Billets", "#3d3d3d", "o"),
    "art": ("innerhalb der Artikel", "#3182bd", "s"),
    "auf": ("innerhalb der Aufträge", "#d95f02", "D"),
}
AGG = {"_n_samples": "Zahl der Messpunkte", "_known_frac": "abgedeckter Anteil", "_frac_on_t": "Zeitanteil",
       "_n_edges": "Zahl der Wechsel", "_mean": "Mittelwert", "_std": "Streuung", "_min": "Minimum",
       "_max": "Maximum", "_last": "letzter Wert", "_range": "Spannweite"}
KANAL = C.D.beschriftungen("eda24_kanal")
STAMM = C.D.beschriftungen("eda24_stamm")
TEXTE = C.D.beschriftungen("eda24_texte")


def _zerlege(name: str) -> tuple[str, str]:
    for s in AGG:
        if name.endswith(s):
            return name[: -len(s)], s
    return name, ""


def _name(gruppe: list[str]) -> str:
    """Zeilenname: gemeinsamer Kanalpräfix mit den Endungen (z. B. X3/4/7_n_samples),
    ein Kanal mit mehreren Aggregaten (X_max/range), sonst das Merkmal selbst."""
    if len(gruppe) == 1:
        return gruppe[0]
    kan, agg = zip(*map(_zerlege, sorted(gruppe)))
    if len(set(agg)) == 1 and all(agg):
        if all(k.startswith(SP["lichtbogen_praefix"]) for k in kan):
            return SP["lichtbogen_praefix"] + "*" + agg[0]
        p = os.path.commonprefix(kan)
        return p + "/".join(k[len(p):] for k in kan) + agg[0]
    if len(set(kan)) == 1:
        return kan[0] + "_" + "/".join(a[1:] for a in agg)
    return " / ".join(gruppe)


def _beschreibung(gruppe: list[str]) -> str:
    gruppe = sorted(gruppe)
    kan, agg = zip(*map(_zerlege, gruppe))
    if not agg[0]:
        return " / ".join(STAMM.get(g, "") for g in gruppe)
    if len(set(agg)) == 1:
        if all(k.startswith(SP["praefix_b"]) for k in kan) and len(kan) > 1:
            kanal = TEXTE.get("praefix_b", " / ".join(kan)).format(", ".join(k[len(SP["praefix_b"]):] for k in kan))
        elif all(k.startswith(SP["lichtbogen_praefix"]) for k in kan) and len(kan) > 1:
            kanal = TEXTE.get("lichtbogen", " / ".join(kan))
        else:
            kanal = " / ".join(KANAL.get(k, k) for k in kan)
        return f"{kanal}, {AGG[agg[0]]}"
    return f"{KANAL.get(kan[0], kan[0])}, " + " / ".join(AGG[a] for a in agg)


def top_je_ebene(T: pd.DataFrame, X: pd.DataFrame, ebene: str) -> list[dict]:
    """Die TOP stärksten Zusammenhänge der Ebene; gleiche Merkmale teilen sich eine
    Zeile."""
    rangfolge = T[f"rho_{ebene}"].dropna().abs().sort_values(ascending=False).index
    raenge = {}

    def rang(c):
        if c not in raenge:
            raenge[c] = C.als_zahl(X[c]).rank()
        return raenge[c]

    zeilen = []
    for c in rangfolge:
        for z in zeilen:
            kopf = z["merkmale"][0]
            if abs(rang(c).corr(rang(kopf))) >= GLEICH:
                z["merkmale"].append(c)
                break
        else:
            if len(zeilen) == TOP:
                break
            zeilen.append({"merkmale": [c]})
    for z in zeilen:
        kopf = z["merkmale"][0]
        z.update(T.loc[kopf, ["quelle", "rang_y", "rho_y", "q_y", "rho_art", "q_art", "konst_art", "rho_auf",
                              "q_auf", "konst_auf", "klasse", "rho_art_oa", "q_art_oa", "rho_auf_oa",
                              "q_auf_oa", "rho_art_masch", "q_art_masch"]].to_dict())
        z["name"], z["beschreibung"] = _name(z["merkmale"]), _beschreibung(z["merkmale"])
        z["nur_abbruch"] = bool(ebene != "y" and z[f"q_{ebene}"] < ALPHA and not (
            z[f"q_{ebene}_oa"] < ALPHA and np.sign(z[f"rho_{ebene}_oa"]) == np.sign(z[f"rho_{ebene}"])))
        z["nur_maschine"] = bool(ebene == "art" and z["q_art"] < ALPHA and not (
            z["q_art_masch"] < ALPHA and np.sign(z["rho_art_masch"]) == np.sign(z["rho_art"])))
    return zeilen


def abb_drei_ebenen(tops: dict, abbruch: int):
    titel = {"y": "a) Die zehn stärksten Zusammenhänge über alle Billets",
             "art": "b) Die zehn stärksten Zusammenhänge innerhalb der Artikel",
             "auf": "c) Die zehn stärksten Zusammenhänge innerhalb der Aufträge"}
    versatz = {"y": -0.27, "art": 0.0, "auf": 0.27}
    fig, axe = plt.subplots(3, 1, figsize=(6.3, 8.8), sharex=True)
    fig.subplots_adjust(left=0.315, right=0.635, top=0.89, bottom=0.055, hspace=0.32)
    for ax, (ebene, zeilen) in zip(axe, tops.items()):
        tr = blended_transform_factory(ax.transAxes, ax.transData)
        for i, z in enumerate(zeilen):
            pfad = []
            for e, (_, farbe, marker) in EBENEN.items():
                r, q = z[f"rho_{e}"], z[f"q_{e}"]
                if pd.isna(r):
                    continue
                pfad.append((r, i + versatz[e]))
                sig = q < ALPHA
                ax.plot(r, i + versatz[e], marker=marker, ls="", ms=5.6 if e == ebene else 4.2, zorder=3,
                        color=farbe, markerfacecolor=farbe if sig else "white", markeredgewidth=0.9)
            ax.plot(*zip(*pfad), color="#c8c8c8", lw=0.9, zorder=1)
            ax.text(-0.03, i, z["name"] + (" †" if z["nur_abbruch"] else "") + (" ‡" if z["nur_maschine"] else ""),
                    transform=tr,
                    ha="right", va="center", fontsize=7)
            fest = ("je Artikel fest" if z["konst_art"] else "je Auftrag fest" if z["konst_auf"] else "")
            ax.text(1.03, i, z["beschreibung"] + (f" ({fest})" if fest else ""), transform=tr,
                    ha="left", va="center", fontsize=6.3, color="#555555")
        ax.set_ylim(len(zeilen) - 0.45, -0.6)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.axvline(0, color="#7f7f7f", lw=0.8, zorder=0)
        ax.set_xlim(-0.27, 0.27)
        ax.set_xticks([-0.2, -0.1, 0, 0.1, 0.2])
        ax.tick_params(axis="x", labelsize=7, labelbottom=True)
        ax.grid(axis="x", color="#e6e6e6", lw=0.6)
        ax.set_axisbelow(True)
        ax.text(-0.98, -1.05, titel[ebene], transform=blended_transform_factory(ax.transAxes, ax.transData),
                ha="left", va="bottom", fontsize=8, fontweight="bold")
    axe[-1].set_xlabel("Spearman-ρ mit dem Ausschussgrad y", fontsize=7.5)
    griffe = [Line2D([], [], marker=m, ls="", color=f, markerfacecolor=f, ms=5, label=t)
              for t, f, m in EBENEN.values()]
    fig.legend(handles=griffe, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3, fontsize=7.5,
               handletextpad=0.3, columnspacing=1.6)
    fig.text(0.5, 0.962, "gefüllt: signifikant (q < 0,05) · leer: nicht signifikant · großer Marker: Ebene, "
             "nach der das Teilbild sortiert ist", ha="center", fontsize=6.8, color="#555555")
    fig.text(0.5, 0.948, f"† ohne die {abbruch} abgebrochenen Schmelzen auf der Ebene des Teilbilds nicht mehr "
             "signifikant", ha="center", fontsize=6.8, color="#555555")
    fig.text(0.5, 0.934, "‡ innerhalb von Artikel und Maschine nicht mehr signifikant", ha="center", fontsize=6.8,
             color="#555555")
    C.aufraeumen(fig)
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ds = C.lade(alle_arten=True)
    X, y, df = C.daten_teil(ds)
    T = merkmalstabelle(X, y, df)
    T_oa = tabelle_ohne_abbrueche(X, y, df)
    T = T.join(T_oa[["rho_y", "q_y", "rho_art", "q_art", "rho_auf", "q_auf", "klasse"]].add_suffix("_oa"))
    maschine = (X if SP["ofen_kennung"] in X.columns else df)[SP["ofen_kennung"]].astype(str)
    w_am = C.within_tabelle(X, y, df[SP["artikel"]].astype(str) + "|" + maschine,
                            T.index.tolist()).set_index("merkmal")
    T["rho_art_masch"], T["q_art_masch"] = w_am.rho.reindex(T.index), w_am.q.reindex(T.index)
    sig = T[T.klasse != "nicht signifikant"]
    innen = (T.q_art < ALPHA) | (T.q_auf < ALPHA)
    verdeckt = T[(T.q_y >= ALPHA) & innen]
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    T.round(4).to_csv(CSV_OUT, sep=";", decimal=",", encoding="utf-8-sig")
    tops = {e: top_je_ebene(T, X, e) for e in EBENEN}
    ab = abbrueche(X, y)
    abb_drei_ebenen(tops, ab["abbruch"])
    spalten_innen = ["quelle", "rho_y", "q_y", "rho_art", "q_art", "rho_auf", "q_auf", "klasse",
                     "rho_art_oa", "q_art_oa", "rho_auf_oa", "q_auf_oa", "rho_art_masch", "q_art_masch"]

    erg = {
        "merkmale": int(len(T)),
        "signifikant": int(len(sig)),
        "klassen": sig.klasse.value_counts().to_dict(),
        "klassen_nach_quelle": pd.crosstab(sig.klasse, sig.quelle).to_dict("index"),
        "klassen_03": sig.klasse_03.value_counts().to_dict(),
        "max_abs_rho_innerhalb_artikel": float(T.rho_art.abs().max()),
        "max_abs_rho_innerhalb_auftrag": float(T.rho_auf.abs().max()),
        "listen": {k: sig[sig.klasse == k][["quelle", "rang_y", "rho_y", "rho_art", "q_art", "rho_auf", "q_auf"]]
                   .round(4).to_dict("index")
                   for k in ("bis Billet", "bis Auftrag", "kehrt sich um")},
        "top10": T.head(10)[["quelle", "rho_y", "q_y", "rho_art", "q_art", "konst_art", "rho_auf", "q_auf",
                             "konst_auf", "klasse"]].round(4).to_dict("index"),
        "signifikant_innerhalb_artikel": int((T.q_art < ALPHA).sum()),
        "signifikant_innerhalb_auftrag": int((T.q_auf < ALPHA).sum()),
        "signifikant_ohne_abbrueche": {"innerhalb_artikel": int((T.q_art_oa < ALPHA).sum()),
                                       "innerhalb_auftrag": int((T.q_auf_oa < ALPHA).sum())},
        "maschine": {"fehlt": int((maschine == "nan").sum()),
                     "artikel_mit_mehreren": int(df.assign(m=maschine).groupby(SP["artikel"]).m.nunique()
                                                 .gt(1).sum()),
                     "artikel": int(df[SP["artikel"]].nunique()),
                     "signifikant_innerhalb_artikel_und_maschine": int((T.q_art_masch < ALPHA).sum()),
                     "von_innerhalb_artikel_erhalten": int(((T.q_art < ALPHA) & (T.q_art_masch < ALPHA) & (
                         np.sign(T.rho_art) == np.sign(T.rho_art_masch))).sum())},
        "innerhalb_auftrag": T[T.q_auf < ALPHA][spalten_innen].round(4).to_dict("index"),
        "verdeckt": {"anzahl": int(len(verdeckt)),
                     "nach_quelle": verdeckt.quelle.value_counts().to_dict(),
                     "merkmale": verdeckt[spalten_innen].round(4).to_dict("index")},
        "top_je_ebene": tops,
        "charge_chemie": charge_und_chemie(X, y, df),
        "abbrueche": ab,
        "ohne_abbrueche": ohne_abbrueche(T_oa, sig[sig.klasse == "bis Billet"].index.tolist()),
        "billetlaenge": billetlaenge(X, df),
        "kanal_a": kanal_a(T, X),
    }
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(erg, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    print(f"{erg['signifikant']} von {erg['merkmale']} Merkmalen signifikant mit y")
    print("Klassen neu:", erg["klassen"])
    print("Klassen alt (03):", erg["klassen_03"])
    print("nach Quelle:", erg["klassen_nach_quelle"])
    for k, v in erg["listen"].items():
        print(f"\n{k}:")
        print(pd.DataFrame(v).T.to_string())
    print("\nTop 10:")
    print(pd.DataFrame(erg["top10"]).T.to_string())
    cc = erg["charge_chemie"]
    print("\nCharge:", {k: v for k, v in cc.items() if not k.startswith("chemie_innerhalb")})
    print("Chemie innerhalb Artikel:\n", pd.DataFrame(cc["chemie_innerhalb_artikel"]).T.to_string())
    print("Chemie innerhalb Auftrag:\n", pd.DataFrame(cc["chemie_innerhalb_auftrag"]).T.to_string())
    print("\nAbbrüche:", erg["abbrueche"])
    oa = erg["ohne_abbrueche"]
    print("Ohne Abbrüche:", {k: oa[k] for k in ("billets", "signifikant", "klassen", "bis_billet")})
    print(pd.DataFrame(oa["vorherige_billet_merkmale"]).T.to_string())
    print("Billetlänge:", erg["billetlaenge"])
    print("kanal_a:", json.dumps(erg["kanal_a"], ensure_ascii=False, default=float))
    print("max |rho| innerhalb Artikel / Auftrag:", round(erg["max_abs_rho_innerhalb_artikel"], 3),
          round(erg["max_abs_rho_innerhalb_auftrag"], 3))
    print("verdeckt:", erg["verdeckt"]["anzahl"], erg["verdeckt"]["nach_quelle"])
    for e, zeilen in tops.items():
        print(f"\nTop {TOP} {EBENEN[e][0]}:")
        print(pd.DataFrame(zeilen).set_index("name")[["rho_y", "q_y", "rho_art", "q_art", "rho_auf", "q_auf",
                                                       "rho_art_oa", "rho_auf_oa", "q_auf_oa", "nur_abbruch",
                                                       "q_art_masch", "nur_maschine", "klasse"]].round(3).to_string())
    print("Maschine:", erg["maschine"], "| ohne Abbrüche signifikant:", erg["signifikant_ohne_abbrueche"])
    print("geschrieben:", JSON_OUT, CSV_OUT, FIG_OUT)


if __name__ == "__main__":
    main()
