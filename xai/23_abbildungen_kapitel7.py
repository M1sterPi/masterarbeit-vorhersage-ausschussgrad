"""23 — Abbildungen zu Kapitel 7.

In der Arbeit: Abbildungen 15, 16, 18, 19, 22, D.1, E.1 und E.2 (Kapitel 7 und
Anhang).

Aus den gespeicherten Ergebnissen von eda/19 und xai/02–16, ohne neue
Rechnung. Ausgabe: reports/figures/kapitel7/.

    python xai/23_abbildungen_kapitel7.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TAB = ROOT / "reports" / "xai" / "tabellen"
ERG = ROOT / "runs" / "xai"
ZIEL = ROOT / "reports" / "figures" / "kapitel7"
ZIEL.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "data_pipeline"))
import data_prep as D  # noqa: E402

import locale  # noqa: E402
from data_prep import SP  # noqa: E402

try:
    locale.setlocale(locale.LC_NUMERIC, "de_DE")
    plt.rcParams["axes.formatter.use_locale"] = True
except locale.Error:
    pass


def de(x, nk=2):
    return f"{x:.{nk}f}".replace(".", ",")


def _zahl(x):
    """Schwelle lesbar: Tausenderpunkt, Dezimalkomma, keine überflüssigen Nullen."""
    nk = 0 if abs(x) >= 100 else (2 if abs(x) >= 1 else 3)
    s = f"{x:,.{nk}f}".replace(",", "§").replace(".", ",").replace("§", ".")
    return s.rstrip("0").rstrip(",") if "," in s else s


plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans"})
FARBE = {"Kanal": "#d95f02", "Influx-Kopf": "#fdae6b", "Stammdaten": "#1f78b4"}

KLAR = D.beschriftungen("xai23_klar")


def speichern(fig, name):
    p = ZIEL / name
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("geschrieben:", p.relative_to(ROOT))


EDA19 = ROOT / "runs" / "eda" / "19_auftrag_vs_artikel.json"
SCHEMA_NAME = {"artikelrein": "artikelrein", "auftragsrein": "auftragsrein",
               "zufällig": "billetweise"}


def abb_ebenen():
    e = json.loads(EDA19.read_text(encoding="utf-8"))["schemata"]
    zeilen, gesamt = [], []
    for modell in ("XGBoost", "TabPFN"):
        for sch in ("artikelrein", "auftragsrein", "zufällig"):
            z = e[sch]["modelle"][modell]["zerlegung"]
            zeilen.append((modell, SCHEMA_NAME[sch], z["artikel"]["beitrag"],
                           z["auftrag"]["beitrag"], z["billet"]["beitrag"]))
            gesamt.append(z["gesamt"])
    for z, g in zip(zeilen, gesamt):
        print(f"  {z[0]:8s} {z[1]:12s} Artikel {z[2]:+.3f}  Auftrag {z[3]:+.3f}  "
              f"Billet {z[4]:+.3f}  R² {g:+.3f}")
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    farben = ["#1f78b4", "#a6cee3", "#d95f02"]
    namen = ["Artikel", "Auftrag im Artikel", "Billet im Auftrag"]
    x = np.arange(len(zeilen))
    for j in range(3):
        w = np.array([z[2 + j] for z in zeilen])
        pos = np.where(w > 0, w, 0)
        neg = np.where(w < 0, w, 0)
        unten_pos = np.array([sum(max(z[2 + k], 0) for k in range(j)) for z in zeilen])
        unten_neg = np.array([sum(min(z[2 + k], 0) for k in range(j)) for z in zeilen])
        ax.bar(x, pos, bottom=unten_pos, color=farben[j], label=namen[j], width=0.65)
        ax.bar(x, neg, bottom=unten_neg, color=farben[j], width=0.65)
    for xi, z, g in zip(x, zeilen, gesamt):
        ax.text(xi, sum(max(v, 0) for v in z[2:]) + 0.006, f"R² = {de(g, 3)}",
                ha="center", fontsize=7.5)
    ax.set_ylim(-0.04, 0.33)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(x, [f"{m}\n{v}" for m, v, *_ in zeilen], fontsize=8)
    ax.set_ylabel("Beitrag zum R² (s_k · R²_k)")
    ax.legend(fontsize=8, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.13))
    ax.grid(axis="y", alpha=0.3)
    speichern(fig, "abb_7_ebenen.png")


def abb_nutzen_sensor():
    fig, axe = plt.subplots(1, 2, figsize=(8.6, 5.2), sharey=True)
    a = pd.read_csv(TAB / "13_sensoren_artikel.csv")
    reihen = a.head(18).Gruppe.tolist()[::-1]
    for ax, ebene, titel in zip(axe, ("artikel", "auftrag"), ("artikelrein", "auftragsrein")):
        t = pd.read_csv(TAB / f"13_sensoren_{ebene}.csv").set_index("Gruppe").reindex(reihen)
        j = json.loads((ERG / f"13_sensoren_{ebene}.json").read_text(encoding="utf-8"))
        yy = np.arange(len(t))
        from matplotlib.colors import to_rgba
        ax.barh(yy, t.delta_mae, color=[to_rgba(FARBE[x], 0.95 if ok else 0.35)
                                        for x, ok in zip(t.art, t.ki_ueber_null)])
        ax.errorbar(t.delta_mae, yy, xerr=[t.delta_mae - t.ki_lo, t.ki_hi - t.delta_mae],
                    fmt="none", ecolor="black", lw=0.7, capsize=2)
        ax.axvline(0, color="black", lw=0.7)
        ax.axvline(j["rauschdecke"], color="#e31a1c", ls="--", lw=1)
        ax.set_title(f"{titel} (Vorsprung vor Median {de(j['mae_median_mittel'] - j['mae_mittel'])} PP)",
                     fontsize=9)
        ax.set_xlabel("ΔMAE bei gemeinsamer Permutation [PP]")
        ax.grid(axis="x", alpha=0.3)
    axe[0].set_yticks(np.arange(len(reihen)), [KLAR.get(g, g) for g in reihen], fontsize=8)
    from matplotlib.patches import Patch
    axe[1].legend(handles=[Patch(color=FARBE["Kanal"], label="Sensor (Influx-Kanal)"),
                           Patch(color=FARBE["Stammdaten"], label="Stammdatenblock"),
                           Patch(color="grey", alpha=0.4, label="KI schließt null ein")],
                  fontsize=7.5, frameon=False, loc="lower right")
    speichern(fig, "abb_7_nutzen_sensor.png")


EBENEN_TITEL = {"artikel": "artikelrein", "auftrag": "auftragsrein", "billet": "billetweise"}


def abb_nutzen_ebene(ebene):
    """Gleiche Zeilen (Tabelle 7.3) und gleiche x-Achse auf allen drei Ebenen,
    sortiert je Ebene nach deren ΔMAE (groesster Nutzen oben)."""
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    T = pd.read_csv(TAB / "tabelle_7_3_sensornutzen.csv", index_col=0)
    reihen = T[T.gesichert_irgendwo].index.tolist()[::-1]
    t = pd.read_csv(TAB / f"13_sensoren_{ebene}.csv").set_index("Gruppe").reindex(reihen)
    t = t.sort_values("delta_mae")
    reihen = t.index.tolist()
    j = json.loads((ERG / f"13_sensoren_{ebene}.json").read_text(encoding="utf-8"))
    xmax = max(T.loc[reihen, f"ki_hi_{e}"].max() for e in EBENEN_TITEL) * 1.05
    xmin = min(T.loc[reihen, f"ki_lo_{e}"].min() for e in EBENEN_TITEL) * 1.05
    fig, ax = plt.subplots(figsize=(6.3, 6.4))
    yy = np.arange(len(t))
    farbe = {**FARBE, "Influx-Kopf": "#6a3d9a"}
    ax.barh(yy, t.delta_mae, color=[farbe[x] for x in t.art])
    ax.errorbar(t.delta_mae, yy, xerr=[t.delta_mae - t.ki_lo, t.ki_hi - t.delta_mae],
                fmt="none", ecolor="black", lw=0.7, capsize=2)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_xlim(xmin, xmax)
    ax.set_yticks(yy, [KLAR.get(g, g) for g in reihen], fontsize=8)
    ax.set_title(f"{EBENEN_TITEL[ebene]} (MAE {de(j['mae_mittel'])}, Vorsprung vor Median "
                 f"{de(j['mae_median_mittel'] - j['mae_mittel'])} PP)", fontsize=9)
    ax.set_xlabel("ΔMAE bei gemeinsamer Permutation [PP]")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(handles=[Patch(color=farbe["Kanal"], label="Sensor (Influx-Kanal)"),
                       Patch(color=farbe["Influx-Kopf"], label="Kopfangaben Kurvendatei"),
                       Patch(color=farbe["Stammdaten"], label="Stammdatenblock"),
                       Line2D([], [], color="black", lw=0.7, marker="|", ms=6,
                              label="Fehlerbalken: 95-%-KI über Artikel")],
              fontsize=7.5, frameon=False, loc="lower right")
    speichern(fig, f"abb_7_nutzen_sensor_{ebene}.png")


def abb_aggregate():
    A = pd.read_csv(TAB / "14_sensorwerte_aggregate_artikel.csv")
    kan = [k for k in pd.read_csv(TAB / "13_sensoren_artikel.csv").query("art == 'Kanal' and ki_ueber_null").Gruppe]
    kan += [k for k in (SP["kanal_f"], SP["kanal_b"]) if k not in kan]
    A = A[A.Kanal.isin(kan)]
    fig, axe = plt.subplots(2, 4, figsize=(9.5, 4.8))
    for ax, k in zip(axe.ravel(), kan):
        t = A[A.Kanal == k].sort_values("shap_anteil_kanal")
        t = t[t.shap_anteil_kanal > 0.005]
        ax.barh(t.Aggregat, 100 * t.shap_anteil_kanal, color=FARBE["Kanal"])
        ax.set_title(KLAR.get(k, k), fontsize=8.5)
        ax.tick_params(labelsize=7)
        ax.set_xlim(0, 100)
        ax.grid(axis="x", alpha=0.3)
    for ax in axe.ravel()[len(kan):]:
        ax.axis("off")
    fig.supxlabel("Anteil am Σ|SHAP| des Kanals [%] (XGBoost, artikelrein, OOF)", fontsize=8.5)
    fig.tight_layout()
    speichern(fig, "abb_7_aggregate.png")


KURZ = D.beschriftungen("xai23_kurz")


def abb_kennzahlauswahl(ebene="artikel"):
    """Ein Balken je Kanal bzw. Block = 100 % seines SHAP-Betrags."""
    teile = [
        ("Sensoren", "Kanal", "shap_anteil_kanal", FARBE["Kanal"],
         pd.read_csv(TAB / f"14_sensorwerte_aggregate_{ebene}.csv"),
         set(pd.read_csv(TAB / f"14_sensorwerte_schwellen_{ebene}.csv").Merkmal)),
        ("Stammdaten", "Block", "shap_anteil_block", FARBE["Stammdaten"],
         pd.read_csv(TAB / f"15_stammdatenwerte_merkmale_{ebene}.csv"),
         set(pd.read_csv(TAB / f"15_stammdatenwerte_schwellen_{ebene}.csv").Merkmal)),
    ]
    hoehen = [d[g].nunique() * (1 if g == "Kanal" else 1.5) for _, g, _, _, d, _ in teile]
    fig, axe = plt.subplots(2, 1, figsize=(7.4, 0.36 * sum(hoehen) + 1.4),
                            gridspec_kw={"height_ratios": hoehen})
    for ax, (titel, g, sp, farbe, d, wahl) in zip(axe, teile):
        reihen = d.groupby(g, sort=False).shap_betrag.sum().sort_values().index
        for y, k in enumerate(reihen):
            t = d[d[g] == k].sort_values("shap_betrag", ascending=False)
            links = 0.0
            for i, (_, r) in enumerate(t[t.Merkmal.isin(wahl)].iterrows()):
                w = 100 * r[sp]
                ax.barh(y, w, left=links, color=farbe, alpha=1.0 if i == 0 else 0.6,
                        edgecolor="white", lw=0.8)
                name = r.Merkmal[len(k) + 1:] if g == "Kanal" else KURZ.get(r.Merkmal, r.Merkmal)
                text = f"{name} {w:.0f} %"
                if len(text) * 0.95 > w:
                    text = f"{name}\n{w:.0f} %"
                if w >= 9:
                    ax.text(links + w / 2, y, text, ha="center", va="center",
                            fontsize=6.5, color="white", linespacing=0.95)
                links += w
            rest = t[~t.Merkmal.isin(wahl)]
            ax.barh(y, 100 - links, left=links, color="#d9d9d9", edgecolor="white", lw=0.8)
            if 100 - links >= 9:
                ax.text(links + (100 - links) / 2, y, f"übrige {len(rest)}", ha="center",
                        va="center", fontsize=6.5, color="#525252")
        ax.set_yticks(range(len(reihen)), [KLAR.get(k, k) for k in reihen], fontsize=8)
        ax.set_xlim(0, 100)
        ax.set_title(f"{titel}: untersuchte {'Kennzahlen' if g == 'Kanal' else 'Merkmale'} "
                     f"(farbig) und übrige (grau)",
                     fontsize=8.5, loc="left")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axe[-1].set_xlabel("Anteil am mittleren SHAP-Betrag des Kanals bzw. Blocks [%]")
    fig.tight_layout()
    speichern(fig, f"abb_7_kennzahlauswahl{'' if ebene == 'artikel' else '_' + ebene}.png")


def abb_werte(merkmale, datei, ebene="artikel"):
    K = pd.read_csv(TAB / f"14_sensorwerte_{ebene}.csv")
    S = pd.read_csv(TAB / f"14_sensorwerte_schwellen_{ebene}.csv").set_index("Merkmal")
    n = len(merkmale)
    sp = 3 if n > 4 else 2
    zl = int(np.ceil(n / sp))
    fig, axe = plt.subplots(zl, sp, figsize=(3.3 * sp, 2.7 * zl), squeeze=False)
    for ax, (m, titel, skala, logx) in zip(axe.ravel(), merkmale):
        k = K[K.Merkmal == m].sort_values("klasse")
        s = S.loc[m]
        x = k.wert_median / skala
        ax.fill_between(x, k.shap_ki_lo, k.shap_ki_hi, color=FARBE["Kanal"], alpha=0.2, lw=0)
        ax.plot(x, k.shap_mittel, "o-", color=FARBE["Kanal"], ms=3, lw=1.2, label="Modell: SHAP")
        ax.errorbar(x + 0, k.y_artikelbereinigt,
                    yerr=[k.y_artikelbereinigt - k.y_art_ki_lo, k.y_art_ki_hi - k.y_artikelbereinigt],
                    fmt="s", color=FARBE["Stammdaten"], ms=3, lw=0.7, capsize=1.5,
                    label="Daten: y − Artikelmittel")
        ax.axhline(0, color="black", lw=0.6)
        if logx:
            ax.set_xscale("log")
        ax.set_title(titel, fontsize=8.5)
        ax.set_xlabel("Wert (Klassenmedian)", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.set_ylabel("Prozentpunkte", fontsize=7.5)
        ax.grid(alpha=0.3)
    for ax in axe.ravel()[n:]:
        ax.axis("off")
    axe[0, 0].legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    speichern(fig, datei)


def abb_stammdaten(merkmale, datei, ebene="artikel"):
    """Wie abb_werte, zusaetzlich das beobachtete y minus Gesamtmittel (grau): bei
    Artikelmerkmalen ist das die einzige Datensicht."""
    K = pd.read_csv(TAB / f"15_stammdatenwerte_{ebene}.csv")
    S = pd.read_csv(TAB / f"15_stammdatenwerte_schwellen_{ebene}.csv").set_index("Merkmal")
    n = len(merkmale)
    sp = 3
    zl = int(np.ceil(n / sp))
    fig, axe = plt.subplots(zl, sp, figsize=(3.3 * sp, 2.9 * zl), squeeze=False)
    for ax, (m, titel) in zip(axe.ravel(), merkmale):
        k = K[K.Merkmal == m].sort_values("klasse")
        s = S.loc[m]
        kat = bool(s.kategorial)
        x = np.arange(len(k)) if kat else k.wert_median.to_numpy()
        ax.fill_between(x, k.shap_ki_lo, k.shap_ki_hi, color=FARBE["Kanal"], alpha=0.2, lw=0)
        ax.plot(x, k.shap_mittel, "o-", color=FARBE["Kanal"], ms=3, lw=1.2, label="Modell: SHAP")
        yr = k.y_mittel - s.y_gesamt
        ax.errorbar(x, yr, yerr=[yr - (k.y_roh_ki_lo - s.y_gesamt), (k.y_roh_ki_hi - s.y_gesamt) - yr],
                    fmt="D", color="#7f7f7f", ms=2.5, lw=0.7, capsize=1.5,
                    label="Daten: y − Gesamtmittel")
        if not s.artikelmerkmal:
            ax.errorbar(x, k.y_artikelbereinigt,
                        yerr=[k.y_artikelbereinigt - k.y_art_ki_lo, k.y_art_ki_hi - k.y_artikelbereinigt],
                        fmt="s", color=FARBE["Stammdaten"], ms=3, lw=0.7, capsize=1.5,
                        label="Daten: y − Artikelmittel")
        ax.axhline(0, color="black", lw=0.6)
        if not kat:
            ax.set_xlabel("Wert (Klassenmedian)", fontsize=7)
        else:
            ax.set_xticks(x, [c[:16] for c in k.kategorien], rotation=40, ha="right", fontsize=6)
        art = "Artikelmerkmal" if s.artikelmerkmal else f"{s.anteil_innen:.0%} Streuung im Artikel"
        ax.set_title(f"{titel}\n({art})".replace("%", " %"), fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_ylabel("Prozentpunkte", fontsize=7.5)
        ax.grid(alpha=0.3)
    for ax in axe.ravel()[n:]:
        ax.axis("off")
    hand, lab = [], []
    for ax in axe.ravel():
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in lab:
                hand.append(h); lab.append(l)
    fig.legend(hand, lab, loc="lower center", ncol=3, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    speichern(fig, datei)


def abb_auswahl(sensoren, stammdaten, datei, ebene="artikel"):
    """Oben Sensoren, unten Stammdaten."""
    K14 = pd.read_csv(TAB / f"14_sensorwerte_{ebene}.csv")
    K15 = pd.read_csv(TAB / f"15_stammdatenwerte_{ebene}.csv")
    S15 = pd.read_csv(TAB / f"15_stammdatenwerte_schwellen_{ebene}.csv").set_index("Merkmal")
    sp = max(len(sensoren or []), len(stammdaten or []))
    if sensoren and stammdaten:
        zl = 2
    else:
        sp, zl = (2, 2) if sp == 4 else (sp, 1)
    fig, axe = plt.subplots(zl, sp, figsize=(3.3 * sp, 3.0 * zl + 0.3), squeeze=False)
    if sensoren and stammdaten:
        zeile_s, zeile_t = axe[0], axe[1]
    else:
        zeile_s = zeile_t = axe.ravel()
    for ax, (m, titel, skala, logx) in zip(zeile_s, sensoren or []):
        k = K14[K14.Merkmal == m].sort_values("klasse")
        x = k.wert_median / skala
        ax.fill_between(x, k.shap_ki_lo, k.shap_ki_hi, color=FARBE["Kanal"], alpha=0.2, lw=0)
        ax.plot(x, k.shap_mittel, "o-", color=FARBE["Kanal"], ms=3, lw=1.2, label="Modell: SHAP")
        ax.errorbar(x, k.y_artikelbereinigt,
                    yerr=[k.y_artikelbereinigt - k.y_art_ki_lo, k.y_art_ki_hi - k.y_artikelbereinigt],
                    fmt="s", color=FARBE["Stammdaten"], ms=3, lw=0.7, capsize=1.5,
                    label="Daten: y − Artikelmittel")
        if logx:
            from matplotlib.ticker import NullFormatter, ScalarFormatter
            ax.set_xscale("log")
            ax.set_xticks([t for t in (20, 50, 100, 200, 300, 500) if x.min() <= t <= x.max() * 1.1])
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(titel, fontsize=8.5)
        ax.set_xlabel("Wert (Klassenmedian" + (", log. Skala)" if logx else ")"), fontsize=7)
    for ax, (m, titel) in zip(zeile_t, stammdaten or []):
        k = K15[K15.Merkmal == m].sort_values("klasse")
        s = S15.loc[m]
        kat = bool(s.kategorial)
        x = np.arange(len(k)) if kat else k.wert_median.to_numpy()
        ax.fill_between(x, k.shap_ki_lo, k.shap_ki_hi, color=FARBE["Kanal"], alpha=0.2, lw=0)
        ax.plot(x, k.shap_mittel, "o-", color=FARBE["Kanal"], ms=3, lw=1.2, label="Modell: SHAP")
        yr = k.y_mittel - s.y_gesamt
        ax.errorbar(x, yr, yerr=[yr - (k.y_roh_ki_lo - s.y_gesamt), (k.y_roh_ki_hi - s.y_gesamt) - yr],
                    fmt="D", color="#7f7f7f", ms=2.5, lw=0.7, capsize=1.5,
                    label="Daten: y − Gesamtmittel")
        if not s.artikelmerkmal:
            ax.errorbar(x, k.y_artikelbereinigt,
                        yerr=[k.y_artikelbereinigt - k.y_art_ki_lo, k.y_art_ki_hi - k.y_artikelbereinigt],
                        fmt="s", color=FARBE["Stammdaten"], ms=3, lw=0.7, capsize=1.5,
                        label="Daten: y − Artikelmittel")
        if kat:
            ax.set_xticks(x, [c[:16] for c in k.kategorien], rotation=40, ha="right", fontsize=6)
        else:
            ax.set_xlabel("Wert (Klassenmedian)", fontsize=7)
        art = "Artikelmerkmal" if s.artikelmerkmal else "variiert im Artikel"
        ax.set_title(f"{titel} ({art})", fontsize=8.5)
    for ax in axe.ravel():
        ax.axhline(0, color="black", lw=0.6)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    for ax in axe[:, 0]:
        ax.set_ylabel("Prozentpunkte", fontsize=7.5)
    hand, lab = [], []
    for ax in axe.ravel():
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in lab:
                hand.append(h); lab.append(l)
    fig.legend(hand, lab, loc="lower center", ncol=3, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0.045 if zl == 2 else 0.08, 1, 1))
    speichern(fig, datei)


SENSOR_PRAEFIX = (SP["praefix_a"], SP["praefix_b"], SP["kanal_f"], SP["lichtbogen_praefix"], SP["phase"], SP["praefix_c"],
                  SP["praefix_f"], SP["praefix_e"], SP["kanal_p"], SP["kanal_q"], SP["kanal_r"],
                  SP["kanal_s"], SP["kanal_t"], SP["kanal_u"], SP["praefix_d"], SP["kanal_v"], SP["kanal_g"])


def abb_lokal(ebene="artikel"):
    d = json.loads((ERG / f"02_lokal_{ebene}.json").read_text(encoding="utf-8"))
    fig, axe = plt.subplots(2, 2, figsize=(9.6, 6.2))
    for ax, f in zip(axe.ravel(), d["faelle"]):
        t = pd.DataFrame(f["beitraege"]).iloc[::-1]
        farbe = [FARBE["Kanal"] if m.startswith(SENSOR_PRAEFIX) else FARBE["Stammdaten"]
                 for m in t.merkmal]
        ax.barh(t.merkmal, t.beitrag, color=farbe)
        ax.axvline(0, color="black", lw=0.7)
        rolle = f["rolle"].split("  ", 1)[-1]
        ax.set_title(f"{rolle}\nAusschuss {de(f['y_wahr'], 1)} %, vorhergesagt "
                     f"{de(f['y_vorhergesagt'], 1)} %", fontsize=8.5)
        ax.tick_params(labelsize=7)
        ax.grid(axis="x", alpha=0.3)
    for ax in axe[1]:
        ax.set_xlabel("SHAP-Beitrag zur Vorhersage [PP]", fontsize=7.5)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=FARBE["Kanal"], label="Sensor"),
                        Patch(color=FARBE["Stammdaten"], label="Stammdaten")],
               loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    speichern(fig, "abb_7_lokal.png")


def abb_erkannt_verfehlt(ebene="artikel"):
    d = json.loads((ERG / f"12_extreme_{ebene}.json").read_text(encoding="utf-8"))["erkannt_vs_verfehlt"]
    t = pd.DataFrame(d["merkmale"]).head(10).iloc[::-1]
    farbe = [FARBE["Kanal"] if m.startswith(SENSOR_PRAEFIX) else FARBE["Stammdaten"] for m in t.merkmal]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.barh(t.merkmal, t.shap_diff, color=farbe)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_xlabel("mittlerer SHAP-Beitrag: erkannt minus verfehlt [PP]", fontsize=8)
    ax.set_title(f"Hoher Ausschuss: erkannt (n = {d['n_erkannt']}) gegen verfehlt "
                 f"(n = {d['n_verfehlt']})", fontsize=9)
    ax.tick_params(labelsize=7.5)
    ax.grid(axis="x", alpha=0.3)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=FARBE["Kanal"], label="Sensor"),
                       Patch(color=FARBE["Stammdaten"], label="Stammdaten")],
              fontsize=7.5, frameon=False, loc="lower right")
    fig.tight_layout()
    speichern(fig, "abb_7_erkannt_verfehlt.png")


def abb_gegenprobe(ebene="artikel"):
    T = pd.read_csv(TAB / f"16_gegenproben_{ebene}.csv")
    S = pd.read_csv(TAB / f"14_sensorwerte_schwellen_{ebene}.csv").set_index("Merkmal")
    vorlage = D.beschriftungen("xai23_gegenprobe")
    felder = [(m, vorlage.get(m, m + " > {schwelle}").format(schwelle=_zahl(S.loc[m, "schwelle"])))
              for m in (SP["kanal_a"] + "_n_samples", SP["kanal_b"] + "_std")]
    quellen = ["beobachteter Ausschuss", "XGBoost", "CatBoost", "TabPFN"]
    namen = ["Ausschuss (Daten)", "XGBoost", "CatBoost", "TabPFN"]
    stil = {"roh": ("#9e9e9e", "o", -0.15, "über alle Billets"),
            "im Artikel": ("#1f78b4", "D", 0.15, "gegenüber dem eigenen Artikel")}
    fig, axe = plt.subplots(1, 2, figsize=(8.6, 3.0), sharey=True)
    for ax, (merkmal, titel) in zip(axe, felder):
        for bezug, (farbe, marker, versatz, label) in stil.items():
            t = T[(T.Merkmal == merkmal) & (T.bezug == bezug)].set_index("quelle").loc[quellen]
            y = np.arange(len(quellen))[::-1] + versatz
            ax.errorbar(t.unterschied, y, xerr=[t.unterschied - t.ki_lo, t.ki_hi - t.unterschied],
                        fmt=marker, color=farbe, ms=5, capsize=2.5, lw=1, label=label)
        ax.axvline(0, color="black", lw=0.7)
        ax.axhline(2.5, color="#cccccc", lw=0.6)
        ax.set_title(titel, fontsize=9)
        ax.set_xlabel("kritischer Bereich minus Rest [PP]", fontsize=8)
        ax.grid(axis="x", alpha=0.3)
        ax.tick_params(labelsize=7.5)
    axe[0].set_yticks(np.arange(len(quellen))[::-1], namen)
    h, l = axe[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    speichern(fig, "abb_7_gegenprobe.png")


def abb_extreme():
    d = json.loads((ERG / "12_extreme_artikel.json").read_text(encoding="utf-8"))
    fig, axe = plt.subplots(1, 2, figsize=(8.6, 3.8))
    for ax, key, titel in zip(axe, ("top_oben", "top_unten"),
                              (f"oberes Dezil (y hoch, n = {d['n_oben']})",
                               f"unteres Dezil (y niedrig, n = {d['n_unten']})")):
        t = pd.DataFrame(d[key]).head(10)
        t = t.sort_values("beitrag", key=abs)
        sensor = (SP["praefix_a"], SP["praefix_b"], SP["kanal_f"], SP["lichtbogen_praefix"], SP["phase"], SP["praefix_c"],
                  SP["praefix_f"], SP["praefix_e"])
        farbe = [FARBE["Kanal"] if m.startswith(sensor) else FARBE["Stammdaten"] for m in t.merkmal]
        ax.barh(t.merkmal, t.beitrag, color=farbe)
        ax.axvline(0, color="black", lw=0.7)
        ax.set_title(titel, fontsize=9)
        ax.set_xlabel("mittlerer SHAP-Beitrag in der Gruppe [PP]")
        ax.tick_params(labelsize=7.5)
        ax.grid(axis="x", alpha=0.3)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=FARBE["Kanal"], label="Sensor"),
                        Patch(color=FARBE["Stammdaten"], label="Stammdaten")],
               loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    speichern(fig, "abb_7_extreme.png")


WERTE = [(m, D.beschriftungen("xai23_werte").get(m, m), skala, log) for m, skala, log in [
    (SP["kanal_a"] + "_n_samples", 1, True),
    (SP["kanal_b"] + "_std", 1, False),
    (SP["kanal_c"] + "_min", 1, False),
    (SP["kanal_f"] + "_mean", 1e3, False),
    (SP["lichtbogen_leistung"] + "_std", 1e3, False),
    (SP["kanal_h"] + "_mean", 1, False),
]]

STAMM = [
    (SP["wand_soll"], "Sollmaße: " + SP["wand_soll"]),
    ("prog__" + SP["rezept_a"], "Rezept: prog__" + SP["rezept_a"]),
    (SP["rezept_c"], "Rezept: " + SP["rezept_c"]),
    ("Glastyp", "Glas/Material: Glastyp"),
    (SP["laenge_erwartet"], "Sollmaße: " + SP["laenge_erwartet"]),
    (SP["rohstoff_artikel_anode"], "Rohstoff: " + SP["rohstoff_artikel_anode"]),
    (SP["granulatdichte_kathode"], "Rohstoff: " + SP["granulatdichte_kathode"]),
    (SP["innenschliff"], "Innenschliff notwendig"),
]

AUSWAHL_SENSOR = [(m, D.beschriftungen("xai23_auswahl").get(m, m), skala, log)
                  for m, skala, log in [
    (SP["kanal_a"] + "_n_samples", 1, True),
    (SP["kanal_c"] + "_min", 1, False),
    (SP["kanal_b"] + "_std", 1, False),
    (SP["kanal_f"] + "_mean", 1e3, False),
]]
AUSWAHL_STAMM = [
    (SP["wand_soll"], SP["wand_soll"]),
    (SP["laenge_erwartet"], SP["laenge_erwartet"]),
    (SP["granulatdichte_kathode"], SP["granulatdichte_kathode"]),
    ("Glastyp", "Glastyp"),
]


if __name__ == "__main__":
    import sys
    abb_ebenen()
    abb_nutzen_sensor()
    if (TAB / "tabelle_7_3_sensornutzen.csv").exists():
        for e in EBENEN_TITEL:
            abb_nutzen_ebene(e)
    abb_extreme()
    abb_lokal()
    abb_erkannt_verfehlt()
    if (TAB / "16_gegenproben_artikel.csv").exists():
        abb_gegenprobe()
    if (TAB / "14_sensorwerte_artikel.csv").exists() and "--ohne14" not in sys.argv:
        abb_aggregate()
        if (TAB / "15_stammdatenwerte_merkmale_artikel.csv").exists():
            abb_kennzahlauswahl()
        abb_werte(WERTE, "abb_7_sensorwerte.png")
    if (TAB / "15_stammdatenwerte_artikel.csv").exists():
        abb_stammdaten(STAMM, "abb_7_stammdatenwerte.png")
        abb_auswahl(AUSWAHL_SENSOR, AUSWAHL_STAMM, "abb_7_auswahl.png")
        abb_auswahl(AUSWAHL_SENSOR, [], "abb_7_auswahl_sensoren.png")
        abb_auswahl([], AUSWAHL_STAMM, "abb_7_auswahl_stammdaten.png")
