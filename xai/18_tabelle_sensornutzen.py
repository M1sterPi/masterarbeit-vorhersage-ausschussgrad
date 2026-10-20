"""18 — Tabelle: Nutzen je Prozesskanal und Stammdatenblock.

In der Arbeit: Tabelle 14 (Anhang C, Nutzen je Gruppe auf drei Ebenen).

Alle drei Ebenen, aus xai/13.
Ausgabe: reports/xai/tabellen/tabelle_7_3_sensornutzen.csv/.md.

    python xai/18_tabelle_sensornutzen.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data_pipeline"))
import data_prep as D  # noqa: E402

TAB = ROOT / "reports" / "xai" / "tabellen"
ERG = ROOT / "runs" / "xai"

EBENEN = [("artikel", "artikelrein"), ("auftrag", "auftragsrein"), ("billet", "billetweise")]

KLAR = D.beschriftungen("xai18_klar")


def de(x, nk=2, vz=True):
    s = f"{x:+.{nk}f}" if vz else f"{x:.{nk}f}"
    return s.replace("-", "−").replace(".", ",")


def laden():
    tabs, meta = {}, {}
    for e, _ in EBENEN:
        p, j = TAB / f"13_sensoren_{e}.csv", ERG / f"13_sensoren_{e}.json"
        if not p.exists() or not j.exists():
            sys.exit(f"{p.name} oder {j.name} fehlt — erst TEST_LEVEL={e} python xai/13_sensoren.py")
        tabs[e] = pd.read_csv(p).set_index("Gruppe")
        meta[e] = json.loads(j.read_text(encoding="utf-8"))
    gruppen = set.union(*(set(t.index) for t in tabs.values()))
    for e, t in tabs.items():
        if set(t.index) != gruppen:
            sys.exit(f"Gruppen auf Ebene {e} weichen ab: {sorted(gruppen ^ set(t.index))}")
    return tabs, meta


def lang(tabs):
    """Eine Zeile je Gruppe mit allen Werten aller Ebenen (CSV zum Nachpruefen)."""
    basis = tabs["artikel"][["art", "n_spalten"]]
    teile = [basis]
    for e, _ in EBENEN:
        t = tabs[e][["delta_mae", "ki_lo", "ki_hi", "n_falten_positiv", "ki_ueber_null", "ueber_decke"]]
        teile.append(t.add_suffix(f"_{e}"))
    df = pd.concat(teile, axis=1)
    df["gesichert_irgendwo"] = df[[f"ki_ueber_null_{e}" for e, _ in EBENEN]].any(axis=1)
    return df.sort_values("delta_mae_artikel", ascending=False)


def zelle(r, e):
    txt = f"{de(r[f'delta_mae_{e}'])} [{de(r[f'ki_lo_{e}'])}; {de(r[f'ki_hi_{e}'])}]"
    if r[f"ki_ueber_null_{e}"]:
        txt = f"**{txt}**"
    return f"{txt} ({int(r[f'n_falten_positiv_{e}'])}/5)"


def markdown(df, meta):
    kopf = "| Gruppe | Spalten | " + " | ".join(f"{t} ΔMAE [KI] (Falten)" for _, t in EBENEN) + " |"
    zeilen = [kopf, "|" + "---|" * (2 + len(EBENEN))]
    for g, r in df[df.gesichert_irgendwo].iterrows():
        name = KLAR.get(g, g)
        name = f"*Stammdaten: {name}*" if r.art == "Stammdaten" else name
        zeilen.append(f"| {name} | {int(r.n_spalten)} | " + " | ".join(zelle(r, e) for e, _ in EBENEN) + " |")
    rest = df[~df.gesichert_irgendwo]
    if len(rest):
        kan = rest[rest.art == "Kanal"]
        spannweite = f"je {int(kan.n_spalten.min())}–{int(kan.n_spalten.max())}" if len(kan) else ""
        zeilen.append(
            f"| übrige {len(kan)} Kanäle und Blöcke | {spannweite} | "
            + " | ".join(f"≤ {de(rest[f'delta_mae_{e}'].max())}" for e, _ in EBENEN) + " |"
        )
    zeilen.append(
        "| Rauschdecke (15 Zufallsgruppen) | 10 | "
        + " | ".join(de(meta[e]["rauschdecke"], 3) for e, _ in EBENEN) + " |"
    )
    zeilen.append(
        "| Vorsprung des Modells vor dem Median | – | "
        + " | ".join(de(meta[e]["mae_median_mittel"] - meta[e]["mae_mittel"]) for e, _ in EBENEN) + " |"
    )
    return "\n".join(zeilen)


def main():
    tabs, meta = laden()
    df = lang(tabs)
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / "tabelle_7_3_sensornutzen.csv", encoding="utf-8", float_format="%.6g")
    md = markdown(df, meta)
    (TAB / "tabelle_7_3_sensornutzen.md").write_text(md + "\n", encoding="utf-8")
    print(md)
    print()
    for e, t in EBENEN:
        kan = df[df.art == "Kanal"]
        print(f"{t:13s} MAE {meta[e]['mae_mittel']:.3f}  Median {meta[e]['mae_median_mittel']:.3f}  "
              f"Decke {meta[e]['rauschdecke']:.4f}  Kanäle über Decke {int(kan[f'ueber_decke_{e}'].sum())}/{len(kan)}  "
              f"KI über null {int(kan[f'ki_ueber_null_{e}'].sum())}")


if __name__ == "__main__":
    main()
