"""03 — XGBoost gegen CatBoost.

In der Arbeit: nicht berichtet; XGBoost gegen CatBoost; liefert die XGBoost-
Werte, die xai/06 mit TabPFN vergleicht.

Übereinstimmung der SHAP-Ranglisten beider Modelle.
Ausgaben unter reports/xai/, runs/xai/ und reports/xai/tabellen/.

    TEST_LEVEL=<billet|auftrag|artikel> python xai/03_vergleich.py
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

import _common as C

TOP = 20
K_GITTER = [5, 10, 20, 30, 50, 75, 100, 150, 200]


def feld_a(ax, wx, wc):
    tx = list(wx.head(TOP).Merkmal)
    tc = list(wc.head(TOP).Merkmal)
    gemeinsam = set(tx) & set(tc)
    for spalte, (namen, farbe, titel) in enumerate(
        ((tx, "#3182bd", "XGBoost"), (tc, "#e6550d", "CatBoost"))
    ):
        for i, m in enumerate(namen):
            ax.text(
                spalte,
                TOP - i,
                f"{i+1:>2}. {m[:30]}",
                fontsize=6.8,
                ha="left",
                va="center",
                color="black" if m in gemeinsam else "#999999",
                weight="bold" if m in gemeinsam else "normal",
            )
        ax.text(
            spalte, TOP + 1.4, titel, fontsize=10, weight="bold", color=farbe, ha="left"
        )
    ax.set_xlim(-0.05, 2.0)
    ax.set_ylim(0, TOP + 2.2)
    ax.axis("off")
    ax.set_title(
        f"A  Top {TOP} je Modell — fett = in beiden Listen "
        f"({len(gemeinsam)} von {TOP})",
        fontsize=10,
    )


def feld_b(ax, wx, wc):
    m = wx.merge(wc, on="Merkmal", suffixes=("_x", "_c"))
    rho = spearmanr(m.mittlerer_betrag_x, m.mittlerer_betrag_c).statistic
    eps = 1e-6
    ax.scatter(
        m.mittlerer_betrag_x + eps,
        m.mittlerer_betrag_c + eps,
        s=10,
        alpha=0.5,
        color="#3182bd",
    )
    lim = [eps, max(m.mittlerer_betrag_x.max(), m.mittlerer_betrag_c.max()) * 1.4]
    ax.plot(lim, lim, color="#999999", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("mittlerer |SHAP| — XGBoost")
    ax.set_ylabel("mittlerer |SHAP| — CatBoost")
    ax.set_title(f"B  Merkmal für Merkmal   Spearman ρ = {rho:+.3f}", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    return float(rho), m


def feld_c(ax, wx, wc):
    anteile = []
    for k in K_GITTER:
        a, b = set(wx.head(k).Merkmal), set(wc.head(k).Merkmal)
        anteile.append(len(a & b) / k)
    ax.plot(K_GITTER, anteile, "o-", color="#3182bd", lw=1.8)
    for k, v in zip(K_GITTER, anteile):
        ax.annotate(
            f"{v:.2f}",
            (k, v),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=7.5,
        )
    ax.axhline(1.0, color="#999999", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("Listenlänge k (log)")
    ax.set_ylabel("Anteil gemeinsamer Merkmale")
    ax.set_title("C  Übereinstimmung der Top-k", fontsize=10)
    ax.grid(alpha=0.3)
    return dict(zip(K_GITTER, anteile))


def feld_d(ax, guete):
    namen = list(guete)
    x = np.arange(len(namen))
    r2 = [guete[n]["r2"] for n in namen]
    mae = [guete[n]["mae"] for n in namen]
    ax.bar(x - 0.2, r2, 0.4, color="#3182bd", label="Test-R²")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, mae, 0.4, color="#e6550d", label="Test-MAE")
    for i, (a, b) in enumerate(zip(r2, mae)):
        ax.annotate(f"{a:+.3f}", (i - 0.2, a), ha="center", va="bottom", fontsize=8)
        ax2.annotate(f"{b:.2f}", (i + 0.2, b), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(namen)
    ax.set_ylabel("Test-R²", color="#3182bd")
    ax2.set_ylabel("Test-MAE [Prozentpunkte]", color="#e6550d")
    ax.set_ylim(0, max(r2) * 1.35)
    ax2.set_ylim(0, max(mae) * 1.35)
    ax.set_title("D  Sind beide Modelle vergleichbar gut?", fontsize=10)
    ax.grid(alpha=0.3, axis="y")


def main():
    ds, X_tr, X_te = C.lade()
    y_te = np.asarray(ds.y_test)

    mx, qx = C.xgb_modell(X_tr, ds.y_train)
    bx, basx = C.shap_xgboost(mx, X_te)
    C.pruefe_additivitaet(bx, basx, mx.predict(X_te), "XGBoost")
    wx = C.globale_wichtigkeit(bx, X_te.columns)

    print("  CatBoost wird angepasst …", flush=True)
    mc, qc = C.catboost_modell(X_tr, ds.y_train)
    bc, basc = C.shap_catboost(mc, X_te, y_te)
    C.pruefe_additivitaet(bc, basc, mc.predict(X_te), "CatBoost")
    wc = C.globale_wichtigkeit(bc, X_te.columns)

    guete = {}
    for name, m in (("XGBoost", mx), ("CatBoost", mc)):
        p = m.predict(X_te)
        guete[name] = {
            "r2": float(r2_score(y_te, p)),
            "mae": float(mean_absolute_error(y_te, p)),
        }
        print(
            f"  {name}: Test-R² {guete[name]['r2']:+.4f}  "
            f"MAE {guete[name]['mae']:.3f}"
        )

    pdf = C.pdf_oeffnen(C.mit_ebene("03_vergleich.pdf"))
    fig, ax = C.seite(
        "Stimmen XGBoost und CatBoost in der Merkmalswichtigkeit " "überein?", 2, 2
    )
    feld_a(ax[0], wx, wc)
    rho, verb = feld_b(ax[1], wx, wc)
    ueb = feld_c(ax[2], wx, wc)
    feld_d(ax[3], guete)
    C.schliesse(pdf, fig, C.fusszeile(ds, X_te, f"XGBoost {qx} | CatBoost {qc}"))

    top20 = len(set(wx.head(20).Merkmal) & set(wc.head(20).Merkmal))
    C.textseite(
        pdf,
        "Lesehinweise und Vorbehalte",
        [
            "Warum dieser Vergleich. Eine Rangliste aus einem einzigen Modell kann "
            "eine Eigenschaft des Algorithmus sein statt der Daten. XGBoost baut "
            "Bäume levelwise mit gewichteter Quantil-Approximation, CatBoost "
            "symmetrische (oblivious) Bäume mit geordnetem Boosting — die "
            "induktiven Verzerrungen sind verschieden. Übereinstimmung ist deshalb "
            "ein Hinweis darauf, dass die Rangfolge in den Daten liegt.",
            f"Ergebnis. Rangkorrelation über alle {len(verb)} Merkmale: "
            f"ρ = {rho:+.3f}. Von den 20 wichtigsten Merkmalen teilen sich beide "
            f"Modelle {top20}. Die Übereinstimmung der Top-k steigt von "
            f"{ueb[5]:.0%} bei k = 5 auf {ueb[100]:.0%} bei k = 100.",
            f"Einordnung der Güte. XGBoost erreicht auf dem Test-Set "
            f"R² {guete['XGBoost']['r2']:+.3f} bei MAE {guete['XGBoost']['mae']:.2f}, "
            f"CatBoost R² {guete['CatBoost']['r2']:+.3f} bei MAE "
            f"{guete['CatBoost']['mae']:.2f}. Ranglisten zu vergleichen setzt "
            f"voraus, dass beide Modelle die Aufgabe ähnlich gut lösen — bei "
            f"deutlich unterschiedlicher Güte verglichen man eine gute mit einer "
            f"schlechten Erklärung.",
            "Vorbehalt zur Konfiguration. XGBoost läuft mit getunten "
            "Hyperparametern, CatBoost mit Defaults, da für CatBoost kein "
            "Tuning-Ergebnis vorliegt. Ausserdem wurde CatBoost hier ohne Early "
            "Stopping angepasst, weil ein Ranking-Vergleich ein deterministisches "
            "Modell braucht; das Trainingsskript nutzt Early Stopping gegen einen "
            "einzelnen Validierungssplit. Beide Abweichungen sprechen dafür, die "
            "Übereinstimmung als untere Schranke zu lesen.",
            "Vorbehalt zur Aussagekraft. Übereinstimmung zweier Baumverfahren "
            "zeigt Robustheit gegenüber der Wahl des Algorithmus, nicht "
            "Richtigkeit. Beide teilen dieselbe Merkmalsmenge und damit dieselben "
            "Redundanzen; korrelierte Merkmale können in beiden Modellen "
            "gleichermassen zufällig gegeneinander ausgetauscht sein.",
        ],
    )
    pdf.close()

    tab = wx[["Merkmal", "mittlerer_betrag"]].rename(
        columns={"mittlerer_betrag": "xgboost"}
    ).merge(
        wc[["Merkmal", "mittlerer_betrag"]].rename(
            columns={"mittlerer_betrag": "catboost"}
        ),
        on="Merkmal",
        how="outer",
    )
    tab[["xgboost", "catboost"]] = tab[["xgboost", "catboost"]].fillna(0.0)
    tab["rang_xgboost"] = tab.xgboost.rank(ascending=False, method="min").astype(int)
    tab["rang_catboost"] = tab.catboost.rank(ascending=False, method="min").astype(int)

    p = C.speichere(
        C.mit_ebene("03_vergleich.json"),
        {
            "spearman_rho": rho,
            "top20_gemeinsam": top20,
            "ueberlappung_top_k": {str(k): v for k, v in ueb.items()},
            "guete": guete,
            "params": {"XGBoost": qx, "CatBoost": qc},
            "top20_xgboost": wx.head(20)[["Merkmal", "mittlerer_betrag"]].to_dict(
                "records"
            ),
            "top20_catboost": wc.head(20)[["Merkmal", "mittlerer_betrag"]].to_dict(
                "records"
            ),
            "rangliste_csv": C.mit_ebene("03_vergleich.csv"),
        },
    )
    pcsv = C.speichere_rangliste(C.mit_ebene("03_vergleich.csv"), tab, "xgboost")
    print(f"\ngeschrieben: {C.OUT_DIR / C.mit_ebene('03_vergleich.pdf')}\n             {p}\n             {pcsv}\n")
    print(f"Spearman ρ = {rho:+.3f}   gemeinsame Top-20: {top20}/20")
    print("Übereinstimmung Top-k:", {k: round(v, 2) for k, v in ueb.items()})


if __name__ == "__main__":
    main()
