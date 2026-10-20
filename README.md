# Ausschussprognose für Quarzglas-Billets – Skripte zur Masterarbeit

Dieses Repository enthält die Skripte, mit denen die Zahlen, Tabellen und
Abbildungen der Masterarbeit berechnet wurden: Datenaufbereitung, Modellvergleich
(XGBoost, CatBoost, TabPFN, MiniRocket, HYDRA, FreshPRINCE, zweistufiges Modell),
explorative Datenanalyse (`eda/`) und erklärbare KI (`xai/`).

**Ohne Daten.** Daten, Ergebnisse und projektspezifische Angaben sind nicht
enthalten:

- Daten und Ausgaben (`data/`, `reports/`, `runs/`) sind von git ausgeschlossen,
  siehe [`data/README.md`](data/README.md).
- Die Liste der Spalten, die erst nach der Schmelze feststehen
  (`_LEAKAGE_KEYWORDS` in `data_pipeline/data_prep.py`), ist leer. Ohne sie gehen
  diese Spalten in die Modelle ein; die Ergebnisse der Arbeit lassen sich damit
  nicht nachrechnen.
- Lesbare Bezeichnungen für Abbildungen und Tabellen liest der Code aus
  `data/beschriftungen.json`, das ebenfalls nicht enthalten ist. Ohne die Datei
  stehen die Spaltennamen in den Abbildungen.
- Spalten- und Kanalnamen der Quelldaten stehen nicht im Code. Er spricht jede
  Spalte über eine Rolle an (`SP["auftrag"]`, `SP["kanal_a"]` …); die
  Zuordnung zu den echten Namen und einige Kenngrößen der Anlage liest
  `data_prep.SP` aus `data/spalten.json`, das ebenfalls nicht enthalten ist.
  Ohne die Datei bricht jedes Skript mit einem Hinweis ab.

## Aufbau

```
data_pipeline/   Daten → Merkmalstabelle → Modellläufe → Auswertung der Läufe
eda/             explorative Analyse und Vorverarbeitung (Kapitel 4 und 5), 01–33
xai/             erklärbare KI und Sensoranalyse (Kapitel 7), 01–25
reproduzieren.py startet alle Schritte in der richtigen Reihenfolge
data/            Eingabedaten (nicht im Repo, siehe data/README.md)
reports/, runs/  Ausgaben (entstehen beim Ausführen)
```

| Ordner | Skripte |
|---|---|
| `data_pipeline/` Daten | `convert_sql_db.py` (ERP-Export → Stammdaten je Billet), `merge_billetv2.py` (InfluxDB → Rohkurve je Billet), `prepare_data.py` (Rohkurven + Stammdaten → Merkmalstabelle), `arc_end_scan.py` (Lichtbogenende je Billet) |
| `data_pipeline/` gemeinsame Module | `data_prep.py` (Laden, Filter, eingefrorene Testmengen, Kreuzvalidierung, Merkmalsverarbeitung), `runs_io.py` (Lauf-Übersicht, Auswahl der finalen Läufe) |
| `data_pipeline/` Modelle | `xgboost_regression.py`, `catboost_regression.py`, `tabpfn_regression.py`, `rocket_regression.py` (MiniRocket), `hydra_regression.py`, `freshprince_regression.py`, `sequenz_runner.py`, `hurdle_regression.py` (zweistufiges Modell), `tune.py` (Optuna) |
| `data_pipeline/` Auswertung | `bootstrap_ci.py` (gepaarter Gruppen-Bootstrap), `ergebnistabelle.py` (Kennzahlen aller finalen Läufe) |
| `eda/` | `_common.py` + 33 nummerierte Skripte; PDFs nach `reports/eda/`, Kennzahlen nach `runs/eda/` |
| `xai/` | `_common.py` + 25 nummerierte Skripte; PDFs nach `reports/xai/`, Kennzahlen nach `runs/xai/`, Tabellen nach `reports/xai/tabellen/` |

## Installation

Python 3.12 und [uv](https://docs.astral.sh/uv/):

```
uv sync                  # legt .venv mit den gepinnten Versionen an
cp .env.example .env     # Pfade und Zugänge eintragen
```

Die Versionen in `pyproject.toml` und `uv.lock` sind die, mit denen die Arbeit
gerechnet wurde. torch kommt als Build für CUDA 13 und läuft ohne Grafikkarte
auf der CPU. TabPFN lädt beim ersten Aufruf seine Modellgewichte aus dem
Internet.

## Ausführen

```
uv run python reproduzieren.py --liste           # alle Schritte mit Bedarf
uv run python reproduzieren.py                   # modelle, auswertung, eda, xai
uv run python reproduzieren.py --nur eda/19      # einzelne Schritte
uv run python reproduzieren.py --ohne-gpu --ohne-kurven
uv run python reproduzieren.py --mit-daten --mit-tuning   # zusätzlich ab Rohdaten
```

Jedes Skript lässt sich auch einzeln aus dem Projektordner starten, etwa
`uv run python eda/05_zeit.py`. Einstellungen über Umgebungsvariablen:

| Variable | Werte | Wirkung |
|---|---|---|
| `TEST_LEVEL` | `billet`, `auftrag`, `artikel` (Standard) | Bewertungsebene: welche eingefrorene Testmenge, bei xai auch welche Falten |
| `CV_LEVEL` | `random`, `auftrag`, `artikel` (Standard) | Falten der Kreuzvalidierung in den Modellskripten; die Arbeit nutzt die Paare billet/random, auftrag/auftrag, artikel/artikel |
| `XGB_PARAMS` | `getunt` (Standard), `default` | XGBoost mit Bibliotheks-Voreinstellungen (eda/20) |
| `HYBRID`, `ZNORM` | `0`/`1` | MiniRocket nur auf Kurven bzw. zusammen mit der Merkmalstabelle; z-Normierung je Kurve (auch HYDRA/FreshPRINCE) |
| `FEATURE_SOURCE` | `beides` (Standard), `sql`, `influx` | Merkmalsquellen einschränken |
| `KURVEN_DIR`, `BEISPIEL_BILLET`, `INFLUX_*`, `SQL_EXPORTS` | über `.env` | Ordner der Rohkurven, Beispiel für eda/32, Zugang für den Datenexport |

Lange Schritte sind TabPFN (xai/05, eda/22, eda/19), FreshPRINCE und der Aufbau
der Merkmalstabelle. Rohkurven (`KURVEN_DIR`) brauchen nur die Schritte, die
`reproduzieren.py --liste` mit „kurven“ kennzeichnet, und auch die nur, solange
der Kurven-Cache in `data/cache/` fehlt.

## Welche Läufe in die Arbeit eingehen

`runs/runs_summary.csv` sammelt jeden Modelllauf, auch Ablations- und
Zwischenläufe. Welche davon berichtet werden, regelt genau eine Funktion:
`runs_io.waehle_finale_laeufe()` mit der Liste `FINALE_KONFIG`. Je Konfiguration
und Bewertungsebene nimmt sie den zuletzt geschriebenen passenden Lauf auf dem
endgültigen Datenstand. eda/10, eda/20, eda/21, `bootstrap_ci.py` und
`ergebnistabelle.py` nutzen sie alle.

## Wo die Ergebnisse der Arbeit herkommen

Abschnitts-, Tabellen- und Abbildungsnummern nach dem Stand vom 19.10.2026.
Jedes Skript nennt im Kopf unter „In der Arbeit:“, wofür es gebraucht wird.
Abbildungen liegen unter `reports/figures/`, Tabellen als CSV/Markdown unter
`reports/eda/` bzw. `reports/xai/tabellen/`, Kennzahlen als JSON unter
`runs/eda/` und `runs/xai/`.

### Kapitel 4 – Datenbasis und explorative Analyse

| In der Arbeit | Skript | Ausgabe |
|---|---|---|
| Tabelle 1, Gruppengrößen und Geschwisteranteil (4.1.1, 4.1.2) | eda/23, eda/01, eda/28 | `runs/eda/23_ebenenstruktur.json`, `01_verteilung.pdf`, `runs/eda/28_spaltenbilanz.json` |
| Abbildung 3 (Ebenenschema) | eda/23 | `ebenen_schema.png` |
| Zielgröße, Abbildung 4 (Untergrenze), 4.1.3 | eda/01, eda/08 | `01_verteilung.pdf`, `y_untergrenze_1prozent.png` |
| Abbildung 5 (Verteilung des Ausschussgrads), 4.2.1 | eda/23, eda/01 | `y_verteilung.png` |
| Abbildung 6 (Schmelzverlauf), Aufzeichnung der Kanäle (4.2.2) | eda/32, eda/16, eda/14 | `kapitel4/sensorverlauf_beispiel.png`, `16_sensorverlaeufe.pdf`, `14_segmentierung.pdf` |
| Fehlende Werte (4.2.3) | eda/17, eda/04 | `17_fehlwerte.pdf`, `04_charge.pdf` |
| Abbildung 7, Korrelationsstruktur (4.2.4) | eda/18 | `kapitel4/kanalpaare.png`, `18_sensorkorrelation.pdf` |
| Tabelle 2 (Varianzanteil je Gruppierung) | eda/02 | `02_ebenen.pdf` |
| Tabelle 3 (Zerlegung auf drei Ebenen) | eda/19 | `19_auftrag_vs_artikel.pdf`, `runs/eda/19_auftrag_vs_artikel.json` |
| Abbildung 8 (vier Artikel) | eda/23 | `ebenen_beispiel.png` |
| Tabelle 4, Merkmale auf drei Ebenen, Rohstoffcharge und -analyse (4.2.6) | eda/24, eda/04 | `24_merkmalsebenen.csv`, `runs/eda/24_merkmalsebenen.json`, `04_charge.pdf` |
| Abgebrochene Schmelzen (4.2.6) | eda/23, eda/09 | `runs/eda/23_ebenenstruktur.json`, `09_prozesskanaele.pdf` |
| Abbildung 9, Tabelle 5 (zeitliche Entwicklung) | eda/23, eda/05 | `zeitverlauf.png`, `runs/eda/05_zeit.json` |

### Kapitel 5 – Datenvorverarbeitung

| In der Arbeit | Skript | Ausgabe |
|---|---|---|
| Abbildung 10 (Ablauf) | eda/27 | `kapitel5/ablauf_vorverarbeitung.png` |
| Zeitfenster und Lichtbogenende (5.1) | `merge_billetv2.py`, `arc_end_scan.py`, eda/25 | `runs/eda/25_lichtbogenende.json` |
| Rasterung, Tabelle 6 (Glättung, 5.2) | `rocket_regression.py`, eda/13 | `13_glaettung.pdf` |
| Kennzahlenkatalog (5.3, Tabelle B.2) | `prepare_data.py`, eda/28 | `runs/eda/28_spaltenbilanz.json` |
| Tabelle 7 (Spaltenbilanz, 5.4) | eda/28 | `28_spaltenbilanz.csv` |
| Datumsfelder, Soll-/Ist-Masse, versteckte Zeitstempel (5.4) | eda/33 | `runs/eda/33_vorhersagezeitpunkt.json` |
| Tabelle 8 (Kardinalitätsschwelle, 5.5) | eda/29 | `29_kardinalitaet.csv`, `runs/eda/29_kardinalitaet.json` |
| Unplausible Werte (5.6) | eda/12, eda/33 | `12_skalierung.pdf`, `runs/eda/33_vorhersagezeitpunkt.json` |
| Fehlende Werte, Tabelle 9 (Imputation, 5.6) | eda/17, eda/11 | `17_fehlwerte.pdf`, `11_imputation.pdf` |
| Skalierung, MiniRocket mit und ohne z-Normierung (5.7) | eda/12, `ergebnistabelle.py` | `12_skalierung.pdf`, `runs/ergebnisse_final.csv` |

### Kapitel 6 – Modellentwicklung

| In der Arbeit | Skript | Ausgabe |
|---|---|---|
| Tabelle 10 (Testmengen, 6.1) | eda/30 | `30_testmengen.csv` |
| Tabelle 11, Tabelle A.1 (Hyperparameter) | `tune.py`, eda/20 | `20_hyperparameter_xgboost.md`, `20_hyperparameter_catboost.md` |
| Abbildung 11 (Voreinstellung gegen optimiert) | eda/20 | `hyperparameter_ebenen.png` |
| Modellkennzahlen aller Verfahren, Testmenge in 8.1 | `*_regression.py`, `ergebnistabelle.py` | `runs/ergebnisse_final.csv` |
| Abbildungen 12 und 13 (CV-MAE, CV-R²) | eda/10 | `kapitel6/abb_modellvergleich_mae.png`, `kapitel6/abb_modellvergleich_r2.png` |
| Abbildung 14 (Beitrag der Datenquellen, Cluster-Bootstrap) | eda/22 | `datenquellen_ebenen.png`, `runs/eda/22_datenquellen.json` |

### Kapitel 7 – Interpretation der Sensoranalyse

| In der Arbeit | Skript | Ausgabe |
|---|---|---|
| Abbildung 15, Tabelle 13 (Güte nach Ebenen, 7.1) | eda/19 → xai/23 | `kapitel7/abb_7_ebenen.png`, `runs/eda/19_auftrag_vs_artikel.json` |
| Abbildung 16 (Top-20 über die Falten) | xai/07 → xai/23 | `kapitel7/abb_7_top20_falten_artikel.png` |
| Anteil Stammdaten / Prozessaufzeichnung am SHAP | xai/17 | `runs/xai/17_herkunftsanteil.json` |
| Abbildung 17 (TabPFN gegen XGBoost) | xai/05, xai/03 → xai/06 | `kapitel7/abb_7_tabpfn_xgboost_top20.png` |
| Abbildungen 18 und 19, Tabelle 14 (Nutzen je Gruppe) | xai/13 → xai/18 → xai/23 | `kapitel7/abb_7_nutzen_sensor_*.png`, `tabellen/tabelle_7_3_sensornutzen.md` |
| Tabelle 15 (Modell-Gegenprobe) | eda/19 + xai/14 → xai/16 → xai/20 | `tabellen/tabelle_16_gegenproben.md` |
| Überlappung der kritischen Bereiche, Werte ohne Abbrüche (7.2, 7.3, 7.5) | xai/13 + xai/14 → xai/25 | `tabellen/tabelle_25_gestoerte_schmelzen.md`, `runs/xai/25_gestoerte_schmelzen.json` |
| Abbildungen 20 und 21 (Wertebereiche) | xai/14, xai/15 → xai/24 | `kapitel7/abb_7_3_sensoren.png`, `kapitel7/abb_7_3_stammdaten.png` |
| Tabelle 16 (Kennwerte der Kanäle) | xai/14 → xai/19 | `tabellen/tabelle_7_4_sensorwerte.md` |
| Abbildung D.1 (Kennzahlauswahl) | xai/22 → xai/23 | `kapitel7/abb_7_kennzahlauswahl.png` |
| Tabellen B.1 und B.3 (Gruppenbildung) | xai/21 | `tabellen/anhang_gruppen_7_4.md` |
| Abbildung 22 (vier Billets), Ausschuss je Glastyp (7.4) | xai/02 → xai/23, eda/31 | `kapitel7/abb_7_lokal.png`, `runs/eda/31_glastypen.json` |
| Gewinn nach Dezilen, Abbildungen E.1 und E.2 (7.4) | eda/21, xai/12 → xai/23 | `21_extreme_billets.pdf`, `kapitel7/abb_7_extreme.png`, `kapitel7/abb_7_erkannt_verfehlt.png` |

Abbildungen 1 und 2 der Arbeit sind Literaturabbildungen; Tabelle 12 und
Tabelle B.2 beschreiben nur die Konfiguration.

### Skripte ohne Ergebnis in der Arbeit

Diese Skripte gehören zur Untersuchung, ihre Ergebnisse werden in der Arbeit
aber nicht berichtet: `hurdle_regression.py` (zweistufiges Modell),
`bootstrap_ci.py` (gepaarte Intervalle der Modellvergleiche), eda/03
(Vorgänger von eda/24), eda/06 (Referenzvorhersagen, Negativkontrolle), eda/07
(Vorarbeit zu eda/17 und eda/18), eda/15 (Vergleich von Extraktionsverfahren),
eda/26 (Prüfung von Wochentag und Stunde), xai/01, xai/04, xai/08–11. xai/03
liefert nur die Vergleichswerte für xai/06.

## Hinweise zum Stand

- **Zwei MiniRocket-Reihen.** `FINALE_KONFIG` führt die Hybrid-Reihe mit und
  ohne z-Normierung getrennt. eda/10 zeigt die Reihe mit, `bootstrap_ci.py`
  rechnet die Intervalle auf der Reihe ohne z-Normierung.
- **HYDRA und FreshPRINCE** liefen je Ebene mit unterschiedlicher z-Normierung;
  `reproduzieren.py` bildet das nach.
- **Faltenmittel und gepoolte Werte.** Kapitel 6 berichtet Faltenmittel
  (`cv_*_mean`). eda/11, eda/12 und eda/29 rechnen gepoolte Out-of-fold-Werte,
  `bootstrap_ci.py` gepoolte Punktschätzer.
- **Eigene Abbildungsdateien.** Einige Abbildungen der Arbeit entstanden als
  Bildschirmfoto oder Ausschnitt einer PDF-Seite. Die Skripte schreiben sie als
  eigene Dateien (eda/32, eda/18, eda/10, xai/07, xai/06); inhaltlich gleich,
  nicht pixelgleich.

## Herkunft

Die Skripte entstanden im Arbeitsrepository des Projekts und wurden für die
Abgabe zusammengestellt (Stand 16.10.2026). Dabei wurde Folgendes geändert:

- Zugangsdaten, Pfade und projektspezifische Bezeichnungen stehen in `.env`,
  `data/beschriftungen.json` bzw. `data/spalten.json`.
- Schalter, für die früher Code editiert wurde, sind Umgebungsvariablen.
- Die Auswahl der finalen Läufe steht an einer Stelle (`runs_io.py`).
- `ergebnistabelle.py` ersetzt eine von Hand gepflegte Excel-Tabelle.
- Werte, die vorher von Hand gerechnet wurden, rechnen jetzt Skripte: eda/25,
  eda/28–32, xai/17, die JSON-Ausgaben von eda/05, eda/19 und eda/23, die
  Tabellen in eda/20 und xai/24 sowie die Zahlen in eda/27.
- Skripte, die nur Word-Dokumente erzeugten, sind nicht enthalten, ebenso
  Explorationswerkzeuge und überholte Experimente.
- Nachgetragen am 17.10.2026: xai/25; xai/16 und xai/20 rechnen seitdem zwei
  weitere Kennzahlen und den Unterschied im Auftrag.
- Nachgetragen am 19.10.2026: eda/33 (Belege zu Abschnitt 5.4 und 5.6, vorher
  nur von Hand geprüft); eda/28 zählt zusätzlich die Stammdatenspalten im
  Rohbestand sowie Kanalkennzahlen und Kopfangaben; jeder Skriptkopf nennt,
  wofür das Skript in der Arbeit gebraucht wird. Spalten- und Kanalnamen sowie
  Kenngrößen der Anlage stehen seitdem in `data/spalten.json` statt im Code.

Diese Skripte kamen aus `tools/` und haben neue Nummern:

| neu | vorher |
|---|---|
| `data_pipeline/convert_sql_db.py` | `tools/convert_sql_db.py` |
| `data_pipeline/arc_end_scan.py` | `tools/_arc_end_scan.py` |
| eda/26, eda/27 | `tools/pruefe_melt_hour.py`, `tools/kapitel5_abbildung.py` |
| xai/18, xai/19, xai/20 | `tools/tabelle_7_3_sensornutzen.py`, `tools/tabellen_7_4_7_5.py`, `tools/tabelle_16_gegenproben.py` |
| xai/21, xai/22 | `tools/anhang_gruppen_7_4.py`, `tools/anhang_kennzahlauswahl.py` |
| xai/23, xai/24 | `tools/kapitel7_abbildungen.py`, `tools/kapitel7_3_abbildungen.py` |
