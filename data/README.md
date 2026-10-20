# Daten (nicht im Repository)

Die Daten stammen aus der Fertigung des Praxispartners und sind vertraulich. Sie
gehören nicht in dieses Repository. Wer die Rechnungen ausführen will, legt die
folgenden Dateien in diesen Ordner. Die Skripte finden sie über feste Pfade
relativ zum Projektordner, die Rohkurven über `KURVEN_DIR` (siehe
`.env.example`).

## Für alle Auswertungen nötig

| Datei | Inhalt | Erzeugt von |
|---|---|---|
| `featurized_batches_trimmed_v6.csv` | Merkmalstabelle, eine Zeile je Billet (Zielgröße `y`, Stammdaten, Kennzahlen der Prozesskanäle). Jeder Modelllauf schreibt den SHA-256 der Datei in sein Run-JSON. | `data_pipeline/prepare_data.py` |
| `billets_master.csv`, `billets_master_columns.csv` | Stammdaten je Billet aus der Fertigungsdatenbank und das Datenverzeichnis dazu | `data_pipeline/convert_sql_db.py` |
| `test_billets_v3.json`, `test_billets_auftrag_v3.json`, `test_billets_artikel_v3.json` | Eingefrorene Testmengen (Listen von `batch_id`) der drei Bewertungsebenen | `data_pipeline/data_prep.py` |
| `best_params_v3.json`, `catboost_best_params.json` | Optimierte Hyperparameter von XGBoost und CatBoost | `data_pipeline/tune.py` |
| `spalten.json` | Zuordnung der Rollen im Code (`SP["auftrag"]`, `SP["kanal_a"]` …) zu den Spalten- und Kanalnamen der Quelldaten, dazu einige Kenngrößen der Anlage (letzte Programmphase, Plausibilitätsgrenzen der Kanäle, Zeilenzahl des endgültigen Datenstands) | von Hand, liegt den Daten bei |
| `cache/featurized_batches_trimmed_v6.process_cols.json` | Klassifikation der Spalten der Rohkurven in Prozesskanäle | `data_pipeline/prepare_data.py` |

Neu gezogene Testmengen (`reproduzieren.py --mit-daten`) weichen von denen der
Arbeit ab. Wer die Zahlen der Arbeit nachrechnen will, braucht die
Originaldateien.

## Optional

| Datei / Ordner | Gebraucht von |
|---|---|
| `beschriftungen.json` | Lesbare Bezeichnungen der Kanäle und Merkmale in Abbildungen und Tabellen (`data_prep.beschriftungen`); ohne die Datei stehen die Spaltennamen da |
| Rohkurven `KURVEN_DIR/billet_<nr>.csv` | `prepare_data.py`, `arc_end_scan.py`, MiniRocket/HYDRA/FreshPRINCE (wenn der Kurven-Cache fehlt), eda/13, eda/14, eda/16, eda/22, eda/26, eda/32 |
| `cache/_rocket_series_<hash>.npz` | Kurventensor; entsteht beim ersten Lauf aus den Rohkurven. Liegt er vor, brauchen MiniRocket, HYDRA, FreshPRINCE, eda/13, eda/14 und eda/22 die Rohkurven nicht. |
| `_arc_end_scan.csv` | eda/25 und eda/26; erzeugt von `data_pipeline/arc_end_scan.py` |
| ERP-Exporte (Ordner mit `index.xml`) | `convert_sql_db.py` (`SQL_EXPORTS` in `.env`) |
| InfluxDB | `merge_billetv2.py`, `arc_end_scan.py` (Zugang über `.env`) |
