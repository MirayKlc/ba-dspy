# ba-dspy: Budgetverteilung auf mehrere Sprachmodelle mit DSPy

Begleitrepository zu meiner Bachelorarbeit. Untersucht wird, wie ein festes
Optimierungsbudget in US-Dollar auf mehrere Sprachmodelle (Kandidatenmodelle) verteilt werden
kann, wenn die Prompts jedes Modells mit DSPy und dem Optimierer MIPROv2 optimiert werden. Das
untersuchte Verfahren gibt in einer Sondierungsphase die Hälfte des Budgets B zu gleichen Teilen
an alle Kandidatenmodelle, scheidet in einer Zwischenbewertung unterlegene Modelle aus, gibt das
Restbudget in einer Fokusphase an die verbleibenden und wählt am Ende ein Programm aus.
Verglichen wird es mit drei Baselines: B1 ohne Optimierung, B2 Gleichverteilung des Budgets,
B3 das gesamte Budget für jeweils ein einzelnes Modell. Aufgabe ist
Multi-Hop-Fragebeantwortung auf HotPotQA im Distractor-Setting.

Das Repository enthält den Code des Verfahrens und der Baselines, die Tests, die Vorstudie, das
Hauptexperiment mit seinem vollständigen Messprotokoll, die gespeicherten Programme und die
Auswertung.

## Ordnerstruktur

| Pfad | Inhalt |
|---|---|
| `dspy_optimizer/` | Experimentcode: Datensatz und Aufteilung, DSPy-Programm, Metrik, Kandidatenmodelle, Kostenerfassung, Kostenmodell, Optimierungslauf, Verfahren, Baselines, Protokoll, Vorstudie, Hauptexperiment, Auswertung, Abbildungen und Tabellen |
| `tests/` | Tests mit `pytest`; Verfahren und Baselines werden gegen eine Attrappe des Optimierungslaufs geprüft, ohne API-Aufrufe |
| `spikes/` | Vortests und Machbarkeitsprüfungen vor dem eigentlichen Experiment (Anbieter, Kostenerfassung, Vorauswahl der Modelle, Lasttests) sowie die nachträgliche Zusatzauswertung `zusatzauswertung.py` |
| `docs/` | Ergänzende Dokumente (siehe unten); `docs/spikes/` enthält die Berichte und Rohdaten der Vortests |
| `ergebnisse/protokoll.jsonl` | Fortlaufendes Messprotokoll aller bezahlten Läufe (Vorstudie, Hauptexperiment, Testbewertung, kleinere Handläufe, dazu Zeilen ohne `zweck` aus der Entwicklungszeit); eine JSON-Zeile je Messlauf, nur angehängt, nie geändert |
| `ergebnisse/handlauf-*.jsonl` | Prüfläufe des Hauptexperiment- und des Verfahrensmoduls, getrennt gehalten, damit sie nicht als Durchgänge zählen |
| `ergebnisse/programme/` | Die im Hauptexperiment gespeicherten optimierten Programme, je gültigem Durchgang ein Ordner |
| `ergebnisse/auswertung/` | Tabellen (CSV), Abbildungen (PDF, PNG) und `bericht.md` der Auswertung; `kapitel-6/` enthält die daraus abgeleiteten Abbildungen und Tabellen der Arbeit |

## Installation

Vorausgesetzt wird Python 3.13. Die Experimente liefen mit den Paketversionen aus
`requirements-lock.txt`; diese Datei sollte zur Reproduktion verwendet werden, weil vor allem
die Kostenberechnung von LiteLLM von der genauen Version abhängt.

```
python -m venv .venv
.venv/bin/pip install -r requirements-lock.txt      # Windows: .venv\Scripts\pip ...
```

`requirements.txt` listet nur die direkten Abhängigkeiten (u. a. `dspy==3.3.1`, `optuna`,
`datasets`, `scipy`, `pandas`, `matplotlib`, `seaborn`, `pytest`).

### Umgebungsvariable

Alle Kandidatenmodelle und das Vorschlagsmodell werden über OpenRouter angesprochen. Für jeden
Lauf mit API-Aufrufen wird ein OpenRouter-Schlüssel in der Umgebungsvariable
`OPENROUTER_API_KEY` benötigt. Die Module laden ihn mit `python-dotenv` aus einer Datei `.env` im
Wurzelverzeichnis, die nicht versioniert wird:

```
OPENROUTER_API_KEY=...
```

Der Datensatz HotPotQA wird beim ersten Aufruf von Hugging Face geladen
(`hotpotqa/hotpot_qa`, Variante `distractor`); dafür ist kein Schlüssel nötig.

## Einstiegspunkte

Alle Befehle werden aus dem Wurzelverzeichnis des Repositorys aufgerufen.

**Kostenpflichtig.** Die folgenden Aufrufe machen bezahlte API-Aufrufe über OpenRouter und hängen
Zeilen an `ergebnisse/protokoll.jsonl` an. Vorstudie und Hauptexperiment setzen am Protokoll an:
sie messen nur, was dort noch fehlt, und wiederholen fertige Läufe nicht.

| Aufruf | Zweck |
|---|---|
| `python -m dspy_optimizer.vorstudie` | Vorstudie: Messungen für Kostenmodell und Kostenobergrenze |
| `python -m dspy_optimizer.hauptexperiment` | Hauptexperiment; misst die fehlenden Durchgänge des laufenden Seeds und hält nach jedem fertigen Seed an |
| `python -m dspy_optimizer.hauptexperiment testbewertung` | Testbewertung auf den Testdaten als letzte Stufe; wird erst nach allen Seeds zugelassen |
| `python -m dspy_optimizer.programm [Kandidatenmodell]` | Prüflauf: das unoptimierte Programm auf fünf Trainingsbeispielen |
| `python -m dspy_optimizer.optimierungslauf [Kandidatenmodell]` | Prüflauf: ein sehr kleiner Optimierungslauf mit zwei Trials |

Vor dem ersten bezahlten Aufruf prüft das Hauptexperiment, dass die Datenaufteilung der
festgehaltenen entspricht und die Vorstudie im Protokoll vollständig ist und die festgelegte
Kostenobergrenze ergibt; andernfalls bricht es ohne API-Aufruf ab.

**Ohne API-Aufruf.** Die Auswertung liest nur das Protokoll und die daraus erzeugten Dateien.

| Aufruf | Ergebnis |
|---|---|
| `python -m dspy_optimizer.datensatz` | Größen der vier Datenteile und drei Beispielfragen (lädt nur den Datensatz) |
| `python -m dspy_optimizer.vorstudie auswertung` | Bericht der Vorstudie aus dem Protokoll |
| `python -m dspy_optimizer.hauptexperiment stand` | Kosten des Hauptexperiments gegen den Plan |
| `python -m dspy_optimizer.auswertung` | Tabellen, Abbildungen und `bericht.md` in `ergebnisse/auswertung/` |
| `python -X utf8 spikes/zusatzauswertung.py > docs/zusatzauswertung-kapitel-6.md` | Zusätzliche Statistiken (u. a. Wilcoxon-Tests); setzt die Auswertung voraus |
| `python -m dspy_optimizer.abbildungen_und_tabellen_kapitel_6` | Abbildungen und Tabellen der Arbeit in `ergebnisse/auswertung/kapitel-6/`; setzt Auswertung und Zusatzauswertung voraus |

Die drei letzten Aufrufe sind in dieser Reihenfolge auszuführen. Die Zusatzauswertung schreibt auf
die Standardausgabe; `-X utf8` sorgt für UTF-8, das der letzte Aufruf erwartet. Unter Windows ist
die Umleitung in `cmd` oder einer Bash-Shell auszuführen, nicht in Windows PowerShell 5.1, deren `>`
UTF-16 schreibt.

## Tests

```
python -m pytest
```

Die Tests machen keine API-Aufrufe an Sprachmodelle und verursachen keine Kosten. Ein Test der
Datenaufteilung lädt HotPotQA von Hugging Face und wird übersprungen, wenn der Datensatz nicht
erreichbar ist.

## Lange Läufe

Ein Seed des Hauptexperiments dauerte ohne Störungen 17 Stunden, mit Ratenbegrenzungen beim
Anbieter und den dadurch nötigen Neustarts bis zu rund 43 Stunden. Lange Läufe sollten deshalb so gestartet
werden, dass sie das Ende der Terminalverbindung überstehen, etwa in einer `tmux`-Sitzung auf
einem Server:

```
tmux new -s lauf
mkdir -p ergebnisse/laeufe
python -u -m dspy_optimizer.hauptexperiment 2>&1 | tee -a ergebnisse/laeufe/<JJJJ-MM-TT>-hauptexperiment.log
```

Die Sitzung wird mit Strg+b, d verlassen, ohne den Lauf zu beenden, und mit
`tmux attach -t lauf` wieder aufgenommen. `-u` schreibt die Ausgabe ungepuffert ins Log;
`tee -a` hängt an ein vorhandenes Log an, sodass ein Neustart am selben Tag das Log des
vorherigen Laufs nicht überschreibt. Der Ordner `ergebnisse/laeufe/` wird nicht versioniert und
muss deshalb vor dem ersten Lauf angelegt werden, weil `tee` die Datei öffnet, bevor das Modul
startet.

Ein Lauf wird mit Strg+C angehalten, nicht durch Beenden der Sitzung: so schreibt der laufende
Messlauf seine Protokollzeile noch. Nach einem Abbruch startet derselbe Befehl den Lauf neu;
wiederholt wird nur der angefangene Durchgang. Meldet ein Durchgang einen Anbieterfehler, hält
das Modul mit `HauptexperimentAngehalten` (bzw. `VorstudieAngehalten`) an; die Ursache steht im
Feld `stoerungen` der letzten Protokollzeilen.

## Dokumentation

| Dokument | Inhalt |
|---|---|
| [`docs/vorstudie-ergebnisse.md`](docs/vorstudie-ergebnisse.md) | Ergebnisse der Vorstudie und die daraus festgelegten Werte |
| [`docs/zusatzauswertung-kapitel-6.md`](docs/zusatzauswertung-kapitel-6.md) | Ausgabe der Zusatzauswertung (erzeugte Datei) |

Kommentare und Docstrings im Code sowie einige erzeugte Ausgaben (`ergebnisse/auswertung/bericht.md`,
die Ausgaben von `hauptexperiment stand` und `vorstudie auswertung`) enthalten Verweise der Form
`ADR-…` und `Issue #…` auf Arbeitsunterlagen, die nicht Teil dieses Repositorys sind.
