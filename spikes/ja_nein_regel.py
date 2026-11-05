"""Nachrechnung: was aendert die Ja/Nein-Regel des offiziellen HotPotQA-F1?

Die Metrik des Experiments ist der allgemeine Wort-F1 (`dspy_optimizer/metrik.py`, gleich
DSPys `f1_score`). Das offizielle HotPotQA-Skript und DSPys `hotpot_f1_score` geben dagegen
0 Punkte, sobald bei einer Ja/Nein-Frage die Antwort nicht exakt "yes" bzw. "no" ist
(`dspy/evaluate/metrics.py`, Z. 183-207). Beispiel: richtig "yes",
geantwortet "yes, they are" -> Wort-F1 0,5, offiziell 0.

Aus den Einzelbewertungen im Protokoll laesst sich das nachrechnen, ohne API-Aufruf: bei einer
Ja/Nein-Goldantwort ist der offizielle Wert 1, wenn der Wort-F1 1 ist, sonst 0. Nicht erfasst
wird die Gegenrichtung (Antwort "yes"/"no", Goldantwort enthaelt das Wort, ist aber laenger);
das ist aus den Einzelbewertungen nicht erkennbar.

Aufruf aus dem Repo-Wurzelordner:
    python -m spikes.ja_nein_regel
"""

import collections
import json
import statistics

from dspy_optimizer.auswertung import Auswertung
from dspy_optimizer.datensatz import lade_aufteilung
from dspy_optimizer.metrik import normalisiert
from dspy_optimizer.protokoll import Protokoll


def ja_nein_stellen(teil):
    return [i for i, beispiel in enumerate(teil) if normalisiert(beispiel.answer) in ("yes", "no")]


def offiziell(einzelbewertungen, stellen):
    werte = list(einzelbewertungen)
    for i in stellen:
        werte[i] = 1.0 if werte[i] == 1.0 else 0.0
    return werte


def testdaten(aufteilung):
    stellen = ja_nein_stellen(aufteilung.testdaten)
    print(f"Testdaten: {len(aufteilung.testdaten)}, davon Ja/Nein: {len(stellen)}")
    zeilen = []
    for _, zeile in Auswertung(Protokoll().zeilen()).testbewertungen().iterrows():
        werte = zeile["einzelbewertungen"]
        teilweise = sum(1 for i in stellen if 0 < werte[i] < 1)
        zeilen.append(
            (zeile["baseline"], zeile["kandidatenmodell"], zeile["seed"],
             statistics.fmean(werte), statistics.fmean(offiziell(werte, stellen)), teilweise)
        )
    for baseline, modell, seed, alt, neu, teilweise in sorted(zeilen):
        print(f"{baseline:18} {modell:30} Seed {seed}  Wort-F1 {alt:.4f}  offiziell {neu:.4f}"
              f"  Differenz {alt - neu:+.4f}  halb richtige Ja/Nein {teilweise}")
    for seed in (1, 2, 3):
        for baseline in ("ohne-optimierung", "gleichverteilung"):
            rangfolge = sorted(
                ((neu, modell) for b, modell, s, _, neu, _ in zeilen if s == seed and b == baseline),
                reverse=True,
            )
            print(f"Seed {seed} {baseline}: " + " > ".join(m for _, m in rangfolge))


def auswahldaten(aufteilung):
    stellen = ja_nein_stellen(aufteilung.auswahldaten)
    print(f"Auswahldaten: {len(aufteilung.auswahldaten)}, davon Ja/Nein: {len(stellen)}")
    je_modell = collections.defaultdict(list)
    with open("ergebnisse/protokoll.jsonl", encoding="utf-8") as datei:
        for zeile in map(json.loads, datei):
            werte = zeile.get("einzelbewertungen")
            if (zeile.get("zweck") != "experiment" or zeile.get("abschnitt") != "nachbewertung"
                    or not werte or len(werte) != len(aufteilung.auswahldaten)):
                continue
            differenz = statistics.fmean(werte) - statistics.fmean(offiziell(werte, stellen))
            teilweise = sum(1 for i in stellen if 0 < werte[i] < 1)
            je_modell[zeile["konfiguration"]["kandidatenmodell"]].append((teilweise, differenz))
    for modell, werte in je_modell.items():
        print(f"{modell:30} Nachbewertungen {len(werte):2}  hoechstens halb richtige "
              f"{max(t for t, _ in werte)}  groesste Differenz {max(d for _, d in werte):.4f}")


if __name__ == "__main__":
    aufteilung = lade_aufteilung()
    testdaten(aufteilung)
    auswahldaten(aufteilung)
