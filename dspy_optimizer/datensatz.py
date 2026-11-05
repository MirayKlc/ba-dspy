"""HotPotQA im Distractor-Setting mit fester Aufteilung in vier Datenteile.

Geladen wird direkt von Hugging Face (`hotpotqa/hotpot_qa`, Variante `distractor`), nicht
ueber DSPys Lader: dieser behaelt nur Frage und Antwort, verwirft beim Testteil den Kontext
und laedt die falsche Variante.
"""

import hashlib
import random
import sys
from typing import NamedTuple

import dspy

QUELLE = "hotpotqa/hotpot_qa"
VARIANTE = "distractor"


class Aufteilung(NamedTuple):
    """Die vier Datenteile.

    MIPROv2 sammelt Few-Shot-Beispiele aus den Trainingsdaten und waehlt auf den
    Validierungsdaten aus. Die Auswahldaten dienen der Nachbewertung: auf ihnen entscheiden
    Zwischenbewertung, Endauswahl und Baselines, also auf Fragen, auf denen MIPROv2 nichts
    ausgewaehlt hat. Die Testdaten bleiben bis zur Auswertung unberuehrt.
    """

    trainingsdaten: list
    validierungsdaten: list
    auswahldaten: list
    testdaten: list


GROESSEN = Aufteilung(trainingsdaten=200, validierungsdaten=200, auswahldaten=200, testdaten=500)

# Fest vorgegeben und ausdruecklich *nicht* der Seed des Verfahrens: die Aufteilung ist ueber
# alle Laeufe und alle Seeds dieselbe, damit sich Laeufe nur im Seed des Verfahrens
# unterscheiden.
MISCHUNG_SEED = 0


def _zu_beispiel(roh):
    """Eine Rohzeile als `dspy.Example`.

    Die Feldnamen sind englisch, obwohl der uebrige Code deutsch ist: sie erscheinen als
    Feldnamen in den Prompts der Kandidatenmodelle, und die Aufgabe selbst ist englisch.
    """
    return dspy.Example(
        id=roh["id"],
        question=roh["question"],
        answer=roh["answer"],
        paragraphs=[
            {"title": titel, "sentences": list(saetze)}
            for titel, saetze in zip(roh["context"]["title"], roh["context"]["sentences"])
        ],
        gold_titles=list(dict.fromkeys(roh["supporting_facts"]["title"])),
    ).with_inputs("question", "paragraphs")


def _ist_brauchbar(roh):
    """Ob das Beispiel brauchbar ist: schwere Frage, zehn Absaetze, zwei davon golden.

    Im Trainingsteil kommen vereinzelt Fragen mit weniger Absaetzen oder mit goldenen Titeln
    vor, die in den Absaetzen nicht vorkommen. Solche Beispiele werden uebergangen, damit
    jedes gezogene Beispiel dieselbe Form hat.

    `level == "hard"` schraenkt auf schwere Fragen ein. Der offizielle Entwicklungsteil,
    aus dem die Testdaten stammen, enthaelt ausschliesslich solche; ohne diese Bedingung waeren
    die Testdaten schwerer als die Daten, mit denen optimiert wird.
    """
    titel = list(roh["context"]["title"])
    saetze = list(roh["context"]["sentences"])
    goldene = set(roh["supporting_facts"]["title"])
    return (
        roh["level"] == "hard"
        and len(titel) == 10
        and all(absatz for absatz in saetze)
        and len(goldene) == 2
        and goldene <= set(titel)
        and bool(roh["answer"].strip())
    )


def _waehle(roh, anzahl, schon_vergeben=frozenset()):
    """`anzahl` brauchbare Beispiele, immer dieselben, gezogen aus dem ganzen Teil.

    Gemischt wird mit einem eigenen Zufallsgenerator, nicht mit dem globalen: sonst hinge
    die Aufteilung davon ab, ob zuvor anderswo `random.seed()` aufgerufen wurde. Die
    Reihenfolge liegt fest; eine spaetere Kuerzung (etwa von 500 auf 300 Testdaten) ist also
    ein Anfangsstueck der hier gezogenen Menge.

    `schon_vergeben` sind die Kennungen, die bereits einem anderen Teil angehoeren. Die beiden
    Teile bei Hugging Face ueberschneiden sich nicht, die Aufteilung setzt das aber nicht
    voraus.
    """
    reihenfolge = list(range(len(roh)))
    random.Random(MISCHUNG_SEED).shuffle(reihenfolge)

    gewaehlt = []
    for i in reihenfolge:
        zeile = roh[i]
        if zeile["id"] not in schon_vergeben and _ist_brauchbar(zeile):
            gewaehlt.append(_zu_beispiel(zeile))
            if len(gewaehlt) == anzahl:
                return gewaehlt

    raise ValueError(
        f"Nur {len(gewaehlt)} brauchbare Beispiele gefunden, gebraucht werden {anzahl}."
    )


def aufteilung_aus(roh_trainingsteil, roh_entwicklungsteil):
    """Die feste Aufteilung aus den beiden Rohdaten-Teilen.

    Trainings-, Validierungs- und Auswahldaten kommen aus derselben Ziehung ueber den
    Trainingsteil, in dieser Reihenfolge; die Testdaten aus dem Entwicklungsteil. Die
    Auswahldaten sind das dritte Stueck, hinten angehaengt: so sind Trainings- und
    Validierungsdaten dieselben Fragen wie in einer Aufteilung ohne Auswahldaten, und die
    Laeufe vor der Vorstudie (Handlaeufe, Lasttest) bleiben vergleichbar.
    """
    bis_validierung = GROESSEN.trainingsdaten + GROESSEN.validierungsdaten
    aus_dem_trainingsteil = _waehle(roh_trainingsteil, bis_validierung + GROESSEN.auswahldaten)
    return Aufteilung(
        trainingsdaten=aus_dem_trainingsteil[: GROESSEN.trainingsdaten],
        validierungsdaten=aus_dem_trainingsteil[GROESSEN.trainingsdaten : bis_validierung],
        auswahldaten=aus_dem_trainingsteil[bis_validierung:],
        testdaten=_waehle(
            roh_entwicklungsteil,
            GROESSEN.testdaten,
            schon_vergeben={beispiel.id for beispiel in aus_dem_trainingsteil},
        ),
    )


def lade_aufteilung():
    """Die feste Aufteilung, geladen von Hugging Face.

    Trainings-, Validierungs- und Auswahldaten stammen aus dem offiziellen Trainingsteil, die
    Testdaten aus dem offiziellen Entwicklungsteil. So verfaehrt auch DSPys eigener Lader;
    einen oeffentlichen Testteil mit Antworten gibt es bei HotPotQA nicht. Der erste Aufruf
    laedt den Datensatz herunter, danach liegt er im Cache von `datasets`.
    """
    # Erst hier importiert: so bleibt `aufteilung_aus` testbar, ohne dass `datasets`
    # samt Arrow-Unterbau geladen werden muss.
    from datasets import load_dataset

    return aufteilung_aus(
        load_dataset(QUELLE, VARIANTE, split="train"),
        load_dataset(QUELLE, VARIANTE, split="validation"),
    )


def berechne_fingerabdruck(teil):
    """Die Kennungen eines Teils, in ihrer Reihenfolge, zu einer kurzen Pruefsumme verdichtet:
    die ersten 12 Zeichen von SHA-256 ueber die durch Kommas verbundenen Kennungen. Aendert
    sich der Datensatz bei Hugging Face, aendert sich der Fingerabdruck."""
    return hashlib.sha256(",".join(beispiel.id for beispiel in teil).encode()).hexdigest()[:12]


def zeige(beispiel):
    """Ein Beispiel als Text, der sich von Hand nachpruefen laesst.

    Die beiden goldenen Absaetze erscheinen vollstaendig, da an ihnen haengt, ob die Antwort
    stimmt. Von den acht Ablenkern genuegt der erste Satz.
    """
    golden = set(beispiel.gold_titles)
    zeilen = [
        f"Frage:   {beispiel.question}",
        f"Antwort: {beispiel.answer}",
        f"Kennung: {beispiel.id}",
        "Absaetze (* = golden und vollstaendig, Ablenker nur erster Satz):",
    ]
    for absatz in beispiel.paragraphs:
        ist_golden = absatz["title"] in golden
        saetze = absatz["sentences"] if ist_golden else absatz["sentences"][:1]
        text = " ".join(satz.strip() for satz in saetze)
        zeilen.append(f"  {'*' if ist_golden else ' '} {absatz['title']}: {text}")
    return "\n".join(zeilen)


def main():
    # HotPotQA enthaelt Umlaute und Akzente; die Windows-Konsole kann die sonst nicht ausgeben.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    aufteilung = lade_aufteilung()
    print(
        f"Trainingsdaten: {len(aufteilung.trainingsdaten)}, "
        f"Validierungsdaten: {len(aufteilung.validierungsdaten)}, "
        f"Auswahldaten: {len(aufteilung.auswahldaten)}, "
        f"Testdaten: {len(aufteilung.testdaten)}"
    )
    # Gezeigt werden Trainingsdaten: die Testdaten bleiben bis zur Auswertung unberuehrt.
    for beispiel in aufteilung.trainingsdaten[:3]:
        print()
        print(zeige(beispiel))


if __name__ == "__main__":
    main()
