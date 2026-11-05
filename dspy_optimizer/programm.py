"""Das zweistufige DSPy-Programm fuer HotPotQA im Distractor-Setting.

Modul 1 zieht aus den zehn Absaetzen die zur Frage passenden Stellen heraus, Modul 2
formuliert daraus die Antwort. Es gibt keinen Retriever: die zehn Absaetze werden
mitgeliefert, nachgesucht wird nicht.

Die Feldnamen und die Anweisungen sind englisch, obwohl der uebrige Code deutsch ist: sie
erscheinen woertlich in den Prompts der Kandidatenmodelle, und die Aufgabe selbst ist
englisch. Zudem sind die Anweisungen genau der Text, den MIPROv2 optimiert.

Ebenso englisch sind die beiden Signaturen und die Klasse `MultiHopQA` samt Namen, Docstring
und Kommentar: ihren Quelltext liest das Vorschlagsmodell mit, wenn es Instruktionen schreibt
(`program_aware_proposer`). Dieser Teil der Datei ist deshalb Bestandteil seines Prompts und
bleibt unveraendert.
"""

import sys
from typing import NamedTuple

import dspy
from dotenv import load_dotenv

from dspy_optimizer.datensatz import lade_aufteilung
from dspy_optimizer.kandidatenmodelle import (
    DIE_VIER,
    kandidatenmodell_mit_namen,
    lm_fuer,
    mess_adapter,
    messkontext,
)
from dspy_optimizer.kostenerfassung import messhygiene, messlauf
from dspy_optimizer.metrik import Bewertung, bewertung_von, zusammenfassung
from dspy_optimizer.protokoll import HANDLAUF, PROTOKOLLDATEI, verzeichnis_fuer_lauf

# So viele Beispiele zeigt der Lauf von Hand (`main`); im Experiment sind es 200 je Datenteil.
BEISPIELE_IM_HANDLAUF = 5


class FindEvidence(dspy.Signature):
    """Find the paragraphs needed to answer the question and quote the sentences that matter."""

    question: str = dspy.InputField()
    paragraphs: str = dspy.InputField(desc="ten paragraphs, only a few of them are relevant")
    relevant_titles: list[str] = dspy.OutputField(
        desc="titles of the paragraphs that are needed to answer the question"
    )
    relevant_text: str = dspy.OutputField(
        desc="the sentences from those paragraphs that are needed, quoted verbatim"
    )


class AnswerFromEvidence(dspy.Signature):
    """Answer the question using only the given sentences. Reply with a short phrase."""

    question: str = dspy.InputField()
    relevant_text: str = dspy.InputField(desc="the sentences that were found to be relevant")
    answer: str = dspy.OutputField(desc="a short phrase, not a sentence")


def format_paragraphs(absaetze):
    """Die Absaetze als ein Textblock, so wie Modul 1 sie sieht."""
    return "\n\n".join(
        f"{absatz['title']}: {' '.join(satz.strip() for satz in absatz['sentences'])}"
        for absatz in absaetze
    )


class MultiHopQA(dspy.Module):
    """Answers a multi-hop question from ten given paragraphs, only two of which contain the
    answer. Two steps: first find the evidence sentences, then answer from those sentences
    alone.
    """

    def __init__(self):
        super().__init__()
        self.find_evidence = dspy.Predict(FindEvidence)
        self.answer_question = dspy.Predict(AnswerFromEvidence)

    def forward(self, question, paragraphs):
        evidence = self.find_evidence(question=question, paragraphs=format_paragraphs(paragraphs))
        answer = self.answer_question(
            question=question, relevant_text=evidence.relevant_text
        ).answer
        # The titles are passed on only for logging (paragraph hit rate); the metric ignores them.
        return dspy.Prediction(
            answer=answer,
            relevant_titles=evidence.relevant_titles,
            relevant_text=evidence.relevant_text,
        )


def lehne_eigenes_lm_ab(programm):
    """Vor jedem Lauf: bricht ab, wenn an einem Praediktor ein eigenes LM haengt.

    `MultiHopQA` gibt keinem Praediktor ein eigenes LM. Welches Kandidatenmodell laeuft,
    entscheidet ausschliesslich der umgebende `dspy.context`; diese Funktion stellt das sicher.

    Ein LM am Praediktor hat Vorrang vor `dspy.context` und vor den globalen Einstellungen
    (DSPy-Quelltext `predict.py` Z. 150). Bliebe eines aus einem frueheren Lauf erhalten,
    liefe der Lauf unbemerkt mit dem falschen Kandidatenmodell, und die Messung waere wertlos,
    ohne dass ein Fehler auftraete. Deshalb ein Abbruch statt einer Warnung.
    """
    praediktoren_mit_eigenem_lm = [
        name for name, praediktor in programm.named_predictors() if praediktor.lm is not None
    ]
    if praediktoren_mit_eigenem_lm:
        raise ValueError(
            "Diese Praediktoren haben ein eigenes LM gesetzt und wuerden den Kontext stechen: "
            + ", ".join(praediktoren_mit_eigenem_lm)
        )


class ModellProgrammPaar(dspy.Module):
    """Das Ergebnis des Verfahrens: ein Kandidatenmodell samt dem dafuer optimierten
    Programm. Eine duenne Huelle um das Programm, die sich ohne weitere Einstellungen laden
    und ausfuehren laesst.

    Dafuer bringt das Paar zwei Dinge selbst mit, die sonst der umgebende Kontext setzt:

    1. **Das Kandidatenmodell steht fest in jedem Praediktor.** DSPy speichert das LM eines
       Praediktors mit (`predict/predict.py` Z. 89) und baut es beim Laden wieder auf, samt
       festgelegtem Anbieter, Denken und Temperatur aus `lm_fuer`. Ein LM im Praediktor hat
       Vorrang vor dem Kontext (`predict/predict.py` Z. 150): das Paar laeuft auf seinem
       Modell, unabhaengig von den Einstellungen des Aufrufers.
    2. **Den Adapter setzt `forward` selbst.** DSPy speichert ihn nicht mit. Ohne die Huelle
       liefe ein geladenes Programm mit DSPys Rueckfall auf JSON, der fuer jeden Lauf
       ausgeschlossen ist (`mess_adapter`).

    Gespeichert werden kann auf beide Arten, die DSPy kennt: nur der Zustand als JSON (geladen
    mit `ModellProgrammPaar().load(pfad)`) oder das ganze Programm (`dspy.load`). Den
    API-Schluessel enthaelt das Paar nicht; er kommt wie ueberall aus der Umgebung.

    Ins Verfahren zurueckgegeben werden darf ein Paar nicht: der Optimierungslauf lehnt
    Praediktoren mit eigenem LM ab (`lehne_eigenes_lm_ab`). Weitergereicht wird dort
    `programm`, das Programm ohne Modell, aus dem das Paar entstanden ist.
    """

    def __init__(self, programm=None):
        super().__init__()
        self.programm = programm if programm is not None else MultiHopQA()

    @classmethod
    def aus_programm_und_modell(cls, programm, kandidatenmodell):
        """Das Paar aus einem Programm ohne eigenes LM und einem Kandidatenmodell. Das
        Programm selbst bleibt unveraendert. Alle Praediktoren teilen sich ein LM, damit
        eine Kostenerfassung jeden Aufruf in einer History findet."""
        paar = cls(programm.deepcopy())
        lm = lm_fuer(kandidatenmodell, klasse=dspy.LM)
        for _, praediktor in paar.named_predictors():
            praediktor.lm = lm
        return paar

    @property
    def kandidatenmodell(self):
        """Der Name des Kandidatenmodells, auf dem das Paar laeuft."""
        (modell,) = {praediktor.lm.model for _, praediktor in self.named_predictors()}
        return modell.removeprefix("openrouter/")

    def programm_ohne_eigenes_lm(self):
        """Eine Kopie des Programms ohne LM in den Praediktoren, mit Instruktionen und Beispielen.

        Fuer die Testbewertung eines gespeicherten Paars:
        `nur_bewerten` verlangt ein Programm ohne eigenes LM, denn gezaehlt
        werden nur Aufrufe ueber das LM des Messkontexts (`lehne_eigenes_lm_ab`). Das
        Kandidatenmodell nennt `kandidatenmodell`; der Messkontext baut sein LM mit denselben
        Einstellungen (`lm_fuer`).
        """
        programm = self.programm.deepcopy()
        for _, praediktor in programm.named_predictors():
            praediktor.lm = None
        return programm

    def forward(self, **eingaben):
        with dspy.context(adapter=mess_adapter()):
            return self.programm(**eingaben)


class BewertetesBeispiel(NamedTuple):
    """Ein Beispiel, die Vorhersage des Programms dazu und deren Bewertung."""

    beispiel: object
    vorhersage: dspy.Prediction
    bewertung: Bewertung


def bewerte_beispiele(programm, beispiele):
    """Das Programm ueber wenige Beispiele laufen lassen und jedes einzeln bewerten.

    Nacheinander und ohne Fehlerbehandlung, fuer den Lauf von Hand. Die parallele,
    fehlertolerante Auswertung im Experiment uebernimmt `dspy.Evaluate`.

    Kandidatenmodell und Adapter kommen von aussen, aus `messkontext`.
    """
    lehne_eigenes_lm_ab(programm)
    bewertete_beispiele = []
    for beispiel in beispiele:
        vorhersage = programm(**beispiel.inputs())
        bewertete_beispiele.append(
            BewertetesBeispiel(beispiel, vorhersage, bewertung_von(beispiel, vorhersage))
        )
    return bewertete_beispiele


def main():
    """Ein Lauf von Hand ueber fuenf Trainingsbeispiele mit einem Kandidatenmodell.

    Aufruf:  python -m dspy_optimizer.programm [Kandidatenmodell]
    Kostet einen Bruchteil eines Cent. Verwendet ausschliesslich Trainingsdaten.

    Auch dieser Lauf geht durch die Kostenerfassung und hinterlaesst eine Zeile im
    fortlaufenden Protokoll, denn protokolliert wird jeder API-Aufruf.
    """
    # HotPotQA enthaelt Umlaute und Akzente; die Windows-Konsole kann die sonst nicht ausgeben.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv()

    kandidatenmodell = (
        kandidatenmodell_mit_namen(sys.argv[1]) if len(sys.argv) > 1 else DIE_VIER[0]
    )
    beispiele = lade_aufteilung().trainingsdaten[:BEISPIELE_IM_HANDLAUF]
    print(f"{kandidatenmodell.name} (festgelegt auf {kandidatenmodell.anbieter}), "
          f"{len(beispiele)} Trainingsbeispiele\n")

    messhygiene()
    laufverzeichnis = verzeichnis_fuer_lauf(kandidatenmodell.name)
    with messkontext(kandidatenmodell) as lm:
        with messlauf(
            lm,
            protokollverzeichnis=laufverzeichnis,
            # Keiner der Abschnitte (Optimierung, Nachbewertung, Testbewertung, Fixkosten), sondern
            # ein Lauf von Hand, der nicht in die Auswertung eingeht.
            abschnitt=None,
            einordnung=HANDLAUF,
            konfiguration={
                "kandidatenmodell": kandidatenmodell.name,
                "anbieter": kandidatenmodell.anbieter,
                "beispiele": len(beispiele),
                "teil": "trainingsdaten",
                "programm": "MultiHopQA",
            },
        ) as kostenerfassung:
            bewertete_beispiele = bewerte_beispiele(MultiHopQA(), beispiele)
            # Dieser Lauf geht nicht durch `dspy.Evaluate` und meldet seine Programmlaeufe
            # deshalb selbst; der Callback wuerde sie sonst nicht zaehlen.
            kostenerfassung.vermerke_programmlaeufe(len(bewertete_beispiele))
            mittelwerte = zusammenfassung(
                [bewertetes_beispiel.bewertung for bewertetes_beispiel in bewertete_beispiele]
            )
            kostenerfassung.vermerke_bewertung(
                mittelwerte.f1, exakt=mittelwerte.exakt, absatztreffer=mittelwerte.absatztreffer
            )
    # Die Protokollzeile ist bereits geschrieben: `messlauf` schreibt sie beim Verlassen.
    messwerte = kostenerfassung.messwerte()

    for bewertetes_beispiel in bewertete_beispiele:
        bewertung = bewertetes_beispiel.bewertung
        print(f"Frage:    {bewertetes_beispiel.beispiel.question}")
        print(f"Richtig:  {bewertetes_beispiel.beispiel.answer}")
        print(f"Antwort:  {bewertetes_beispiel.vorhersage.answer}")
        print(f"Absaetze: {', '.join(bewertetes_beispiel.vorhersage.relevant_titles)}"
              f"   (golden: {', '.join(bewertetes_beispiel.beispiel.gold_titles)})")
        print(f"          exakt {bewertung.exakt:.0f}   F1 {bewertung.f1:.2f}   "
              f"goldene Absaetze getroffen {bewertung.absatztreffer:.0%}")
        print()

    print(f"Exact Match {mittelwerte.exakt:.2f}   F1 {mittelwerte.f1:.2f}   "
          "(optimiert wird auf F1)")
    print(f"Goldene Absaetze getroffen: {mittelwerte.absatztreffer:.0%} "
          "- nur protokolliert, geht nicht in die Bewertung ein")
    # Der bedienende Anbieter wird je Aufruf aus der Antwort belegt, nicht vorausgesetzt.
    aufrufe_je_anbieter = ", ".join(
        f"{name} ({aufrufe})" for name, aufrufe in messwerte.anbieter.items()
    )
    print(f"{messwerte.lm_aufrufe} LM-Aufrufe fuer {messwerte.programmlaeufe} Programmlaeufe, "
          f"${messwerte.dollar:.6f} (${messwerte.dollar_je_programmlauf:.6f} je Programmlauf), "
          f"{messwerte.tokens.gesamt} Tokens, bedient von: {aufrufe_je_anbieter}")
    for stoerung in messwerte.stoerungen:
        print(f"Stoerung: {stoerung}")

    print(f"\nProtokollzeile in {PROTOKOLLDATEI}, "
          f"Programme je Trial spaeter in {laufverzeichnis}")


if __name__ == "__main__":
    main()
