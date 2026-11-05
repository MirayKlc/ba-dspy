"""Die Attrappe des Optimierungslaufs und ihre Zahlen - fuer die Tests des Verfahrens
(`test_verfahren.py`) und der Baselines (`test_baselines.py`).

Beide haengen nur an der Naht `Optimierungslauf`: ihre Logik
haengt an dem, was ein Optimierungslauf zurueckgibt, nicht daran, wie er es misst. Die
Attrappe gibt vorgegebene Zahlen zurueck - kein einziger API-Aufruf.

Die Einzelbewertungen sind von Hand so gebaut, dass der gepaarte t-Test genau das Ergebnis
hat, das ein Test braucht. Der Kniff: weicht ein Kandidatenmodell vom Fuehrenden nur in k von
n Beispielen ab, und dort jeweils um denselben Betrag, haengt der t-Wert nur an k und n,
nicht am Betrag: t = -sqrt(k (n-1) / (n-k)). Bei n = 40 Beispielen ergibt das einseitig
p = 0,16 (k = 1), p = 0,080 (k = 2) und p = 0,042 (k = 3).
"""

import random
import statistics
from typing import NamedTuple

import dspy

from dspy_optimizer.kandidatenmodelle import DIE_VIER
from dspy_optimizer.kostenmodell import Kostenmodell, Parameter
from dspy_optimizer.optimierungslauf import Bewertungsergebnis, Optimierungsergebnis
from dspy_optimizer.protokoll import Einordnung

# Die vier Kandidatenmodelle, in ihrer festen Reihenfolge.
A, B, C, D = (modell.name for modell in DIE_VIER)

N = 40

# Eine Bewertungsreihe, die nach F1 aussieht: Werte zwischen 0,4 und 1,0, fest gewuerfelt.
_wuerfel = random.Random(0)
BASIS = tuple(round(_wuerfel.uniform(0.4, 1.0), 2) for _ in range(N))


def schlechter(an, um=0.2, basis=BASIS):
    """Die Basisreihe, an den Beispielen `an` jeweils um `um` niedriger."""
    return tuple(round(wert - um, 10) if i in an else wert for i, wert in enumerate(basis))


def vertauscht(i, j, basis=BASIS):
    """Die Basisreihe mit zwei vertauschten Werten: anderer Verlauf, exakt derselbe Mittelwert."""
    reihe = list(basis)
    reihe[i], reihe[j] = reihe[j], reihe[i]
    return tuple(reihe)


# Weit hinten: in der Haelfte der Beispiele um 0,3 schlechter. p liegt weit unter 0,05.
WEIT_HINTEN = schlechter(an=range(0, N, 2), um=0.3)

# Dollar je Programmlauf; die Kostenobergrenze steht in Dollar je 1000 Programmlaeufe.
GUENSTIG = 0.0003
TEURER = 0.00035
KOSTENOBERGRENZE = 0.4
ZU_TEUER = 0.0005

OPTIMIERUNGSBUDGET = 4.0


def kostenmodell(name, kosten_je_trial):
    return Kostenmodell(
        kandidatenmodell=name,
        fixkosten=Parameter(0.05, "erfunden"),
        kosten_je_trial=Parameter(kosten_je_trial, "erfunden"),
        ausfuehrungskosten=Parameter(GUENSTIG, "erfunden"),
    )


# Verschieden teure Trials: fuer denselben Dollar-Betrag bekommt A die meisten.
KOSTENMODELLE = (
    kostenmodell(A, 0.01),
    kostenmodell(B, 0.02),
    kostenmodell(C, 0.02),
    kostenmodell(D, 0.03),
)


class Vorgabe(NamedTuple):
    """Was die Attrappe fuer ein Kandidatenmodell zurueckgibt."""

    bewertungen: tuple[float, ...]
    ausfuehrungskosten: float = GUENSTIG
    # Um wie viel der Lauf teurer war als geplant: das Verfahren plant das Budget, es regelt
    # es nicht.
    mehrkosten: float = 0.0
    formatfehler: int = 0
    anbieterfehler: int = 0


class Aufruf(NamedTuple):
    kandidatenmodell: str
    num_trials: int
    seed: int
    einordnung: Einordnung
    plan: object


class Bewertungsaufruf(NamedTuple):
    """Ein Aufruf des zweiten Eingangs, `nur_bewerten`."""

    kandidatenmodell: str
    programm: object
    seed: int
    einordnung: Einordnung
    abschnitt: str


class AttrappeLauf:
    """Ein Optimierungslauf, der nichts optimiert und nichts kostet: er gibt Vorgaben zurueck.

    Er hat dieselbe Aufrufform wie `Optimierungslauf` und dieselbe Rueckgabe
    (`Optimierungsergebnis`, samt Format- und Anbieterfehlerzahlen). Ausgegeben wird genau der Plan,
    zuzueglich der vorgegebenen Mehrkosten.

    `fokus` legt fest, was ein Kandidatenmodell in der Fokusphase zurueckgibt; wer dort
    fehlt, gibt dasselbe zurueck wie in der Sondierung. Das zurueckgegebene Programm traegt,
    wie ein optimiertes, eine eigene Instruktion - sie sagt, aus welcher Phase (bei einer
    Baseline: aus welcher Baseline) und von welchem Kandidatenmodell es stammt - und ein
    Few-Shot-Beispiel je Praediktor.

    Auch der zweite Eingang, `nur_bewerten`, gibt Vorgaben zurueck: dieselben wie die
    Sondierung. Die Bewertung kostet genau einen Programmlauf je Auswahlbeispiel zu den
    vorgegebenen Ausfuehrungskosten, zuzueglich der Mehrkosten.
    """

    def __init__(self, vorgaben, auswahlbeispiele=N, fokus=None):
        self.vorgaben = vorgaben
        self.fokus = fokus or {}
        self.auswahldaten = list(range(auswahlbeispiele))
        self.aufrufe = []
        self.startprogramme = []
        self.bewertungsaufrufe = []

    def vorgabe(self, name, phase):
        if phase == "fokus":
            return self.fokus.get(name, self.vorgaben[name])
        return self.vorgaben[name]

    def __call__(self, kandidatenmodell, startprogramm, num_trials, seed, *, einordnung, plan=None):
        self.aufrufe.append(Aufruf(kandidatenmodell.name, num_trials, seed, einordnung, plan))
        self.startprogramme.append(startprogramm)
        vorgabe = self.vorgabe(kandidatenmodell.name, einordnung.phase)
        return Optimierungsergebnis(
            programm=optimiert(startprogramm, f"{herkunft(einordnung)}: {kandidatenmodell.name}"),
            bewertung=statistics.fmean(vorgabe.bewertungen),
            bewertungen_je_beispiel=tuple(vorgabe.bewertungen),
            optimierungskosten=plan.dollar_optimierung + vorgabe.mehrkosten,
            nachbewertungskosten=plan.dollar_nachbewertung,
            ausfuehrungskosten=vorgabe.ausfuehrungskosten,
            programmlaeufe=35 * num_trials,
            formatfehler=vorgabe.formatfehler,
            anbieterfehler=vorgabe.anbieterfehler,
        )

    def nur_bewerten(self, kandidatenmodell, programm, seed, *, einordnung, abschnitt):
        self.bewertungsaufrufe.append(
            Bewertungsaufruf(kandidatenmodell.name, programm, seed, einordnung, abschnitt)
        )
        vorgabe = self.vorgaben[kandidatenmodell.name]
        programmlaeufe = len(self.auswahldaten)
        return Bewertungsergebnis(
            bewertung=statistics.fmean(vorgabe.bewertungen),
            bewertungen_je_beispiel=tuple(vorgabe.bewertungen),
            bewertungskosten=programmlaeufe * (vorgabe.ausfuehrungskosten or 0.0)
            + vorgabe.mehrkosten,
            ausfuehrungskosten=vorgabe.ausfuehrungskosten,
            formatfehler=vorgabe.formatfehler,
            anbieterfehler=vorgabe.anbieterfehler,
        )


def herkunft(einordnung):
    """Die Phase des Verfahrens, sonst der Name der Baseline."""
    return einordnung.phase or einordnung.baseline


def optimiert(startprogramm, instruktion):
    """Was MIPROv2 aus einem Startprogramm macht, im Kleinen: neue Instruktion, ein Beispiel."""
    programm = startprogramm.deepcopy()
    for _, praediktor in programm.named_predictors():
        praediktor.signature = praediktor.signature.with_instructions(instruktion)
        praediktor.demos = [dspy.Example(question="Frage", answer="Antwort")]
    return programm
