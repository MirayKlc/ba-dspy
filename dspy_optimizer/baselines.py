"""Die Baselines, mit denen das Verfahren verglichen wird.

Alle laufen ueber dieselbe Schnittstelle wie das Verfahren (`Optimierungslauf`) und bekommen
dasselbe Optimierungsbudget B in Dollar, damit die *Verteilung* der einzige Unterschied ist:

- **Gleichverteilung** - B zu gleichen Dollar-Teilen auf alle vier Kandidatenmodelle, je B/4,
  in einer einzigen Phase. Der Hauptvergleich der Arbeit.
- **Ein Kandidatenmodell bekommt alles** - ein Optimierungslauf mit ganz B, einmal je
  Kandidatenmodell. Nur so laesst sich sagen, ob das Verfahren besser war, als irgendein
  Modell blind zu waehlen.
- **Ohne Optimierung** - die vier unoptimierten Programme, nur bewertet. Diese Baseline
  optimiert nicht und hat deshalb kein Optimierungsbudget.

Die **teure Referenz** - alle vier mit je B - wird nicht ausgefuehrt. Ihr Protokollwert
`teure-referenz` ist im Protokollformat weiterhin vorgesehen, wird hier aber nicht
geschrieben.

Auch die Baselines planen ueber `num_trials` vorab, bezahlen die Nachbewertung aus ihrem
Optimierungsbudget und brechen bei einem Anbieterfehler ab - ueber dieselben Funktionen wie
das Verfahren (`verfahren.plane`, `verfahren.optimiere`).

**Jede Baseline waehlt nach der Regel der Endauswahl:** erst der Filter ueber die
Kostenobergrenze, dann die hoechste Bewertung aus der Nachbewertung, bei Gleichstand die
niedrigeren Ausfuehrungskosten. Das sind die ersten beiden Schritte der Endauswahl, und
gerufen werden dafuer dieselben Funktionen, `kostenfilter` und `rangfolge`; deren Erster ist
der Fuehrende der Endauswahl. Den gepaarten Test und "hoechstens zwei" gibt es hier nicht:
eine Baseline siebt nichts aus, sie waehlt nur. Liefen diese Schritte mit, stuende in ihrer
Zeile ein Ausscheiden durch einen Test, den die Auswahlregel nicht vorsieht.
"""

from typing import NamedTuple

from dspy_optimizer.kandidatenmodelle import kandidatenmodell_mit_namen
from dspy_optimizer.programm import ModellProgrammPaar
from dspy_optimizer.protokoll import Einordnung, Protokoll
from dspy_optimizer.verfahren import (
    KOSTENFILTER,
    Stand,
    ausgegeben,
    kostenfilter,
    optimiere,
    plane,
    pruefe_messung,
    rangfolge,
    staende_aus,
)


class Baselineverbrauch(NamedTuple):
    """Was ein Durchgang einer Baseline tatsaechlich ausgegeben hat, einschliesslich der
    Nachbewertungen. `uebrig` ist negativ, wenn die Baseline ueberzogen hat: sie plant das
    Budget wie das Verfahren nur vorab und hat, anders als das Verfahren, keine zweite Phase,
    die eine Abweichung ausgleichen koennte.

    Die Baseline ohne Optimierung hat kein Optimierungsbudget: dann sind
    `optimierungsbudget` und `uebrig` `None`, und `ausgegeben` sind die Dollar der vier
    Bewertungen."""

    optimierungsbudget: float | None
    ausgegeben: float

    @property
    def uebrig(self):
        if self.optimierungsbudget is None:
            return None
        return self.optimierungsbudget - self.ausgegeben


class Baselineergebnis(NamedTuple):
    """Was ein Durchgang einer Baseline zurueckgibt.

    `programm` ist das Modell-Programm-Paar des Gewinners, oder `None`, wenn es keinen gibt;
    dann sagt `meldung` warum. `ausgeschieden` nennt, wer an der
    Kostenobergrenze gescheitert ist - einen anderen Grund gibt es bei einer Baseline nicht.

    `programme` haelt je Kandidatenmodell das Programm des Durchgangs, **ohne** eigenes LM -
    so, wie `nur_bewerten` es entgegennimmt. Die Testbewertung braucht alle, nicht nur das
    des Gewinners: Forschungsfrage FF1 vergleicht die Rangfolge aller vier,
    und auch ein Durchgang ohne Gewinner bekommt eine Testbewertung.
    """

    baseline: str
    gewinner: str | None
    programm: ModellProgrammPaar | None
    meldung: str | None
    programme: dict
    staende: tuple[Stand, ...]
    ausgeschieden: dict[str, str]
    budgetverbrauch: Baselineverbrauch

    @property
    def kein_gewinner(self):
        return self.gewinner is None


class Baselines:
    """Die Baselines, konfiguriert mit denselben Angaben wie das `Verfahren`.

    `kostenmodelle` ist je Kandidatenmodell ein `Kostenmodell`; ihre Reihenfolge ist die
    Reihenfolge der Laeufe. `kostenobergrenze` in Dollar je 1000 Programmlaeufe.
    """

    def __init__(self, *, optimierungslauf, kostenmodelle, kostenobergrenze, protokoll=None):
        self.optimierungslauf = optimierungslauf
        self.kostenmodelle = tuple(kostenmodelle)
        self.kostenobergrenze = kostenobergrenze
        self.protokoll = protokoll or Protokoll()

    def gleichverteilung(self, startprogramm, optimierungsbudget, *, seed, durchgang):
        """Ein Durchgang der Gleichverteilung: jedes Kandidatenmodell bekommt B/4.

        Geplant wird fuer alle vier, bevor der erste Lauf Kosten verursacht; ein
        Anbieterfehler bricht den Durchgang ab. Beides geschieht wie im Verfahren
        (`verfahren.plane`, `verfahren.optimiere`).
        """
        return self._optimiere_und_waehle(
            Einordnung.experiment("gleichverteilung", durchgang=durchgang),
            self.kostenmodelle,
            startprogramm,
            optimierungsbudget=optimierungsbudget,
            anteil=optimierungsbudget / len(self.kostenmodelle),
            seed=seed,
        )

    def ein_modell(self, kandidatenmodell, startprogramm, optimierungsbudget, *, seed, durchgang):
        """Ein Durchgang der Baseline "ein Kandidatenmodell bekommt alles": ein einziger
        Optimierungslauf mit dem ganzen Optimierungsbudget B, fuer das Kandidatenmodell mit
        diesem Namen.

        Ausgefuehrt wird sie fuer **jedes** der vier Kandidatenmodelle, und jedes bekommt
        dabei einen eigenen Durchgang mit eigener Nummer: ein Durchgang ist ein Ablauf, dessen
        Optimierungslaeufe sich *ein* Optimierungsbudget teilen, und jedes der vier hat sein
        eigenes B. Unter einer Nummer stuenden sonst 4·B, und die Kosten eines Durchgangs
        (Forschungsfrage FF3) waeren nicht mehr mit denen des Verfahrens vergleichbar. Alle
        vier auszufuehren und die Nummern zu vergeben, ist Aufgabe des Hauptexperiments.

        Reisst das Programm die Kostenobergrenze, ist das Ergebnis "kein Gewinner".
        """
        kostenmodelle = {modell.kandidatenmodell: modell for modell in self.kostenmodelle}
        if kandidatenmodell not in kostenmodelle:
            raise ValueError(
                f"Fuer {kandidatenmodell!r} gibt es kein Kostenmodell. Moeglich sind: "
                f"{', '.join(kostenmodelle)}"
            )
        return self._optimiere_und_waehle(
            Einordnung.experiment("ein-modell", durchgang=durchgang),
            [kostenmodelle[kandidatenmodell]],
            startprogramm,
            optimierungsbudget=optimierungsbudget,
            anteil=optimierungsbudget,
            seed=seed,
        )

    def ohne_optimierung(self, startprogramm, *, seed, durchgang):
        """Ein Durchgang der Baseline ohne Optimierung: das Startprogramm auf jedem
        Kandidatenmodell nur bewertet, ueber die Auswahldaten als Nachbewertung - mit derselben
        Messung wie die Nachbewertung eines optimierten Programms. `num_trials = 0` waere kein
        Ersatz: es zahlte die Fixkosten und suchte nicht. Mit `num_trials = 0` laeuft nur die
        Fixkosten-Messung der Vorstudie (`nur_fixkosten`).

        Kein Optimierungsbudget, kein Plan: die Baseline zaehlt nicht zu B, und das
        Kostenmodell schaetzt Nachbewertungen mit den Ausfuehrungskosten *optimierter*
        Programme. Was die Bewertungen gekostet haben, steht trotzdem in der Zeile.
        """
        einordnung = Einordnung.experiment("ohne-optimierung", durchgang=durchgang)
        ergebnisse = {}
        for modell in self.kostenmodelle:
            name = modell.kandidatenmodell
            ergebnis = self.optimierungslauf.nur_bewerten(
                kandidatenmodell_mit_namen(name),
                startprogramm,
                seed,
                einordnung=einordnung,
                abschnitt="nachbewertung",
            )
            pruefe_messung(name, ergebnis, einordnung)
            ergebnisse[name] = ergebnis
        return self._waehle(
            einordnung,
            seed,
            staende_aus(ergebnisse),
            {name: startprogramm for name in ergebnisse},
            Baselineverbrauch(
                None, sum(ergebnis.bewertungskosten for ergebnis in ergebnisse.values())
            ),
        )

    def _optimiere_und_waehle(
        self, einordnung, kostenmodelle, startprogramm, *, optimierungsbudget, anteil, seed
    ):
        """Jedes Kandidatenmodell bekommt `anteil` Dollar, alle mit demselben Startprogramm
        und Seed; danach die Auswahl."""
        plaene = plane(self.optimierungslauf, kostenmodelle, anteil)
        ergebnisse = optimiere(
            self.optimierungslauf,
            {plan.kandidatenmodell: startprogramm for plan in plaene},
            plaene,
            seed,
            einordnung,
        )
        return self._waehle(
            einordnung,
            seed,
            staende_aus(ergebnisse),
            {name: ergebnis.programm for name, ergebnis in ergebnisse.items()},
            Baselineverbrauch(optimierungsbudget, ausgegeben(ergebnisse)),
        )

    def _waehle(self, einordnung, seed, staende, programme, verbrauch):
        """Die Auswahl nach der Regel der Endauswahl, als Entscheidungszeile im Protokoll:
        Gewinner, wer an der Kostenobergrenze gescheitert ist, je Kandidatenmodell die Zahlen,
        aus denen die Auswahl nachzurechnen ist, und der Verbrauch des Durchgangs."""
        unter_der_obergrenze = kostenfilter(staende, self.kostenobergrenze)
        ausgeschieden = {
            stand.name: KOSTENFILTER for stand in staende if stand not in unter_der_obergrenze
        }
        gewinner = rangfolge(unter_der_obergrenze)[0].name if unter_der_obergrenze else None
        if gewinner is None:
            programm = None
            meldung = (
                f"Kein Gewinner: in der Baseline {einordnung.baseline} reisst "
                f"{', '.join(stand.name for stand in staende)} die Kostenobergrenze von "
                f"{self.kostenobergrenze} $ je 1000 Programmlaeufe (ADR-0020)."
            )
        else:
            programm = ModellProgrammPaar.aus_programm_und_modell(
                programme[gewinner], kandidatenmodell_mit_namen(gewinner)
            )
            meldung = None
        self.protokoll.schreibe_entscheidung(
            abschnitt="endauswahl",
            einordnung=einordnung,
            seed=seed,
            entscheidung={
                "gewinner": gewinner,
                "kein_gewinner": gewinner is None,
                "ausgeschieden": ausgeschieden,
                "kostenobergrenze": self.kostenobergrenze,
                "kandidatenmodelle": {
                    stand.name: {
                        "bewertung": stand.bewertung,
                        "ausfuehrungskosten_je_1000": stand.ausfuehrungskosten_je_1000,
                    }
                    for stand in staende
                },
                "optimierungsbudget": verbrauch.optimierungsbudget,
                "ausgegeben": verbrauch.ausgegeben,
                "uebrig": verbrauch.uebrig,
            },
        )
        return Baselineergebnis(
            baseline=einordnung.baseline,
            gewinner=gewinner,
            programm=programm,
            meldung=meldung,
            programme=programme,
            staende=staende,
            ausgeschieden=ausgeschieden,
            budgetverbrauch=verbrauch,
        )
