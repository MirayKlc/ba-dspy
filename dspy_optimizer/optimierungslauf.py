"""Die eine Naht zwischen dem Verfahren und MIPROv2.

Hinein gehen vier Dinge: ein Kandidatenmodell, ein Startprogramm, eine Zahl `num_trials`
und ein Seed - dazu, fuer das Protokoll, wozu der Lauf gehoert (`einordnung`).
Heraus kommen sieben: das optimierte Programm, seine Bewertung auf den Auswahldaten,
die Bewertungen **je einzelnem Auswahlbeispiel**, die Optimierungskosten, die Kosten
der Nachbewertung, die Ausfuehrungskosten und die Zahl der tatsaechlichen Programmlaeufe.
Dazu, getrennt, wie viele Aufrufe an einem Formatfehler und wie viele an einem
Anbieterfehler gescheitert sind: nur so kann das Verfahren erkennen, dass eine
Einzelbewertung von 0,0 nichts ueber das Modell sagt.

Die Einzelbewertungen muessen mit durch die Naht, weil die Zwischenbewertung sie fuer ihren
gepaarten Test braucht. Alles Uebrige -
Trainings-, Validierungs- und Auswahldaten, Vorschlagsmodell, `num_candidates`,
`minibatch_size` - steht beim Bau des `Optimierungslauf` fest und aendert sich zwischen den
Laeufen nicht. Das ist die Bedingung dafuer, dass ein Unterschied zwischen zwei Laeufen
genau einer Ursache zuzuordnen ist.

Ein Lauf zerfaellt in **zwei Messlaeufe** mit je einer Protokollzeile:

1. `optimierung` - der `MIPROv2.compile()`-Aufruf. Was er kostet, sind Optimierungskosten.
   MIPROv2 sammelt aus den Trainingsdaten und waehlt auf den Validierungsdaten aus.
2. `nachbewertung` - eine vollstaendige Bewertung des fertigen Programms ueber die
   Auswahldaten. Sie liefert die Bewertungen je Einzelbeispiel und, als Kosten je
   Programmlauf, die Ausfuehrungskosten. Nicht ueber die Validierungsdaten: auf denen hat
   MIPROv2 das Programm gerade ausgewaehlt, und ihre Bewertung waere zu optimistisch - umso
   mehr, je mehr Trials ein Lauf hatte.

Die beiden sind getrennt, weil Optimierungs- und Ausfuehrungskosten verschiedene Fragen
beantworten und nie miteinander verrechnet werden. Der zweite
Abschnitt kostet zusaetzlich `len(auswahldaten)` Programmlaeufe. Billiger geht es
nicht: MIPROv2 haengt zwar eine Bewertung ans Programm (`best_program.score`), aber keine
Einzelwerte - und diese Bewertung ist *innerhalb* des Optimierungslaufs angefallen und
taugt deshalb nicht als Ausfuehrungskosten.

Die Naht hat einen **zweiten Eingang**, `nur_bewerten`: ein Programm wird nur
bewertet, ohne Optimierung davor - fuer die Baseline ohne Optimierung, die Spannweite der
Vorstudie und die Testbewertung. Er misst mit derselben Methode wie die Nachbewertung oben
und hinterlaesst genau eine Zeile: `nachbewertung` auf den Auswahldaten,
`testbewertung` auf den Testdaten.

Und einen **dritten**, `nur_fixkosten`: die Fixkosten-Messung der Vorstudie. Er
laesst MIPROv2 mit `num_trials = 0` laufen, also nur den festen Teil, und hinterlaesst genau
eine Zeile `fixkosten`, ohne Nachbewertung.
"""

import statistics
import sys
from typing import NamedTuple

import dspy
from dotenv import load_dotenv

from dspy_optimizer.datensatz import lade_aufteilung
from dspy_optimizer.kandidatenmodelle import (
    DAS_VORSCHLAGSMODELL,
    DIE_VIER,
    kandidatenmodell_mit_namen,
    lm_fuer,
    messkontext,
)
from dspy_optimizer.kostenerfassung import messhygiene, messlauf
from dspy_optimizer.metrik import bewerte, bewertung_von, zusammenfassung
from dspy_optimizer.programm import MultiHopQA, lehne_eigenes_lm_ab
from dspy_optimizer.protokoll import (
    HANDLAUF,
    PROTOKOLLDATEI,
    pruefe_einordnung,
    verzeichnis_fuer_lauf,
)

# Wie viele Instruktionen und Beispielsaetze MIPROv2 je Praediktor erzeugt. Ueber alle
# Kandidatenmodelle und Phasen konstant, damit `num_trials` der einzige Hebel bleibt.
# Sechs ist die Zahl, die DSPy selbst fuer `auto="light"` einsetzt
# (`mipro_optimizer_v2.py` Z. 46-50) - `auto` bleibt aus, aber die Groesse der
# Suchraeume von dort zu uebernehmen, ist besser belegt als eine selbst gewaehlte Zahl.
NUM_CANDIDATES = 6

# DSPys Voreinstellungen (`mipro_optimizer_v2.py` Z. 66-67, `compile` Z. 120).
# Sie stehen hier, weil sie ueber alle Laeufe gleich bleiben muessen, nicht weil sie
# besonders gut waeren.
MAX_BOOTSTRAPPED_DEMOS = 4
MAX_LABELED_DEMOS = 4
MINIBATCH_FULL_EVAL_STEPS = 5

# Wie viele Beispiele ein Trial bewertet - DSPys Voreinstellung (`mipro_optimizer_v2.py`
# Z. 121). Eine Festlegung des Versuchsaufbaus, kein Messwert. Das Kostenmodell kennt sie
# nicht: es rechnet mit den gemessenen Kosten je Trial, und in denen steckt sie schon.
# Wer sie aendert, muss neu kalibrieren; deshalb steht sie in jeder Zeile.
MINIBATCH_SIZE = 35

# So viele Beispiele wertet DSPy gleichzeitig aus. Die Zahl geht in die Messung ein: unter
# Last fallen Aufrufe anders aus als einzeln. Der Lasttest vor der Vorstudie
# (docs/spikes/lasttest.json) lief mit genau acht, je Kandidatenmodell ein Optimierungslauf
# in Experimentgroesse, und kam ohne einen einzigen Anbieterfehler durch. DSPys
# Voreinstellung ist dieselbe (`settings.py` Z. 31).
NUM_THREADS = 8


class Optimierungsergebnis(NamedTuple):
    """Was ein Optimierungslauf hinterlaesst - die Rueckgabeseite der Naht.

    `bewertung` ist der Mittelwert ueber `bewertungen_je_beispiel` und liegt wie jede
    Bewertung dieses Projekts zwischen 0 und 1, nicht in Prozent.

    `nachbewertungskosten` sind die Dollar der ganzen Nachbewertung. Zusammen mit den
    `optimierungskosten` ist das, was der Lauf tatsaechlich aus seinem Budgetanteil bezahlt
    hat - und daraus rechnet das Verfahren sein Restbudget.

    `ausfuehrungskosten` sind die Dollar **eines** spaeteren Programmlaufs, gemessen am
    fertigen Programm. Sie sind `None`, wenn nichts gelaufen ist: keine Messung ist etwas
    anderes als eine Messung von null.

    `programmlaeufe` zaehlt den **Optimierungslauf**, nicht die abschliessende Bewertung.
    Gegen den Plan gehalten wird diese Zahl nicht - der Plan kennt Trials, Dollar und die
    Zahl der Auswahlbeispiele, und die Planabweichung rechnet in
    Dollar. Gezaehlt werden dabei nur Programmlaeufe, die durch eine **Evaluation**
    gelaufen sind: die erste vollstaendige Bewertung, die Minibatches der Trials und die
    vollstaendigen Bewertungen dazwischen.
    Die Laeufe des **Bootstrappings** sind nicht darunter - sie gehen nicht durch
    `dspy.Evaluate`, und die Kostenerfassung sieht sie deshalb nur im Geld, nicht im Stueck.
    `optimierungskosten / programmlaeufe` ist folglich **keine**
    Ausfuehrungskosten-Schaetzung; dafuer gibt es `ausfuehrungskosten`.

    `formatfehler` und `anbieterfehler` zaehlen die gescheiterten Aufrufe beider Messlaeufe
    zusammen, nach Ursache getrennt. In der Nachbewertung macht jeder von ihnen
    aus einem Programmlauf still eine Einzelbewertung von 0,0. Beim Formatfehler ist das
    richtig: die Antwort kam an, der Adapter konnte sie nicht lesen - eine Schwaeche des
    Modells. Beim Anbieterfehler nicht: er bleibt nach den Wiederholungen uebrig
    und sagt nichts ueber das Modell. Wie mit einem Anbieterfehler umgegangen wird, regelt
    das Verfahren; die Naht meldet ihn nur. In der Optimierung zaehlen auch Fehler des
    Bootstrappings und des Vorschlagsmodells mit, die keine Bewertung beruehren.
    """

    programm: dspy.Module
    bewertung: float
    bewertungen_je_beispiel: tuple[float, ...]
    optimierungskosten: float
    nachbewertungskosten: float
    ausfuehrungskosten: float | None
    programmlaeufe: int
    formatfehler: int
    anbieterfehler: int


class Bewertungsergebnis(NamedTuple):
    """Was eine Bewertung ohne Optimierung hinterlaesst - die Rueckgabe von `nur_bewerten`.

    Dieselbe Messung wie die Nachbewertung eines Optimierungslaufs, und deshalb dieselben
    Zahlen. `bewertungskosten` sind die Dollar der ganzen Bewertung, `ausfuehrungskosten` die eines
    einzelnen Programmlaufs (`None`, wenn nichts gelaufen ist). `formatfehler` und
    `anbieterfehler` zaehlen nur diese eine Bewertung.

    Exact Match, Absatztreffer, Dauer und Latenz stehen in der Protokollzeile, nicht hier -
    wie beim `Optimierungsergebnis`: die Auswertung liest nur das Protokoll, und wer hier
    etwas entscheidet, braucht sie nicht.
    """

    bewertung: float
    bewertungen_je_beispiel: tuple[float, ...]
    bewertungskosten: float
    ausfuehrungskosten: float | None
    formatfehler: int
    anbieterfehler: int


class Fixkostenergebnis(NamedTuple):
    """Was eine Fixkosten-Messung hinterlaesst - die Rueckgabe von `nur_fixkosten`.

    `fixkosten` sind die Dollar bis zum Ende der ersten vollstaendigen Bewertung, `None`, wenn
    sie nicht zu Ende kam; `kosten` die Dollar der ganzen
    Messung. `bewertung` ist MIPROv2s eigene Bewertung des unveraenderten Programms auf den
    Validierungsdaten, zwischen 0 und 1, `None`, wenn der Lauf kein Programm zurueckgab.
    `formatfehler` und `anbieterfehler` wie beim `Optimierungsergebnis`.
    """

    fixkosten: float | None
    kosten: float
    bewertung: float | None
    formatfehler: int
    anbieterfehler: int


class Optimierungslauf:
    """Die Naht. Gebaut mit allem, was konstant bleibt; gerufen mit dem, was variiert.

    Der Aufruf erwartet, dass die Messhygiene steht - `messhygiene()` einmal am Anfang des
    Skripts. Ob sie steht, prueft jeder Messlauf selbst und bricht sonst ab, bevor Geld fliesst.

    Das Ergebnisprogramm traegt **kein** eigenes LM in seinen Praediktoren. Welches Modell
    es spaeter ausfuehrt, entscheidet weiter der umgebende Kontext. Erst die Endauswahl des
    Verfahrens bindet das Gewinnermodell fest an das Programm; wuerde es hier schon geschehen,
    scheiterte der Warm-Start in die Fokusphase an `lehne_eigenes_lm_ab`.
    """

    def __init__(
        self,
        *,
        trainingsdaten,
        validierungsdaten,
        auswahldaten,
        vorschlagsmodell=DAS_VORSCHLAGSMODELL,
        num_candidates=NUM_CANDIDATES,
        minibatch_size=MINIBATCH_SIZE,
        protokoll=None,
        testdaten=(),
    ):
        self.trainingsdaten = list(trainingsdaten)
        self.validierungsdaten = list(validierungsdaten)
        # Nur fuer die Nachbewertung; MIPROv2 sieht sie nie. Ohne Voreinstellung:
        # jeder Optimierungslauf bewertet danach auf ihnen, und eine vergessene Angabe soll
        # nicht still auf andere Daten ausweichen.
        self.auswahldaten = list(auswahldaten)
        if not self.auswahldaten:
            # Eine Nachbewertung ueber null Beispiele hinterliesse eine Zeile mit Bewertung
            # 0,0: eine erfundene Messung statt einer fehlenden.
            raise ValueError(
                "Ohne Auswahldaten gibt es keine Nachbewertung und damit nichts, worauf "
                "Zwischenbewertung, Endauswahl und Baselines entscheiden koennten (ADR-0025)."
            )
        # Nur fuer die Testbewertung (`nur_bewerten`); die Optimierung sieht sie nie. Leer
        # ist die Voreinstellung: Vorstudie und Handlauf bauen die Naht ohne sie.
        self.testdaten = list(testdaten)
        self.vorschlagsmodell = vorschlagsmodell
        self.num_candidates = num_candidates
        self.minibatch_size = minibatch_size
        self.protokoll = protokoll
        # Einmal gebaut und danach fuer jeden Lauf dasselbe Objekt: das Vorschlagsmodell
        # bleibt ueber alle Kandidatenmodelle konstant, sonst variierten zwei Dinge
        # zugleich.
        self.vorschlags_lm = lm_fuer(vorschlagsmodell)

    def __call__(self, kandidatenmodell, startprogramm, num_trials, seed, *, einordnung, plan=None):
        """Ein vollstaendiger Optimierungslauf samt Nachbewertung.

        `einordnung` sagt, wozu der Lauf gehoert (eine `dspy_optimizer.protokoll.Einordnung`:
        Vorstudie, Handlauf oder Experiment mit Baseline, Phase und Durchgang). Sie landet in
        beiden Protokollzeilen und hat keinen Standardwert: eine vergessene Angabe waere
        sonst still eine falsche.

        `plan` ist der `dspy_optimizer.kostenmodell.Plan`, aus dem `num_trials` stammt,
        sofern es einen gab. Dann steht in beiden Protokollzeilen die Planabweichung, jede
        gegen ihren eigenen Teil des Plans.
        """
        if plan is not None:
            self._pruefe_plan(plan, kandidatenmodell, num_trials)
        # Beide Messlaeufe pruefen das selbst, aber erst, wenn der erste schon Geld kosten
        # koennte und der zweite schon bezahlt ist. Hier faellt es vor beiden auf.
        pruefe_einordnung("optimierung", einordnung)
        # Vor allem anderen, und ausdruecklich vor dem ersten bezahlten Aufruf: ein
        # Praediktor mit eigenem LM staeche Kontext und Settings (`predict.py` Z. 150), der
        # Lauf liefe still auf dem falschen Modell und die Messung waere wertlos.
        lehne_eigenes_lm_ab(startprogramm)

        verzeichnis = verzeichnis_fuer_lauf(kandidatenmodell.name)
        konfiguration = self._konfiguration(kandidatenmodell, num_trials)

        with messkontext(kandidatenmodell) as lm:
            optimiert, optimierung = self._optimiere(
                lm, startprogramm, num_trials, seed, verzeichnis, konfiguration, einordnung, plan
            )
            nachbewertung = self._bewerte(
                lm,
                optimiert,
                self.auswahldaten,
                "nachbewertung",
                seed,
                verzeichnis,
                konfiguration,
                einordnung,
                plan,
            )

        return Optimierungsergebnis(
            programm=optimiert,
            bewertung=nachbewertung.bewertung,
            bewertungen_je_beispiel=nachbewertung.bewertungen_je_beispiel,
            optimierungskosten=optimierung.dollar,
            nachbewertungskosten=nachbewertung.bewertungskosten,
            ausfuehrungskosten=nachbewertung.ausfuehrungskosten,
            programmlaeufe=optimierung.programmlaeufe,
            formatfehler=optimierung.formatfehler + nachbewertung.formatfehler,
            anbieterfehler=optimierung.anbieterfehler + nachbewertung.anbieterfehler,
        )

    def nur_bewerten(
        self,
        kandidatenmodell,
        programm,
        seed,
        *,
        einordnung,
        abschnitt,
        zusaetzliche_konfiguration=None,
    ):
        """Der zweite Eingang: ein Programm nur bewerten, ohne es vorher zu optimieren.

        Gebraucht fuer die Baseline ohne Optimierung, die Spannweite der Kandidatenmodelle
        ohne Optimierung in der Vorstudie und die Testbewertung jedes fertigen
        Modell-Programm-Paars. `num_trials = 0` waere kein Ersatz: es zahlte die Fixkosten
        und suchte nicht - genau das tut, mit Absicht, nur `nur_fixkosten`.

        Gemessen wird mit **derselben** Methode wie die Nachbewertung im Optimierungslauf
        (`_bewerte`) - kein zweiter Weg mit eigenen Fehlern. Deshalb sind die
        Ausfuehrungskosten eines unoptimierten und eines optimierten Programms unmittelbar
        vergleichbar.

        `abschnitt` waehlt die Daten und hat keinen Standardwert, wie `einordnung`:
        `nachbewertung` laeuft ueber die Auswahldaten wie die Nachbewertung eines
        Optimierungslaufs, `testbewertung` ueber die Testdaten. Eine
        vergessene Angabe darf nicht still die Testdaten beruehren - und auch nicht still an
        ihnen vorbeigehen.

        `programm` ist ein Programm **ohne** eigenes LM, wie beim Optimierungslauf; bei
        einem Modell-Programm-Paar ist das `paar.programm_ohne_eigenes_lm()`. Das Programm
        selbst bleibt unveraendert.

        Einen Plan nimmt `nur_bewerten` nicht: eine Testbewertung bezahlt keinen Budgetanteil,
        und das Kostenmodell schaetzt die Nachbewertung mit den Ausfuehrungskosten eines
        **optimierten** Programms.

        `zusaetzliche_konfiguration` sind weitere Felder fuer die Konfiguration der Zeile, reine
        JSON-Werte: bei der Testbewertung, welches gespeicherte Paar bewertet wurde und ob es
        aus einem Durchgang ohne Gewinner stammt. Ein Feld, das die Naht selbst setzt, darf
        darunter nicht sein - es ersetzte still den gemessenen Wert.
        """
        daten = self._daten_fuer(abschnitt)
        # Alles vor dem ersten bezahlten Aufruf, aus denselben Gruenden wie in `__call__`.
        pruefe_einordnung(abschnitt, einordnung)
        lehne_eigenes_lm_ab(programm)
        konfiguration = self._konfiguration_der_bewertung(kandidatenmodell, abschnitt, daten)
        zusaetzliche_konfiguration = zusaetzliche_konfiguration or {}
        doppelt = sorted(set(konfiguration) & set(zusaetzliche_konfiguration))
        if doppelt:
            raise ValueError(
                f"Die zusaetzliche Konfiguration bringt die Felder {', '.join(doppelt)} mit, die "
                "die Naht selbst setzt; sie ersetzten still deren Werte (ADR-0019)."
            )
        konfiguration = {**konfiguration, **zusaetzliche_konfiguration}

        verzeichnis = verzeichnis_fuer_lauf(kandidatenmodell.name)

        with messkontext(kandidatenmodell) as lm:
            return self._bewerte(
                lm, programm, daten, abschnitt, seed, verzeichnis, konfiguration, einordnung
            )

    def nur_fixkosten(self, kandidatenmodell, startprogramm, seed, *, einordnung, wiederholung):
        """Der dritte Eingang: die Fixkosten-Messung der Vorstudie.

        MIPROv2 laeuft mit `num_trials = 0`: Beispiele sammeln, Instruktionen vorschlagen und
        die erste vollstaendige Bewertung des unveraenderten Programms
        (`mipro_optimizer_v2.py` Z. 530-538), danach haelt Optuna an. Derselbe Weg wie die
        Optimierung eines Optimierungslaufs (`_optimiere`), damit der feste Teil genau so
        entsteht wie dort.

        Keine Nachbewertung: das Programm ist unveraendert, und seine Bewertung kennt die
        Vorstudie aus der Bewertung ohne Optimierung. Kein Plan: die Messung bezahlt keinen
        Budgetanteil.

        `wiederholung` ist die Nummer der Messung je Kandidatenmodell (1, 2, 3); sie steht in
        der Konfiguration, sonst saehen die Zeilen gleich aus.
        `einordnung` hat keinen Standardwert wie bei den anderen Eingaengen und muss die
        Vorstudie sein.
        """
        # Alles vor dem ersten bezahlten Aufruf, aus denselben Gruenden wie in `__call__`.
        pruefe_einordnung("fixkosten", einordnung)
        lehne_eigenes_lm_ab(startprogramm)

        verzeichnis = verzeichnis_fuer_lauf(kandidatenmodell.name)
        konfiguration = {
            **self._konfiguration(kandidatenmodell, 0),
            "wiederholung": wiederholung,
        }

        with messkontext(kandidatenmodell) as lm:
            programm, messung = self._optimiere(
                lm, startprogramm, 0, seed, verzeichnis, konfiguration, einordnung, None,
                abschnitt="fixkosten",
            )

        return Fixkostenergebnis(
            fixkosten=messung.fixkosten,
            kosten=messung.dollar,
            bewertung=_bewertung_von_mipro(programm),
            formatfehler=messung.formatfehler,
            anbieterfehler=messung.anbieterfehler,
        )

    def _daten_fuer(self, abschnitt):
        """Welche Daten ein Abschnitt bewertet. Die Optimierung bewertet nicht nur."""
        if abschnitt == "nachbewertung":
            return self.auswahldaten
        if abschnitt == "testbewertung":
            if not self.testdaten:
                # Eine Bewertung ueber null Beispiele hinterliesse eine Zeile mit Bewertung
                # 0,0: eine erfundene Messung statt einer fehlenden.
                raise ValueError(
                    "Diese Naht hat keine Testdaten. Fuer die Testbewertung wird sie mit "
                    "`testdaten=` gebaut - nur im Hauptexperiment (ADR-0008, Issue #12)."
                )
            return self.testdaten
        raise ValueError(
            f"Nur bewerten geht als nachbewertung (Auswahldaten) oder testbewertung "
            f"(Testdaten), nicht als {abschnitt!r}."
        )

    def _pruefe_plan(self, plan, kandidatenmodell, num_trials):
        """Ob der Plan fuer genau diesen Lauf gemacht ist - vor dem ersten bezahlten Aufruf.

        Drei Dinge muessen passen: die Trial-Zahl, das Kandidatenmodell und die Zahl der
        Auswahlbeispiele, nach der die Nachbewertung geplant wurde. Sonst plante man
        fuer ein Kandidatenmodell und bezahlte den Lauf eines anderen, und die
        Protokollzeilen rechneten die Abweichung gegen einen Plan fuer einen anderen Lauf -
        lautlos, waehrend die Planabweichung der Grund ist, warum der Plan ueberhaupt
        mitgereicht wird.
        """
        geplant_und_gelaufen = [
            ("Trials", plan.num_trials, num_trials),
            ("Kandidatenmodell", plan.kandidatenmodell, kandidatenmodell.name),
            ("Auswahlbeispiele", plan.auswahlbeispiele, len(self.auswahldaten)),
        ]
        for was, geplant, gelaufen in geplant_und_gelaufen:
            if geplant != gelaufen:
                raise ValueError(
                    f"Der Plan gilt fuer {was} {geplant}, gelaufen werden soll mit {gelaufen}. "
                    "Die Protokollzeilen wuerden die Abweichung gegen einen Plan fuer einen "
                    "anderen Lauf rechnen (ADR-0014, Issue #17)."
                )

    def _konfiguration(self, kandidatenmodell, num_trials):
        """Was in beide Protokollzeilen gehoert, damit sie ohne den Code lesbar bleiben.

        Die konstant gehaltenen Groessen stehen mit drin: ohne sie liesse sich spaeter nicht
        mehr sagen, ob eine Abweichung am Lauf lag oder an einer geaenderten Einstellung.
        """
        return {
            "kandidatenmodell": kandidatenmodell.name,
            "anbieter": kandidatenmodell.anbieter,
            "vorschlagsmodell": self.vorschlagsmodell.name,
            "programm": "MultiHopQA",
            "num_trials": num_trials,
            "num_candidates": self.num_candidates,
            "minibatch_size": self.minibatch_size,
            "trainingsdaten": len(self.trainingsdaten),
            "validierungsdaten": len(self.validierungsdaten),
            "auswahldaten": len(self.auswahldaten),
        }

    def _konfiguration_der_bewertung(self, kandidatenmodell, abschnitt, daten):
        """Die Konfiguration einer Zeile aus `nur_bewerten`.

        Ohne die Groessen der Optimierung, auch ohne die Zahl der Validierungsdaten: in diesem
        Messlauf hat weder das Vorschlagsmodell noch MIPROv2 etwas getan, und eine Zeile soll
        nichts behaupten, was nicht gelaufen ist. `num_trials` steht leer da statt zu fehlen,
        damit die Spalte in jeder Bewertungszeile existiert: leer heisst, vor dieser Bewertung
        lief in derselben Naht keine Optimierung. Ob das Programm frueher einmal optimiert
        wurde, sagt die Einordnung - etwa die Baseline `ohne-optimierung`.
        """
        return {
            "kandidatenmodell": kandidatenmodell.name,
            "anbieter": kandidatenmodell.anbieter,
            "programm": "MultiHopQA",
            "num_trials": None,
            ("auswahldaten" if abschnitt == "nachbewertung" else "testdaten"): len(daten),
        }

    def _optimiere(
        self,
        lm,
        startprogramm,
        num_trials,
        seed,
        verzeichnis,
        konfiguration,
        einordnung,
        plan,
        abschnitt="optimierung",
    ):
        """Abschnitt 1: der `compile()`-Aufruf. Was er kostet, sind Optimierungskosten.

        Gezaehlt werden **beide** Sprachmodelle: das Kandidatenmodell bewertet und sammelt
        Beispiele, das Vorschlagsmodell schreibt die Instruktionen. Beides ist bezahlte
        Optimierung, und das Vorschlagen gehoert zu den Fixkosten. Die Zeile traegt
        die Fixkosten des Laufs: die Dollar bis zum Ende der ersten vollstaendigen Bewertung.

        `abschnitt` ist `fixkosten` bei der Fixkosten-Messung (`nur_fixkosten`), sonst
        `optimierung`.
        """
        with messlauf(
            lm,
            self.vorschlags_lm,
            protokollverzeichnis=verzeichnis,
            abschnitt=abschnitt,
            einordnung=einordnung,
            konfiguration=konfiguration,
            seed=seed,
            plan=plan,
            protokoll=self.protokoll,
        ) as erfassung:
            optimiert = self._optimierer(lm, seed, verzeichnis).compile(
                startprogramm.deepcopy(),
                trainset=self.trainingsdaten,
                valset=self.validierungsdaten,
                num_trials=num_trials,
                seed=seed,
                minibatch=True,
                minibatch_size=self.minibatch_size,
                minibatch_full_eval_steps=MINIBATCH_FULL_EVAL_STEPS,
                provide_traceback=True,
            )
            erfassung.vermerke_bewertung(_bewertung_von_mipro(optimiert))
        return optimiert, erfassung.messwerte()

    def _optimierer(self, lm, seed, verzeichnis):
        """MIPROv2, eingestellt wie im Experiment.

        `auto=None` mit ausdruecklichem `num_candidates` und `num_trials`: `auto` koppelt
        fuenf Groessen aneinander, ueberschreibt Nutzerangaben und kuerzt die
        Validierungsdaten stillschweigend (`mipro_optimizer_v2.py` Z. 289-317). Fuer ein
        Verfahren, das ein Budget verteilt, ist das der falsche Hebel.

        `teacher_settings={}` steht ausdruecklich da, obwohl es DSPys Voreinstellung ist:
        das Kandidatenmodell sammelt seine Few-Shot-Beispiele **selbst**. Ein gemeinsames
        Lehrermodell machte aus dem Vergleich eine Distillation, und der datenabhaengige
        Aufwand des Sammelns - der unsicherste Posten des Kostenmodells - gehoert zum
        Kandidatenmodell: schwache Modelle scheitern oefter und verbrauchen mehr.

        `log_dir` ist der einzige Absturzschutz: bei jedem Abbruch geht das beste Programm
        sonst verloren.

        `track_stats=True` steht ebenfalls ausdruecklich da, obwohl es DSPys Voreinstellung
        ist. Nur damit haengt MIPROv2 seine Bewertung ans Ergebnisprogramm
        (`mipro_optimizer_v2.py` Z. 675), und daran haengt mehr, als es aussieht: die
        Protokollzeile eines Optimierungslaufs traegt genau dann eine Bewertung, wenn der
        Lauf zu Ende gekommen ist - und das ist der Marker, an dem die Kalibrierung
        abgebrochene von gelungenen Laeufen unterscheidet. Faellt die
        Voreinstellung eines Tages um, waeren alle Bewertungen leer und die Kalibrierung
        faende still keine einzige brauchbare Zeile mehr.
        """
        return dspy.MIPROv2(
            metric=bewerte,
            task_model=lm,
            prompt_model=self.vorschlags_lm,
            teacher_settings={},
            track_stats=True,
            auto=None,
            num_candidates=self.num_candidates,
            max_bootstrapped_demos=MAX_BOOTSTRAPPED_DEMOS,
            max_labeled_demos=MAX_LABELED_DEMOS,
            num_threads=NUM_THREADS,
            max_errors=self._max_errors(self.validierungsdaten),
            seed=seed,
            log_dir=str(verzeichnis),
        )

    def _max_errors(self, daten):
        """Wie viele gescheiterte Programmlaeufe eine Bewertung ueber `daten` ueberlebt.

        DSPys Voreinstellung ist 10 (`settings.py` Z. 32). Das ist hier zu wenig: im
        Lasttest (docs/spikes/lasttest.json) hatten DeepSeek V4.1 Flash 3 und GPT-6 Luna 5
        Formatfehler von 470 in der Optimierung - Text, den der Adapter nicht auswerten kann,
        gegen den Wiederholungen nicht helfen und der als null Punkte zaehlen soll. Eine
        feste Grenze muesste raten, wie viele davon ein
        Optimierungslauf noch bringt; ein knapper Wert risse ihn mitten in der Optimierung ab,
        nachdem das Geld ausgegeben ist.

        Die Grenze liegt deshalb **ein Beispiel ueber** der Groesse der bewerteten Daten:
        selbst wenn jedes Beispiel scheitert, kommt die Bewertung zu Ende. Bewertet werden in
        MIPROv2 die Validierungsdaten, in der Nachbewertung die Auswahldaten und in der
        Testbewertung die Testdaten; die Testbewertung braucht mit 500 Beispielen eine hoehere
        Grenze. Das eine Beispiel mehr ist
        kein Sicherheitspuffer, sondern noetig - DSPy bricht bei `error_count >= max_errors` ab
        (`utils/parallelizer.py` Z. 66), also schon beim letzten Beispiel. Genau bei
        `len(daten)` wuerde ein Kandidatenmodell, dessen Endpunkt jede Anfrage ablehnt, die
        Bewertung bezahlen und den Messlauf dann doch abbrechen - der Fall, den diese Grenze
        verhindern soll.

        Still ist das nicht: der Kostenzaehler meldet gescheiterte Programmlaeufe als
        Stoerung ins Protokoll.
        """
        return len(daten) + 1

    def _bewerte(
        self, lm, programm, daten, abschnitt, seed, verzeichnis, konfiguration, einordnung, plan=None
    ):
        """Die eine Messung fuer jede Bewertung: das Programm ueber `daten`, ohne Optimierung.

        Im Optimierungslauf ist das Abschnitt 2, die Nachbewertung des fertigen Programms
        ueber die Auswahldaten. `nur_bewerten` ruft dieselbe Methode - auf den
        Auswahldaten als Nachbewertung, auf den Testdaten als Testbewertung -, damit
        es fuer beide Eingaenge nur einen Weg gibt.

        Hier entstehen die Bewertungen je Einzelbeispiel. Ihre Reihenfolge ist die der
        Daten, auch bei acht Threads: `Evaluate` legt die Ergebnisse am Ende
        wieder gegen den Datensatz (`evaluate/evaluate.py` Z. 177-180). Auf diese
        Reihenfolge stuetzt sich der gepaarte Test der Zwischenbewertung.

        Nur das Kandidatenmodell wird gezaehlt: das Vorschlagsmodell hat mit dem Ausfuehren
        nichts zu tun, und seine Kosten sind Optimierungskosten.

        Die Protokollzeile traegt zusaetzlich die Einzelbewertungen und die Latenz je
        Programmlauf: die Auswertung liest nur das Protokoll, nicht diese
        Rueckgabe. Die Latenz wird nur berichtet und geht in keine Entscheidung ein.
        """
        with messlauf(
            lm,
            protokollverzeichnis=verzeichnis,
            abschnitt=abschnitt,
            einordnung=einordnung,
            konfiguration=konfiguration,
            seed=seed,
            plan=plan,
            protokoll=self.protokoll,
        ) as erfassung:
            ergebnis = dspy.Evaluate(
                devset=daten,
                metric=bewerte,
                num_threads=NUM_THREADS,
                max_errors=self._max_errors(daten),
                display_progress=False,
                display_table=False,
            )(erfassung.mit_stoppuhr(programm))
            bewertungen = tuple(float(punkte) for *_, punkte in ergebnis.results)
            # Exact Match und Absatztreffer steuern nichts, gehoeren aber je Lauf ins
            # Protokoll. Nur hier liegen die Vorhersagen dafuer vor.
            nebenmasse = zusammenfassung(
                [
                    bewertung_von(beispiel, vorhersage)
                    for beispiel, vorhersage, _ in ergebnis.results
                ]
            )
            bewertung = statistics.fmean(bewertungen) if bewertungen else 0.0
            erfassung.vermerke_bewertung(
                bewertung,
                exakt=nebenmasse.exakt,
                absatztreffer=nebenmasse.absatztreffer,
                einzelbewertungen=bewertungen,
            )
        messwerte = erfassung.messwerte()
        return Bewertungsergebnis(
            bewertung=bewertung,
            bewertungen_je_beispiel=bewertungen,
            bewertungskosten=messwerte.dollar,
            ausfuehrungskosten=messwerte.dollar_je_programmlauf,
            formatfehler=messwerte.formatfehler,
            anbieterfehler=messwerte.anbieterfehler,
        )


def _bewertung_von_mipro(optimiert):
    """Was MIPROv2 selbst gemessen hat, ohne dafuer erneut zu zahlen.

    Die Zahl steht in Prozent (`evaluate.py` Z. 222), im Protokoll stehen Bewertungen zwischen
    0 und 1. Sie fehlt, wenn der Lauf kein Programm gefunden hat - dann bleibt das Feld leer
    statt bei einer erfundenen Null zu landen.
    """
    gemessen = getattr(optimiert, "score", None)
    return None if gemessen is None else gemessen / 100


# So viele Beispiele und Trials benutzt der Lauf von Hand (`main`). Im Experiment sind es
# je 200 Validierungs- und Auswahlbeispiele und so viele Trials, wie der Budgetanteil
# hergibt.
HANDLAUF_GROESSEN = {
    "trainingsdaten": 20,
    "validierungsdaten": 40,
    "auswahldaten": 40,
    "num_trials": 2,
    "seed": 0,
}


def main():
    """Ein winziger echter Optimierungslauf mit zwei Trials, zum Hinsehen.

    Aufruf:  python -m dspy_optimizer.optimierungslauf [Kandidatenmodell]
    Kostet wenige Cent. Ausschliesslich Trainings-, Validierungs- und Auswahldaten, nie die
    Testdaten.

    Der Lauf hinterlaesst zwei Zeilen im fortlaufenden Protokoll, eine je Messlauf.
    """
    # HotPotQA enthaelt Umlaute und Akzente; die Windows-Konsole kann die sonst nicht ausgeben.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv()
    messhygiene()

    gewaehlt = kandidatenmodell_mit_namen(sys.argv[1]) if len(sys.argv) > 1 else DIE_VIER[0]
    aufteilung = lade_aufteilung()
    lauf = Optimierungslauf(
        trainingsdaten=aufteilung.trainingsdaten[: HANDLAUF_GROESSEN["trainingsdaten"]],
        validierungsdaten=aufteilung.validierungsdaten[: HANDLAUF_GROESSEN["validierungsdaten"]],
        auswahldaten=aufteilung.auswahldaten[: HANDLAUF_GROESSEN["auswahldaten"]],
    )
    print(
        f"{gewaehlt.name} (festgenagelt auf {gewaehlt.anbieter}), "
        f"Vorschlagsmodell {lauf.vorschlagsmodell.name}, "
        f"{HANDLAUF_GROESSEN['num_trials']} Trials, "
        f"{len(lauf.validierungsdaten)} Validierungs- und "
        f"{len(lauf.auswahldaten)} Auswahlbeispiele\n"
    )

    # Ein Handlauf, keine Vorstudie: sonst kalibrierte er das Kostenmodell mit.
    ergebnis = lauf(
        gewaehlt,
        MultiHopQA(),
        HANDLAUF_GROESSEN["num_trials"],
        HANDLAUF_GROESSEN["seed"],
        einordnung=HANDLAUF,
    )

    print(f"Bewertung auf den Auswahldaten: {ergebnis.bewertung:.3f} (F1)")
    print(
        f"Einzelbewertungen: {len(ergebnis.bewertungen_je_beispiel)} Stueck, "
        f"die ersten fuenf: "
        + ", ".join(f"{wert:.2f}" for wert in ergebnis.bewertungen_je_beispiel[:5])
    )
    print(
        f"Optimierungskosten ${ergebnis.optimierungskosten:.6f} "
        f"fuer {ergebnis.programmlaeufe} Programmlaeufe"
    )
    print(
        f"Formatfehler {ergebnis.formatfehler}, Anbieterfehler {ergebnis.anbieterfehler} "
        "(beide Messlaeufe)"
    )
    # `None`, wenn kein einziger Programmlauf durchkam. Keine Messung ist etwas anderes als
    # eine Messung von null - und ein Formatfehler hier risse den Bericht eines Laufs mit,
    # der schon bezahlt ist.
    if ergebnis.ausfuehrungskosten is None:
        print("Ausfuehrungskosten: nicht gemessen, es ist kein Programmlauf durchgekommen")
    else:
        print(f"Ausfuehrungskosten ${ergebnis.ausfuehrungskosten:.6f} je Programmlauf")
    for name, instruktion in (
        (name, praediktor.signature.instructions)
        for name, praediktor in ergebnis.programm.named_predictors()
    ):
        print(f"\nInstruktion {name}:\n  {instruktion}")

    print(f"\nZwei Protokollzeilen in {PROTOKOLLDATEI}")


if __name__ == "__main__":
    main()
