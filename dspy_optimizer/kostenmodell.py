"""Das Kostenmodell: Dollar in `num_trials` uebersetzen und zurueck.

Die Form ist

    Kosten(Kandidatenmodell) = Fixkosten + num_trials * Kosten je Trial
                               + Ausfuehrungskosten * Zahl der Auswahlbeispiele

Die ersten beiden Summanden sind die Optimierung, der dritte die **Nachbewertung**, die jeder
Optimierungslauf danach selbst bezahlt, ein Programmlauf je Frage der Auswahldaten. Sie kommt
aus demselben Budgetanteil: vor dem Planen wird sie abgezogen, nur der Rest wird in
`num_trials` uebersetzt. Einen Sicherheitsaufschlag gibt es nicht - das Verfahren plant nach
bestem Wissen und weist die Abweichung hinterher aus.

Die **Fixkosten** sind der Block, der bei *jedem* Optimierungslauf erneut anfaellt: Beispiele
sammeln, Instruktionen vorschlagen und eine erste vollstaendige Bewertung. Er haengt nicht an
`num_trials` und ist der Grund, warum ein zu kleiner Budgetanteil nicht einfach wenig Suche
bedeutet, sondern gar keine. Er wird **gemessen** (Fixkosten-Messung der Vorstudie), nicht als
Achsenabschnitt einer Geraden abgelesen.

Die **Kosten je Trial** sind die Steigung: was ein zusaetzlicher Trial im Mittel kostet.
Das ist nicht der Preis eines Programmlaufs mal `minibatch_size`. Die vollstaendigen
Bewertungen *innerhalb* der Schleife feuern alle `minibatch_full_eval_steps + 1` Trials und
stecken mit in der Steigung; sie ist deshalb etwa doppelt so hoch wie die Kosten der
Programmlaeufe eines Minibatches. Die Gerade mittelt ueber diese Treppe hinweg; was sie nicht
trifft, steht hinterher als Planabweichung in der Protokollzeile, statt als ungepruefte
Formel im Code.

`auto` bleibt ausgeschaltet, und `num_candidates`, `minibatch_size` und die Groessen der
Validierungs- und Auswahldaten sind ueber alle Kandidatenmodelle und Phasen konstant -
deshalb ist `num_trials` der einzige Stellhebel, und deshalb genuegt diese einfache Form.

Alle drei gemessenen Parameter kommen **von aussen** und tragen die Herkunft ihres Wertes.
Das Modell bringt selbst keine gemessene Zahl mit: die im DSPy-Quelltext hinterlegte Formel
`_estimate_lm_calls()` ist toter Code und unterschaetzt die Prompt-Model-Aufrufe um etwa
Faktor 3 - gefuellt wird aus Messungen, mit `kalibriere()`.
"""

import math
import statistics
from typing import NamedTuple

# Weniger als ein Trial ist kein Optimierungslauf, sondern ein bezahlter Leerlauf: die
# Fixkosten fallen an, ausprobiert wird nichts. Das gilt fuers Planen. Die Fixkosten-Messung
# der Vorstudie ist kein Optimierungslauf und wird nicht geplant: sie laeuft mit
# `num_trials = 0`, um genau diesen festen Teil zu messen.
MINDESTENS_TRIALS = 1

# Wie weit eine Division danebenliegen darf, um noch als aufgegangen zu gelten. Gemeint sind
# Rundungsfehler der letzten Nachkommastellen, nicht Betraege: relativ 1e-9 sind bei einem
# Lauf fuer 2 $ Milliardstel Cent.
SPIELRAUM = 1e-9


class BudgetReichtNicht(ValueError):
    """Ein Budgetanteil reicht fuer keinen einzigen Trial samt Nachbewertung.

    Eine eigene Klasse, damit ein Aufrufer genau diesen Fall erkennt: das Hauptexperiment haelt
    dann an, statt still weiterzurechnen.
    Ein `ValueError` bleibt sie, weil es ein unbrauchbarer Wert ist.
    """


class Parameter(NamedTuple):
    """Ein Parameter des Kostenmodells samt Herkunft seines Wertes.

    Die Herkunft haengt an der Zahl und nicht in einem Kommentar daneben, weil sie in jede
    Protokollzeile mitwandert: nur so laesst sich spaeter sagen, ob eine Abweichung am Lauf
    lag oder an einem inzwischen neu kalibrierten Modell.
    """

    betrag: float
    herkunft: str


class Messung(NamedTuple):
    """Ein gemessener Optimierungslauf, wie ihn die Kalibrierung braucht: die geplante
    Trial-Zahl und die dafuer tatsaechlich angefallenen Optimierungskosten."""

    num_trials: int
    dollar: float


class Kostenmodell(NamedTuple):
    """Die Uebersetzung zwischen einem Dollar-Betrag und `num_trials`, fuer *ein*
    Kandidatenmodell: alle drei Parameter haengen am Modell und an seinem Preis.

    `kandidatenmodell` ist dessen Name. Er steht hier, weil ein Plan aus diesem Modell nur
    fuer dieses Kandidatenmodell gilt und der Optimierungslauf das pruefen koennen muss.

    `ausfuehrungskosten` sind die Dollar **eines** Programmlaufs eines *optimierten*
    Programms dieses Kandidatenmodells, gemessen in der Vorstudie. Optimierte Programme sind
    teurer als unoptimierte, weil Instruktionen und Beispiele jede Anfrage verlaengern -
    gemessen am unoptimierten Programm waere die Nachbewertung systematisch unterschaetzt.
    """

    kandidatenmodell: str
    fixkosten: Parameter
    kosten_je_trial: Parameter
    ausfuehrungskosten: Parameter

    def _kosten_je_trial_geprueft(self):
        """Die Kosten je Trial, geprueft.

        Die Parameter sind von aussen setzbar, also wird hier geprueft, was `kalibriere()`
        schon beim Fuellen prueft. Ohne diese Pruefung fiele ein Wert von 0,00 $ nicht als
        Fehler auf, sondern als `ZeroDivisionError` mitten in der Budgetplanung.
        """
        if self.kosten_je_trial.betrag <= 0:
            raise ValueError(
                f"Ein Trial muesste Geld kosten, dieses Kostenmodell sagt "
                f"{self.kosten_je_trial.betrag} $ ({self.kosten_je_trial.herkunft}). "
                "Damit ist kein Budgetanteil in Trials zu uebersetzen."
            )
        return self.kosten_je_trial.betrag

    def dollar_nachbewertung(self, auswahlbeispiele):
        """Was die Nachbewertung nach Plan kostet: ein Programmlauf je Auswahlbeispiel.

        Eine Nachbewertung zum Nulltarif gibt es nicht. Eine vergessene Null bei den
        Ausfuehrungskosten waere sonst kein Rechenfehler, sondern ein still zu grosser Plan,
        und jeder Lauf ginge um die ganze Nachbewertung ueber seinen Budgetanteil.
        """
        if auswahlbeispiele < 1:
            raise ValueError(
                f"Eine Nachbewertung ueber {auswahlbeispiele} Auswahlbeispiele "
                "liefert keine Einzelbewertungen, und ohne sie gibt es keinen gepaarten Test "
                "(ADR-0004). So ein Lauf laesst sich nicht planen."
            )
        if self.ausfuehrungskosten.betrag <= 0:
            raise ValueError(
                f"Die Nachbewertung muesste Geld kosten, dieses Kostenmodell sagt "
                f"{self.ausfuehrungskosten.betrag} $ je Programmlauf "
                f"({self.ausfuehrungskosten.herkunft}). Der Plan ginge still um die ganze "
                "Nachbewertung ueber den Budgetanteil (ADR-0017)."
            )
        return auswahlbeispiele * self.ausfuehrungskosten.betrag

    def dollar_optimierung(self, num_trials):
        """Was die Optimierung mit dieser Trial-Zahl nach Plan kostet, ohne Nachbewertung.

        Unterhalb von `MINDESTENS_TRIALS` gibt es keine Antwort, weil es den Lauf nicht gibt,
        nach dem gefragt wird - in beide Richtungen dieselbe Grenze, sonst schloesse sich der
        Rundlauf an ihrem Rand nicht.
        """
        if num_trials < MINDESTENS_TRIALS:
            raise ValueError(
                f"num_trials = {num_trials} ist kein Optimierungslauf. Ein Lauf ohne Trials "
                f"zahlt die Fixkosten von {self.fixkosten.betrag:.6f} $ und sucht nicht."
            )
        return self.fixkosten.betrag + num_trials * self._kosten_je_trial_geprueft()

    def dollar_fuer(self, num_trials, auswahlbeispiele):
        """Der Rueckweg: was ein Optimierungslauf samt Nachbewertung nach Plan kostet."""
        return self.dollar_optimierung(num_trials) + self.dollar_nachbewertung(
            auswahlbeispiele
        )

    def mindestbudget(self, auswahlbeispiele):
        """Der kleinste Budgetanteil, aus dem ein Lauf mit einem Trial samt Nachbewertung wird."""
        return self.dollar_fuer(MINDESTENS_TRIALS, auswahlbeispiele)

    def num_trials_fuer(self, budgetanteil, auswahlbeispiele):
        """Der Hinweg: wie viele Trials von diesem Betrag nach der Nachbewertung bezahlbar sind.

        Abgerundet - ein angefangener Trial wird trotzdem ganz bezahlt. Ein Betrag, der
        rechnerisch genau fuer n Trials reicht, darf dabei nicht an einem Rundungsfehler in
        der letzten Nachkommastelle scheitern und bei n-1 landen; sonst verschenkte jede
        Umrechnung einen Trial. Der Plan kann den Betrag dadurch um bis zu `SPIELRAUM`
        ueberschreiten - Milliardstel Cent, weit unter dem kleinsten abgerechneten Betrag.
        """
        nachbewertung = self.dollar_nachbewertung(auswahlbeispiele)
        genau = (budgetanteil - nachbewertung - self.fixkosten.betrag) / self._kosten_je_trial_geprueft()
        gerundet = round(genau)
        if math.isclose(genau, gerundet, rel_tol=SPIELRAUM, abs_tol=SPIELRAUM):
            num_trials = gerundet
        else:
            num_trials = math.floor(genau)

        if num_trials < MINDESTENS_TRIALS:
            raise BudgetReichtNicht(
                f"Der Budgetanteil von {budgetanteil:.6f} $ reicht fuer keinen einzigen Trial. "
                f"Die Fixkosten liegen bei {self.fixkosten.betrag:.6f} $ "
                f"({self.fixkosten.herkunft}), ein Trial kostet "
                f"{self.kosten_je_trial.betrag:.6f} $ ({self.kosten_je_trial.herkunft}), "
                f"und die Nachbewertung ueber {auswahlbeispiele} Auswahlbeispiele "
                f"kostet {nachbewertung:.6f} $ ({self.ausfuehrungskosten.herkunft}). "
                f"Noetig sind mindestens {self.mindestbudget(auswahlbeispiele):.6f} $ "
                "samt Nachbewertung."
            )
        return num_trials

    def plane(self, budgetanteil, auswahlbeispiele):
        """Der Plan fuer einen Optimierungslauf: Trial-Zahl und die Kosten beider Messlaeufe."""
        num_trials = self.num_trials_fuer(budgetanteil, auswahlbeispiele)
        return Plan(
            num_trials=num_trials,
            auswahlbeispiele=auswahlbeispiele,
            budgetanteil=budgetanteil,
            dollar_optimierung=self.dollar_optimierung(num_trials),
            dollar_nachbewertung=self.dollar_nachbewertung(auswahlbeispiele),
            kostenmodell=self,
        )

    def als_felder(self):
        """Flach und als reine JSON-Werte - so geht das Modell mit ins Protokoll."""
        return {
            "kandidatenmodell": self.kandidatenmodell,
            "fixkosten": self.fixkosten.betrag,
            "fixkosten_herkunft": self.fixkosten.herkunft,
            "kosten_je_trial": self.kosten_je_trial.betrag,
            "kosten_je_trial_herkunft": self.kosten_je_trial.herkunft,
            "ausfuehrungskosten": self.ausfuehrungskosten.betrag,
            "ausfuehrungskosten_herkunft": self.ausfuehrungskosten.herkunft,
        }


class Plan(NamedTuple):
    """Was vor einem Optimierungslauf feststeht - fuer beide seiner Messlaeufe.

    Ein Plan gilt fuer genau ein Kandidatenmodell und genau eine Zahl von
    Auswahlbeispielen. Der Optimierungslauf prueft beides, bevor Geld fliesst: sonst
    plante man fuer ein Kandidatenmodell und bezahlte den Lauf eines anderen, und die
    Planabweichung waere bedeutungslos, ohne dass es auffiele.

    `budgetanteil` ist der zugeteilte Betrag, `dollar_gesamt` der davon verplante - die
    Differenz ist der Rest, der fuer keinen weiteren Trial mehr reicht.
    """

    num_trials: int
    auswahlbeispiele: int
    budgetanteil: float
    dollar_optimierung: float
    dollar_nachbewertung: float
    kostenmodell: Kostenmodell

    @property
    def kandidatenmodell(self):
        """Der Name des Kandidatenmodells, fuer das geplant wurde - das des Kostenmodells."""
        return self.kostenmodell.kandidatenmodell

    @property
    def dollar_gesamt(self):
        """Beide Messlaeufe zusammen: das, was der Budgetanteil bezahlen muss.

        Heisst bewusst nicht `dollar_geplant`: so heisst in der Protokollzeile der Plan fuer
        *einen* Messlauf, und ein Name darf nicht zwei Dinge meinen.
        """
        return self.dollar_optimierung + self.dollar_nachbewertung

    def geplant_fuer(self, abschnitt):
        """Was dieser Messlauf nach Plan kosten soll.

        Geplant werden nur die beiden Messlaeufe eines Optimierungslaufs. Eine Testbewertung
        etwa bezahlt kein Budgetanteil - gegen welchen Plan sollte sie rechnen? `messlauf`
        fragt hier schon vor dem ersten bezahlten Aufruf nach, damit der Fehler nicht erst
        beim Schreiben der Protokollzeile auffaellt und sie mitreisst.
        """
        geplant = {
            "optimierung": self.dollar_optimierung,
            "nachbewertung": self.dollar_nachbewertung,
        }
        if abschnitt not in geplant:
            raise ValueError(
                f"Fuer den Abschnitt {abschnitt!r} gibt es keinen Plan. Geplant werden nur "
                f"die beiden Messlaeufe eines Optimierungslaufs: {', '.join(geplant)}."
            )
        return geplant[abschnitt]

    def abweichung(self, abschnitt, dollar_tatsaechlich):
        """Der Abgleich nach einem Messlauf, gegen dessen Teil des Plans.

        Das Verfahren ist budget-planend:
        es kann einen laufenden Lauf nicht anhalten, die Planabweichung ist deshalb selbst
        ein Messergebnis und kein zu vermeidender Fehler. Jeder der beiden Messlaeufe
        schreibt seine eigene Zeile, also rechnet jede Zeile gegen ihren eigenen Teil - die
        Optimierung gegen die geplante Optimierung, die Nachbewertung gegen die geplante
        Nachbewertung.
        """
        return Planabweichung(
            plan=self, geplant=self.geplant_fuer(abschnitt), tatsaechlich=dollar_tatsaechlich
        )


class Planabweichung(NamedTuple):
    """Geplante und tatsaechliche Kosten eines Messlaufs nebeneinander."""

    plan: Plan
    geplant: float
    tatsaechlich: float

    @property
    def differenz(self):
        """Positiv, wenn der Messlauf teurer war als geplant."""
        return self.tatsaechlich - self.geplant

    @property
    def anteil(self):
        """Die Differenz im Verhaeltnis zum Plan, oder `None`, wenn nichts geplant war.

        Ein Plan ueber 0,00 $ entsteht aus `plane()` nie. Die Abfrage steht hier trotzdem,
        weil diese Zeile am Ende eines bezahlten Laufs geschrieben wird: ein
        `ZeroDivisionError` risse die einzige Aufzeichnung mit, die der Lauf
        hinterlaesst.
        """
        if not self.geplant:
            return None
        return self.differenz / self.geplant

    def als_felder(self):
        """Flach und als reine JSON-Werte - so geht der Abgleich in die Protokollzeile.

        `dollar_geplant` ist der Plan fuer *diesen* Messlauf und steht damit neben `dollar`
        derselben Zeile. `num_trials_geplant` heisst so und nicht `num_trials`, weil die
        Konfiguration des Laufs dieselbe Zahl fuehrt: in einer Zeile duerfen nicht zwei
        Felder gleich heissen und Verschiedenes bedeuten.
        """
        return {
            "num_trials_geplant": self.plan.num_trials,
            "dollar_budgetanteil": self.plan.budgetanteil,
            "dollar_geplant": self.geplant,
            "dollar_abweichung": self.differenz,
            "dollar_abweichung_anteil": self.anteil,
            "kostenmodell": self.plan.kostenmodell.als_felder(),
        }


def kalibriere(messungen, herkunft, *, kandidatenmodell, fixkosten, ausfuehrungskosten):
    """Ein Kostenmodell aus gemessenen Fixkosten und Optimierungslaeufen der Vorstudie.

    `fixkosten` ist ein `Parameter`: die **gemessenen** Fixkosten des Kandidatenmodells, in
    der Vorstudie der Mittelwert seiner Fixkosten-Messungen. Nicht der Achsenabschnitt einer
    Regression: weil die Trials mit wachsender Zahl teurer werden, waere er fuer gpt-oss-120b
    negativ.

    `messungen` sind `Messung`-Paare aus `num_trials` und den gemessenen Optimierungskosten,
    aus Laeufen **desselben** Kandidatenmodells bei sonst gleicher Konfiguration. Aus ihnen
    kommt nur noch die Steigung, die Kosten je Trial: eine Regression durch den Nullpunkt
    ueber das, was jeder Lauf ueber die Fixkosten hinaus gekostet hat. Ohne Aufschlag -
    nach bestem Wissen planen und die Abweichung ausweisen.

    Die Nachbewertung steckt in diesen Dollar nicht - kalibriert wird aus
    Optimierungszeilen (`protokoll.kalibrierbar`). Ihre Kosten kommen aus einer eigenen
    Messung der Vorstudie und werden als `ausfuehrungskosten` (ein `Parameter`)
    mitgegeben.

    Rechnerisch genuegte ein Lauf. Verlangt werden trotzdem zwei verschiedene Trial-Zahlen:
    erst mehrere Laeufe zeigen, wie stark die Steigung streut.
    """
    trials = [float(messung[0]) for messung in messungen]
    dollar = [float(messung[1]) for messung in messungen]
    if len(set(trials)) < 2:
        raise ValueError(
            "Zum Kalibrieren braucht es mindestens zwei Laeufe mit verschiedenen Trial-Zahlen: "
            "erst mehrere Laeufe zeigen, wie stark die Kosten je Trial streuen (ADR-0014, "
            "Nachtrag 10.10.2026, E6)."
        )
    if fixkosten is None or fixkosten.betrag is None:
        raise ValueError(
            "Die Fixkosten fehlen. Sie werden gemessen (Fixkosten-Messung der Vorstudie), und "
            "eine fehlende Messung ist keine Messung von 0 (ADR-0014, Nachtrag 10.10.2026)."
        )
    if fixkosten.betrag <= 0:
        raise ValueError(
            f"Die gemessenen Fixkosten sind {fixkosten.betrag:.6f} $ ({fixkosten.herkunft}). "
            "Ein Lauf ohne Trials kostet mindestens das Sammeln der Beispiele und die erste "
            "vollstaendige Bewertung. Messung pruefen."
        )

    ueber_den_fixkosten = [betrag - fixkosten.betrag for betrag in dollar]
    steigung, _ = statistics.linear_regression(trials, ueber_den_fixkosten, proportional=True)
    if steigung <= 0:
        raise ValueError(
            f"Ueber den Fixkosten von {fixkosten.betrag:.6f} $ bleibt keine Steigung "
            f"({steigung:.6f} $ je Trial). Negative Kosten je Trial wuerden jeden Budgetanteil "
            "in beliebig viele Trials uebersetzen - erst die Rechnung wuerde das aufdecken. "
            "Messreihe pruefen."
        )
    for messung in messungen:
        if messung[1] <= fixkosten.betrag:
            raise ValueError(
                f"Der Lauf mit {messung[0]} Trials hat {messung[1]:.6f} $ gekostet, nicht mehr "
                f"als die gemessenen Fixkosten von {fixkosten.betrag:.6f} $ allein "
                f"({fixkosten.herkunft}). Das passt nicht zusammen; Messreihe pruefen "
                "(ADR-0014, Nachtrag 10.10.2026, E7)."
            )

    return Kostenmodell(
        kandidatenmodell=kandidatenmodell,
        fixkosten=fixkosten,
        kosten_je_trial=Parameter(
            steigung,
            f"{herkunft}; Regression durch den Nullpunkt ueber {len(messungen)} "
            "Optimierungslaeufe abzueglich der gemessenen Fixkosten",
        ),
        ausfuehrungskosten=ausfuehrungskosten,
    )
