"""Das Verfahren: Sondierungsphase, Zwischenbewertung, Fokusphase und Endauswahl.

Das Verfahren haengt nur von der Schnittstelle `Optimierungslauf` und vom Kostenmodell ab,
nicht von MIPROv2 selbst. Deshalb ist seine gesamte Logik gegen eine Attrappe des
Optimierungslaufs pruefbar, ohne API-Aufrufe.

**Sondierungsphase.** Die Haelfte des Optimierungsbudgets geht zu gleichen Dollar-Teilen an
die vier Kandidatenmodelle, jedes bekommt also B/8. Jeder Anteil wird ueber das Kostenmodell
des Kandidatenmodells in `num_trials` uebersetzt und bezahlt auch die Nachbewertung. Gleiche
Dollar bedeuten nicht gleich viel Suche: Ein Kandidatenmodell mit billigeren Trials bekommt
mehr davon.

**Zwischenbewertung.** Vier Schritte in fester Reihenfolge:

1. Kostenfilter: wer ueber der Kostenobergrenze liegt, scheidet aus.
2. Der Fuehrende: das Kandidatenmodell mit der hoechsten Bewertung unter den Uebrigen.
3. Gepaarter t-Test, einseitig, alpha = 0,05, jedes uebrige gegen den Fuehrenden. Wer
   nachweisbar schlechter ist, scheidet aus - aber nur, wenn auch alle schlechter bewerteten
   nachweisbar schlechter sind. Der Fuehrende geht also nur allein weiter, wenn der Test
   alle anderen als schlechter erkennt, und ein besser bewertetes Kandidatenmodell wird nie
   von einem schlechter bewerteten ueberholt.
4. Obergrenze: bleiben mehr als zwei, gehen die besten zwei weiter.

Stuende der Test vor dem Kostenfilter, koennte er alle gegen einen Fuehrenden
hinauswerfen, den der Kostenfilter danach selbst entfernt - dann bliebe niemand.

Schritt 3 wirft nicht einfach jedes nachweisbar schlechtere hinaus, weil der Test misst, wie
*sicher* ein Kandidatenmodell schlechter ist, nicht wie *viel*. Ein Zweitbester mit kleinem,
aber sehr gleichmaessigem Rueckstand schiede sonst aus, waehrend ein Drittbester mit
groesserem, unregelmaessigem Rueckstand bliebe und an seiner Stelle weiterginge.

Bei Gleichstand der Bewertung - beim Fuehrenden wie bei "hoechstens zwei" - gewinnen die
niedrigeren Ausfuehrungskosten. Sind auch diese gleich, entscheidet die Reihenfolge, in der
die Kandidatenmodelle gelaufen sind - im Experiment die feste Reihenfolge aus `DIE_VIER` -,
damit dieselben Eingaben immer dieselbe Entscheidung ergeben.

**Fokusphase.** Das Restbudget - B minus die tatsaechlich ausgegebenen Dollar der Sondierung -
geht zu gleichen Dollar-Teilen an die Ueberlebenden; ueberlebt nur eines, bekommt es alles.
Jedes startet mit seinem Programm aus der Sondierung, aber ohne dessen Few-Shot-Beispiele
(Warm-Start).

**Endauswahl.** Dieselben vier Schritte wie die Zwischenbewertung, auf den Laeufen der
Fokusphase; Gewinner ist der Fuehrende. Reisst jedes verbliebene Modell-Programm-Paar die
Kostenobergrenze, gibt es keinen Gewinner, sondern eine Meldung und als Zusatzangabe das
guenstigere Programm der Sondierung. Zurueckgegeben wird ein `ModellProgrammPaar`, das sich
direkt laden und ausfuehren laesst.
"""

import math
import statistics
from typing import NamedTuple

from scipy import stats

from dspy_optimizer.kandidatenmodelle import kandidatenmodell_mit_namen
from dspy_optimizer.programm import ModellProgrammPaar
from dspy_optimizer.protokoll import Einordnung, Protokoll

# Die Sondierung bekommt die Haelfte des Optimierungsbudgets.
SONDIERUNGSANTEIL = 0.5

# Irrtumswahrscheinlichkeit, ein Kandidatenmodell hinauszuwerfen, das in Wahrheit gleich gut
# ist wie der Fuehrende: einer von zwanzig Faellen. Ohne Korrektur fuer Mehrfachtests,
# wie bei CAPO (Zehle et al. 2025).
ALPHA = 0.05

# Hoechstens so viele Kandidatenmodelle gehen in die Fokusphase.
HOECHSTENS_UEBERLEBENDE = 2

# Die Kostenobergrenze ist in Dollar je 1000 Programmlaeufe angegeben, die Ausfuehrungskosten
# aus dem Optimierungslauf in Dollar je Programmlauf.
PROGRAMMLAEUFE_JE_KOSTENOBERGRENZE = 1000

# Gruende fuer das Ausscheiden eines Kandidatenmodells, wie sie in der Entscheidungszeile
# stehen. Der dritte Grund ist Schritt 4: nicht nachweisbar schlechter, aber nicht unter den
# besten zwei.
KOSTENFILTER = "kostenfilter"
TEST = "test"
NICHT_UNTER_DEN_ZWEI = "nicht-unter-den-zwei"


class UngueltigerDurchgang(RuntimeError):
    """Ein Durchgang, auf dessen Grundlage keine Entscheidung fallen darf.

    Er wird mit einer neuen Durchgangsnummer wiederholt; seine Zeilen bleiben im Protokoll
    stehen, da die Kosten trotzdem angefallen sind.
    """


class Stand(NamedTuple):
    """Die Angaben, die die Zwischenbewertung ueber ein Kandidatenmodell benoetigt.

    `ausfuehrungskosten` in Dollar je Programmlauf, wie sie aus der Nachbewertung kommen.
    """

    name: str
    bewertungen_je_beispiel: tuple[float, ...]
    ausfuehrungskosten: float

    @property
    def bewertung(self):
        """Die Bewertung: der Mittelwert der Einzelbewertungen."""
        return statistics.fmean(self.bewertungen_je_beispiel)

    @property
    def ausfuehrungskosten_je_1000(self):
        """In der Einheit der Kostenobergrenze: Dollar je 1000 Programmlaeufe."""
        return self.ausfuehrungskosten * PROGRAMMLAEUFE_JE_KOSTENOBERGRENZE


def kostenfilter(staende, kostenobergrenze):
    """Schritt 1: die Staende, die die Kostenobergrenze einhalten. Der Grenzwert selbst gilt
    als eingehalten."""
    return [stand for stand in staende if stand.ausfuehrungskosten_je_1000 <= kostenobergrenze]


def rangfolge(staende):
    """Die Staende von der hoechsten Bewertung abwaerts; bei Gleichstand die niedrigeren
    Ausfuehrungskosten zuerst. `sorted` ist stabil: sind auch die Kosten gleich, bleibt die
    Reihenfolge der Eingabe."""
    return sorted(staende, key=lambda stand: (-stand.bewertung, stand.ausfuehrungskosten))


def p_wert(stand, fuehrender):
    """Schritt 3 fuer ein Kandidatenmodell: der einseitige p-Wert des gepaarten t-Tests.

    Gepaart heisst: verglichen wird Frage fuer Frage, auf denselben Auswahlbeispielen
    in derselben Reihenfolge. Einseitig, weil die Richtung feststeht - der
    Fuehrende hat den hoechsten Mittelwert, gefragt ist nur, ob der andere *schlechter* ist.

    `None`, wenn die beiden in keinem einzigen Beispiel voneinander abweichen: dann ist der
    t-Wert 0/0, es gibt keinen Unterschied und damit auch nichts nachzuweisen.
    """
    if len(stand.bewertungen_je_beispiel) != len(fuehrender.bewertungen_je_beispiel):
        raise ValueError(
            f"{stand.name} hat {len(stand.bewertungen_je_beispiel)} Einzelbewertungen, der "
            f"Fuehrende {fuehrender.name} {len(fuehrender.bewertungen_je_beispiel)}. Bewertungen "
            "ueber verschiedene Auswahlbeispiele lassen sich nicht paaren (ADR-0004)."
        )
    ergebnis = stats.ttest_rel(
        stand.bewertungen_je_beispiel, fuehrender.bewertungen_je_beispiel, alternative="less"
    )
    p = float(ergebnis.pvalue)
    return None if math.isnan(p) else p


class Entscheidung(NamedTuple):
    """Das Ergebnis der Zwischenbewertung.

    `weiter` in der Rangfolge, der Fuehrende zuerst. `ausgeschieden` sagt je
    Kandidatenmodell, woran es gescheitert ist: `kostenfilter`, `test` oder
    `nicht-unter-den-zwei`. Ein nachweisbar schlechteres, das bleibt, weil ein schlechter
    bewertetes nicht nachweisbar schlechter ist, steht je nach Rang unter `weiter` oder als
    `nicht-unter-den-zwei` da.
    `p_werte` hat einen Eintrag je Vergleich mit dem Fuehrenden, also fuer jedes, das den
    Kostenfilter ueberstanden hat und nicht selbst der Fuehrende ist.
    """

    weiter: tuple[str, ...]
    ausgeschieden: dict[str, str]
    fuehrender: str | None
    p_werte: dict[str, float | None]

    @property
    def test_hat_gefunden(self):
        """Ob mindestens ein Kandidatenmodell nachweisbar schlechter ist als der Fuehrende.
        Findet der Test nie etwas, ist das ein Befund ueber kleine Sondierungsbudgets
        (Forschungsfrage FF2).

        Gefragt ist, ob die Sondierung signifikant unterscheiden konnte, nicht, ob der Test
        jemanden hinausgeworfen hat. Das ist nicht dasselbe: ein nachweisbar schlechteres
        Kandidatenmodell kann bleiben, weil ein schlechter bewertetes nicht nachweisbar
        schlechter ist. Wen der Test hinausgeworfen hat, steht unter `ausgeschieden`.
        """
        return any(p is not None and p < ALPHA for p in self.p_werte.values())

    @property
    def vergleiche(self):
        """Anzahl der Vergleiche mit dem Fuehrenden."""
        return len(self.p_werte)


def zwischenbewertung(staende, kostenobergrenze):
    """Die vier Schritte in ihrer festen Reihenfolge. Zwischenbewertung und Endauswahl rufen
    beide diese Funktion.

    `kostenobergrenze` in Dollar je 1000 Programmlaeufe. Laesst der Kostenfilter niemanden
    durch, gibt es weder einen Fuehrenden noch einen Test.
    """
    ausgeschieden = {}

    uebrig = kostenfilter(staende, kostenobergrenze)
    for stand in staende:
        if stand not in uebrig:
            ausgeschieden[stand.name] = KOSTENFILTER
    if not uebrig:
        return Entscheidung(weiter=(), ausgeschieden=ausgeschieden, fuehrender=None, p_werte={})

    fuehrender, *andere = rangfolge(uebrig)

    # Jedes Kandidatenmodell wird gegen den Fuehrenden getestet; jeder p-Wert wird protokolliert.
    p_werte = {
        stand.name: p_wert(stand, fuehrender) for stand in uebrig if stand is not fuehrender
    }

    # Schritt 3, vom schlechtesten Rang aufwaerts: wer nachweisbar schlechter ist, scheidet nur
    # aus, wenn alle schlechter bewerteten ebenfalls ausscheiden. Das erste Kandidatenmodell
    # von hinten, das bleibt, haelt alle besser bewerteten im Rennen.
    nach_dem_test = [fuehrender, *andere]
    for stand in reversed(andere):
        p = p_werte[stand.name]
        if p is None or p >= ALPHA:
            break
        ausgeschieden[stand.name] = TEST
        nach_dem_test.remove(stand)

    weiter = nach_dem_test[:HOECHSTENS_UEBERLEBENDE]
    for stand in nach_dem_test:
        if stand not in weiter:
            ausgeschieden[stand.name] = NICHT_UNTER_DEN_ZWEI

    return Entscheidung(
        weiter=tuple(stand.name for stand in weiter),
        ausgeschieden=ausgeschieden,
        fuehrender=fuehrender.name,
        p_werte=p_werte,
    )


class Zwischenstand(NamedTuple):
    """Das Ergebnis der Sondierung, soweit die Fokusphase es benoetigt.

    `ergebnisse` sind die Optimierungsergebnisse je Kandidatenmodell, in der Reihenfolge der
    Laeufe; aus ihnen kommen die Programme fuer den Warm-Start. `ausgegeben` sind die
    tatsaechlichen Dollar aller Messlaeufe der Sondierung, Optimierung und Nachbewertung.
    """

    ergebnisse: dict
    staende: tuple[Stand, ...]
    entscheidung: Entscheidung
    kostenobergrenze: float
    optimierungsbudget: float
    ausgegeben: float

    @property
    def restbudget(self):
        """Das Optimierungsbudget minus die tatsaechlich ausgegebenen, nicht die geplanten
        Dollar."""
        return self.optimierungsbudget - self.ausgegeben

    @property
    def kein_gewinner(self):
        """Hat schon der Kostenfilter alle ausgesiebt, endet das Verfahren hier: die
        Fokusphase startet nicht, das Restbudget bleibt unausgegeben."""
        return not self.entscheidung.weiter

    def als_felder(self):
        """Die Entscheidungszeile, aus der sich die Zwischenbewertung zusammen mit den
        Einzelbewertungen in den Zeilen der Nachbewertung nachrechnen laesst."""
        return {
            **entscheidungsfelder(
                self.staende, self.entscheidung, self.kostenobergrenze, self.kein_gewinner
            ),
            "optimierungsbudget": self.optimierungsbudget,
            "ausgegeben": self.ausgegeben,
            "restbudget": self.restbudget,
        }


def entscheidungsfelder(staende, entscheidung, kostenobergrenze, kein_gewinner):
    """Die gemeinsamen Felder jeder Entscheidungszeile, Zwischenbewertung wie Endauswahl: wer
    weiter ist, wer warum ausscheidet, ob der Test etwas gefunden hat, und je Kandidatenmodell
    die Zahlen, aus denen die Entscheidung nachzurechnen ist."""
    return {
        "weiter": list(entscheidung.weiter),
        "fuehrender": entscheidung.fuehrender,
        "ausgeschieden": dict(entscheidung.ausgeschieden),
        "test_hat_gefunden": entscheidung.test_hat_gefunden,
        "vergleiche": entscheidung.vergleiche,
        "kein_gewinner": kein_gewinner,
        "kostenobergrenze": kostenobergrenze,
        "alpha": ALPHA,
        "kandidatenmodelle": {
            stand.name: {
                "bewertung": stand.bewertung,
                "ausfuehrungskosten_je_1000": stand.ausfuehrungskosten_je_1000,
                "p_wert": entscheidung.p_werte.get(stand.name),
            }
            for stand in staende
        },
    }


def staende_aus(ergebnisse):
    """Je Optimierungsergebnis die Angaben, die eine Entscheidung benoetigt."""
    return tuple(
        Stand(name, ergebnis.bewertungen_je_beispiel, ergebnis.ausfuehrungskosten)
        for name, ergebnis in ergebnisse.items()
    )


def ausgegeben(ergebnisse):
    """Die tatsaechlichen Dollar aller Messlaeufe: Optimierung und Nachbewertung."""
    return sum(
        ergebnis.optimierungskosten + ergebnis.nachbewertungskosten
        for ergebnis in ergebnisse.values()
    )


def ohne_beispiele(programm):
    """Der Warm-Start: das Programm mit seinen Instruktionen, aber ohne Few-Shot-Beispiele.

    MIPROv2 misst den Startwert mit den mitgegebenen Beispielen, verbucht ihn aber als
    "ohne Beispiele" - die Suche bekaeme sonst eine Information, die nicht zu ihren
    Parametern gehoert. `reset_copy` ist genau das, womit MIPROv2 selbst sein Beispielset 0
    baut: eine Kopie, deren Praediktoren keine Beispiele haben, die Instruktion aber behalten
    (DSPy, `predict/predict.py` Z. 65-69).
    """
    return programm.reset_copy()


def plane(optimierungslauf, kostenmodelle, anteil):
    """Je Kandidatenmodell der Plan fuer denselben Dollar-Betrag, fuer alle, bevor der erste
    Lauf Kosten verursacht. Verfahren und Baselines planen ueber diese Funktion."""
    auswahlbeispiele = len(optimierungslauf.auswahldaten)
    return [modell.plane(anteil, auswahlbeispiele) for modell in kostenmodelle]


def optimiere(optimierungslauf, startprogramme, plaene, seed, einordnung):
    """Die Optimierungslaeufe einer Phase oder einer Baseline, nacheinander, alle mit demselben
    Seed, damit Unterschiede zwischen den Laeufen nicht auf verschiedene Zufallszahlen
    zurueckgehen.

    Meldet ein Lauf auch nur einen Anbieterfehler, bricht der Durchgang sofort ohne
    Entscheidung ab: er ist damit ohnehin ungueltig, und jeder weitere Lauf verursachte
    Kosten, ohne noch etwas entscheiden zu duerfen.
    """
    ergebnisse = {}
    for plan in plaene:
        kandidatenmodell = kandidatenmodell_mit_namen(plan.kandidatenmodell)
        ergebnis = optimierungslauf(
            kandidatenmodell,
            startprogramme[kandidatenmodell.name],
            plan.num_trials,
            seed,
            einordnung=einordnung,
            plan=plan,
        )
        pruefe_messung(kandidatenmodell.name, ergebnis, einordnung)
        ergebnisse[kandidatenmodell.name] = ergebnis
    return ergebnisse


def pruefe_messung(name, ergebnis, einordnung):
    """Prueft nach jedem Lauf, ob optimiert oder nur bewertet, ob auf diesem Ergebnis eine
    Entscheidung fallen darf; andernfalls `UngueltigerDurchgang`."""
    if ergebnis.anbieterfehler:
        raise UngueltigerDurchgang(
            f"{name} meldet {ergebnis.anbieterfehler} Anbieterfehler in {_ort_des_laufs(einordnung)}. "
            "Auf dieser Grundlage faellt keine Entscheidung: ein gedrosseltes Modell saehe "
            "aus wie ein schlechtes (ADR-0018). Den Durchgang mit einer neuen Nummer "
            "wiederholen (ADR-0019)."
        )
    if ergebnis.ausfuehrungskosten is None:
        raise UngueltigerDurchgang(
            f"Fuer {name} sind in {_ort_des_laufs(einordnung)} keine Ausfuehrungskosten gemessen: die "
            "Nachbewertung hat keinen Programmlauf gezaehlt. Ohne sie kann der Kostenfilter "
            "nicht pruefen, und eine fehlende Messung ist keine Messung von null (ADR-0004, "
            "ADR-0021)."
        )


def _ort_des_laufs(einordnung):
    """Wo ein Lauf stattfand, fuer die Meldung eines ungueltigen Durchgangs."""
    if einordnung.phase is not None:
        phase = {"sondierung": "Sondierung", "fokus": "Fokusphase"}[einordnung.phase]
        return f"der {phase} von Durchgang {einordnung.durchgang}"
    return f"der Baseline {einordnung.baseline}, Durchgang {einordnung.durchgang}"


class Platz(NamedTuple):
    """Ein Kandidatenmodell in der Rangliste des Ergebnisses.

    `phase` ist die letzte Phase, in der es optimiert wurde; Bewertung und
    Ausfuehrungskosten stammen aus deren Nachbewertung. `ausgeschieden` ist der Grund aus
    der Entscheidung am Ende dieser Phase - `None`, wenn es dort nicht ausgeschieden ist.
    """

    kandidatenmodell: str
    phase: str
    bewertung: float
    ausfuehrungskosten_je_1000: float
    ausgeschieden: str | None


def plaetze(staende, entscheidung, phase):
    """Die Staende einer Phase in der Reihenfolge der Rangliste: wer die Kostenobergrenze
    einhaelt, vor allen, die sie reissen; sonst wie beim Fuehrenden - die hoehere Bewertung,
    bei Gleichstand die niedrigeren Ausfuehrungskosten, dann die Reihenfolge der Laeufe."""
    reihenfolge = sorted(
        rangfolge(staende),
        key=lambda stand: entscheidung.ausgeschieden.get(stand.name) == KOSTENFILTER,
    )
    return [
        Platz(
            stand.name,
            phase,
            stand.bewertung,
            stand.ausfuehrungskosten_je_1000,
            entscheidung.ausgeschieden.get(stand.name),
        )
        for stand in reihenfolge
    ]


class Budgetverbrauch(NamedTuple):
    """Was ein Durchgang vom Optimierungsbudget tatsaechlich ausgegeben hat, je Phase und
    einschliesslich der Nachbewertungen. `uebrig` ist negativ, wenn das Verfahren ueberzogen
    hat: es plant das Budget vorab, regelt es aber nicht waehrend des Laufs."""

    optimierungsbudget: float
    sondierung: float
    fokus: float

    @property
    def gesamt(self):
        return self.sondierung + self.fokus

    @property
    def uebrig(self):
        return self.optimierungsbudget - self.gesamt


class Verfahrensergebnis(NamedTuple):
    """Was ein Durchgang des Verfahrens zurueckgibt.

    `programm` ist das Modell-Programm-Paar des Gewinners, oder `None`, wenn es keinen gibt.
    Dann sagt `meldung` warum, und `zusatzangabe` haelt - nur wenn erst die Endauswahl
    niemanden mehr durchliess - das guenstigere Paar aus der Sondierung: ausdruecklich
    **nicht** das Ergebnis.

    `endauswahl` ist `None`, wenn die Fokusphase gar nicht gestartet ist.

    `ergebnisse_der_fokusphase` sind die Optimierungsergebnisse der Fokusphase je
    Kandidatenmodell, wie `zwischenstand.ergebnisse` die der Sondierung; leer, wenn die
    Fokusphase nicht gestartet ist. Ihre Programme tragen kein eigenes LM. Das Hauptexperiment
    braucht sie fuer die Testbewertung ohne Gewinner.
    """

    gewinner: str | None
    programm: ModellProgrammPaar | None
    meldung: str | None
    zusatzangabe: ModellProgrammPaar | None
    rangliste: tuple[Platz, ...]
    budgetverbrauch: Budgetverbrauch
    zwischenstand: Zwischenstand
    endauswahl: Entscheidung | None
    ergebnisse_der_fokusphase: dict

    @property
    def kein_gewinner(self):
        return self.gewinner is None

    @property
    def fuehrender_ohne_kostenfilter(self):
        """Das Kandidatenmodell mit der hoechsten Bewertung der Fokusphase, als gaebe es den
        Kostenfilter nicht: bei Gleichstand die niedrigeren Ausfuehrungskosten, dann die
        Reihenfolge der Laeufe (`rangfolge`). Hat das Verfahren keinen Gewinner, bekommt es
        neben der Zusatzangabe eine Testbewertung, als "kein Gewinner" gekennzeichnet.
        `None`, wenn die Fokusphase nicht gestartet ist."""
        if not self.ergebnisse_der_fokusphase:
            return None
        return rangfolge(staende_aus(self.ergebnisse_der_fokusphase))[0].name


class Verfahren:
    """Das Verfahren, konfiguriert mit allen Angaben, die ueber die Durchgaenge gleich bleiben.

    `kostenmodelle` ist je Kandidatenmodell ein `Kostenmodell`; das Kandidatenmodell ergibt
    sich aus dessen Namen. Ihre Reihenfolge ist die Reihenfolge der Laeufe.
    `kostenobergrenze` in Dollar je 1000 Programmlaeufe.
    """

    def __init__(self, *, optimierungslauf, kostenmodelle, kostenobergrenze, protokoll=None):
        self.optimierungslauf = optimierungslauf
        self.kostenmodelle = tuple(kostenmodelle)
        self.kostenobergrenze = kostenobergrenze
        self.protokoll = protokoll or Protokoll()

    def sondiere(self, startprogramm, optimierungsbudget, *, seed, durchgang):
        """Sondierungsphase und Zwischenbewertung eines Durchgangs.

        Geplant wird fuer alle Kandidatenmodelle, bevor der erste Lauf Kosten verursacht:
        reicht ein Anteil fuer keinen Trial, bricht das Kostenmodell ab, bevor Geld
        ausgegeben ist. Alle laufen mit demselben Seed; ein Anbieterfehler bricht den
        Durchgang ab (`optimiere`).

        Gibt den `Zwischenstand` zurueck. Den vollstaendigen Durchgang fuehrt `__call__` aus.
        """
        einordnung = Einordnung.experiment("verfahren", durchgang=durchgang, phase="sondierung")
        anteil = optimierungsbudget * SONDIERUNGSANTEIL / len(self.kostenmodelle)
        plaene = plane(self.optimierungslauf, self.kostenmodelle, anteil)
        ergebnisse = optimiere(
            self.optimierungslauf,
            {plan.kandidatenmodell: startprogramm for plan in plaene},
            plaene,
            seed,
            einordnung,
        )

        staende = staende_aus(ergebnisse)
        zwischenstand = Zwischenstand(
            ergebnisse=ergebnisse,
            staende=staende,
            entscheidung=zwischenbewertung(staende, self.kostenobergrenze),
            kostenobergrenze=self.kostenobergrenze,
            optimierungsbudget=optimierungsbudget,
            ausgegeben=ausgegeben(ergebnisse),
        )
        self.protokoll.schreibe_entscheidung(
            abschnitt="zwischenbewertung",
            einordnung=einordnung,
            seed=seed,
            entscheidung=zwischenstand.als_felder(),
        )
        return zwischenstand

    def __call__(self, startprogramm, optimierungsbudget, *, seed, durchgang):
        """Ein vollstaendiger Durchgang: Sondierung, Zwischenbewertung, Fokusphase,
        Endauswahl. Zurueck kommt ein `Verfahrensergebnis`.

        Hat schon der Kostenfilter der Zwischenbewertung alle ausgesiebt, endet der Durchgang
        nach der Sondierung: die Fokusphase startet nicht, und eine Endauswahl gibt es nicht.
        """
        zwischenstand = self.sondiere(
            startprogramm, optimierungsbudget, seed=seed, durchgang=durchgang
        )
        if zwischenstand.kein_gewinner:
            return Verfahrensergebnis(
                gewinner=None,
                programm=None,
                meldung=(
                    "Kein Gewinner: schon nach der Sondierung reisst jedes Kandidatenmodell "
                    f"die Kostenobergrenze von {self.kostenobergrenze} $ je 1000 "
                    "Programmlaeufe. Unter dieser Obergrenze findet das Verfahren kein "
                    "Modell-Programm-Paar; das Restbudget von "
                    f"{zwischenstand.restbudget:.6f} $ bleibt unausgegeben (ADR-0004, Randfall 1)."
                ),
                zusatzangabe=None,
                rangliste=tuple(
                    plaetze(zwischenstand.staende, zwischenstand.entscheidung, "sondierung")
                ),
                budgetverbrauch=Budgetverbrauch(
                    optimierungsbudget, zwischenstand.ausgegeben, 0.0
                ),
                zwischenstand=zwischenstand,
                endauswahl=None,
                ergebnisse_der_fokusphase={},
            )
        return self._fokussiere(zwischenstand, seed=seed, durchgang=durchgang)

    def _fokussiere(self, zwischenstand, *, seed, durchgang):
        """Fokusphase und Endauswahl.

        Geplant wird wieder fuer alle, bevor der erste Lauf Kosten verursacht: reicht der
        Anteil eines Ueberlebenden fuer keinen Trial, bricht das Kostenmodell mit dem
        Mindestbudget ab. Die Laeufe folgen der Reihenfolge der Zwischenbewertung, der
        Fuehrende zuerst.
        """
        einordnung = Einordnung.experiment("verfahren", durchgang=durchgang, phase="fokus")
        weiter = zwischenstand.entscheidung.weiter
        kostenmodelle = {modell.kandidatenmodell: modell for modell in self.kostenmodelle}
        plaene = plane(
            self.optimierungslauf,
            [kostenmodelle[name] for name in weiter],
            zwischenstand.restbudget / len(weiter),
        )
        ergebnisse = optimiere(
            self.optimierungslauf,
            {name: ohne_beispiele(zwischenstand.ergebnisse[name].programm) for name in weiter},
            plaene,
            seed,
            einordnung,
        )

        staende = staende_aus(ergebnisse)
        endauswahl = zwischenbewertung(staende, self.kostenobergrenze)
        gewinner = endauswahl.fuehrender
        if gewinner is not None:
            programm = self._paar(ergebnisse[gewinner].programm, gewinner)
            zusatzangabe = meldung = None
        else:
            programm = None
            zusatzangabe = self._zusatzangabe(zwischenstand)
            meldung = self._meldung_ohne_gewinner(zusatzangabe)
        verbrauch = Budgetverbrauch(
            zwischenstand.optimierungsbudget, zwischenstand.ausgegeben, ausgegeben(ergebnisse)
        )
        ausgeschieden_in_der_sondierung = [
            stand for stand in zwischenstand.staende if stand.name not in weiter
        ]
        rangliste = tuple(
            plaetze(staende, endauswahl, "fokus")
            + plaetze(ausgeschieden_in_der_sondierung, zwischenstand.entscheidung, "sondierung")
        )

        self.protokoll.schreibe_entscheidung(
            abschnitt="endauswahl",
            einordnung=einordnung,
            seed=seed,
            entscheidung={
                "gewinner": gewinner,
                **entscheidungsfelder(
                    staende, endauswahl, self.kostenobergrenze, gewinner is None
                ),
                "zusatzangabe": None if zusatzangabe is None else zusatzangabe.kandidatenmodell,
                "rangliste": [platz.kandidatenmodell for platz in rangliste],
                "optimierungsbudget": verbrauch.optimierungsbudget,
                "ausgegeben_sondierung": verbrauch.sondierung,
                "ausgegeben_fokus": verbrauch.fokus,
                "ausgegeben": verbrauch.gesamt,
                "uebrig": verbrauch.uebrig,
            },
        )
        return Verfahrensergebnis(
            gewinner=gewinner,
            programm=programm,
            meldung=meldung,
            zusatzangabe=zusatzangabe,
            rangliste=rangliste,
            budgetverbrauch=verbrauch,
            zwischenstand=zwischenstand,
            endauswahl=endauswahl,
            ergebnisse_der_fokusphase=ergebnisse,
        )

    @staticmethod
    def _paar(programm, name):
        return ModellProgrammPaar.aus_programm_und_modell(
            programm, kandidatenmodell_mit_namen(name)
        )

    def _zusatzangabe(self, zwischenstand):
        """Das guenstigere Programm der Sondierung unter den Ueberlebenden. Es hielt
        die Kostenobergrenze ein - sonst haette es die Zwischenbewertung nicht ueberlebt.
        Bei gleichen Ausfuehrungskosten das mit der hoeheren Bewertung."""
        weiter = zwischenstand.entscheidung.weiter
        guenstigstes = min(
            (stand for stand in zwischenstand.staende if stand.name in weiter),
            key=lambda stand: (stand.ausfuehrungskosten, -stand.bewertung),
        )
        return self._paar(zwischenstand.ergebnisse[guenstigstes.name].programm, guenstigstes.name)

    def _meldung_ohne_gewinner(self, zusatzangabe):
        return (
            "Kein Gewinner: nach der Fokusphase reisst jedes verbliebene Modell-Programm-Paar "
            f"die Kostenobergrenze von {self.kostenobergrenze} $ je 1000 Programmlaeufe. "
            "Unter dieser Obergrenze findet das Verfahren keines (ADR-0005). Als "
            f"Zusatzangabe, nicht als Ergebnis: das Programm von {zusatzangabe.kandidatenmodell} "
            "aus der Sondierungsphase, das die Obergrenze dort noch einhielt."
        )
