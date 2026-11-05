"""Die Vorstudie: den Messplan abarbeiten und aus dem Protokoll auswerten.

Die Vorstudie beantwortet vier Messfragen, auf deren Nummern sich Code und Tests beziehen:

1. Kostenmodell: Fixkosten, Kosten je Trial und Ausfuehrungskosten je Kandidatenmodell.
2. Spannweite: Abstand der besten zur schlechtesten Bewertung ohne Optimierung.
3. Anstieg: um welchen Faktor die Optimierung einen Programmlauf verteuert.
4. Budget: das kleinste B, bei dem jedes Kandidatenmodell in der Sondierung genug Trials
   bekommt, und ob das Hauptexperiment damit in die verfuegbaren Mittel passt.

Der Messplan: je Kandidatenmodell eine Bewertung ohne Optimierung (Messfrage 2) und drei
Optimierungslaeufe mit 10, 20 und 40 Trials (Messfragen 1 und 3), alle mit Seed 0. Begonnen
wird mit dem teuersten Kandidatenmodell: sprengt es den Plan, zeigt sich das nach wenigen
Dollar. Danach folgen in derselben Reihenfolge je drei Fixkosten-Messungen mit den
Seeds 0, 4 und 5; die Fixkosten werden also gemessen und nicht aus der Geraden abgelesen.

Verwendet werden ausschliesslich Trainings-, Validierungs- und Auswahldaten. Die Naht wird ohne
Testdaten gebaut, und das Protokoll lehnt eine Testbewertung mit Zweck Vorstudie ohnehin ab.

Fortgesetzt wird aus dem Protokoll: was dort als gueltiger Lauf steht, wird nicht noch einmal
bezahlt. Ein Abbruch kostet so hoechstens den Lauf, der gerade lief. Ausgewertet wird
ebenfalls nur aus dem Protokoll, ohne einen API-Aufruf.

Aufruf (lange Laeufe etwa in `tmux` auf einem Server):
    python -u -m dspy_optimizer.vorstudie              misst, was fehlt, dann der Bericht
    python -m dspy_optimizer.vorstudie auswertung      nur der Bericht, kostet nichts
"""

import datetime
import statistics
import sys
from typing import NamedTuple

from dotenv import load_dotenv

from dspy_optimizer.datensatz import GROESSEN, lade_aufteilung
from dspy_optimizer.kandidatenmodelle import DIE_VIER, kandidatenmodell_mit_namen
from dspy_optimizer.kostenerfassung import messhygiene
from dspy_optimizer.kostenmodell import Kostenmodell, Messung, Parameter, kalibriere
from dspy_optimizer.optimierungslauf import Optimierungslauf
from dspy_optimizer.programm import MultiHopQA
from dspy_optimizer.protokoll import VORSTUDIE, Protokoll, kalibrierbar
from dspy_optimizer.verfahren import PROGRAMMLAEUFE_JE_KOSTENOBERGRENZE, SONDIERUNGSANTEIL

# Die Trial-Zahlen der Optimierungslaeufe im Messplan.
TRIALS = (10, 20, 40)

# Der Seed der Vorstudie; das Hauptexperiment nimmt 1, 2 und 3.
SEED = 0

# Die Seeds der Fixkosten-Messungen 1, 2 und 3: verschiedene Reihenfolgen des Bootstrappings,
# aber keine Seeds des Hauptexperiments.
FIXKOSTEN_SEEDS = (0, 4, 5)

# `DIE_VIER` rueckwaerts. Deren feste Reihenfolge wurde nach aufsteigendem Listenpreis gelegt,
# Luna steht dort zuletzt - rueckwaerts also zuerst, damit das teuerste Kandidatenmodell zuerst
# gemessen wird. Seit dem Endpunktwechsel von DeepSeek folgt die Reihenfolge nicht mehr dem
# Preis, bleibt aber fest.
REIHENFOLGE = tuple(reversed(DIE_VIER))

# So viele Trials bekommt jedes Kandidatenmodell in der Sondierung mindestens: unter
# 1,5 * `num_candidates` arbeitet die Bayes'sche Suche kaum.
TRIALS_IN_DER_SONDIERUNG = 9

# Das Hauptexperiment: drei Seeds, je Seed sechs Durchgaenge mit Budget B - Verfahren,
# Gleichverteilung und die vier Einzelmodelle.
SEEDS_IM_HAUPTEXPERIMENT = 3
DURCHGAENGE_MIT_B_JE_SEED = 6


class Messpunkt(NamedTuple):
    """Ein Lauf des Messplans. `num_trials` ist `None` fuer die Bewertung ohne Optimierung und
    0 fuer eine Fixkosten-Messung; nur diese traegt eine `wiederholung` (1, 2, 3), sonst
    zaehlte von den drei Messungen nur eine."""

    kandidatenmodell: str
    num_trials: int | None
    wiederholung: int | None = None

    @property
    def ist_fixkosten_messung(self):
        return self.wiederholung is not None

    @property
    def seed(self):
        if self.ist_fixkosten_messung:
            return FIXKOSTEN_SEEDS[self.wiederholung - 1]
        return SEED


MESSPLAN = tuple(
    Messpunkt(modell.name, num_trials)
    for modell in REIHENFOLGE
    for num_trials in (None, *TRIALS)
) + tuple(
    Messpunkt(modell.name, 0, wiederholung)
    for modell in REIHENFOLGE
    for wiederholung in range(1, len(FIXKOSTEN_SEEDS) + 1)
)


class Lauf(NamedTuple):
    """Ein fertiger, gueltiger Lauf der Vorstudie, wie er im Protokoll steht.

    `optimierung` ist die Optimierungszeile (`None` bei der Bewertung ohne Optimierung),
    `nachbewertung` die Zeile mit Bewertung und Ausfuehrungskosten. Eine Fixkosten-Messung
    hat nur ihre eine Zeile, `fixkosten_zeile`, und keine der beiden anderen.
    """

    punkt: Messpunkt
    optimierung: dict | None
    nachbewertung: dict | None
    fixkosten_zeile: dict | None = None

    @property
    def letzte_zeile(self):
        return self.fixkosten_zeile or self.nachbewertung

    @property
    def gemessene_fixkosten(self):
        """Die Fixkosten einer Fixkosten-Messung in Dollar."""
        return self.fixkosten_zeile["fixkosten"]


def gueltige_laeufe(zeilen):
    """Die fertigen, gueltigen Laeufe des Messplans im Protokoll, je Messpunkt der erste.

    Gueltig heisst: Zweck Vorstudie; gemessen bei dem Anbieter, auf den das Kandidatenmodell
    festgenagelt ist - ein Lauf bei einem frueheren Anbieter ist eine andere Messung und wird
    neu gemessen; bei einem Optimierungslauf eine
    kalibrierbare Optimierungszeile (mit Bewertung) und direkt danach ihre Nachbewertung; in
    keiner der beiden ein Anbieterfehler; und in der Nachbewertung eine Bewertung - der Marker
    fuer einen zu Ende gekommenen Messlauf - und gemessene Ausfuehrungskosten. Was davon
    abweicht - ein Absturz, ein Anbieterfehler -, hat Geld gekostet, zaehlt aber weder fuer
    Kostenmodell noch Kostenobergrenze und wird beim Fortsetzen wiederholt.

    Die Nachbewertung gehoert zu der Optimierungszeile direkt vor ihr: die Naht schreibt die
    beiden nacheinander, und waehrend der Vorstudie schreibt kein anderer Lauf ins Protokoll.

    Eine Fixkosten-Messung ist eine Zeile fuer sich (`_ist_gueltige_fixkosten_messung`).
    Aeltere Zeilen der Vorstudie tragen kein Feld `fixkosten`; fuer Bewertungen und
    Optimierungslaeufe ist es nicht noetig, sie bleiben fertig.
    """
    laeufe = {}
    optimierung = None
    for zeile in zeilen:
        if zeile.get("zweck") != "vorstudie":
            continue
        if zeile.get("abschnitt") == "fixkosten":
            optimierung = None
            if _ist_gueltige_fixkosten_messung(zeile):
                punkt = _messpunkt_der_zeile(zeile)
                laeufe.setdefault(punkt, Lauf(punkt, None, None, fixkosten_zeile=zeile))
            continue
        if zeile.get("abschnitt") == "optimierung":
            optimierung = zeile
            continue
        if zeile.get("abschnitt") != "nachbewertung":
            continue
        davor, optimierung = optimierung, None
        punkt = _messpunkt_der_zeile(zeile)
        if (
            punkt not in MESSPLAN
            or punkt.ist_fixkosten_messung
            or not _ist_beim_festgenagelten_anbieter_gemessen(zeile)
        ):
            continue
        if punkt.num_trials is None:
            davor = None
        elif davor is None or not kalibrierbar(davor) or _messpunkt_der_zeile(davor) != punkt:
            continue
        if zeile.get("anbieterfehler") or (davor and davor.get("anbieterfehler")):
            continue
        if zeile.get("bewertung") is None or zeile.get("dollar_je_programmlauf") is None:
            continue
        laeufe.setdefault(punkt, Lauf(punkt, davor, zeile))
    return laeufe


def _ist_gueltige_fixkosten_messung(zeile):
    """Fertig ist eine Fixkosten-Messung, wenn ihre Zeile zu einem Messpunkt des Plans gehoert,
    mit dessen Seed und beim festgenagelten Anbieter gemessen ist, keinen Anbieterfehler hat
    und gemessene Fixkosten traegt. Leere Fixkosten heissen: die erste vollstaendige Bewertung
    kam nicht zu Ende."""
    punkt = _messpunkt_der_zeile(zeile)
    return (
        punkt in MESSPLAN
        and punkt.ist_fixkosten_messung
        and zeile.get("seed") == punkt.seed
        and _ist_beim_festgenagelten_anbieter_gemessen(zeile)
        and not zeile.get("anbieterfehler")
        and zeile.get("fixkosten") is not None
    )


def _ist_beim_festgenagelten_anbieter_gemessen(zeile):
    """Ob die Zeile bei dem Anbieter gemessen ist, auf den ihr Kandidatenmodell heute
    festgenagelt ist (`DIE_VIER`)."""
    konfiguration = zeile.get("konfiguration") or {}
    festgenagelter_anbieter = kandidatenmodell_mit_namen(
        konfiguration.get("kandidatenmodell")
    ).anbieter
    return konfiguration.get("anbieter") == festgenagelter_anbieter


def _messpunkt_der_zeile(zeile):
    konfiguration = zeile.get("konfiguration") or {}
    return Messpunkt(
        konfiguration.get("kandidatenmodell"),
        konfiguration.get("num_trials"),
        konfiguration.get("wiederholung"),
    )


class VorstudieAngehalten(RuntimeError):
    """Die Vorstudie hat angehalten, weil auf einem Lauf nichts entschieden werden darf."""


class Vorstudie:
    """Arbeitet den Messplan ab, Lauf fuer Lauf, durch die Naht `Optimierungslauf`."""

    def __init__(self, optimierungslauf, protokoll):
        self.optimierungslauf = optimierungslauf
        self.protokoll = protokoll

    def __call__(self):
        fertige_laeufe = gueltige_laeufe(self.protokoll.zeilen())
        for punkt in offen(fertige_laeufe):
            modell = kandidatenmodell_mit_namen(punkt.kandidatenmodell)
            print(f"[{_jetzt()}] {_beschreibe(punkt)} beginnt", flush=True)
            if punkt.ist_fixkosten_messung:
                ergebnis = self.optimierungslauf.nur_fixkosten(
                    modell,
                    MultiHopQA(),
                    punkt.seed,
                    einordnung=VORSTUDIE,
                    wiederholung=punkt.wiederholung,
                )
                gemessen = f"Fixkosten {_zahl(ergebnis.fixkosten, 4)} $"
            elif punkt.num_trials is None:
                ergebnis = self.optimierungslauf.nur_bewerten(
                    modell, MultiHopQA(), SEED, einordnung=VORSTUDIE, abschnitt="nachbewertung"
                )
                gemessen = f"Bewertung {ergebnis.bewertung:.3f}"
            else:
                ergebnis = self.optimierungslauf(
                    modell, MultiHopQA(), punkt.num_trials, SEED, einordnung=VORSTUDIE
                )
                gemessen = f"Bewertung {ergebnis.bewertung:.3f}"
            print(
                f"[{_jetzt()}] fertig: {gemessen}, "
                f"Formatfehler {ergebnis.formatfehler}, Anbieterfehler {ergebnis.anbieterfehler}",
                flush=True,
            )
            if ergebnis.anbieterfehler:
                # Angehalten statt weitergemessen: erst klaeren, ob es ein Rate Limit war -
                # dann wird die Zahl gleichzeitiger Anfragen fuer alle Kandidatenmodelle gleich
                # gesenkt - oder etwa ein leeres Guthaben. Beim Neustart wird nur dieser Lauf
                # wiederholt.
                raise VorstudieAngehalten(
                    f"{_beschreibe(punkt)}: {ergebnis.anbieterfehler} Anbieterfehler. "
                    "Der Lauf zaehlt nicht (ADR-0018). Welcher Fehler es war, steht unter "
                    "`stoerungen` in seinen Protokollzeilen (bei einem Optimierungslauf: "
                    "Optimierung oder Nachbewertung, die letzten beiden; bei einer "
                    "Fixkosten-Messung die letzte). Nach der Klaerung "
                    "neu starten; fertige Laeufe werden nicht wiederholt."
                )


def offen(fertig):
    """Die Messpunkte, die noch keinen gueltigen Lauf haben, in der Reihenfolge des Plans."""
    return [punkt for punkt in MESSPLAN if punkt not in fertig]


def _jetzt():
    return datetime.datetime.now().strftime("%d.%m. %H:%M")


def _beschreibe(punkt):
    if punkt.ist_fixkosten_messung:
        return (
            f"Fixkosten-Messung {punkt.wiederholung} von {len(FIXKOSTEN_SEEDS)} von "
            f"{punkt.kandidatenmodell} (Seed {punkt.seed})"
        )
    if punkt.num_trials is None:
        return f"Bewertung ohne Optimierung von {punkt.kandidatenmodell}"
    return f"Optimierungslauf von {punkt.kandidatenmodell} mit {punkt.num_trials} Trials"


class Modellergebnis(NamedTuple):
    """Was die Vorstudie ueber ein Kandidatenmodell weiss: seine vier gueltigen Laeufe, seine
    drei Fixkosten-Messungen und das daraus kalibrierte Kostenmodell."""

    kandidatenmodell: str
    ohne_optimierung: Lauf
    optimiert: tuple[Lauf, ...]
    fixkosten_messungen: tuple[Lauf, ...]
    kostenmodell: Kostenmodell

    @property
    def bewertung_ohne(self):
        return self.ohne_optimierung.nachbewertung["bewertung"]

    @property
    def bewertung_optimiert(self):
        """Der Mittelwert ueber die drei optimierten Laeufe, nur berichtet."""
        return statistics.fmean(lauf.nachbewertung["bewertung"] for lauf in self.optimiert)

    @property
    def ausfuehrungskosten_ohne(self):
        return self.ohne_optimierung.nachbewertung["dollar_je_programmlauf"]

    @property
    def anstieg(self):
        """Messfrage 3: um welchen Faktor die Optimierung einen Programmlauf verteuert."""
        return self.kostenmodell.ausfuehrungskosten.betrag / self.ausfuehrungskosten_ohne

    @property
    def latenz_ohne(self):
        """Median der Latenz je Programmlauf, in Sekunden."""
        return self.ohne_optimierung.nachbewertung["latenz_median_sekunden"]

    @property
    def latenz_optimiert(self):
        """Der Mittelwert der Mediane der optimierten Laeufe; `None`, wo nichts gestoppt wurde."""
        gestoppt = [
            lauf.nachbewertung["latenz_median_sekunden"]
            for lauf in self.optimiert
            if lauf.nachbewertung["latenz_median_sekunden"] is not None
        ]
        return statistics.fmean(gestoppt) if gestoppt else None

    @property
    def stunden_je_dollar(self):
        """Wie lange ein Dollar dieses Kandidatenmodells rechnet, ueber die gueltigen Zeilen
        der Bewertung ohne Optimierung und der Optimierungslaeufe. Die Fixkosten-Messungen
        zaehlen nicht mit."""
        zeilen = [self.ohne_optimierung.nachbewertung] + [
            zeile for lauf in self.optimiert for zeile in (lauf.optimierung, lauf.nachbewertung)
        ]
        sekunden = sum(zeile["dauer_sekunden"] or 0.0 for zeile in zeilen)
        return sekunden / 3600 / sum(zeile["dollar"] for zeile in zeilen)


class Auswertung(NamedTuple):
    """Die Antworten der Vorstudie, nur aus dem Protokoll gerechnet.

    `ergebnisse` enthaelt die Kandidatenmodelle, deren Messplan vollstaendig ist, in der
    Reihenfolge der Vorstudie; `offen` die Messpunkte, die noch fehlen.
    """

    ergebnisse: dict
    offen: tuple[Messpunkt, ...]
    # Die Dollar aller Messlaeufe der Vorstudie, auch der ungueltigen.
    ausgegeben: float
    # Alle Anbieterfehler der Vorstudie, auch in nicht gewerteten Laeufen; der Bericht weist
    # sie aus.
    anbieterfehler: int
    # Woher die Zahlen stammen, wie es im Kostenmodell steht.
    herkunft: str

    @property
    def spannweite(self):
        """Messfrage 2: Abstand der besten zur schlechtesten Bewertung ohne Optimierung."""
        bewertungen = [ergebnis.bewertung_ohne for ergebnis in self._vollstaendig().values()]
        return max(bewertungen) - min(bewertungen)

    @property
    def kostenmodelle(self):
        return {name: ergebnis.kostenmodell for name, ergebnis in self.ergebnisse.items()}

    def _vollstaendig(self):
        if self.offen:
            raise ValueError(
                f"Die Vorstudie ist noch nicht vollstaendig: es fehlen {len(self.offen)} "
                f"Laeufe, als naechster {_beschreibe(self.offen[0])}."
            )
        return self.ergebnisse

    @property
    def zweitteuerstes(self):
        """Das Kandidatenmodell mit den zweithoechsten Ausfuehrungskosten optimiert.

        Gleichstand bricht die Reihenfolge der Vorstudie; der Wert der Kostenobergrenze haengt
        daran nicht, er ist bei Gleichstand derselbe.
        """
        nach_kosten = sorted(
            self._vollstaendig().values(),
            key=lambda ergebnis: -ergebnis.kostenmodell.ausfuehrungskosten.betrag,
        )
        return nach_kosten[1].kandidatenmodell

    @property
    def kostenobergrenze(self):
        """Die Ausfuehrungskosten optimiert des zweitteuersten, in Dollar je 1000
        Programmlaeufe."""
        return _je_1000(self.kostenmodelle[self.zweitteuerstes].ausfuehrungskosten.betrag)

    @property
    def b_mindestens(self):
        """Das kleinste B, bei dem jedes Kandidatenmodell in der Sondierung mindestens
        `TRIALS_IN_DER_SONDIERUNG` Trials bekommt (Messfrage 4).

        Die Sondierung bekommt `SONDIERUNGSANTEIL` von B, gleich verteilt auf die vier; jeder
        Anteil bezahlt Optimierung und Nachbewertung. Massgeblich ist das
        Kandidatenmodell, das dafuer am meisten braucht.
        """
        anteil_je_modell = SONDIERUNGSANTEIL / len(DIE_VIER)
        return max(
            ergebnis.kostenmodell.dollar_fuer(TRIALS_IN_DER_SONDIERUNG, GROESSEN.auswahldaten)
            for ergebnis in self._vollstaendig().values()
        ) / anteil_je_modell

    def hauptexperiment(self, b):
        """Was das Hauptexperiment mit diesem B voraussichtlich kostet (Messfrage 4).

        Je Seed die sechs Durchgaenge mit B und die Bewertungen ausserhalb von B: die Baseline
        ohne Optimierung auf den Auswahldaten und die Testbewertungen -
        die vier unoptimierten Programme, die vier der Gleichverteilung, die vier der
        Einzelmodelle und das Ergebnis des Verfahrens. Fuer dieses ist noch offen, welches
        Kandidatenmodell es ist; gerechnet wird mit dem teuersten. Optimierte Programme
        kosten so viel wie im Mittel der Vorstudie.
        """
        ergebnisse = self._vollstaendig().values()
        ohne = sum(ergebnis.ausfuehrungskosten_ohne for ergebnis in ergebnisse)
        optimiert = [ergebnis.kostenmodell.ausfuehrungskosten.betrag for ergebnis in ergebnisse]
        bewertungen = (
            GROESSEN.auswahldaten * ohne
            + GROESSEN.testdaten * (ohne + 2 * sum(optimiert) + max(optimiert))
        )
        return SEEDS_IM_HAUPTEXPERIMENT * (DURCHGAENGE_MIT_B_JE_SEED * b + bewertungen)


def werte_aus(zeilen):
    """Die Vorstudie aus Protokollzeilen auswerten - ohne einen einzigen API-Aufruf."""
    zeilen = list(zeilen)
    laeufe = gueltige_laeufe(zeilen)
    herkunft = _herkunft(laeufe.values())
    ergebnisse = {}
    for modell in REIHENFOLGE:
        punkte = [punkt for punkt in MESSPLAN if punkt.kandidatenmodell == modell.name]
        if any(punkt not in laeufe for punkt in punkte):
            # Auch wenn nur Fixkosten-Messungen fehlen: ohne sie gibt es kein Kostenmodell,
            # und was fehlt, nennt `offen`.
            continue
        ohne = laeufe[Messpunkt(modell.name, None)]
        optimiert = tuple(laeufe[Messpunkt(modell.name, num_trials)] for num_trials in TRIALS)
        fixkosten_messungen = tuple(
            laeufe[punkt] for punkt in punkte if punkt.ist_fixkosten_messung
        )
        ergebnisse[modell.name] = Modellergebnis(
            kandidatenmodell=modell.name,
            ohne_optimierung=ohne,
            optimiert=optimiert,
            fixkosten_messungen=fixkosten_messungen,
            kostenmodell=_kalibriere(modell.name, optimiert, fixkosten_messungen, herkunft),
        )
    vorstudie = [zeile for zeile in zeilen if zeile.get("zweck") == "vorstudie"]
    return Auswertung(
        ergebnisse=ergebnisse,
        offen=tuple(offen(laeufe)),
        ausgegeben=sum(zeile.get("dollar") or 0.0 for zeile in vorstudie),
        anbieterfehler=sum(zeile.get("anbieterfehler") or 0 for zeile in vorstudie),
        herkunft=herkunft,
    )


def _kalibriere(name, optimiert, fixkosten_messungen, herkunft):
    """Messfrage 1 und 3: die Fixkosten als Mittelwert der Fixkosten-Messungen, die Gerade
    durch die Optimierungskosten darueber und, als Ausfuehrungskosten, der Mittelwert ueber die
    Nachbewertungen der optimierten Laeufe - eine Zahl fuer Kostenmodell und Kostenobergrenze."""
    ausfuehrungskosten = statistics.fmean(
        lauf.nachbewertung["dollar_je_programmlauf"] for lauf in optimiert
    )
    fixkosten = statistics.fmean(lauf.gemessene_fixkosten for lauf in fixkosten_messungen)
    seeds = ", ".join(str(lauf.punkt.seed) for lauf in fixkosten_messungen)
    return kalibriere(
        [Messung(lauf.punkt.num_trials, lauf.optimierung["dollar"]) for lauf in optimiert],
        herkunft,
        kandidatenmodell=name,
        fixkosten=Parameter(
            fixkosten,
            f"{herkunft}; Mittelwert aus {len(fixkosten_messungen)} Fixkosten-Messungen "
            f"(Seeds {seeds}; ADR-0014, Nachtrag 10.10.2026)",
        ),
        ausfuehrungskosten=Parameter(
            ausfuehrungskosten,
            f"{herkunft}; Mittelwert der Nachbewertungen von {len(optimiert)} "
            "optimierten Laeufen (ADR-0021)",
        ),
    )


def _herkunft(laeufe):
    """Woher die Zahlen kommen, so wie es in jede Protokollzeile eines geplanten Laufs geht."""
    zeitpunkte = sorted(lauf.letzte_zeile["zeitpunkt"][:10] for lauf in laeufe)
    if not zeitpunkte:
        return "Vorstudie (Issue #11)"
    return f"Vorstudie (Issue #11), gemessen {zeitpunkte[0]} bis {zeitpunkte[-1]}"


def bericht(auswertung):
    """Die Auswertung als Markdown-Tabellen, fuer das Log und fuer Kapitel 6 der Arbeit."""
    zeilen = [
        "# Vorstudie (Issue #11)",
        "",
        f"Herkunft der Zahlen: {auswertung.herkunft}",
        "",
        f"Ausgegeben in der Vorstudie, alle Messlaeufe: {auswertung.ausgegeben:.4f} $",
        f"Anbieterfehler in der Vorstudie: {auswertung.anbieterfehler} (auch in nicht "
        "gewerteten Laeufen; ADR-0018)",
        "",
        "## Je Kandidatenmodell",
        "",
        "Fixkosten $ ist der Mittelwert der Fixkosten-Messungen, $ je Trial die Steigung durch "
        "die Optimierungslaeufe darueber (ADR-0014, Nachtrag 10.10.2026).",
        "",
        "| Kandidatenmodell | F1 ohne | F1 optimiert | $/1000 ohne | $/1000 optimiert | Anstieg "
        "| Fixkosten $ | Fixkosten min-max $ | $ je Trial | Latenz ohne s | Latenz optimiert s "
        "| h je $ |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for ergebnis in auswertung.ergebnisse.values():
        modell = ergebnis.kostenmodell
        gemessen = [lauf.gemessene_fixkosten for lauf in ergebnis.fixkosten_messungen]
        zeilen.append(
            f"| {ergebnis.kandidatenmodell} | {ergebnis.bewertung_ohne:.3f} "
            f"| {ergebnis.bewertung_optimiert:.3f} "
            f"| {_je_1000(ergebnis.ausfuehrungskosten_ohne):.3f} "
            f"| {_je_1000(modell.ausfuehrungskosten.betrag):.3f} "
            f"| {ergebnis.anstieg:.2f} | {modell.fixkosten.betrag:.4f} "
            f"| {min(gemessen):.4f}-{max(gemessen):.4f} "
            f"| {modell.kosten_je_trial.betrag:.5f} | {_zahl(ergebnis.latenz_ohne, 2)} "
            f"| {_zahl(ergebnis.latenz_optimiert, 2)} | {ergebnis.stunden_je_dollar:.2f} |"
        )
    zeilen += [
        "",
        "## Je Fixkosten-Messung",
        "",
        "Nur der feste Teil eines Optimierungslaufs: Beispiele sammeln, Instruktionen vorschlagen, "
        "erste vollstaendige Bewertung (`num_trials = 0`, ADR-0014, Nachtrag 10.10.2026).",
        "",
        "| Kandidatenmodell | Messung | Seed | Fixkosten $ | Dauer min |",
        "|---|---|---|---|---|",
    ]
    for ergebnis in auswertung.ergebnisse.values():
        for lauf in ergebnis.fixkosten_messungen:
            zeilen.append(
                f"| {ergebnis.kandidatenmodell} | {lauf.punkt.wiederholung} | {lauf.punkt.seed} "
                f"| {lauf.gemessene_fixkosten:.4f} "
                f"| {(lauf.fixkosten_zeile['dauer_sekunden'] or 0.0) / 60:.0f} |"
            )
    zeilen += [
        "",
        "## Je Optimierungslauf",
        "",
        "Die Gerade ist das kalibrierte Kostenmodell; ihre Abweichung von der Messung zeigt, ob "
        "die Kosten auf einer Geraden liegen (ADR-0014, Nachtrag 07.10.2026). Abweichung ist "
        "(Optimierung - Gerade) / Gerade: positiv, wo die Gerade zu wenig geplant haette.",
        "",
        "| Kandidatenmodell | Trials | Optimierung $ | Gerade $ | Abweichung | F1 | $/1000 "
        "| Formatfehler | abgeschnitten (Kandidat/Vorschlag) | Dauer min |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for ergebnis in auswertung.ergebnisse.values():
        for lauf in ergebnis.optimiert:
            optimierung, nachbewertung = lauf.optimierung, lauf.nachbewertung
            beide = (optimierung, nachbewertung)
            # Das Kandidatenmodell antwortet in beiden Messlaeufen, das Vorschlagsmodell nur in
            # der Optimierung.
            abgeschnitten_kandidat = sum(
                (zeile.get("abgeschnittene_antworten") or {}).get("kandidatenmodell") or 0
                for zeile in beide
            )
            abgeschnitten_vorschlag = (optimierung.get("abgeschnittene_antworten") or {}).get(
                "vorschlagsmodell"
            )
            minuten = sum(zeile["dauer_sekunden"] or 0.0 for zeile in beide) / 60
            gerade = ergebnis.kostenmodell.dollar_optimierung(lauf.punkt.num_trials)
            zeilen.append(
                f"| {ergebnis.kandidatenmodell} | {lauf.punkt.num_trials} "
                f"| {optimierung['dollar']:.4f} | {gerade:.4f} "
                f"| {(optimierung['dollar'] - gerade) / gerade * 100:+.1f} % "
                f"| {nachbewertung['bewertung']:.3f} "
                f"| {_je_1000(nachbewertung['dollar_je_programmlauf']):.3f} "
                f"| {sum(zeile['formatfehler'] for zeile in beide)} "
                f"| {abgeschnitten_kandidat}/{abgeschnitten_vorschlag} "
                f"| {minuten:.0f} |"
            )
    zeilen.append("")
    if auswertung.offen:
        fehlende_fixkosten = sum(punkt.ist_fixkosten_messung for punkt in auswertung.offen)
        if fehlende_fixkosten:
            # Ohne Fixkosten kein Kostenmodell; das steht hier im Bericht und nicht erst als
            # Fehler der Kalibrierung.
            zeilen.append(
                f"Fixkosten fehlen noch: {fehlende_fixkosten} Fixkosten-Messungen sind offen, "
                "ohne sie gibt es kein Kostenmodell (ADR-0014, Nachtrag 10.10.2026)."
            )
        zeilen.append(
            f"Es fehlen noch {len(auswertung.offen)} Laeufe, als naechster "
            f"{_beschreibe(auswertung.offen[0])}."
        )
        return "\n".join(zeilen)
    b = auswertung.b_mindestens
    zeilen += [
        "## Festzulegen",
        "",
        f"Spannweite ohne Optimierung (F1): {auswertung.spannweite:.3f}",
        f"Zweitteuerstes optimiert: {auswertung.zweitteuerstes}",
        f"Kostenobergrenze: {auswertung.kostenobergrenze:.3f} $ je 1000 Programmlaeufe",
        f"B mindestens: {b:.2f} $ ({TRIALS_IN_DER_SONDIERUNG} Trials je Kandidatenmodell "
        "in der Sondierung)",
        f"Hauptexperiment bei diesem B: {auswertung.hauptexperiment(b):.2f} $, je weiterem "
        f"Dollar B: {SEEDS_IM_HAUPTEXPERIMENT * DURCHGAENGE_MIT_B_JE_SEED} $ mehr",
        "Vorstudie und Hauptexperiment zusammen: "
        f"{auswertung.ausgegeben + auswertung.hauptexperiment(b):.2f} $ (gegen 200 EUR zu "
        "halten, zum Kurs des Tages, an dem B festgelegt wird; ADR-0002)",
    ]
    return "\n".join(zeilen)


def _je_1000(dollar_je_programmlauf):
    """In der Einheit der Kostenobergrenze: Dollar je 1000 Programmlaeufe."""
    return dollar_je_programmlauf * PROGRAMMLAEUFE_JE_KOSTENOBERGRENZE


def _zahl(wert, stellen):
    return "-" if wert is None else f"{wert:.{stellen}f}"


def main():
    # HotPotQA enthaelt Umlaute und Akzente; die Windows-Konsole kann die sonst nicht ausgeben.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    protokoll = Protokoll()
    if sys.argv[1:] == ["auswertung"]:
        _drucke_bericht(protokoll)
        return

    load_dotenv()
    messhygiene()
    aufteilung = lade_aufteilung()
    # Ohne Testdaten: die Vorstudie sieht sie nie.
    naht = Optimierungslauf(
        trainingsdaten=aufteilung.trainingsdaten,
        validierungsdaten=aufteilung.validierungsdaten,
        auswahldaten=aufteilung.auswahldaten,
        protokoll=protokoll,
    )
    try:
        Vorstudie(naht, protokoll)()
    finally:
        _drucke_bericht(protokoll)


def _drucke_bericht(protokoll):
    """Der Bericht nach dem Lauf. Scheitert die Auswertung - etwa weil `kalibriere` eine
    Messreihe abweist -, steht das im Log, ohne die Meldung zu verdecken, mit der die Vorstudie
    angehalten hat."""
    try:
        print("\n" + bericht(werte_aus(protokoll.zeilen())), flush=True)
    except ValueError as fehler:
        print(f"\nKein Bericht moeglich: {fehler}", flush=True)


if __name__ == "__main__":
    main()
