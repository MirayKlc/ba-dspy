"""Auswertung des Hauptexperiments: Tabellen, Abbildungen und Bericht fuer Kapitel 6.

Die Auswertung liest ausschliesslich das Protokoll und ruft keine API auf. Jede ausgegebene
Zahl laesst sich damit aus dem Protokoll nachrechnen.

Gueltigkeit: Ein Durchgang zaehlt, wenn seine Entscheidungszeile im Protokoll steht und keine
seiner Zeilen einen Anbieterfehler enthaelt. Das ist dieselbe Regel wie in
`hauptexperiment.finde_fertige_durchgaenge`, nur ohne Pruefung der gespeicherten Programme. Je
Seed, Baseline und Kandidatenmodell zaehlt der erste gueltige Durchgang. Eine Testbewertung
zaehlt, wenn sie ohne Anbieterfehler und mit Bewertung abgeschlossen wurde.

Zwei Vergleiche mit verschiedenen Stichproben: Beim Vergleich Verfahren gegen
Gleichverteilung sind die drei Seeds die Stichprobe; das reicht nicht fuer eine
Signifikanzaussage, berichtet werden Mittelwert und Standardabweichung. Zwei fertige Programme
auf den Testdaten haben 500 gepaarte Beispiele; dort wird ein zweiseitiger gepaarter t-Test mit
alpha = 0,05 gerechnet.

Jede Ergebniszahl wird als Mittelwert ± Standardabweichung ueber die drei Seeds angegeben,
mit der Stichprobenstandardabweichung (n - 1 im Nenner). Tabellen "je Seed" enthalten die
Messwerte, aus denen die zusammengefassten Tabellen berechnet sind.

Zwei Preisfassungen: `abgerechnet` sind die gemessenen Dollar
und bilden das Hauptergebnis. `festgehalten` ist die Gegenprobe, bei der die Kosten von
DeepSeek V4.1 Flash so umgerechnet werden, als haette der Preis vom Stichtag weiter gegolten
(`dollar_zum_festgehaltenen_preis`).

Aufruf (schreibt nach `ergebnisse/auswertung/`):
    python -m dspy_optimizer.auswertung
"""

import datetime
import functools
import itertools
import math
import pathlib
import statistics
import sys
from typing import NamedTuple

import pandas as pd
from scipy import stats

from dspy_optimizer.hauptexperiment import B, KOSTENOBERGRENZE
from dspy_optimizer.kandidatenmodelle import DIE_VIER, kandidatenmodell_mit_namen
from dspy_optimizer.protokoll import WURZEL, Protokoll
from dspy_optimizer.verfahren import (
    PROGRAMMLAEUFE_JE_KOSTENOBERGRENZE,
    SONDIERUNGSANTEIL,
    Stand,
    kostenfilter,
    rangfolge,
    zwischenbewertung,
)

AUSGABE = WURZEL / "ergebnisse" / "auswertung"

# Vergleich zweier fertiger Programme auf den Testdaten: zweiseitig, weil die Richtung des
# Unterschieds vorher offen ist. Als belastbar gilt ein Platzwechsel, wenn er in mindestens
# zwei von drei Seeds signifikant ist.
ALPHA_ZWEISEITIG = 0.05
BELASTBAR_AB_SEEDS = 2

NAMEN = tuple(modell.name for modell in DIE_VIER)
KURZNAMEN = {
    "openai/gpt-oss-120b": "gpt-oss-120b",
    "google/gemma-4-31b-it": "Gemma 4 31B",
    "deepseek/deepseek-v4.1-flash": "DeepSeek V4.1 Flash",
    "openai/gpt-6-luna": "GPT-6 Luna",
}
BASELINENAMEN = {
    "verfahren": "Verfahren",
    "gleichverteilung": "Gleichverteilung (B/4 je Modell)",
    "ein-modell": "Einzelmodell (volles Budget B)",
    "ohne-optimierung": "Ohne Optimierung",
}
KEIN_GEWINNER = "kein Gewinner"
PREISFASSUNGEN = ("abgerechnet", "festgehalten")

# --- Der Preis von DeepSeek -----------------------------------------------------------------

DEEPSEEK = "deepseek/deepseek-v4.1-flash"
_DEEPSEEK_ZUM_STICHTAG = kandidatenmodell_mit_namen(DEEPSEEK)


class Preis(NamedTuple):
    """Listenpreis in Dollar je 1 Mio. Tokens, gueltig ab `ab` (ISO-Zeitpunkt)."""

    ab: str
    prompt: float
    antwort: float


# Ab wann jeder Preis galt, laesst sich nur eingrenzen. Verwendet wird der frueheste Zeitpunkt,
# zu dem das Protokoll ihn belegt: der zweite Preis durch die Bewertung ohne Optimierung in
# Durchgang 20, der dritte durch die Nachbewertung von DeepSeek in Durchgang 23. Zeilen zwischen
# der tatsaechlichen Erhoehung und diesem Beleg werden mit dem aelteren Preis gerechnet; die
# Gegenprobe unterschaetzt die Preiserhoehung dort also eher.
PREISE_DEEPSEEK = (
    Preis("", _DEEPSEEK_ZUM_STICHTAG.preis_prompt, _DEEPSEEK_ZUM_STICHTAG.preis_antwort),
    Preis("2026-10-13T15:11:07+02:00", 0.077, 0.55),
    Preis("2026-10-14T11:36:16+02:00", 0.10, 0.66),
)


def _preis_zur_zeit(zeitpunkt):
    zeitpunkt_der_zeile = datetime.datetime.fromisoformat(zeitpunkt)
    gueltiger_preis = PREISE_DEEPSEEK[0]
    for preis in PREISE_DEEPSEEK[1:]:
        if zeitpunkt_der_zeile >= datetime.datetime.fromisoformat(preis.ab):
            gueltiger_preis = preis
    return gueltiger_preis


def _listenkosten(zeile, preis):
    return (zeile["tokens_prompt"] * preis.prompt + zeile["tokens_antwort"] * preis.antwort) / 1e6


def dollar_zum_festgehaltenen_preis(zeile):
    """Die Dollar einer Zeile, als haette DeepSeek seinen Preis vom Stichtag behalten.

    Berechnet als abgerechneter Betrag mal festgehaltenem durch damaligen Listenpreis, beide
    fuer die Tokens dieser Zeile. Das setzt voraus, dass der Rabatt fuer kuerzlich gesehene
    Eingaben denselben Anteil hatte, da das Protokoll die rabattierten Tokens nicht getrennt
    ausweist. In einer Optimierungszeile sind auch die Tokens des Vorschlagsmodells enthalten
    (47 von ueber 6000 Aufrufen); sie werden mit umgerechnet, der Fehler liegt unter einem
    Prozent. Zeilen anderer Kandidatenmodelle bleiben unveraendert.
    """
    dollar = zeile.get("dollar")
    if not dollar or (zeile.get("konfiguration") or {}).get("kandidatenmodell") != DEEPSEEK:
        return dollar
    damals = _preis_zur_zeit(zeile["zeitpunkt"])
    if damals == PREISE_DEEPSEEK[0] or not _listenkosten(zeile, damals):
        return dollar
    return dollar * _listenkosten(zeile, PREISE_DEEPSEEK[0]) / _listenkosten(zeile, damals)


# --- Rechenregeln ---------------------------------------------------------------------------


def mittel_sd(werte):
    """Mittelwert und Standardabweichung der Stichprobe; bei einem Wert keine Streuung."""
    werte = [wert for wert in werte if wert is not None and not _fehlt(wert)]
    if not werte:
        return None, None
    return statistics.fmean(werte), statistics.stdev(werte) if len(werte) > 1 else None


def formatiere(mittel, sd, stellen=3):
    """"0,820 ± 0,010", deutsch, wie es in der Arbeit steht."""
    return f"{_zahl(mittel, stellen)} ± {_zahl(sd, stellen)}"


def _fehlt(wert):
    return wert is None or (isinstance(wert, float) and math.isnan(wert))


def _zahl(wert, stellen=3):
    if _fehlt(wert):
        return "–"
    return f"{wert:,.{stellen}f}".replace(",", " ").replace(".", ",")


def kendall_tau(erste, zweite):
    """Kendalls Tau zweier Rangfolgen derselben Kandidatenmodelle: je Paar wird geprueft, ob
    beide Rangfolgen dasselbe Modell vorne sehen; (uebereinstimmend - abweichend) / Zahl der
    Paare. Ein Gleichstand zaehlt zu keiner der beiden Gruppen."""
    paare = list(itertools.combinations(erste, 2))
    summe = 0
    for a, b in paare:
        summe += _vorzeichen(erste[a] - erste[b]) * _vorzeichen(zweite[a] - zweite[b])
    return summe / len(paare)


def _vorzeichen(x):
    return (x > 0) - (x < 0)


def gepaarter_test(erste, zweite):
    """Zweiseitiger p-Wert des gepaarten t-Tests zweier Programme auf denselben Beispielen.
    `None`, wenn sie in keinem Beispiel voneinander abweichen."""
    if len(erste) != len(zweite):
        raise ValueError("Gepaart wird nur ueber dieselben Beispiele.")
    p = float(stats.ttest_rel(erste, zweite).pvalue)
    return None if math.isnan(p) else p


def waehle_nach_der_regel(staende, kostenobergrenze):
    """Die Auswahlregel der Baselines: erst der Kostenfilter, dann die hoechste Bewertung.
    `None`, wenn kein Kandidatenmodell unter der Obergrenze liegt."""
    unter_der_obergrenze = kostenfilter(staende, kostenobergrenze)
    return rangfolge(unter_der_obergrenze)[0].name if unter_der_obergrenze else None


def break_even(optimierungskosten, ausfuehrungskosten_gewaehlt, ausfuehrungskosten_referenz):
    """N* = C_opt / (c_exec,Referenz - c_exec,gewaehlt) in Programmlaeufen (FF4), beide
    Ausfuehrungskosten in Dollar je Programmlauf.
    `None`, wenn das gewaehlte Paar nicht billiger ist: dann gibt es keinen Break-even ueber
    die Kosten."""
    if ausfuehrungskosten_gewaehlt >= ausfuehrungskosten_referenz:
        return None
    return optimierungskosten / (ausfuehrungskosten_referenz - ausfuehrungskosten_gewaehlt)


def pareto_dominiert(kosten, qualitaet):
    """Je Punkt, ob ein anderer Punkt hoechstens so teuer und mindestens so gut ist und in
    mindestens einer der beiden Groessen echt besser."""
    punkte = list(zip(kosten, qualitaet))
    return [
        any(
            k <= k0 and q >= q0 and (k < k0 or q > q0)
            for k, q in punkte
        )
        for k0, q0 in punkte
    ]


# --- Durchgaenge ----------------------------------------------------------------------------


class Durchgang(NamedTuple):
    """Ein gueltiger Durchgang: Nummer, Seed, Baseline, beim Einzelmodell das Kandidatenmodell,
    seine Zeilen ohne die Testbewertungen und seine letzte Entscheidungszeile."""

    nummer: int
    seed: int
    baseline: str
    kandidatenmodell: str | None
    zeilen: tuple
    entscheidung: dict


def _zeilen_je_durchgang(zeilen):
    """Die Zeilen des Experiments je Durchgang, ohne Testbewertungen; diese entstehen erst in
    der letzten Stufe des Ablaufs."""
    je_durchgang = {}
    for zeile in zeilen:
        if zeile.get("zweck") != "experiment" or zeile.get("abschnitt") == "testbewertung":
            continue
        je_durchgang.setdefault(zeile["durchgang"], []).append(zeile)
    return dict(sorted(je_durchgang.items()))


def _entscheidungszeile(zeilen):
    for zeile in zeilen:
        if zeile.get("abschnitt") == "endauswahl":
            return zeile
        if zeile.get("abschnitt") == "zwischenbewertung" and zeile.get("kein_gewinner"):
            return zeile
    return None


def _ist_gueltig(zeilen):
    return _entscheidungszeile(zeilen) is not None and not any(
        zeile.get("anbieterfehler") for zeile in zeilen
    )


def finde_gueltige_durchgaenge(zeilen):
    """Die gueltigen Durchgaenge, je Seed, Baseline und Kandidatenmodell der erste."""
    gefunden = {}
    for nummer, zeilen_des_durchgangs in _zeilen_je_durchgang(zeilen).items():
        if not _ist_gueltig(zeilen_des_durchgangs):
            continue
        entscheidung = _entscheidungszeile(zeilen_des_durchgangs)
        kandidatenmodell = None
        if entscheidung["baseline"] == "ein-modell":
            (kandidatenmodell,) = entscheidung["kandidatenmodelle"]
        aufgabe = (entscheidung["seed"], entscheidung["baseline"], kandidatenmodell)
        gefunden.setdefault(
            aufgabe,
            Durchgang(
                nummer,
                entscheidung["seed"],
                entscheidung["baseline"],
                kandidatenmodell,
                tuple(zeilen_des_durchgangs),
                entscheidung,
            ),
        )
    return sorted(gefunden.values(), key=lambda durchgang: durchgang.nummer)


def _kandidatenmodell_der_zeile(zeile):
    return (zeile.get("konfiguration") or {}).get("kandidatenmodell")


def _mal(wert, faktor):
    return None if wert is None else wert * faktor


def _unter_der_kostenobergrenze(je_1000):
    """`None`, wenn die Ausfuehrungskosten nicht im Protokoll stehen."""
    return None if je_1000 is None else je_1000 <= KOSTENOBERGRENZE


def _je_1000(dollar_je_programmlauf):
    if dollar_je_programmlauf is None:
        return None
    return dollar_je_programmlauf * PROGRAMMLAEUFE_JE_KOSTENOBERGRENZE


def _ist_gueltige_testbewertung(zeile):
    """Eine Testbewertung ist gueltig, wenn sie ohne Anbieterfehler und mit Bewertung
    abgeschlossen wurde."""
    return not zeile.get("anbieterfehler") and zeile.get("bewertung") is not None


def _faktor_zum_festgehaltenen_preis(zeile):
    """Faktor von abgerechnet zu festgehaltenem Preis fuer diese Zeile."""
    dollar = zeile.get("dollar")
    return dollar_zum_festgehaltenen_preis(zeile) / dollar if dollar else 1.0


class Auswertung:
    """Die ganze Auswertung aus den Zeilen des Protokolls. Jede Methode gibt eine Tabelle
    (`pandas.DataFrame`) zurueck; `schreibe` legt alle Tabellen, Abbildungen und den Bericht ab.
    """

    def __init__(self, zeilen):
        self.zeilen = list(zeilen)
        self.durchgaenge = finde_gueltige_durchgaenge(self.zeilen)
        self._nach_nummer = {durchgang.nummer: durchgang for durchgang in self.durchgaenge}

    def seeds(self):
        return sorted({durchgang.seed for durchgang in self.durchgaenge})

    def _durchgang(self, seed, baseline, kandidatenmodell=None):
        for durchgang in self.durchgaenge:
            if (durchgang.seed, durchgang.baseline, durchgang.kandidatenmodell) == (
                seed,
                baseline,
                kandidatenmodell,
            ):
                return durchgang
        return None

    # --- Ungueltiges und Wiederholtes -------------------------------------------------------

    def ungueltige_durchgaenge(self):
        """Je ungueltigem Durchgang: Seed, Baseline, Kandidatenmodelle, Anbieterfehler, Ursache
        (aus `stoerungen`) und Kosten. Die Ursache `KeyboardInterrupt` bedeutet einen Abbruch
        von Hand (Strg+C)."""
        eintraege = []
        for nummer, zeilen in _zeilen_je_durchgang(self.zeilen).items():
            if _ist_gueltig(zeilen):
                continue
            ursachen = []
            for zeile in zeilen:
                for stoerung in zeile.get("stoerungen") or []:
                    if stoerung.startswith("Anbieterfehler: "):
                        ursache = stoerung.removeprefix("Anbieterfehler: ")
                        if ursache not in ursachen:
                            ursachen.append(ursache)
            eintraege.append(
                {
                    "durchgang": nummer,
                    "seed": zeilen[0]["seed"],
                    "baseline": zeilen[0]["baseline"],
                    "kandidatenmodelle": ", ".join(
                        dict.fromkeys(
                            _kandidatenmodell_der_zeile(zeile)
                            for zeile in zeilen
                            if _kandidatenmodell_der_zeile(zeile)
                        )
                    ),
                    "anbieterfehler": sum(zeile.get("anbieterfehler") or 0 for zeile in zeilen),
                    "ursache": ", ".join(ursachen) or "ohne Entscheidung abgebrochen",
                    "dollar": sum(zeile.get("dollar") or 0.0 for zeile in zeilen),
                }
            )
        return pd.DataFrame(
            eintraege,
            columns=[
                "durchgang", "seed", "baseline", "kandidatenmodelle", "anbieterfehler",
                "ursache", "dollar",
            ],
        )

    def _testzeilen(self):
        return [
            zeile
            for zeile in self.zeilen
            if zeile.get("zweck") == "experiment" and zeile.get("abschnitt") == "testbewertung"
        ]

    def wiederholte_testbewertungen(self):
        """Wie viele Testbewertungen nach einem Anbieterfehler wiederholt werden mussten, also
        die Zahl der ungueltigen Testbewertungszeilen."""
        return sum(1 for zeile in self._testzeilen() if not _ist_gueltige_testbewertung(zeile))

    # --- Messwerte je Programm --------------------------------------------------------------

    def testbewertungen(self):
        """Je Modell-Programm-Paar seine gueltige Testbewertung, mit beiden Preisfassungen."""
        return self._tests.copy()

    @functools.cached_property
    def _tests(self):
        eintraege = []
        gesehen = set()
        for zeile in self._testzeilen():
            if not _ist_gueltige_testbewertung(zeile):
                continue
            durchgang = self._nach_nummer.get(zeile["durchgang"])
            paar = zeile["konfiguration"].get("paar")
            if durchgang is None or paar in gesehen:
                continue
            gesehen.add(paar)
            preisfaktor = _faktor_zum_festgehaltenen_preis(zeile)
            eintraege.append(
                {
                    "durchgang": durchgang.nummer,
                    "seed": durchgang.seed,
                    "baseline": durchgang.baseline,
                    "phase": zeile.get("phase"),
                    "kandidatenmodell": _kandidatenmodell_der_zeile(zeile),
                    "rolle": zeile["konfiguration"].get("rolle"),
                    "paar": paar,
                    "testbeispiele": zeile["konfiguration"].get("testdaten"),
                    "bewertung": zeile["bewertung"],
                    "exakt": zeile.get("exakt"),
                    "absatztreffer": zeile.get("absatztreffer"),
                    "dollar": zeile["dollar"],
                    "dollar_festgehalten": zeile["dollar"] * preisfaktor,
                    "je_1000": _je_1000(zeile["dollar_je_programmlauf"]),
                    "je_1000_festgehalten": _je_1000(
                        _mal(zeile["dollar_je_programmlauf"], preisfaktor)
                    ),
                    "latenz_median": zeile.get("latenz_median_sekunden"),
                    "latenz_mittelwert": zeile.get("latenz_mittelwert_sekunden"),
                    "einzelbewertungen": tuple(zeile["einzelbewertungen"]),
                }
            )
        return pd.DataFrame(eintraege)

    def _nachbewertungen(self, nummer, phase):
        # Aus einer Tabelle kommt eine fehlende Phase als NaN zurueck, nicht als None.
        phase = None if _fehlt(phase) else phase
        return [
            zeile
            for zeile in self._nach_nummer[nummer].zeilen
            if zeile["abschnitt"] == "nachbewertung" and zeile.get("phase") == phase
        ]

    def staende(self, nummer, *, phase, preis="abgerechnet"):
        """Die Grundlage einer Entscheidung in diesem Durchgang und dieser Phase: je
        Kandidatenmodell die Einzelbewertungen und Ausfuehrungskosten aus der Nachbewertung, in
        der Reihenfolge der Laeufe. Diese Reihenfolge entscheidet einen verbleibenden
        Gleichstand."""
        _lehne_unbekannte_preisfassung_ab(preis)
        return tuple(
            Stand(
                _kandidatenmodell_der_zeile(zeile),
                tuple(zeile["einzelbewertungen"]),
                {
                    "dollar": zeile["dollar_je_programmlauf"],
                    "dollar_festgehalten": zeile["dollar_je_programmlauf"]
                    * _faktor_zum_festgehaltenen_preis(zeile),
                }[_spalte("dollar", preis)],
            )
            for zeile in self._nachbewertungen(nummer, phase)
        )

    def programme(self):
        """Je Seed und Modell-Programm-Paar mit Testbewertung: Bewertung und Kosten auf den
        Auswahldaten (Nachbewertung) und auf den Testdaten, ob es unter der Kostenobergrenze
        liegt, und die Latenz. Ohne das Etikett "kein Gewinner", damit auch Programme ueber der
        Kostenobergrenze berichtet werden."""
        eintraege = []
        for test in self._tests.itertuples():
            (nachbewertung,) = [
                zeile
                for zeile in self._nachbewertungen(test.durchgang, test.phase)
                if _kandidatenmodell_der_zeile(zeile) == test.kandidatenmodell
            ]
            auswahl_je_1000 = _je_1000(nachbewertung["dollar_je_programmlauf"])
            auswahl_festgehalten = _mal(
                auswahl_je_1000, _faktor_zum_festgehaltenen_preis(nachbewertung)
            )
            eintraege.append(
                {
                    "seed": test.seed,
                    "durchgang": test.durchgang,
                    "baseline": test.baseline,
                    "kandidatenmodell": test.kandidatenmodell,
                    "rolle": test.rolle,
                    "bewertung_auswahl": nachbewertung["bewertung"],
                    "bewertung_test": test.bewertung,
                    "exakt_test": test.exakt,
                    "je_1000_auswahl": auswahl_je_1000,
                    "je_1000_auswahl_festgehalten": auswahl_festgehalten,
                    "je_1000_test": test.je_1000,
                    "je_1000_test_festgehalten": test.je_1000_festgehalten,
                    "unter_grenze": _unter_der_kostenobergrenze(auswahl_je_1000),
                    "unter_grenze_festgehalten": _unter_der_kostenobergrenze(auswahl_festgehalten),
                    "latenz_median_test": test.latenz_median,
                }
            )
        return pd.DataFrame(eintraege)

    def programme_zusammengefasst(self):
        """Je Baseline und Kandidatenmodell ueber die Seeds: Mittelwert ± Standardabweichung,
        und in wie vielen Seeds das Programm unter der Kostenobergrenze lag. Beim Verfahren nur
        das Ergebnis."""
        programme = self.programme()
        eintraege = []
        for (baseline, modell), gruppe in programme.groupby(
            ["baseline", "kandidatenmodell"], sort=False
        ):
            eintraege.append(
                {
                    "baseline": BASELINENAMEN[baseline],
                    "kandidatenmodell": KURZNAMEN[modell],
                    "seeds": len(gruppe),
                    "F1 Auswahldaten": _mittel_sd_als_text(gruppe["bewertung_auswahl"]),
                    "F1 Testdaten": _mittel_sd_als_text(gruppe["bewertung_test"]),
                    "$ je 1000 Auswahldaten": _mittel_sd_als_text(gruppe["je_1000_auswahl"]),
                    "$ je 1000 Testdaten": _mittel_sd_als_text(gruppe["je_1000_test"]),
                    "$ je 1000 Testdaten, festgehaltener Preis": _mittel_sd_als_text(
                        gruppe["je_1000_test_festgehalten"]
                    ),
                    "unter der Kostenobergrenze?": _ja_in_wie_vielen(gruppe["unter_grenze"]),
                    "unter der Kostenobergrenze? (festgehaltener Preis)": _ja_in_wie_vielen(
                        gruppe["unter_grenze_festgehalten"]
                    ),
                    "Latenz Median s (Testdaten)": _mittel_sd_als_text(
                        gruppe["latenz_median_test"], 2
                    ),
                }
            )
        return _ordne(pd.DataFrame(eintraege))

    # --- Entscheidungen -------------------------------------------------------------------

    def zwischenbewertungen(self, *, kostenobergrenze=KOSTENOBERGRENZE, preis="abgerechnet"):
        """Je Verfahren die Zwischenbewertung, nachgerechnet aus den Einzelbewertungen der
        Sondierung: wer weiter ist, ob der Test etwas gefunden hat (FF2) und je
        Kandidatenmodell Grund und p-Wert. Mit `kostenobergrenze=math.inf` ohne Kostenfilter."""
        eintraege = []
        for durchgang in self.durchgaenge:
            if durchgang.baseline != "verfahren":
                continue
            entscheidung = zwischenbewertung(
                self.staende(durchgang.nummer, phase="sondierung", preis=preis),
                kostenobergrenze,
            )
            eintrag = {
                "seed": durchgang.seed,
                "durchgang": durchgang.nummer,
                "weiter": ", ".join(entscheidung.weiter) or KEIN_GEWINNER,
                "fuehrender": entscheidung.fuehrender,
                "vergleiche": entscheidung.vergleiche,
                "test_hat_gefunden": entscheidung.test_hat_gefunden,
            }
            for name in NAMEN:
                eintrag[f"ausgeschieden {name}"] = entscheidung.ausgeschieden.get(name, "")
                eintrag[f"p_wert {name}"] = entscheidung.p_werte.get(name)
            eintraege.append(eintrag)
        return pd.DataFrame(eintraege)

    def _waehle(self, durchgang, kostenobergrenze, preis):
        """Die Auswahl eines Durchgangs, nachgerechnet nach seiner Regel. Beim Verfahren ist das
        die Endauswahl; sie ist nur nachrechenbar, wenn dieselben Kandidatenmodelle in die
        Fokusphase gegangen waeren, da nur deren Laeufe im Protokoll stehen."""
        if durchgang.baseline != "verfahren":
            staende = self.staende(durchgang.nummer, phase=None, preis=preis)
            return waehle_nach_der_regel(staende, kostenobergrenze) or KEIN_GEWINNER
        sondierung = zwischenbewertung(
            self.staende(durchgang.nummer, phase="sondierung", preis=preis), kostenobergrenze
        )
        fokus = self.staende(durchgang.nummer, phase="fokus", preis=preis)
        if not sondierung.weiter:
            return KEIN_GEWINNER
        if set(sondierung.weiter) != {stand.name for stand in fokus}:
            return "nicht nachrechenbar"
        return zwischenbewertung(fokus, kostenobergrenze).fuehrender or KEIN_GEWINNER

    def auswahlen(self):
        """Jede Auswahl in vier Fassungen: wie protokolliert, nachgerechnet (muss mit der
        protokollierten uebereinstimmen), ohne Kostenobergrenze und zum festgehaltenen Preis.
        Die Einzelmodelle treffen
        keine Auswahl; ob ihr Programm unter der Obergrenze liegt, steht in `programme`."""
        eintraege = []
        for durchgang in self.durchgaenge:
            if durchgang.baseline == "ein-modell":
                continue
            eintraege.append(
                {
                    "seed": durchgang.seed,
                    "durchgang": durchgang.nummer,
                    "baseline": durchgang.baseline,
                    "protokolliert": durchgang.entscheidung.get("gewinner") or KEIN_GEWINNER,
                    "nachgerechnet": self._waehle(durchgang, KOSTENOBERGRENZE, "abgerechnet"),
                    "ohne_obergrenze": self._waehle(durchgang, math.inf, "abgerechnet"),
                    "festgehaltener_preis": self._waehle(
                        durchgang, KOSTENOBERGRENZE, "festgehalten"
                    ),
                }
            )
        return pd.DataFrame(eintraege)

    def _gewaehltes_programm(self, durchgang, preis):
        """Die Testbewertung des Programms, das der Durchgang in dieser Preisfassung waehlt;
        beim Einzelmodell sein Programm, auch ueber der Kostenobergrenze."""
        tests = self._tests[self._tests["durchgang"] == durchgang.nummer]
        if durchgang.baseline == "ein-modell":
            return tests.iloc[0]
        gewaehltes_kandidatenmodell = self._waehle(durchgang, KOSTENOBERGRENZE, preis)
        if durchgang.baseline == "verfahren":
            tests = tests[tests["phase"] == "fokus"]
        treffer = tests[tests["kandidatenmodell"] == gewaehltes_kandidatenmodell]
        return treffer.iloc[0] if len(treffer) else None

    # --- FF1 -----------------------------------------------------------------------------

    def _testbewertungen_je_kandidatenmodell(self, durchgang):
        tests = self._tests[self._tests["durchgang"] == durchgang.nummer]
        return {zeile.kandidatenmodell: zeile for zeile in tests.itertuples()}

    def _ohne_optimierung_und_gleichverteilung(self):
        """Je Seed, in dem beide Durchgaenge gueltig sind, die Testbewertungen der vier
        unoptimierten Programme und der vier Programme der Gleichverteilung, je
        Kandidatenmodell."""
        for seed in self.seeds():
            ohne_optimierung = self._durchgang(seed, "ohne-optimierung")
            gleichverteilung = self._durchgang(seed, "gleichverteilung")
            if ohne_optimierung is None or gleichverteilung is None:
                continue
            yield (
                seed,
                self._testbewertungen_je_kandidatenmodell(ohne_optimierung),
                self._testbewertungen_je_kandidatenmodell(gleichverteilung),
            )

    def ff1(self):
        """Je Seed die Rangfolge ohne Optimierung gegen die nach Optimierung mit gleichem Anteil
        (Gleichverteilung), auf den Testdaten: Kendalls Tau und ob die Spitze wechselt."""
        eintraege = []
        for seed, ohne, mit in self._ohne_optimierung_und_gleichverteilung():
            bewertung_ohne_optimierung = {name: ohne[name].bewertung for name in NAMEN}
            bewertung_gleichverteilung = {name: mit[name].bewertung for name in NAMEN}
            rang_ohne = sorted(NAMEN, key=lambda name: -bewertung_ohne_optimierung[name])
            rang_mit = sorted(NAMEN, key=lambda name: -bewertung_gleichverteilung[name])
            eintraege.append(
                {
                    "seed": seed,
                    "tau": kendall_tau(bewertung_ohne_optimierung, bewertung_gleichverteilung),
                    "spitze_ohne": rang_ohne[0],
                    "spitze_mit": rang_mit[0],
                    "spitze_wechselt": rang_ohne[0] != rang_mit[0],
                    "rangfolge_ohne": " > ".join(KURZNAMEN[name] for name in rang_ohne),
                    "rangfolge_mit": " > ".join(KURZNAMEN[name] for name in rang_mit),
                    **{
                        f"bewertung_ohne_optimierung {name}": bewertung_ohne_optimierung[name]
                        for name in NAMEN
                    },
                    **{
                        f"bewertung_gleichverteilung {name}": bewertung_gleichverteilung[name]
                        for name in NAMEN
                    },
                }
            )
        return pd.DataFrame(eintraege)

    def platzwechsel(self):
        """Je Seed jedes Paar von Kandidatenmodellen, das mit Optimierung die Plaetze tauscht,
        mit dem zweiseitigen gepaarten Test auf den 500 Testbeispielen: zwischen den beiden
        optimierten Programmen (der Unterschied der neuen Rangfolge) und, zum Vergleich, den
        beiden unoptimierten."""
        eintraege = []
        for seed, ohne, mit in self._ohne_optimierung_und_gleichverteilung():
            for a, b in itertools.combinations(NAMEN, 2):
                richtung_ohne = _vorzeichen(ohne[a].bewertung - ohne[b].bewertung)
                richtung_mit = _vorzeichen(mit[a].bewertung - mit[b].bewertung)
                # Ein Gleichstand ist kein Platzwechsel, wie bei Kendalls Tau.
                if richtung_ohne * richtung_mit >= 0:
                    continue
                vorne_ohne = a if richtung_ohne > 0 else b
                vorne_mit = a if richtung_mit > 0 else b
                p_mit = gepaarter_test(mit[a].einzelbewertungen, mit[b].einzelbewertungen)
                eintraege.append(
                    {
                        "seed": seed,
                        "paar": f"{KURZNAMEN[a]} / {KURZNAMEN[b]}",
                        "vorne_ohne": KURZNAMEN[vorne_ohne],
                        "vorne_mit": KURZNAMEN[vorne_mit],
                        "p_mit": p_mit,
                        "p_ohne": gepaarter_test(
                            ohne[a].einzelbewertungen, ohne[b].einzelbewertungen
                        ),
                        "signifikant": p_mit is not None and p_mit < ALPHA_ZWEISEITIG,
                    }
                )
        return pd.DataFrame(
            eintraege,
            columns=["seed", "paar", "vorne_ohne", "vorne_mit", "p_mit", "p_ohne", "signifikant"],
        )

    def platzwechsel_zusammengefasst(self):
        """Je Paar: in wie vielen Seeds es die Plaetze tauscht, in wie vielen davon signifikant,
        und ob der Wechsel belastbar ist (mindestens zwei von drei Seeds)."""
        wechsel = self.platzwechsel()
        eintraege = []
        for paar, gruppe in wechsel.groupby("paar", sort=False):
            signifikant = int(gruppe["signifikant"].sum())
            eintraege.append(
                {
                    "paar": paar,
                    "wechsel in Seeds": f"{len(gruppe)} von {len(self.seeds())}",
                    "davon signifikant": signifikant,
                    "belastbar": signifikant >= BELASTBAR_AB_SEEDS,
                }
            )
        return pd.DataFrame(
            eintraege, columns=["paar", "wechsel in Seeds", "davon signifikant", "belastbar"]
        )

    # --- Vergleiche fertiger Programme --------------------------------------------------

    def vergleiche_auf_den_testdaten(self, preis="abgerechnet"):
        """Je Seed das Ergebnis des Verfahrens gegen das gewaehlte Programm jeder Baseline, auf
        denselben 500 Testbeispielen: Differenz der Bewertung und zweiseitiger gepaarter Test.
        Bei den Einzelmodellen jedes Programm, mit der Angabe, ob es unter der Obergrenze lag."""
        _lehne_unbekannte_preisfassung_ab(preis)
        programme = self.programme().set_index("durchgang")
        eintraege = []
        for seed in self.seeds():
            durchgang_des_verfahrens = self._durchgang(seed, "verfahren")
            if durchgang_des_verfahrens is None:
                continue
            verfahren = self._gewaehltes_programm(durchgang_des_verfahrens, preis)
            if verfahren is None:
                continue
            gegner = [
                self._durchgang(seed, "gleichverteilung"),
                self._durchgang(seed, "ohne-optimierung"),
            ] + [self._durchgang(seed, "ein-modell", name) for name in NAMEN]
            for durchgang in gegner:
                if durchgang is None:
                    continue
                gegen = self._gewaehltes_programm(durchgang, preis)
                if gegen is None:
                    continue
                programme_des_gegners = programme.loc[[durchgang.nummer]]
                unter_grenze = programme_des_gegners[
                    programme_des_gegners["kandidatenmodell"] == gegen.kandidatenmodell
                ][_spalte("unter_grenze", preis)].iloc[0]
                eintraege.append(
                    {
                        "seed": seed,
                        "gegen": _name_des_gegners(durchgang),
                        "verfahren_waehlt": KURZNAMEN[verfahren.kandidatenmodell],
                        "bewertung_verfahren": verfahren.bewertung,
                        "gegen_waehlt": KURZNAMEN[gegen.kandidatenmodell],
                        "bewertung_gegner": gegen.bewertung,
                        "differenz": verfahren.bewertung - gegen.bewertung,
                        "p_wert": gepaarter_test(
                            verfahren.einzelbewertungen, gegen.einzelbewertungen
                        ),
                        "gegen_unter_grenze": bool(unter_grenze),
                    }
                )
        tabelle = pd.DataFrame(
            eintraege,
            columns=[
                "seed", "gegen", "verfahren_waehlt", "bewertung_verfahren", "gegen_waehlt",
                "bewertung_gegner", "differenz", "p_wert", "gegen_unter_grenze",
            ],
        )
        tabelle["signifikant"] = tabelle["p_wert"].map(
            lambda p: p is not None and not _fehlt(p) and p < ALPHA_ZWEISEITIG
        )
        return tabelle

    def vergleiche_zusammengefasst(self, preis="abgerechnet"):
        vergleiche = self.vergleiche_auf_den_testdaten(preis)
        eintraege = []
        for gegen, gruppe in vergleiche.groupby("gegen", sort=False):
            eintraege.append(
                {
                    "Verfahren gegen": gegen,
                    "F1 Verfahren": _mittel_sd_als_text(gruppe["bewertung_verfahren"]),
                    "F1 Gegner": _mittel_sd_als_text(gruppe["bewertung_gegner"]),
                    "Differenz": _mittel_sd_als_text(gruppe["differenz"]),
                    "Programmtest signifikant (500 Beispiele)": (
                        f"{int(gruppe['signifikant'].sum())} von {len(gruppe)}"
                    ),
                    "Gegner unter der Obergrenze": _ja_in_wie_vielen(gruppe["gegen_unter_grenze"]),
                }
            )
        return pd.DataFrame(eintraege)

    def endqualitaet(self):
        """Je Seed und Strategie die Bewertung auf den Testdaten des Programms, das die Strategie
        waehlt (FF3): ohne Optimierung, jedes Einzelmodell mit seinem Programm, die
        Gleichverteilung in beiden Preisfassungen (in Seed 3 waehlt sie je nach Preis ein
        anderes Paar) und das Verfahren, dessen Wahl nicht vom Preis abhing."""
        strategien = [("Ohne Optimierung", "ohne-optimierung", None, "abgerechnet")]
        strategien += [
            (f"Einzelmodell: {KURZNAMEN[name]}", "ein-modell", name, "abgerechnet")
            for name in NAMEN
        ]
        strategien += [
            ("Gleichverteilung (abgerechnet)", "gleichverteilung", None, "abgerechnet"),
            ("Gleichverteilung (Preis vom Stichtag)", "gleichverteilung", None, "festgehalten"),
            ("Verfahren", "verfahren", None, "abgerechnet"),
        ]
        eintraege = []
        for strategie, baseline, kandidatenmodell, preis in strategien:
            for seed in self.seeds():
                durchgang = self._durchgang(seed, baseline, kandidatenmodell)
                gewaehltes_paar = (
                    None if durchgang is None else self._gewaehltes_programm(durchgang, preis)
                )
                if gewaehltes_paar is None:
                    continue
                eintraege.append(
                    {
                        "strategie": strategie,
                        "seed": seed,
                        "kandidatenmodell": gewaehltes_paar["kandidatenmodell"],
                        "bewertung_test": gewaehltes_paar["bewertung"],
                    }
                )
        return pd.DataFrame(
            eintraege, columns=["strategie", "seed", "kandidatenmodell", "bewertung_test"]
        )

    def endqualitaet_zusammengefasst(self):
        endqualitaet = self.endqualitaet()
        return pd.DataFrame(
            [
                {
                    "Strategie": strategie,
                    "F1 Testdaten": _mittel_sd_als_text(gruppe["bewertung_test"]),
                    "gewaehlt": ", ".join(
                        f"{KURZNAMEN[name]} ({anzahl})"
                        for name, anzahl in gruppe["kandidatenmodell"]
                        .value_counts(sort=False)
                        .items()
                    ),
                }
                for strategie, gruppe in endqualitaet.groupby("strategie", sort=False)
            ]
        )

    # --- Kosten -------------------------------------------------------------------------

    def kosten_je_durchgang(self, preis="abgerechnet"):
        """Je Durchgang die Optimierungskosten, getrennt in Optimierung und Nachbewertung (die
        Nachbewertung zaehlt zum Budget), die Zahl der Nachbewertungen, die geplanten Kosten
        und die Abweichung davon. Bei
        der Baseline ohne Optimierung sind es die vier Bewertungen, ohne Budget und ohne
        Plan."""
        _lehne_unbekannte_preisfassung_ab(preis)
        dollar = _spalte("dollar", preis)
        eintraege = []
        for durchgang in self.durchgaenge:
            messlaeufe = [
                {
                    "abschnitt": zeile["abschnitt"],
                    "dollar": zeile.get("dollar") or 0.0,
                    "dollar_festgehalten": dollar_zum_festgehaltenen_preis(zeile) or 0.0,
                    "geplant": zeile.get("dollar_geplant"),
                }
                for zeile in durchgang.zeilen
                if zeile["abschnitt"] in ("optimierung", "nachbewertung")
            ]
            optimierung = sum(m[dollar] for m in messlaeufe if m["abschnitt"] == "optimierung")
            nachbewertung = sum(
                m[dollar] for m in messlaeufe if m["abschnitt"] == "nachbewertung"
            )
            geplant = [m["geplant"] for m in messlaeufe if m["geplant"] is not None]
            gesamt = optimierung + nachbewertung
            mit_budget = durchgang.baseline != "ohne-optimierung"
            eintraege.append(
                {
                    "seed": durchgang.seed,
                    "durchgang": durchgang.nummer,
                    "baseline": durchgang.baseline,
                    "kandidatenmodell": durchgang.kandidatenmodell,
                    "optimierung": optimierung,
                    "nachbewertung": nachbewertung,
                    "nachbewertungen": sum(
                        1 for m in messlaeufe if m["abschnitt"] == "nachbewertung"
                    ),
                    "c_opt": gesamt,
                    "budget": B if mit_budget else None,
                    "geplant": sum(geplant) if geplant else None,
                    "abweichung_vom_plan": gesamt - sum(geplant) if geplant else None,
                    "abweichung_vom_plan_anteil": (gesamt - sum(geplant)) / sum(geplant)
                    if geplant
                    else None,
                    "abweichung_von_b_anteil": (gesamt - B) / B if mit_budget else None,
                }
            )
        return pd.DataFrame(eintraege)

    def kosten_zusammengefasst(self, preis="abgerechnet"):
        kosten = self.kosten_je_durchgang(preis)
        eintraege = []
        for (baseline, modell), gruppe in kosten.groupby(
            ["baseline", kosten["kandidatenmodell"].fillna("")], sort=False
        ):
            eintraege.append(
                {
                    "baseline": BASELINENAMEN[baseline],
                    "kandidatenmodell": KURZNAMEN.get(modell, ""),
                    "Optimierung $": _mittel_sd_als_text(gruppe["optimierung"]),
                    "Nachbewertung $": _mittel_sd_als_text(gruppe["nachbewertung"]),
                    "Nachbewertungen": _mittel_sd_als_text(gruppe["nachbewertungen"], 1),
                    "C_opt $": _mittel_sd_als_text(gruppe["c_opt"]),
                    "geplant $": _mittel_sd_als_text(gruppe["geplant"]),
                    "Abweichung vom Plan": _mittel_sd_in_prozent(
                        gruppe["abweichung_vom_plan_anteil"]
                    ),
                    "Abweichung von B": _mittel_sd_in_prozent(gruppe["abweichung_von_b_anteil"]),
                }
            )
        return _ordne(pd.DataFrame(eintraege))

    def kosten_je_stufe(self):
        """Die Gesamtkosten des Experiments je Stufe, einschliesslich ungueltiger Durchgaenge,
        in beiden Preisfassungen."""
        eintraege = []
        gueltige_nummern = {durchgang.nummer for durchgang in self.durchgaenge}
        for seed in self.seeds():
            zeilen = [
                zeile
                for zeile in self.zeilen
                if zeile.get("zweck") == "experiment"
                and zeile.get("abschnitt") != "testbewertung"
                and zeile["seed"] == seed
            ]
            eintraege.append(
                {
                    "stufe": f"Seed {seed}",
                    "abgerechnet": sum(zeile.get("dollar") or 0.0 for zeile in zeilen),
                    "festgehaltener_preis": sum(
                        dollar_zum_festgehaltenen_preis(zeile) or 0.0 for zeile in zeilen
                    ),
                    "davon_ungueltig": sum(
                        zeile.get("dollar") or 0.0
                        for zeile in zeilen
                        if zeile["durchgang"] not in gueltige_nummern
                    ),
                }
            )
        tests = self._testzeilen()
        eintraege.append(
            {
                "stufe": "Testbewertung",
                "abgerechnet": sum(zeile.get("dollar") or 0.0 for zeile in tests),
                "festgehaltener_preis": sum(
                    dollar_zum_festgehaltenen_preis(zeile) or 0.0 for zeile in tests
                ),
                "davon_ungueltig": sum(
                    zeile.get("dollar") or 0.0
                    for zeile in tests
                    if not _ist_gueltige_testbewertung(zeile)
                ),
            }
        )
        return pd.DataFrame(eintraege)

    def ff4(self, preis="abgerechnet"):
        """Je Seed und Durchgang mit Optimierungsbudget: die Optimierungskosten in Programmlaeufen
        des gewaehlten Paars (C_opt / c_exec) und der Break-even gegen das Paar der Baseline ohne
        Optimierung. c_exec stammt aus der Testbewertung."""
        _lehne_unbekannte_preisfassung_ab(preis)
        kosten = self.kosten_je_durchgang(preis).set_index("durchgang")
        spalte = _spalte("dollar", preis)
        eintraege = []
        for durchgang in self.durchgaenge:
            if durchgang.baseline == "ohne-optimierung":
                continue
            durchgang_ohne_optimierung = self._durchgang(durchgang.seed, "ohne-optimierung")
            if durchgang_ohne_optimierung is None:
                continue
            gewaehltes_paar = self._gewaehltes_programm(durchgang, preis)
            referenz = self._gewaehltes_programm(durchgang_ohne_optimierung, preis)
            if gewaehltes_paar is None or referenz is None:
                continue
            c_opt = kosten.loc[durchgang.nummer, "c_opt"]
            c_exec = gewaehltes_paar[spalte] / gewaehltes_paar["testbeispiele"]
            c_ref = referenz[spalte] / referenz["testbeispiele"]
            eintraege.append(
                {
                    "seed": durchgang.seed,
                    "durchgang": durchgang.nummer,
                    "baseline": durchgang.baseline,
                    "gewaehltes_kandidatenmodell": gewaehltes_paar["kandidatenmodell"],
                    "referenz": referenz["kandidatenmodell"],
                    "modellwechsel": gewaehltes_paar["kandidatenmodell"]
                    != referenz["kandidatenmodell"],
                    "c_opt": c_opt,
                    "c_exec_je_1000": c_exec * PROGRAMMLAEUFE_JE_KOSTENOBERGRENZE,
                    "c_ref_je_1000": c_ref * PROGRAMMLAEUFE_JE_KOSTENOBERGRENZE,
                    "c_opt_in_programmlaeufen": c_opt / c_exec,
                    "break_even": break_even(c_opt, c_exec, c_ref),
                }
            )
        return pd.DataFrame(eintraege)

    def ff4_zusammengefasst(self, preis="abgerechnet"):
        """Je Baseline und gewaehltem Kandidatenmodell ueber die Seeds. Getrennt nach
        Kandidatenmodell, weil ein Mittelwert ueber verschiedene gewaehlte Paare nicht sinnvoll
        deutbar ist. Ist das gewaehlte Kandidatenmodell dasselbe wie das der Referenz, beschreibt
        der Break-even keinen Modellwechsel, sondern den Vergleich von optimiertem und
        unoptimiertem Programm desselben Modells."""
        ff4 = self.ff4(preis)
        eintraege = []
        for (baseline, modell), gruppe in ff4.groupby(
            ["baseline", "gewaehltes_kandidatenmodell"], sort=False
        ):
            ohne_break_even = int(gruppe["break_even"].isna().sum())
            eintraege.append(
                {
                    "baseline": BASELINENAMEN[baseline],
                    "kandidatenmodell": KURZNAMEN[modell],
                    "Seeds": len(gruppe),
                    "Referenz (ohne Optimierung)": ", ".join(
                        KURZNAMEN[name] for name in dict.fromkeys(gruppe["referenz"])
                    ),
                    "Modellwechsel": _ja_in_wie_vielen(gruppe["modellwechsel"]),
                    "C_opt $": _mittel_sd_als_text(gruppe["c_opt"]),
                    "c_exec $ je 1000": _mittel_sd_als_text(gruppe["c_exec_je_1000"]),
                    "C_opt in Programmlaeufen": _mittel_sd_als_text(
                        gruppe["c_opt_in_programmlaeufen"], 0
                    ),
                    "Break-even in Programmlaeufen (nur Seeds mit Break-even)": (
                        _mittel_sd_als_text(gruppe["break_even"].dropna(), 0)
                        if ohne_break_even < len(gruppe)
                        else "–"
                    ),
                    "kein Break-even ueber Kosten, nur ueber Qualitaet": (
                        f"{ohne_break_even} von {len(gruppe)}"
                    ),
                }
            )
        return _ordne(pd.DataFrame(eintraege))

    # --- Lernkurven und Pareto-Front ------------------------------------------------------

    def lernkurven(self):
        """Je Seed und Kandidatenmodell drei Budgetstufen aus vorhandenen Laeufen: Sondierung
        (B/8), Gleichverteilung (B/4) und Einzelmodell (B), je mit der Bewertung der
        Nachbewertung. Die Punkte stammen aus verschiedenen Laeufen und bilden keine
        durchgehende Kurve."""
        stufen = (
            ("Sondierung", "verfahren", "sondierung", B * SONDIERUNGSANTEIL / len(NAMEN)),
            ("Gleichverteilung", "gleichverteilung", None, B / len(NAMEN)),
            ("Einzelmodell", "ein-modell", None, B),
        )
        eintraege = []
        for seed in self.seeds():
            for stufe, baseline, phase, budget in stufen:
                for name in NAMEN:
                    durchgang = self._durchgang(
                        seed, baseline, name if baseline == "ein-modell" else None
                    )
                    if durchgang is None:
                        continue
                    for zeile in self._nachbewertungen(durchgang.nummer, phase):
                        if _kandidatenmodell_der_zeile(zeile) == name:
                            eintraege.append(
                                {
                                    "seed": seed,
                                    "kandidatenmodell": name,
                                    "stufe": stufe,
                                    "budgetanteil": budget,
                                    "bewertung": zeile["bewertung"],
                                }
                            )
        return pd.DataFrame(eintraege)

    def lernkurven_zusammengefasst(self):
        lernkurven = self.lernkurven()
        tabelle = {}
        for (name, stufe), gruppe in lernkurven.groupby(
            ["kandidatenmodell", "stufe"], sort=False
        ):
            tabelle.setdefault(KURZNAMEN[name], {})[
                f"{stufe} ({_zahl(gruppe['budgetanteil'].iloc[0], 2)} $)"
            ] = _mittel_sd_als_text(gruppe["bewertung"])
        return (
            pd.DataFrame.from_dict(tabelle, orient="index")
            .rename_axis("kandidatenmodell")
            .reset_index()
        )

    def pareto(self, preis="abgerechnet"):
        """Je Baseline und Kandidatenmodell ueber die Seeds: Bewertung und Ausfuehrungskosten auf
        den Testdaten und ob ein anderes Paar diesen Punkt dominiert. Beim Verfahren nur sein
        Ergebnis."""
        _lehne_unbekannte_preisfassung_ab(preis)
        programme = self.programme()
        spalte = _spalte("je_1000_test", preis)
        punkte = (
            programme.groupby(["baseline", "kandidatenmodell"], sort=False)
            .agg(
                bewertung=("bewertung_test", "mean"),
                bewertung_sd=("bewertung_test", "std"),
                kosten=(spalte, "mean"),
                kosten_sd=(spalte, "std"),
                seeds=("seed", "count"),
            )
            .reset_index()
        )
        punkte["dominiert"] = pareto_dominiert(punkte["kosten"], punkte["bewertung"])
        return punkte

    # --- Ablegen ------------------------------------------------------------------------

    def schreibe(self, ordner=AUSGABE):
        """Schreibt alle Tabellen als CSV (Semikolon, Dezimalkomma, fuer Excel lesbar), alle
        Abbildungen als PDF und PNG und den Bericht `bericht.md` mit den zusammengefassten
        Tabellen."""
        ordner = pathlib.Path(ordner)
        ordner.mkdir(parents=True, exist_ok=True)
        tabellen = {
            "ungueltige-durchgaenge": self.ungueltige_durchgaenge(),
            "testbewertungen": self.testbewertungen().drop(columns="einzelbewertungen"),
            "programme-je-seed": self.programme(),
            "programme": self.programme_zusammengefasst(),
            "ff1-je-seed": self.ff1(),
            "platzwechsel-je-seed": self.platzwechsel(),
            "platzwechsel": self.platzwechsel_zusammengefasst(),
            "zwischenbewertungen": self.zwischenbewertungen(),
            "zwischenbewertungen-ohne-obergrenze": self.zwischenbewertungen(
                kostenobergrenze=math.inf
            ),
            "auswahlen": self.auswahlen(),
            "endqualitaet-je-seed": self.endqualitaet(),
            "endqualitaet": self.endqualitaet_zusammengefasst(),
            "lernkurven-je-seed": self.lernkurven(),
            "lernkurven": self.lernkurven_zusammengefasst(),
            "kosten-je-stufe": self.kosten_je_stufe(),
        }
        for preis in PREISFASSUNGEN:
            tabellen[f"vergleiche-je-seed-{preis}"] = self.vergleiche_auf_den_testdaten(preis)
            tabellen[f"vergleiche-{preis}"] = self.vergleiche_zusammengefasst(preis)
            tabellen[f"kosten-je-durchgang-{preis}"] = self.kosten_je_durchgang(preis)
            tabellen[f"kosten-{preis}"] = self.kosten_zusammengefasst(preis)
            tabellen[f"ff4-je-seed-{preis}"] = self.ff4(preis)
            tabellen[f"ff4-{preis}"] = self.ff4_zusammengefasst(preis)
            tabellen[f"pareto-{preis}"] = self.pareto(preis)
        for name, tabelle in tabellen.items():
            tabelle.to_csv(
                ordner / f"{name}.csv", sep=";", decimal=",", index=False, encoding="utf-8-sig"
            )
        _zeichne(self, ordner)
        (ordner / "bericht.md").write_text(fasse_als_bericht_zusammen(self), encoding="utf-8")


# --- Hilfen fuer Tabellen -------------------------------------------------------------------


def _spalte(name, preis):
    """Der Spaltenname einer Groesse in dieser Preisfassung: `name` oder `name_festgehalten`."""
    _lehne_unbekannte_preisfassung_ab(preis)
    return name if preis == "abgerechnet" else f"{name}_festgehalten"


def _lehne_unbekannte_preisfassung_ab(preis):
    if preis not in PREISFASSUNGEN:
        raise ValueError(f"Preisfassung {preis!r}: moeglich sind {', '.join(PREISFASSUNGEN)}.")


def _mittel_sd_als_text(spalte, stellen=3):
    return formatiere(*mittel_sd(list(spalte)), stellen)


def _mittel_sd_in_prozent(spalte):
    mittel, sd = mittel_sd(list(spalte))
    if mittel is None:
        return "–"
    return f"{_zahl(100 * mittel, 1)} % ± {_zahl(None if sd is None else 100 * sd, 1)} %"


def _ja_in_wie_vielen(spalte):
    return f"ja in {int(spalte.sum())} von {len(spalte)}"


def _name_des_gegners(durchgang):
    if durchgang.baseline == "ein-modell":
        return f"{BASELINENAMEN['ein-modell']}: {KURZNAMEN[durchgang.kandidatenmodell]}"
    return BASELINENAMEN[durchgang.baseline]


def _ordne(tabelle):
    """Baselines in der Reihenfolge des Experiments, Kandidatenmodelle in der von `DIE_VIER`."""
    if tabelle.empty:
        return tabelle
    reihenfolge_baseline = list(BASELINENAMEN.values())
    reihenfolge_modell = ["", *KURZNAMEN.values()]
    return tabelle.sort_values(
        ["baseline", "kandidatenmodell"],
        key=lambda spalte: spalte.map(
            (reihenfolge_baseline if spalte.name == "baseline" else reihenfolge_modell).index
        ),
    ).reset_index(drop=True)


def _markdown(tabelle, stellen=3):
    """Eine Tabelle als Markdown; Zahlen deutsch, fehlende als Gedankenstrich."""
    if tabelle.empty:
        return "_(keine Zeilen)_"

    def zelle(wert):
        if isinstance(wert, bool):
            return "ja" if wert else "nein"
        if isinstance(wert, float):
            return _zahl(wert, stellen)
        if wert is None:
            return "–"
        return str(KURZNAMEN.get(wert, wert)) if isinstance(wert, str) else str(wert)

    kopf = "| " + " | ".join(map(str, tabelle.columns)) + " |"
    trenner = "|" + "---|" * len(tabelle.columns)
    zeilen = [
        "| " + " | ".join(zelle(wert) for wert in zeile) + " |"
        for zeile in tabelle.itertuples(index=False)
    ]
    return "\n".join([kopf, trenner, *zeilen])


def _p_wert_als_text(wert):
    if _fehlt(wert):
        return "–"
    return "< 0,0001" if wert < 0.0001 else _zahl(wert, 4)


# --- Abbildungen ----------------------------------------------------------------------------

FARBEN = dict(zip(NAMEN, ("#4C72B0", "#55A868", "#C44E52", "#8172B2")))
MARKER = {"ohne-optimierung": "o", "gleichverteilung": "s", "ein-modell": "^", "verfahren": "*"}


def _zeichne(auswertung, ordner):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import FuncFormatter, ScalarFormatter

    sns.set_theme(style="whitegrid", context="paper")

    def ablegen(abbildung, name):
        for achse in abbildung.axes:
            for richtung in (achse.xaxis, achse.yaxis):
                if isinstance(richtung.get_major_formatter(), ScalarFormatter):
                    richtung.set_major_formatter(
                        FuncFormatter(lambda wert, _: f"{wert:g}".replace(".", ","))
                    )
        abbildung.tight_layout()
        abbildung.savefig(ordner / f"{name}.pdf")
        abbildung.savefig(ordner / f"{name}.png", dpi=200)
        plt.close(abbildung)

    endqualitaet = auswertung.endqualitaet()
    strategien = list(dict.fromkeys(endqualitaet["strategie"]))
    abbildung, achse = plt.subplots(figsize=(7, 4.2))
    for position, strategie in enumerate(strategien):
        werte = endqualitaet.loc[endqualitaet["strategie"] == strategie, "bewertung_test"]
        mittel, sd = mittel_sd(list(werte))
        hervorgehoben = strategie == "Verfahren"
        farbe = "#C44E52" if hervorgehoben else "0.25"
        # Die einzelnen Seeds, leicht versetzt, damit gleiche Werte nicht aufeinanderliegen.
        achse.scatter(
            werte, [position + 0.18] * len(werte), s=14, color=farbe, alpha=0.5, zorder=2,
            linewidths=0,
        )
        achse.errorbar(
            mittel, position, xerr=0 if sd is None else sd, fmt="D" if hervorgehoben else "o",
            color=farbe, markersize=7, capsize=3, elinewidth=1.2, zorder=3,
        )
    achse.set_yticks(range(len(strategien)), strategien)
    achse.invert_yaxis()
    achse.set_xlabel("F1-Score (Testdaten, n = 500)")
    achse.legend(
        handles=[
            plt.Line2D([], [], color="0.25", marker="o", linestyle="-",
                       label="Mittelwert ± Standardabweichung (3 Seeds)"),
            plt.Line2D([], [], color="0.25", marker="o", linestyle="", alpha=0.5,
                       markersize=4, label="einzelner Seed"),
        ],
        fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False,
    )
    ablegen(abbildung, "endqualitaet")

    for preis in PREISFASSUNGEN:
        punkte = auswertung.pareto(preis)
        abbildung, achse = plt.subplots(figsize=(7, 4.5))
        for punkt in punkte.itertuples():
            gefuellt = not punkt.dominiert
            achse.errorbar(
                punkt.kosten,
                punkt.bewertung,
                xerr=0 if _fehlt(punkt.kosten_sd) else punkt.kosten_sd,
                yerr=0 if _fehlt(punkt.bewertung_sd) else punkt.bewertung_sd,
                fmt=MARKER[punkt.baseline],
                color=FARBEN[punkt.kandidatenmodell],
                markerfacecolor=FARBEN[punkt.kandidatenmodell] if gefuellt else "white",
                markersize=14 if punkt.baseline == "verfahren" else 7,
                markeredgecolor="black" if punkt.baseline == "verfahren" else None,
                capsize=2,
                elinewidth=0.8,
                zorder=3 if punkt.baseline == "verfahren" else 2,
            )
        front = punkte[~punkte["dominiert"]].sort_values("kosten")
        achse.step(
            front["kosten"], front["bewertung"], where="post", color="0.4", linewidth=1, zorder=0
        )
        achse.axvline(KOSTENOBERGRENZE, color="0.3", linestyle="--", linewidth=1)
        achse.text(
            KOSTENOBERGRENZE,
            # Unterhalb der Punkte um die Grenze, damit die Beschriftung nichts verdeckt.
            achse.get_ylim()[0] + 0.2 * (achse.get_ylim()[1] - achse.get_ylim()[0]),
            f"Kostenobergrenze {_zahl(KOSTENOBERGRENZE, 4)} $ (Auswahldaten) ",
            rotation=90, va="bottom", ha="right", fontsize=7, color="0.3",
        )
        achse.set_xlabel(
            "Ausführungskosten ($ je 1000 Programmläufe, Testdaten"
            + (", zum Preis vom Stichtag)" if preis == "festgehalten" else ")")
        )
        achse.set_ylabel("F1-Score (Testdaten, n = 500)")
        legende = [
            plt.Line2D([], [], color=FARBEN[name], marker="o", linestyle="", label=KURZNAMEN[name])
            for name in NAMEN
        ] + [
            plt.Line2D([], [], color="0.3", marker=MARKER[baseline], linestyle="",
                       label=BASELINENAMEN[baseline])
            for baseline in MARKER
        ] + [
            plt.Line2D([], [], color="0.3", marker="o", markerfacecolor="white", linestyle="",
                       label="Pareto-dominiert (offenes Symbol)"),
            plt.Line2D([], [], color="0.4", linewidth=1, label="Pareto-Front"),
        ]
        achse.legend(handles=legende, fontsize=7, loc="lower right", ncol=2)
        ablegen(abbildung, f"pareto-{preis}")

    lernkurven = auswertung.lernkurven()
    lernkurven["Kandidatenmodell"] = lernkurven["kandidatenmodell"].map(KURZNAMEN)
    abbildung, achse = plt.subplots(figsize=(7.5, 4))
    sns.lineplot(
        data=lernkurven, x="budgetanteil", y="bewertung", hue="Kandidatenmodell",
        palette={KURZNAMEN[name]: FARBEN[name] for name in NAMEN},
        errorbar="sd", marker="o", err_style="bars", ax=achse,
    )
    achse.set_xscale("log")
    stufen = sorted(lernkurven["budgetanteil"].unique())
    achse.set_xticks(stufen, [f"{_zahl(stufe, 2)} $" for stufe in stufen])
    achse.minorticks_off()
    achse.legend(
        title="Kandidatenmodell", fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5)
    )
    achse.set_xlabel("Budgetanteil je Kandidatenmodell (B/8, B/4, B; logarithmisch)")
    achse.set_ylabel("F1-Score (Auswahldaten, n = 200)")
    ablegen(abbildung, "lernkurven")

    ff1 = auswertung.ff1()
    abbildung, achse = plt.subplots(figsize=(5, 4))
    for name in NAMEN:
        spalten = [f"bewertung_ohne_optimierung {name}", f"bewertung_gleichverteilung {name}"]
        mittel = [ff1[spalte].mean() for spalte in spalten]
        sd = [ff1[spalte].std() for spalte in spalten]
        achse.errorbar(
            [0, 1], mittel, yerr=sd, color=FARBEN[name], marker="o", capsize=3,
            label=KURZNAMEN[name],
        )
    achse.set_xticks([0, 1], ["Ohne Optimierung", "Optimiert\n(Gleichverteilung, B/4)"])
    achse.set_xlim(-0.3, 1.3)
    achse.set_ylabel("F1-Score (Testdaten, n = 500)")
    achse.legend(title="Kandidatenmodell", fontsize=7)
    ablegen(abbildung, "ff1-rangfolge")


# --- Bericht --------------------------------------------------------------------------------


def fasse_als_bericht_zusammen(auswertung):
    """Der Bericht fuer Kapitel 6: die zusammengefassten Tabellen, jeweils mit ihrer Quelle."""
    seeds = auswertung.seeds()
    ff1 = auswertung.ff1()
    ungueltige = auswertung.ungueltige_durchgaenge()
    tests = auswertung.testbewertungen()
    zwischen = auswertung.zwischenbewertungen()
    zwischen_ohne = auswertung.zwischenbewertungen(kostenobergrenze=math.inf)
    endauswahlen = [d.entscheidung for d in auswertung.durchgaenge if d.baseline == "verfahren"]
    auswahlen = auswertung.auswahlen()
    abweichende_auswahlen = auswahlen[
        (auswahlen["ohne_obergrenze"] != auswahlen["protokolliert"])
        | (auswahlen["festgehaltener_preis"] != auswahlen["protokolliert"])
    ]
    platzwechsel = auswertung.platzwechsel()
    kosten = auswertung.kosten_je_stufe()
    kosten_je_durchgang = auswertung.kosten_je_durchgang()
    nachbewertungen = {
        baseline: _verschiedene_werte(
            kosten_je_durchgang.loc[kosten_je_durchgang["baseline"] == baseline, "nachbewertungen"]
        )
        for baseline in ("verfahren", "gleichverteilung")
    }

    def p_tabelle(tabelle):
        spalten = ["seed", "durchgang", "weiter", "test_hat_gefunden"]
        ausgabe = tabelle[spalten].copy()
        for name in NAMEN:
            ausgabe[f"p {KURZNAMEN[name]}"] = tabelle[f"p_wert {name}"].map(_p_wert_als_text)
            ausgabe[f"Grund {KURZNAMEN[name]}"] = tabelle[f"ausgeschieden {name}"]
        return ausgabe

    wechsel = platzwechsel.copy()
    wechsel["p_mit"] = wechsel["p_mit"].map(_p_wert_als_text)
    wechsel["p_ohne"] = wechsel["p_ohne"].map(_p_wert_als_text)

    def vergleich(preis):
        je_seed = auswertung.vergleiche_auf_den_testdaten(preis)
        je_seed = je_seed[je_seed["gegen"] == BASELINENAMEN["gleichverteilung"]].copy()
        je_seed["p_wert"] = je_seed["p_wert"].map(_p_wert_als_text)
        return je_seed.drop(columns=["gegen"])

    text = [
        "# Auswertung des Hauptexperiments (Issue #13)",
        "",
        f"Erzeugt am {datetime.date.today():%d.%m.%Y} von `python -m dspy_optimizer.auswertung`, "
        f"allein aus `ergebnisse/protokoll.jsonl` ({len(auswertung.zeilen)} Zeilen), ohne einen "
        "API-Aufruf. Jede Ergebniszahl ist Mittelwert ± Standardabweichung (Stichprobe, n − 1) "
        f"über die Seeds {', '.join(map(str, seeds))}. Werte je Seed stehen im Bericht nur, wo "
        "ein Seed für sich zählt: Kendalls Tau je Durchgang (FF1), der gepaarte Test zweier "
        "Programme und Seed 3 in beiden Preisfassungen (ADR-0028); alle übrigen Tabellen „je "
        "Seed“ liegen als CSV daneben.",
        "",
        "Die Stichproben: **Verfahren gegen Gleichverteilung** hat als Stichprobe die drei Seeds "
        "— zu wenig für eine belastbare Signifikanzaussage; berichtet wird die Streuung. **Zwei "
        "fertige Programme auf den Testdaten** haben 500 gepaarte Beispiele; dort trägt ein "
        "gepaarter t-Test (zweiseitig, α = 0,05, ADR-0022, Nachtrag 28.09.2026).",
        "",
        f"Gültig: {len(auswertung.durchgaenge)} Durchgänge, {len(tests)} Modell-Programm-Paare "
        f"mit je einer gültigen Testbewertung über {_verschiedene_werte(tests['testbeispiele'])} "
        "Testbeispiele.",
        "",
        "## Ungültige Durchgänge und wiederholte Testbewertungen",
        "",
        f"{len(ungueltige)} ungültige Durchgänge, zusammen "
        f"{_zahl(ungueltige['dollar'].sum(), 2)} $. "
        "Ursache laut `stoerungen`: `LMRateLimitError` ist HTTP 429 von DeepInfra (überlasteter "
        "gemeinsamer Pool, ADR-0027, Nachtrag 15.10.2026); `KeyboardInterrupt` ist ein Abbruch "
        "von Hand (Strg+C). Ein Durchgang zählt nicht, sobald eine Zeile einen Anbieterfehler hat "
        "(ADR-0018).",
        "",
        _markdown(
            ungueltige.assign(
                kandidatenmodelle=ungueltige["kandidatenmodelle"].map(
                    lambda namen: ", ".join(KURZNAMEN[name] for name in namen.split(", "))
                )
            ),
            2,
        ),
        "",
        f"Wiederholte Testbewertungen nach Anbieterfehlern (ADR-0027, Punkt 12): "
        f"{auswertung.wiederholte_testbewertungen()}.",
        "",
        "Abweichung nach dem Stichtag: nach dem ungültigen Durchgang 14 (Seed 2) wiederholt jeder "
        "Aufruf bis zu 20-mal statt 10-mal (ADR-0003, Nachtrag 12.10.2026). Die gültigen "
        "Durchgänge 1–8, 12 und 13 liefen mit 10; in keinem davon hatte ein Aufruf die 10 "
        "Wiederholungen aufgebraucht, mit 20 wären sie also genauso verlaufen.",
        "",
        "## Kosten je Stufe",
        "",
        "Abgerechnet ist das Hauptergebnis (ADR-0012); der festgehaltene Preis ist die Gegenprobe "
        "zur Preiserhöhung bei DeepSeek (ADR-0028).",
        "",
        _markdown(kosten, 2),
        "",
        "## FF1: Rangfolge ohne gegen mit Optimierung (Testdaten)",
        "",
        f"Kendalls Tau: **{_mittel_sd_als_text(ff1['tau'])}**; die Spitze wechselt in "
        f"{int(ff1['spitze_wechselt'].sum())} von {len(ff1)} Seeds.",
        "",
        _markdown(ff1[["seed", "tau", "rangfolge_ohne", "rangfolge_mit", "spitze_wechselt"]]),
        "",
        "Platzwechsel je Seed, gepaarter t-Test zweiseitig auf den 500 Testbeispielen: `p_mit` "
        "zwischen den beiden Programmen der Gleichverteilung (der Unterschied der neuen "
        "Rangfolge), `p_ohne` zum Vergleich zwischen den beiden unoptimierten.",
        "",
        _markdown(wechsel) if len(wechsel) else "Kein Paar tauscht in einem Seed die Plätze.",
        "",
        "Belastbar ist ein Platzwechsel bei Signifikanz in mindestens zwei von drei Seeds "
        "(ADR-0022, Nachtrag 28.09.2026)"
        + (
            ":\n\n" + _markdown(auswertung.platzwechsel_zusammengefasst())
            if len(wechsel)
            else "; ohne Platzwechsel gibt es keinen zu prüfen."
        ),
        "",
        "## FF2: Zwischenbewertung — wie oft der Test etwas gefunden hat",
        "",
        f"Mit Kostenobergrenze fand der Test in {int(zwischen['test_hat_gefunden'].sum())} von "
        f"{len(zwischen)} Zwischenbewertungen etwas (einseitig, α = 0,05, ADR-0004). In der "
        "Endauswahl gab es "
        f"{sum(e.get('vergleiche') or 0 for e in endauswahlen)} Vergleiche: in die Fokusphase ging "
        f"jeweils {', '.join(str(len(e.get('kandidatenmodelle') or {})) for e in endauswahlen)} "
        "Kandidatenmodell(e). Der Zusatzbeleg „sortiert die Sondierung schon richtig?“ (Rangfolge "
        "Sondierung gegen Fokusphase, ADR-0022) lässt sich deshalb nicht rechnen: mit einem "
        "Kandidatenmodell in der Fokusphase gibt es dort keine Rangfolge.",
        "",
        _markdown(p_tabelle(zwischen)),
        "",
        "Ohne Kostenobergrenze nachgerechnet (ADR-0021, Nachtrag 15.10.2026):",
        "",
        _markdown(p_tabelle(zwischen_ohne)),
        "",
        "## Jede Auswahl mit und ohne Kostenobergrenze",
        "",
        "Nachgerechnet aus den Einzelbewertungen und Ausführungskosten der Nachbewertungen. "
        "`nachgerechnet` muss `protokolliert` gleichen"
        + (" — tut es." if (auswahlen["nachgerechnet"] == auswahlen["protokolliert"]).all()
           else " — **tut es nicht, prüfen!**")
        + " Die Einzelmodelle wählen nichts; ob ihr Programm unter der Obergrenze liegt, steht "
        "unter „Programme“ (ADR-0021, Nachtrag 15.10.2026).",
        "",
        "Abweichend von der protokollierten Auswahl:",
        "",
        _markdown(abweichende_auswahlen),
        "",
        "Alle Auswahlen: `auswahlen.csv`.",
        "",
        "## Verfahren gegen Gleichverteilung",
        "",
        "Beide zahlen ihre Nachbewertungen aus dem Budget B (ADR-0017). Geplant waren beim "
        "Verfahren höchstens 6 (vier in der Sondierung, zwei in der Fokusphase); tatsächlich "
        f"zahlte es {nachbewertungen['verfahren']} je Durchgang, weil jeweils nur ein "
        "Kandidatenmodell in die Fokusphase ging. Die Gleichverteilung zahlte "
        f"{nachbewertungen['gleichverteilung']}. Kosten unter „Kosten“.",
        "",
        "**Über die drei Seeds** (Stichprobe: Seeds; keine Signifikanzaussage, nur die Streuung):",
        "",
        _markdown(
            pd.DataFrame(
                [
                    {
                        "Preisfassung": {
                            "abgerechnet": "abgerechnet",
                            "festgehalten": "zum Preis vom Stichtag",
                        }[preis],
                        **auswertung.vergleiche_zusammengefasst(preis)
                        .iloc[0][["F1 Verfahren", "F1 Gegner", "Differenz"]]
                        .rename({"F1 Gegner": "F1 Gleichverteilung"})
                        .to_dict(),
                    }
                    for preis in PREISFASSUNGEN
                ]
            )
        ),
        "",
        "**Programmvergleich je Seed** (Stichprobe: die 500 Testbeispiele, gepaarter t-Test "
        "zweiseitig). Er sagt, ob sich zwei fertige Programme unterscheiden, nicht, ob das "
        "Verfahren der Gleichverteilung überlegen ist. In Seed 1 und 2 sind beide Programme von "
        "DeepSeek. Seed 3 in beiden Preisfassungen (ADR-0028, Entscheidung 3).",
        "",
        "Abgerechnet:",
        "",
        _markdown(vergleich("abgerechnet")),
        "",
        "Zum festgehaltenen Preis:",
        "",
        _markdown(vergleich("festgehalten")),
        "",
        "Gegen jede Baseline, abgerechnet:",
        "",
        _markdown(auswertung.vergleiche_zusammengefasst("abgerechnet")),
        "",
        "Gegen jede Baseline, zum festgehaltenen Preis:",
        "",
        _markdown(auswertung.vergleiche_zusammengefasst("festgehalten")),
        "",
        "## Endqualität je Strategie (FF3)",
        "",
        "F1 auf den Testdaten des Programms, das jede Strategie wählt; beim Einzelmodell sein "
        "Programm, auch über der Kostenobergrenze (dazu „Programme“). In Klammern, wie oft welches "
        "Kandidatenmodell gewählt wurde. Abbildung `endqualitaet.pdf`.",
        "",
        _markdown(auswertung.endqualitaet_zusammengefasst()),
        "",
        "## Programme der Baselines, ohne Gewinner-Etikett",
        "",
        "Je Baseline und Kandidatenmodell; „unter der Kostenobergrenze?“ nach den "
        "Ausführungskosten auf den Auswahldaten, nach denen die Regel wählt (ADR-0020). Die "
        "Latenz wird nur berichtet und geht in keine Auswahl ein (ADR-0019, Punkt 5).",
        "",
        _markdown(auswertung.programme_zusammengefasst()),
        "",
        "## Kosten: Optimierung, Nachbewertung, Plan",
        "",
        "C_opt = Optimierung + Nachbewertung (ADR-0022). Abweichung vom Plan: gegen die "
        "geplanten Dollar beider Messläufe (ADR-0017); Abweichung von B: gegen das "
        "Optimierungsbudget von 6 $. Die Ausführungskosten stehen getrennt davon unter "
        "„Programme“ und „FF4“.",
        "",
        "Abgerechnet:",
        "",
        _markdown(auswertung.kosten_zusammengefasst("abgerechnet")),
        "",
        "Zum festgehaltenen Preis:",
        "",
        _markdown(auswertung.kosten_zusammengefasst("festgehalten")),
        "",
        "## FF4: Optimierungskosten gegen Ausführungskosten",
        "",
        "C_opt in Programmläufen = C_opt / c_exec des gewählten Paars; Break-even gegen das Paar "
        "der Baseline ohne Optimierung im selben Seed, nur wenn das gewählte billiger ist "
        "(ADR-0022). c_exec aus der Testbewertung. Getrennt je gewähltem Kandidatenmodell, weil "
        "ein Mittel über verschiedene Paare nichts bedeutet. **Ohne Modellwechsel** (gewählt ist "
        "dasselbe Kandidatenmodell wie ohne Optimierung) ist ein Break-even kein Modellwechsel im "
        "Sinne von ADR-0022, sondern optimiertes gegen unoptimiertes Programm desselben Modells; "
        "die Kostendifferenz ist dann oft winzig und der Break-even entsprechend groß und "
        "unsicher.",
        "",
        "Abgerechnet:",
        "",
        _markdown(auswertung.ff4_zusammengefasst("abgerechnet")),
        "",
        "Zum festgehaltenen Preis:",
        "",
        _markdown(auswertung.ff4_zusammengefasst("festgehalten")),
        "",
        "## Lernkurven (FF3)",
        "",
        "Drei Budgetstufen aus vorhandenen Läufen, Bewertung der Nachbewertung auf den "
        "Auswahldaten (ADR-0022, Nachtrag 28.09.2026). Einschränkung: verschiedene Läufe, keine "
        "durchgehende Kurve.",
        "",
        _markdown(auswertung.lernkurven_zusammengefasst()),
        "",
        "## Abbildungen",
        "",
        "- `endqualitaet.pdf/.png`: F1 auf den Testdaten je Strategie, Mittelwert ± "
        "Standardabweichung und die drei Seeds als Punkte (FF3).",
        "- `pareto-abgerechnet.pdf/.png`, `pareto-festgehalten.pdf/.png`: Qualität gegen "
        "Ausführungskosten auf den Testdaten, je Baseline (Form) und Kandidatenmodell (Farbe), "
        "Mittelwert ± Standardabweichung; offene Symbole = Pareto-dominiert; graue Linie = "
        "Pareto-Front.",
        "- `lernkurven.pdf/.png`: F1 je Budgetstufe, Mittelwert ± Standardabweichung.",
        "- `ff1-rangfolge.pdf/.png`: F1 auf den Testdaten ohne Optimierung und nach "
        "Gleichverteilung je Kandidatenmodell.",
        "",
    ]
    return "\n".join(text)


def _verschiedene_werte(spalte):
    werte = sorted(set(spalte))
    return ", ".join(map(str, werte))


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    auswertung = Auswertung(Protokoll().zeilen())
    auswertung.schreibe(AUSGABE)
    print(f"Auswertung geschrieben nach {AUSGABE}")


if __name__ == "__main__":
    main()
