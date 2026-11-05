"""Tests zur Kostenerfassung.

Gemessen wird gegen eine Attrappe, die sich wie ein bezahltes Modell verhaelt: sie meldet je
Aufruf Tokens, einen Dollar-Betrag und einen bedienenden Anbieter. Damit laesst sich die
Erfassung vollstaendig pruefen, ohne einen Aufruf zu bezahlen - auch der Fall, auf den es
ankommt: die Aufrufe aus der **parallelen** Evaluation.
"""

import time

import dspy
import pytest
from dspy.dsp.utils.utils import dotdict
from dspy.utils.callback import BaseCallback
from dspy.utils.dummies import DummyLM

from dspy_optimizer.kandidatenmodelle import mess_adapter
from dspy_optimizer.kostenerfassung import (
    Kostenzaehler,
    Tokens,
    messhygiene,
    messlauf,
    pruefe_messhygiene,
)
from dspy_optimizer.metrik import bewerte
from dspy_optimizer.programm import MultiHopQA
from dspy_optimizer.protokoll import VORSTUDIE, Einordnung, Protokoll

# Was die Attrappe je Aufruf meldet. Zwei Aufrufe sind ein Programmlauf.
PROMPT_TOKENS = 10
ANTWORT_TOKENS = 5
DOLLAR_JE_AUFRUF = 0.001
ANBIETER = "novita"

ABSAETZE = [{"title": "A", "sentences": ["Der Eiffelturm steht in Paris."]}]


class BezahlteAttrappe(DummyLM):
    """Ein DummyLM, das Tokens, Kosten und den bedienenden Anbieter mitliefert.

    Geantwortet wird nach Inhalt des Prompts und nicht der Reihe nach: bei acht Threads
    greifen sonst mehrere Threads gleichzeitig auf denselben Antwort-Iterator zu, und die
    Antworten landen beim falschen Modul.
    """

    def __init__(self):
        super().__init__({
            # Schluessel = ein Feldname, der nur im Prompt des jeweiligen Moduls vorkommt.
            "relevant_titles": {"relevant_titles": ["A"], "relevant_text": "In Paris."},
            "answer": {"answer": "France"},
        })

    def forward(self, prompt=None, messages=None, **kwargs):
        antwort = super().forward(prompt=prompt, messages=messages, **kwargs)
        antwort.usage = dotdict(
            prompt_tokens=PROMPT_TOKENS,
            completion_tokens=ANTWORT_TOKENS,
            total_tokens=PROMPT_TOKENS + ANTWORT_TOKENS,
        )
        antwort._hidden_params = {"response_cost": DOLLAR_JE_AUFRUF}
        antwort.provider = ANBIETER
        return antwort


@pytest.fixture
def beispiele():
    return [
        dspy.Example(
            question=f"Frage {nummer}",
            answer="France",
            paragraphs=ABSAETZE,
            gold_titles=["A"],
        ).with_inputs("question", "paragraphs")
        for nummer in range(4)
    ]


@pytest.fixture
def verzeichnis(tmp_path):
    """Ein Protokollverzeichnis, wie es MIPROv2 als `log_dir` bekommt."""
    return tmp_path / "lauf"


@pytest.fixture
def protokoll(tmp_path):
    """Das Protokoll des Tests - nie das des Repos."""
    return Protokoll(tmp_path / "protokoll.jsonl")


@pytest.fixture(autouse=True)
def saubere_messhygiene(verzeichnis):
    """Jeder Test startet mit abgeschaltetem Cache und einem vorhandenen Verzeichnis."""
    verzeichnis.mkdir(parents=True, exist_ok=True)
    messhygiene()


@pytest.fixture
def abschnitt(verzeichnis, protokoll):
    """Ein Messlauf mit den Angaben, die in jedem Test gleich sind."""

    def baue(*sprachmodelle, **abweichend):
        felder = {
            "protokollverzeichnis": verzeichnis,
            "abschnitt": "nachbewertung",
            "einordnung": VORSTUDIE,
            "konfiguration": {"kandidatenmodell": "attrappe"},
            "protokoll": protokoll,
        }
        return messlauf(*sprachmodelle, **{**felder, **abweichend})

    return baue


def auswertung(beispiele, threads=4):
    return dspy.Evaluate(devset=beispiele, metric=bewerte, num_threads=threads)


def test_der_zaehler_erfasst_die_aufrufe_aus_der_parallelen_evaluation(beispiele, abschnitt):
    """Der zentrale Befund: der Evaluationsanteil laeuft in Arbeits-Threads.

    Vier Beispiele, je zwei LM-Aufrufe. Faellt der Evaluationsanteil heraus, steht hier 0.
    """
    lm = BezahlteAttrappe()

    with dspy.context(lm=lm):
        with abschnitt(lm) as erfassung:
            auswertung(beispiele)(MultiHopQA())
    messwerte = erfassung.messwerte()

    assert messwerte.programmlaeufe == 4
    assert messwerte.lm_aufrufe == 8
    assert messwerte.tokens_aus_evaluationen == Tokens(
        prompt=8 * PROMPT_TOKENS, antwort=8 * ANTWORT_TOKENS
    )
    assert messwerte.dollar == pytest.approx(8 * DOLLAR_JE_AUFRUF)


def test_je_aufruf_wird_der_bedienende_anbieter_mitgeschrieben(beispiele, abschnitt):
    """Belegt statt geglaubt, und bezifferbar: ein einziger ausgewichener Aufruf faellt auf.

    Gezaehlt wird je Anbieter, nicht nur vermerkt, dass er vorkam.
    """
    lm = BezahlteAttrappe()

    with dspy.context(lm=lm):
        with abschnitt(lm) as erfassung:
            auswertung(beispiele)(MultiHopQA())

    assert erfassung.messwerte().anbieter == {ANBIETER: 8}


def test_der_hauptthread_anteil_ist_die_differenz_zum_evaluationsanteil(beispiele, abschnitt):
    """Bootstrapping und Proposer laufen im Hauptthread und gehen durch keine Evaluation.

    Sie stehen trotzdem in der History - der Hauptthread-Anteil ist deshalb die Differenz
    zwischen History und Evaluationsanteil.
    """
    lm = BezahlteAttrappe()

    with dspy.context(lm=lm):
        with abschnitt(lm) as erfassung:
            MultiHopQA()(**beispiele[0].inputs())  # Hauptthread: zwei Aufrufe
            auswertung(beispiele)(MultiHopQA())  # Evaluation: acht Aufrufe

    messwerte = erfassung.messwerte()

    assert messwerte.lm_aufrufe == 10
    assert messwerte.tokens_aus_evaluationen.gesamt == 8 * (PROMPT_TOKENS + ANTWORT_TOKENS)
    assert messwerte.tokens_aus_dem_hauptthread.gesamt == 2 * (PROMPT_TOKENS + ANTWORT_TOKENS)


def test_ein_zweiter_abschnitt_zaehlt_die_aufrufe_des_ersten_nicht_noch_einmal(
    beispiele, abschnitt
):
    """Genau so laeuft das Verfahren: erst optimieren, dann das fertige Programm ausfuehren.

    Die History haelt beide Abschnitte. Zaehlte der zweite von vorn, steckten die
    Optimierungskosten in den Ausfuehrungskosten - und die Trennung waere hinfaellig.
    """
    lm = BezahlteAttrappe()

    with dspy.context(lm=lm):
        with abschnitt(lm) as erster:
            auswertung(beispiele)(MultiHopQA())
        with abschnitt(lm) as zweiter:
            auswertung(beispiele[:2])(MultiHopQA())

    assert erster.messwerte().lm_aufrufe == 8
    assert zweiter.messwerte().lm_aufrufe == 4
    assert zweiter.messwerte().dollar == pytest.approx(4 * DOLLAR_JE_AUFRUF)


def test_ein_aeusserer_usage_tracker_schaltet_die_erste_quelle_stumm(beispiele):
    """Gemessen, nicht angenommen - und der Grund, warum die Kostenerfassung nicht auf
    `dspy.track_usage()` aufbaut.

    `dspy.track_usage()` legt einen Tracker in die Thread-Ueberschreibungen. Der
    Arbeits-Thread bekommt eine Kopie davon, und weil damit schon ein Tracker gesetzt ist,
    legt das Programm-Modul keinen eigenen mehr an (`primitives/module.py` Z. 101-106) - die
    Einzelergebnisse tragen dann keinen Verbrauch mehr.
    """
    lm = BezahlteAttrappe()
    verbrauch = []

    class Mitschrift(BaseCallback):
        def on_evaluate_end(self, call_id, outputs, exception=None):
            verbrauch.extend(ergebnis.get_lm_usage() for _, ergebnis, _ in outputs.results)

    with dspy.context(lm=lm, track_usage=True, callbacks=[Mitschrift()]):
        with dspy.track_usage():
            auswertung(beispiele)(MultiHopQA())

    assert verbrauch == [None, None, None, None]


def test_ein_aeusserer_usage_tracker_wird_vor_dem_lauf_abgelehnt(abschnitt):
    """Deshalb ein harter Abbruch statt einer stillen Untererfassung."""
    lm = BezahlteAttrappe()

    with dspy.track_usage():
        with pytest.raises(ValueError, match="track_usage"):
            with abschnitt(lm):
                pass


@pytest.mark.parametrize(
    "abweichend, gesucht",
    [
        ({"abschnitt": "ausfuehrung"}, "ausfuehrung"),
        ({"einordnung": "probelauf"}, "Einordnung"),
        ({"abschnitt": None, "einordnung": Einordnung.experiment("ein-modell", durchgang=1)},
         "Abschnitt"),
    ],
)
def test_eine_ungueltige_angabe_faellt_auf_bevor_der_lauf_geld_kostet(
    abschnitt, protokoll, abweichend, gesucht
):
    """Sonst stuende die Zeile erst nach dem bezahlten Lauf zur Debatte."""
    lm = BezahlteAttrappe()
    betreten = False

    with pytest.raises(ValueError, match=gesucht):
        with abschnitt(lm, **abweichend):
            betreten = True

    assert not betreten
    assert protokoll.zeilen() == []


def test_programmlaeufe_ausserhalb_einer_evaluation_werden_vermerkt(abschnitt):
    """Der Lauf von Hand geht nicht durch `dspy.Evaluate` und meldet sich deshalb selbst."""
    lm = BezahlteAttrappe()

    with dspy.context(lm=lm):
        with abschnitt(lm) as erfassung:
            erfassung.vermerke_programmlaeufe(3)

    assert erfassung.messwerte().programmlaeufe == 3


def test_die_ausfuehrungskosten_sind_die_kosten_eines_einzelnen_programmlaufs(
    beispiele, abschnitt
):
    """Ausfuehrungskosten sind je Programmlauf bemessen, nicht je Lauf."""
    lm = BezahlteAttrappe()

    with dspy.context(lm=lm):
        with abschnitt(lm) as erfassung:
            auswertung(beispiele)(MultiHopQA())

    assert erfassung.messwerte().dollar_je_programmlauf == pytest.approx(2 * DOLLAR_JE_AUFRUF)


def test_ohne_programmlauf_sind_die_kosten_je_programmlauf_unbekannt_statt_null(abschnitt):
    lm = BezahlteAttrappe()

    with dspy.context(lm=lm):
        with abschnitt(lm) as erfassung:
            pass

    assert erfassung.messwerte().dollar_je_programmlauf is None


def test_jeder_abschnitt_hinterlaesst_genau_eine_protokollzeile(beispiele, abschnitt, protokoll):
    """Je Lauf eine Zeile - geschrieben von derselben Stelle, die auch zaehlt."""
    lm = BezahlteAttrappe()

    with dspy.context(lm=lm):
        with abschnitt(lm, seed=7) as erfassung:
            auswertung(beispiele)(MultiHopQA())
            erfassung.vermerke_bewertung(0.75)

    (zeile,) = protokoll.zeilen()
    assert zeile["abschnitt"] == "nachbewertung"
    assert zeile["zweck"] == "vorstudie"
    assert zeile["seed"] == 7
    assert zeile["bewertung"] == 0.75
    assert zeile["dollar"] == pytest.approx(8 * DOLLAR_JE_AUFRUF)
    assert zeile["anbieter"] == {ANBIETER: 8}


def test_exact_match_und_absatztreffer_gehen_mit_in_die_zeile(beispiele, abschnitt, protokoll):
    """Der Zaehler reicht sie nur durch - er weiss nicht, was ein Lauf wert war."""
    lm = BezahlteAttrappe()

    with dspy.context(lm=lm):
        with abschnitt(lm) as erfassung:
            erfassung.vermerke_bewertung(0.75, exakt=0.5, absatztreffer=1.0)

    (zeile,) = protokoll.zeilen()
    assert zeile["bewertung"] == 0.75
    assert zeile["exakt"] == 0.5
    assert zeile["absatztreffer"] == 1.0


def test_auch_ein_abgestuerzter_lauf_hinterlaesst_seine_zeile(beispiele, abschnitt, protokoll):
    """Das Geld ist ausgegeben, ob der Lauf durchlief oder nicht.

    Der Fehler selbst wird durchgereicht: verschwiegen wird hier nichts.
    """
    lm = BezahlteAttrappe()

    with pytest.raises(RuntimeError, match="Anbieter weg"):
        with dspy.context(lm=lm):
            with abschnitt(lm):
                auswertung(beispiele)(MultiHopQA())
                raise RuntimeError("Anbieter weg")

    (zeile,) = protokoll.zeilen()
    assert zeile["dollar"] == pytest.approx(8 * DOLLAR_JE_AUFRUF)
    assert any("abgebrochen" in stoerung for stoerung in zeile["stoerungen"])


def evaluationsergebnis(ergebnisse):
    return dspy.evaluate.evaluate.EvaluationResult(score=0.0, results=ergebnisse)


def test_eine_verschluckte_evaluation_wird_als_stoerung_sichtbar():
    """DSPy macht aus einem Fehler in der Evaluation still eine Bewertung von 0,0.

    `eval_candidate_program` faengt jede Exception und gibt `score=0.0, results=[]` zurueck
    (`teleprompt/utils.py` Z. 59-62). Ein Modell mit Rate-Limit-Problemen saehe damit aus wie
    ein schlechtes Modell.
    """
    zaehler = Kostenzaehler()

    zaehler.on_evaluate_end("id", evaluationsergebnis([]))

    assert len(zaehler.messwerte().stoerungen) == 1
    assert "0,0" in zaehler.messwerte().stoerungen[0]


def test_eine_abgebrochene_evaluation_wird_als_stoerung_sichtbar():
    zaehler = Kostenzaehler()

    zaehler.on_evaluate_end("id", None, exception=TimeoutError("Anbieter antwortet nicht"))

    assert "Anbieter antwortet nicht" in zaehler.messwerte().stoerungen[0]


def test_ein_fehlgeschlagener_programmlauf_wird_gezaehlt_und_nicht_verschwiegen():
    """Faellt ein einzelnes Beispiel aus, ersetzt DSPy es durch eine leere Prediction mit
    der Bewertung 0,0 (`evaluate/evaluate.py` Z. 178). Die zaehlt hier als Fehlschlag."""
    gelungen = dspy.Prediction(answer="France")
    gelungen.set_lm_usage({"dummy": {"prompt_tokens": 1, "completion_tokens": 1}})
    zaehler = Kostenzaehler()

    zaehler.on_evaluate_end(
        "id",
        evaluationsergebnis([("beispiel", gelungen, 1.0), ("beispiel", dspy.Prediction(), 0.0)]),
    )
    messwerte = zaehler.messwerte()

    assert messwerte.programmlaeufe == 2
    assert messwerte.fehlgeschlagene_programmlaeufe == 1
    assert messwerte.stoerungen  # sichtbar, nicht nur gezaehlt


def test_die_history_wird_zwischen_den_evaluationen_abgeholt(beispiele, abschnitt):
    """Die History ist ein Ringpuffer mit 10.000 Eintraegen (`base_lm.py` Z. 786-800).

    Ein Optimierungslauf sprengt den; wer erst am Ende summiert, misst zu wenig. Hier fasst
    der Puffer eine Evaluation, aber nicht zwei - trotzdem muessen alle 16 Aufrufe in der
    Summe stehen.
    """
    lm = BezahlteAttrappe()

    with dspy.context(lm=lm, max_history_size=10):
        with abschnitt(lm) as erfassung:
            auswertung(beispiele)(MultiHopQA())
            auswertung(beispiele)(MultiHopQA())

    messwerte = erfassung.messwerte()

    assert messwerte.lm_aufrufe == 16
    assert messwerte.dollar == pytest.approx(16 * DOLLAR_JE_AUFRUF)


def test_ein_ueberlaufender_ringpuffer_wird_gemeldet_statt_still_zu_verschlucken(
    beispiele, abschnitt
):
    """Laeuft der Puffer *innerhalb* einer Evaluation ueber, ist die Dollar-Summe zu klein.

    Das darf nicht unbemerkt bleiben: die Gegenprobe zwischen beiden Quellen deckt es auf.
    """
    lm = BezahlteAttrappe()

    with dspy.context(lm=lm, max_history_size=3):
        with abschnitt(lm) as erfassung:
            auswertung(beispiele)(MultiHopQA())

    messwerte = erfassung.messwerte()

    assert messwerte.lm_aufrufe < 8
    assert any("Ringpuffer" in stoerung for stoerung in messwerte.stoerungen)


def test_mehrere_sprachmodelle_werden_zusammen_erfasst(beispiele, abschnitt):
    """Im Verfahren laeuft neben dem Kandidatenmodell ein zweites Modell als Proposer.

    Es hat eine eigene History; beide muessen in dieselbe Rechnung.
    """
    kandidatenmodell = BezahlteAttrappe()
    proposer = BezahlteAttrappe()

    with abschnitt(kandidatenmodell, proposer) as erfassung:
        with dspy.context(lm=kandidatenmodell):
            auswertung(beispiele)(MultiHopQA())
        with dspy.context(lm=proposer):
            MultiHopQA()(**beispiele[0].inputs())

    assert erfassung.messwerte().lm_aufrufe == 10


def test_die_messhygiene_schaltet_beide_cache_schichten_ab():
    """Ein Cache-Treffer kostet nichts und meldet weder Tokens noch Kosten."""
    dspy.configure_cache(enable_disk_cache=True, enable_memory_cache=True)

    messhygiene()

    assert dspy.cache.enable_disk_cache is False
    assert dspy.cache.enable_memory_cache is False


def test_ein_eingeschalteter_cache_bricht_den_messlauf_ab(verzeichnis):
    dspy.configure_cache(enable_disk_cache=True, enable_memory_cache=True)

    with pytest.raises(ValueError, match="Cache"):
        pruefe_messhygiene(verzeichnis)


def test_ein_fehlendes_protokollverzeichnis_bricht_den_messlauf_ab(tmp_path):
    """`log_dir` ist der einzige Absturzschutz von MIPROv2."""
    with pytest.raises(ValueError, match="Protokollverzeichnis"):
        pruefe_messhygiene(tmp_path / "gibt-es-nicht")


def test_eingeschaltetes_streaming_bricht_den_messlauf_ab(verzeichnis):
    with dspy.context(send_stream=object()):
        with pytest.raises(ValueError, match="Streaming"):
            pruefe_messhygiene(verzeichnis)


def test_der_messlauf_schaltet_traceback_und_verbrauchserfassung_ein(abschnitt):
    """Ohne `provide_traceback` sieht ein Rate-Limit-Problem aus wie ein schlechtes Modell."""
    lm = BezahlteAttrappe()

    with abschnitt(lm):
        assert dspy.settings.provide_traceback is True
        assert dspy.settings.track_usage is True

    assert dspy.settings.provide_traceback is False


# Wie lange die wartende Attrappe je LM-Aufruf braucht. Zwei Aufrufe sind ein Programmlauf.
WARTEZEIT = 0.05


class WartendeAttrappe(BezahlteAttrappe):
    """Eine bezahlte Attrappe mit fester Wartezeit je Aufruf - eine Latenz ohne API-Aufruf."""

    def forward(self, prompt=None, messages=None, **kwargs):
        time.sleep(WARTEZEIT)
        return super().forward(prompt=prompt, messages=messages, **kwargs)


def test_die_zeile_traegt_die_dauer_des_messlaufs(abschnitt, protokoll):
    """Wie lange ein Abschnitt lief, in Sekunden."""
    with abschnitt(BezahlteAttrappe()):
        time.sleep(WARTEZEIT)

    (zeile,) = protokoll.zeilen()
    assert WARTEZEIT <= zeile["dauer_sekunden"] < 1.0


def test_auch_ein_abgebrochener_messlauf_traegt_seine_dauer(abschnitt, protokoll):
    """Die Zeile eines Abbruchs ist genauso vollstaendig wie jede andere."""
    with pytest.raises(RuntimeError, match="Anbieter weg"):
        with abschnitt(BezahlteAttrappe()):
            time.sleep(WARTEZEIT)
            raise RuntimeError("Anbieter weg")

    (zeile,) = protokoll.zeilen()
    assert zeile["dauer_sekunden"] >= WARTEZEIT


def test_die_latenz_ist_die_dauer_eines_programmlaufs(beispiele, abschnitt, protokoll):
    """Die Wartezeit auf eine beantwortete Frage, nicht auf einen LM-Aufruf.

    Zwei Aufrufe je Programmlauf, also mindestens die doppelte Wartezeit - auch wenn vier
    Threads gleichzeitig laufen.
    """
    lm = WartendeAttrappe()

    with dspy.context(lm=lm):
        with abschnitt(lm) as erfassung:
            auswertung(beispiele)(erfassung.mit_stoppuhr(MultiHopQA()))

    (zeile,) = protokoll.zeilen()
    assert 2 * WARTEZEIT <= zeile["latenz_mittelwert_sekunden"] < 2 * WARTEZEIT + 0.5
    assert 2 * WARTEZEIT <= zeile["latenz_median_sekunden"] < 2 * WARTEZEIT + 0.5
    assert zeile["programmlaeufe"] == len(beispiele)


def test_gescheiterte_programmlaeufe_zaehlen_nicht_zur_latenz(beispiele, abschnitt, protokoll):
    """Ein gescheiterter Programmlauf hat keine Frage beantwortet.

    Er wartet hier viel laenger als die gelungenen, bevor er scheitert; zaehlte er mit,
    laege der Mittelwert weit ueber der doppelten Wartezeit.
    """

    class ScheitertBeiFrageNull(MultiHopQA):
        def forward(self, question, paragraphs):
            if question == "Frage 0":
                time.sleep(20 * WARTEZEIT)
                raise RuntimeError("Formatfehler")
            return super().forward(question=question, paragraphs=paragraphs)

    lm = WartendeAttrappe()

    with dspy.context(lm=lm):
        with abschnitt(lm) as erfassung:
            dspy.Evaluate(devset=beispiele, metric=bewerte, num_threads=4, max_errors=10)(
                erfassung.mit_stoppuhr(ScheitertBeiFrageNull())
            )

    (zeile,) = protokoll.zeilen()
    assert zeile["latenz_mittelwert_sekunden"] < 10 * WARTEZEIT
    assert zeile["fehlgeschlagene_programmlaeufe"] == 1


def test_ohne_programmlauf_ist_die_latenz_unbekannt_statt_null(abschnitt, protokoll):
    """Keine Messung ist etwas anderes als eine Messung von null."""
    with abschnitt(BezahlteAttrappe()):
        pass

    (zeile,) = protokoll.zeilen()
    assert zeile["latenz_mittelwert_sekunden"] is None
    assert zeile["latenz_median_sekunden"] is None


def test_die_einzelbewertungen_gehen_in_der_gemeldeten_reihenfolge_in_die_zeile(
    abschnitt, protokoll
):
    """Der gepaarte Test ueber die Testbeispiele braucht sie."""
    with abschnitt(BezahlteAttrappe()) as erfassung:
        erfassung.vermerke_bewertung(0.5, einzelbewertungen=(1.0, 0.0, 0.5, 0.5))

    (zeile,) = protokoll.zeilen()
    assert zeile["einzelbewertungen"] == [1.0, 0.0, 0.5, 0.5]


def test_ohne_gemeldete_einzelbewertungen_bleibt_das_feld_leer(abschnitt, protokoll):
    with abschnitt(BezahlteAttrappe()) as erfassung:
        erfassung.vermerke_bewertung(0.5)

    (zeile,) = protokoll.zeilen()
    assert zeile["einzelbewertungen"] is None


def test_latenz_und_fehlschlaege_folgen_derselben_definition(beispiele, abschnitt):
    """Gestoppt wird genau, was der Zaehler als gelungenen Programmlauf zaehlt.

    Scheitert erst die Metrik, ist das Programm schon durchgelaufen - aber DSPy ersetzt das
    Ergebnis durch eine leere Prediction mit 0,0 (`evaluate/evaluate.py` Z. 170-178), und der
    Zaehler fuehrt den Lauf als gescheitert. Dann darf er auch nicht in der Latenz stehen.
    """

    def metrik_scheitert_bei_frage_null(beispiel, vorhersage, trace=None):
        if beispiel.question == "Frage 0":
            raise RuntimeError("Metrik kaputt")
        return bewerte(beispiel, vorhersage, trace)

    lm = BezahlteAttrappe()

    with dspy.context(lm=lm):
        with abschnitt(lm) as erfassung:
            dspy.Evaluate(
                devset=beispiele, metric=metrik_scheitert_bei_frage_null, num_threads=4,
                max_errors=10,
            )(erfassung.mit_stoppuhr(MultiHopQA()))

    messwerte = erfassung.messwerte()
    assert messwerte.fehlgeschlagene_programmlaeufe == 1
    assert len(messwerte.latenzen) == messwerte.programmlaeufe - 1


# Formatfehler und Anbieterfehler. Beide machen aus einem Programmlauf
# still eine Einzelbewertung von 0,0; nur der Formatfehler sagt dabei etwas ueber das Modell.


class AbgelehnteAttrappe(BezahlteAttrappe):
    """Ein Endpunkt, der jede Anfrage ablehnt - auch nach allen Wiederholungen."""

    def forward(self, prompt=None, messages=None, **kwargs):
        raise RuntimeError("429 Too Many Requests")


class UnlesbareAttrappe(BezahlteAttrappe):
    """Die Antwort kommt an und kostet, aber ohne die Marken, die der ChatAdapter sucht."""

    def __init__(self):
        super().__init__()
        # Kein Schluessel passt mehr: DummyLM antwortet dann "No more responses".
        self.answers = {}


class FlackerndeAttrappe(BezahlteAttrappe):
    """Wiederholt selbst, wie LiteLLM es innerhalb von `dspy.LM.forward` tut (`num_retries`).

    Jede Anfrage scheitert beim ersten Versuch und gelingt beim zweiten - ausser die zu
    "Frage 0", die bei jedem Versuch scheitert.
    """

    num_retries = 3

    def forward(self, prompt=None, messages=None, **kwargs):
        inhalt = (messages or [{"content": prompt}])[-1]["content"]
        for versuch in range(self.num_retries + 1):
            if versuch > 0 and "Frage 0" not in inhalt:
                return super().forward(prompt=prompt, messages=messages, **kwargs)
        raise RuntimeError("503 Service Unavailable")


def werte_aus(lm, beispiele, abschnitt):
    """Eine Nachbewertung mit dem Adapter, mit dem gemessen wird."""
    with dspy.context(lm=lm, adapter=mess_adapter()):
        with abschnitt(lm) as erfassung:
            ergebnis = dspy.Evaluate(
                devset=beispiele, metric=bewerte, num_threads=4, max_errors=len(beispiele) + 1
            )(MultiHopQA())
    return erfassung, [punkte for *_, punkte in ergebnis.results]


def test_ein_endpunkt_der_jede_anfrage_ablehnt_ist_ein_anbieterfehler(
    beispiele, abschnitt, protokoll
):
    """Sonst saehe ein gedrosseltes Modell aus wie ein schlechtes.

    Die Art des Fehlers steht einmal als Stoerung in der Zeile, nicht einmal je Aufruf: nicht
    jeder Fehler an dieser Grenze liegt beim Anbieter.
    """
    werte_aus(AbgelehnteAttrappe(), beispiele, abschnitt)

    (zeile,) = protokoll.zeilen()
    assert zeile["anbieterfehler"] == len(beispiele)
    assert zeile["formatfehler"] == 0
    assert zeile["stoerungen"].count("Anbieterfehler: RuntimeError") == 1


def test_eine_unlesbare_antwort_ist_ein_formatfehler_und_zaehlt_null(
    beispiele, abschnitt, protokoll
):
    """Eine Schwaeche des Modells: sie kostet Punkte, aber keinen Lauf."""
    _, einzelbewertungen = werte_aus(UnlesbareAttrappe(), beispiele, abschnitt)

    (zeile,) = protokoll.zeilen()
    assert zeile["formatfehler"] == len(beispiele)
    assert zeile["anbieterfehler"] == 0
    assert einzelbewertungen == [0.0] * len(beispiele)


def test_gezaehlt_wird_nur_was_nach_den_wiederholungen_uebrig_bleibt(beispiele, abschnitt):
    """Die Wiederholungen bleiben; ein Aussetzer, den sie auffangen, ist keiner.

    Eine Anfrage, die bei allen vier Versuchen scheitert, ist **ein** Anbieterfehler und
    nicht vier - gezaehlt wird der Aufruf, nicht der Versuch.
    """
    erfassung, einzelbewertungen = werte_aus(FlackerndeAttrappe(), beispiele, abschnitt)

    messwerte = erfassung.messwerte()
    assert messwerte.anbieterfehler == 1
    assert messwerte.formatfehler == 0
    assert einzelbewertungen == [0.0, 1.0, 1.0, 1.0]


# Abgeschnittene Antworten. Eine am `max_tokens`-Limit abgeschnittene Antwort zaehlt als
# Formatfehler und waere im Protokoll sonst nicht von einer Antwort im falschen Format zu
# unterscheiden.


class AbgeschnitteneAttrappe(BezahlteAttrappe):
    """Jede Antwort endet am `max_tokens`-Limit (`finish_reason == "length"`). Der Text bleibt
    lesbar: gezaehlt wird, was der Anbieter meldet, nicht, ob der Adapter scheitert."""

    def forward(self, prompt=None, messages=None, **kwargs):
        antwort = super().forward(prompt=prompt, messages=messages, **kwargs)
        for wahl in antwort.choices:
            wahl.finish_reason = "length"
        return antwort


@pytest.mark.parametrize(
    ("kandidatenmodell", "vorschlagsmodell", "erwartet"),
    [
        (AbgeschnitteneAttrappe, BezahlteAttrappe, {"kandidatenmodell": 8, "vorschlagsmodell": 0}),
        (BezahlteAttrappe, AbgeschnitteneAttrappe, {"kandidatenmodell": 0, "vorschlagsmodell": 2}),
        (BezahlteAttrappe, BezahlteAttrappe, {"kandidatenmodell": 0, "vorschlagsmodell": 0}),
    ],
    ids=["kandidatenmodell", "vorschlagsmodell", "keines"],
)
def test_abgeschnittene_antworten_stehen_je_modell_in_der_zeile(
    beispiele, abschnitt, protokoll, kandidatenmodell, vorschlagsmodell, erwartet
):
    """Getrennt nach Kandidaten- und Vorschlagsmodell: beim Kandidatenmodell kostet eine
    abgeschnittene Antwort einen Punkt, beim Vorschlagsmodell bricht sie den Optimierungslauf
    ab. Das Kandidatenmodell beantwortet vier Fragen mit je zwei Aufrufen, das
    Vorschlagsmodell eine mit zwei - wie in `test_mehrere_sprachmodelle_werden_zusammen_erfasst`.
    """
    kandidaten_lm, vorschlags_lm = kandidatenmodell(), vorschlagsmodell()

    with abschnitt(kandidaten_lm, vorschlags_lm, abschnitt="optimierung") as erfassung:
        with dspy.context(lm=kandidaten_lm):
            auswertung(beispiele)(MultiHopQA())
        with dspy.context(lm=vorschlags_lm):
            MultiHopQA()(**beispiele[0].inputs())

    (zeile,) = protokoll.zeilen()
    assert zeile["abgeschnittene_antworten"] == erwartet
    assert erfassung.messwerte().abgeschnittene_antworten == erwartet


def test_ohne_vorschlagsmodell_bleibt_dessen_zahl_leer(beispiele, abschnitt, protokoll):
    """In einer Bewertung laeuft nur das Kandidatenmodell. Null hiesse "gezaehlt, keine
    gefunden" - fuer ein Modell, das gar nicht lief, waere das eine erfundene Messung."""
    lm = AbgeschnitteneAttrappe()

    with dspy.context(lm=lm):
        with abschnitt(lm):
            auswertung(beispiele)(MultiHopQA())

    (zeile,) = protokoll.zeilen()
    assert zeile["abgeschnittene_antworten"] == {"kandidatenmodell": 8, "vorschlagsmodell": None}


def test_ein_zweiter_abschnitt_zaehlt_die_abgeschnittenen_des_ersten_nicht_mit(
    beispiele, abschnitt, protokoll
):
    """Dieselbe Grenze wie bei den Dollar: gezaehlt wird ab Beginn des Abschnitts."""
    lm = AbgeschnitteneAttrappe()

    with dspy.context(lm=lm):
        with abschnitt(lm):
            auswertung(beispiele)(MultiHopQA())
        with abschnitt(lm):
            auswertung(beispiele[:1])(MultiHopQA())

    erste, zweite = protokoll.zeilen()
    assert erste["abgeschnittene_antworten"]["kandidatenmodell"] == 8
    assert zweite["abgeschnittene_antworten"]["kandidatenmodell"] == 2


# --- Fixkosten ---------------------------------------------------------------------------

VOLL = {"metric_key": "eval_full"}
MINIBATCH = {"metric_key": "eval_minibatch"}


def test_die_fixkosten_sind_die_dollar_bis_zum_ende_der_ersten_vollstaendigen_bewertung(
    beispiele, abschnitt, protokoll
):
    """So beginnt jeder MIPROv2-Lauf: Beispiele sammeln und Instruktionen vorschlagen im
    Hauptthread, dann die erste vollstaendige Bewertung (`teleprompt/utils.py` Z. 50-51).
    Was danach kommt - Minibatches, weitere vollstaendige Bewertungen -, sind die Trials.
    Von Hand: zwei Aufrufe im Hauptthread und acht in der ersten Bewertung, also 10 * 0,001 $;
    der ganze Abschnitt kostet 10 + 4 + 8 = 22 Aufrufe."""
    lm = BezahlteAttrappe()

    with dspy.context(lm=lm):
        with abschnitt(lm, abschnitt="optimierung") as erfassung:
            MultiHopQA()(**beispiele[0].inputs())
            auswertung(beispiele)(MultiHopQA(), callback_metadata=VOLL)
            auswertung(beispiele[:2])(MultiHopQA(), callback_metadata=MINIBATCH)
            auswertung(beispiele)(MultiHopQA(), callback_metadata=VOLL)

    (zeile,) = protokoll.zeilen()
    assert zeile["fixkosten"] == pytest.approx(10 * DOLLAR_JE_AUFRUF)
    assert zeile["dollar"] == pytest.approx(22 * DOLLAR_JE_AUFRUF)


def test_bricht_die_erste_vollstaendige_bewertung_ab_bleiben_die_fixkosten_leer():
    """Ein Bruchteil der ersten Bewertung waere eine zu kleine Messung. Auch eine spaetere
    vollstaendige Bewertung springt nicht ein: sie bewertet schon einen Trial, nicht mehr das
    unveraenderte Programm. Leer heisst "nicht gemessen", nicht 0."""
    zaehler = Kostenzaehler()

    zaehler.on_evaluate_start("erste", None, {"callback_metadata": VOLL})
    zaehler.on_evaluate_end("erste", None, exception=TimeoutError("Anbieter antwortet nicht"))
    zaehler.on_evaluate_start("zweite", None, {"callback_metadata": VOLL})
    zaehler.on_evaluate_end("zweite", evaluationsergebnis([]))

    assert zaehler.messwerte().fixkosten is None


def test_eine_bewertung_des_fertigen_programms_hat_keine_fixkosten(beispiele, abschnitt, protokoll):
    """Die Nachbewertung traegt kein Kennzeichen von MIPROv2; dort gibt es keinen festen Teil."""
    lm = BezahlteAttrappe()

    with dspy.context(lm=lm):
        with abschnitt(lm):
            auswertung(beispiele)(MultiHopQA())

    (zeile,) = protokoll.zeilen()
    assert zeile["fixkosten"] is None
