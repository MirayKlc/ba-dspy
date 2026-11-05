"""Tests zur Naht zwischen Verfahren und MIPROv2.

Gelaufen wird ein **vollstaendiger** MIPROv2-Lauf - Beispiele sammeln, Instruktionen
vorschlagen, Trials bewerten -, nur eben gegen eine Attrappe statt gegen bezahlte
Endpunkte. Das ist Absicht: geprueft werden soll, was aus der Naht herauskommt, nicht in
welcher Reihenfolge sie intern Hilfsfunktionen ruft.

Die Attrappe liest aus dem Prompt, welche Ausgabefelder der Adapter erwartet, und fuellt
genau diese. Anders waere ein echter MIPROv2-Lauf nicht zu bedienen: er fragt nicht nur die
beiden Module des Programms, sondern auch den Proposer mit seinen sechs weiteren
Signaturen, und deren Feldnamen legt DSPy fest, nicht dieses Projekt.
"""

import contextlib
import re
import threading
import time

import dspy
import pytest
from dspy.dsp.utils.utils import dotdict
from dspy.utils.dummies import DummyLM

from dspy_optimizer import optimierungslauf as modul
from dspy_optimizer.kandidatenmodelle import (
    DAS_VORSCHLAGSMODELL,
    Kandidatenmodell,
    Vorschlagsmodell,
)
from dspy_optimizer.kostenerfassung import messhygiene
from dspy_optimizer.kostenmodell import Kostenmodell, Parameter
from dspy_optimizer.metrik import bewerte
from dspy_optimizer.optimierungslauf import Optimierungslauf
from dspy_optimizer.programm import MultiHopQA
from dspy_optimizer.protokoll import HANDLAUF, VORSTUDIE, Einordnung, Protokoll

# Was die Attrappe je Aufruf meldet. Zwei Aufrufe sind ein Programmlauf, einer je Modul.
PROMPT_TOKENS = 10
ANTWORT_TOKENS = 5
DOLLAR_JE_AUFRUF = 0.001

ABSAETZE = [{"title": "A", "sentences": ["The Eiffel Tower is in Paris."]}]

# Die Marken, mit denen der ChatAdapter seine Ausgabefelder ankuendigt.
MARKE = re.compile(r"\[\[ ## (\w+) ## \]\]")

# Womit die Attrappe ein Feld fuellt, wenn nichts anderes verlangt ist. `relevant_titles`
# ist eine Liste, weil die Signatur `list[str]` fordert; alles Uebrige ist Text.
FUELLWERTE = {"relevant_titles": ["A"], "answer": "France"}


class Attrappe(DummyLM):
    """Ein Sprachmodell, das jede Signatur bedient und dabei Tokens, Dollar und Anbieter meldet.

    `antworten` kann einzelne Felder festlegen; alles Uebrige kommt aus `FUELLWERTE`. Kopien
    schreiben in dieselbe History, wie `MitgezaehltesLM` es fuer die echten Sprachmodelle
    tut - genau darueber ruft MIPROv2s Proposer an.
    """

    def __init__(self, **antworten):
        super().__init__([])
        self.antworten = antworten
        # Die Gegenprobe zur Kostenerfassung: was die Attrappe selbst gezaehlt hat. Als
        # Liste, weil DSPys `copy()` flach kopiert - so zaehlen Kopien in denselben Topf,
        # ganz unabhaengig davon, ob die History geteilt wird. Acht Threads schreiben
        # darauf, deshalb ein Schloss.
        self._zaehler = [0]
        self._schloss = threading.Lock()

    @property
    def aufrufe(self):
        return self._zaehler[0]

    def copy(self, **kwargs):
        """Dasselbe wie `MitgezaehltesLM.copy` - `DummyLM` erbt nicht von `dspy.LM`."""
        kopie = super().copy(**kwargs)
        kopie.history = self.history
        return kopie

    def forward(self, prompt=None, messages=None, **kwargs):
        with self._schloss:
            self._zaehler[0] += 1
        inhalt = (messages or [{"content": prompt}])[-1]["content"]
        # Der Adapter zaehlt die erwarteten Felder im letzten Absatz auf; die Marken weiter
        # oben gehoeren zu Beispielen und zur Aufgabenbeschreibung.
        schluss = inhalt.rsplit("Respond with the corresponding output fields", 1)[-1]
        felder = [name for name in MARKE.findall(schluss) if name != "completed"]

        antwort = dotdict(
            choices=[
                dotdict(
                    message=dotdict(
                        content=self._format_answer_fields(
                            {feld: self._wert(feld) for feld in felder}
                        ),
                        tool_calls=None,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=dotdict(
                prompt_tokens=PROMPT_TOKENS,
                completion_tokens=ANTWORT_TOKENS,
                total_tokens=PROMPT_TOKENS + ANTWORT_TOKENS,
            ),
            model="attrappe",
        )
        antwort._hidden_params = {"response_cost": DOLLAR_JE_AUFRUF}
        antwort.provider = "attrappe"
        return antwort

    def _wert(self, feld):
        return self.antworten.get(feld, FUELLWERTE.get(feld, f"ein {feld}"))


class KaputteAttrappe(Attrappe):
    """Ein Endpunkt, der jede Anfrage ablehnt - auch nach allen Wiederholungen."""

    def forward(self, prompt=None, messages=None, **kwargs):
        raise RuntimeError("429 Too Many Requests")


def kandidatenmodell_antwortet_wie(monkeypatch, attrappenklasse):
    """Das Kandidatenmodell des Laufs wird eine `attrappenklasse`, gemessen mit dem Adapter
    des Experiments (ChatAdapter ohne JSON-Rueckfall). Das Vorschlagsmodell bleibt die
    gewoehnliche Attrappe."""

    @contextlib.contextmanager
    def kontext(kandidatenmodell):
        with dspy.context(
            lm=attrappenklasse(), adapter=dspy.ChatAdapter(use_json_adapter_fallback=False)
        ):
            yield dspy.settings.lm

    monkeypatch.setattr(modul, "messkontext", kontext)


def beispiel(nummer, antwort):
    return dspy.Example(
        id=f"beispiel-{nummer}",
        question=f"Question {nummer}?",
        answer=antwort,
        paragraphs=ABSAETZE,
        gold_titles=["A"],
    ).with_inputs("question", "paragraphs")


@pytest.fixture
def trainingsdaten():
    return [beispiel(nummer, "France") for nummer in range(6)]


@pytest.fixture
def validierungsdaten():
    """Sechs Beispiele, bei denen jedes zweite eine andere richtige Antwort hat.

    Die Attrappe antwortet immer "France". Die Einzelbewertungen sind damit abwechselnd 1
    und 0 - eine vertauschte Reihenfolge faellt auf, eine gleichfoermige Folge nicht.
    """
    return [
        beispiel(nummer, "France" if nummer % 2 == 0 else "Australia") for nummer in range(6)
    ]


@pytest.fixture
def auswahldaten():
    """Acht Beispiele, auf denen die Nachbewertung misst: zwei richtig, zwei
    falsch, zweimal. Mehr als Validierungsdaten und anders gemischt - so faellt auf, wenn die
    Nachbewertung doch ueber die Validierungsdaten liefe."""
    return [
        beispiel(300 + nummer, "France" if nummer % 4 < 2 else "Australia")
        for nummer in range(8)
    ]


@pytest.fixture
def protokoll(tmp_path):
    """Das Protokoll des Tests - nie das des Repos."""
    return Protokoll(tmp_path / "protokoll.jsonl")


@pytest.fixture(autouse=True)
def saubere_messhygiene():
    messhygiene()


@pytest.fixture(autouse=True)
def laeufe_im_tmp(tmp_path, monkeypatch):
    """Die `log_dir`-Verzeichnisse der Tests landen nicht in `ergebnisse/laeufe`."""

    def anlegen(name):
        pfad = tmp_path / re.sub(r"[^A-Za-z0-9._-]+", "-", name)
        pfad.mkdir(parents=True, exist_ok=True)
        return pfad

    monkeypatch.setattr(modul, "verzeichnis_fuer_lauf", anlegen)


@pytest.fixture
def attrappen(monkeypatch):
    """Statt bezahlter Endpunkte Attrappen - eine je Rolle, und sie werden mitgezaehlt.

    Getrennte Objekte, damit sich pruefen laesst, dass das Vorschlagsmodell ein anderes ist
    als das Kandidatenmodell und ueber alle Laeufe dasselbe bleibt.
    """
    gebaut = {"kandidatenmodelle": [], "vorschlagsmodelle": []}

    @contextlib.contextmanager
    def kontext(kandidatenmodell):
        lm = Attrappe()
        gebaut["kandidatenmodelle"].append(lm)
        with dspy.context(lm=lm, adapter=dspy.ChatAdapter(use_json_adapter_fallback=False)):
            yield lm

    def vorschlag(vorschlagsmodell):
        lm = Attrappe()
        gebaut["vorschlagsmodelle"].append(lm)
        return lm

    monkeypatch.setattr(modul, "messkontext", kontext)
    monkeypatch.setattr(modul, "lm_fuer", vorschlag)
    return gebaut


@pytest.fixture
def lauf(trainingsdaten, validierungsdaten, auswahldaten, protokoll, attrappen):
    """Die Naht, klein genug fuer einen Test: zwei Kandidaten, Minibatch von drei."""
    return Optimierungslauf(
        trainingsdaten=trainingsdaten,
        validierungsdaten=validierungsdaten,
        auswahldaten=auswahldaten,
        num_candidates=2,
        minibatch_size=3,
        protokoll=protokoll,
    )


KANDIDATENMODELL = Kandidatenmodell("haus/modell-a", "anbieter", 0.1, 0.2, {"enabled": False})
ANDERES_KANDIDATENMODELL = Kandidatenmodell("haus/modell-b", "anbieter", 0.3, 0.4, {"enabled": False})

# Wozu die Laeufe der Tests gehoeren. Welche es ist, spielt fuer die meisten Tests keine
# Rolle - nur fehlen darf sie nicht.
EXPERIMENT = Einordnung.experiment("verfahren", durchgang=1, phase="sondierung")


def test_die_naht_gibt_alle_sechs_rueckgaben(lauf, auswahldaten):
    """Das Kernversprechen der Schnittstelle.

    Dass der Lauf ueberhaupt durchlaeuft, belegt zugleich zwei weitere Zusagen: MIPROv2
    laesst `auto` und `num_trials` nur entweder-oder zu und wirft sonst einen `ValueError`
    (`mipro_optimizer_v2.py` Z. 150-167). Ein Lauf, der zurueckkommt, hatte also `auto=None`
    und ein ausdrueckliches `num_trials`.
    """
    ergebnis = lauf(
        KANDIDATENMODELL, MultiHopQA(), num_trials=2, seed=0, einordnung=EXPERIMENT
    )

    assert isinstance(ergebnis.programm, dspy.Module)
    assert 0.0 <= ergebnis.bewertung <= 1.0
    assert len(ergebnis.bewertungen_je_beispiel) == len(auswahldaten)
    assert ergebnis.optimierungskosten > 0
    assert ergebnis.ausfuehrungskosten > 0
    assert ergebnis.programmlaeufe > 0


def test_die_bewertungen_kommen_je_auswahlbeispiel_in_stabiler_reihenfolge(lauf):
    """Der gepaarte Test der Zwischenbewertung haengt an dieser Reihenfolge.

    Ausgewertet wird mit acht Threads; die Reihenfolge darf davon nicht abhaengen. Die
    Auswahldaten wechseln paarweise zwischen richtig und falsch - waere die Reihenfolge
    die der fertig gewordenen Threads, stuende hier eine andere Folge.
    """
    ergebnis = lauf(
        KANDIDATENMODELL, MultiHopQA(), num_trials=2, seed=0, einordnung=EXPERIMENT
    )

    assert ergebnis.bewertungen_je_beispiel == (1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0)


def test_ohne_auswahldaten_laesst_sich_die_naht_nicht_bauen(trainingsdaten, validierungsdaten):
    """Eine Nachbewertung ueber null Beispiele hinterliesse eine Zeile mit Bewertung 0,0: eine
    erfundene Messung statt einer fehlenden."""
    with pytest.raises(ValueError, match="Auswahldaten"):
        Optimierungslauf(
            trainingsdaten=trainingsdaten, validierungsdaten=validierungsdaten, auswahldaten=[]
        )


def gestellt(lm, beispiel):
    """In wie vielen Aufrufen an `lm` die Frage dieses Beispiels vorkam."""
    return sum(beispiel.question in str(eintrag["messages"]) for eintrag in lm.history)


def test_mipro_waehlt_auf_den_validierungsdaten_die_nachbewertung_misst_auf_den_auswahldaten(
    lauf, attrappen, validierungsdaten, auswahldaten
):
    """Die Nachbewertung misst auf eigenen Auswahldaten, also auf Fragen, auf denen MIPROv2
    nichts ausgewaehlt hat. Jede Frage der Auswahldaten wird genau zweimal gestellt - je
    Modul einmal, in der Nachbewertung -, MIPROv2 hat sie also nie gesehen. Jede Frage der
    Validierungsdaten hat MIPROv2 dagegen benutzt."""
    lauf(KANDIDATENMODELL, MultiHopQA(), num_trials=2, seed=0, einordnung=EXPERIMENT)

    kandidaten_lm = attrappen["kandidatenmodelle"][0]
    assert [gestellt(kandidaten_lm, beispiel) for beispiel in auswahldaten] == [2] * len(
        auswahldaten
    )
    assert all(gestellt(kandidaten_lm, beispiel) > 0 for beispiel in validierungsdaten)


def test_die_bewertung_ist_der_mittelwert_der_einzelbewertungen(lauf):
    """Beide Zahlen kommen aus derselben Auswertung; sie duerfen nicht auseinanderlaufen."""
    ergebnis = lauf(
        KANDIDATENMODELL, MultiHopQA(), num_trials=2, seed=0, einordnung=EXPERIMENT
    )

    assert ergebnis.bewertung == pytest.approx(0.5)


def test_ein_praediktor_mit_eigenem_lm_bricht_ab_bevor_geld_fliesst(lauf, attrappen):
    """Ein eigenes LM sticht Kontext und Settings (`predict.py` Z. 150).

    Der Lauf liefe still auf dem falschen Modell, und die Messung waere wertlos, ohne dass
    irgendwo ein Fehler auftauchte. Geprueft wird deshalb vor dem ersten Aufruf: hinterher ist
    das Geld ausgegeben.
    """
    programm = MultiHopQA()
    programm.answer_question.set_lm(Attrappe())

    with pytest.raises(ValueError, match="answer_question"):
        lauf(KANDIDATENMODELL, programm, num_trials=2, seed=0, einordnung=EXPERIMENT)

    assert attrappen["kandidatenmodelle"] == []


def test_das_vorschlagsmodell_ist_ueber_alle_laeufe_dasselbe_und_nicht_das_kandidatenmodell(
    lauf, attrappen, protokoll
):
    """Sonst variierten zwei Dinge zugleich."""
    lauf(KANDIDATENMODELL, MultiHopQA(), num_trials=2, seed=0, einordnung=EXPERIMENT)
    lauf(
        ANDERES_KANDIDATENMODELL,
        MultiHopQA(),
        num_trials=2,
        seed=0,
        einordnung=EXPERIMENT,
    )

    # Ein einziges Vorschlagsmodell, gebaut beim Bau der Naht, fuer beide Laeufe.
    assert len(attrappen["vorschlagsmodelle"]) == 1
    assert len(attrappen["kandidatenmodelle"]) == 2
    assert attrappen["vorschlagsmodelle"][0] not in attrappen["kandidatenmodelle"]

    namen = {zeile["konfiguration"]["vorschlagsmodell"] for zeile in protokoll.zeilen()}
    assert namen == {DAS_VORSCHLAGSMODELL.name}


def test_der_lauf_hinterlaesst_je_eine_zeile_fuer_optimierung_und_nachbewertung(
    lauf, protokoll
):
    """Optimierungs- und Ausfuehrungskosten werden nie miteinander verrechnet."""
    lauf(KANDIDATENMODELL, MultiHopQA(), num_trials=2, seed=0, einordnung=EXPERIMENT)

    zeilen = protokoll.zeilen()
    assert [zeile["abschnitt"] for zeile in zeilen] == ["optimierung", "nachbewertung"]
    assert all(zeile["konfiguration"]["kandidatenmodell"] == "haus/modell-a" for zeile in zeilen)


def test_beide_zeilen_sagen_wozu_der_lauf_gehoert(lauf, protokoll):
    """Die Auswertung liest nur das Protokoll: ohne Baseline, Phase und Durchgang
    liessen sich weder die Phasen vergleichen noch die Kosten eines Durchgangs summieren."""
    einordnung = Einordnung.experiment("verfahren", durchgang=3, phase="fokus")

    lauf(KANDIDATENMODELL, MultiHopQA(), num_trials=2, seed=0, einordnung=einordnung)

    for zeile in protokoll.zeilen():
        assert zeile["zweck"] == "experiment"
        assert zeile["baseline"] == "verfahren"
        assert zeile["phase"] == "fokus"
        assert zeile["durchgang"] == 3


def test_ein_handlauf_ueber_die_naht_ist_nicht_kalibrierbar(lauf, protokoll):
    """So ist am 20.09.2026 eine `optimierung`-Zeile entstanden, die mitkalibriert worden
    waere. Mit dem Zweck Handlauf faellt sie aus der Kalibrierung heraus."""
    lauf(KANDIDATENMODELL, MultiHopQA(), num_trials=2, seed=0, einordnung=HANDLAUF)

    optimierung = protokoll.zeilen()[0]
    assert optimierung["zweck"] == "handlauf"
    assert optimierung["bewertung"] is not None
    assert protokoll.kalibrierbare_zeilen() == []


@pytest.mark.parametrize("einordnung", [None, "experiment"])
def test_ohne_gueltige_einordnung_bricht_der_lauf_ab_bevor_geld_fliesst(
    lauf, attrappen, protokoll, einordnung
):
    with pytest.raises(ValueError, match="Einordnung"):
        lauf(KANDIDATENMODELL, MultiHopQA(), num_trials=2, seed=0, einordnung=einordnung)

    assert attrappen["kandidatenmodelle"] == []
    assert protokoll.zeilen() == []


def test_ohne_einordnung_laesst_sich_der_lauf_gar_nicht_rufen(lauf, attrappen):
    """Eine vergessene Angabe ist kein stiller Standardwert."""
    with pytest.raises(TypeError, match="einordnung"):
        lauf(KANDIDATENMODELL, MultiHopQA(), num_trials=2, seed=0)

    assert attrappen["kandidatenmodelle"] == []


def test_die_ausfuehrungskosten_sind_die_kosten_eines_laufs_des_fertigen_programms(
    lauf, auswahldaten, protokoll
):
    """Ein Programmlauf sind zwei LM-Aufrufe - einer je Modul."""
    ergebnis = lauf(
        KANDIDATENMODELL, MultiHopQA(), num_trials=2, seed=0, einordnung=EXPERIMENT
    )

    assert ergebnis.ausfuehrungskosten == pytest.approx(2 * DOLLAR_JE_AUFRUF)

    nachbewertung = protokoll.zeilen()[-1]
    assert nachbewertung["programmlaeufe"] == len(auswahldaten)
    assert nachbewertung["dollar"] == pytest.approx(len(auswahldaten) * 2 * DOLLAR_JE_AUFRUF)


def test_kein_bezahlter_aufruf_faellt_unter_den_tisch(lauf, attrappen, protokoll):
    """Die Gegenprobe: die Attrappe zaehlt selbst mit, und beide Zahlen muessen aufgehen.

    Der heikle Teil sind die Instruktionsvorschlaege. MIPROv2 ruft den Proposer ueber eine
    **Kopie** des Vorschlagsmodells an, um den Cache zu umgehen (`grounded_proposer.py`
    Z. 395-397), und DSPys `copy()` gibt der Kopie eine frische History
    (`base_lm.py` Z. 755-757). Ohne die geteilte History stuende das Vorschlagen - ein Teil
    der Fixkosten - mit 0,00 $ im Protokoll, und niemand saehe es.
    """
    lauf(KANDIDATENMODELL, MultiHopQA(), num_trials=2, seed=0, einordnung=EXPERIMENT)

    vorschlags_lm = attrappen["vorschlagsmodelle"][0]
    kandidaten_lm = attrappen["kandidatenmodelle"][0]
    assert vorschlags_lm.aufrufe > 0

    gezaehlt = sum(zeile["lm_aufrufe"] for zeile in protokoll.zeilen())
    assert gezaehlt == kandidaten_lm.aufrufe + vorschlags_lm.aufrufe


def test_die_optimierungskosten_sind_die_dollar_beider_beteiligten_modelle(lauf, protokoll):
    """Kandidatenmodell und Vorschlagsmodell zahlen beide auf die Optimierungskosten ein."""
    ergebnis = lauf(
        KANDIDATENMODELL, MultiHopQA(), num_trials=2, seed=0, einordnung=EXPERIMENT
    )

    optimierung = protokoll.zeilen()[0]
    assert ergebnis.optimierungskosten == pytest.approx(
        optimierung["lm_aufrufe"] * DOLLAR_JE_AUFRUF
    )


def test_die_naht_gibt_die_tatsaechlichen_kosten_der_nachbewertung_zurueck(lauf, protokoll):
    """Das Restbudget ist das Optimierungsbudget minus die tatsaechlich ausgegebenen Dollar beider
    Messlaeufe. Dafuer muss das Verfahren auch die Nachbewertung in Dollar kennen, nicht nur
    je Programmlauf."""
    ergebnis = lauf(
        KANDIDATENMODELL, MultiHopQA(), num_trials=2, seed=0, einordnung=EXPERIMENT
    )

    optimierung, nachbewertung = protokoll.zeilen()
    assert ergebnis.optimierungskosten == optimierung["dollar"]
    assert ergebnis.nachbewertungskosten == nachbewertung["dollar"]
    assert ergebnis.nachbewertungskosten > 0


def test_gleicher_seed_und_gleiche_eingaben_geben_dasselbe_ergebnis(
    trainingsdaten, validierungsdaten, auswahldaten, attrappen, tmp_path
):
    """Ein Lauf muss wiederholbar sein."""

    def einmal():
        lauf = Optimierungslauf(
            trainingsdaten=trainingsdaten,
            validierungsdaten=validierungsdaten,
            auswahldaten=auswahldaten,
            num_candidates=2,
            minibatch_size=3,
            protokoll=Protokoll(tmp_path / "wegwerf.jsonl"),
        )
        return lauf(
            KANDIDATENMODELL, MultiHopQA(), num_trials=2, seed=7, einordnung=EXPERIMENT
        )

    erster, zweiter = einmal(), einmal()

    assert erster.bewertungen_je_beispiel == zweiter.bewertungen_je_beispiel
    assert erster.optimierungskosten == pytest.approx(zweiter.optimierungskosten)
    assert erster.programmlaeufe == zweiter.programmlaeufe


def test_das_ergebnisprogramm_traegt_kein_eigenes_lm(lauf):
    """Sonst scheiterte der Warm-Start in die Fokusphase an der eigenen Pruefung.

    Das Gewinnermodell wird erst in der Endauswahl eingebrannt.
    """
    ergebnis = lauf(
        KANDIDATENMODELL, MultiHopQA(), num_trials=2, seed=0, einordnung=EXPERIMENT
    )

    assert all(praediktor.lm is None for _, praediktor in ergebnis.programm.named_predictors())


def test_das_startprogramm_bleibt_unveraendert(lauf):
    """Die Fokusphase reicht ein Programm weiter; der Aufrufer behaelt seines."""
    startprogramm = MultiHopQA()
    vorher = startprogramm.find_evidence.signature.instructions

    ergebnis = lauf(KANDIDATENMODELL, startprogramm, num_trials=2, seed=0, einordnung=EXPERIMENT)

    assert startprogramm.find_evidence.signature.instructions == vorher
    assert ergebnis.programm is not startprogramm


def test_das_vorschlagsmodell_ist_kein_kandidatenmodell():
    """Es konkurriert um kein Budget und kann nie Gewinner werden."""
    assert isinstance(DAS_VORSCHLAGSMODELL, Vorschlagsmodell)
    assert not isinstance(DAS_VORSCHLAGSMODELL, Kandidatenmodell)


def plan_fuer(kandidatenmodell, auswahlbeispiele, num_trials=2):
    """Ein Plan: 1,00 $ Fixkosten, 0,03 $ je Trial, 0,002 $ je Programmlauf.

    Der Budgetanteil ist so gewaehlt, dass nach der Nachbewertung genau `num_trials` bleiben.
    """
    kostenmodell = Kostenmodell(
        kandidatenmodell=kandidatenmodell.name,
        fixkosten=Parameter(1.0, "erfunden"),
        kosten_je_trial=Parameter(0.03, "erfunden"),
        ausfuehrungskosten=Parameter(2 * DOLLAR_JE_AUFRUF, "erfunden"),
    )
    return kostenmodell.plane(
        budgetanteil=kostenmodell.dollar_fuer(num_trials, auswahlbeispiele),
        auswahlbeispiele=auswahlbeispiele,
    )


@pytest.mark.parametrize(
    "falsch, meldung",
    [
        ({"num_trials": 3}, "Trials"),
        ({"kandidatenmodell": ANDERES_KANDIDATENMODELL}, "haus/modell-b"),
        ({"auswahlbeispiele": 200}, "Auswahlbeispiele"),
    ],
)
def test_ein_plan_fuer_einen_anderen_lauf_bricht_ab_bevor_geld_fliesst(
    lauf, attrappen, protokoll, auswahldaten, falsch, meldung
):
    """Sonst plante man fuer ein Kandidatenmodell und bezahlte den Lauf eines anderen, und
    die Protokollzeile rechnete die Abweichung gegen einen Plan fuer einen anderen Lauf -
    lautlos, waehrend die Planabweichung der Grund ist, warum der Plan mitgereicht wird.

    Geplant wird hier mit 200 Auswahlbeispielen oder fuer ein anderes Kandidatenmodell,
    gelaufen wird mit acht und `haus/modell-a`. Die Nachbewertung misst auf den
    Auswahldaten, also zaehlt deren Zahl."""
    geplant = {
        "num_trials": 2,
        "kandidatenmodell": KANDIDATENMODELL,
        "auswahlbeispiele": len(auswahldaten),
        **falsch,
    }
    plan = plan_fuer(**geplant)

    with pytest.raises(ValueError, match=meldung):
        lauf(
            KANDIDATENMODELL,
            MultiHopQA(),
            num_trials=2,
            seed=0,
            einordnung=EXPERIMENT,
            plan=plan,
        )

    assert attrappen["kandidatenmodelle"] == []
    assert protokoll.zeilen() == []


def test_beide_zeilen_rechnen_gegen_ihren_teil_des_plans(lauf, protokoll, auswahldaten):
    """Plan gegen Ist gilt fuer beide Messlaeufe: die Nachbewertung wird aus dem
    Budgetanteil bezahlt, also wird auch sie geplant und gegen ihren Plan gehalten.

    Die Attrappe kostet je Programmlauf genau das, was der Plan annimmt - die
    Nachbewertung trifft ihren Plan deshalb auf den Cent.
    """
    plan = plan_fuer(KANDIDATENMODELL, len(auswahldaten))

    lauf(
        KANDIDATENMODELL,
        MultiHopQA(),
        num_trials=2,
        seed=0,
        einordnung=EXPERIMENT,
        plan=plan,
    )

    optimierung, nachbewertung = protokoll.zeilen()
    assert optimierung["dollar_geplant"] == pytest.approx(1.06)  # 1,00 + 2 * 0,03
    assert optimierung["dollar_abweichung"] == pytest.approx(optimierung["dollar"] - 1.06)
    assert nachbewertung["dollar_geplant"] == pytest.approx(0.016)  # 8 * 0,002
    assert nachbewertung["dollar_abweichung"] == pytest.approx(0.0)
    for zeile in (optimierung, nachbewertung):
        assert zeile["dollar_budgetanteil"] == pytest.approx(1.076)
        assert zeile["num_trials_geplant"] == 2
        assert zeile["kostenmodell"]["kandidatenmodell"] == "haus/modell-a"


def test_der_lauf_uebersteht_eine_bewertung_in_der_jedes_beispiel_scheitert(
    trainingsdaten, validierungsdaten, protokoll, monkeypatch
):
    """Ein Kandidatenmodell, dessen Endpunkt jede Anfrage ablehnt, darf den Lauf nicht
    abbrechen, nachdem die Bewertung bezahlt ist.

    DSPy bricht bei `error_count >= max_errors` ab (`utils/parallelizer.py` Z. 66) - genau
    bei `len(validierungsdaten)` waere also schon Schluss. Geprueft wird die Grenze an der
    abschliessenden Bewertung: sie laeuft mit derselben Zahl.
    """
    kandidatenmodell_antwortet_wie(monkeypatch, KaputteAttrappe)
    monkeypatch.setattr(modul, "lm_fuer", lambda modell: Attrappe())
    auswertung = dspy.Evaluate(
        devset=validierungsdaten,
        metric=bewerte,
        num_threads=modul.NUM_THREADS,
        max_errors=Optimierungslauf(
            trainingsdaten=trainingsdaten,
            validierungsdaten=validierungsdaten,
            auswahldaten=validierungsdaten,
        )._max_errors(validierungsdaten),
        display_progress=False,
    )

    with dspy.context(lm=KaputteAttrappe(), adapter=dspy.ChatAdapter(
        use_json_adapter_fallback=False
    )):
        ergebnis = auswertung(MultiHopQA())

    # Kein Abbruch, sondern eine Bewertung von 0,0 je Beispiel - laut und nachlesbar.
    assert [punkte for *_, punkte in ergebnis.results] == [0.0] * len(validierungsdaten)


def test_die_optimierungszeile_traegt_keine_dollar_je_programmlauf(lauf, protokoll):
    """Dort sind es weder Ausfuehrungskosten noch sonst etwas Deutbares.

    Im ersten echten Lauf lagen die beiden Zahlen nur 10 % auseinander (0,000301 gegen
    0,000334) - ein Verwechseln waere beim Auswerten nicht aufgefallen.
    """
    lauf(KANDIDATENMODELL, MultiHopQA(), num_trials=2, seed=0, einordnung=EXPERIMENT)

    optimierung, nachbewertung = protokoll.zeilen()
    assert optimierung["dollar_je_programmlauf"] is None
    assert nachbewertung["dollar_je_programmlauf"] == pytest.approx(2 * DOLLAR_JE_AUFRUF)


def test_die_nachbewertung_traegt_exact_match_und_absatztreffer(lauf, protokoll):
    """Je Lauf drei Zahlen (F1, Exact Match, Absatztreffer): sie entstehen nur hier, bei der
    eigenen Bewertung.

    Die Attrappe antwortet immer "France" und nennt immer Absatz "A": Exact Match trifft
    also jedes zweite Beispiel, die Absatztreffer jedes. Die Optimierungszeile kennt nur
    MIPROv2s F1 und laesst beide Spalten leer.
    """
    lauf(KANDIDATENMODELL, MultiHopQA(), num_trials=2, seed=0, einordnung=EXPERIMENT)

    optimierung, nachbewertung = protokoll.zeilen()
    assert nachbewertung["exakt"] == pytest.approx(0.5)
    assert nachbewertung["absatztreffer"] == pytest.approx(1.0)
    assert optimierung["exakt"] is None
    assert optimierung["absatztreffer"] is None


def test_die_optimierungszeile_traegt_eine_bewertung_wenn_der_lauf_zu_ende_kam(lauf, protokoll):
    """Daran unterscheidet die Kalibrierung abgebrochene von gelungenen Laeufen.

    MIPROv2 haengt seine Bewertung nur bei `track_stats` ans Ergebnisprogramm
    (`mipro_optimizer_v2.py` Z. 675). Faellt das weg, waere jede Bewertung leer - und die
    Kalibrierung faende still keine einzige brauchbare Zeile mehr.
    """
    lauf(KANDIDATENMODELL, MultiHopQA(), num_trials=2, seed=0, einordnung=EXPERIMENT)

    optimierung = protokoll.zeilen()[0]
    assert optimierung["bewertung"] is not None
    assert 0.0 <= optimierung["bewertung"] <= 1.0


def test_ein_abgebrochener_lauf_hinterlaesst_eine_zeile_ohne_bewertung(lauf, protokoll):
    """Ausgegeben ist das Geld trotzdem - aber ein Abbruch ist kein Messpunkt.

    Genau diese Zeile darf die Kalibrierung nicht mitnehmen: sie zoege die Gerade nach
    unten, die Kosten je Trial kaemen zu niedrig heraus, und jeder daraus geplante Lauf waere
    unterfinanziert (die Fixkosten stammen aus eigenen Fixkosten-Messungen).
    """

    class AttrappeDieAufgibt(Attrappe):
        """Antwortet, bis der Proposer dran ist, und bricht dann ab."""

        def forward(self, prompt=None, messages=None, **kwargs):
            inhalt = (messages or [{"content": prompt}])[-1]["content"]
            if "proposed_instruction" in inhalt:
                raise RuntimeError("Der Endpunkt hat die Anfrage abgelehnt.")
            return super().forward(prompt=prompt, messages=messages, **kwargs)

    lauf.vorschlags_lm = AttrappeDieAufgibt()

    with pytest.raises(Exception):  # noqa: B017 - *dass* es durchschlaegt, ist der Punkt
        lauf(KANDIDATENMODELL, MultiHopQA(), num_trials=2, seed=0, einordnung=EXPERIMENT)

    abbruch = protokoll.zeilen()[-1]
    assert abbruch["abschnitt"] == "optimierung"
    assert abbruch["bewertung"] is None
    assert abbruch["dollar"] > 0
    assert abbruch["stoerungen"]


def test_die_nachbewertung_traegt_die_einzelbewertungen_in_der_reihenfolge_der_rueckgabe(
    lauf, protokoll
):
    """Die Auswertung liest nur das Protokoll; ohne sie kein gepaarter Test."""
    ergebnis = lauf(
        KANDIDATENMODELL, MultiHopQA(), num_trials=2, seed=0, einordnung=EXPERIMENT
    )

    optimierung, nachbewertung = protokoll.zeilen()
    assert nachbewertung["einzelbewertungen"] == list(ergebnis.bewertungen_je_beispiel)
    assert nachbewertung["einzelbewertungen"] == [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    # MIPROv2 gibt keine Einzelwerte heraus; die Optimierungszeile hat keine.
    assert optimierung["einzelbewertungen"] is None


# Wie lange die wartende Attrappe je LM-Aufruf braucht. Zwei Aufrufe sind ein Programmlauf.
WARTEZEIT = 0.02


class WartendeAttrappe(Attrappe):
    """Die Attrappe mit fester Wartezeit je Aufruf - eine Latenz ohne API-Aufruf."""

    def forward(self, prompt=None, messages=None, **kwargs):
        time.sleep(WARTEZEIT)
        return super().forward(prompt=prompt, messages=messages, **kwargs)


def test_beide_zeilen_tragen_ihre_dauer_und_die_nachbewertung_die_latenz(
    lauf, protokoll, monkeypatch
):
    """Die Latenz ist die Dauer eines Programmlaufs, nicht eines LM-Aufrufs.

    Sie steht nur bei der Nachbewertung: in der Optimierung laeuft kein fertiges Programm.
    """
    kandidatenmodell_antwortet_wie(monkeypatch, WartendeAttrappe)

    lauf(KANDIDATENMODELL, MultiHopQA(), num_trials=2, seed=0, einordnung=EXPERIMENT)

    optimierung, nachbewertung = protokoll.zeilen()
    assert optimierung["dauer_sekunden"] > 0
    assert optimierung["latenz_mittelwert_sekunden"] is None
    assert optimierung["latenz_median_sekunden"] is None
    for feld in ("latenz_mittelwert_sekunden", "latenz_median_sekunden"):
        assert 2 * WARTEZEIT <= nachbewertung[feld] < 2 * WARTEZEIT + 0.5
    assert nachbewertung["dauer_sekunden"] >= nachbewertung["latenz_median_sekunden"]


def test_die_rueckgabe_der_naht_kennt_weder_dauer_noch_latenz(lauf):
    """Dauer und Latenz gehen nur ins Protokoll; das Verfahren sieht sie nicht.

    Die beiden Fehlerzahlen dagegen braucht es: auf ihnen entscheidet es, ob ein Durchgang
    gilt. Und das Ergebnisprogramm ist das Programm selbst, nicht die Stoppuhr
    darum.
    """
    ergebnis = lauf(
        KANDIDATENMODELL, MultiHopQA(), num_trials=2, seed=0, einordnung=EXPERIMENT
    )

    assert modul.Optimierungsergebnis._fields == (
        "programm",
        "bewertung",
        "bewertungen_je_beispiel",
        "optimierungskosten",
        "nachbewertungskosten",
        "ausfuehrungskosten",
        "programmlaeufe",
        "formatfehler",
        "anbieterfehler",
    )
    assert isinstance(ergebnis.programm, MultiHopQA)


def test_ein_lauf_ohne_stoerung_meldet_weder_formatfehler_noch_anbieterfehler(lauf, protokoll):
    ergebnis = lauf(
        KANDIDATENMODELL, MultiHopQA(), num_trials=2, seed=0, einordnung=EXPERIMENT
    )

    assert (ergebnis.formatfehler, ergebnis.anbieterfehler) == (0, 0)
    for zeile in protokoll.zeilen():
        assert (zeile["formatfehler"], zeile["anbieterfehler"]) == (0, 0)


class AbgeschnitteneAttrappe(Attrappe):
    """Jede Antwort endet am `max_tokens`-Limit, bleibt aber lesbar."""

    def forward(self, prompt=None, messages=None, **kwargs):
        antwort = super().forward(prompt=prompt, messages=messages, **kwargs)
        antwort.choices[0].finish_reason = "length"
        return antwort


def test_abgeschnittene_antworten_des_vorschlagsmodells_landen_nicht_beim_kandidatenmodell(
    lauf, protokoll
):
    """Die Naht reicht beide Sprachmodelle in fester Reihenfolge an die Kostenerfassung, und
    an dieser Reihenfolge haengt die Trennung der beiden Zaehler. Die
    Nachbewertung kennt kein Vorschlagsmodell; dort bleibt dessen Zahl leer."""
    vorschlags_lm = AbgeschnitteneAttrappe()
    lauf.vorschlags_lm = vorschlags_lm

    lauf(KANDIDATENMODELL, MultiHopQA(), num_trials=2, seed=0, einordnung=EXPERIMENT)

    optimierung, nachbewertung = protokoll.zeilen()
    assert vorschlags_lm.aufrufe > 0
    assert optimierung["abgeschnittene_antworten"] == {
        "kandidatenmodell": 0,
        "vorschlagsmodell": vorschlags_lm.aufrufe,
    }
    assert nachbewertung["abgeschnittene_antworten"] == {
        "kandidatenmodell": 0,
        "vorschlagsmodell": None,
    }


class StotterndeAttrappe(Attrappe):
    """Antwortet auf "Question 2?" (Validierungsdaten) und "Question 300?" (Auswahldaten) im
    Antwortmodul in einer Form, die der Adapter nicht liest.

    Nur im Antwortmodul, damit der Proposer unberuehrt bleibt; und nur am Ende des Prompts,
    wo die aktuelle Frage steht - weiter oben koennte sie als Few-Shot-Beispiel stehen.
    """

    def forward(self, prompt=None, messages=None, **kwargs):
        antwort = super().forward(prompt=prompt, messages=messages, **kwargs)
        inhalt = (messages or [{"content": prompt}])[-1]["content"]
        stottert = "Question 2?" in inhalt or "Question 300?" in inhalt
        if stottert and "[[ ## answer ## ]]" in antwort.choices[0].message.content:
            antwort.choices[0].message.content = "Frankreich, glaube ich."
        return antwort


def test_die_naht_meldet_formatfehler_ueber_beide_messlaeufe(lauf, protokoll, monkeypatch):
    """Ein Formatfehler zaehlt als 0,0 - und wird trotzdem gesondert gemeldet.

    Beide Fragen haetten die richtige Antwort "France" und stuenden sonst bei 1,0. Die
    Optimierung bewertet "Question 2?" mindestens einmal, bei der ersten vollstaendigen
    Bewertung der Validierungsdaten; die Nachbewertung "Question 300?", das erste
    Auswahlbeispiel. Die Rueckgabe nennt die Summe beider Messlaeufe.
    """
    kandidatenmodell_antwortet_wie(monkeypatch, StotterndeAttrappe)

    ergebnis = lauf(
        KANDIDATENMODELL, MultiHopQA(), num_trials=2, seed=0, einordnung=EXPERIMENT
    )

    optimierung, nachbewertung = protokoll.zeilen()
    assert ergebnis.bewertungen_je_beispiel == (0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
    assert nachbewertung["formatfehler"] == 1
    assert optimierung["formatfehler"] >= 1
    assert ergebnis.formatfehler == optimierung["formatfehler"] + nachbewertung["formatfehler"]
    assert ergebnis.anbieterfehler == 0


def test_ein_endpunkt_der_jede_anfrage_ablehnt_erscheint_als_anbieterfehler(
    lauf, protokoll, auswahldaten, monkeypatch
):
    """Nicht als Formatfehler: sonst saehe ein gedrosseltes Modell aus wie ein schlechtes
    und floege im gepaarten Test heraus."""
    kandidatenmodell_antwortet_wie(monkeypatch, KaputteAttrappe)

    ergebnis = lauf(
        KANDIDATENMODELL, MultiHopQA(), num_trials=2, seed=0, einordnung=EXPERIMENT
    )

    optimierung, nachbewertung = protokoll.zeilen()
    assert nachbewertung["anbieterfehler"] == len(auswahldaten)
    assert optimierung["anbieterfehler"] > 0
    assert ergebnis.anbieterfehler == optimierung["anbieterfehler"] + len(auswahldaten)
    assert ergebnis.formatfehler == 0


def test_das_verfahren_sondiert_durch_die_echte_naht(lauf, protokoll):
    """Die Gegenprobe zu `tests/test_verfahren.py`: dort laeuft das Verfahren gegen eine
    Attrappe der Naht. Hier laeuft es durch die echte, nur mit Attrappen statt bezahlter
    Endpunkte. So faellt auf, wenn Attrappe und Naht auseinanderlaufen - etwa wenn der Plan
    des Verfahrens die Pruefung des Optimierungslaufs nicht besteht.

    Und das Restbudget laesst sich hier gegen das Protokoll halten: Budget minus die Dollar
    aller acht Messlaeufe, so wie die Auswertung es spaeter nachrechnet.
    """
    from dspy_optimizer.kandidatenmodelle import DIE_VIER
    from dspy_optimizer.verfahren import Verfahren

    kostenmodelle = [
        Kostenmodell(
            kandidatenmodell=modell.name,
            fixkosten=Parameter(0.05, "erfunden"),
            kosten_je_trial=Parameter(0.01, "erfunden"),
            ausfuehrungskosten=Parameter(2 * DOLLAR_JE_AUFRUF, "erfunden"),
        )
        for modell in DIE_VIER
    ]
    verfahren = Verfahren(
        optimierungslauf=lauf,
        kostenmodelle=kostenmodelle,
        kostenobergrenze=5.0,
        protokoll=protokoll,
    )

    zwischenstand = verfahren.sondiere(MultiHopQA(), 0.64, seed=0, durchgang=1)

    *messlaeufe, entscheidung = protokoll.zeilen()
    assert [zeile["abschnitt"] for zeile in messlaeufe] == ["optimierung", "nachbewertung"] * 4
    assert [zeile["konfiguration"]["kandidatenmodell"] for zeile in messlaeufe[::2]] == [
        modell.name for modell in DIE_VIER
    ]
    assert {zeile["phase"] for zeile in protokoll.zeilen()} == {"sondierung"}
    assert [zeile["dollar_budgetanteil"] for zeile in messlaeufe] == [pytest.approx(0.64 / 8)] * 8
    assert entscheidung["abschnitt"] == "zwischenbewertung"
    ausgegeben = sum(zeile["dollar"] for zeile in messlaeufe)
    assert zwischenstand.ausgegeben == pytest.approx(ausgegeben)
    assert entscheidung["restbudget"] == pytest.approx(0.64 - ausgegeben)
    # Alle vier antworten gleich: Gleichstand ueberall, der erste fuehrt, keiner ist schlechter.
    assert entscheidung["weiter"] == [DIE_VIER[0].name, DIE_VIER[1].name]
    assert entscheidung["test_hat_gefunden"] is False


def test_ein_ganzer_durchgang_des_verfahrens_laeuft_durch_die_echte_naht(lauf, protokoll):
    """Wie oben, nur bis zum Ende: Fokusphase und Endauswahl. Hier faellt auf, ob
    der Warm-Start durch die echte Naht geht - das Programm der Sondierung kommt aus
    MIPROv2 und muss ohne Few-Shot-Beispiele und ohne eigenes LM wieder hinein.

    Und der Budgetverbrauch laesst sich gegen das Protokoll halten: Budget minus die Dollar
    aller zwoelf Messlaeufe, Nachbewertungen eingeschlossen.
    """
    from dspy_optimizer.kandidatenmodelle import DIE_VIER
    from dspy_optimizer.verfahren import Verfahren

    kostenmodelle = [
        Kostenmodell(
            kandidatenmodell=modell.name,
            fixkosten=Parameter(0.05, "erfunden"),
            kosten_je_trial=Parameter(0.01, "erfunden"),
            ausfuehrungskosten=Parameter(2 * DOLLAR_JE_AUFRUF, "erfunden"),
        )
        for modell in DIE_VIER
    ]
    verfahren = Verfahren(
        optimierungslauf=lauf,
        kostenmodelle=kostenmodelle,
        kostenobergrenze=5.0,
        protokoll=protokoll,
    )

    ergebnis = verfahren(MultiHopQA(), 0.64, seed=0, durchgang=1)

    zeilen = protokoll.zeilen()
    assert [zeile["abschnitt"] for zeile in zeilen] == (
        ["optimierung", "nachbewertung"] * 4
        + ["zwischenbewertung"]
        + ["optimierung", "nachbewertung"] * 2
        + ["endauswahl"]
    )
    *sondierung, zwischen = zeilen[:9]
    *fokus, endauswahl = zeilen[9:]
    assert {zeile["phase"] for zeile in fokus + [endauswahl]} == {"fokus"}
    assert [zeile["konfiguration"]["kandidatenmodell"] for zeile in fokus[::2]] == [
        DIE_VIER[0].name,
        DIE_VIER[1].name,
    ]
    assert [zeile["dollar_budgetanteil"] for zeile in fokus] == [
        pytest.approx(zwischen["restbudget"] / 2)
    ] * 4
    alle_dollar = sum(zeile["dollar"] for zeile in sondierung + fokus)
    assert endauswahl["ausgegeben"] == pytest.approx(alle_dollar)
    assert endauswahl["uebrig"] == pytest.approx(0.64 - alle_dollar)
    assert ergebnis.budgetverbrauch.gesamt == pytest.approx(alle_dollar)
    # Alle antworten gleich: Gleichstand, die feste Reihenfolge entscheidet.
    assert endauswahl["gewinner"] == ergebnis.gewinner == DIE_VIER[0].name
    assert ergebnis.programm.kandidatenmodell == DIE_VIER[0].name


# --- Nur bewerten -----------------------------------------------------------------------
#
# Der zweite Eingang der Naht: ein Programm bewerten, ohne es vorher zu optimieren - fuer
# die Baseline ohne Optimierung, die Spannweite der Vorstudie und die Testbewertung.
# Gemessen wird genau wie in der Nachbewertung eines Optimierungslaufs.

OHNE_OPTIMIERUNG = Einordnung.experiment("ohne-optimierung", durchgang=1)


def test_ein_programm_laesst_sich_auf_den_auswahldaten_bewerten_ohne_zu_optimieren(
    lauf, attrappen, protokoll
):
    """Genau eine Zeile, die Nachbewertung - keine Optimierung davor, kein Vorschlagsmodell.
    Sie misst wie die Nachbewertung eines Optimierungslaufs auf den Auswahldaten.

    Die Attrappe antwortet immer "France": die Einzelbewertungen wechseln paarweise zwischen
    1 und 0, in der Reihenfolge der Auswahldaten.
    """
    ergebnis = lauf.nur_bewerten(
        KANDIDATENMODELL,
        MultiHopQA(),
        seed=0,
        einordnung=OHNE_OPTIMIERUNG,
        abschnitt="nachbewertung",
    )

    (zeile,) = protokoll.zeilen()
    assert zeile["abschnitt"] == "nachbewertung"
    assert zeile["baseline"] == "ohne-optimierung"
    assert zeile["konfiguration"]["auswahldaten"] == 8
    assert ergebnis.bewertungen_je_beispiel == (1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
    assert ergebnis.bewertung == pytest.approx(0.5)
    assert attrappen["vorschlagsmodelle"][0].aufrufe == 0


@pytest.fixture
def testdaten():
    """Vier Beispiele, anders gemischt als Validierungs- und Auswahldaten: richtig, falsch,
    falsch, richtig. So faellt auf, wenn die Testbewertung doch ueber andere Daten liefe."""
    return [
        beispiel(100 + nummer, antwort)
        for nummer, antwort in enumerate(["France", "Australia", "Peru", "France"])
    ]


@pytest.fixture
def lauf_mit_testdaten(
    trainingsdaten, validierungsdaten, auswahldaten, testdaten, protokoll, attrappen
):
    """Die Naht, wie das Hauptexperiment sie baut: mit Testdaten."""
    return Optimierungslauf(
        trainingsdaten=trainingsdaten,
        validierungsdaten=validierungsdaten,
        auswahldaten=auswahldaten,
        testdaten=testdaten,
        num_candidates=2,
        minibatch_size=3,
        protokoll=protokoll,
    )


def test_ein_programm_laesst_sich_auf_den_testdaten_bewerten(lauf_mit_testdaten, protokoll):
    """Genau eine Zeile mit Abschnitt Testbewertung, ueber die Testdaten."""
    ergebnis = lauf_mit_testdaten.nur_bewerten(
        KANDIDATENMODELL,
        MultiHopQA(),
        seed=0,
        einordnung=OHNE_OPTIMIERUNG,
        abschnitt="testbewertung",
    )

    (zeile,) = protokoll.zeilen()
    assert zeile["abschnitt"] == "testbewertung"
    assert zeile["konfiguration"]["testdaten"] == 4
    assert zeile["programmlaeufe"] == 4
    assert ergebnis.bewertungen_je_beispiel == (1.0, 0.0, 0.0, 1.0)
    assert zeile["einzelbewertungen"] == [1.0, 0.0, 0.0, 1.0]


@pytest.mark.parametrize(
    ("einordnung", "abschnitt", "meldung"),
    [
        (None, "nachbewertung", "Einordnung"),
        ("experiment", "testbewertung", "Einordnung"),
        (OHNE_OPTIMIERUNG, "optimierung", "Nur bewerten geht"),
        (OHNE_OPTIMIERUNG, None, "Nur bewerten geht"),
    ],
)
def test_nur_bewerten_prueft_einordnung_und_abschnitt_bevor_geld_fliesst(
    lauf_mit_testdaten, attrappen, protokoll, einordnung, abschnitt, meldung
):
    """Zweck, Baseline, Phase und Durchgang wie beim Optimierungslauf - und der
    Abschnitt: eine Optimierung gibt es auf diesem Eingang nicht."""
    with pytest.raises(ValueError, match=meldung):
        lauf_mit_testdaten.nur_bewerten(
            KANDIDATENMODELL,
            MultiHopQA(),
            seed=0,
            einordnung=einordnung,
            abschnitt=abschnitt,
        )

    assert attrappen["kandidatenmodelle"] == []
    assert protokoll.zeilen() == []


def test_nur_bewerten_laesst_sich_ohne_einordnung_und_abschnitt_gar_nicht_rufen(lauf, attrappen):
    """Eine vergessene Angabe ist kein stiller Standardwert - schon gar nicht der, der die
    Testdaten beruehrt."""
    with pytest.raises(TypeError, match="einordnung"):
        lauf.nur_bewerten(
            KANDIDATENMODELL, MultiHopQA(), seed=0, abschnitt="nachbewertung"
        )
    with pytest.raises(TypeError, match="abschnitt"):
        lauf.nur_bewerten(KANDIDATENMODELL, MultiHopQA(), seed=0, einordnung=HANDLAUF)

    assert attrappen["kandidatenmodelle"] == []


def test_ein_praediktor_mit_eigenem_lm_bricht_auch_das_bewerten_ab_bevor_geld_fliesst(
    lauf, attrappen, protokoll
):
    programm = MultiHopQA()
    programm.answer_question.lm = Attrappe()

    with pytest.raises(ValueError, match="eigenes LM"):
        lauf.nur_bewerten(
            KANDIDATENMODELL,
            programm,
            seed=0,
            einordnung=OHNE_OPTIMIERUNG,
            abschnitt="nachbewertung",
        )

    assert attrappen["kandidatenmodelle"] == []
    assert protokoll.zeilen() == []


def test_ohne_testdaten_gibt_es_keine_testbewertung(lauf, attrappen, protokoll):
    """Eine Bewertung ueber null Beispiele haette eine Zeile mit Bewertung 0,0 hinterlassen -
    eine erfundene Messung statt einer fehlenden."""
    with pytest.raises(ValueError, match="Testdaten"):
        lauf.nur_bewerten(
            KANDIDATENMODELL,
            MultiHopQA(),
            seed=0,
            einordnung=OHNE_OPTIMIERUNG,
            abschnitt="testbewertung",
        )

    assert attrappen["kandidatenmodelle"] == []
    assert protokoll.zeilen() == []


@pytest.mark.parametrize("einordnung", [VORSTUDIE, HANDLAUF])
def test_die_testdaten_beruehrt_nur_das_hauptexperiment(
    lauf_mit_testdaten, attrappen, protokoll, einordnung
):
    """Die Testdaten bleiben bis zur abschliessenden Auswertung unberuehrt. Eine Vorstudie
    oder ein Handlauf auf ihnen haette sie schon vorher gesehen."""
    with pytest.raises(ValueError, match="ADR-0008"):
        lauf_mit_testdaten.nur_bewerten(
            KANDIDATENMODELL,
            MultiHopQA(),
            seed=0,
            einordnung=einordnung,
            abschnitt="testbewertung",
        )

    assert attrappen["kandidatenmodelle"] == []
    assert protokoll.zeilen() == []


@pytest.mark.parametrize("abschnitt", ["nachbewertung", "testbewertung"])
def test_nur_bewerten_misst_alles_was_die_nachbewertung_misst(
    lauf_mit_testdaten, protokoll, monkeypatch, abschnitt
):
    """Einzelbewertungen, Bewertung, Exact Match, Absatztreffer, Ausfuehrungskosten, Latenz
    und Fehlerzahlen. Die Attrappe antwortet immer "France" und nennt immer
    Absatz "A"; jeder LM-Aufruf wartet `WARTEZEIT`, ein Programmlauf also doppelt so lang."""
    kandidatenmodell_antwortet_wie(monkeypatch, WartendeAttrappe)

    ergebnis = lauf_mit_testdaten.nur_bewerten(
        KANDIDATENMODELL,
        MultiHopQA(),
        seed=0,
        einordnung=OHNE_OPTIMIERUNG,
        abschnitt=abschnitt,
    )

    (zeile,) = protokoll.zeilen()
    assert zeile["einzelbewertungen"] == list(ergebnis.bewertungen_je_beispiel)
    assert zeile["bewertung"] == pytest.approx(ergebnis.bewertung) == pytest.approx(0.5)
    assert zeile["exakt"] == pytest.approx(0.5)
    assert zeile["absatztreffer"] == pytest.approx(1.0)
    assert ergebnis.ausfuehrungskosten == pytest.approx(2 * DOLLAR_JE_AUFRUF)
    assert zeile["dollar_je_programmlauf"] == pytest.approx(ergebnis.ausfuehrungskosten)
    assert ergebnis.bewertungskosten == zeile["dollar"] > 0
    for feld in ("latenz_mittelwert_sekunden", "latenz_median_sekunden"):
        assert 2 * WARTEZEIT <= zeile[feld] < 2 * WARTEZEIT + 0.5
    assert zeile["dauer_sekunden"] > 0
    assert (ergebnis.formatfehler, ergebnis.anbieterfehler) == (0, 0)
    assert (zeile["formatfehler"], zeile["anbieterfehler"]) == (0, 0)


def test_unoptimiert_und_optimiert_stehen_in_denselben_spalten(lauf, protokoll):
    """Die Ausfuehrungskosten eines unoptimierten und eines optimierten Programms sind direkt
    vergleichbar (Messfrage 3 der Vorstudie): dieselbe Messung, dieselben Felder. Was die beiden
    Zeilen unterscheidet, steht in der Konfiguration - und leer bei `num_trials`, wo vor der
    Bewertung keine Optimierung lief."""
    lauf(KANDIDATENMODELL, MultiHopQA(), num_trials=2, seed=0, einordnung=VORSTUDIE)
    lauf.nur_bewerten(
        KANDIDATENMODELL,
        MultiHopQA(),
        seed=0,
        einordnung=VORSTUDIE,
        abschnitt="nachbewertung",
    )

    _, optimiert, unoptimiert = protokoll.zeilen()
    assert optimiert["abschnitt"] == unoptimiert["abschnitt"] == "nachbewertung"
    assert optimiert.keys() == unoptimiert.keys()
    assert optimiert["konfiguration"]["num_trials"] == 2
    assert unoptimiert["konfiguration"]["num_trials"] is None
    assert optimiert["konfiguration"]["validierungsdaten"] == 6
    assert optimiert["konfiguration"]["auswahldaten"] == 8
    assert unoptimiert["konfiguration"]["auswahldaten"] == 8
    # Ohne Optimierung hat MIPROv2 keine Validierungsdaten benutzt.
    assert "validierungsdaten" not in unoptimiert["konfiguration"]
    assert "vorschlagsmodell" not in unoptimiert["konfiguration"]
    # Keine Nachbewertung kalibriert, ob nach Optimierung oder ohne: dafuer braucht es die
    # Optimierungszeile.
    assert protokoll.kalibrierbare_zeilen() == [protokoll.zeilen()[0]]


@pytest.mark.parametrize("abschnitt", ["nachbewertung", "testbewertung"])
def test_jede_bewertung_uebersteht_mehr_gescheiterte_beispiele_als_die_validierungsdaten_haben(
    trainingsdaten, validierungsdaten, protokoll, attrappen, monkeypatch, abschnitt
):
    """Die Fehlergrenze richtet sich nach den jeweils bewerteten Daten: bei MIPROv2 die
    Validierungsdaten, in der Nachbewertung die Auswahldaten, in der Testbewertung die
    Testdaten - mit 500 gegen 200 Beispielen. Mit der Grenze der Validierungsdaten braeche
    ein Endpunkt, der jede Anfrage ablehnt, die Bewertung mittendrin ab - nachdem sie bezahlt
    ist. Stattdessen: eine Zeile, jedes Beispiel ein Anbieterfehler."""
    kandidatenmodell_antwortet_wie(monkeypatch, KaputteAttrappe)
    bewertet = [beispiel(200 + nummer, "France") for nummer in range(2 * len(validierungsdaten))]
    lauf = Optimierungslauf(
        trainingsdaten=trainingsdaten,
        validierungsdaten=validierungsdaten,
        auswahldaten=bewertet,
        testdaten=bewertet,
        protokoll=protokoll,
    )

    ergebnis = lauf.nur_bewerten(
        KANDIDATENMODELL,
        MultiHopQA(),
        seed=0,
        einordnung=OHNE_OPTIMIERUNG,
        abschnitt=abschnitt,
    )

    (zeile,) = protokoll.zeilen()
    assert ergebnis.anbieterfehler == zeile["anbieterfehler"] == len(bewertet)
    assert ergebnis.bewertungen_je_beispiel == (0.0,) * len(bewertet)


# --- Fixkosten-Messung ------------------------------------------------------------------


def test_eine_fixkosten_messung_hinterlaesst_genau_eine_zeile_ohne_nachbewertung(
    lauf, protokoll, validierungsdaten
):
    """Nur der feste Teil: MIPROv2 mit `num_trials = 0` sammelt Beispiele, schlaegt
    Instruktionen vor und bewertet das unveraenderte Programm einmal vollstaendig
    (`mipro_optimizer_v2.py` Z. 530-538), danach haelt Optuna an. Eine Nachbewertung gibt es
    nicht. Alles, was die Messung kostet, sind Fixkosten."""
    ergebnis = lauf.nur_fixkosten(
        KANDIDATENMODELL, MultiHopQA(), seed=4, einordnung=VORSTUDIE, wiederholung=2
    )

    (zeile,) = protokoll.zeilen()
    assert zeile["abschnitt"] == "fixkosten"
    assert zeile["zweck"] == "vorstudie"
    assert zeile["seed"] == 4
    assert zeile["konfiguration"]["num_trials"] == 0
    assert zeile["konfiguration"]["wiederholung"] == 2
    assert zeile["fixkosten"] > 0
    assert zeile["fixkosten"] == pytest.approx(zeile["dollar"])
    assert zeile["programmlaeufe"] == len(validierungsdaten)
    assert zeile["dollar_je_programmlauf"] is None
    assert protokoll.kalibrierbare_zeilen() == []
    assert ergebnis.fixkosten == zeile["fixkosten"]
    assert ergebnis.bewertung == zeile["bewertung"]
    assert (ergebnis.formatfehler, ergebnis.anbieterfehler) == (0, 0)


def test_der_feste_teil_haengt_nicht_an_der_zahl_der_trials(lauf, protokoll):
    """Bei `auto=None` geht `num_trials` erst in Schritt 3 ein (`mipro_optimizer_v2.py`
    Z. 261-272, 299-302): Optimierungslaeufe mit 2 und 4 Trials und die Fixkosten-Messung
    haben denselben festen Teil. Die Attrappe kostet je Aufruf gleich viel und antwortet
    immer gleich; jeder Unterschied hiesse, dass Trials in die Fixkosten rutschen."""
    lauf.nur_fixkosten(KANDIDATENMODELL, MultiHopQA(), seed=0, einordnung=VORSTUDIE, wiederholung=1)
    for num_trials in (2, 4):
        lauf(KANDIDATENMODELL, MultiHopQA(), num_trials=num_trials, seed=0, einordnung=VORSTUDIE)

    messung, zwei, _, vier, _ = protokoll.zeilen()
    assert zwei["fixkosten"] == pytest.approx(messung["fixkosten"])
    assert vier["fixkosten"] == pytest.approx(messung["fixkosten"])
    assert messung["fixkosten"] < zwei["dollar"] < vier["dollar"]


@pytest.mark.parametrize("einordnung", [HANDLAUF, EXPERIMENT])
def test_eine_fixkosten_messung_gibt_es_nur_in_der_vorstudie_und_das_vor_dem_ersten_aufruf(
    lauf, attrappen, protokoll, einordnung
):
    with pytest.raises(ValueError, match="Vorstudie"):
        lauf.nur_fixkosten(
            KANDIDATENMODELL, MultiHopQA(), seed=0, einordnung=einordnung, wiederholung=1
        )

    assert attrappen["kandidatenmodelle"] == []
    assert protokoll.zeilen() == []


def test_ein_endpunkt_der_jede_anfrage_ablehnt_meldet_sich_auch_in_der_fixkosten_messung(
    lauf, protokoll, monkeypatch
):
    """Die Vorstudie haelt nach einem Anbieterfehler an; dafuer muss die Fixkosten-Messung
    ihn melden wie jeder andere Lauf."""
    kandidatenmodell_antwortet_wie(monkeypatch, KaputteAttrappe)

    ergebnis = lauf.nur_fixkosten(
        KANDIDATENMODELL, MultiHopQA(), seed=0, einordnung=VORSTUDIE, wiederholung=1
    )

    assert ergebnis.anbieterfehler > 0
    (zeile,) = protokoll.zeilen()
    assert zeile["anbieterfehler"] == ergebnis.anbieterfehler


def test_eine_testbewertung_traegt_den_vermerk_in_ihrer_konfiguration(
    lauf_mit_testdaten, protokoll
):
    """Welches gespeicherte Paar bewertet wurde und ob es aus einem
    Durchgang ohne Gewinner stammt, steht in der Zeile - nur so ist aus dem Protokoll
    auszuzaehlen, dass jedes Paar genau eine gueltige Testbewertung hat."""
    zusaetzliche_konfiguration = {
        "paar": "ergebnisse/programme/durchgang-001/a.json",
        "kein_gewinner": True,
    }

    lauf_mit_testdaten.nur_bewerten(
        KANDIDATENMODELL,
        MultiHopQA(),
        seed=1,
        einordnung=OHNE_OPTIMIERUNG,
        abschnitt="testbewertung",
        zusaetzliche_konfiguration=zusaetzliche_konfiguration,
    )

    (zeile,) = protokoll.zeilen()
    assert zeile["konfiguration"]["paar"] == zusaetzliche_konfiguration["paar"]
    assert zeile["konfiguration"]["kein_gewinner"] is True
    assert zeile["konfiguration"]["kandidatenmodell"] == KANDIDATENMODELL.name


def test_die_zusaetzliche_konfiguration_ersetzt_kein_feld_der_naht(
    lauf_mit_testdaten, attrappen, protokoll
):
    """Ein Feld wie `kandidatenmodell` ersetzte still den Wert, den die Naht setzt; abgelehnt
    wird, bevor Geld fliesst."""
    with pytest.raises(ValueError, match="kandidatenmodell"):
        lauf_mit_testdaten.nur_bewerten(
            KANDIDATENMODELL,
            MultiHopQA(),
            seed=1,
            einordnung=OHNE_OPTIMIERUNG,
            abschnitt="testbewertung",
            zusaetzliche_konfiguration={"kandidatenmodell": "anderes"},
        )

    assert attrappen["kandidatenmodelle"] == []
    assert protokoll.zeilen() == []
