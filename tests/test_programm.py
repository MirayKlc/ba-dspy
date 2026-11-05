"""Tests zum zweistufigen DSPy-Programm.

Gearbeitet wird mit DSPys Attrappen-LM: es beantwortet jeden Aufruf aus einer vorgegebenen
Liste. Damit laesst sich pruefen, *was* das Programm die Module fragt und *was* es
zurueckgibt, ohne einen einzigen Dollar auszugeben.
"""

import dspy
import pytest
from dspy.dsp.utils.utils import dotdict
from dspy.utils.dummies import DummyLM

from dspy_optimizer.kandidatenmodelle import DIE_VIER
from dspy_optimizer.programm import (
    ModellProgrammPaar,
    MultiHopQA,
    format_paragraphs,
    bewerte_beispiele,
    lehne_eigenes_lm_ab,
)

ABSAETZE = [
    {"title": "Eiffel Tower", "sentences": ["The Eiffel Tower is in Paris."]},
    {"title": "Paris", "sentences": ["Paris is the capital of France."]},
    {"title": "Ablenker", "sentences": ["Kangaroos live in Australia."]},
]


@pytest.fixture
def beispiel():
    return dspy.Example(
        question="In which country is the Eiffel Tower?",
        answer="France",
        paragraphs=ABSAETZE,
        gold_titles=["Eiffel Tower", "Paris"],
    ).with_inputs("question", "paragraphs")


def attrappe(titel=("Eiffel Tower", "Paris"), text="The Eiffel Tower is in Paris.",
             antwort="France"):
    """Ein LM, das erst wie Modul 1 und dann wie Modul 2 antwortet."""
    return DummyLM([
        {"relevant_titles": list(titel), "relevant_text": text},
        {"answer": antwort},
    ])


def test_das_programm_liefert_eine_prediction_mit_antwort_und_gewaehlten_absaetzen(beispiel):
    """Eine saubere Prediction ist Bedingung fuer die Kostenerfassung."""
    programm = MultiHopQA()

    with dspy.context(lm=attrappe()):
        vorhersage = programm(**beispiel.inputs())

    assert isinstance(vorhersage, dspy.Prediction)
    assert vorhersage.answer == "France"
    assert vorhersage.relevant_titles == ["Eiffel Tower", "Paris"]
    assert vorhersage.relevant_text == "The Eiffel Tower is in Paris."


def test_ein_programmlauf_sind_genau_zwei_lm_aufrufe(beispiel):
    """Ein Programmlauf ist nicht ein LM-Aufruf, sondern zwei - einer je Modul."""
    lm = attrappe()
    programm = MultiHopQA()

    with dspy.context(lm=lm):
        programm(**beispiel.inputs())

    assert len(lm.history) == 2


def test_modul_zwei_sieht_nur_die_herausgezogenen_stellen_nicht_die_ablenker(beispiel):
    """Der Sinn der zweiten Stufe: die acht Ablenker sind bis dahin aussortiert."""
    lm = attrappe()
    programm = MultiHopQA()

    with dspy.context(lm=lm):
        programm(**beispiel.inputs())

    erster_prompt, zweiter_prompt = (str(eintrag["messages"]) for eintrag in lm.history)
    assert "Kangaroos" in erster_prompt
    assert "Kangaroos" not in zweiter_prompt
    assert "The Eiffel Tower is in Paris." in zweiter_prompt


def test_die_absaetze_gehen_mit_titel_und_saetzen_in_den_prompt(beispiel):
    text = format_paragraphs(beispiel.paragraphs)

    for absatz in ABSAETZE:
        assert absatz["title"] in text
        assert absatz["sentences"][0] in text


def test_ohne_eigenes_lm_je_praediktor_geht_die_pruefung_durch():
    lehne_eigenes_lm_ab(MultiHopQA())


def test_ein_praediktor_mit_eigenem_lm_wird_vor_dem_lauf_bemerkt():
    """Ein eigenes LM sticht Kontext und Settings - der Lauf maesse ein anderes Modell."""
    programm = MultiHopQA()
    programm.answer_question.set_lm(attrappe())

    with pytest.raises(ValueError, match="answer_question"):
        lehne_eigenes_lm_ab(programm)


def test_ein_lauf_ueber_mehrere_beispiele_gibt_je_beispiel_eine_bewertung(beispiel):
    lm = DummyLM([
        {"relevant_titles": ["Eiffel Tower", "Paris"], "relevant_text": "In Paris."},
        {"answer": "France"},
        {"relevant_titles": ["Ablenker"], "relevant_text": "Kangaroos."},
        {"answer": "Australia"},
    ])
    programm = MultiHopQA()

    with dspy.context(lm=lm):
        ergebnisse = bewerte_beispiele(programm, [beispiel, beispiel])

    assert [ergebnis.bewertung.exakt for ergebnis in ergebnisse] == [1.0, 0.0]
    assert [ergebnis.bewertung.absatztreffer for ergebnis in ergebnisse] == [1.0, 0.0]
    assert [ergebnis.vorhersage.answer for ergebnis in ergebnisse] == ["France", "Australia"]


def test_ein_lauf_prueft_vorher_dass_kein_praediktor_ein_eigenes_lm_hat(beispiel):
    programm = MultiHopQA()
    programm.find_evidence.set_lm(attrappe())

    with pytest.raises(ValueError, match="find_evidence"):
        with dspy.context(lm=attrappe()):
            bewerte_beispiele(programm, [beispiel])


# --- Das Modell-Programm-Paar: die Rueckgabe des Verfahrens ----------------------------

GEWINNER = DIE_VIER[1]

# So antwortet ein Kandidatenmodell im Format des ChatAdapters - erst Modul 1, dann Modul 2.
MODUL_1 = (
    '[[ ## relevant_titles ## ]]\n["Eiffel Tower", "Paris"]\n\n'
    "[[ ## relevant_text ## ]]\nThe Eiffel Tower is in Paris.\n\n"
    "[[ ## completed ## ]]"
)
MODUL_2 = "[[ ## answer ## ]]\nFrance\n\n[[ ## completed ## ]]"

# Eine Antwort ohne die Marken des ChatAdapters: ein Formatfehler.
OHNE_MARKEN = "Die Antwort ist France."


class Endpunkt:
    """Statt OpenRouter: antwortet der Reihe nach aus `antworten` und merkt sich, welches
    Sprachmodell mit welchen Einstellungen gefragt wurde. Kein Aufruf verlaesst den Rechner."""

    def __init__(self, antworten):
        self.antworten = list(antworten)
        self.gefragt = []

    def forward(self, lm, prompt=None, messages=None, **kwargs):
        self.gefragt.append((lm.model, lm.kwargs))
        antwort = dotdict(
            choices=[
                dotdict(
                    message=dotdict(content=self.antworten.pop(0), tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=dotdict(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            model=lm.model,
        )
        antwort._hidden_params = {"response_cost": 0.0}
        return antwort


@pytest.fixture
def endpunkt(monkeypatch):
    """Jedes `dspy.LM` fragt den `Endpunkt` statt das Netz."""
    endpunkt = Endpunkt([MODUL_1, MODUL_2])
    monkeypatch.setattr(
        dspy.LM, "forward", lambda lm, *args, **kwargs: endpunkt.forward(lm, *args, **kwargs)
    )
    return endpunkt


def optimiertes_programm():
    """Ein Programm, wie es aus der Optimierung kommt: eigene Instruktion, ein Beispiel."""
    programm = MultiHopQA()
    programm.answer_question.signature = programm.answer_question.signature.with_instructions(
        "Antworte knapp."
    )
    programm.answer_question.demos = [
        dspy.Example(question="Wo steht der Eiffelturm?", relevant_text="In Paris.", answer="Paris")
    ]
    return programm


def test_das_paar_traegt_das_kandidatenmodell_fest_in_jedem_praediktor():
    """In den Praediktoren ist das Gewinnermodell fest hinterlegt - mit allem, was ein
    Messlauf auch hat: festgenagelter Anbieter, Denken, Temperatur."""
    programm = optimiertes_programm()

    paar = ModellProgrammPaar.aus_programm_und_modell(programm, GEWINNER)

    assert paar.kandidatenmodell == GEWINNER.name
    for _, praediktor in paar.named_predictors():
        assert praediktor.lm.model == f"openrouter/{GEWINNER.name}"
        assert praediktor.lm.kwargs["extra_body"]["provider"] == {
            "only": [GEWINNER.anbieter],
            "allow_fallbacks": False,
        }
        assert praediktor.lm.kwargs["temperature"] == 0.0
    assert paar.programm.answer_question.signature.instructions == "Antworte knapp."
    assert len(paar.programm.answer_question.demos) == 1
    # Das Programm, aus dem das Paar entsteht, bleibt ohne eigenes LM.
    lehne_eigenes_lm_ab(programm)


def test_das_paar_laeuft_auf_seinem_modell_auch_wenn_der_kontext_ein_anderes_setzt(
    endpunkt, beispiel
):
    paar = ModellProgrammPaar.aus_programm_und_modell(optimiertes_programm(), GEWINNER)
    anderes = attrappe(antwort="Germany")

    with dspy.context(lm=anderes):
        vorhersage = paar(**beispiel.inputs())

    assert vorhersage.answer == "France"
    assert [modell for modell, _ in endpunkt.gefragt] == [f"openrouter/{GEWINNER.name}"] * 2
    assert anderes.history == []


def speichern_und_laden(paar, tmp_path, art):
    """Die beiden Arten, wie DSPy ein Programm speichert: nur den Zustand als JSON, oder das
    ganze Programm mit cloudpickle."""
    if art == "zustand":
        paar.save(tmp_path / "paar.json")
        geladen = ModellProgrammPaar()
        geladen.load(tmp_path / "paar.json")
        return geladen
    paar.save(tmp_path / "paar", save_program=True)
    return dspy.load(tmp_path / "paar", allow_pickle=True)


@pytest.mark.parametrize("art", ["zustand", "programm"])
def test_nach_speichern_und_laden_laeuft_das_paar_ohne_weitere_konfiguration(
    endpunkt, beispiel, tmp_path, art
):
    """Laden und ausfuehren genuegt - kein `dspy.configure`, kein Kontext."""
    geladen = speichern_und_laden(
        ModellProgrammPaar.aus_programm_und_modell(optimiertes_programm(), GEWINNER), tmp_path, art
    )

    vorhersage = geladen(**beispiel.inputs())

    assert vorhersage.answer == "France"
    assert geladen.kandidatenmodell == GEWINNER.name
    assert geladen.programm.answer_question.signature.instructions == "Antworte knapp."
    assert len(geladen.programm.answer_question.demos) == 1
    for modell, einstellungen in endpunkt.gefragt:
        assert modell == f"openrouter/{GEWINNER.name}"
        assert einstellungen["extra_body"]["provider"]["only"] == [GEWINNER.anbieter]
        assert einstellungen["extra_body"]["reasoning"] == GEWINNER.denken


@pytest.mark.parametrize("art", ["zustand", "programm"])
def test_nach_dem_laden_gibt_es_keinen_rueckfall_auf_json(endpunkt, beispiel, tmp_path, art):
    """DSPy speichert den Adapter nicht mit. Ohne die Huelle liefe ein geladenes Programm mit
    dem JSON-Rueckfall und fragte nach einem Formatfehler ein zweites Mal - bei einem frueher
    erwogenen Kandidatenmodell (Kimi K2) endete das mit HTTP 400. Mit der Huelle bleibt es beim
    einen Aufruf und dem Fehler."""
    geladen = speichern_und_laden(
        ModellProgrammPaar.aus_programm_und_modell(optimiertes_programm(), GEWINNER), tmp_path, art
    )
    endpunkt.antworten = [OHNE_MARKEN, OHNE_MARKEN]

    with pytest.raises(Exception, match="Adapter"):
        geladen(**beispiel.inputs())

    assert len(endpunkt.gefragt) == 1


def test_ein_geladenes_paar_gibt_sein_programm_ohne_modell_heraus(tmp_path):
    """Die Testbewertung laedt gespeicherte Paare und misst sie durch
    `nur_bewerten`, das ein Programm ohne eigenes LM verlangt - sonst liefen die Aufrufe an der
    Kostenerfassung vorbei. Instruktionen und Beispiele bleiben, nur das LM faellt weg; das
    geladene Paar selbst behaelt seines."""
    geladen = speichern_und_laden(
        ModellProgrammPaar.aus_programm_und_modell(optimiertes_programm(), GEWINNER),
        tmp_path,
        "zustand",
    )

    programm = geladen.programm_ohne_eigenes_lm()

    lehne_eigenes_lm_ab(programm)
    assert isinstance(programm, MultiHopQA)
    assert programm.answer_question.signature.instructions == "Antworte knapp."
    assert len(programm.answer_question.demos) == 1
    assert geladen.kandidatenmodell == GEWINNER.name
