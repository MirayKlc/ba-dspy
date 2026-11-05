"""Tests zu den vier Kandidatenmodellen und ihrem LM-Bau.

`dspy.LM` zu bauen kostet nichts und braucht kein Netz - erst der Aufruf tut das. Geprueft
wird deshalb, *wie* das LM eingestellt ist, bevor im Experiment Geld daran haengt.
"""

import dspy
import pytest

from dspy_optimizer.kandidatenmodelle import (
    DAS_VORSCHLAGSMODELL,
    DIE_VIER,
    kandidatenmodell_mit_namen,
    lm_fuer,
    mess_adapter,
    messkontext,
)


def test_es_sind_genau_die_vier_kandidatenmodelle_in_fester_reihenfolge():
    """Die Reihenfolge bricht als letzte Regel einen Gleichstand und muss deshalb fest sein -
    seit dem Endpunktwechsel von DeepSeek nicht mehr nach Listenpreis."""
    assert [modell.name for modell in DIE_VIER] == [
        "openai/gpt-oss-120b",
        "google/gemma-4-31b-it",
        "deepseek/deepseek-v4.1-flash",
        "openai/gpt-6-luna",
    ]


def test_jedes_kandidatenmodell_ist_auf_genau_einen_endpunkt_festgenagelt():
    """Auf den Endpunkt samt Variante, nicht nur auf die Firma: DeepInfra fuehrt gpt-oss-120b
    unter demselben Namen auch als `turbo` und `fp8`, zu vier- bis fuenffachem Preis."""
    anbieter = [modell.anbieter for modell in DIE_VIER]

    assert anbieter == ["deepinfra/bf16", "deepinfra/turbo", "inference-net/fp8", "openai"]


def test_jedes_kandidatenmodell_bittet_um_so_wenig_denken_wie_sein_endpunkt_zulaesst():
    """Gemessen, nicht vermutet: DeepInfra lehnt bei gpt-oss-120b abgeschaltetes Denken mit
    HTTP 400 ab, gedrosselt wird es dort. "Alle auf low" ist gemessen verworfen: Gemma und
    DeepSeek fingen damit erst an zu denken und schnitten Antworten ab."""
    gebeten = {modell.name: modell.denken for modell in DIE_VIER}

    assert gebeten == {
        "openai/gpt-oss-120b": {"effort": "low"},
        "google/gemma-4-31b-it": {"enabled": False},
        "deepseek/deepseek-v4.1-flash": {"enabled": False},
        "openai/gpt-6-luna": {"enabled": False},
    }


def test_ein_unbekanntes_modell_ist_ein_fehler_und_kein_stiller_aufruf():
    with pytest.raises(KeyError, match="gibt-es-nicht"):
        kandidatenmodell_mit_namen("gibt-es-nicht")


@pytest.fixture
def lm():
    return lm_fuer(kandidatenmodell_mit_namen("google/gemma-4-31b-it"))


def test_das_lm_laeuft_ueber_openrouter(lm):
    assert lm.model == "openrouter/google/gemma-4-31b-it"


def test_das_lm_nagelt_den_anbieter_fest_und_verbietet_ausweichen(lm):
    """Ohne das koennte OpenRouter denselben Namen von einem anderen Anbieter bedienen lassen -
    unter Umstaenden in einer anderen Quantisierung, also als anderes Modell."""
    assert lm.kwargs["extra_body"]["provider"] == {
        "only": ["deepinfra/turbo"],
        "allow_fallbacks": False,
    }


def test_das_lm_gibt_die_denk_einstellung_des_kandidatenmodells_weiter(lm):
    """Gemma laesst das Denken abschalten; bei den Pflicht-Denkern steht `effort`."""
    assert lm.kwargs["extra_body"]["reasoning"] == {"enabled": False}
    gpt_oss = lm_fuer(kandidatenmodell_mit_namen("openai/gpt-oss-120b"))
    assert gpt_oss.kwargs["extra_body"]["reasoning"] == {"effort": "low"}


def test_bei_messlaeufen_ist_der_cache_aus_und_die_temperatur_null(lm):
    """Mit Cache waere der zweite Lauf gratis - und die Dollar-Messung waere hinfaellig."""
    assert lm.cache is False
    assert lm.kwargs["temperature"] == 0.0


def test_wiederholungen_sind_eingeschaltet(lm):
    """Ohne sie fiel ein frueheres Kandidatenmodell unter Last bei 3 von 24 Aufrufen aus; mit 3
    fiel DeepSeek in der Vorstudie an kurzen Aussetzern aus; mit 10 scheiterten im
    Hauptexperiment Durchgaenge an der Ueberlastung von DeepInfra."""
    assert lm.num_retries == 20


def test_der_adapter_faellt_nicht_still_auf_json_zurueck():
    """Der Rueckfall wuerde ein Kandidatenmodell mitten im Lauf anders fragen als die anderen -
    und bei einem frueher erwogenen Kandidatenmodell (Kimi K2 bei Novita) brach er mit HTTP 400
    ab, weil der Anbieter keine strukturierten Ausgaben kann."""
    gesetzter = mess_adapter()

    assert isinstance(gesetzter, dspy.ChatAdapter)
    assert gesetzter.use_json_adapter_fallback is False


def test_der_messkontext_setzt_lm_und_adapter_in_einem():
    """Die eine Stelle, durch die jeder Messlauf geht - damit der abgeschaltete JSON-Rueckfall
    nicht davon abhaengt, dass jeder Aufrufer von sich aus an den Adapter denkt."""
    vorher = dspy.settings.adapter

    with messkontext(kandidatenmodell_mit_namen("google/gemma-4-31b-it")) as lm:
        assert dspy.settings.lm is lm
        assert dspy.settings.adapter.use_json_adapter_fallback is False

    assert dspy.settings.adapter is vorher


def test_beim_vorschlagsmodell_ist_das_denken_abgeschaltet():
    """Hier geht es nicht ums Sparen, sondern darum, dass die Antwort ankommt.

    Mit eingeschaltetem Denken schrieb das fruehere Vorschlagsmodell DeepSeek V4 auf langen
    Anfragen die vollstaendige Antwort samt Marken nach `reasoning_content` und liess `content`
    leer; der ChatAdapter findet dort nichts und der Optimierungslauf bricht ab
    (`docs/spikes/vorschlagsmodell.json`). qwen3.7-flash erzeugt mit Denken rund 2000
    Denk-Tokens je Vorschlag (`docs/spikes/modellwechsel.json`).
    Festgenagelt wird der Anbieter wie bei jedem Modell des Experiments.
    """
    lm = lm_fuer(DAS_VORSCHLAGSMODELL)

    assert lm.model == "openrouter/qwen/qwen3.7-flash"
    assert lm.kwargs["extra_body"] == {
        "provider": {"only": ["alibaba"], "allow_fallbacks": False},
        "reasoning": {"enabled": False},
    }
    assert lm.cache is False


def test_das_vorschlagsmodell_hat_mehr_platz_als_ein_kandidatenmodell():
    """Es schreibt ganze Instruktionen. Eine abgeschnittene Antwort waere hier kein
    verlorener Punkt, sondern ein Abbruch: der ChatAdapter braucht die Schlussmarke, und
    einen Rueckfall auf JSON gibt es nicht."""
    vorschlag = lm_fuer(DAS_VORSCHLAGSMODELL)
    kandidat = lm_fuer(kandidatenmodell_mit_namen("google/gemma-4-31b-it"))

    assert vorschlag.kwargs["max_tokens"] > kandidat.kwargs["max_tokens"]


def test_kopien_eines_lm_schreiben_in_dieselbe_history():
    """Der Fund, an dem die Erfassung der Fixkosten haengt.

    MIPROv2s Proposer ruft ueber `prompt_model.copy(rollout_id=...)` an, um den Cache zu
    umgehen (`propose/grounded_proposer.py` Z. 395-397). DSPys `copy()` gibt der Kopie eine
    frische, leere History (`clients/base_lm.py` Z. 755-757) - die Kostenerfassung summiert
    aber je Sprachmodell aus dessen History. Ohne diese Klasse stuende das
    Vorschlagen der Instruktionen mit 0,00 $ im Protokoll.
    """
    lm = lm_fuer(DAS_VORSCHLAGSMODELL)
    kopie = lm.copy(rollout_id=1, temperature=1.0)

    assert kopie.history is lm.history
    # Zum Vergleich das Verhalten, das umgangen wird: DSPys eigenes LM teilt nicht.
    blank = dspy.LM("openrouter/deepseek/deepseek-v4-flash-0731", cache=False)
    assert blank.copy(rollout_id=1).history is not blank.history


def test_die_kopie_behaelt_die_abweichenden_einstellungen():
    """Die geteilte History darf den Zweck der Kopie nicht aushebeln: der Proposer braucht
    eine andere `rollout_id` und eine andere Temperatur, sonst greift der Cache."""
    lm = lm_fuer(DAS_VORSCHLAGSMODELL)

    kopie = lm.copy(rollout_id=42, temperature=1.0)

    assert kopie.kwargs["temperature"] == 1.0
    assert lm.kwargs["temperature"] == 0.0
