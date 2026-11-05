"""Die Sprachmodelle des Experiments und der Aufbau ihrer `dspy.LM`-Objekte.

Alle Einstellungen, mit denen ein Sprachmodell angesprochen wird, stehen an dieser einen
Stelle, damit jeder Lauf des Experiments dieselben benutzt. Preise in US-Dollar je 1 Mio.
Tokens beim festgelegten Endpunkt, Stand 27.09.2026 (DeepSeek: 10.10.2026), ohne befristeten
Rabatt.

Es gibt zwei getrennte Rollen: die vier **Kandidatenmodelle** konkurrieren um Budget und
loesen die Aufgabe; das **Vorschlagsmodell** schreibt nur die Instruktionen und bleibt ueber
alle Laeufe dasselbe.
"""

import contextlib
from typing import NamedTuple

import dspy


# So wenig Denken (Reasoning), wie der jeweilige Endpunkt zulaesst. Beide Werte sind der
# `reasoning`-Block, den OpenRouter im Anfrage-Koerper erwartet.
AUS = {"enabled": False}
SO_WENIG_WIE_MOEGLICH = {"effort": "low"}


class Kandidatenmodell(NamedTuple):
    name: str
    anbieter: str
    preis_prompt: float
    preis_antwort: float
    denken: dict


class Vorschlagsmodell(NamedTuple):
    """Das Modell, das die Instruktionen vorschlaegt; kein Kandidatenmodell.

    Eigener Typ und nicht `Kandidatenmodell`, obwohl die Felder dieselben sind: es
    konkurriert um kein Budget, loest die Aufgabe nie und kann nicht ausgewaehlt werden.
    Wo der Unterschied im Code etwas aendert, steht er in `lm_fuer`.
    """

    name: str
    anbieter: str
    preis_prompt: float
    preis_antwort: float
    denken: dict


# Die Reihenfolge ist fest: aufsteigender Listenpreis zum Stand 28.09.2026. Sie bleibt auch,
# nachdem DeepSeek mit dem Wechsel zu inference.net (10.10.2026) mehr je Antwort-Token kostet
# als Luna. Sie entscheidet nur, welches Modell zuerst laeuft, und bricht als letzte Regel
# einen Gleichstand; dafuer muss sie fest sein und darf nicht vom aktuellen Preis abhaengen.
#
# Festgelegt wird der Endpunkt samt Variante, nicht nur der Anbieter: DeepInfra fuehrt
# gpt-oss-120b unter demselben Namen auch als `turbo` und `fp8`, zum vier- bis fuenffachen
# Preis.
DIE_VIER = (
    # DeepInfra lehnt abgeschaltetes Denken hier mit HTTP 400 ab: "Reasoning is mandatory
    # for this endpoint and cannot be disabled." Reduzieren laesst es sich.
    Kandidatenmodell(
        "openai/gpt-oss-120b", "deepinfra/bf16", 0.037, 0.170, SO_WENIG_WIE_MOEGLICH
    ),
    # fp4. Mit `effort: low` beginnt es zu denken und schneidet Antworten ab.
    Kandidatenmodell("google/gemma-4-31b-it", "deepinfra/turbo", 0.090, 0.340, AUS),
    # fp8. Seit dem 10.10.2026; vorher OpenInference (fp4), das in der Vorstudie mit HTTP 504
    # ausfiel und danach bei OpenRouter gestoert war. Der Endpunkt des Herstellers ist ueber die
    # Datenschutz-Einstellung des Kontos gesperrt.
    Kandidatenmodell("deepseek/deepseek-v4.1-flash", "inference-net/fp8", 0.070, 0.600, AUS),
    # Nimmt keine Temperatur an; OpenRouter verwirft den Wert.
    Kandidatenmodell("openai/gpt-6-luna", "openai", 0.100, 0.500, AUS),
)

# Das Vorschlagsmodell ist ueber alle Kandidatenmodelle und Phasen dasselbe. Es stammt aus
# einer Modellfamilie, die unter den Kandidatenmodellen nicht vorkommt; das zuvor vorgesehene
# DeepSeek V4 haette sonst die Instruktionen fuer das verwandte DeepSeek V4.1 geschrieben.
# Angesprochen beim Hersteller Alibaba, ohne befristeten Rabatt.
#
# Das Denken ist nicht aus Kostengruenden abgeschaltet: das fruehere Vorschlagsmodell
# DeepSeek V4 schrieb mit Denken bei langen Anfragen die **vollstaendige** Antwort nach
# `reasoning_content` und liess `content` leer; der ChatAdapter findet dort nichts, und der
# Optimierungslauf bricht ab. qwen3.7-flash erzeugt mit Denken 2000 Denk-Tokens je Vorschlag,
# ohne Denken keine (`docs/spikes/modellwechsel.json`).
DAS_VORSCHLAGSMODELL = Vorschlagsmodell("qwen/qwen3.7-flash", "alibaba", 0.03, 0.13, AUS)

# Grosszuegig bemessen, damit ein Modell nicht am Platz scheitert statt an der Aufgabe.
MAX_TOKENS = 1000

# Das Vorschlagsmodell bekommt mehr Platz: es schreibt ganze Instruktionen und fasst den
# Datensatz zusammen, waehrend ein Kandidatenmodell nur eine kurze Antwort liefert. Eine
# abgeschnittene Antwort waere hier kein verlorener Punkt, sondern ein Abbruch: der
# ChatAdapter braucht die Schlussmarke, und der Rueckfall auf JSON ist abgeschaltet
# (`mess_adapter`). Ein Abbruch mitten im Lauf entwertet alle bis dahin angefallenen Kosten.
MAX_TOKENS_VORSCHLAG = 4000

# Wiederholungen bei voruebergehenden Ausfaellen der Anbieter (HTTP 504 bei einem frueheren
# Anbieter von DeepSeek, HTTP 429 bei DeepInfra wegen Ueberlastung). LiteLLM wartet zwischen
# den Versuchen zusammen 165 s (LiteLLM 1.101.0, `main.py` Z. 5874-5878), zuzueglich der Dauer
# der gescheiterten Versuche. Ein gescheiterter Versuch kostet nichts, und an einem Lauf ohne
# Ausfaelle aendert die Zahl nichts.
WIEDERHOLUNGEN = 20


class MitgezaehltesLM(dspy.LM):
    """Ein Sprachmodell, dessen Kopien in **dieselbe** History schreiben.

    DSPys `copy()` gibt der Kopie eine neue, leere History-Liste
    (`clients/base_lm.py` Z. 755-757). Der Proposer von MIPROv2 ruft das Vorschlagsmodell
    aber ueber eine solche Kopie auf - `self.prompt_model.copy(rollout_id=..., temperature=...)`
    (`propose/grounded_proposer.py` Z. 395-397), um den Cache zu umgehen.

    Die Kostenerfassung summiert Dollar und Anbieter je Sprachmodell aus dessen `history`.
    Ohne diese Klasse landeten alle Aufrufe des Vorschlagsmodells in einer Liste, die nicht
    ausgewertet wird: das Vorschlagen der Instruktionen, ein Teil der **Fixkosten**, stuende
    mit 0,00 $ im Protokoll. Doppelt gezaehlt wird nichts, weil die Kostenerfassung die
    Eintraege ueber ihre `uuid` unterscheidet.
    """

    def copy(self, **kwargs):
        kopie = super().copy(**kwargs)
        kopie.history = self.history
        return kopie


def kandidatenmodell_mit_namen(name):
    """Das Kandidatenmodell zu einem OpenRouter-Namen."""
    for kandidatenmodell in DIE_VIER:
        if kandidatenmodell.name == name:
            return kandidatenmodell
    raise KeyError(
        f"{name!r} ist kein Kandidatenmodell. Die vier sind: "
        + ", ".join(modell.name for modell in DIE_VIER)
    )


def lm_fuer(modell, *, klasse=MitgezaehltesLM):
    """Das `dspy.LM` fuer ein Kandidatenmodell oder das Vorschlagsmodell, wie im Experiment.

    Der Anbieter wird festgelegt und das Ausweichen auf andere Anbieter verboten. Welcher
    Anbieter einen Aufruf tatsaechlich bedient hat, steht danach je Aufruf in `lm.history`
    unter `response.provider` und wird von der Kostenerfassung protokolliert.

    Jedes Modell wird mit so wenig Denken angesprochen, wie sein Endpunkt zulaesst; das ist je
    Endpunkt gemessen: einer der vier Kandidaten-Endpunkte (gpt-oss-120b bei DeepInfra) lehnt
    das Abschalten ab, die uebrigen und der des Vorschlagsmodells nehmen es an.

    Der einzige Unterschied zwischen den beiden Rollen ist die Obergrenze der Antwort-Tokens.
    Ein Kandidatenmodell gibt eine kurze Antwort, das Vorschlagsmodell schreibt ganze
    Instruktionen; eine abgeschnittene Antwort waere dort kein verlorener Punkt, sondern ein
    Abbruch.

    `cache=False`, weil ein Cache-Treffer nichts kostet und die Dollar-Messung, auf der das
    Verfahren beruht, damit zu niedrig ausfiele.

    `klasse` ist nur fuer das Modell-Programm-Paar ein gewoehnliches `dspy.LM`
    (`dspy_optimizer.programm.ModellProgrammPaar`): DSPy laedt ein gespeichertes Programm
    mit einer eigenen LM-Klasse nur mit einem zusaetzlichen Schalter (`clients/base_lm.py`
    Z. 720-724). Die Einstellungen sind dieselben.
    """
    return klasse(
        f"openrouter/{modell.name}",
        cache=False,
        max_tokens=(
            MAX_TOKENS if isinstance(modell, Kandidatenmodell) else MAX_TOKENS_VORSCHLAG
        ),
        temperature=0.0,
        num_retries=WIEDERHOLUNGEN,
        extra_body={
            "provider": {"only": [modell.anbieter], "allow_fallbacks": False},
            "reasoning": modell.denken,
        },
    )


def mess_adapter():
    """Der Adapter, mit dem gemessen wird: ChatAdapter, ohne Rueckfall auf JSON.

    DSPy faellt bei einem Formatfehler von sich aus auf den JSONAdapter zurueck. Fuer die
    Messung ist das schaedlich: der Rueckfall fragt dasselbe Kandidatenmodell mitten im Lauf
    in einem anderen Format und kostet einen zusaetzlichen Aufruf. Bei einem frueher
    erwogenen Kandidatenmodell (Kimi K2 bei Novita) scheiterte er zudem mit HTTP 400
    ("does not support feature: structured-outputs"); aus einem Formatfehler, der null Punkte
    ergeben soll, wurde so ein Abbruch.
    """
    return dspy.ChatAdapter(use_json_adapter_fallback=False)


@contextlib.contextmanager
def messkontext(kandidatenmodell):
    """Der Kontext, in dem gemessen wird: das LM des Kandidatenmodells und der feste Adapter.

    Jeder Messlauf geht durch diese Stelle, der Lauf von Hand ebenso wie der
    Optimierungslauf. So haengt der feste Adapter nicht davon ab, dass jeder Aufrufer selbst
    `mess_adapter()` setzt.

    Gibt das LM zurueck, weil danach seine `history` gebraucht wird: dort stehen die Kosten je
    Aufruf und der bedienende Anbieter.
    """
    lm = lm_fuer(kandidatenmodell)
    with dspy.context(lm=lm, adapter=mess_adapter()):
        yield lm
