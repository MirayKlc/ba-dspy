"""Vortest: eignet sich das Vorschlagsmodell fuer den Proposer von MIPROv2?

Der erste echte Optimierungslauf ist am Instruktionsvorschlag abgebrochen: DeepSeek V4 Flash
bei DeepInfra hat die vollstaendige Antwort - samt der Marken `[[ ## proposed_instruction ## ]]`
und `[[ ## completed ## ]]` - in `reasoning_content` geschrieben und `content` leer gelassen.
Der ChatAdapter findet dort nichts und bricht ab, da ohne Rueckfall auf den JSONAdapter
gemessen wird.

Gefragt ist deshalb nicht "denkt es viel oder wenig" wie in `denken.py`, sondern: **kommt die
Antwort im richtigen Feld an?** Geprueft wird das mit einer Signatur in derselben Form, die
MIPROv2s Proposer stellt - eine laengere Anweisung, ein laengerer Freitext als Ausgabe.

Aufruf:  python spikes/vorschlagsmodell.py
Kosten:  wenige Zehntelcent.
"""

import json
import pathlib
import sys
import time

from dotenv import load_dotenv

load_dotenv()

import dspy  # noqa: E402  (erst nach load_dotenv, damit der Schluessel steht)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from dspy_optimizer.kandidatenmodelle import (  # noqa: E402
    DAS_VORSCHLAGSMODELL,
    MAX_TOKENS_VORSCHLAG,
    mess_adapter,
)

BELEG = pathlib.Path(__file__).resolve().parent.parent / "docs" / "spikes" / "vorschlagsmodell.json"

VARIANTEN = {
    "nichts": None,
    "enabled_false": {"enabled": False},
    "effort_low": {"effort": "low"},
}

# Ersatzkandidaten, falls der erste Anlauf in keiner Variante durchgeht. Beide sind in der
# Vorauswahl gemessen und je auf einen Anbieter festgenagelt.
ERSATZ = [
    ("nousresearch/hermes-4-405b", "nebius"),
    ("google/gemma-4-31b-it", "novita"),
]


class WieDerProposer(dspy.Signature):
    """Propose a new instruction for a module of a DSPy program, using the information given."""

    basic_instruction: str = dspy.InputField(desc="the instruction the module has today")
    module_description: str = dspy.InputField(desc="what the module does")
    proposed_instruction: str = dspy.OutputField(desc="the new instruction, one paragraph")


EINGABE = {
    "basic_instruction": "Answer the question using only the given sentences. "
    "Reply with a short phrase.",
    "module_description": "The module receives a question and the sentences that were found "
    "to be relevant, and produces a short answer.",
}


def lm_mit(name, anbieter, reasoning):
    koerper = {"provider": {"only": [anbieter], "allow_fallbacks": False}}
    if reasoning is not None:
        koerper["reasoning"] = reasoning
    return dspy.LM(f"openrouter/{name}", cache=False, max_tokens=MAX_TOKENS_VORSCHLAG,
                   temperature=1.0, num_retries=0, extra_body=koerper)


def versuche(name, anbieter, reasoning):
    """Ein Vorschlag mit dieser Einstellung. Ob er im richtigen Feld ankommt, ist der Befund."""
    lm = lm_mit(name, anbieter, reasoning)
    try:
        with dspy.context(lm=lm, adapter=mess_adapter()):
            vorhersage = dspy.Predict(WieDerProposer)(**EINGABE)
    except Exception as fehler:  # noqa: BLE001 - *woran* es scheitert, ist der Befund
        gescheitert = {"gelungen": False, "fehler": f"{type(fehler).__name__}: {fehler}"[:200]}
        if lm.history:
            antwort = lm.history[-1]["response"].choices[0].message
            gescheitert["content_leer"] = not getattr(antwort, "content", None)
            gescheitert["hat_reasoning_content"] = bool(
                getattr(antwort, "reasoning_content", None)
            )
        return gescheitert

    eintrag = lm.history[-1]
    einzelheiten = getattr(eintrag["response"].usage, "completion_tokens_details", None)
    return {
        "gelungen": True,
        "vorschlag": vorhersage.proposed_instruction[:200],
        "denk_tokens": getattr(einzelheiten, "reasoning_tokens", None),
        "antwort_tokens": eintrag["response"].usage.completion_tokens,
        "kosten_usd": eintrag["cost"],
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    modelle = [(DAS_VORSCHLAGSMODELL.name, DAS_VORSCHLAGSMODELL.anbieter), *ERSATZ]
    gemessen = {}
    for name, anbieter in modelle:
        print(f"\n{name} bei {anbieter}")
        for variante, reasoning in VARIANTEN.items():
            ergebnis = versuche(name, anbieter, reasoning)
            gemessen[f"{name} / {variante}"] = ergebnis
            if ergebnis["gelungen"]:
                # `cost` ist `None`, wenn OpenRouter keinen Betrag mitschickt. Ein Fehler in
                # dieser Ausgabe verhinderte sonst das Schreiben des bereits bezahlten Belegs.
                print(f"  {variante:<16}ja    denk={str(ergebnis['denk_tokens']):>5}  "
                      f"antwort={ergebnis['antwort_tokens']:>5}  "
                      f"${ergebnis['kosten_usd'] or 0.0:.7f}")
            else:
                print(f"  {variante:<16}NEIN  content_leer="
                      f"{ergebnis.get('content_leer')} "
                      f"reasoning_content={ergebnis.get('hat_reasoning_content')}  "
                      f"{ergebnis['fehler'][:70]}")

    BELEG.parent.mkdir(parents=True, exist_ok=True)
    BELEG.write_text(
        json.dumps({"gelaufen_am": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "frage": "Signatur WieDerProposer, siehe spikes/vorschlagsmodell.py",
                    "gemessen": gemessen}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nBeleg geschrieben: {BELEG}")


if __name__ == "__main__":
    main()
