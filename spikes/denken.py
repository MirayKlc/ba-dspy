"""Vortest: laesst sich das Denken der vier Kandidatenmodelle abschalten?

Laut OpenRouter fuehren drei der vier Kandidatenmodelle den `reasoning`-Parameter als
unterstuetzt. Ob mit oder ohne Denken gemessen wird, ist damit noch nicht entschieden, denn
"unterstuetzt" heisst nicht "abschaltbar": dieses Skript prueft es.

Zwei Teile:

1. **Was nimmt der Endpunkt an?** Je Kandidatenmodell jede Variante des `reasoning`-Blocks
   aus `VARIANTEN`.
   Gemessen wird, ob der Aufruf durchgeht und wie viele Denk-Tokens dabei anfallen.
2. **Was bringt die Drosselung?** Fuer die Endpunkte, die das Abschalten verweigern, dieselbe
   HotpotQA-artige Frage dreimal mit und dreimal ohne `effort: low`, mit reichlich Platz, damit
   die Zahlen nicht an der Obergrenze haengen.

Aufruf:  python spikes/denken.py
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
from dspy_optimizer.kandidatenmodelle import DIE_VIER  # noqa: E402

BELEG = pathlib.Path(__file__).resolve().parent.parent / "docs" / "spikes" / "denken.json"

VARIANTEN = {
    "nichts": None,
    "enabled_false": {"enabled": False},
    "effort_low": {"effort": "low"},
}

# Eine Frage in der Form, die das Programm im Experiment stellt; kurz gehalten, damit der
# Vortest wenig kostet.
FRAGE = (
    "Paragraphs: A: The Eiffel Tower is in Paris. B: Paris is the capital of France. "
    "Question: In which country is the Eiffel Tower? Answer with a short phrase."
)

AUFRUFE_JE_EINSTELLUNG = 3


def lm_mit(kandidatenmodell, reasoning, max_tokens):
    koerper = {"provider": {"only": [kandidatenmodell.anbieter], "allow_fallbacks": False}}
    if reasoning is not None:
        koerper["reasoning"] = reasoning
    return dspy.LM(f"openrouter/{kandidatenmodell.name}", cache=False, max_tokens=max_tokens,
                   temperature=0.0, num_retries=0, extra_body=koerper)


def denk_tokens(eintrag):
    verbrauch = getattr(eintrag["response"], "usage", None)
    einzelheiten = getattr(verbrauch, "completion_tokens_details", None)
    return getattr(einzelheiten, "reasoning_tokens", None)


def versuche(kandidatenmodell, reasoning, max_tokens, aufrufe=1):
    """Einen Aufruf (oder mehrere) mit dieser Einstellung; Fehler sind hier ein Ergebnis."""
    lm = lm_mit(kandidatenmodell, reasoning, max_tokens)
    try:
        for _ in range(aufrufe):
            lm(FRAGE)
    except Exception as fehler:  # noqa: BLE001 - *dass* und *woran* es scheitert, ist der Befund
        return {"angenommen": False, "fehler": f"{type(fehler).__name__}: {fehler}"[:300]}
    return {
        "angenommen": True,
        "denk_tokens": [denk_tokens(e) for e in lm.history],
        "antwort_tokens": [e["response"].usage.completion_tokens for e in lm.history],
        "kosten_usd": [e["cost"] for e in lm.history],
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("Teil 1 - was nimmt der Endpunkt an?\n")
    angenommen = {}
    for kandidatenmodell in DIE_VIER:
        for name, reasoning in VARIANTEN.items():
            ergebnis = versuche(kandidatenmodell, reasoning, max_tokens=60)
            angenommen[f"{kandidatenmodell.name} / {name}"] = ergebnis
            zusatz = (f"denk_tokens={ergebnis['denk_tokens']}" if ergebnis["angenommen"]
                      else ergebnis["fehler"][:90])
            print(f"  {kandidatenmodell.name:<26}{name:<16}"
                  f"{'ja ' if ergebnis['angenommen'] else 'NEIN'}  {zusatz}")

    pflicht_denker = [
        kandidatenmodell for kandidatenmodell in DIE_VIER
        if not angenommen[f"{kandidatenmodell.name} / enabled_false"]["angenommen"]
    ]
    print(f"\nTeil 2 - was bringt die Drosselung bei den {len(pflicht_denker)} Pflicht-Denkern?\n")
    gedrosselt = {}
    for kandidatenmodell in pflicht_denker:
        for name in ("nichts", "effort_low"):
            ergebnis = versuche(kandidatenmodell, VARIANTEN[name], max_tokens=600,
                                aufrufe=AUFRUFE_JE_EINSTELLUNG)
            gedrosselt[f"{kandidatenmodell.name} / {name}"] = ergebnis
            mittel = sum(ergebnis["kosten_usd"]) / len(ergebnis["kosten_usd"])
            print(f"  {kandidatenmodell.name:<26}{name:<16}denk={ergebnis['denk_tokens']} "
                  f"antwort={ergebnis['antwort_tokens']} ${mittel:.7f}/Aufruf")

    BELEG.parent.mkdir(parents=True, exist_ok=True)
    BELEG.write_text(
        json.dumps({"gelaufen_am": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "frage": FRAGE,
                    "teil1_angenommen": angenommen,
                    "teil2_gedrosselt": gedrosselt}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nBeleg geschrieben: {BELEG}")


if __name__ == "__main__":
    main()
