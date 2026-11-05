"""Vortest zum Modellwechsel (27.09.2026): eignen sich die neuen Kandidatenmodelle?

Das bisherige Feld der Kandidatenmodelle wird ersetzt: das teuerste Kandidatenmodell soll
GPT-6 Luna sein, nichts darueber, damit das Budget reicht. Vorgeschlagen sind gpt-oss-120b,
Gemma 4 31B, DeepSeek V4.1 Flash und GPT-6 Luna, dazu qwen3.7-flash als Vorschlagsmodell.
Geprueft wird ausserdem `effort: low` fuer **alle** Kandidatenmodelle statt "so wenig Denken
wie moeglich"; nach diesem Vortest verworfen.

Drei Teile, alle ausschliesslich auf Trainingsdaten:

1. **Denken:** Was nimmt jeder Endpunkt an (nichts / abgeschaltet / `effort: low`), und wie
   viele Denk-Tokens entstehen dabei? "Unterstuetzt" heisst nicht "abschaltbar".
2. **Das echte Programm unter Last:** das zweistufige Programm ueber 24 Fragen, acht
   gleichzeitig, mit drei Wiederholungen, `effort: low`, `max_tokens` 1000 - so, wie das
   Experiment laufen wird. Gemessen: beantwortet, Fehler nach Art, F1, Dollar je
   Programmlauf, Denk-Tokens, abgeschnittene Antworten, Fragen je Minute, bedienender Anbieter.
3. **Das Vorschlagsmodell:** kommt die Antwort von qwen3.7-flash im richtigen Feld an
   (mit Denken landete sie bei DeepSeek in `reasoning_content`)?

Festgenagelt wird auf den **Endpunkt** (OpenRouter-Kuerzel samt Variante, etwa
`deepinfra/turbo`), nicht nur auf die Firma - Anbieter fuehren dasselbe Modell in mehreren
Quantisierungen.

Aufruf:  python spikes/modellwechsel.py
Kosten:  wenige Cent.
"""

import collections
import concurrent.futures
import json
import pathlib
import statistics
import sys
import time

from dotenv import load_dotenv

load_dotenv()

import dspy  # noqa: E402  (erst nach load_dotenv, damit der Schluessel steht)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from dspy_optimizer.datensatz import lade_aufteilung  # noqa: E402
from dspy_optimizer.kandidatenmodelle import (  # noqa: E402
    MAX_TOKENS,
    MAX_TOKENS_VORSCHLAG,
    WIEDERHOLUNGEN,
    mess_adapter,
)
from dspy_optimizer.metrik import bewertung_von  # noqa: E402
from dspy_optimizer.programm import ZweistufigesProgramm  # noqa: E402

BELEG = pathlib.Path(__file__).resolve().parent.parent / "docs" / "spikes" / "modellwechsel.json"

ANZAHL_FRAGEN = 24
GLEICHZEITIG = 8
LOW = {"effort": "low"}

# (Modell, Endpunkt-Kuerzel). Preise laut OpenRouter am 27.09.2026, je 1 Mio. Tokens, kein
# befristeter Rabatt an diesen Endpunkten.
KANDIDATEN = [
    ("openai/gpt-oss-120b", "deepinfra/bf16"),  # 0,037 / 0,17
    ("google/gemma-4-31b-it", "deepinfra/turbo"),  # fp4, 0,09 / 0,34
    ("google/gemma-4-31b-it", "reka"),  # 0,08 / 0,30
    ("deepseek/deepseek-v4.1-flash", "deepseek"),  # Hersteller, 0,15 / 0,60
    ("deepseek/deepseek-v4.1-flash", "open-inference/fp4"),  # 0,10 / 0,50
    ("openai/gpt-6-luna", "openai"),  # Standard-Endpunkt, 0,10 / 0,50
]

VORSCHLAG = ("qwen/qwen3.7-flash", "alibaba")  # Hersteller, 0,03 / 0,13

VARIANTEN = {
    "nichts": None,
    "enabled_false": {"enabled": False},
    "effort_low": LOW,
}

FRAGE = (
    "Paragraphs: A: The Eiffel Tower is in Paris. B: Paris is the capital of France. "
    "Question: In which country is the Eiffel Tower? Answer with a short phrase."
)


def lm_mit(modell, endpunkt, reasoning, max_tokens, temperatur=0.0, wiederholungen=0):
    koerper = {"provider": {"only": [endpunkt], "allow_fallbacks": False}}
    if reasoning is not None:
        koerper["reasoning"] = reasoning
    return dspy.LM(f"openrouter/{modell}", cache=False, max_tokens=max_tokens,
                   temperature=temperatur, num_retries=wiederholungen, extra_body=koerper)


def denk_tokens(eintrag):
    verbrauch = getattr(eintrag["response"], "usage", None)
    einzelheiten = getattr(verbrauch, "completion_tokens_details", None)
    return getattr(einzelheiten, "reasoning_tokens", None)


def abgeschnitten(eintrag):
    return eintrag["response"].choices[0].finish_reason == "length"


# --- Teil 1 ------------------------------------------------------------------------------

def denken(modell, endpunkt, reasoning):
    lm = lm_mit(modell, endpunkt, reasoning, MAX_TOKENS)
    try:
        antwort = lm(FRAGE)
    except Exception as fehler:  # noqa: BLE001 - *dass* und *woran* es scheitert, ist der Befund
        return {"angenommen": False, "fehler": f"{type(fehler).__name__}: {fehler}"[:300]}
    eintrag = lm.history[-1]
    return {
        "angenommen": True,
        # DSPy gibt je Antwort einen Text oder, bei Denkmodellen, ein Woerterbuch zurueck.
        "antwort": str(antwort[0] if antwort else "")[:80],
        "denk_tokens": denk_tokens(eintrag),
        "antwort_tokens": eintrag["response"].usage.completion_tokens,
        "kosten_usd": eintrag["cost"],
        "anbieter": str(getattr(eintrag["response"], "provider", None)),
    }


# --- Teil 2 ------------------------------------------------------------------------------

def fehlerart(ausnahme):
    name = type(ausnahme).__name__
    text = str(ausnahme)
    if "RateLimit" in name or "429" in text:
        return "rate_limit"
    if "AdapterParseError" in name:
        return "format"
    return name


def ein_versuch(programm, lm, beispiel):
    try:
        with dspy.context(lm=lm, adapter=mess_adapter()):
            vorhersage = programm(**beispiel.inputs())
    except Exception as ausnahme:  # noqa: BLE001 - im Vortest zaehlt, woran es scheitert
        return None, fehlerart(ausnahme)
    return bewertung_von(beispiel, vorhersage), None


def programm_unter_last(modell, endpunkt, beispiele, reasoning=LOW):
    lm = lm_mit(modell, endpunkt, reasoning, MAX_TOKENS, wiederholungen=WIEDERHOLUNGEN)
    programm = ZweistufigesProgramm()

    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=GLEICHZEITIG) as pool:
        ausgang = list(pool.map(lambda b: ein_versuch(programm, lm, b), beispiele))
    dauer = time.time() - start

    bewertungen = [b for (b, _) in ausgang if b is not None]
    f1 = [b.f1 for b in bewertungen]
    fehler = collections.Counter(art for (_, art) in ausgang if art)
    kosten = sum(eintrag["cost"] or 0.0 for eintrag in lm.history)
    denk = [denk_tokens(e) or 0 for e in lm.history]
    bedient = collections.Counter(
        str(getattr(e["response"], "provider", None)) for e in lm.history
    )
    return {
        "modell": modell,
        "endpunkt": endpunkt,
        "bedient_haben": dict(bedient),
        "beantwortet": len(bewertungen),
        "gestellt": len(beispiele),
        "fehler": dict(fehler),
        "f1": statistics.fmean(f1) if f1 else 0.0,
        "f1_standardfehler": (statistics.stdev(f1) / len(f1) ** 0.5) if len(f1) > 1 else None,
        "exakt": statistics.fmean(b.exakt for b in bewertungen) if bewertungen else 0.0,
        "lm_aufrufe": len(lm.history),
        "kosten_usd": kosten,
        "kosten_je_programmlauf_usd": kosten / len(beispiele),
        "denk_tokens_je_aufruf": statistics.fmean(denk) if denk else None,
        "denk_tokens_max": max(denk) if denk else None,
        "abgeschnitten": sum(abgeschnitten(e) for e in lm.history),
        "dauer_s": dauer,
        "fragen_je_minute": len(beispiele) / dauer * 60,
    }


# --- Teil 3 ------------------------------------------------------------------------------

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


def vorschlag(modell, endpunkt, reasoning):
    lm = lm_mit(modell, endpunkt, reasoning, MAX_TOKENS_VORSCHLAG, temperatur=1.0)
    try:
        with dspy.context(lm=lm, adapter=mess_adapter()):
            vorhersage = dspy.Predict(WieDerProposer)(**EINGABE)
    except Exception as fehler:  # noqa: BLE001 - *woran* es scheitert, ist der Befund
        gescheitert = {"gelungen": False, "fehler": f"{type(fehler).__name__}: {fehler}"[:200]}
        if lm.history:
            nachricht = lm.history[-1]["response"].choices[0].message
            gescheitert["content_leer"] = not getattr(nachricht, "content", None)
            gescheitert["hat_reasoning_content"] = bool(
                getattr(nachricht, "reasoning_content", None)
            )
        return gescheitert
    eintrag = lm.history[-1]
    return {
        "gelungen": True,
        "vorschlag": vorhersage.proposed_instruction[:200],
        "denk_tokens": denk_tokens(eintrag),
        "antwort_tokens": eintrag["response"].usage.completion_tokens,
        "kosten_usd": eintrag["cost"],
        "anbieter": str(getattr(eintrag["response"], "provider", None)),
    }


# Gegenprobe zu Teil 2: so wenig Denken, wie der Endpunkt zulaesst (die fruehere Regel).
# Der Endpunkt des Herstellers DeepSeek fehlt, weil die Datenschutz-Einstellung des Kontos
# ihn sperrt (Teil 1).
SO_WENIG_WIE_MOEGLICH = [
    ("openai/gpt-oss-120b", "deepinfra/bf16", LOW),  # Abschalten verweigert
    ("google/gemma-4-31b-it", "deepinfra/turbo", {"enabled": False}),
    ("google/gemma-4-31b-it", "reka", {"enabled": False}),
    ("deepseek/deepseek-v4.1-flash", "open-inference/fp4", {"enabled": False}),
    ("openai/gpt-6-luna", "openai", {"enabled": False}),
]


def gegenprobe():
    """Nur Teil 2, mit der frueheren Regel "so wenig wie moeglich" statt `effort: low` fuer alle."""
    beleg = BELEG.with_name("modellwechsel-so-wenig-wie-moeglich.json")
    beispiele = lade_aufteilung().trainingsdaten[:ANZAHL_FRAGEN]
    zeilen = []
    for modell, endpunkt, reasoning in SO_WENIG_WIE_MOEGLICH:
        zeile = programm_unter_last(modell, endpunkt, beispiele, reasoning)
        zeile["reasoning"] = reasoning
        zeilen.append(zeile)
        print(f"  {modell:<30}{endpunkt:<20}{zeile['beantwortet']}/{zeile['gestellt']}  "
              f"F1 {zeile['f1']:.2f}  EM {zeile['exakt']:.2f}  "
              f"${zeile['kosten_je_programmlauf_usd']:.6f}/Programmlauf  "
              f"denk/Aufruf {zeile['denk_tokens_je_aufruf'] or 0:.0f}  "
              f"abgeschnitten {zeile['abgeschnitten']}  {zeile['fragen_je_minute']:.0f} F/min  "
              f"{zeile['fehler'] or ''} {zeile['bedient_haben']}")
    beleg.write_text(
        json.dumps({"gelaufen_am": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "anzahl_fragen": len(beispiele),
                    "gleichzeitig": GLEICHZEITIG,
                    "wiederholungen": WIEDERHOLUNGEN,
                    "programm": zeilen}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nBeleg geschrieben: {beleg}")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.argv[1:] == ["gegenprobe"]:
        gegenprobe()
        return

    print("Teil 1 - Denken: was nimmt der Endpunkt an?\n")
    teil1 = {}
    for modell, endpunkt in KANDIDATEN:
        for name, reasoning in VARIANTEN.items():
            ergebnis = denken(modell, endpunkt, reasoning)
            teil1[f"{modell} @ {endpunkt} / {name}"] = ergebnis
            zusatz = (f"denk={ergebnis['denk_tokens']} antwort={ergebnis['antwort_tokens']}"
                      if ergebnis["angenommen"] else ergebnis["fehler"][:90])
            print(f"  {modell:<30}{endpunkt:<20}{name:<15}"
                  f"{'ja ' if ergebnis['angenommen'] else 'NEIN'}  {zusatz}")

    beispiele = lade_aufteilung().trainingsdaten[:ANZAHL_FRAGEN]
    print(f"\nTeil 2 - das echte Programm: {len(beispiele)} Fragen, {GLEICHZEITIG} gleichzeitig, "
          f"{WIEDERHOLUNGEN} Wiederholungen, effort low\n")
    teil2 = []
    for modell, endpunkt in KANDIDATEN:
        zeile = programm_unter_last(modell, endpunkt, beispiele)
        teil2.append(zeile)
        print(f"  {modell:<30}{endpunkt:<20}{zeile['beantwortet']}/{zeile['gestellt']}  "
              f"F1 {zeile['f1']:.2f}  ${zeile['kosten_je_programmlauf_usd']:.6f}/Programmlauf  "
              f"denk/Aufruf {zeile['denk_tokens_je_aufruf'] or 0:.0f}  "
              f"abgeschnitten {zeile['abgeschnitten']}  {zeile['fragen_je_minute']:.0f} F/min  "
              f"{zeile['fehler'] or ''} {zeile['bedient_haben']}")

    print("\nTeil 3 - Vorschlagsmodell\n")
    teil3 = {}
    for name, reasoning in VARIANTEN.items():
        ergebnis = vorschlag(*VORSCHLAG, reasoning)
        teil3[f"{VORSCHLAG[0]} @ {VORSCHLAG[1]} / {name}"] = ergebnis
        if ergebnis["gelungen"]:
            print(f"  {name:<16}ja    denk={ergebnis['denk_tokens']} "
                  f"antwort={ergebnis['antwort_tokens']}  ${ergebnis['kosten_usd'] or 0.0:.7f}")
        else:
            print(f"  {name:<16}NEIN  content_leer={ergebnis.get('content_leer')} "
                  f"reasoning_content={ergebnis.get('hat_reasoning_content')}  "
                  f"{ergebnis['fehler'][:70]}")

    BELEG.parent.mkdir(parents=True, exist_ok=True)
    BELEG.write_text(
        json.dumps({"gelaufen_am": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "anzahl_fragen": len(beispiele),
                    "gleichzeitig": GLEICHZEITIG,
                    "wiederholungen": WIEDERHOLUNGEN,
                    "teil1_denken": teil1,
                    "teil2_programm": teil2,
                    "teil3_vorschlagsmodell": teil3}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nBeleg geschrieben: {BELEG}")


if __name__ == "__main__":
    main()
