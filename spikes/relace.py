"""Vortest (10.10.2026): eignet sich Relace als Endpunkt fuer DeepSeek V4.1 Flash?

In der Vorstudie scheiterte der Optimierungslauf von DeepSeek V4.1 Flash mit 40 Trials zweimal
an Anbieterfehlern: HTTP 504 von OpenInference (`open-inference/fp4`), in kurzen Aussetzern.
Laut OpenRouter hatte Relace vom 07. bis 10.10. 100 % Uptime (OpenInference 91,4 %). Vor der
Entscheidung ueber einen Wechsel werden drei Fragen geprueft:

1. **Denken:** Laesst es sich bei Relace abschalten? "Unterstuetzt" heisst nicht
   "abschaltbar". Je Variante (nichts /
   abgeschaltet / `effort: low`) ein Aufruf.
2. **Das echte Programm unter Last:** MultiHopQA ueber 48 Trainingsfragen, acht gleichzeitig
   wie im Experiment, mit genau dessen LM-Einstellungen (`lm_fuer`) - einmal bei
   Relace, einmal bei OpenInference zum Vergleich. Gemessen: beantwortet, Fehler nach Art,
   F1, Dollar je Programmlauf, Denk-Tokens, abgeschnittene Antworten, Latenz, bedienender
   Anbieter.
3. **Precision:** Was OpenRouter ueber die Quantisierung des Endpunkts angibt (kostenlos,
   aus der Endpunkt-Liste).

Ergaenzt, weil der erste Aufruf in Teil 1 mit HTTP 429 scheiterte: **Teil 4** im Beleg ist
Teil 2 noch einmal auf den naechsten 48 Trainingsfragen (`trainingsdaten[48:96]`), mit
`WIEDERHOLUNGEN = 0` - er zaehlt die Fehler, die die Wiederholungen sonst verdecken.

Verwendet ausschliesslich Trainingsdaten. Schreibt nicht ins Protokoll, da es kein Messlauf ist.

Aufruf:  python spikes/relace.py
Kosten:  rund 2 Cent.
"""

import collections
import concurrent.futures
import json
import pathlib
import statistics
import sys
import time
import urllib.request

from dotenv import load_dotenv

load_dotenv()

import dspy  # noqa: E402  (erst nach load_dotenv, damit der Schluessel steht)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from dspy_optimizer.datensatz import lade_aufteilung  # noqa: E402
from dspy_optimizer.kandidatenmodelle import (  # noqa: E402
    WIEDERHOLUNGEN,
    kandidatenmodell_mit_namen,
    lm_fuer,
    mess_adapter,
)
from dspy_optimizer.metrik import bewertung_von  # noqa: E402
from dspy_optimizer.programm import MultiHopQA  # noqa: E402

BELEG = pathlib.Path(__file__).resolve().parent.parent / "docs" / "spikes" / "relace.json"
ENDPUNKTE_URL = "https://openrouter.ai/api/v1/models/deepseek/deepseek-v4.1-flash/endpoints"

ANZAHL_FRAGEN = 48
GLEICHZEITIG = 8

DEEPSEEK = kandidatenmodell_mit_namen("deepseek/deepseek-v4.1-flash")
RELACE = DEEPSEEK._replace(anbieter="relace")
OPENINFERENCE = DEEPSEEK._replace(anbieter="open-inference/fp4")  # bis 10.10.2026 festgenagelt

VARIANTEN = {
    "nichts": None,
    "enabled_false": {"enabled": False},
    "effort_low": {"effort": "low"},
}

FRAGE = (
    "Paragraphs: A: The Eiffel Tower is in Paris. B: Paris is the capital of France. "
    "Question: In which country is the Eiffel Tower? Answer with a short phrase."
)


def denk_tokens(eintrag):
    verbrauch = getattr(eintrag["response"], "usage", None)
    einzelheiten = getattr(verbrauch, "completion_tokens_details", None)
    return getattr(einzelheiten, "reasoning_tokens", None)


def abgeschnitten(eintrag):
    return eintrag["response"].choices[0].finish_reason == "length"


def anbieter(eintrag):
    return str(getattr(eintrag["response"], "provider", None))


# --- Teil 1 ------------------------------------------------------------------------------

def denken(reasoning):
    koerper = {"provider": {"only": [RELACE.anbieter], "allow_fallbacks": False}}
    if reasoning is not None:
        koerper["reasoning"] = reasoning
    lm = dspy.LM(f"openrouter/{RELACE.name}", cache=False, max_tokens=1000, temperature=0.0,
                 num_retries=0, extra_body=koerper)
    try:
        antwort = lm(FRAGE)
    except Exception as fehler:  # noqa: BLE001 - *dass* und *woran* es scheitert, ist der Befund
        return {"angenommen": False, "fehler": f"{type(fehler).__name__}: {fehler}"[:300]}
    eintrag = lm.history[-1]
    return {
        "angenommen": True,
        "antwort": str(antwort[0] if antwort else "")[:80],
        "denk_tokens": denk_tokens(eintrag),
        "antwort_tokens": eintrag["response"].usage.completion_tokens,
        "kosten_usd": eintrag["cost"],
        "anbieter": anbieter(eintrag),
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
    start = time.monotonic()
    try:
        with dspy.context(lm=lm, adapter=mess_adapter()):
            vorhersage = programm(**beispiel.inputs())
    except Exception as ausnahme:  # noqa: BLE001 - im Vortest zaehlt, woran es scheitert
        return None, fehlerart(ausnahme), time.monotonic() - start
    return bewertung_von(beispiel, vorhersage), None, time.monotonic() - start


def programm_unter_last(modell, beispiele):
    # Genau die LM-Einstellungen des Experiments, nur ohne die mitgezaehlte History-Klasse.
    lm = lm_fuer(modell, klasse=dspy.LM)
    programm = MultiHopQA()

    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=GLEICHZEITIG) as pool:
        ausgang = list(pool.map(lambda b: ein_versuch(programm, lm, b), beispiele))
    dauer = time.monotonic() - start

    bewertungen = [b for (b, _, _) in ausgang if b is not None]
    f1 = [b.f1 for b in bewertungen]
    latenzen = [s for (b, _, s) in ausgang if b is not None]
    fehler = collections.Counter(art for (_, art, _) in ausgang if art)
    kosten = sum(eintrag["cost"] or 0.0 for eintrag in lm.history)
    denk = [denk_tokens(e) or 0 for e in lm.history]
    return {
        "modell": modell.name,
        "endpunkt": modell.anbieter,
        "reasoning": modell.denken,
        "bedient_haben": dict(collections.Counter(anbieter(e) for e in lm.history)),
        "beantwortet": len(bewertungen),
        "gestellt": len(beispiele),
        "fehler": dict(fehler),
        "f1": statistics.fmean(f1) if f1 else 0.0,
        "f1_standardfehler": (statistics.stdev(f1) / len(f1) ** 0.5) if len(f1) > 1 else None,
        "lm_aufrufe": len(lm.history),
        "kosten_usd": kosten,
        "kosten_je_programmlauf_usd": kosten / len(beispiele),
        "denk_tokens_je_aufruf": statistics.fmean(denk) if denk else None,
        "denk_tokens_max": max(denk) if denk else None,
        "abgeschnitten": sum(abgeschnitten(e) for e in lm.history),
        "latenz_median_s": statistics.median(latenzen) if latenzen else None,
        "dauer_s": dauer,
    }


# --- Teil 3 ------------------------------------------------------------------------------

def endpunkte():
    with urllib.request.urlopen(ENDPUNKTE_URL, timeout=30) as antwort:
        daten = json.load(antwort)["data"]["endpoints"]
    return [
        {
            "tag": e["tag"],
            "quantization": e.get("quantization"),
            "preis_prompt_je_mio": float(e["pricing"]["prompt"]) * 1e6,
            "preis_antwort_je_mio": float(e["pricing"]["completion"]) * 1e6,
            "uptime_last_1d": e.get("uptime_last_1d"),
            "supported_parameters": e.get("supported_parameters"),
        }
        for e in daten
        if e["tag"] in (RELACE.anbieter, OPENINFERENCE.anbieter)
    ]


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("Teil 1 - Denken bei Relace\n")
    teil1 = {}
    for name, reasoning in VARIANTEN.items():
        ergebnis = denken(reasoning)
        teil1[name] = ergebnis
        zusatz = (f"denk={ergebnis['denk_tokens']} antwort={ergebnis['antwort_tokens']} "
                  f"{ergebnis['anbieter']}" if ergebnis["angenommen"]
                  else ergebnis["fehler"][:120])
        print(f"  {name:<15}{'ja ' if ergebnis['angenommen'] else 'NEIN'}  {zusatz}")

    beispiele = lade_aufteilung().trainingsdaten[:ANZAHL_FRAGEN]
    print(f"\nTeil 2 - MultiHopQA: {len(beispiele)} Fragen, {GLEICHZEITIG} gleichzeitig, "
          f"{WIEDERHOLUNGEN} Wiederholungen\n")
    teil2 = []
    for modell in (RELACE, OPENINFERENCE):
        zeile = programm_unter_last(modell, beispiele)
        teil2.append(zeile)
        print(f"  {modell.anbieter:<20}{zeile['beantwortet']}/{zeile['gestellt']}  "
              f"F1 {zeile['f1']:.3f}  ${zeile['kosten_je_programmlauf_usd']:.6f}/Programmlauf  "
              f"denk/Aufruf {zeile['denk_tokens_je_aufruf'] or 0:.0f}  "
              f"abgeschnitten {zeile['abgeschnitten']}  Median {zeile['latenz_median_s']:.1f} s  "
              f"{zeile['fehler'] or ''} {zeile['bedient_haben']}")

    print("\nTeil 3 - Angaben von OpenRouter\n")
    teil3 = endpunkte()
    for e in teil3:
        print(f"  {e['tag']:<20}quantization={e['quantization']}  "
              f"{e['preis_prompt_je_mio']:.3f} / {e['preis_antwort_je_mio']:.3f} $ je Mio.  "
              f"uptime 1d {e['uptime_last_1d']}")

    BELEG.write_text(
        json.dumps({"gelaufen_am": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "anzahl_fragen": len(beispiele),
                    "gleichzeitig": GLEICHZEITIG,
                    "wiederholungen": WIEDERHOLUNGEN,
                    "teil1_denken_relace": teil1,
                    "teil2_programm": teil2,
                    "teil3_openrouter_angaben": teil3}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nBeleg geschrieben: {BELEG}")


if __name__ == "__main__":
    main()
