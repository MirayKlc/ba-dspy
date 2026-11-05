"""Vortest: eignet sich ein Endpunkt fuer DeepSeek V4.1 Flash?

OpenInference (`open-inference/fp4`) ist bei OpenRouter seit dem 10.10.2026 gestoert; DeepSeek
V4.1 Flash braucht einen neuen Endpunkt. Getestet werden die auf der Kommandozeile genannten
Endpunkte aus der Endpunktliste von OpenRouter, je Endpunkt mit denselben Fragen wie in
`spikes/relace.py`:

1. **Denken:** laesst es sich abschalten?
   Je Variante ein Aufruf.
2. **Das echte Programm unter Last:** MultiHopQA ueber 96 Trainingsfragen, acht gleichzeitig
   wie im Experiment, mit dessen LM-Einstellungen (`lm_fuer`), aber **ohne Wiederholungen**,
   damit jeder Fehler des Anbieters sichtbar wird statt verdeckt.
3. **Angaben von OpenRouter:** Precision, Preise, Rabatt, Erreichbarkeit.

Verwendet ausschliesslich Trainingsdaten. Schreibt nicht ins Protokoll, da es kein Messlauf ist.

Aufruf:  python spikes/endpunkt_deepseek.py <endpunkt> [...]   etwa inference-net/fp8
Kosten:  rund 2 Cent je Endpunkt.
"""

import json
import pathlib
import sys
import time
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import relace  # noqa: E402  (laedt .env; Messfunktionen aus dem Relace-Test)

from dspy_optimizer import kandidatenmodelle  # noqa: E402
from dspy_optimizer.datensatz import lade_aufteilung  # noqa: E402

BELEGE = pathlib.Path(__file__).resolve().parent.parent / "docs" / "spikes"
ANZAHL_FRAGEN = 96


def angaben(tag):
    with urllib.request.urlopen(relace.ENDPUNKTE_URL, timeout=30) as antwort:
        daten = json.load(antwort)["data"]["endpoints"]
    for e in daten:
        if e["tag"] == tag:
            return {
                "anbieter": e["provider_name"],
                "quantization": e.get("quantization"),
                "preis_prompt_je_mio": float(e["pricing"]["prompt"]) * 1e6,
                "preis_antwort_je_mio": float(e["pricing"]["completion"]) * 1e6,
                "rabatt": e["pricing"].get("discount"),
                "uptime_last_1d": e.get("uptime_last_1d"),
                "status": e.get("status"),
            }
    raise KeyError(f"{tag!r} fuehrt OpenRouter fuer DeepSeek V4.1 Flash nicht")


def teste(tag, beispiele):
    modell = relace.DEEPSEEK._replace(anbieter=tag)
    relace.RELACE = modell  # `relace.denken` fragt den Endpunkt in `RELACE` an
    print(f"\n=== {tag}")
    beleg = {"endpunkt": tag, "openrouter_angaben": angaben(tag), "denken": {}}
    print(f"  {beleg['openrouter_angaben']}")
    for name, reasoning in relace.VARIANTEN.items():
        ergebnis = relace.denken(reasoning)
        beleg["denken"][name] = ergebnis
        zusatz = (f"denk={ergebnis['denk_tokens']} antwort={ergebnis['antwort_tokens']} "
                  f"{ergebnis['anbieter']}" if ergebnis["angenommen"]
                  else ergebnis["fehler"][:120])
        print(f"  Denken {name:<15}{'ja ' if ergebnis['angenommen'] else 'NEIN'}  {zusatz}")

    kandidatenmodelle.WIEDERHOLUNGEN = 0  # `lm_fuer` liest den Wert beim Aufruf
    zeile = relace.programm_unter_last(modell, beispiele)
    beleg["programm_ohne_wiederholungen"] = zeile
    print(f"  Programm {zeile['beantwortet']}/{zeile['gestellt']}  F1 {zeile['f1']:.3f}  "
          f"${zeile['kosten_je_programmlauf_usd']:.6f}/Programmlauf  "
          f"denk/Aufruf {zeile['denk_tokens_je_aufruf'] or 0:.0f}  "
          f"abgeschnitten {zeile['abgeschnitten']}  Median {zeile['latenz_median_s'] or 0:.1f} s  "
          f"Dauer {zeile['dauer_s']:.0f} s  {zeile['fehler'] or ''} {zeile['bedient_haben']}")
    return beleg


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    beispiele = lade_aufteilung().trainingsdaten[:ANZAHL_FRAGEN]
    for tag in sys.argv[1:]:
        beleg = teste(tag, beispiele)
        beleg.update(gelaufen_am=time.strftime("%Y-%m-%d %H:%M:%S"),
                     anzahl_fragen=len(beispiele), gleichzeitig=relace.GLEICHZEITIG,
                     wiederholungen=0)
        datei = BELEGE / f"endpunkt-deepseek-{tag.replace('/', '-')}.json"
        datei.write_text(json.dumps(beleg, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Beleg geschrieben: {datei}")


if __name__ == "__main__":
    main()
