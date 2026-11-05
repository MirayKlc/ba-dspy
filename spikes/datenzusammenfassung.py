"""Vortest: warum schreibt das Vorschlagsmodell beim Zusammenfassen der Trainingsdaten
bis an das Limit von 4000 Tokens?

Im ersten Optimierungslauf der Vorstudie (GPT-6 Luna, 10 Trials) endeten zwei Antworten von
qwen3.7-flash am `max_tokens`-Limit, beide vor dem Vorschlagen der Instruktionen, also in
DSPys Datenzusammenfassung (`propose/dataset_summary_generator.py`). Den Wortlaut speichert das
Protokoll nicht. Dieses Skript ruft genau diese DSPy-Funktion noch einmal auf - dieselben 200
Trainingsdaten, dasselbe Sprachmodell mit demselben Limit, derselbe Adapter, `view_data_batch_size`
10 wie in MIPROv2 - und haelt je Aufruf fest: Laenge, ob abgeschnitten, und ob sich der Text
wiederholt. Die Frage: Schleife (ein hoeheres Limit hilft nicht) oder viel sinnvoller Text
(ein hoeheres Limit hilft)?

Verwendet nur Trainingsdaten und schreibt nicht ins Protokoll, da es kein Messlauf der
Vorstudie ist.

Aufruf:  python spikes/datenzusammenfassung.py [Wiederholungen]
Kosten:  wenige Cent.
"""

import json
import pathlib
import sys

import dspy
from dotenv import load_dotenv
from dspy.propose.dataset_summary_generator import create_dataset_summary

from dspy_optimizer.datensatz import lade_aufteilung
from dspy_optimizer.kandidatenmodelle import DAS_VORSCHLAGSMODELL, lm_fuer, mess_adapter

ZIEL = pathlib.Path(__file__).resolve().parent.parent / "docs" / "spikes" / "datenzusammenfassung.json"


def wiederholung(text):
    """Anteil der Zeilen, die schon einmal vorkamen - nahe 1 heisst Schleife."""
    zeilen = [zeile.strip() for zeile in text.splitlines() if zeile.strip()]
    if not zeilen:
        return None
    return round(1 - len(set(zeilen)) / len(zeilen), 3)


def aufruf(eintrag):
    antwort = eintrag["response"]
    text = antwort.choices[0].message.content or ""
    verbrauch = getattr(antwort, "usage", None)
    return {
        "signatur": eintrag.get("messages", [{}])[0].get("content", "")[:80],
        "abgeschnitten": antwort.choices[0].finish_reason == "length",
        "tokens_eingabe": getattr(verbrauch, "prompt_tokens", None),
        "tokens_ausgabe": getattr(verbrauch, "completion_tokens", None),
        "zeichen": len(text),
        "wiederholte_zeilen": wiederholung(text),
        "anfang": text[:600],
        "ende": text[-600:],
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv()
    wiederholungen = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    trainingsdaten = lade_aufteilung().trainingsdaten
    ergebnis = []
    for nummer in range(1, wiederholungen + 1):
        lm = lm_fuer(DAS_VORSCHLAGSMODELL)
        with dspy.context(adapter=mess_adapter()):
            zusammenfassung = create_dataset_summary(
                trainset=trainingsdaten, view_data_batch_size=10, prompt_model=lm, verbose=True
            )
        aufrufe = [aufruf(eintrag) for eintrag in lm.history]
        dollar = sum(eintrag.get("cost") or 0.0 for eintrag in lm.history)
        ergebnis.append(
            {"durchgang": nummer, "zusammenfassung": zusammenfassung, "dollar": dollar,
             "aufrufe": aufrufe}
        )
        print(f"\nDurchgang {nummer}: {len(aufrufe)} Aufrufe, {dollar:.4f} $")
        for i, a in enumerate(aufrufe):
            print(f"  {i}: {a['tokens_ausgabe']} Tokens, abgeschnitten={a['abgeschnitten']}, "
                  f"wiederholte Zeilen={a['wiederholte_zeilen']}")
    ZIEL.write_text(json.dumps(ergebnis, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nGeschrieben: {ZIEL}")


if __name__ == "__main__":
    main()
