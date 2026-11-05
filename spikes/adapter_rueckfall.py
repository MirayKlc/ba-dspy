"""Vortest: was geschieht beim Rueckfall des ChatAdapters auf den JSONAdapter?

DSPy faellt bei einem Formatfehler von sich aus auf den JSONAdapter zurueck. In einem Lauf
ueber fuenf Beispiele ist das bei `moonshotai/kimi-k2` einmal eingetreten und mit HTTP 400
abgebrochen. Ob der Rueckfall eintritt, haengt vom zufaelligen Auftreten eines Formatfehlers
ab, die Ursache des Abbruchs aber nicht: das Skript ruft beide Adapter direkt auf und zeigt,
dass der JSONAdapter bei Kimi K2 *immer* scheitert, waehrend der ChatAdapter die Antwort
verarbeitet. Gemessen wird deshalb mit dem ChatAdapter ohne JSON-Rueckfall.

Aufruf:  python spikes/adapter_rueckfall.py
Kosten:  wenige Zehntelcent.
"""

import pathlib
import sys

from dotenv import load_dotenv

load_dotenv()

import dspy  # noqa: E402  (erst nach load_dotenv, damit der Schluessel steht)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from dspy_optimizer.datensatz import lade_aufteilung  # noqa: E402
from dspy_optimizer.kandidatenmodelle import DIE_VIER, lm_fuer  # noqa: E402
from dspy_optimizer.programm import RelevantesHerausziehen, als_text  # noqa: E402

ADAPTER = {
    "ChatAdapter": lambda: dspy.ChatAdapter(use_json_adapter_fallback=False),
    "JSONAdapter": dspy.JSONAdapter,
}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    beispiel = lade_aufteilung().trainingsdaten[0]

    for kandidatenmodell in DIE_VIER:
        for name, baue_adapter in ADAPTER.items():
            lm = lm_fuer(kandidatenmodell)
            try:
                with dspy.context(lm=lm, adapter=baue_adapter()):
                    vorhersage = dspy.Predict(RelevantesHerausziehen)(
                        question=beispiel.question, paragraphs=als_text(beispiel.paragraphs)
                    )
                ausgang = f"ja    {vorhersage.relevant_titles}"
            except Exception as fehler:  # noqa: BLE001 - *woran* es scheitert, ist der Befund
                ausgang = f"NEIN  {type(fehler).__name__}: {fehler}"[:160]
            print(f"{kandidatenmodell.name:<26}{name:<14}{ausgang}")


if __name__ == "__main__":
    main()
