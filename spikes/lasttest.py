"""Lasttest (03.10.2026): halten die vier Endpunkte einen echten Optimierungslauf aus?

Der Vortest zum Modellwechsel (`spikes/modellwechsel.py`) hat jedes Kandidatenmodell nur kurz
belastet: 24 Fragen, acht gleichzeitig, rund 48 Aufrufe in unter einer Minute. Einen echten
Optimierungslauf hatte nur gpt-oss-120b hinter sich (Handlauf vom 28.09.2026, 40
Validierungsbeispiele). Ein Lauf im Experiment schickt aber mehr als tausend Aufrufe ueber
Minuten, und die Limits der Anbieter sind nicht abfragbar, da sich alle OpenRouter-Nutzer
beim Anbieter ein gemeinsames Kontingent teilen. Ob ein Endpunkt unter Dauerlast drosselt,
zeigt erst der Versuch.

Deshalb je Kandidatenmodell ein Lauf ueber die Schnittstelle des Experiments
(`Optimierungslauf`) in der **Groesse des Experiments**: 200 Trainings- und 200
Validierungsbeispiele, acht gleichzeitige Anfragen, nur 2 Trials - die Last entsteht durch die
vollstaendigen Bewertungen ueber 200 Beispiele, nicht durch die Zahl der Trials. Gemessen:
Anbieterfehler und Formatfehler, Stoerungen, bedienende Anbieter, Dauer, Latenz, Kosten.

Eingeordnet als Handlauf: die Zeilen im Protokoll kalibrieren das Kostenmodell nicht.
Ausschliesslich Trainings- und Validierungsdaten.

Aufruf:  python spikes/lasttest.py [Kandidatenmodell ...]
Kosten:  grob 1 bis 1,5 $ fuer alle vier.
"""

import json
import pathlib
import sys
import time

from dotenv import load_dotenv

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from dspy_optimizer.datensatz import lade_aufteilung  # noqa: E402
from dspy_optimizer.kandidatenmodelle import DIE_VIER, kandidatenmodell_mit_namen  # noqa: E402
from dspy_optimizer.kostenerfassung import messhygiene  # noqa: E402
from dspy_optimizer.optimierungslauf import Optimierungslauf  # noqa: E402
from dspy_optimizer.programm import ZweistufigesProgramm  # noqa: E402
from dspy_optimizer.protokoll import HANDLAUF, PROTOKOLLDATEI  # noqa: E402

BELEG = pathlib.Path(__file__).resolve().parent.parent / "docs" / "spikes" / "lasttest.json"
NUM_TRIALS = 2
SEED = 0

# Was aus den beiden Protokollzeilen eines Laufs in den Beleg uebernommen wird.
FELDER = (
    "dollar", "lm_aufrufe", "programmlaeufe", "fehlgeschlagene_programmlaeufe",
    "formatfehler", "anbieterfehler", "anbieter", "stoerungen", "dauer_sekunden",
    "latenz_mittelwert_sekunden", "latenz_median_sekunden", "bewertung",
)


def neue_protokollzeilen(vorher):
    zeilen = PROTOKOLLDATEI.read_text(encoding="utf-8").splitlines()[vorher:]
    return [json.loads(zeile) for zeile in zeilen if zeile.strip()]


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv()
    messhygiene()

    modelle = [kandidatenmodell_mit_namen(n) for n in sys.argv[1:]] or list(DIE_VIER)
    aufteilung = lade_aufteilung()
    lauf = Optimierungslauf(
        trainingsdaten=aufteilung.trainingsdaten,
        validierungsdaten=aufteilung.validierungsdaten,
    )

    ergebnisse = []
    for modell in modelle:
        print(f"\n=== {modell.name} @ {modell.anbieter}", flush=True)
        vorher = len(PROTOKOLLDATEI.read_text(encoding="utf-8").splitlines())
        start = time.monotonic()
        eintrag = {"modell": modell.name, "endpunkt": modell.anbieter}
        try:
            ergebnis = lauf(modell, ZweistufigesProgramm(), NUM_TRIALS, SEED, einordnung=HANDLAUF)
            eintrag["abgeschlossen"] = True
            eintrag["ausfuehrungskosten_je_programmlauf"] = ergebnis.ausfuehrungskosten
        except Exception as fehler:  # ein Endpunkt, der ausfaellt, soll die anderen nicht mitreissen
            eintrag["abgeschlossen"] = False
            eintrag["abbruch"] = f"{type(fehler).__name__}: {fehler}"[:500]
        eintrag["wanduhr_sekunden"] = round(time.monotonic() - start, 1)
        for zeile in neue_protokollzeilen(vorher):
            eintrag[zeile["abschnitt"] or "ohne_abschnitt"] = {f: zeile.get(f) for f in FELDER}
        ergebnisse.append(eintrag)
        print(json.dumps(eintrag, ensure_ascii=False, indent=1), flush=True)

        BELEG.write_text(
            json.dumps(
                {
                    "gelaufen_am": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "num_trials": NUM_TRIALS,
                    "trainingsdaten": len(lauf.trainingsdaten),
                    "validierungsdaten": len(lauf.validierungsdaten),
                    "laeufe": ergebnisse,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
