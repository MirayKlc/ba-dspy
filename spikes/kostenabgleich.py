"""Vortest: Abgleich der eigenen Kostensumme mit der OpenRouter-Abrechnung.

Die Kostenerfassung (`dspy_optimizer/kostenerfassung.py`) summiert die Dollar-Betraege, die
LiteLLM je Aufruf in die History schreibt. Dieses Skript prueft, ob diese Summe stimmt, und
zwar an dem massgeblichen Fall: einer **parallelen** Auswertung ueber 20 Beispiele, bei der
die Aufrufe in Arbeits-Threads laufen und der naheliegende Weg zu wenig messen wuerde.

Gegengeprueft wird doppelt:

1. je Aufruf gegen `/api/v1/generation` - dieselbe Zahl, die auf der Abrechnung landet;
2. ueber den Kontostand (`/api/v1/credits`) vor und nach dem Lauf. Der hinkt bis zu einer
   Minute hinterher und wird deshalb abgewartet, bis er sich nicht mehr bewegt.

Nebenbei ergibt sich die erste gemessene Zahl fuer die Kosten je Programmlauf. Die Zeile im
fortlaufenden Protokoll traegt den Zweck `vorstudie` und keinen Abschnitt: bewertet werden
**Trainingsdaten**, eine Nachbewertung laeuft dagegen ueber die Auswahldaten. Ohne
Abschnitt Optimierung kalibriert die Zeile auch nichts. Die Zeile vom 20.09.2026 traegt noch
das fruehere Feld `art: vorstudie`.

Aufruf:  python spikes/kostenabgleich.py [Kandidatenmodell]
Kosten:  rund einen halben Cent. Ausschliesslich Trainingsdaten.
"""

import datetime
import json
import os
import pathlib
import sys
import time

import requests
from dotenv import load_dotenv

load_dotenv()

import dspy  # noqa: E402  (erst nach load_dotenv, damit der Schluessel steht)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from dspy_optimizer.datensatz import lade_aufteilung  # noqa: E402
from dspy_optimizer.kandidatenmodelle import (  # noqa: E402
    DIE_VIER,
    kandidatenmodell_mit_namen,
    messkontext,
)
from dspy_optimizer.kostenerfassung import messhygiene, messlauf  # noqa: E402
from dspy_optimizer.metrik import bewerte  # noqa: E402
from dspy_optimizer.programm import ZweistufigesProgramm  # noqa: E402
from dspy_optimizer.protokoll import (  # noqa: E402
    PROTOKOLLDATEI,
    VORSTUDIE,
    verzeichnis_fuer_lauf,
)

BELEG = pathlib.Path(__file__).resolve().parent.parent / "docs" / "spikes" / "kostenabgleich.json"

# Umfang der parallelen Auswertung fuer den Abgleich.
ANZAHL_BEISPIELE = 20
# Acht Threads sind DSPys Voreinstellung und damit das, was MIPROv2 spaeter benutzt.
THREADS = 8

KOPF = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}


def abgerechnet():
    """Was OpenRouter fuer dieses Konto bisher abgerechnet hat, in US-Dollar."""
    antwort = requests.get("https://openrouter.ai/api/v1/credits", headers=KOPF, timeout=30)
    antwort.raise_for_status()
    return float(antwort.json()["data"]["total_usage"])


def abgerechnet_stabil(ruhe=3, takt=10, geduld=240):
    """Wie `abgerechnet`, wartet aber, bis sich der Wert nicht mehr bewegt.

    Die Kontoabrechnung haengt den Aufrufen hinterher; ohne Warten misst man zu wenig
    (siehe `spikes/openrouter_spike.py`, Frage 4).
    """
    letzter, gleich, start = None, 0, time.time()
    while time.time() - start < geduld:
        jetzt = abgerechnet()
        gleich = gleich + 1 if jetzt == letzter else 0
        letzter = jetzt
        if gleich >= ruhe:
            return jetzt
        time.sleep(takt)
    return letzter


def einzelabrechnung(aufruf_id, geduld=120):
    """Der von OpenRouter je Aufruf abgerechnete Betrag, aus `/api/v1/generation`."""
    start = time.time()
    while time.time() - start < geduld:
        antwort = requests.get(
            "https://openrouter.ai/api/v1/generation",
            params={"id": aufruf_id},
            headers=KOPF,
            timeout=30,
        )
        if antwort.status_code == 200:
            return float(antwort.json()["data"]["total_cost"])
        time.sleep(5)
    return None


def abweichung(eigene_summe, vergleich):
    """Absolut und in Prozent - und `None`, wenn es nichts zu vergleichen gibt."""
    if vergleich is None:
        return None
    absolut = eigene_summe - vergleich
    return {
        "eigene_summe_usd": round(eigene_summe, 8),
        "vergleich_usd": round(vergleich, 8),
        "abweichung_usd": round(absolut, 8),
        "abweichung_prozent": round(100 * absolut / vergleich, 4) if vergleich else None,
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    gewaehlt = kandidatenmodell_mit_namen(sys.argv[1]) if len(sys.argv) > 1 else DIE_VIER[0]
    beispiele = lade_aufteilung().trainingsdaten[:ANZAHL_BEISPIELE]

    print(f"{gewaehlt.name} (festgenagelt auf {gewaehlt.anbieter}), "
          f"{len(beispiele)} Trainingsbeispiele, {THREADS} Threads")

    messhygiene()
    vorher = abgerechnet_stabil()
    print(f"Kontostand vorher: {vorher:.6f} $")

    verzeichnis = verzeichnis_fuer_lauf(f"kostenabgleich-{gewaehlt.name}")
    with messkontext(gewaehlt) as lm:
        with messlauf(
            lm,
            protokollverzeichnis=verzeichnis,
            abschnitt=None,  # Trainingsdaten: keine Nachbewertung, daher kein Abschnitt
            einordnung=VORSTUDIE,
            konfiguration={
                "kandidatenmodell": gewaehlt.name,
                "anbieter": gewaehlt.anbieter,
                "beispiele": len(beispiele),
                "teil": "trainingsdaten",
                "threads": THREADS,
                "programm": "ZweistufigesProgramm (unoptimiert)",
                "anlass": "Kostenabgleich Issue #5",
            },
        ) as erfassung:
            auswertung = dspy.Evaluate(
                devset=beispiele, metric=bewerte, num_threads=THREADS, display_progress=True
            )
            ergebnis = auswertung(ZweistufigesProgramm())
            # `Evaluate` gibt Prozent zurueck; im Protokoll steht der F1 wie ueberall als Anteil.
            erfassung.vermerke_bewertung(ergebnis.score / 100)
    # Die Protokollzeile steht jetzt schon: `messlauf` schreibt sie beim Verlassen.
    messwerte = erfassung.messwerte()

    # Die Aufruf-Kennungen stehen nur in der History; fuer 40 Aufrufe ist das ueberschaubar.
    aufrufe = [
        {
            "id": getattr(eintrag["response"], "id", None),
            "kosten_usd": eintrag["cost"],
            "anbieter": getattr(eintrag["response"], "provider", None),
            "prompt_tokens": eintrag["usage"].get("prompt_tokens"),
            "completion_tokens": eintrag["usage"].get("completion_tokens"),
        }
        for eintrag in lm.history
    ]
    print(f"{len(aufrufe)} Aufrufe, eigene Summe {messwerte.dollar:.6f} $ - "
          "hole jetzt die Einzelabrechnungen")

    for aufruf in aufrufe:
        aufruf["abgerechnet_usd"] = einzelabrechnung(aufruf["id"]) if aufruf["id"] else None
    abgerechnete = [aufruf for aufruf in aufrufe if aufruf["abgerechnet_usd"] is not None]
    je_aufruf = (
        sum(aufruf["abgerechnet_usd"] for aufruf in abgerechnete) if abgerechnete else None
    )

    nachher = abgerechnet_stabil()
    print(f"Kontostand nachher: {nachher:.6f} $")

    beleg = {
        "kandidatenmodell": gewaehlt.name,
        "anbieter": gewaehlt.anbieter,
        "beispiele": len(beispiele),
        "threads": THREADS,
        "zeitpunkt": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "eigene_erfassung": {**messwerte.als_felder(), "bewertung_f1": ergebnis.score / 100},
        "gegenprobe_je_aufruf": abweichung(messwerte.dollar, je_aufruf),
        "gegenprobe_kontostand": abweichung(messwerte.dollar, nachher - vorher),
        "kontostand": {"vorher_usd": vorher, "nachher_usd": nachher},
        "aufrufe": aufrufe,
    }
    BELEG.parent.mkdir(parents=True, exist_ok=True)
    BELEG.write_text(json.dumps(beleg, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print(json.dumps(
        {schluessel: beleg[schluessel]
         for schluessel in ("eigene_erfassung", "gegenprobe_je_aufruf", "gegenprobe_kontostand")},
        indent=2, ensure_ascii=False,
    ))
    print(f"\nBeleg in {BELEG}, Protokollzeile in {PROTOKOLLDATEI}")


if __name__ == "__main__":
    main()
