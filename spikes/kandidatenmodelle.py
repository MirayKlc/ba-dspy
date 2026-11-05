"""Machbarkeitspruefung OpenRouter, Teil B: Suche nach Kandidatenmodellen.

Fragt die oeffentliche OpenRouter-Modellliste ab und sucht Modelle, die
a) von genau einem Anbieter bedient werden und
b) in der guenstigen Preisklasse liegen.

Verursacht keine Kosten: beide Endpunkte sind reine Leseabfragen ohne LM-Aufruf.
Das vollstaendige Ergebnis wird als Beleg nach docs/spikes/kandidatenmodelle.json
geschrieben; die Auswahl daraus erfolgt von Hand.
"""

import datetime
import json
import pathlib
import sys
import time

import requests

BASIS = "https://openrouter.ai/api/v1"

BELEG = pathlib.Path(__file__).resolve().parent.parent / "docs" / "spikes" / "kandidatenmodelle.json"

# Obergrenze der "guenstigen Preisklasse", in US-Dollar je 1 Mio. Tokens.
# Die Versuchsparameter verlangen die guenstige Preisklasse, nennen aber keine Zahl; die
# beiden Schwellen sind eine eigene Festlegung fuer diese Suche.
MAX_PREIS_PROMPT = 1.00
MAX_PREIS_ANTWORT = 3.00


def hole(pfad):
    antwort = requests.get(f"{BASIS}{pfad}", timeout=30)
    antwort.raise_for_status()
    return antwort.json()["data"]


def preis_pro_million(modell, feld):
    """OpenRouter gibt Preise als Dollar je *einzelnem* Token an."""
    roh = modell.get("pricing", {}).get(feld)
    return float(roh) * 1_000_000 if roh is not None else None


def ist_guenstiger_textchat(modell):
    architektur = modell.get("architecture", {})
    if "text" not in (architektur.get("input_modalities") or []):
        return False
    if "text" not in (architektur.get("output_modalities") or []):
        return False
    prompt = preis_pro_million(modell, "prompt")
    antwort = preis_pro_million(modell, "completion")
    if prompt is None or antwort is None:
        return False
    # Kostenlose Modelle entfallen: ohne Preis lassen sich keine Kosten messen.
    if prompt == 0 or antwort == 0:
        return False
    return prompt <= MAX_PREIS_PROMPT and antwort <= MAX_PREIS_ANTWORT


def veroeffentlicht(modell):
    """Erscheinungsdatum als YYYY-MM-DD; OpenRouter liefert es als Unix-Zeit."""
    roh = modell.get("created")
    return datetime.date.fromtimestamp(roh).isoformat() if roh else None


def endpunkte_von(modell_id):
    """Die Endpunkte, ueber die dieses Modell auf OpenRouter bedient wird.

    Ein Anbieter kann mehrere Endpunkte haben (z. B. Azure und Azure/Schweden),
    deshalb werden Endpunkte und Anbieter getrennt gezaehlt.
    """
    endpunkte = hole(f"/models/{modell_id}/endpoints")["endpoints"]
    return [
        {
            "anbieter": e.get("provider_name"),
            "kuerzel": e.get("tag"),
            "quantisierung": e.get("quantization"),
            "kontext": e.get("context_length"),
            "status": e.get("status"),
        }
        for e in endpunkte
    ]


def main():
    alle = hole("/models")
    guenstig = [m for m in alle if ist_guenstiger_textchat(m)]
    print(f"{len(alle)} Modelle insgesamt, {len(guenstig)} davon guenstig und rein textbasiert",
          file=sys.stderr)

    ergebnis = []
    for i, modell in enumerate(guenstig, start=1):
        modell_id = modell["id"]
        try:
            endpunkte = endpunkte_von(modell_id)
        except requests.HTTPError as fehler:
            print(f"  uebersprungen: {modell_id} ({fehler})", file=sys.stderr)
            continue
        ergebnis.append({
            "id": modell_id,
            "name": modell.get("name"),
            "veroeffentlicht": veroeffentlicht(modell),
            "preis_prompt": preis_pro_million(modell, "prompt"),
            "preis_antwort": preis_pro_million(modell, "completion"),
            "kontext": modell.get("context_length"),
            "anzahl_endpunkte": len(endpunkte),
            "anzahl_anbieter": len({e["anbieter"] for e in endpunkte}),
            "endpunkte": endpunkte,
        })
        print(f"  [{i}/{len(guenstig)}] {modell_id}: {len(endpunkte)} Endpunkte, "
              f"{ergebnis[-1]['anzahl_anbieter']} Anbieter", file=sys.stderr)
        time.sleep(0.2)

    ein_anbieter = sorted((m for m in ergebnis if m["anzahl_anbieter"] == 1),
                          key=lambda m: m["veroeffentlicht"] or "", reverse=True)
    ein_endpunkt = [m for m in ein_anbieter if m["anzahl_endpunkte"] == 1]
    beleg = {
        "abgefragt_am": time.strftime("%Y-%m-%d %H:%M:%S"),
        "schwellen_usd_je_million": {"prompt": MAX_PREIS_PROMPT, "antwort": MAX_PREIS_ANTWORT},
        "anzahl_guenstig": len(ergebnis),
        "anzahl_ein_anbieter": len(ein_anbieter),
        "anzahl_ein_endpunkt": len(ein_endpunkt),
        "ein_anbieter": ein_anbieter,
        "alle": ergebnis,
    }
    BELEG.parent.mkdir(parents=True, exist_ok=True)
    BELEG.write_text(json.dumps(beleg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(ergebnis)} Modelle in der guenstigen Klasse, davon {len(ein_anbieter)} von genau "
          f"einem Anbieter und {len(ein_endpunkt)} ueber genau einen Endpunkt.")
    print(f"Beleg geschrieben: {BELEG}")


if __name__ == "__main__":
    main()
