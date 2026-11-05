"""Machbarkeitspruefung OpenRouter, Teil A.

Prueft vier Annahmen zur Anbindung der Sprachmodelle ueber OpenRouter gegen die echte
Schnittstelle:

1. Kommt bei einem Aufruf ueber OpenRouter eine Kostenangabe in US-Dollar an?
2. Welcher Anbieter hat bedient, und laesst sich das je Aufruf mitschreiben?
3. Geht eine Provider-Vorgabe durch? Zwei Wege werden probiert:
   (a) `extra_body={"provider": ...}`  (b) `provider=...` als gewoehnliches Schluesselwort.
4. Stimmt eine selbst gezaehlte Summe mit dem ueberein, was OpenRouter abgerechnet hat?

Teil 5 misst zusaetzlich die Token-Grundlast je Aufruf, die zwischen Anbietern stark abweicht.

Das Skript schreibt sein Protokoll nach docs/spikes/spike-protokoll.json und gibt es
zusaetzlich aus. Es kostet echtes Geld, aber in der Groessenordnung von Bruchteilen
eines Cents (ein vollstaendiger Lauf: rund 0,0005 $).
"""

import json
import os
import pathlib
import sys
import time

import requests
from dotenv import load_dotenv

load_dotenv()

import dspy  # noqa: E402  (erst nach load_dotenv, damit der Schluessel steht)
import litellm  # noqa: E402

SCHLUESSEL = os.environ["OPENROUTER_API_KEY"]

PROTOKOLL = pathlib.Path(__file__).resolve().parent.parent / "docs" / "spikes" / "spike-protokoll.json"

# Frage 3 braucht ein Modell mit *mehreren* Anbietern, sonst gibt es nichts zu lenken.
MEHR_ANBIETER_MODELL = "openai/gpt-4o-mini"

# Die vier Kandidatenmodelle der damaligen Auswahl, je mit dem Anbieter, auf den sie
# festgenagelt sind. Ausgewaehlt in drei Vorauswahl-Runden (spikes/vorauswahl.py,
# spikes/vorauswahl2.py); spaeter ersetzt.
KANDIDATENMODELLE = [
    ("openai/gpt-oss-120b", "deepinfra"),
    ("google/gemma-4-31b-it", "novita"),
    ("meta/muse-glimmer-30b", "deepinfra"),
    ("moonshotai/kimi-k2", "novita"),
]

FRAGEN = [
    "Answer in one word: what is the capital of France?",
    "Answer in one word: which planet is known as the red planet?",
    "Answer in one word: what is the largest ocean on Earth?",
    "Answer in one word: who wrote the play Hamlet?",
    "Answer in one word: what is the chemical symbol for gold?",
]


KOPF = {"Authorization": f"Bearer {SCHLUESSEL}"}


def abgerechnet():
    """Was OpenRouter fuer dieses Konto bisher abgerechnet hat, in US-Dollar."""
    antwort = requests.get("https://openrouter.ai/api/v1/credits", headers=KOPF, timeout=30)
    antwort.raise_for_status()
    return float(antwort.json()["data"]["total_usage"])


def abgerechnet_stabil(ruhe=3, takt=10, geduld=240):
    """Wie `abgerechnet`, wartet aber, bis sich der Wert nicht mehr bewegt.

    Die Kontoabrechnung haengt den Aufrufen hinterher; ohne Warten misst man zu wenig.
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
    """Der von OpenRouter je Aufruf abgerechnete Betrag, aus /generation.

    Das ist die belastbarste Quelle: dieselbe Zahl, die auf der Abrechnung landet.
    """
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


def sprachmodell(modell, anbieter=None):
    """Ein LM ohne Cache und mit kurzer Antwort, fuer jeden Aufruf dieser Pruefung gleich.

    Ist ein Anbieter angegeben, wird er festgenagelt; OpenRouter lehnt dann eher ab, als
    stillschweigend an einen anderen Anbieter umzuleiten.
    """
    zusatz = {}
    if anbieter:
        zusatz["extra_body"] = {"provider": {"only": [anbieter], "allow_fallbacks": False}}
    return dspy.LM(f"openrouter/{modell}", cache=False, max_tokens=20, temperature=0.0, **zusatz)


def messwerte(eintrag):
    """Die drei Dinge, auf die es ankommt, aus einem dspy-History-Eintrag."""
    antwort = eintrag["response"]
    return {
        "prompt_tokens": eintrag["usage"].get("prompt_tokens"),
        "completion_tokens": eintrag["usage"].get("completion_tokens"),
        "kosten_usd": eintrag["cost"],
        "anbieter": getattr(antwort, "provider", None),
        "antwort_modell": eintrag["response_model"],
        "aufruf_id": getattr(antwort, "id", None),
    }


def frage_1_und_2():
    """Ein Aufruf, und alles was dabei zurueckkommt."""
    print("\n=== Frage 1+2: Kosten und bedienender Anbieter ===")
    lm = sprachmodell(*KANDIDATENMODELLE[0])
    ausgabe = lm(FRAGEN[0])
    werte = messwerte(lm.history[-1])
    print(f"Antwort: {ausgabe}")
    print(json.dumps(werte, indent=2, ensure_ascii=False))
    return werte


def anbieter_bei_vorgabe(weg, gewuenscht):
    """Ruft mit Provider-Vorgabe auf und meldet, wer tatsaechlich bedient hat.

    weg='extra_body'  -> {"extra_body": {"provider": ...}}
    weg='schluesselwort' -> {"provider": ...} direkt als Parameter
    """
    vorgabe = {"only": [gewuenscht], "allow_fallbacks": False}
    zusatz = {"extra_body": {"provider": vorgabe}} if weg == "extra_body" else {"provider": vorgabe}
    try:
        antwort = litellm.completion(
            model=f"openrouter/{MEHR_ANBIETER_MODELL}",
            messages=[{"role": "user", "content": FRAGEN[0]}],
            max_tokens=20,
            temperature=0.0,
            **zusatz,
        )
        return {"weg": weg, "gewuenscht": gewuenscht, "bedient_hat": getattr(antwort, "provider", None)}
    except Exception as fehler:  # noqa: BLE001 - der Fehlertext ist hier das Ergebnis
        return {"weg": weg, "gewuenscht": gewuenscht, "fehler": f"{type(fehler).__name__}: {fehler}"[:300]}


def frage_3():
    """Beide Wege der Provider-Vorgabe, je zweimal, plus eine Gegenprobe."""
    print(f"\n=== Frage 3: Provider-Vorgabe ({MEHR_ANBIETER_MODELL}) ===")
    ergebnisse = []
    for weg in ("extra_body", "schluesselwort"):
        for gewuenscht in ("openai", "azure"):
            ergebnisse.append(anbieter_bei_vorgabe(weg, gewuenscht))
            time.sleep(0.5)
    # Gegenprobe: ein Anbieter, der dieses Modell gar nicht bedient. Kommt die Vorgabe
    # an, muss OpenRouter den Aufruf ablehnen statt ihn still woanders hin zu leiten.
    ergebnisse.append(anbieter_bei_vorgabe("extra_body", "deepinfra"))
    ergebnisse.append(anbieter_bei_vorgabe("schluesselwort", "deepinfra"))

    # Derselbe Weg noch einmal ueber dspy.LM, denn diesen Weg benutzt das Experiment.
    # `provider=` ist bei dspy.LM bereits belegt, daher bleibt nur extra_body.
    lm = dspy.LM(
        f"openrouter/{MEHR_ANBIETER_MODELL}",
        cache=False,
        max_tokens=20,
        temperature=0.0,
        extra_body={"provider": {"only": ["openai"], "allow_fallbacks": False}},
    )
    lm(FRAGEN[0])
    ergebnisse.append({
        "weg": "extra_body ueber dspy.LM",
        "gewuenscht": "openai",
        "bedient_hat": messwerte(lm.history[-1])["anbieter"],
    })
    print(json.dumps(ergebnisse, indent=2, ensure_ascii=False))
    return ergebnisse


def frage_4():
    """Selbst gezaehlte Summe gegen die Abrechnung von OpenRouter halten."""
    print(f"\n=== Frage 4: {len(KANDIDATENMODELLE) * len(FRAGEN)} Aufrufe, Summe gegen Abrechnung ===")
    vorher = abgerechnet_stabil()
    protokoll = []
    for modell, anbieter in KANDIDATENMODELLE:
        lm = sprachmodell(modell, anbieter)
        for frage in FRAGEN:
            lm(frage)
            zeile = {"modell": modell, **messwerte(lm.history[-1])}
            protokoll.append(zeile)
            kosten = f"${zeile['kosten_usd']:.8f}" if zeile["kosten_usd"] is not None else "Kosten fehlen"
            print(f"  {modell:<45} {str(zeile['anbieter']):<16} {kosten}")

    nachher = abgerechnet_stabil()
    for zeile in protokoll:
        zeile["kosten_laut_abrechnung"] = einzelabrechnung(zeile["aufruf_id"])

    selbst = sum(z["kosten_usd"] or 0.0 for z in protokoll)
    je_aufruf = sum(z["kosten_laut_abrechnung"] or 0.0 for z in protokoll)
    konto = nachher - vorher
    ungleich = [z for z in protokoll if z["kosten_usd"] != z["kosten_laut_abrechnung"]]
    bericht = {
        "aufrufe": len(protokoll),
        "selbst_gezaehlt_usd": selbst,
        "laut_einzelabrechnung_usd": je_aufruf,
        "abweichung_zur_einzelabrechnung_usd": selbst - je_aufruf,
        "aufrufe_mit_abweichung": len(ungleich),
        "kontostand_vorher_usd": vorher,
        "kontostand_nachher_usd": nachher,
        "laut_kontostand_usd": konto,
        "abweichung_zum_kontostand_usd": selbst - konto,
        "abweichung_zum_kontostand_prozent": ((selbst - konto) / konto * 100) if konto else None,
        "anbieter_je_modell": {m: sorted({z["anbieter"] for z in protokoll if z["modell"] == m})
                               for m, _ in KANDIDATENMODELLE},
    }
    print(json.dumps(bericht, indent=2, ensure_ascii=False))
    return {"bericht": bericht, "aufrufe": protokoll}


def frage_5():
    """Wie viele Prompt-Tokens rechnet welcher Anbieter fuer denselben Text ab?

    Einmal mit einem Mini-Prompt, einmal mit rund 1.000 Tokens - nur so laesst sich
    eine feste Grundlast von einem Aufschlag je Token unterscheiden.
    """
    print("\n=== Frage 5: Token-Grundlast je Aufruf ===")
    lang = ("Read the following passage and answer with one word. "
            + "The quick brown fox jumps over the lazy dog near the riverbank at dawn. " * 60
            + " Question: which animal jumps?")
    ergebnisse = []
    for laenge, frage in (("kurz", FRAGEN[0]), ("lang", lang)):
        for modell, anbieter in KANDIDATENMODELLE:
            lm = sprachmodell(modell, anbieter)
            lm(frage)
            werte = messwerte(lm.history[-1])
            ergebnisse.append({"prompt_laenge": laenge, "modell": modell, **werte})
            print(f"  {laenge:<5} {modell:<45} prompt_tokens={werte['prompt_tokens']:<6} "
                  f"${werte['kosten_usd']:.8f}")
    print(json.dumps(ergebnisse, indent=2, ensure_ascii=False))
    return ergebnisse


def main():
    teile = {"1": frage_1_und_2, "3": frage_3, "4": frage_4, "5": frage_5}
    gewuenscht = sys.argv[1:] or list(teile)
    unbekannt = [t for t in gewuenscht if t not in teile]
    if unbekannt:
        sys.exit(f"Unbekannter Teil: {', '.join(unbekannt)}. Moeglich sind: {', '.join(teile)}")

    protokoll = {"gelaufen_am": time.strftime("%Y-%m-%d %H:%M:%S")}
    for t in gewuenscht:
        protokoll[f"teil_{t}"] = teile[t]()
    PROTOKOLL.parent.mkdir(parents=True, exist_ok=True)
    PROTOKOLL.write_text(json.dumps(protokoll, indent=2, ensure_ascii=False, default=str),
                         encoding="utf-8")
    print(f"\nProtokoll geschrieben: {PROTOKOLL}")


if __name__ == "__main__":
    main()
