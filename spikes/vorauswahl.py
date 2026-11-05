"""Vorauswahl der Kandidatenmodelle nach gemessener Qualitaet (Runde 1).

Die Auswahl der vier Kandidatenmodelle beruhte bis dahin nur auf Preis und Anbieterzahl. Ob
ein Modell die Aufgabe ueberhaupt loest, war nicht geprueft. Dieses Skript holt das nach: jedes
Modell aus der Vorauswahl beantwortet dieselben 20 HotPotQA-Fragen mit demselben,
*unoptimierten* Prompt. Gemessen werden Antwortqualitaet und die dabei angefallenen Dollar.

Das ist ausdruecklich **kein** Ergebnis der Arbeit, sondern ein Vorauswahlschritt:
- ein einstufiges Programm, waehrend das Experiment ein zweistufiges benutzt,
- 20 Beispiele statt 200, also mit grosser Streuung,
- ausschliesslich Trainingsdaten. Die Testdaten bleiben unberuehrt.

Aufruf:  python spikes/vorauswahl.py
Kosten:  rund 20 bis 40 Cent fuer den ganzen Lauf.
"""

import collections
import json
import pathlib
import re
import string
import sys
import time

from dotenv import load_dotenv

load_dotenv()

import dspy  # noqa: E402  (erst nach load_dotenv, damit der Schluessel steht)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from dspy_optimizer.datensatz import lade_aufteilung  # noqa: E402

ERGEBNIS = pathlib.Path(__file__).resolve().parent.parent / "docs" / "spikes" / "vorauswahl.json"

ANZAHL_FRAGEN = 20

# Grosszuegig, damit denkende Modelle nicht schon am Platz scheitern statt an der Aufgabe.
# Was sie an Denk-Tokens verbrauchen, schlaegt sich dann im gemessenen Preis nieder.
MAX_TOKENS = 1000

# Alle mit genau einem Anbieter und in der guenstigen Preisklasse
# (docs/spikes/kandidatenmodelle.json). Die unteren vier sind die bis dahin getroffene
# Auswahl und laufen als Vergleichsmassstab mit.
KANDIDATENMODELLE = [
    "qwen/qwen3.7-flash",
    "inception/mercury-2.5",
    "upstage/solar-pro4",
    "stepfun/step-3.5-flash",
    "bytedance-seed/seed-2.0-mini",
    "mistralai/ministral-8b-2512",
    "mistralai/mistral-small-2603",
    "qwen/qwen3.8-flash",
    "meituan/longcat-2.0",
    "z-ai/glm-5.3-flashx",
    "aion-labs/aion-3.0-mini",
    "tencent/hy4-preview",
    "mistralai/mistral-small-24b-instruct-2501",
    "openai/gpt-4o-mini-2024-07-18",
    "anthropic/claude-3-haiku",
    "moonshotai/kimi-k2",
]


class FrageBeantworten(dspy.Signature):
    """Answer the question using the given paragraphs. Reply with a short phrase."""

    paragraphs: str = dspy.InputField(desc="ten paragraphs, only two of them are relevant")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="a short phrase, not a sentence")


def als_text(beispiel):
    """Die zehn Absaetze als ein Textblock, so wie das Modell sie sieht."""
    return "\n\n".join(
        f"{absatz['title']}: {' '.join(satz.strip() for satz in absatz['sentences'])}"
        for absatz in beispiel.paragraphs
    )


def normalisiert(text):
    """Kleinschreibung, ohne Artikel, ohne Satzzeichen - der uebliche HotPotQA-Vergleich."""
    text = text.lower()
    text = "".join(z for z in text if z not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def punkte(vorhergesagt, richtig):
    """Exakte Uebereinstimmung und Wort-F1, beide auf der normalisierten Form."""
    a, b = normalisiert(vorhergesagt), normalisiert(richtig)
    exakt = float(a == b)
    gemeinsam = collections.Counter(a.split()) & collections.Counter(b.split())
    treffer = sum(gemeinsam.values())
    if treffer == 0:
        return exakt, 0.0
    genauigkeit = treffer / len(a.split())
    trefferquote = treffer / len(b.split())
    return exakt, 2 * genauigkeit * trefferquote / (genauigkeit + trefferquote)


def pruefe(modell, beispiele):
    """Ein Modell gegen alle Beispiele, mit Qualitaet und Kosten."""
    lm = dspy.LM(f"openrouter/{modell}", cache=False, max_tokens=MAX_TOKENS, temperature=0.0)
    programm = dspy.Predict(FrageBeantworten)
    exakt, f1, fehler = [], [], 0

    for beispiel in beispiele:
        try:
            with dspy.context(lm=lm):
                antwort = programm(paragraphs=als_text(beispiel), question=beispiel.question).answer
        except Exception as ausnahme:  # noqa: BLE001 - im Vortest zaehlt, *dass* es scheitert
            fehler += 1
            print(f"    Fehler: {type(ausnahme).__name__}", file=sys.stderr)
            continue
        e, f = punkte(antwort, beispiel.answer)
        exakt.append(e)
        f1.append(f)

    kosten = sum(eintrag["cost"] or 0.0 for eintrag in lm.history)
    anbieter = sorted({getattr(eintrag["response"], "provider", None) for eintrag in lm.history})
    return {
        "modell": modell,
        "exakt": sum(exakt) / len(exakt) if exakt else 0.0,
        "f1": sum(f1) / len(f1) if f1 else 0.0,
        "fehler": fehler,
        "kosten_usd": kosten,
        "kosten_je_frage_usd": kosten / len(beispiele),
        "anbieter": anbieter,
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    beispiele = lade_aufteilung().trainingsdaten[:ANZAHL_FRAGEN]
    print(f"{len(beispiele)} Trainingsbeispiele, {len(KANDIDATENMODELLE)} Modelle\n")

    ergebnisse = []
    for i, modell in enumerate(KANDIDATENMODELLE, start=1):
        print(f"[{i}/{len(KANDIDATENMODELLE)}] {modell}")
        zeile = pruefe(modell, beispiele)
        ergebnisse.append(zeile)
        print(f"    exakt {zeile['exakt']:.2f}  F1 {zeile['f1']:.2f}  "
              f"${zeile['kosten_je_frage_usd']:.6f}/Frage  {zeile['fehler']} Fehler  "
              f"{','.join(str(a) for a in zeile['anbieter'])}")

    ergebnisse.sort(key=lambda z: z["f1"], reverse=True)
    ERGEBNIS.parent.mkdir(parents=True, exist_ok=True)
    ERGEBNIS.write_text(
        json.dumps({"gelaufen_am": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "anzahl_fragen": len(beispiele),
                    "ergebnisse": ergebnisse}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n{'Modell':<45}{'exakt':>7}{'F1':>7}{'$/Frage':>11}{'Fehler':>8}")
    print("-" * 78)
    for z in ergebnisse:
        print(f"{z['modell']:<45}{z['exakt']:>7.2f}{z['f1']:>7.2f}"
              f"{z['kosten_je_frage_usd']:>11.6f}{z['fehler']:>8}")
    print(f"\nErgebnis geschrieben: {ERGEBNIS}")


if __name__ == "__main__":
    main()
