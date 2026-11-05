"""Vorauswahl der Kandidatenmodelle, Runden 2 bis 4: unter Last, mit Provider-Pinning.

Runde 1 (`spikes/vorauswahl.py`) hat nur Qualitaet und Preis gemessen, und zwar Aufruf fuer
Aufruf nacheinander. Dabei fielen einzelne Modelle mit Rate-Limit-Fehlern aus. Genau das ist
aber die entscheidende Frage: das Hauptexperiment stellt **Hunderte** von Anfragen, und
MIPROv2 wertet sie **parallel** aus. Ein Modell, das dabei mit HTTP 429 dichtmacht, ist
unbrauchbar, egal wie gut es antwortet.

Runde 2 misst deshalb drei Dinge gleichzeitig:

1. **Belastbarkeit** - wie viele Anfragen scheitern, wenn mehrere gleichzeitig laufen?
   Fehler werden nach Art getrennt gezaehlt; Rate-Limit-Fehler sind der Knackpunkt.
2. **Qualitaet** - exakte Uebereinstimmung und Wort-F1, diesmal mit Einzelwerten, damit
   sich eine Streuung ausrechnen laesst statt sie zu schaetzen.
3. **Preis** - die tatsaechlich abgerechneten Dollar je Frage.

Neu ist ausserdem das **Provider-Pinning**: Modelle mit mehreren Anbietern kommen ins Feld,
werden aber auf genau einen Anbieter festgenagelt (`provider.only` mit `allow_fallbacks:
false`). Dass die Vorgabe wirklich ankommt, prueft das Skript nach: es vergleicht bei jedem
Aufruf den tatsaechlich bedienenden Anbieter mit dem gewuenschten.

Wiederholungen sind standardmaessig **abgeschaltet** (`num_retries=0`). Sonst wuerde DSPy
Rate-Limit-Fehler stillschweigend abfangen, die hier gerade gemessen werden sollen. Das ist der
Schlimmstfall; das Hauptexperiment laeuft mit Wiederholungen. Deshalb laesst sich die Zahl als
zweites Argument uebergeben, um beides zu vergleichen.

Aufruf:  python spikes/vorauswahl2.py [Runde] [Wiederholungen]
Kosten:  rund 30 bis 60 Cent fuer den ganzen Lauf.
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
from vorauswahl import FrageBeantworten, als_text, punkte  # noqa: E402

ERGEBNIS_MUSTER = "vorauswahl-runde{}.json"

ANZAHL_FRAGEN = 24
# So viele Anfragen laufen gleichzeitig. MIPROv2 wertet ebenfalls parallel aus; ohne
# Gleichzeitigkeit wuerde ein Rate-Limit gar nicht erst auftreten.
GLEICHZEITIG = 8
MAX_TOKENS = 1000
WIEDERHOLUNGEN = 0

# (Modell, Anbieter zum Festnageln oder None). Das Kuerzel ist der OpenRouter-Slug, also
# "deepinfra", nicht "deepinfra/fp8". Ausgewaehlt nach Verfuegbarkeit der letzten 30 Minuten,
# moeglichst schonender Quantisierung und Preis (docs/spikes/kandidatenmodelle.json).
KANDIDATENMODELLE = [
    # Mehrere Anbieter, deshalb festgenagelt
    ("deepseek/deepseek-v4-flash-0731", "deepinfra"),
    ("z-ai/glm-5.3-flash", "novita"),
    ("qwen/qwen3.6-35b-a3b", "deepinfra"),
    ("minimax/minimax-m3", "together"),
    ("google/gemma-4-26b-a4b-it", "novita"),
    ("nvidia/nemotron-3.5-lightning", "deepinfra"),
    ("inclusionai/ling-3.0-flash", "novita"),
    ("tencent/hy3", "novita"),
    ("xiaomi/mimo-v2.5", "deepinfra"),
    # Genau ein Anbieter, nichts festzunageln - die Besten aus Runde 1 als Vergleich
    ("mistralai/mistral-small-24b-instruct-2501", None),
    ("qwen/qwen3.7-flash", None),
    ("mistralai/ministral-8b-2512", None),
    ("inception/mercury-2.5", None),
    ("meituan/longcat-2.0", None),
    ("moonshotai/kimi-k2", None),
    # Die drei aus Runde 1, deren Messung an Fehlern gescheitert ist
    ("mistralai/mistral-small-2603", None),
    ("stepfun/step-3.5-flash", None),
]

# Runde 3: westliche Modellfamilien, damit das Feld nicht nur aus chinesischen Modellen
# besteht. Ohne Pinning waere davon fast nichts zulaessig - genau dafuer ist es da.
WESTLICHE_MODELLE = [
    ("openai/gpt-oss-120b", "deepinfra"),
    ("openai/gpt-oss-20b", "deepinfra"),
    ("openai/gpt-5.6-luna", "openai"),
    ("google/gemini-3.1-flash-lite", "google-vertex"),
    ("google/gemma-4-31b-it", "deepinfra"),
    ("meta/muse-glimmer-30b", "deepinfra"),
    ("meta-llama/llama-4-maverick", "deepinfra"),
]

# Runde 4: die vier ausgewaehlten Kandidatenmodelle, genau so festgenagelt, wie sie im
# Experiment laufen sollten. Drei davon unkomprimiert (bf16), Kimi K2 nur als fp8
# zu haben. Bei gemma-4-31b ist Novita gewaehlt, weil DeepInfra nur fp4 anbietet.
DIE_VIER = [
    ("openai/gpt-oss-120b", "deepinfra"),
    ("google/gemma-4-31b-it", "novita"),
    ("meta/muse-glimmer-30b", "deepinfra"),
    ("moonshotai/kimi-k2", "novita"),
]

GRUPPEN = {"2": KANDIDATENMODELLE, "3": WESTLICHE_MODELLE, "4": DIE_VIER}


def fehlerart(ausnahme):
    """Rate-Limit, Format oder sonst was - die Unterscheidung ist der Kern dieser Runde."""
    name = type(ausnahme).__name__
    text = str(ausnahme)
    if "RateLimit" in name or "429" in text:
        return "rate_limit"
    if "Parse" in name or "Adapter" in name:
        return "format"
    return name


def ein_versuch(programm, beispiel):
    """Eine Frage stellen; gibt (Punkte, Fehlerart) zurueck, genau eins von beidem."""
    try:
        antwort = programm(paragraphs=als_text(beispiel), question=beispiel.question).answer
    except Exception as ausnahme:  # noqa: BLE001 - im Vortest zaehlt, *dass* und *woran* es scheitert
        return None, fehlerart(ausnahme)
    return punkte(antwort, beispiel.answer), None


def pruefe(modell, gewuenschter_anbieter, beispiele):
    """Ein Modell unter Last: Qualitaet, Fehler nach Art, Preis, tatsaechlicher Anbieter."""
    zusatz = {}
    if gewuenschter_anbieter:
        zusatz["extra_body"] = {
            "provider": {"only": [gewuenschter_anbieter], "allow_fallbacks": False}
        }
    lm = dspy.LM(f"openrouter/{modell}", cache=False, max_tokens=MAX_TOKENS,
                 temperature=0.0, num_retries=WIEDERHOLUNGEN, **zusatz)
    programm = dspy.Predict(FrageBeantworten)
    programm.set_lm(lm)

    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=GLEICHZEITIG) as pool:
        ausgang = list(pool.map(lambda b: ein_versuch(programm, b), beispiele))
    dauer = time.time() - start

    exakt = [e for (p, _) in ausgang if p for e in [p[0]]]
    f1 = [f for (p, _) in ausgang if p for f in [p[1]]]
    fehler = collections.Counter(art for (_, art) in ausgang if art)

    kosten = sum(eintrag["cost"] or 0.0 for eintrag in lm.history)
    bedient = collections.Counter(
        str(getattr(eintrag["response"], "provider", None)) for eintrag in lm.history
    )
    return {
        "modell": modell,
        "gewuenschter_anbieter": gewuenschter_anbieter,
        "bedient_haben": dict(bedient),
        "pinning_gehalten": (
            None if not gewuenschter_anbieter
            else len(bedient) <= 1  # ein einziger Anbieter hat alles bedient
        ),
        "beantwortet": len(f1),
        "gestellt": len(beispiele),
        "fehler": dict(fehler),
        "rate_limit_fehler": fehler.get("rate_limit", 0),
        "exakt": statistics.fmean(exakt) if exakt else 0.0,
        "f1": statistics.fmean(f1) if f1 else 0.0,
        "f1_streuung": statistics.stdev(f1) if len(f1) > 1 else None,
        "f1_einzelwerte": f1,
        "kosten_usd": kosten,
        "kosten_je_frage_usd": kosten / len(f1) if f1 else None,
        "dauer_s": dauer,
        "fragen_je_minute": len(beispiele) / dauer * 60,
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    runde = sys.argv[1] if len(sys.argv) > 1 else "2"
    if runde not in GRUPPEN:
        sys.exit(f"Unbekannte Runde: {runde}. Moeglich sind: {', '.join(GRUPPEN)}")
    global KANDIDATENMODELLE, ERGEBNIS, WIEDERHOLUNGEN
    KANDIDATENMODELLE = GRUPPEN[runde]
    WIEDERHOLUNGEN = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    ERGEBNIS = (pathlib.Path(__file__).resolve().parent.parent / "docs" / "spikes"
                / ERGEBNIS_MUSTER.format(f"{runde}-mit-{WIEDERHOLUNGEN}-wiederholungen"
                                         if WIEDERHOLUNGEN else runde))
    beispiele = lade_aufteilung().trainingsdaten[:ANZAHL_FRAGEN]
    print(f"{len(beispiele)} Fragen, je {GLEICHZEITIG} gleichzeitig, "
          f"{len(KANDIDATENMODELLE)} Modelle, keine Wiederholungen\n")

    ergebnisse = []
    for i, (modell, anbieter) in enumerate(KANDIDATENMODELLE, start=1):
        print(f"[{i}/{len(KANDIDATENMODELLE)}] {modell}"
              + (f"  (festgenagelt auf {anbieter})" if anbieter else ""))
        zeile = pruefe(modell, anbieter, beispiele)
        ergebnisse.append(zeile)
        kosten = (f"${zeile['kosten_je_frage_usd']:.6f}/Frage"
                  if zeile["kosten_je_frage_usd"] else "keine Antwort")
        print(f"    {zeile['beantwortet']}/{zeile['gestellt']} beantwortet, "
              f"{zeile['rate_limit_fehler']} Rate-Limit  |  F1 {zeile['f1']:.2f}  |  {kosten}  |  "
              f"{zeile['fragen_je_minute']:.0f} Fragen/min  |  "
              f"{','.join(zeile['bedient_haben']) or '-'}")

    ERGEBNIS.parent.mkdir(parents=True, exist_ok=True)
    ERGEBNIS.write_text(
        json.dumps({"gelaufen_am": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "anzahl_fragen": len(beispiele),
                    "gleichzeitig": GLEICHZEITIG,
                    "ergebnisse": ergebnisse}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    brauchbar = [z for z in ergebnisse if z["beantwortet"] == z["gestellt"]]
    brauchbar.sort(key=lambda z: z["f1"], reverse=True)
    print(f"\n{len(brauchbar)} von {len(ergebnisse)} Modellen ohne einen einzigen Fehler:\n")
    print(f"{'Modell':<44}{'F1':>6}{'±':>6}{'$/Frage':>11}{'F/min':>7}  Anbieter")
    print("-" * 92)
    for z in brauchbar:
        streuung = f"{z['f1_streuung']:.2f}" if z["f1_streuung"] else "-"
        print(f"{z['modell']:<44}{z['f1']:>6.2f}{streuung:>6}"
              f"{z['kosten_je_frage_usd']:>11.6f}{z['fragen_je_minute']:>7.0f}  "
              f"{','.join(z['bedient_haben'])}")
    ausgefallen = [z for z in ergebnisse if z["beantwortet"] < z["gestellt"]]
    if ausgefallen:
        print(f"\nMit Ausfaellen (fuer das Hauptexperiment fraglich):")
        for z in sorted(ausgefallen, key=lambda z: z["beantwortet"]):
            print(f"  {z['modell']:<44}{z['beantwortet']:>3}/{z['gestellt']}  {z['fehler']}")
    print(f"\nErgebnis geschrieben: {ERGEBNIS}")


if __name__ == "__main__":
    main()
