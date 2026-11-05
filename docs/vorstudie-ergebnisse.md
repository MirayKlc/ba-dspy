# Ergebnisse der Vorstudie

Stand 10.10.2026, nach den zwölf Fixkosten-Messungen. Erzeugt mit
`python -m dspy_optimizer.vorstudie auswertung` (ohne API-Aufruf) aus `ergebnisse/protokoll.jsonl`,
Commit `ab15e50`. Aus den Ergebnissen wurde festgelegt:

- **B = 6,00 $** je Durchgang. Hauptexperiment
  geschätzt 115,82 $, mit Vorstudie 127,31 $ = 113,61 € (EZB-Referenzkurs 09.10.2026:
  1 € = 1,1206 $).
- **Kostenobergrenze 0,2895 $ je 1000 Programmläufe** (Gemma 4 31B optimiert).
- **Fixkosten = Mittelwert der drei Fixkosten-Messungen**.

Der Bericht unten nennt „B mindestens“ (4,74 $) und das Hauptexperiment bei diesem Mindestwert;
festgelegt ist B = 6 $. Der Bericht ist die Ausgabe des genannten Befehls mit einer redaktionellen
Anpassung: Die Ausgabe verweist auf interne Arbeitsunterlagen (`ADR-…`, `Issue #…`); hier sind
diese Verweise entfernt.
Zahlen und übriger Wortlaut sind unverändert, einschließlich der Schreibweise ohne Umlaute.

---


# Vorstudie

Herkunft der Zahlen: Vorstudie, gemessen 2026-10-09 bis 2026-10-10

Ausgegeben in der Vorstudie, alle Messlaeufe: 11.4996 $
Anbieterfehler in der Vorstudie: 11 (auch in nicht gewerteten Laeufen)

## Je Kandidatenmodell

Fixkosten $ ist der Mittelwert der Fixkosten-Messungen, $ je Trial die Steigung durch die Optimierungslaeufe darueber.

| Kandidatenmodell | F1 ohne | F1 optimiert | $/1000 ohne | $/1000 optimiert | Anstieg | Fixkosten $ | Fixkosten min-max $ | $ je Trial | Latenz ohne s | Latenz optimiert s | h je $ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| openai/gpt-6-luna | 0.590 | 0.726 | 0.287 | 0.898 | 3.13 | 0.0608 | 0.0431-0.0846 | 0.03907 | 2.97 | 2.81 | 0.28 |
| deepseek/deepseek-v4.1-flash | 0.851 | 0.846 | 0.219 | 0.227 | 1.04 | 0.0417 | 0.0342-0.0552 | 0.01682 | 1.90 | 1.77 | 0.45 |
| google/gemma-4-31b-it | 0.790 | 0.816 | 0.207 | 0.290 | 1.40 | 0.0516 | 0.0511-0.0519 | 0.02552 | 3.88 | 3.98 | 0.56 |
| openai/gpt-oss-120b | 0.788 | 0.804 | 0.112 | 0.163 | 1.46 | 0.0320 | 0.0315-0.0324 | 0.01642 | 5.18 | 3.16 | 0.86 |

## Je Fixkosten-Messung

Nur der feste Teil eines Optimierungslaufs: Beispiele sammeln, Instruktionen vorschlagen, erste vollstaendige Bewertung (`num_trials = 0`).

| Kandidatenmodell | Messung | Seed | Fixkosten $ | Dauer min |
|---|---|---|---|---|
| openai/gpt-6-luna | 1 | 0 | 0.0846 | 6 |
| openai/gpt-6-luna | 2 | 4 | 0.0431 | 7 |
| openai/gpt-6-luna | 3 | 5 | 0.0548 | 10 |
| deepseek/deepseek-v4.1-flash | 1 | 0 | 0.0552 | 5 |
| deepseek/deepseek-v4.1-flash | 2 | 4 | 0.0357 | 4 |
| deepseek/deepseek-v4.1-flash | 3 | 5 | 0.0342 | 4 |
| google/gemma-4-31b-it | 1 | 0 | 0.0511 | 7 |
| google/gemma-4-31b-it | 2 | 4 | 0.0519 | 7 |
| google/gemma-4-31b-it | 3 | 5 | 0.0518 | 7 |
| openai/gpt-oss-120b | 1 | 0 | 0.0324 | 9 |
| openai/gpt-oss-120b | 2 | 4 | 0.0315 | 9 |
| openai/gpt-oss-120b | 3 | 5 | 0.0323 | 8 |

## Je Optimierungslauf

Die Gerade ist das kalibrierte Kostenmodell; ihre Abweichung von der Messung zeigt, ob die Kosten auf einer Geraden liegen. Abweichung ist (Optimierung - Gerade) / Gerade: positiv, wo die Gerade zu wenig geplant haette.

| Kandidatenmodell | Trials | Optimierung $ | Gerade $ | Abweichung | F1 | $/1000 | Formatfehler | abgeschnitten (Kandidat/Vorschlag) | Dauer min |
|---|---|---|---|---|---|---|---|---|---|
| openai/gpt-6-luna | 10 | 0.5386 | 0.4516 | +19.3 % | 0.708 | 0.832 | 6 | 0/2 | 14 |
| openai/gpt-6-luna | 20 | 0.7600 | 0.8423 | -9.8 % | 0.710 | 1.037 | 7 | 0/0 | 18 |
| openai/gpt-6-luna | 40 | 1.6431 | 1.6237 | +1.2 % | 0.759 | 0.825 | 2 | 0/0 | 27 |
| deepseek/deepseek-v4.1-flash | 10 | 0.2306 | 0.2099 | +9.9 % | 0.860 | 0.230 | 0 | 0/0 | 9 |
| deepseek/deepseek-v4.1-flash | 20 | 0.3766 | 0.3780 | -0.4 % | 0.828 | 0.223 | 3 | 0/0 | 13 |
| deepseek/deepseek-v4.1-flash | 40 | 0.7098 | 0.7143 | -0.6 % | 0.849 | 0.228 | 2 | 0/0 | 19 |
| google/gemma-4-31b-it | 10 | 0.2884 | 0.3068 | -6.0 % | 0.815 | 0.310 | 0 | 0/0 | 15 |
| google/gemma-4-31b-it | 20 | 0.5320 | 0.5620 | -5.4 % | 0.802 | 0.234 | 0 | 0/0 | 23 |
| google/gemma-4-31b-it | 40 | 1.0921 | 1.0724 | +1.8 % | 0.830 | 0.324 | 0 | 0/0 | 32 |
| openai/gpt-oss-120b | 10 | 0.1749 | 0.1962 | -10.9 % | 0.821 | 0.182 | 0 | 0/0 | 17 |
| openai/gpt-oss-120b | 20 | 0.2981 | 0.3605 | -17.3 % | 0.793 | 0.181 | 0 | 0/1 | 21 |
| openai/gpt-oss-120b | 40 | 0.7254 | 0.6889 | +5.3 % | 0.798 | 0.128 | 0 | 0/1 | 28 |

## Festzulegen

Spannweite ohne Optimierung (F1): 0.261
Zweitteuerstes optimiert: google/gemma-4-31b-it
Kostenobergrenze: 0.290 $ je 1000 Programmlaeufe
B mindestens: 4.74 $ (9 Trials je Kandidatenmodell in der Sondierung)
Hauptexperiment bei diesem B: 93.08 $, je weiterem Dollar B: 18 $ mehr
Vorstudie und Hauptexperiment zusammen: 104.58 $ (gegen 200 EUR zu halten, zum Kurs des Tages, an dem B festgelegt wird)
