# Auswertung des Hauptexperiments (Issue #13)

Erzeugt am 15.10.2026 von `python -m dspy_optimizer.auswertung`, allein aus `ergebnisse/protokoll.jsonl` (230 Zeilen), ohne einen API-Aufruf. Jede Ergebniszahl ist Mittelwert ± Standardabweichung (Stichprobe, n − 1) über die Seeds 1, 2, 3. Werte je Seed stehen im Bericht nur, wo ein Seed für sich zählt: Kendalls Tau je Durchgang (FF1), der gepaarte Test zweier Programme und Seed 3 in beiden Preisfassungen (ADR-0028); alle übrigen Tabellen „je Seed“ liegen als CSV daneben.

Die Stichproben: **Verfahren gegen Gleichverteilung** hat als Stichprobe die drei Seeds — zu wenig für eine belastbare Signifikanzaussage; berichtet wird die Streuung. **Zwei fertige Programme auf den Testdaten** haben 500 gepaarte Beispiele; dort trägt ein gepaarter t-Test (zweiseitig, α = 0,05, ADR-0022, Nachtrag 28.09.2026).

Gültig: 21 Durchgänge, 39 Modell-Programm-Paare mit je einer gültigen Testbewertung über 500 Testbeispiele.

## Ungültige Durchgänge und wiederholte Testbewertungen

6 ungültige Durchgänge, zusammen 16,13 $. Ursache laut `stoerungen`: `LMRateLimitError` ist HTTP 429 von DeepInfra (überlasteter gemeinsamer Pool, ADR-0027, Nachtrag 15.10.2026); `KeyboardInterrupt` ist ein Abbruch von Hand (Strg+C). Ein Durchgang zählt nicht, sobald eine Zeile einen Anbieterfehler hat (ADR-0018).

| durchgang | seed | baseline | kandidatenmodelle | anbieterfehler | ursache | dollar |
|---|---|---|---|---|---|---|
| 9 | 2 | verfahren | gpt-oss-120b, Gemma 4 31B | 21 | LMRateLimitError | 1,65 |
| 10 | 2 | verfahren | gpt-oss-120b | 2 | LMRateLimitError, KeyboardInterrupt | 0,00 |
| 11 | 2 | verfahren | gpt-oss-120b | 1 | KeyboardInterrupt | 0,00 |
| 14 | 2 | ein-modell | gpt-oss-120b | 1 | LMRateLimitError | 6,25 |
| 16 | 2 | ein-modell | Gemma 4 31B | 9 | LMRateLimitError | 7,45 |
| 22 | 3 | gleichverteilung | gpt-oss-120b | 7 | LMRateLimitError | 0,77 |

Wiederholte Testbewertungen nach Anbieterfehlern (ADR-0027, Punkt 12): 0.

Abweichung nach dem Stichtag: nach dem ungültigen Durchgang 14 (Seed 2) wiederholt jeder Aufruf bis zu 20-mal statt 10-mal (ADR-0003, Nachtrag 12.10.2026). Die gültigen Durchgänge 1–8, 12 und 13 liefen mit 10; in keinem davon hatte ein Aufruf die 10 Wiederholungen aufgebraucht, mit 20 wären sie also genauso verlaufen.

## Kosten je Stufe

Abgerechnet ist das Hauptergebnis (ADR-0012); der festgehaltene Preis ist die Gegenprobe zur Preiserhöhung bei DeepSeek (ADR-0028).

| stufe | abgerechnet | festgehaltener_preis | davon_ungueltig |
|---|---|---|---|
| Seed 1 | 31,53 | 31,53 | 0,00 |
| Seed 2 | 51,42 | 51,42 | 15,36 |
| Seed 3 | 39,50 | 37,20 | 0,77 |
| Testbewertung | 6,47 | 6,02 | 0,00 |

## FF1: Rangfolge ohne gegen mit Optimierung (Testdaten)

Kendalls Tau: **1,000 ± 0,000**; die Spitze wechselt in 0 von 3 Seeds.

| seed | tau | rangfolge_ohne | rangfolge_mit | spitze_wechselt |
|---|---|---|---|---|
| 1 | 1,000 | DeepSeek V4.1 Flash > Gemma 4 31B > gpt-oss-120b > GPT-6 Luna | DeepSeek V4.1 Flash > Gemma 4 31B > gpt-oss-120b > GPT-6 Luna | nein |
| 2 | 1,000 | DeepSeek V4.1 Flash > Gemma 4 31B > gpt-oss-120b > GPT-6 Luna | DeepSeek V4.1 Flash > Gemma 4 31B > gpt-oss-120b > GPT-6 Luna | nein |
| 3 | 1,000 | DeepSeek V4.1 Flash > Gemma 4 31B > gpt-oss-120b > GPT-6 Luna | DeepSeek V4.1 Flash > Gemma 4 31B > gpt-oss-120b > GPT-6 Luna | nein |

Platzwechsel je Seed, gepaarter t-Test zweiseitig auf den 500 Testbeispielen: `p_mit` zwischen den beiden Programmen der Gleichverteilung (der Unterschied der neuen Rangfolge), `p_ohne` zum Vergleich zwischen den beiden unoptimierten.

Kein Paar tauscht in einem Seed die Plätze.

Belastbar ist ein Platzwechsel bei Signifikanz in mindestens zwei von drei Seeds (ADR-0022, Nachtrag 28.09.2026); ohne Platzwechsel gibt es keinen zu prüfen.

## FF2: Zwischenbewertung — wie oft der Test etwas gefunden hat

Mit Kostenobergrenze fand der Test in 3 von 3 Zwischenbewertungen etwas (einseitig, α = 0,05, ADR-0004). In der Endauswahl gab es 0 Vergleiche: in die Fokusphase ging jeweils 1, 1, 1 Kandidatenmodell(e). Der Zusatzbeleg „sortiert die Sondierung schon richtig?“ (Rangfolge Sondierung gegen Fokusphase, ADR-0022) lässt sich deshalb nicht rechnen: mit einem Kandidatenmodell in der Fokusphase gibt es dort keine Rangfolge.

| seed | durchgang | weiter | test_hat_gefunden | p gpt-oss-120b | Grund gpt-oss-120b | p Gemma 4 31B | Grund Gemma 4 31B | p DeepSeek V4.1 Flash | Grund DeepSeek V4.1 Flash | p GPT-6 Luna | Grund GPT-6 Luna |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | DeepSeek V4.1 Flash | ja | 0,0026 | test | – | kostenfilter | – |  | – | kostenfilter |
| 2 | 12 | DeepSeek V4.1 Flash | ja | 0,0004 | test | – | kostenfilter | – |  | – | kostenfilter |
| 3 | 21 | DeepSeek V4.1 Flash | ja | 0,0006 | test | – | kostenfilter | – |  | – | kostenfilter |

Ohne Kostenobergrenze nachgerechnet (ADR-0021, Nachtrag 15.10.2026):

| seed | durchgang | weiter | test_hat_gefunden | p gpt-oss-120b | Grund gpt-oss-120b | p Gemma 4 31B | Grund Gemma 4 31B | p DeepSeek V4.1 Flash | Grund DeepSeek V4.1 Flash | p GPT-6 Luna | Grund GPT-6 Luna |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | DeepSeek V4.1 Flash | ja | 0,0026 | test | 0,0351 | test | – |  | < 0,0001 | test |
| 2 | 12 | DeepSeek V4.1 Flash | ja | 0,0004 | test | 0,0143 | test | – |  | < 0,0001 | test |
| 3 | 21 | DeepSeek V4.1 Flash | ja | 0,0006 | test | 0,0011 | test | – |  | < 0,0001 | test |

## Jede Auswahl mit und ohne Kostenobergrenze

Nachgerechnet aus den Einzelbewertungen und Ausführungskosten der Nachbewertungen. `nachgerechnet` muss `protokolliert` gleichen — tut es. Die Einzelmodelle wählen nichts; ob ihr Programm unter der Obergrenze liegt, steht unter „Programme“ (ADR-0021, Nachtrag 15.10.2026).

Abweichend von der protokollierten Auswahl:

| seed | durchgang | baseline | protokolliert | nachgerechnet | ohne_obergrenze | festgehaltener_preis |
|---|---|---|---|---|---|---|
| 3 | 23 | gleichverteilung | gpt-oss-120b | gpt-oss-120b | DeepSeek V4.1 Flash | DeepSeek V4.1 Flash |

Alle Auswahlen: `auswahlen.csv`.

## Verfahren gegen Gleichverteilung

Beide zahlen ihre Nachbewertungen aus dem Budget B (ADR-0017). Geplant waren beim Verfahren höchstens 6 (vier in der Sondierung, zwei in der Fokusphase); tatsächlich zahlte es 5 je Durchgang, weil jeweils nur ein Kandidatenmodell in die Fokusphase ging. Die Gleichverteilung zahlte 4. Kosten unter „Kosten“.

**Über die drei Seeds** (Stichprobe: Seeds; keine Signifikanzaussage, nur die Streuung):

| Preisfassung | F1 Verfahren | F1 Gleichverteilung | Differenz |
|---|---|---|---|
| abgerechnet | 0,805 ± 0,011 | 0,783 ± 0,035 | 0,023 ± 0,032 |
| zum Preis vom Stichtag | 0,805 ± 0,011 | 0,803 ± 0,001 | 0,003 ± 0,010 |

**Programmvergleich je Seed** (Stichprobe: die 500 Testbeispiele, gepaarter t-Test zweiseitig). Er sagt, ob sich zwei fertige Programme unterscheiden, nicht, ob das Verfahren der Gleichverteilung überlegen ist. In Seed 1 und 2 sind beide Programme von DeepSeek. Seed 3 in beiden Preisfassungen (ADR-0028, Entscheidung 3).

Abgerechnet:

| seed | verfahren_waehlt | bewertung_verfahren | gegen_waehlt | bewertung_gegner | differenz | p_wert | gegen_unter_grenze | signifikant |
|---|---|---|---|---|---|---|---|---|
| 1 | DeepSeek V4.1 Flash | 0,818 | DeepSeek V4.1 Flash | 0,804 | 0,014 | 0,0606 | ja | nein |
| 2 | DeepSeek V4.1 Flash | 0,798 | DeepSeek V4.1 Flash | 0,802 | -0,004 | 0,6296 | ja | nein |
| 3 | DeepSeek V4.1 Flash | 0,800 | gpt-oss-120b | 0,742 | 0,058 | < 0,0001 | ja | ja |

Zum festgehaltenen Preis:

| seed | verfahren_waehlt | bewertung_verfahren | gegen_waehlt | bewertung_gegner | differenz | p_wert | gegen_unter_grenze | signifikant |
|---|---|---|---|---|---|---|---|---|
| 1 | DeepSeek V4.1 Flash | 0,818 | DeepSeek V4.1 Flash | 0,804 | 0,014 | 0,0606 | ja | nein |
| 2 | DeepSeek V4.1 Flash | 0,798 | DeepSeek V4.1 Flash | 0,802 | -0,004 | 0,6296 | ja | nein |
| 3 | DeepSeek V4.1 Flash | 0,800 | DeepSeek V4.1 Flash | 0,803 | -0,003 | 0,7045 | ja | nein |

Gegen jede Baseline, abgerechnet:

| Verfahren gegen | F1 Verfahren | F1 Gegner | Differenz | Programmtest signifikant (500 Beispiele) | Gegner unter der Obergrenze |
|---|---|---|---|---|---|
| Gleichverteilung (B/4 je Modell) | 0,805 ± 0,011 | 0,783 ± 0,035 | 0,023 ± 0,032 | 1 von 3 | ja in 3 von 3 |
| Ohne Optimierung | 0,805 ± 0,011 | 0,796 ± 0,001 | 0,009 ± 0,010 | 1 von 3 | ja in 3 von 3 |
| Einzelmodell (volles Budget B): gpt-oss-120b | 0,805 ± 0,011 | 0,758 ± 0,014 | 0,047 ± 0,003 | 3 von 3 | ja in 3 von 3 |
| Einzelmodell (volles Budget B): Gemma 4 31B | 0,805 ± 0,011 | 0,796 ± 0,003 | 0,009 ± 0,014 | 1 von 3 | ja in 0 von 3 |
| Einzelmodell (volles Budget B): DeepSeek V4.1 Flash | 0,805 ± 0,011 | 0,800 ± 0,004 | 0,005 ± 0,008 | 0 von 3 | ja in 2 von 3 |
| Einzelmodell (volles Budget B): GPT-6 Luna | 0,805 ± 0,011 | 0,716 ± 0,027 | 0,089 ± 0,022 | 3 von 3 | ja in 0 von 3 |

Gegen jede Baseline, zum festgehaltenen Preis:

| Verfahren gegen | F1 Verfahren | F1 Gegner | Differenz | Programmtest signifikant (500 Beispiele) | Gegner unter der Obergrenze |
|---|---|---|---|---|---|
| Gleichverteilung (B/4 je Modell) | 0,805 ± 0,011 | 0,803 ± 0,001 | 0,003 ± 0,010 | 0 von 3 | ja in 3 von 3 |
| Ohne Optimierung | 0,805 ± 0,011 | 0,796 ± 0,001 | 0,009 ± 0,010 | 1 von 3 | ja in 3 von 3 |
| Einzelmodell (volles Budget B): gpt-oss-120b | 0,805 ± 0,011 | 0,758 ± 0,014 | 0,047 ± 0,003 | 3 von 3 | ja in 3 von 3 |
| Einzelmodell (volles Budget B): Gemma 4 31B | 0,805 ± 0,011 | 0,796 ± 0,003 | 0,009 ± 0,014 | 1 von 3 | ja in 0 von 3 |
| Einzelmodell (volles Budget B): DeepSeek V4.1 Flash | 0,805 ± 0,011 | 0,800 ± 0,004 | 0,005 ± 0,008 | 0 von 3 | ja in 3 von 3 |
| Einzelmodell (volles Budget B): GPT-6 Luna | 0,805 ± 0,011 | 0,716 ± 0,027 | 0,089 ± 0,022 | 3 von 3 | ja in 0 von 3 |

## Endqualität je Strategie (FF3)

F1 auf den Testdaten des Programms, das jede Strategie wählt; beim Einzelmodell sein Programm, auch über der Kostenobergrenze (dazu „Programme“). In Klammern, wie oft welches Kandidatenmodell gewählt wurde. Abbildung `endqualitaet.pdf`.

| Strategie | F1 Testdaten | gewaehlt |
|---|---|---|
| Ohne Optimierung | 0,796 ± 0,001 | DeepSeek V4.1 Flash (3) |
| Einzelmodell: gpt-oss-120b | 0,758 ± 0,014 | gpt-oss-120b (3) |
| Einzelmodell: Gemma 4 31B | 0,796 ± 0,003 | Gemma 4 31B (3) |
| Einzelmodell: DeepSeek V4.1 Flash | 0,800 ± 0,004 | DeepSeek V4.1 Flash (3) |
| Einzelmodell: GPT-6 Luna | 0,716 ± 0,027 | GPT-6 Luna (3) |
| Gleichverteilung (abgerechnet) | 0,783 ± 0,035 | DeepSeek V4.1 Flash (2), gpt-oss-120b (1) |
| Gleichverteilung (Preis vom Stichtag) | 0,803 ± 0,001 | DeepSeek V4.1 Flash (3) |
| Verfahren | 0,805 ± 0,011 | DeepSeek V4.1 Flash (3) |

## Programme der Baselines, ohne Gewinner-Etikett

Je Baseline und Kandidatenmodell; „unter der Kostenobergrenze?“ nach den Ausführungskosten auf den Auswahldaten, nach denen die Regel wählt (ADR-0020). Die Latenz wird nur berichtet und geht in keine Auswahl ein (ADR-0019, Punkt 5).

| baseline | kandidatenmodell | seeds | F1 Auswahldaten | F1 Testdaten | $ je 1000 Auswahldaten | $ je 1000 Testdaten | $ je 1000 Testdaten, festgehaltener Preis | unter der Kostenobergrenze? | unter der Kostenobergrenze? (festgehaltener Preis) | Latenz Median s (Testdaten) |
|---|---|---|---|---|---|---|---|---|---|---|
| Verfahren | DeepSeek V4.1 Flash | 3 | 0,862 ± 0,006 | 0,805 ± 0,011 | 0,236 ± 0,018 | 0,284 ± 0,011 | 0,213 ± 0,006 | ja in 3 von 3 | ja in 3 von 3 | 1,93 ± 0,04 |
| Gleichverteilung (B/4 je Modell) | gpt-oss-120b | 3 | 0,796 ± 0,006 | 0,751 ± 0,008 | 0,181 ± 0,119 | 0,182 ± 0,119 | 0,182 ± 0,119 | ja in 2 von 3 | ja in 2 von 3 | 3,94 ± 0,38 |
| Gleichverteilung (B/4 je Modell) | Gemma 4 31B | 3 | 0,823 ± 0,008 | 0,789 ± 0,008 | 0,414 ± 0,166 | 0,412 ± 0,165 | 0,412 ± 0,165 | ja in 1 von 3 | ja in 1 von 3 | 3,91 ± 0,17 |
| Gleichverteilung (B/4 je Modell) | DeepSeek V4.1 Flash | 3 | 0,860 ± 0,004 | 0,803 ± 0,001 | 0,264 ± 0,047 | 0,314 ± 0,028 | 0,231 ± 0,015 | ja in 2 von 3 | ja in 3 von 3 | 2,15 ± 0,18 |
| Gleichverteilung (B/4 je Modell) | GPT-6 Luna | 3 | 0,738 ± 0,027 | 0,711 ± 0,016 | 0,749 ± 0,219 | 0,749 ± 0,219 | 0,749 ± 0,219 | ja in 0 von 3 | ja in 0 von 3 | 3,49 ± 0,59 |
| Einzelmodell (volles Budget B) | gpt-oss-120b | 3 | 0,806 ± 0,007 | 0,758 ± 0,014 | 0,146 ± 0,033 | 0,147 ± 0,034 | 0,147 ± 0,034 | ja in 3 von 3 | ja in 3 von 3 | 5,45 ± 0,47 |
| Einzelmodell (volles Budget B) | Gemma 4 31B | 3 | 0,833 ± 0,004 | 0,796 ± 0,003 | 0,429 ± 0,075 | 0,425 ± 0,073 | 0,425 ± 0,073 | ja in 0 von 3 | ja in 0 von 3 | 3,88 ± 0,62 |
| Einzelmodell (volles Budget B) | DeepSeek V4.1 Flash | 3 | 0,851 ± 0,003 | 0,800 ± 0,004 | 0,270 ± 0,049 | 0,316 ± 0,018 | 0,233 ± 0,007 | ja in 2 von 3 | ja in 3 von 3 | 1,92 ± 0,10 |
| Einzelmodell (volles Budget B) | GPT-6 Luna | 3 | 0,757 ± 0,032 | 0,716 ± 0,027 | 0,590 ± 0,128 | 0,591 ± 0,126 | 0,591 ± 0,126 | ja in 0 von 3 | ja in 0 von 3 | 3,29 ± 0,18 |
| Ohne Optimierung | gpt-oss-120b | 3 | 0,791 ± 0,006 | 0,751 ± 0,006 | 0,113 ± 0,001 | 0,113 ± 0,000 | 0,113 ± 0,000 | ja in 3 von 3 | ja in 3 von 3 | 4,30 ± 0,28 |
| Ohne Optimierung | Gemma 4 31B | 3 | 0,790 ± 0,008 | 0,772 ± 0,005 | 0,207 ± 0,000 | 0,207 ± 0,000 | 0,207 ± 0,000 | ja in 3 von 3 | ja in 3 von 3 | 3,41 ± 0,10 |
| Ohne Optimierung | DeepSeek V4.1 Flash | 3 | 0,846 ± 0,009 | 0,796 ± 0,001 | 0,228 ± 0,011 | 0,288 ± 0,001 | 0,222 ± 0,000 | ja in 3 von 3 | ja in 3 von 3 | 1,96 ± 0,08 |
| Ohne Optimierung | GPT-6 Luna | 3 | 0,600 ± 0,012 | 0,566 ± 0,003 | 0,288 ± 0,001 | 0,288 ± 0,000 | 0,288 ± 0,000 | ja in 3 von 3 | ja in 3 von 3 | 3,03 ± 0,19 |

## Kosten: Optimierung, Nachbewertung, Plan

C_opt = Optimierung + Nachbewertung (ADR-0022). Abweichung vom Plan: gegen die geplanten Dollar beider Messläufe (ADR-0017); Abweichung von B: gegen das Optimierungsbudget von 6 $. Die Ausführungskosten stehen getrennt davon unter „Programme“ und „FF4“.

Abgerechnet:

| baseline | kandidatenmodell | Optimierung $ | Nachbewertung $ | Nachbewertungen | C_opt $ | geplant $ | Abweichung vom Plan | Abweichung von B |
|---|---|---|---|---|---|---|---|---|
| Verfahren |  | 5,285 ± 0,179 | 0,384 ± 0,065 | 5,0 ± 0,0 | 5,669 ± 0,140 | 5,525 ± 0,147 | 2,6 % ± 0,7 % | -5,5 % ± 2,3 % |
| Gleichverteilung (B/4 je Modell) |  | 6,178 ± 0,649 | 0,322 ± 0,092 | 4,0 ± 0,0 | 6,500 ± 0,722 | 5,971 ± 0,000 | 8,9 % ± 12,1 % | 8,3 % ± 12,0 % |
| Einzelmodell (volles Budget B) | gpt-oss-120b | 5,923 ± 0,733 | 0,029 ± 0,007 | 1,0 ± 0,0 | 5,952 ± 0,729 | 5,992 ± 0,000 | -0,7 % ± 12,2 % | -0,8 % ± 12,2 % |
| Einzelmodell (volles Budget B) | Gemma 4 31B | 6,604 ± 0,800 | 0,086 ± 0,015 | 1,0 ± 0,0 | 6,690 ± 0,811 | 5,979 ± 0,000 | 11,9 % ± 13,6 % | 11,5 % ± 13,5 % |
| Einzelmodell (volles Budget B) | DeepSeek V4.1 Flash | 5,353 ± 1,857 | 0,054 ± 0,010 | 1,0 ± 0,0 | 5,407 ± 1,865 | 5,989 ± 0,000 | -9,7 % ± 31,1 % | -9,9 % ± 31,1 % |
| Einzelmodell (volles Budget B) | GPT-6 Luna | 4,939 ± 0,578 | 0,118 ± 0,026 | 1,0 ± 0,0 | 5,057 ± 0,555 | 5,984 ± 0,000 | -15,5 % ± 9,3 % | -15,7 % ± 9,3 % |
| Ohne Optimierung |  | 0,000 ± 0,000 | 0,167 ± 0,002 | 4,0 ± 0,0 | 0,167 ± 0,002 | – ± – | – | – |

Zum festgehaltenen Preis:

| baseline | kandidatenmodell | Optimierung $ | Nachbewertung $ | Nachbewertungen | C_opt $ | geplant $ | Abweichung vom Plan | Abweichung von B |
|---|---|---|---|---|---|---|---|---|
| Verfahren |  | 5,235 ± 0,246 | 0,383 ± 0,065 | 5,0 ± 0,0 | 5,618 ± 0,222 | 5,525 ± 0,147 | 1,7 % ± 2,0 % | -6,4 % ± 3,7 % |
| Gleichverteilung (B/4 je Modell) |  | 6,141 ± 0,596 | 0,316 ± 0,090 | 4,0 ± 0,0 | 6,457 ± 0,665 | 5,971 ± 0,000 | 8,1 % ± 11,1 % | 7,6 % ± 11,1 % |
| Einzelmodell (volles Budget B) | gpt-oss-120b | 5,923 ± 0,733 | 0,029 ± 0,007 | 1,0 ± 0,0 | 5,952 ± 0,729 | 5,992 ± 0,000 | -0,7 % ± 12,2 % | -0,8 % ± 12,2 % |
| Einzelmodell (volles Budget B) | Gemma 4 31B | 6,604 ± 0,800 | 0,086 ± 0,015 | 1,0 ± 0,0 | 6,690 ± 0,811 | 5,979 ± 0,000 | 11,9 % ± 13,6 % | 11,5 % ± 13,5 % |
| Einzelmodell (volles Budget B) | DeepSeek V4.1 Flash | 4,688 ± 0,754 | 0,048 ± 0,002 | 1,0 ± 0,0 | 4,736 ± 0,752 | 5,989 ± 0,000 | -20,9 % ± 12,6 % | -21,1 % ± 12,5 % |
| Einzelmodell (volles Budget B) | GPT-6 Luna | 4,939 ± 0,578 | 0,118 ± 0,026 | 1,0 ± 0,0 | 5,057 ± 0,555 | 5,984 ± 0,000 | -15,5 % ± 9,3 % | -15,7 % ± 9,3 % |
| Ohne Optimierung |  | 0,000 ± 0,000 | 0,167 ± 0,002 | 4,0 ± 0,0 | 0,167 ± 0,002 | – ± – | – | – |

## FF4: Optimierungskosten gegen Ausführungskosten

C_opt in Programmläufen = C_opt / c_exec des gewählten Paars; Break-even gegen das Paar der Baseline ohne Optimierung im selben Seed, nur wenn das gewählte billiger ist (ADR-0022). c_exec aus der Testbewertung. Getrennt je gewähltem Kandidatenmodell, weil ein Mittel über verschiedene Paare nichts bedeutet. **Ohne Modellwechsel** (gewählt ist dasselbe Kandidatenmodell wie ohne Optimierung) ist ein Break-even kein Modellwechsel im Sinne von ADR-0022, sondern optimiertes gegen unoptimiertes Programm desselben Modells; die Kostendifferenz ist dann oft winzig und der Break-even entsprechend groß und unsicher.

Abgerechnet:

| baseline | kandidatenmodell | Seeds | Referenz (ohne Optimierung) | Modellwechsel | C_opt $ | c_exec $ je 1000 | C_opt in Programmlaeufen | Break-even in Programmlaeufen (nur Seeds mit Break-even) | kein Break-even ueber Kosten, nur ueber Qualitaet |
|---|---|---|---|---|---|---|---|---|---|
| Verfahren | DeepSeek V4.1 Flash | 3 | DeepSeek V4.1 Flash | ja in 0 von 3 | 5,669 ± 0,140 | 0,284 ± 0,011 | 19 969 ± 567 | 569 465 ± 191 685 | 1 von 3 |
| Gleichverteilung (B/4 je Modell) | gpt-oss-120b | 1 | DeepSeek V4.1 Flash | ja in 1 von 1 | 7,153 ± – | 0,113 ± – | 63 182 ± – | 40 763 ± – | 0 von 1 |
| Gleichverteilung (B/4 je Modell) | DeepSeek V4.1 Flash | 2 | DeepSeek V4.1 Flash | ja in 0 von 2 | 6,173 ± 0,634 | 0,312 ± 0,040 | 20 108 ± 4 587 | 1 474 462 ± – | 1 von 2 |
| Einzelmodell (volles Budget B) | gpt-oss-120b | 3 | DeepSeek V4.1 Flash | ja in 3 von 3 | 5,952 ± 0,729 | 0,147 ± 0,034 | 42 349 ± 12 853 | 43 308 ± 7 326 | 0 von 3 |
| Einzelmodell (volles Budget B) | Gemma 4 31B | 3 | DeepSeek V4.1 Flash | ja in 3 von 3 | 6,690 ± 0,811 | 0,425 ± 0,073 | 15 895 ± 1 835 | – | 3 von 3 |
| Einzelmodell (volles Budget B) | DeepSeek V4.1 Flash | 3 | DeepSeek V4.1 Flash | ja in 0 von 3 | 5,407 ± 1,865 | 0,316 ± 0,018 | 17 105 ± 5 568 | – | 3 von 3 |
| Einzelmodell (volles Budget B) | GPT-6 Luna | 3 | DeepSeek V4.1 Flash | ja in 3 von 3 | 5,057 ± 0,555 | 0,591 ± 0,126 | 8 967 ± 2 980 | – | 3 von 3 |

Zum festgehaltenen Preis:

| baseline | kandidatenmodell | Seeds | Referenz (ohne Optimierung) | Modellwechsel | C_opt $ | c_exec $ je 1000 | C_opt in Programmlaeufen | Break-even in Programmlaeufen (nur Seeds mit Break-even) | kein Break-even ueber Kosten, nur ueber Qualitaet |
|---|---|---|---|---|---|---|---|---|---|
| Verfahren | DeepSeek V4.1 Flash | 3 | DeepSeek V4.1 Flash | ja in 0 von 3 | 5,618 ± 0,222 | 0,213 ± 0,006 | 26 392 ± 1 154 | 843 047 ± 610 764 | 0 von 3 |
| Gleichverteilung (B/4 je Modell) | DeepSeek V4.1 Flash | 3 | DeepSeek V4.1 Flash | ja in 0 von 3 | 6,457 ± 0,665 | 0,231 ± 0,015 | 28 082 ± 4 222 | 1 183 624 ± – | 2 von 3 |
| Einzelmodell (volles Budget B) | gpt-oss-120b | 3 | DeepSeek V4.1 Flash | ja in 3 von 3 | 5,952 ± 0,729 | 0,147 ± 0,034 | 42 349 ± 12 853 | 89 823 ± 34 139 | 0 von 3 |
| Einzelmodell (volles Budget B) | Gemma 4 31B | 3 | DeepSeek V4.1 Flash | ja in 3 von 3 | 6,690 ± 0,811 | 0,425 ± 0,073 | 15 895 ± 1 835 | – | 3 von 3 |
| Einzelmodell (volles Budget B) | DeepSeek V4.1 Flash | 3 | DeepSeek V4.1 Flash | ja in 0 von 3 | 4,736 ± 0,752 | 0,233 ± 0,007 | 20 364 ± 3 302 | – | 3 von 3 |
| Einzelmodell (volles Budget B) | GPT-6 Luna | 3 | DeepSeek V4.1 Flash | ja in 3 von 3 | 5,057 ± 0,555 | 0,591 ± 0,126 | 8 967 ± 2 980 | – | 3 von 3 |

## Lernkurven (FF3)

Drei Budgetstufen aus vorhandenen Läufen, Bewertung der Nachbewertung auf den Auswahldaten (ADR-0022, Nachtrag 28.09.2026). Einschränkung: verschiedene Läufe, keine durchgehende Kurve.

| kandidatenmodell | Sondierung (0,75 $) | Gleichverteilung (1,50 $) | Einzelmodell (6,00 $) |
|---|---|---|---|
| gpt-oss-120b | 0,809 ± 0,006 | 0,796 ± 0,006 | 0,806 ± 0,007 |
| Gemma 4 31B | 0,821 ± 0,013 | 0,823 ± 0,008 | 0,833 ± 0,004 |
| DeepSeek V4.1 Flash | 0,863 ± 0,001 | 0,860 ± 0,004 | 0,851 ± 0,003 |
| GPT-6 Luna | 0,727 ± 0,027 | 0,738 ± 0,027 | 0,757 ± 0,032 |

## Abbildungen

- `endqualitaet.pdf/.png`: F1 auf den Testdaten je Strategie, Mittelwert ± Standardabweichung und die drei Seeds als Punkte (FF3).
- `pareto-abgerechnet.pdf/.png`, `pareto-festgehalten.pdf/.png`: Qualität gegen Ausführungskosten auf den Testdaten, je Baseline (Form) und Kandidatenmodell (Farbe), Mittelwert ± Standardabweichung; offene Symbole = Pareto-dominiert; graue Linie = Pareto-Front.
- `lernkurven.pdf/.png`: F1 je Budgetstufe, Mittelwert ± Standardabweichung.
- `ff1-rangfolge.pdf/.png`: F1 auf den Testdaten ohne Optimierung und nach Gleichverteilung je Kandidatenmodell.
