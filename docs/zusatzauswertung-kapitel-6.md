# Zusatzauswertung Kapitel 6

Quelle: `protokoll.jsonl` (230 Zeilen), kein API-Aufruf. 21 gültige Durchgänge, 39 Testbewertungen. Nachbau geprüft gegen die Referenz-CSV in `ergebnisse/auswertung`: 696 Werte gleich (F1 je Programm, FF1, p-Werte der Zwischenbewertungen mit/ohne Obergrenze, Auswahlen, Vergleiche je Seed beider Preisfassungen, FF4 je Seed, C_opt und Planabweichung je Durchgang, Lernkurve Sondierung).

Konventionen: Differenzen in F1 (0..1). p-Werte 4 Stellen. α = 0.05. scipy 1.18.1; Wilcoxon `scipy.stats.wilcoxon(x, y, zero_method=…)` mit `method='auto'` (bei n > 50 bzw. Nullen/Bindungen Normalapproximation, ohne Stetigkeitskorrektur). t-Test `scipy.stats.ttest_rel`. Test = Testdaten (n = 500), Auswahl = Auswahldaten (n = 200).

## 1. Paarweise Modellvergleiche auf den Testdaten (t-Test, zweiseitig)

Differenz = F1(erstes) − F1(zweites). B3 = Einzelmodell-Programm je Modell (verschiedene Läufe).

### B1 (ohne-optimierung)

| Paar | Seed | F1 A | F1 B | Differenz | p (t) | sig. |
|---|---|---|---|---|---|---|
| DeepSeek − Gemma | 1 | 0.798 | 0.776 | 0.021 | 0.0656 | nein |
| DeepSeek − Gemma | 2 | 0.797 | 0.774 | 0.023 | 0.0560 | nein |
| DeepSeek − Gemma | 3 | 0.795 | 0.767 | 0.028 | 0.0180 | ja |
| DeepSeek − gpt-oss | 1 | 0.798 | 0.751 | 0.047 | <0.0001 | ja |
| DeepSeek − gpt-oss | 2 | 0.797 | 0.757 | 0.039 | 0.0020 | ja |
| DeepSeek − gpt-oss | 3 | 0.795 | 0.745 | 0.050 | <0.0001 | ja |
| DeepSeek − Luna | 1 | 0.798 | 0.563 | 0.235 | <0.0001 | ja |
| DeepSeek − Luna | 2 | 0.797 | 0.567 | 0.230 | <0.0001 | ja |
| DeepSeek − Luna | 3 | 0.795 | 0.569 | 0.226 | <0.0001 | ja |
| Gemma − gpt-oss | 1 | 0.776 | 0.751 | 0.025 | 0.0392 | ja |
| Gemma − gpt-oss | 2 | 0.774 | 0.757 | 0.016 | 0.2270 | nein |
| Gemma − gpt-oss | 3 | 0.767 | 0.745 | 0.022 | 0.0878 | nein |
| Gemma − Luna | 1 | 0.776 | 0.563 | 0.213 | <0.0001 | ja |
| Gemma − Luna | 2 | 0.774 | 0.567 | 0.207 | <0.0001 | ja |
| Gemma − Luna | 3 | 0.767 | 0.569 | 0.198 | <0.0001 | ja |
| gpt-oss − Luna | 1 | 0.751 | 0.563 | 0.188 | <0.0001 | ja |
| gpt-oss − Luna | 2 | 0.757 | 0.567 | 0.191 | <0.0001 | ja |
| gpt-oss − Luna | 3 | 0.745 | 0.569 | 0.176 | <0.0001 | ja |

### B2 (gleichverteilung)

| Paar | Seed | F1 A | F1 B | Differenz | p (t) | sig. |
|---|---|---|---|---|---|---|
| DeepSeek − Gemma | 1 | 0.804 | 0.788 | 0.016 | 0.1336 | nein |
| DeepSeek − Gemma | 2 | 0.802 | 0.798 | 0.004 | 0.7156 | nein |
| DeepSeek − Gemma | 3 | 0.803 | 0.782 | 0.021 | 0.0704 | nein |
| DeepSeek − gpt-oss | 1 | 0.804 | 0.753 | 0.051 | <0.0001 | ja |
| DeepSeek − gpt-oss | 2 | 0.802 | 0.759 | 0.043 | 0.0003 | ja |
| DeepSeek − gpt-oss | 3 | 0.803 | 0.742 | 0.060 | <0.0001 | ja |
| DeepSeek − Luna | 1 | 0.804 | 0.729 | 0.074 | <0.0001 | ja |
| DeepSeek − Luna | 2 | 0.802 | 0.697 | 0.105 | <0.0001 | ja |
| DeepSeek − Luna | 3 | 0.803 | 0.707 | 0.096 | <0.0001 | ja |
| Gemma − gpt-oss | 1 | 0.788 | 0.753 | 0.035 | 0.0049 | ja |
| Gemma − gpt-oss | 2 | 0.798 | 0.759 | 0.039 | 0.0016 | ja |
| Gemma − gpt-oss | 3 | 0.782 | 0.742 | 0.040 | 0.0027 | ja |
| Gemma − Luna | 1 | 0.788 | 0.729 | 0.059 | <0.0001 | ja |
| Gemma − Luna | 2 | 0.798 | 0.697 | 0.100 | <0.0001 | ja |
| Gemma − Luna | 3 | 0.782 | 0.707 | 0.075 | <0.0001 | ja |
| gpt-oss − Luna | 1 | 0.753 | 0.729 | 0.023 | 0.0691 | nein |
| gpt-oss − Luna | 2 | 0.759 | 0.697 | 0.062 | <0.0001 | ja |
| gpt-oss − Luna | 3 | 0.742 | 0.707 | 0.035 | 0.0113 | ja |

### B3 (ein-modell)

| Paar | Seed | F1 A | F1 B | Differenz | p (t) | sig. |
|---|---|---|---|---|---|---|
| DeepSeek − Gemma | 1 | 0.804 | 0.792 | 0.012 | 0.2921 | nein |
| DeepSeek − Gemma | 2 | 0.796 | 0.798 | -0.002 | 0.8563 | nein |
| DeepSeek − Gemma | 3 | 0.800 | 0.798 | 0.003 | 0.7589 | nein |
| DeepSeek − gpt-oss | 1 | 0.804 | 0.775 | 0.029 | 0.0063 | ja |
| DeepSeek − gpt-oss | 2 | 0.796 | 0.749 | 0.047 | 0.0005 | ja |
| DeepSeek − gpt-oss | 3 | 0.800 | 0.752 | 0.049 | <0.0001 | ja |
| DeepSeek − Luna | 1 | 0.804 | 0.739 | 0.065 | <0.0001 | ja |
| DeepSeek − Luna | 2 | 0.796 | 0.724 | 0.072 | <0.0001 | ja |
| DeepSeek − Luna | 3 | 0.800 | 0.686 | 0.115 | <0.0001 | ja |
| Gemma − gpt-oss | 1 | 0.792 | 0.775 | 0.018 | 0.1546 | nein |
| Gemma − gpt-oss | 2 | 0.798 | 0.749 | 0.049 | 0.0002 | ja |
| Gemma − gpt-oss | 3 | 0.798 | 0.752 | 0.046 | 0.0003 | ja |
| Gemma − Luna | 1 | 0.792 | 0.739 | 0.053 | <0.0001 | ja |
| Gemma − Luna | 2 | 0.798 | 0.724 | 0.074 | <0.0001 | ja |
| Gemma − Luna | 3 | 0.798 | 0.686 | 0.112 | <0.0001 | ja |
| gpt-oss − Luna | 1 | 0.775 | 0.739 | 0.036 | 0.0077 | ja |
| gpt-oss − Luna | 2 | 0.749 | 0.724 | 0.025 | 0.1004 | nein |
| gpt-oss − Luna | 3 | 0.752 | 0.686 | 0.066 | <0.0001 | ja |

### Zusammenfassung: signifikant in x von 3 Seeds (t-Test)

| Paar | B1 sig. | B1 Ø Diff. | B1 p (S1, S2, S3) | B2 sig. | B2 Ø Diff. | B2 p (S1, S2, S3) | B3 sig. | B3 Ø Diff. | B3 p (S1, S2, S3) |
|---|---|---|---|---|---|---|---|---|---|
| DeepSeek − Gemma | 1/3 | 0.024 | 0.0656, 0.0560, 0.0180 | 0/3 | 0.014 | 0.1336, 0.7156, 0.0704 | 0/3 | 0.004 | 0.2921, 0.8563, 0.7589 |
| DeepSeek − gpt-oss | 3/3 | 0.045 | <0.0001, 0.0020, <0.0001 | 3/3 | 0.051 | <0.0001, 0.0003, <0.0001 | 3/3 | 0.042 | 0.0063, 0.0005, <0.0001 |
| DeepSeek − Luna | 3/3 | 0.230 | <0.0001, <0.0001, <0.0001 | 3/3 | 0.092 | <0.0001, <0.0001, <0.0001 | 3/3 | 0.084 | <0.0001, <0.0001, <0.0001 |
| Gemma − gpt-oss | 1/3 | 0.021 | 0.0392, 0.2270, 0.0878 | 3/3 | 0.038 | 0.0049, 0.0016, 0.0027 | 2/3 | 0.038 | 0.1546, 0.0002, 0.0003 |
| Gemma − Luna | 3/3 | 0.206 | <0.0001, <0.0001, <0.0001 | 3/3 | 0.078 | <0.0001, <0.0001, <0.0001 | 3/3 | 0.080 | <0.0001, <0.0001, <0.0001 |
| gpt-oss − Luna | 3/3 | 0.185 | <0.0001, <0.0001, <0.0001 | 2/3 | 0.040 | 0.0691, <0.0001, 0.0113 | 2/3 | 0.042 | 0.0077, 0.1004, <0.0001 |

**Hervorgehoben:**

- DeepSeek − Gemma, B1: Differenz je Seed S1 0.021 (p 0.0656), S2 0.023 (p 0.0560), S3 0.028 (p 0.0180); Ø 0.024; signifikant in 1/3
- DeepSeek − Gemma, B2: Differenz je Seed S1 0.016 (p 0.1336), S2 0.004 (p 0.7156), S3 0.021 (p 0.0704); Ø 0.014; signifikant in 0/3
- DeepSeek − Gemma, B3: Differenz je Seed S1 0.012 (p 0.2921), S2 -0.002 (p 0.8563), S3 0.003 (p 0.7589); Ø 0.004; signifikant in 0/3
- Gemma − gpt-oss, B1: Differenz je Seed S1 0.025 (p 0.0392), S2 0.016 (p 0.2270), S3 0.022 (p 0.0878); Ø 0.021; signifikant in 1/3
- Gemma − gpt-oss, B2: Differenz je Seed S1 0.035 (p 0.0049), S2 0.039 (p 0.0016), S3 0.040 (p 0.0027); Ø 0.038; signifikant in 3/3
- Gemma − gpt-oss, B3: Differenz je Seed S1 0.018 (p 0.1546), S2 0.049 (p 0.0002), S3 0.046 (p 0.0003); Ø 0.038; signifikant in 2/3

## 2. Dieselben Paare mit dem Wilcoxon-Vorzeichen-Rang-Test (zweiseitig)

| Strategie | Paar | Seed | Fragen mit Δ≠0 | p t-Test | p wilcox | p pratt | p zsplit |
|---|---|---|---|---|---|---|---|
| B1 | DeepSeek − Gemma | 1 | 101 | 0.0656 | 0.0684 | 0.0370 | 0.1177 |
| B1 | DeepSeek − Gemma | 2 | 106 | 0.0560 | 0.0404 | 0.0469 | 0.1297 |
| B1 | DeepSeek − Gemma | 3 | 98 | 0.0180 | 0.0166 | 0.0028 | 0.0262 |
| B1 | DeepSeek − gpt-oss | 1 | 92 | <0.0001 | <0.0001 | <0.0001 | 0.0036 |
| B1 | DeepSeek − gpt-oss | 2 | 96 | 0.0020 | 0.0016 | 0.0033 | 0.0305 |
| B1 | DeepSeek − gpt-oss | 3 | 101 | <0.0001 | <0.0001 | <0.0001 | 0.0020 |
| B1 | DeepSeek − Luna | 1 | 404 | <0.0001 | <0.0001 | <0.0001 | <0.0001 |
| B1 | DeepSeek − Luna | 2 | 402 | <0.0001 | <0.0001 | <0.0001 | <0.0001 |
| B1 | DeepSeek − Luna | 3 | 405 | <0.0001 | <0.0001 | <0.0001 | <0.0001 |
| B1 | Gemma − gpt-oss | 1 | 110 | 0.0392 | 0.0402 | 0.1446 | 0.2602 |
| B1 | Gemma − gpt-oss | 2 | 118 | 0.2270 | 0.2223 | 0.4100 | 0.5155 |
| B1 | Gemma − gpt-oss | 3 | 118 | 0.0878 | 0.0955 | 0.1708 | 0.2797 |
| B1 | Gemma − Luna | 1 | 390 | <0.0001 | <0.0001 | <0.0001 | <0.0001 |
| B1 | Gemma − Luna | 2 | 398 | <0.0001 | <0.0001 | <0.0001 | <0.0001 |
| B1 | Gemma − Luna | 3 | 395 | <0.0001 | <0.0001 | <0.0001 | <0.0001 |
| B1 | gpt-oss − Luna | 1 | 401 | <0.0001 | <0.0001 | <0.0001 | <0.0001 |
| B1 | gpt-oss − Luna | 2 | 400 | <0.0001 | <0.0001 | <0.0001 | <0.0001 |
| B1 | gpt-oss − Luna | 3 | 399 | <0.0001 | <0.0001 | <0.0001 | <0.0001 |
| B2 | DeepSeek − Gemma | 1 | 80 | 0.1336 | 0.1552 | 0.2452 | 0.4220 |
| B2 | DeepSeek − Gemma | 2 | 93 | 0.7156 | 0.6841 | 0.8878 | 0.9180 |
| B2 | DeepSeek − Gemma | 3 | 93 | 0.0704 | 0.0759 | 0.2181 | 0.3691 |
| B2 | DeepSeek − gpt-oss | 1 | 96 | <0.0001 | <0.0001 | 0.0001 | 0.0050 |
| B2 | DeepSeek − gpt-oss | 2 | 87 | 0.0003 | 0.0004 | 0.0003 | 0.0108 |
| B2 | DeepSeek − gpt-oss | 3 | 88 | <0.0001 | <0.0001 | <0.0001 | <0.0001 |
| B2 | DeepSeek − Luna | 1 | 141 | <0.0001 | <0.0001 | <0.0001 | <0.0001 |
| B2 | DeepSeek − Luna | 2 | 217 | <0.0001 | <0.0001 | <0.0001 | <0.0001 |
| B2 | DeepSeek − Luna | 3 | 183 | <0.0001 | <0.0001 | <0.0001 | <0.0001 |
| B2 | Gemma − gpt-oss | 1 | 100 | 0.0049 | 0.0049 | 0.0014 | 0.0168 |
| B2 | Gemma − gpt-oss | 2 | 112 | 0.0016 | 0.0035 | 0.0038 | 0.0247 |
| B2 | Gemma − gpt-oss | 3 | 111 | 0.0027 | 0.0021 | 0.0001 | 0.0030 |
| B2 | Gemma − Luna | 1 | 120 | <0.0001 | <0.0001 | <0.0001 | <0.0001 |
| B2 | Gemma − Luna | 2 | 202 | <0.0001 | <0.0001 | <0.0001 | <0.0001 |
| B2 | Gemma − Luna | 3 | 164 | <0.0001 | <0.0001 | <0.0001 | <0.0001 |
| B2 | gpt-oss − Luna | 1 | 145 | 0.0691 | 0.0729 | 0.0025 | 0.0111 |
| B2 | gpt-oss − Luna | 2 | 228 | <0.0001 | <0.0001 | <0.0001 | <0.0001 |
| B2 | gpt-oss − Luna | 3 | 182 | 0.0113 | 0.0038 | <0.0001 | <0.0001 |
| B3 | DeepSeek − Gemma | 1 | 78 | 0.2921 | 0.2721 | 0.3497 | 0.5220 |
| B3 | DeepSeek − Gemma | 2 | 77 | 0.8563 | 0.7855 | 0.8961 | 0.9291 |
| B3 | DeepSeek − Gemma | 3 | 75 | 0.7589 | 0.7113 | 0.8881 | 0.9243 |
| B3 | DeepSeek − gpt-oss | 1 | 90 | 0.0063 | 0.0071 | 0.0056 | 0.0457 |
| B3 | DeepSeek − gpt-oss | 2 | 100 | 0.0005 | 0.0008 | 0.0001 | 0.0045 |
| B3 | DeepSeek − gpt-oss | 3 | 89 | <0.0001 | <0.0001 | <0.0001 | 0.0040 |
| B3 | DeepSeek − Luna | 1 | 122 | <0.0001 | <0.0001 | <0.0001 | <0.0001 |
| B3 | DeepSeek − Luna | 2 | 136 | <0.0001 | <0.0001 | <0.0001 | <0.0001 |
| B3 | DeepSeek − Luna | 3 | 256 | <0.0001 | <0.0001 | <0.0001 | <0.0001 |
| B3 | Gemma − gpt-oss | 1 | 100 | 0.1546 | 0.1609 | 0.0509 | 0.1444 |
| B3 | Gemma − gpt-oss | 2 | 106 | 0.0002 | 0.0002 | 0.0004 | 0.0064 |
| B3 | Gemma − gpt-oss | 3 | 101 | 0.0003 | 0.0003 | 0.0004 | 0.0078 |
| B3 | Gemma − Luna | 1 | 123 | <0.0001 | <0.0001 | <0.0001 | <0.0001 |
| B3 | Gemma − Luna | 2 | 126 | <0.0001 | <0.0001 | <0.0001 | <0.0001 |
| B3 | Gemma − Luna | 3 | 256 | <0.0001 | <0.0001 | <0.0001 | <0.0001 |
| B3 | gpt-oss − Luna | 1 | 146 | 0.0077 | 0.0060 | 0.0002 | 0.0020 |
| B3 | gpt-oss − Luna | 2 | 151 | 0.1004 | 0.0880 | 0.0228 | 0.0533 |
| B3 | gpt-oss − Luna | 3 | 251 | <0.0001 | <0.0001 | <0.0001 | <0.0001 |

### Zusammenfassung Wilcoxon: signifikant in x von 3 Seeds (wilcox / pratt / zsplit; t-Test zum Vergleich)

| Paar | B1 | B2 | B3 |
|---|---|---|---|
| DeepSeek − Gemma | 2 / 3 / 1 (t: 1) | 0 / 0 / 0 (t: 0) | 0 / 0 / 0 (t: 0) |
| DeepSeek − gpt-oss | 3 / 3 / 3 (t: 3) | 3 / 3 / 3 (t: 3) | 3 / 3 / 3 (t: 3) |
| DeepSeek − Luna | 3 / 3 / 3 (t: 3) | 3 / 3 / 3 (t: 3) | 3 / 3 / 3 (t: 3) |
| Gemma − gpt-oss | 1 / 0 / 0 (t: 1) | 3 / 3 / 3 (t: 3) | 2 / 2 / 2 (t: 2) |
| Gemma − Luna | 3 / 3 / 3 (t: 3) | 3 / 3 / 3 (t: 3) | 3 / 3 / 3 (t: 3) |
| gpt-oss − Luna | 3 / 3 / 3 (t: 3) | 2 / 3 / 3 (t: 2) | 2 / 3 / 2 (t: 2) |

### Fälle, in denen ein Wilcoxon-Urteil vom t-Test-Urteil abweicht (Modellpaare)

| Strategie | Paar | Seed | zero_method | p t | p Wilcoxon | Richtung |
|---|---|---|---|---|---|---|
| B1 | DeepSeek − Gemma | 1 | pratt | 0.0656 | 0.0370 | Wilcoxon sig., t nicht |
| B1 | DeepSeek − Gemma | 2 | wilcox | 0.0560 | 0.0404 | Wilcoxon sig., t nicht |
| B1 | DeepSeek − Gemma | 2 | pratt | 0.0560 | 0.0469 | Wilcoxon sig., t nicht |
| B1 | Gemma − gpt-oss | 1 | pratt | 0.0392 | 0.1446 | t sig., Wilcoxon nicht |
| B1 | Gemma − gpt-oss | 1 | zsplit | 0.0392 | 0.2602 | t sig., Wilcoxon nicht |
| B2 | gpt-oss − Luna | 1 | pratt | 0.0691 | 0.0025 | Wilcoxon sig., t nicht |
| B2 | gpt-oss − Luna | 1 | zsplit | 0.0691 | 0.0111 | Wilcoxon sig., t nicht |
| B3 | gpt-oss − Luna | 2 | pratt | 0.1004 | 0.0228 | Wilcoxon sig., t nicht |

## 3. Wilcoxon-Gegenprobe für die Programmvergleiche und die Zwischenbewertung

### 3a. Verfahren gegen jede Baseline (Testdaten, zweiseitig), wie `vergleiche-je-seed-*.csv`

**abgerechnet**

| Seed | gegen | Gegner-Modell | Differenz | Fragen mit Δ≠0 | p t | t sig. | p wilcox | p pratt | p zsplit |
|---|---|---|---|---|---|---|---|---|---|
| 1 | B2 | DeepSeek | 0.014 | 38 | 0.0606 | nein | 0.0685 | 0.0236 | 0.2462 |
| 1 | B1 | DeepSeek | 0.020 | 44 | 0.0185 | ja | 0.0193 | 0.0326 | 0.2439 |
| 1 | B3 gpt-oss | gpt-oss | 0.043 | 94 | <0.0001 | ja | <0.0001 | <0.0001 | 0.0022 |
| 1 | B3 Gemma | Gemma | 0.026 | 84 | 0.0220 | ja | 0.0201 | 0.0261 | 0.1177 |
| 1 | B3 DeepSeek | DeepSeek | 0.014 | 45 | 0.0589 | nein | 0.0614 | 0.0515 | 0.2837 |
| 1 | B3 Luna | Luna | 0.079 | 131 | <0.0001 | ja | <0.0001 | <0.0001 | <0.0001 |
| 2 | B2 | DeepSeek | -0.004 | 36 | 0.6296 | nein | 0.5236 | 0.5035 | 0.7378 |
| 2 | B1 | DeepSeek | 0.002 | 47 | 0.8353 | nein | 0.9746 | 0.4899 | 0.6986 |
| 2 | B3 gpt-oss | gpt-oss | 0.049 | 99 | 0.0002 | ja | 0.0002 | <0.0001 | 0.0033 |
| 2 | B3 Gemma | Gemma | 0.000 | 67 | 0.9929 | nein | 0.8389 | 0.7214 | 0.8177 |
| 2 | B3 DeepSeek | DeepSeek | 0.002 | 39 | 0.8205 | nein | 0.8005 | 0.8875 | 0.9416 |
| 2 | B3 Luna | Luna | 0.074 | 127 | <0.0001 | ja | <0.0001 | <0.0001 | <0.0001 |
| 3 | B2 | gpt-oss | 0.058 | 100 | <0.0001 | ja | <0.0001 | <0.0001 | 0.0002 |
| 3 | B1 | DeepSeek | 0.005 | 51 | 0.5628 | nein | 0.7423 | 0.6763 | 0.8087 |
| 3 | B3 gpt-oss | gpt-oss | 0.049 | 92 | <0.0001 | ja | <0.0001 | 0.0001 | 0.0051 |
| 3 | B3 Gemma | Gemma | 0.002 | 85 | 0.8082 | nein | 0.7956 | 0.9440 | 0.9604 |
| 3 | B3 DeepSeek | DeepSeek | -0.000 | 32 | 0.9582 | nein | 0.8956 | 0.7366 | 0.8730 |
| 3 | B3 Luna | Luna | 0.114 | 259 | <0.0001 | ja | <0.0001 | <0.0001 | <0.0001 |

**festgehalten**

| Seed | gegen | Gegner-Modell | Differenz | Fragen mit Δ≠0 | p t | t sig. | p wilcox | p pratt | p zsplit |
|---|---|---|---|---|---|---|---|---|---|
| 1 | B2 | DeepSeek | 0.014 | 38 | 0.0606 | nein | 0.0685 | 0.0236 | 0.2462 |
| 1 | B1 | DeepSeek | 0.020 | 44 | 0.0185 | ja | 0.0193 | 0.0326 | 0.2439 |
| 1 | B3 gpt-oss | gpt-oss | 0.043 | 94 | <0.0001 | ja | <0.0001 | <0.0001 | 0.0022 |
| 1 | B3 Gemma | Gemma | 0.026 | 84 | 0.0220 | ja | 0.0201 | 0.0261 | 0.1177 |
| 1 | B3 DeepSeek | DeepSeek | 0.014 | 45 | 0.0589 | nein | 0.0614 | 0.0515 | 0.2837 |
| 1 | B3 Luna | Luna | 0.079 | 131 | <0.0001 | ja | <0.0001 | <0.0001 | <0.0001 |
| 2 | B2 | DeepSeek | -0.004 | 36 | 0.6296 | nein | 0.5236 | 0.5035 | 0.7378 |
| 2 | B1 | DeepSeek | 0.002 | 47 | 0.8353 | nein | 0.9746 | 0.4899 | 0.6986 |
| 2 | B3 gpt-oss | gpt-oss | 0.049 | 99 | 0.0002 | ja | 0.0002 | <0.0001 | 0.0033 |
| 2 | B3 Gemma | Gemma | 0.000 | 67 | 0.9929 | nein | 0.8389 | 0.7214 | 0.8177 |
| 2 | B3 DeepSeek | DeepSeek | 0.002 | 39 | 0.8205 | nein | 0.8005 | 0.8875 | 0.9416 |
| 2 | B3 Luna | Luna | 0.074 | 127 | <0.0001 | ja | <0.0001 | <0.0001 | <0.0001 |
| 3 | B2 | DeepSeek | -0.003 | 27 | 0.7045 | nein | 0.6552 | 0.5651 | 0.7998 |
| 3 | B1 | DeepSeek | 0.005 | 51 | 0.5628 | nein | 0.7423 | 0.6763 | 0.8087 |
| 3 | B3 gpt-oss | gpt-oss | 0.049 | 92 | <0.0001 | ja | <0.0001 | 0.0001 | 0.0051 |
| 3 | B3 Gemma | Gemma | 0.002 | 85 | 0.8082 | nein | 0.7956 | 0.9440 | 0.9604 |
| 3 | B3 DeepSeek | DeepSeek | -0.000 | 32 | 0.9582 | nein | 0.8956 | 0.7366 | 0.8730 |
| 3 | B3 Luna | Luna | 0.114 | 259 | <0.0001 | ja | <0.0001 | <0.0001 | <0.0001 |

Abweichende Urteile (Wilcoxon vs. t-Test):

| Preis | Seed | gegen | zero_method | p t | p Wilcoxon |
|---|---|---|---|---|---|
| abgerechnet | 1 | B2 | pratt | 0.0606 | 0.0236 |
| abgerechnet | 1 | B1 | zsplit | 0.0185 | 0.2439 |
| abgerechnet | 1 | B3 Gemma | zsplit | 0.0220 | 0.1177 |
| festgehalten | 1 | B2 | pratt | 0.0606 | 0.0236 |
| festgehalten | 1 | B1 | zsplit | 0.0185 | 0.2439 |
| festgehalten | 1 | B3 Gemma | zsplit | 0.0220 | 0.1177 |

### 3b. Zwischenbewertung des Verfahrens (Auswahldaten, n = 200, einseitig `less`, gegen den Führenden)

Regel aus ADR-0004 (Nachtrag 08./09.10.2026) unverändert, nur der Test getauscht. Führender = höchster Mittelwert.

**ohne Kostenobergrenze (wie `zwischenbewertungen-ohne-obergrenze.csv`)**

| Seed | Test | p gpt-oss | p Gemma | p DeepSeek | p Luna | weiter |
|---|---|---|---|---|---|---|
| 1 | t-Test | 0.0026 (Δ≠0: 26) | 0.0351 (Δ≠0: 36) | Führender | <0.0001 (Δ≠0: 92) | DeepSeek |
| 1 | Wilcoxon wilcox | 0.0030 (Δ≠0: 26) | 0.0316 (Δ≠0: 36) | Führender | <0.0001 (Δ≠0: 92) | DeepSeek |
| 1 | Wilcoxon pratt | 0.0028 (Δ≠0: 26) | 0.0791 (Δ≠0: 36) | Führender | <0.0001 (Δ≠0: 92) | DeepSeek, Gemma |
| 1 | Wilcoxon zsplit | 0.0385 (Δ≠0: 26) | 0.1547 (Δ≠0: 36) | Führender | <0.0001 (Δ≠0: 92) | DeepSeek, Gemma |
| 2 | t-Test | 0.0004 (Δ≠0: 26) | 0.0143 (Δ≠0: 31) | Führender | <0.0001 (Δ≠0: 79) | DeepSeek |
| 2 | Wilcoxon wilcox | 0.0003 (Δ≠0: 26) | 0.0192 (Δ≠0: 31) | Führender | <0.0001 (Δ≠0: 79) | DeepSeek |
| 2 | Wilcoxon pratt | 0.0002 (Δ≠0: 26) | 0.0882 (Δ≠0: 31) | Führender | <0.0001 (Δ≠0: 79) | DeepSeek, Gemma |
| 2 | Wilcoxon zsplit | 0.0117 (Δ≠0: 26) | 0.1782 (Δ≠0: 31) | Führender | <0.0001 (Δ≠0: 79) | DeepSeek, Gemma |
| 3 | t-Test | 0.0006 (Δ≠0: 30) | 0.0011 (Δ≠0: 41) | Führender | <0.0001 (Δ≠0: 89) | DeepSeek |
| 3 | Wilcoxon wilcox | 0.0005 (Δ≠0: 30) | 0.0010 (Δ≠0: 41) | Führender | <0.0001 (Δ≠0: 89) | DeepSeek |
| 3 | Wilcoxon pratt | 0.0001 (Δ≠0: 30) | 0.0066 (Δ≠0: 41) | Führender | <0.0001 (Δ≠0: 89) | DeepSeek |
| 3 | Wilcoxon zsplit | 0.0069 (Δ≠0: 30) | 0.0310 (Δ≠0: 41) | Führender | <0.0001 (Δ≠0: 89) | DeepSeek |

**mit Kostenobergrenze (wie im Experiment)**

| Seed | Test | p gpt-oss | p Gemma | p DeepSeek | p Luna | weiter |
|---|---|---|---|---|---|---|
| 1 | t-Test | 0.0026 (Δ≠0: 26) | kostenfilter | Führender | kostenfilter | DeepSeek |
| 1 | Wilcoxon wilcox | 0.0030 (Δ≠0: 26) | kostenfilter | Führender | kostenfilter | DeepSeek |
| 1 | Wilcoxon pratt | 0.0028 (Δ≠0: 26) | kostenfilter | Führender | kostenfilter | DeepSeek |
| 1 | Wilcoxon zsplit | 0.0385 (Δ≠0: 26) | kostenfilter | Führender | kostenfilter | DeepSeek |
| 2 | t-Test | 0.0004 (Δ≠0: 26) | kostenfilter | Führender | kostenfilter | DeepSeek |
| 2 | Wilcoxon wilcox | 0.0003 (Δ≠0: 26) | kostenfilter | Führender | kostenfilter | DeepSeek |
| 2 | Wilcoxon pratt | 0.0002 (Δ≠0: 26) | kostenfilter | Führender | kostenfilter | DeepSeek |
| 2 | Wilcoxon zsplit | 0.0117 (Δ≠0: 26) | kostenfilter | Führender | kostenfilter | DeepSeek |
| 3 | t-Test | 0.0006 (Δ≠0: 30) | kostenfilter | Führender | kostenfilter | DeepSeek |
| 3 | Wilcoxon wilcox | 0.0005 (Δ≠0: 30) | kostenfilter | Führender | kostenfilter | DeepSeek |
| 3 | Wilcoxon pratt | 0.0001 (Δ≠0: 30) | kostenfilter | Führender | kostenfilter | DeepSeek |
| 3 | Wilcoxon zsplit | 0.0069 (Δ≠0: 30) | kostenfilter | Führender | kostenfilter | DeepSeek |

Zweites Modell in der Fokusphase unter einer Testvariante: ohne Kostenobergrenze, Seed 1, Wilcoxon pratt: DeepSeek, Gemma; ohne Kostenobergrenze, Seed 1, Wilcoxon zsplit: DeepSeek, Gemma; ohne Kostenobergrenze, Seed 2, Wilcoxon pratt: DeepSeek, Gemma; ohne Kostenobergrenze, Seed 2, Wilcoxon zsplit: DeepSeek, Gemma

F1 der Sondierung (Auswahldaten) je Seed, zur Einordnung:

| Seed | F1 gpt-oss | F1 Gemma | F1 DeepSeek | F1 Luna | $/1000 gpt-oss | $/1000 Gemma | $/1000 DeepSeek | $/1000 Luna |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.816 | 0.830 | 0.864 | 0.718 | 0.129 | 0.333 | 0.268 | 0.617 |
| 2 | 0.806 | 0.828 | 0.861 | 0.758 | 0.197 | 0.553 | 0.233 | 1.039 |
| 3 | 0.805 | 0.807 | 0.863 | 0.706 | 0.112 | 0.470 | 0.220 | 0.879 |

## 4. FF1 mit vollem Budget: Rangfolge B1 gegen B3 (Testdaten)

| Seed | Kendalls τ | Rang B1 | Rang B3 (F1) | Spitze B1 | Spitze B3 |
|---|---|---|---|---|---|
| 1 | 1.000 | DeepSeek > Gemma > gpt-oss > Luna | DeepSeek 0.804 > Gemma 0.792 > gpt-oss 0.775 > Luna 0.739 | DeepSeek | DeepSeek |
| 2 | 0.667 | DeepSeek > Gemma > gpt-oss > Luna | Gemma 0.798 > DeepSeek 0.796 > gpt-oss 0.749 > Luna 0.724 | DeepSeek | Gemma |
| 3 | 1.000 | DeepSeek > Gemma > gpt-oss > Luna | DeepSeek 0.800 > Gemma 0.798 > gpt-oss 0.752 > Luna 0.686 | DeepSeek | DeepSeek |

τ Mittel ± SD: 0.889 ± 0.192

Platzwechsel (B3 gegen B1), t-Test zweiseitig zwischen den beiden B3-Programmen (p_B3) und den beiden B1-Programmen (p_B1):

| Seed | Wechsel (Ordnung in B3) | Δ F1 in B3 | p_B3 (t) | p_B1 (t) | sig. | p_B3 wilcox | p_B3 pratt | p_B3 zsplit |
|---|---|---|---|---|---|---|---|---|
| 2 | Gemma vor DeepSeek | 0.002 | 0.8563 | 0.0560 | nein | 0.7855 | 0.8961 | 0.9291 |

- DeepSeek / Gemma: Wechsel in 1 von 3 Seeds, davon signifikant 0; belastbar (≥ 2 von 3): nein

## 5. Kleinster nachweisbarer Unterschied (MDE), 80 % Power, zweiseitig α = 0,05, n = 500 gepaart

MDE = (z_0,975 + z_0,8) · SD_diff / √n = 2.8016 · SD_diff / √500; SD_diff = Stichproben-SD der Differenzen je Frage.

| Vergleich | Programme | Seed | SD_diff | MDE | Fragen mit Δ≠0 |
|---|---|---|---|---|---|
| Verfahren − B1 | DeepSeek − DeepSeek | 1 | 0.192 | 0.024 | 44 |
| Verfahren − B2 (abgerechnet) | DeepSeek − DeepSeek | 1 | 0.170 | 0.021 | 38 |
| Verfahren − B2 (festgehalten) | DeepSeek − DeepSeek | 1 | 0.170 | 0.021 | 38 |
| Verfahren − B3 DeepSeek | DeepSeek − DeepSeek | 1 | 0.164 | 0.021 | 45 |
| DeepSeek − Gemma (B1) | DeepSeek − Gemma | 1 | 0.259 | 0.033 | 101 |
| Gemma − gpt-oss (B1) | Gemma − gpt-oss | 1 | 0.272 | 0.034 | 110 |
| gpt-oss − Luna (B1) | gpt-oss − Luna | 1 | 0.329 | 0.041 | 401 |
| DeepSeek − Gemma (B2) | DeepSeek − Gemma | 1 | 0.235 | 0.029 | 80 |
| Gemma − gpt-oss (B2) | Gemma − gpt-oss | 1 | 0.279 | 0.035 | 100 |
| gpt-oss − Luna (B2) | gpt-oss − Luna | 1 | 0.285 | 0.036 | 145 |
| Verfahren − B1 | DeepSeek − DeepSeek | 2 | 0.186 | 0.023 | 47 |
| Verfahren − B2 (abgerechnet) | DeepSeek − DeepSeek | 2 | 0.170 | 0.021 | 36 |
| Verfahren − B2 (festgehalten) | DeepSeek − DeepSeek | 2 | 0.170 | 0.021 | 36 |
| Verfahren − B3 DeepSeek | DeepSeek − DeepSeek | 2 | 0.203 | 0.025 | 39 |
| DeepSeek − Gemma (B1) | DeepSeek − Gemma | 2 | 0.268 | 0.034 | 106 |
| Gemma − gpt-oss (B1) | Gemma − gpt-oss | 2 | 0.297 | 0.037 | 118 |
| gpt-oss − Luna (B1) | gpt-oss − Luna | 2 | 0.323 | 0.040 | 400 |
| DeepSeek − Gemma (B2) | DeepSeek − Gemma | 2 | 0.264 | 0.033 | 93 |
| Gemma − gpt-oss (B2) | Gemma − gpt-oss | 2 | 0.272 | 0.034 | 112 |
| gpt-oss − Luna (B2) | gpt-oss − Luna | 2 | 0.335 | 0.042 | 228 |
| Verfahren − B1 | DeepSeek − DeepSeek | 3 | 0.200 | 0.025 | 51 |
| Verfahren − B2 (abgerechnet) | DeepSeek − gpt-oss | 3 | 0.262 | 0.033 | 100 |
| Verfahren − B2 (festgehalten) | DeepSeek − DeepSeek | 3 | 0.159 | 0.020 | 27 |
| Verfahren − B3 DeepSeek | DeepSeek − DeepSeek | 3 | 0.151 | 0.019 | 32 |
| DeepSeek − Gemma (B1) | DeepSeek − Gemma | 3 | 0.267 | 0.033 | 98 |
| Gemma − gpt-oss (B1) | Gemma − gpt-oss | 3 | 0.287 | 0.036 | 118 |
| gpt-oss − Luna (B1) | gpt-oss − Luna | 3 | 0.328 | 0.041 | 399 |
| DeepSeek − Gemma (B2) | DeepSeek − Gemma | 3 | 0.256 | 0.032 | 93 |
| Gemma − gpt-oss (B2) | Gemma − gpt-oss | 3 | 0.294 | 0.037 | 111 |
| gpt-oss − Luna (B2) | gpt-oss − Luna | 3 | 0.311 | 0.039 | 182 |

Spannen je Vergleich:

| Vergleich | MDE min … max |
|---|---|
| Verfahren − B1 | 0.023 … 0.025 |
| Verfahren − B2 (abgerechnet) | 0.021 … 0.033 |
| Verfahren − B2 (festgehalten) | 0.020 … 0.021 |
| Verfahren − B3 DeepSeek | 0.019 … 0.025 |
| DeepSeek − Gemma (B1) | 0.033 … 0.034 |
| Gemma − gpt-oss (B1) | 0.034 … 0.037 |
| gpt-oss − Luna (B1) | 0.040 … 0.041 |
| DeepSeek − Gemma (B2) | 0.029 … 0.033 |
| Gemma − gpt-oss (B2) | 0.034 … 0.037 |
| gpt-oss − Luna (B2) | 0.036 … 0.042 |

- Beide Programme DeepSeek (Verfahren gegen B1, B2 mit DeepSeek, B3 DeepSeek): 0.019 … 0.025
- Verschiedene Modelle: 0.029 … 0.042

## 6. 95-%-t-Intervall über die Seeds (df = 2) für die mittlere Differenz

t_0,975;2 = 4.3027

| Differenz | Preis | je Seed | Mittel | SD | untere Grenze | obere Grenze |
|---|---|---|---|---|---|---|
| Verfahren − B1 | beide gleich | 0.020, 0.002, 0.005 | 0.009 | 0.010 | -0.015 | 0.034 |
| Verfahren − B2 | abgerechnet | 0.014, -0.004, 0.058 | 0.023 | 0.032 | -0.056 | 0.101 |
| Verfahren − B3 DeepSeek | beide gleich | 0.014, 0.002, -0.000 | 0.005 | 0.008 | -0.014 | 0.024 |
| Verfahren − B3 Gemma | beide gleich | 0.026, 0.000, 0.002 | 0.009 | 0.014 | -0.026 | 0.044 |
| Verfahren − B3 gpt-oss | beide gleich | 0.043, 0.049, 0.049 | 0.047 | 0.003 | 0.039 | 0.056 |
| Verfahren − B3 Luna | beide gleich | 0.079, 0.074, 0.114 | 0.089 | 0.022 | 0.035 | 0.143 |
| Verfahren − B2 | festgehalten | 0.014, -0.004, -0.003 | 0.003 | 0.010 | -0.022 | 0.028 |

## 7. FF4 je Seed, beide Preisfassungen

**abgerechnet**

| Seed | Verf. C_opt $ | Verf. c_exec $/1000 | B1 c_exec $/1000 | Verf. C_opt in Läufen | Verf. Break-even vs B1 | B3 gpt-oss C_opt $ | B3 gpt-oss c_exec $/1000 | Faktor gpt-oss/B1 | gpt-oss Break-even vs B1 | ΔF1 gpt-oss − B1 | ΔF1 gpt-oss − Verfahren |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 5.809 | 0.296 | 0.288 | 19,602 | keiner | 5.154 | 0.180 | 0.626 | 47,950 | -0.023 | -0.043 |
| 2 | 5.671 | 0.275 | 0.288 | 20,622 | 433,923 | 6.583 | 0.148 | 0.515 | 47,112 | -0.048 | -0.049 |
| 3 | 5.528 | 0.281 | 0.289 | 19,683 | 705,007 | 6.119 | 0.113 | 0.392 | 34,863 | -0.043 | -0.049 |

**festgehalten**

| Seed | Verf. C_opt $ | Verf. c_exec $/1000 | B1 c_exec $/1000 | Verf. C_opt in Läufen | Verf. Break-even vs B1 | B3 gpt-oss C_opt $ | B3 gpt-oss c_exec $/1000 | Faktor gpt-oss/B1 | gpt-oss Break-even vs B1 | ΔF1 gpt-oss − B1 | ΔF1 gpt-oss − Verfahren |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 5.809 | 0.218 | 0.222 | 26,655 | 1,534,789 | 5.154 | 0.180 | 0.813 | 124,204 | -0.023 | -0.043 |
| 2 | 5.671 | 0.207 | 0.222 | 27,392 | 378,211 | 6.583 | 0.148 | 0.668 | 89,333 | -0.048 | -0.049 |
| 3 | 5.374 | 0.214 | 0.223 | 25,129 | 616,141 | 6.119 | 0.113 | 0.508 | 55,931 | -0.043 | -0.049 |

Schnittpunkt der mittleren kumulierten Kostenlinien B1 und B3 gpt-oss (Mittel von C_opt und c_exec über die Seeds):

| Preis | Ø C_opt gpt-oss B3 $ | Ø c_exec gpt-oss $/1000 | Ø c_exec B1 $/1000 | Ø Bewertungskosten B1 $ | Schnitt (B1 ohne Fixkosten) | Schnitt (B1 mit 0,167 $ Fixkosten) |
|---|---|---|---|---|---|---|
| abgerechnet | 5.952 | 0.1472 | 0.2882 | 0.167 | 42,239 | 41,053 |
| festgehalten | 5.952 | 0.1472 | 0.2221 | 0.167 | 79,505 | 77,277 |

## 8. Planabweichung je Optimierungslauf

Plan = `dollar_geplant` der Optimierungszeile = Kostenmodell für die geplanten Trials (Fixkosten + num_trials · Kosten je Trial, ADR-0014/0017), **ohne** Nachbewertung. Ist = `dollar` der Optimierungszeile (abgerechnet; festgehalten nach ADR-0028). Daneben die Abweichung vom Budgetanteil (`dollar_budgetanteil`). Nur Läufe gültiger Durchgänge.

| Seed | DG | Strategie/Phase | Modell | Trials | Budgetanteil $ | Plan $ | Ist $ | Abw. vom Plan (abger.) | Abw. vom Plan (fest) | Abw. vom Budgetanteil (abger.) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | Verfahren sondierung | gpt-oss | 41 | 0.750 | 0.705 | 0.698 | -1.0 % | -1.0 % | -6.9 % |
| 1 | 2 | Verfahren sondierung | Gemma | 25 | 0.750 | 0.690 | 0.716 | +3.8 % | +3.8 % | -4.6 % |
| 1 | 2 | Verfahren sondierung | DeepSeek | 39 | 0.750 | 0.698 | 0.770 | +10.3 % | +10.3 % | +2.6 % |
| 1 | 2 | Verfahren sondierung | Luna | 13 | 0.750 | 0.569 | 0.830 | +45.9 % | +45.9 % | +10.6 % |
| 1 | 2 | Verfahren fokus | DeepSeek | 156 | 2.718 | 2.665 | 2.475 | -7.1 % | -7.1 % | -8.9 % |
| 1 | 3 | B2 | gpt-oss | 87 | 1.500 | 1.461 | 1.305 | -10.7 % | -10.7 % | -13.0 % |
| 1 | 3 | B2 | Gemma | 54 | 1.500 | 1.430 | 1.544 | +8.0 % | +8.0 % | +3.0 % |
| 1 | 3 | B2 | DeepSeek | 84 | 1.500 | 1.454 | 1.282 | -11.9 % | -11.9 % | -14.5 % |
| 1 | 3 | B2 | Luna | 32 | 1.500 | 1.311 | 1.375 | +4.9 % | +4.9 % | -8.3 % |
| 1 | 4 | B3 | gpt-oss | 361 | 6.000 | 5.960 | 5.118 | -14.1 % | -14.1 % | -14.7 % |
| 1 | 5 | B3 | Gemma | 230 | 6.000 | 5.921 | 5.836 | -1.4 % | -1.4 % | -2.7 % |
| 1 | 6 | B3 | DeepSeek | 351 | 6.000 | 5.944 | 3.960 | -33.4 % | -33.4 % | -34.0 % |
| 1 | 7 | B3 | Luna | 147 | 6.000 | 5.804 | 4.641 | -20.0 % | -20.0 % | -22.6 % |
| 2 | 12 | Verfahren sondierung | gpt-oss | 41 | 0.750 | 0.705 | 0.759 | +7.7 % | +7.7 % | +1.3 % |
| 2 | 12 | Verfahren sondierung | Gemma | 25 | 0.750 | 0.690 | 0.689 | -0.0 % | -0.0 % | -8.1 % |
| 2 | 12 | Verfahren sondierung | DeepSeek | 39 | 0.750 | 0.698 | 0.835 | +19.7 % | +19.7 % | +11.3 % |
| 2 | 12 | Verfahren sondierung | Luna | 13 | 0.750 | 0.569 | 0.791 | +39.1 % | +39.1 % | +5.5 % |
| 2 | 12 | Verfahren fokus | DeepSeek | 144 | 2.520 | 2.463 | 2.146 | -12.9 % | -12.9 % | -14.9 % |
| 2 | 13 | B2 | gpt-oss | 87 | 1.500 | 1.461 | 1.543 | +5.7 % | +5.7 % | +2.9 % |
| 2 | 13 | B2 | Gemma | 54 | 1.500 | 1.430 | 1.698 | +18.8 % | +18.8 % | +13.2 % |
| 2 | 13 | B2 | DeepSeek | 84 | 1.500 | 1.454 | 1.311 | -9.9 % | -9.9 % | -12.6 % |
| 2 | 13 | B2 | Luna | 32 | 1.500 | 1.311 | 1.674 | +27.7 % | +27.7 % | +11.6 % |
| 2 | 15 | B3 | gpt-oss | 361 | 6.000 | 5.960 | 6.554 | +10.0 % | +10.0 % | +9.2 % |
| 2 | 17 | B3 | Gemma | 230 | 6.000 | 5.921 | 6.545 | +10.5 % | +10.5 % | +9.1 % |
| 2 | 18 | B3 | DeepSeek | 351 | 6.000 | 5.944 | 4.638 | -22.0 % | -22.0 % | -22.7 % |
| 2 | 19 | B3 | Luna | 147 | 6.000 | 5.804 | 5.606 | -3.4 % | -3.4 % | -6.6 % |
| 3 | 21 | Verfahren sondierung | gpt-oss | 41 | 0.750 | 0.705 | 0.842 | +19.4 % | +19.4 % | +12.3 % |
| 3 | 21 | Verfahren sondierung | Gemma | 25 | 0.750 | 0.690 | 0.847 | +22.8 % | +22.8 % | +12.9 % |
| 3 | 21 | Verfahren sondierung | DeepSeek | 39 | 0.750 | 0.698 | 0.742 | +6.4 % | -0.1 % | -1.1 % |
| 3 | 21 | Verfahren sondierung | Luna | 13 | 0.750 | 0.569 | 0.802 | +40.9 % | +40.9 % | +6.9 % |
| 3 | 21 | Verfahren fokus | DeepSeek | 139 | 2.431 | 2.379 | 1.915 | -19.5 % | -23.9 % | -21.2 % |
| 3 | 23 | B2 | gpt-oss | 87 | 1.500 | 1.461 | 1.625 | +11.2 % | +11.2 % | +8.3 % |
| 3 | 23 | B2 | Gemma | 54 | 1.500 | 1.430 | 1.706 | +19.3 % | +19.3 % | +13.7 % |
| 3 | 23 | B2 | DeepSeek | 84 | 1.500 | 1.454 | 1.821 | +25.2 % | +17.5 % | +21.4 % |
| 3 | 23 | B2 | Luna | 32 | 1.500 | 1.311 | 1.651 | +25.9 % | +25.9 % | +10.1 % |
| 3 | 24 | B3 | gpt-oss | 361 | 6.000 | 5.960 | 6.097 | +2.3 % | +2.3 % | +1.6 % |
| 3 | 25 | B3 | Gemma | 230 | 6.000 | 5.921 | 7.432 | +25.5 % | +25.5 % | +23.9 % |
| 3 | 26 | B3 | DeepSeek | 351 | 6.000 | 5.944 | 7.460 | +25.5 % | -8.0 % | +24.3 % |
| 3 | 27 | B3 | Luna | 147 | 6.000 | 5.804 | 4.571 | -21.3 % | -21.3 % | -23.8 % |

Je Strategie/Phase und Modell (Spanne über die Seeds):

| Strategie/Phase | Modell | Abw. abgerechnet | Abw. festgehalten |
|---|---|---|---|
| Verfahren sondierung | gpt-oss | -1.0 % … +19.4 % | -1.0 % … +19.4 % |
| Verfahren sondierung | Gemma | -0.0 % … +22.8 % | -0.0 % … +22.8 % |
| Verfahren sondierung | DeepSeek | +6.4 % … +19.7 % | -0.1 % … +19.7 % |
| Verfahren sondierung | Luna | +39.1 % … +45.9 % | +39.1 % … +45.9 % |
| Verfahren fokus | DeepSeek | -19.5 % … -7.1 % | -23.9 % … -7.1 % |
| B2 | gpt-oss | -10.7 % … +11.2 % | -10.7 % … +11.2 % |
| B2 | Gemma | +8.0 % … +19.3 % | +8.0 % … +19.3 % |
| B2 | DeepSeek | -11.9 % … +25.2 % | -11.9 % … +17.5 % |
| B2 | Luna | +4.9 % … +27.7 % | +4.9 % … +27.7 % |
| B3 | gpt-oss | -14.1 % … +10.0 % | -14.1 % … +10.0 % |
| B3 | Gemma | -1.4 % … +25.5 % | -1.4 % … +25.5 % |
| B3 | DeepSeek | -33.4 % … +25.5 % | -33.4 % … -8.0 % |
| B3 | Luna | -21.3 % … -3.4 % | -21.3 % … -3.4 % |

Alle Optimierungsläufe: -33.4 % … +45.9 % (abgerechnet), -33.4 % … +45.9 % (festgehalten).

Je Durchgang (wie `kosten-je-durchgang-*.csv`: C_opt = Optimierung + Nachbewertung gegen die Summe der geplanten Dollar beider Messläufe):

| Preis | Abweichung je Durchgang |
|---|---|
| abgerechnet | -33.0 % … +25.8 % |
| abgerechnet: Mittel je Strategie | Verfahren +2.6 %; B2 +8.9 %; B3 gpt-oss -0.7 %; B3 Gemma +11.9 %; B3 DeepSeek -9.7 %; B3 Luna -15.5 % |
| abgerechnet: Spanne der Strategie-Mittel | -15.5 % … +11.9 % |
| festgehalten | -33.0 % … +25.8 % |
| festgehalten: Mittel je Strategie | Verfahren +1.7 %; B2 +8.1 %; B3 gpt-oss -0.7 %; B3 Gemma +11.9 %; B3 DeepSeek -20.9 %; B3 Luna -15.5 % |
| festgehalten: Spanne der Strategie-Mittel | -20.9 % … +11.9 % |

## 9. Messrauschen: B1 je Seed und Modell

| Modell | Seed | F1 Auswahl | F1 Test | $/1000 Auswahl | $/1000 Auswahl fest | $/1000 Test | $/1000 Test fest |
|---|---|---|---|---|---|---|---|
| gpt-oss | 1 | 0.797 | 0.751 | 0.1132 | 0.1132 | 0.1129 | 0.1129 |
| gpt-oss | 2 | 0.788 | 0.757 | 0.1131 | 0.1131 | 0.1132 | 0.1132 |
| gpt-oss | 3 | 0.788 | 0.745 | 0.1118 | 0.1118 | 0.1130 | 0.1130 |
| gpt-oss | Ø ± SD | 0.791 ± 0.006 | 0.751 ± 0.006 | 0.1127 ± 0.0008 | 0.1127 ± 0.0008 | 0.1130 ± 0.0002 | 0.1130 ± 0.0002 |
| gpt-oss | Spanne max−min | 0.0099 | 0.0128 | 0.0014 | 0.0014 | 0.0003 | 0.0003 |
| Gemma | 1 | 0.799 | 0.776 | 0.2076 | 0.2076 | 0.2068 | 0.2068 |
| Gemma | 2 | 0.786 | 0.774 | 0.2074 | 0.2074 | 0.2064 | 0.2064 |
| Gemma | 3 | 0.785 | 0.767 | 0.2074 | 0.2074 | 0.2068 | 0.2068 |
| Gemma | Ø ± SD | 0.790 ± 0.008 | 0.772 ± 0.005 | 0.2074 ± 0.0001 | 0.2074 ± 0.0001 | 0.2067 ± 0.0002 | 0.2067 ± 0.0002 |
| Gemma | Spanne max−min | 0.0138 | 0.0097 | 0.0002 | 0.0002 | 0.0004 | 0.0004 |
| DeepSeek | 1 | 0.851 | 0.798 | 0.2399 | 0.2399 | 0.2877 | 0.2217 |
| DeepSeek | 2 | 0.851 | 0.797 | 0.2183 | 0.2183 | 0.2881 | 0.2220 |
| DeepSeek | 3 | 0.836 | 0.795 | 0.2253 | 0.2191 | 0.2887 | 0.2226 |
| DeepSeek | Ø ± SD | 0.846 ± 0.009 | 0.796 ± 0.001 | 0.2278 ± 0.0110 | 0.2258 ± 0.0123 | 0.2882 ± 0.0005 | 0.2221 ± 0.0004 |
| DeepSeek | Spanne max−min | 0.0150 | 0.0028 | 0.0217 | 0.0217 | 0.0010 | 0.0009 |
| Luna | 1 | 0.607 | 0.563 | 0.2871 | 0.2871 | 0.2879 | 0.2879 |
| Luna | 2 | 0.586 | 0.567 | 0.2882 | 0.2882 | 0.2886 | 0.2886 |
| Luna | 3 | 0.606 | 0.569 | 0.2886 | 0.2886 | 0.2881 | 0.2881 |
| Luna | Ø ± SD | 0.600 ± 0.012 | 0.566 ± 0.003 | 0.2880 ± 0.0008 | 0.2880 ± 0.0008 | 0.2882 ± 0.0004 | 0.2882 ± 0.0004 |
| Luna | Spanne max−min | 0.0211 | 0.0054 | 0.0015 | 0.0015 | 0.0007 | 0.0007 |

## 10. Lücke Auswahl − Test (B1) und Vorsprung DeepSeek vor Gemma

| Modell (B1) | S1 | S2 | S3 | Ø ± SD |
|---|---|---|---|---|
| gpt-oss | 0.046 | 0.030 | 0.043 | 0.040 ± 0.009 |
| Gemma | 0.022 | 0.012 | 0.018 | 0.018 ± 0.005 |
| DeepSeek | 0.053 | 0.054 | 0.041 | 0.050 ± 0.007 |
| Luna | 0.044 | 0.019 | 0.038 | 0.034 ± 0.013 |

| Datensatz | Δ S1 | Δ S2 | Δ S3 | Ø Δ | Ø F1 DeepSeek | Ø F1 Gemma |
|---|---|---|---|---|---|---|
| B1 Auswahl | 0.052 | 0.065 | 0.051 | 0.056 | 0.846 | 0.790 |
| B1 Test | 0.021 | 0.023 | 0.028 | 0.024 | 0.796 | 0.772 |
| Sondierung B/8 Auswahl | 0.034 | 0.033 | 0.056 | 0.041 | 0.863 | 0.821 |
| B2 Auswahl | 0.047 | 0.024 | 0.041 | 0.037 | 0.860 | 0.823 |
| B2 Test | 0.016 | 0.004 | 0.021 | 0.014 | 0.803 | 0.789 |

Sondierung hat keine Testbewertung (nur DeepSeeks Fokus-Programm wurde getestet).

## 11. Latenz (Median je Testbewertung, s): B1 gegen B2 und B3

| Modell | B1 S1/S2/S3 | B1 Ø | B2 S1/S2/S3 | B2 Ø | Δ B2−B1 | Seeds langsamer | B3 S1/S2/S3 | B3 Ø | Δ B3−B1 | Seeds langsamer |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt-oss | 4.31 / 4.02 / 4.59 | 4.30 | 3.82 / 3.64 / 4.37 | 3.94 | -0.36 | 0/3 | 4.92 / 5.82 / 5.60 | 5.45 | +1.14 | 3/3 |
| Gemma | 3.30 / 3.49 / 3.44 | 3.41 | 3.73 / 4.06 / 3.95 | 3.91 | +0.51 | 3/3 | 3.87 / 4.50 / 3.26 | 3.88 | +0.47 | 2/3 |
| DeepSeek | 1.87 / 2.02 / 2.00 | 1.96 | 1.96 / 2.31 / 2.16 | 2.15 | +0.18 | 3/3 | 2.02 / 1.82 / 1.92 | 1.92 | -0.04 | 1/3 |
| Luna | 2.91 / 3.25 / 2.93 | 3.03 | 3.00 / 3.31 / 4.15 | 3.49 | +0.46 | 3/3 | 3.14 / 3.26 / 3.49 | 3.29 | +0.26 | 3/3 |

Im Mittel langsamer als B1 — mit B2: Gemma, DeepSeek, Luna (3 von 4); mit B3: gpt-oss, Gemma, Luna (3 von 4).

## 12. Pareto-Front (Mittel über die Seeds, Testdaten, F1 gegen $ je 1000)

**abgerechnet**: Front (nach Kosten) = B1 gpt-oss, B3 gpt-oss, B1 Gemma, Verfahren DeepSeek

| Programm | Ø F1 Test | Ø $/1000 Test | dominiert |
|---|---|---|---|
| B1 gpt-oss | 0.751 | 0.113 | nein |
| B1 Gemma | 0.772 | 0.207 | nein |
| B1 DeepSeek | 0.796 | 0.288 | ja |
| B1 Luna | 0.566 | 0.288 | ja |
| Verfahren DeepSeek | 0.805 | 0.284 | nein |
| B2 gpt-oss | 0.751 | 0.182 | ja |
| B2 Gemma | 0.789 | 0.412 | ja |
| B2 DeepSeek | 0.803 | 0.314 | ja |
| B2 Luna | 0.711 | 0.749 | ja |
| B3 gpt-oss | 0.758 | 0.147 | nein |
| B3 Gemma | 0.796 | 0.425 | ja |
| B3 DeepSeek | 0.800 | 0.316 | ja |
| B3 Luna | 0.716 | 0.591 | ja |

**festgehalten**: Front (nach Kosten) = B1 gpt-oss, B3 gpt-oss, B1 Gemma, Verfahren DeepSeek

| Programm | Ø F1 Test | Ø $/1000 Test | dominiert |
|---|---|---|---|
| B1 gpt-oss | 0.751 | 0.113 | nein |
| B1 Gemma | 0.772 | 0.207 | nein |
| B1 DeepSeek | 0.796 | 0.222 | ja |
| B1 Luna | 0.566 | 0.288 | ja |
| Verfahren DeepSeek | 0.805 | 0.213 | nein |
| B2 gpt-oss | 0.751 | 0.182 | ja |
| B2 Gemma | 0.789 | 0.412 | ja |
| B2 DeepSeek | 0.803 | 0.231 | ja |
| B2 Luna | 0.711 | 0.749 | ja |
| B3 gpt-oss | 0.758 | 0.147 | nein |
| B3 Gemma | 0.796 | 0.425 | ja |
| B3 DeepSeek | 0.800 | 0.233 | ja |
| B3 Luna | 0.716 | 0.591 | ja |

