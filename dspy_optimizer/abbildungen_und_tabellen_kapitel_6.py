"""Abbildungen und Tabellen fuer Kapitel 6, nach `docs/checkliste-kapitel-6.md`, Teil E.

Liest nur die CSV-Dateien, die `dspy_optimizer.auswertung` nach `ergebnisse/auswertung/`
geschrieben hat, dazu die Wilcoxon-Tabellen aus `docs/zusatzauswertung-kapitel-6.md`; fuer das
Budget der Fokusphase und die Pruefsummen liest es `ergebnisse/protokoll.jsonl` (nur lesend).
Kein API-Aufruf. Geschrieben wird nur nach `ergebnisse/auswertung/kapitel-6/`; die vorhandenen
CSV-Dateien und `bericht.md` bleiben unberuehrt.

Mittelwerte und Standardabweichungen rechnet das Modul aus den Tabellen "je Seed" selbst
(Stichproben-Standardabweichung, n - 1 im Nenner, wie `auswertung.mittel_sd`) und prueft sie
gegen die zusammengefassten Tabellen der Auswertung: weicht ein Wert ab, bricht es ab.

Aufruf:
    python -m dspy_optimizer.abbildungen_und_tabellen_kapitel_6
"""

import hashlib
import json
import re
import statistics
import subprocess
import sys

import pandas as pd

from dspy_optimizer.auswertung import AUSGABE
from dspy_optimizer.protokoll import WURZEL

ZIEL = AUSGABE / "kapitel-6"
PROTOKOLL = WURZEL / "ergebnisse" / "protokoll.jsonl"
ZUSATZAUSWERTUNG = WURZEL / "docs" / "zusatzauswertung-kapitel-6.md"

GPT_OSS = "openai/gpt-oss-120b"
GEMMA = "google/gemma-4-31b-it"
DEEPSEEK = "deepseek/deepseek-v4.1-flash"
LUNA = "openai/gpt-6-luna"
# Reihenfolge wie `DIE_VIER` in `kandidatenmodelle.py`.
KANDIDATENMODELLE = (GPT_OSS, GEMMA, DEEPSEEK, LUNA)
KURZNAMEN = {GPT_OSS: "gpt-oss", GEMMA: "Gemma", DEEPSEEK: "DeepSeek", LUNA: "Luna"}
LANGNAMEN = {
    GPT_OSS: "gpt-oss-120b",
    GEMMA: "Gemma 4 31B",
    DEEPSEEK: "DeepSeek V4.1 Flash",
    LUNA: "GPT-6 Luna",
}
# Dieselben Strategienamen wie im Text der Arbeit (Teil E).
STRATEGIENAMEN = {
    "verfahren": "Verfahren",
    "ohne-optimierung": "B1 ohne Optimierung",
    "gleichverteilung": "B2 Gleichverteilung",
    "ein-modell": "B3 Einzelmodell",
}
# Wie die Auswertung die Strategien in ihren zusammengefassten Tabellen nennt.
BASELINENAMEN_DER_AUSWERTUNG = {
    "verfahren": "Verfahren",
    "gleichverteilung": "Gleichverteilung (B/4 je Modell)",
    "ein-modell": "Einzelmodell (volles Budget B)",
    "ohne-optimierung": "Ohne Optimierung",
}
PREISFASSUNGEN = {"abgerechnet": "abgerechnet", "festgehalten": "Stichtagspreis"}

# Schwarz-weiss druckbar: je Kandidatenmodell eigene Markerform, Linienart und Grauton.
MARKER_JE_MODELL = {GPT_OSS: "o", GEMMA: "s", DEEPSEEK: "D", LUNA: "^"}
LINIENART_JE_MODELL = {GPT_OSS: "-", GEMMA: "--", DEEPSEEK: "-.", LUNA: ":"}
GRAU_JE_MODELL = {GPT_OSS: "0.0", GEMMA: "0.3", DEEPSEEK: "0.5", LUNA: "0.65"}
MARKER_JE_STRATEGIE = {"ohne-optimierung": "o", "gleichverteilung": "s", "ein-modell": "^",
                       "verfahren": "*"}

TOLERANZ = 5e-4


# --- Lesen und Rechnen ----------------------------------------------------------------------


def lies_csv(name):
    return pd.read_csv(AUSGABE / name, sep=";", decimal=",", encoding="utf-8-sig")


def mittel_sd(werte):
    werte = list(werte)
    return statistics.mean(werte), (statistics.stdev(werte) if len(werte) > 1 else None)


def zahl(wert, stellen=3):
    """Dezimalkomma und Tausenderpunkt, wie in der Arbeit."""
    if wert is None or (isinstance(wert, float) and pd.isna(wert)):
        return "–"
    text = f"{wert:,.{stellen}f}"
    return text.replace(",", " ").replace(".", ",").replace(" ", ".")


def mittel_sd_als_text(werte, stellen=3):
    mittel, sd = mittel_sd(werte)
    return f"{zahl(mittel, stellen)} ± {zahl(sd, stellen)}"


def prozent(anteil, stellen=1):
    return f"{'+' if anteil > 0 else ''}{zahl(100 * anteil, stellen)} %"


def mittel_aus_text(text):
    """Den Mittelwert aus einer Zelle wie '0,805 ± 0,011' oder '19 969 ± 567'."""
    vorne = str(text).split("±")[0].strip().replace(" ", "").replace(" ", "")
    vorne = vorne.replace("%", "").replace(",", ".")
    return float(vorne)


def pruefe_gleich(eigener_wert, wert_der_auswertung, was, toleranz=TOLERANZ):
    if abs(eigener_wert - wert_der_auswertung) > toleranz:
        sys.exit(
            f"Abweichung bei {was}: hier {eigener_wert}, in der Auswertung {wert_der_auswertung}"
        )


def lies_protokoll():
    with PROTOKOLL.open(encoding="utf-8") as datei:
        return [json.loads(zeile) for zeile in datei if zeile.strip()]


def schreibe_csv(tabelle, name):
    tabelle.to_csv(ZIEL / name, sep=";", index=False, encoding="utf-8-sig")


# --- Tabellen -------------------------------------------------------------------------------


def tabelle_6_3(kosten, protokoll):
    """Strategien im Ueberblick: Budget, Laeufe, Fixkostenbloecke, Nachbewertungen, C_opt."""
    fokusbudgets = [
        zeile["dollar_budgetanteil"]
        for zeile in protokoll
        if zeile.get("abschnitt") == "optimierung" and zeile.get("phase") == "fokus"
        and zeile.get("zweck") == "experiment"
    ]
    if len(fokusbudgets) != 3:
        sys.exit(f"Erwartet drei Fokuslaeufe im Protokoll, gefunden {len(fokusbudgets)}")

    def c_opt(baseline):
        return kosten.loc[kosten["baseline"] == baseline, "c_opt"]

    def nachbewertungen(baseline):
        werte = set(kosten.loc[kosten["baseline"] == baseline, "nachbewertungen"])
        return " / ".join(str(int(wert)) for wert in sorted(werte))

    b3_je_modell = [
        mittel_sd(kosten.loc[(kosten["baseline"] == "ein-modell")
                             & (kosten["kandidatenmodell"] == modell), "c_opt"])[0]
        for modell in KANDIDATENMODELLE
    ]
    zeilen = [
        {
            "Strategie": "Verfahren",
            "Budget je Modell": (
                f"Sondierung 0,75 $ je Modell (B/8); Fokus {zahl(min(fokusbudgets), 2)}–"
                f"{zahl(max(fokusbudgets), 2)} $ (Restbudget, je Seed)"
            ),
            "Optimierungsläufe": "5 (4 Sondierung + 1 Fokus; geplant höchstens 6)",
            "Fixkostenblöcke": "5 (höchstens 6)",
            "Nachbewertungen": nachbewertungen("verfahren"),
            "wählt aus?": "ja (Zwischenbewertung, dann Endauswahl)",
            "tatsächlich ausgegeben (C_opt) $": mittel_sd_als_text(c_opt("verfahren"), 2),
        },
        {
            "Strategie": "B1 ohne Optimierung",
            "Budget je Modell": "0 $ (keine Optimierung)",
            "Optimierungsläufe": "0",
            "Fixkostenblöcke": "0",
            "Nachbewertungen": nachbewertungen("ohne-optimierung"),
            "wählt aus?": "ja (nach derselben Regel)",
            "tatsächlich ausgegeben (C_opt) $": mittel_sd_als_text(c_opt("ohne-optimierung"), 2),
        },
        {
            "Strategie": "B2 Gleichverteilung",
            "Budget je Modell": "1,50 $ (B/4)",
            "Optimierungsläufe": "4",
            "Fixkostenblöcke": "4",
            "Nachbewertungen": nachbewertungen("gleichverteilung"),
            "wählt aus?": "ja (nach derselben Regel)",
            "tatsächlich ausgegeben (C_opt) $": mittel_sd_als_text(c_opt("gleichverteilung"), 2),
        },
        {
            "Strategie": "B3 Einzelmodell",
            "Budget je Modell": "6,00 $ (B, ein Modell je Durchgang)",
            "Optimierungsläufe": "1",
            "Fixkostenblöcke": "1",
            "Nachbewertungen": nachbewertungen("ein-modell"),
            "wählt aus?": "nein (nur Prüfung gegen die Kostenobergrenze)",
            "tatsächlich ausgegeben (C_opt) $": (
                f"{zahl(min(b3_je_modell), 2)}–{zahl(max(b3_je_modell), 2)} "
                "(Mittel je Modell)"
            ),
        },
    ]
    return pd.DataFrame(zeilen)


def tabelle_6_4(ff1, test):
    """Die vier Kandidatenmodelle vor und nach der Optimierung (B/4), Testdaten."""
    ohne_je_modell = {
        modell: list(ff1[f"bewertung_ohne_optimierung {modell}"]) for modell in KANDIDATENMODELLE
    }
    mit_je_modell = {
        modell: list(ff1[f"bewertung_gleichverteilung {modell}"]) for modell in KANDIDATENMODELLE
    }

    def rang(werte_je_modell):
        mittel = {modell: statistics.mean(werte) for modell, werte in werte_je_modell.items()}
        reihenfolge = sorted(mittel, key=mittel.get, reverse=True)
        return {modell: reihenfolge.index(modell) + 1 for modell in mittel}

    rang_ohne, rang_mit = rang(ohne_je_modell), rang(mit_je_modell)

    def exakt(baseline, modell):
        auswahl = test[(test["baseline"] == baseline) & (test["kandidatenmodell"] == modell)]
        return list(auswahl.sort_values("seed")["exakt"])

    zeilen = []
    for modell in sorted(KANDIDATENMODELLE, key=rang_ohne.get):
        zugewinn = [mit - ohne for ohne, mit in zip(ohne_je_modell[modell], mit_je_modell[modell])]
        zeilen.append({
            "Kandidatenmodell": LANGNAMEN[modell],
            "F1 ohne": mittel_sd_als_text(ohne_je_modell[modell]),
            "F1 B/4": mittel_sd_als_text(mit_je_modell[modell]),
            "Zugewinn": mittel_sd_als_text(zugewinn),
            "Rang ohne/mit": f"{rang_ohne[modell]}/{rang_mit[modell]}",
            "Exact Match ohne": mittel_sd_als_text(exakt("ohne-optimierung", modell)),
            "Exact Match B/4": mittel_sd_als_text(exakt("gleichverteilung", modell)),
        })
    return pd.DataFrame(zeilen)


def tabelle_6_5(vergleiche_je_seed, programme_je_seed):
    """Verfahren gegen jede Baseline: F1, Differenz, Programmtest, Obergrenze."""

    def ueber_die_seeds(tabelle, gegen):
        return tabelle[tabelle["gegen"] == gegen].sort_values("seed")

    def unter_der_grenze(baseline, modell, spalte):
        auswahl = programme_je_seed[(programme_je_seed["baseline"] == baseline)
                                    & (programme_je_seed["kandidatenmodell"] == modell)]
        return int(auswahl[spalte].sum())

    def zeile(name, je_seed, unter_grenze):
        return {
            "Gegner": name,
            "F1 Verfahren": mittel_sd_als_text(je_seed["bewertung_verfahren"]),
            "F1 Gegner": mittel_sd_als_text(je_seed["bewertung_gegner"]),
            "Differenz": mittel_sd_als_text(je_seed["differenz"]),
            "Programmtest signifikant (x von 3)": f"{int(je_seed['signifikant'].sum())} von 3",
            "Gegner unter der Obergrenze (x von 3, abgerechnet / Stichtagspreis)": unter_grenze,
        }

    abgerechnet, festgehalten = vergleiche_je_seed["abgerechnet"], vergleiche_je_seed["festgehalten"]
    b1 = BASELINENAMEN_DER_AUSWERTUNG["ohne-optimierung"]
    b2 = BASELINENAMEN_DER_AUSWERTUNG["gleichverteilung"]
    b1_unter = (
        f"{int(ueber_die_seeds(abgerechnet, b1)['gegen_unter_grenze'].sum())} / "
        f"{int(ueber_die_seeds(festgehalten, b1)['gegen_unter_grenze'].sum())}"
    )
    zeilen = [
        zeile("B1 ohne Optimierung", ueber_die_seeds(abgerechnet, b1), b1_unter),
        zeile("B2 Gleichverteilung, abgerechnet", ueber_die_seeds(abgerechnet, b2),
              f"{int(ueber_die_seeds(abgerechnet, b2)['gegen_unter_grenze'].sum())} / –"),
        zeile("B2 Gleichverteilung, Stichtagspreis", ueber_die_seeds(festgehalten, b2),
              f"– / {int(ueber_die_seeds(festgehalten, b2)['gegen_unter_grenze'].sum())}"),
    ]
    for modell in KANDIDATENMODELLE:
        gegen = f"{BASELINENAMEN_DER_AUSWERTUNG['ein-modell']}: {LANGNAMEN[modell]}"
        # Die Bewertung haengt nicht am Preis: dieselben Zahlen in beiden Fassungen.
        zeilen.append(zeile(
            f"B3 Einzelmodell {LANGNAMEN[modell]}",
            ueber_die_seeds(abgerechnet, gegen),
            f"{unter_der_grenze('ein-modell', modell, 'unter_grenze')} / "
            f"{unter_der_grenze('ein-modell', modell, 'unter_grenze_festgehalten')}",
        ))
    return pd.DataFrame(zeilen)


def _ausgleich_als_text(werte):
    """Ausgleichspunkt je Seed in Programmlaeufen; 'nie' fuer Seeds ohne Ausgleich."""
    if all(pd.isna(wert) for wert in werte):
        return "nie"
    return " · ".join(
        f"S{seed} {'nie' if pd.isna(wert) else zahl(wert, 0)}"
        for seed, wert in enumerate(werte, start=1)
    )


def tabelle_6_6(kosten_je_durchgang, ff4_je_seed, test):
    """Kosten je Strategie; Stichtagspreis jeweils in Klammern."""

    def kosten_von(preis, baseline, modell=None):
        tabelle = kosten_je_durchgang[preis]
        auswahl = tabelle[tabelle["baseline"] == baseline]
        if modell is not None:
            auswahl = auswahl[auswahl["kandidatenmodell"] == modell]
        return auswahl.sort_values("seed")

    def ff4_von(preis, baseline, modell=None):
        tabelle = ff4_je_seed[preis]
        auswahl = tabelle[tabelle["baseline"] == baseline]
        if modell is not None:
            auswahl = auswahl[auswahl["gewaehltes_kandidatenmodell"] == modell]
        return auswahl.sort_values("seed")

    def beide(abgerechnet, festgehalten):
        return abgerechnet if abgerechnet == festgehalten else f"{abgerechnet} ({festgehalten})"

    def zeile(name, baseline, modell=None):
        werte = {}
        for preis in ("abgerechnet", "festgehalten"):
            kosten = kosten_von(preis, baseline, modell)
            ff4 = ff4_von(preis, baseline, modell)
            werte[preis] = {
                "c_opt": mittel_sd(kosten["c_opt"])[0],
                "plan": mittel_sd(kosten["abweichung_vom_plan_anteil"])[0],
                "c_exec": mittel_sd(ff4["c_exec_je_1000"])[0],
                "laeufe": mittel_sd(ff4["c_opt_in_programmlaeufen"])[0],
                "wechsel": int(ff4["modellwechsel"].sum()),
                "ausgleich": _ausgleich_als_text(list(ff4["break_even"])),
            }
        a, f = werte["abgerechnet"], werte["festgehalten"]
        return {
            "Strategie": name,
            "C_opt $": beide(zahl(a["c_opt"], 2), zahl(f["c_opt"], 2)),
            "Abweichung vom Plan": beide(prozent(a["plan"]), prozent(f["plan"])),
            "c_exec $ je 1000": beide(zahl(a["c_exec"]), zahl(f["c_exec"])),
            "C_opt in Programmläufen": beide(zahl(a["laeufe"], 0), zahl(f["laeufe"], 0)),
            "Modellwechsel?": (
                "– wählt nicht" if baseline == "ein-modell"
                else beide(f"ja in {a['wechsel']} von 3", f"ja in {f['wechsel']} von 3")
            ),
            "Ausgleichspunkt gegen B1 (Spanne je Seed)": beide(a["ausgleich"], f["ausgleich"]),
        }

    b1_test = test[(test["baseline"] == "ohne-optimierung") & (test["rolle"] == "gewinner")]
    b1_kosten = {preis: kosten_von(preis, "ohne-optimierung") for preis in ("abgerechnet", "festgehalten")}
    b1 = {
        "Strategie": "B1 ohne Optimierung (Referenz, DeepSeek)",
        "C_opt $": beide(zahl(mittel_sd(b1_kosten["abgerechnet"]["c_opt"])[0], 2),
                         zahl(mittel_sd(b1_kosten["festgehalten"]["c_opt"])[0], 2)),
        "Abweichung vom Plan": "–",
        "c_exec $ je 1000": beide(zahl(mittel_sd(b1_test["je_1000"])[0]),
                                  zahl(mittel_sd(b1_test["je_1000_festgehalten"])[0])),
        "C_opt in Programmläufen": "– (Referenz; im Ausgleichspunkt mit 0 gerechnet)",
        "Modellwechsel?": "– (Referenz)",
        "Ausgleichspunkt gegen B1 (Spanne je Seed)": "–",
    }
    zeilen = [zeile("Verfahren", "verfahren"), b1, zeile("B2 Gleichverteilung", "gleichverteilung")]
    for modell in KANDIDATENMODELLE:
        zeilen.append(zeile(f"B3 Einzelmodell {LANGNAMEN[modell]}", "ein-modell", modell))
    return pd.DataFrame(zeilen)


def tabelle_alle_programme(test):
    """Anhang: F1, Exact Match und Absatz-Trefferquote aller Programme (ADR-0009)."""
    reihenfolge = ("verfahren", "ohne-optimierung", "gleichverteilung", "ein-modell")
    zusammengefasst, je_seed = [], []
    for baseline in reihenfolge:
        for modell in KANDIDATENMODELLE:
            auswahl = test[(test["baseline"] == baseline) & (test["kandidatenmodell"] == modell)]
            if auswahl.empty:
                continue
            auswahl = auswahl.sort_values("seed")
            zusammengefasst.append({
                "Strategie": STRATEGIENAMEN[baseline],
                "Kandidatenmodell": LANGNAMEN[modell],
                "Seeds": len(auswahl),
                "F1": mittel_sd_als_text(auswahl["bewertung"]),
                "Exact Match": mittel_sd_als_text(auswahl["exakt"]),
                "Absatz-Trefferquote": mittel_sd_als_text(auswahl["absatztreffer"]),
            })
            for messung in auswahl.itertuples():
                je_seed.append({
                    "Strategie": STRATEGIENAMEN[baseline],
                    "Kandidatenmodell": LANGNAMEN[modell],
                    "Seed": messung.seed,
                    "Durchgang": messung.durchgang,
                    "Rolle": messung.rolle,
                    "F1": round(messung.bewertung, 4),
                    "Exact Match": round(messung.exakt, 4),
                    "Absatz-Trefferquote": round(messung.absatztreffer, 4),
                })
    return pd.DataFrame(zusammengefasst), pd.DataFrame(je_seed)


def tabelle_ungueltige_durchgaenge(ungueltige, protokoll):
    """Anhang: die ungueltigen Durchgaenge mit den Stoerungen, wie sie im Protokoll stehen."""
    stoerungen_je_durchgang = {}
    for zeile in protokoll:
        if zeile.get("durchgang") in set(ungueltige["durchgang"]):
            for stoerung in zeile.get("stoerungen") or []:
                if stoerung.startswith("Anbieterfehler"):
                    stoerungen_je_durchgang.setdefault(zeile["durchgang"], set()).add(stoerung)
    hinweise = {
        10: "1 Ratenbegrenzung (HTTP 429), dann Abbruch von Hand",
        11: "im Protokoll kein 429, nur Abbruch von Hand (KeyboardInterrupt)",
        22: "von Hand abgebrochen, nachdem der Durchgang schon ungültig war",
    }
    zeilen = []
    for messung in ungueltige.sort_values("durchgang").itertuples():
        zeilen.append({
            "Durchgang": messung.durchgang,
            "Seed": messung.seed,
            "Strategie": STRATEGIENAMEN[messung.baseline],
            "Kandidatenmodelle": ", ".join(
                KURZNAMEN[name.strip()] for name in messung.kandidatenmodelle.split(",")
            ),
            "Anbieterfehler (gezählt)": messung.anbieterfehler,
            "Ursache laut Protokoll": messung.ursache,
            "Hinweis": hinweise.get(messung.durchgang, ""),
            "Kosten $": zahl(messung.dollar, 3),
        })
    summe = ungueltige["dollar"].sum()
    zeilen.append({
        "Durchgang": "zusammen", "Seed": "", "Strategie": "", "Kandidatenmodelle": "",
        "Anbieterfehler (gezählt)": int(ungueltige["anbieterfehler"].sum()),
        "Ursache laut Protokoll": "", "Hinweis": "",
        "Kosten $": zahl(summe, 3),
    })
    return pd.DataFrame(zeilen)


def wilcoxon_tabellen():
    """Anhang: die Wilcoxon-Gegenprobe, aus `docs/zusatzauswertung-kapitel-6.md` uebernommen."""
    text = ZUSATZAUSWERTUNG.read_text(encoding="utf-8")

    def tabelle_nach(ueberschrift, nummer=0):
        abschnitt = text.split(ueberschrift, 1)[1]
        bloecke = re.findall(r"((?:^\|.*\|\n)+)", abschnitt, flags=re.MULTILINE)
        zeilen = [zeile.strip().strip("|").split("|") for zeile in bloecke[nummer].splitlines()]
        kopf = [zelle.strip() for zelle in zeilen[0]]
        rumpf = [[zelle.strip().replace(".", ",") for zelle in zeile] for zeile in zeilen[2:]]
        return pd.DataFrame(rumpf, columns=kopf)

    return {
        "anhang-wilcoxon-modellpaare.csv": tabelle_nach(
            "## 2. Dieselben Paare mit dem Wilcoxon-Vorzeichen-Rang-Test"),
        "anhang-wilcoxon-modellpaare-zusammenfassung.csv": tabelle_nach(
            "### Zusammenfassung Wilcoxon"),
        "anhang-wilcoxon-verfahren-abgerechnet.csv": tabelle_nach("**abgerechnet**"),
        "anhang-wilcoxon-verfahren-stichtagspreis.csv": tabelle_nach("**festgehalten**"),
        "anhang-wilcoxon-zwischenbewertung-ohne-obergrenze.csv": tabelle_nach(
            "**ohne Kostenobergrenze (wie"),
        "anhang-wilcoxon-zwischenbewertung-mit-obergrenze.csv": tabelle_nach(
            "**mit Kostenobergrenze (wie im Experiment)**"),
    }


# --- Abbildungen ----------------------------------------------------------------------------


def _zeichenwerkzeug():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
    })
    komma = FuncFormatter(lambda wert, _: f"{wert:g}".replace(".", ","))
    return plt, komma


def _ablegen(plt, abbildung, name):
    abbildung.tight_layout()
    abbildung.savefig(ZIEL / f"{name}.pdf")
    abbildung.savefig(ZIEL / f"{name}.png", dpi=200)
    plt.close(abbildung)


def abbildung_6_1(ff1):
    """F1 je Modell ohne Optimierung -> B/4, eine Linie je Modell, Seeds als duenne Linien."""
    plt, komma = _zeichenwerkzeug()
    abbildung, achse = plt.subplots(figsize=(5.6, 4.2))
    for modell in KANDIDATENMODELLE:
        ohne = list(ff1[f"bewertung_ohne_optimierung {modell}"])
        mit = list(ff1[f"bewertung_gleichverteilung {modell}"])
        for wert_ohne, wert_mit in zip(ohne, mit):
            achse.plot([0, 1], [wert_ohne, wert_mit], color="0.6", linewidth=0.6,
                       linestyle=LINIENART_JE_MODELL[modell], zorder=1)
        mittel_ohne, sd_ohne = mittel_sd(ohne)
        mittel_mit, sd_mit = mittel_sd(mit)
        achse.errorbar(
            [0, 1], [mittel_ohne, mittel_mit], yerr=[sd_ohne, sd_mit], color="black",
            linestyle=LINIENART_JE_MODELL[modell], marker=MARKER_JE_MODELL[modell],
            markerfacecolor="white", markersize=6, capsize=3, linewidth=1.4, zorder=3,
        )
        achse.annotate(
            LANGNAMEN[modell], (1, mittel_mit), xytext=(10, 0), textcoords="offset points",
            va="center", fontsize=8,
        )
    achse.set_xticks([0, 1], ["ohne Optimierung", "optimiert (B/4 je Modell)"])
    achse.set_xlim(-0.15, 1.55)
    achse.set_ylabel("F1 (Testdaten, n = 500)")
    achse.yaxis.set_major_formatter(komma)
    achse.grid(axis="y", color="0.9", linewidth=0.6)
    _ablegen(plt, abbildung, "abb-6-1-rangfolge")


def _zeilen_abbildung_6_2(endqualitaet, programme_je_seed):
    """Die Reihen der Abb. 6.2 mit ihren drei Seeds und der Zahl der Seeds ueber der Grenze."""

    def ueber_der_grenze(baseline, modell=None, spalte="unter_grenze"):
        auswahl = programme_je_seed[(programme_je_seed["baseline"] == baseline)
                                    & (programme_je_seed["rolle"] == "gewinner")]
        if modell is not None:
            auswahl = programme_je_seed[(programme_je_seed["baseline"] == baseline)
                                        & (programme_je_seed["kandidatenmodell"] == modell)]
        return int((~auswahl[spalte].astype(bool)).sum())

    def werte(strategie):
        return list(endqualitaet.loc[endqualitaet["strategie"] == strategie, "bewertung_test"])

    reihen = [
        ("Verfahren", werte("Verfahren"), ueber_der_grenze("verfahren")),
        ("B1 ohne Optimierung", werte("Ohne Optimierung"), ueber_der_grenze("ohne-optimierung")),
        ("B2 Gleichverteilung, abgerechnet", werte("Gleichverteilung (abgerechnet)"),
         ueber_der_grenze("gleichverteilung")),
        ("B2 Gleichverteilung, Stichtagspreis", werte("Gleichverteilung (Preis vom Stichtag)"),
         ueber_der_grenze("gleichverteilung", spalte="unter_grenze_festgehalten")),
    ]
    for modell in KANDIDATENMODELLE:
        reihen.append((
            f"B3 {KURZNAMEN[modell]}", werte(f"Einzelmodell: {LANGNAMEN[modell]}"),
            ueber_der_grenze("ein-modell", modell),
        ))
    return reihen


def abbildung_6_2(endqualitaet, programme_je_seed, f1_b1):
    """F1 je Strategie als Punktdiagramm: Seeds als Punkte, Mittelwert als Strich."""
    plt, komma = _zeichenwerkzeug()
    reihen = _zeilen_abbildung_6_2(endqualitaet, programme_je_seed)
    abbildung, achse = plt.subplots(figsize=(6.2, 3.9))
    beschriftungen = []
    for position, (name, werte, ueber) in enumerate(reihen):
        hervorgehoben = name == "Verfahren"
        farbe = "black" if hervorgehoben else "0.45"
        achse.scatter(werte, [position] * len(werte), s=26 if hervorgehoben else 18,
                      facecolors="white" if not hervorgehoben else "black",
                      edgecolors=farbe, linewidths=0.9, zorder=3)
        mittel = statistics.mean(werte)
        achse.plot([mittel, mittel], [position - 0.3, position + 0.3], color=farbe,
                   linewidth=2.4 if hervorgehoben else 1.6, zorder=2)
        beschriftungen.append(name + (f" † {ueber}/3" if ueber else ""))
    achse.axvline(f1_b1, color="0.3", linestyle="--", linewidth=0.9, zorder=1)
    achse.annotate(f"B1 = {zahl(f1_b1)}", (f1_b1, -0.75), xytext=(3, 0),
                   textcoords="offset points", ha="left", va="center", fontsize=7, color="0.3")
    # Trennlinien zwischen den Gruppen Verfahren | B1 | B2 | B3.
    for grenze in (0.5, 1.5, 3.5):
        achse.axhline(grenze, color="0.85", linewidth=0.6)
    achse.set_yticks(range(len(reihen)), beschriftungen)
    achse.get_yticklabels()[0].set_fontweight("bold")
    achse.set_ylim(len(reihen) - 0.4, -1.0)
    achse.set_xlim(0.68, None)
    achse.set_xlabel("F1 (Testdaten, n = 500)")
    achse.xaxis.set_major_formatter(komma)
    achse.grid(axis="x", color="0.92", linewidth=0.6)
    _ablegen(plt, abbildung, "abb-6-2-endqualitaet")


# Wo die Modellnamen an den Punkten der Abb. 6.3 stehen (Punkte Versatz), damit sich nichts
# ueberdeckt. Schluessel: (Strategie, Kandidatenmodell).
VERSATZ_ABBILDUNG_6_3 = {
    ("ohne-optimierung", GPT_OSS): (6, -10),
    ("ohne-optimierung", GEMMA): (6, -10),
    ("ohne-optimierung", DEEPSEEK): (-62, -4),
    ("gleichverteilung", GPT_OSS): (6, 2),
    ("gleichverteilung", GEMMA): (6, -11),
    ("gleichverteilung", DEEPSEEK): (22, 14),
    ("gleichverteilung", LUNA): (6, 3),
    ("ein-modell", GPT_OSS): (6, 3),
    ("ein-modell", GEMMA): (6, 3),
    ("ein-modell", DEEPSEEK): (5, -38),
    ("ein-modell", LUNA): (6, -10),
    ("verfahren", DEEPSEEK): (-62, 10),
}


def abbildung_6_3(pareto, preis, name):
    """F1 gegen c_exec mit Pareto-Front; Markerform = Strategie, Modell als Text am Punkt."""
    plt, komma = _zeichenwerkzeug()
    abbildung, achse = plt.subplots(figsize=(6.4, 4.4))
    unterer_rand = 0.70
    for punkt in pareto.itertuples():
        grau = GRAU_JE_MODELL[punkt.kandidatenmodell]
        verfahren = punkt.baseline == "verfahren"
        if punkt.bewertung < unterer_rand:
            # Luna ohne Optimierung liegt weit unten: als Pfeil am unteren Rand.
            achse.annotate(
                f"Luna, ohne Optimierung: {zahl(punkt.bewertung)}",
                xy=(punkt.kosten, unterer_rand), xytext=(punkt.kosten, unterer_rand + 0.012),
                ha="center", va="bottom", fontsize=7,
                arrowprops={"arrowstyle": "-|>", "color": grau, "linewidth": 1},
            )
            continue
        achse.errorbar(
            punkt.kosten, punkt.bewertung, xerr=punkt.kosten_sd, yerr=punkt.bewertung_sd,
            fmt=MARKER_JE_STRATEGIE[punkt.baseline], color=grau,
            markerfacecolor="white" if punkt.dominiert else grau,
            markeredgecolor="black" if verfahren else grau,
            markersize=13 if verfahren else 6.5, capsize=2, elinewidth=0.7,
            zorder=4 if verfahren else 3,
        )
        versatz = VERSATZ_ABBILDUNG_6_3.get((punkt.baseline, punkt.kandidatenmodell), (6, 3))
        # Weit abgesetzte Namen (die dicht liegenden DeepSeek-Punkte) bekommen eine Hinweislinie.
        hinweislinie = ({"arrowstyle": "-", "color": "0.5", "linewidth": 0.5}
                        if max(abs(versatz[0]), abs(versatz[1])) > 12 else None)
        achse.annotate(
            KURZNAMEN[punkt.kandidatenmodell], (punkt.kosten, punkt.bewertung), xytext=versatz,
            textcoords="offset points", fontsize=7, color="0.15", va="center",
            arrowprops=hinweislinie,
        )
    front = pareto[~pareto["dominiert"]].sort_values("kosten")
    front_sichtbar = front[front["bewertung"] >= unterer_rand]
    achse.step(front_sichtbar["kosten"], front_sichtbar["bewertung"], where="post", color="0.2",
               linewidth=0.9, zorder=1)
    achse.set_ylim(unterer_rand, None)
    achse.set_xlim(0, None)
    achse.set_xlabel(
        "c_exec ($ je 1000 Programmläufe, Testdaten, "
        + ("zum Stichtagspreis)" if preis == "festgehalten" else "abgerechnet)")
    )
    achse.set_ylabel("F1 (Testdaten, n = 500)")
    achse.xaxis.set_major_formatter(komma)
    achse.yaxis.set_major_formatter(komma)
    achse.grid(color="0.92", linewidth=0.6)
    legende = [
        plt.Line2D([], [], color="0.3", marker=MARKER_JE_STRATEGIE[baseline], linestyle="",
                   markersize=9 if baseline == "verfahren" else 6, label=STRATEGIENAMEN[baseline])
        for baseline in ("verfahren", "ohne-optimierung", "gleichverteilung", "ein-modell")
    ] + [
        plt.Line2D([], [], color="0.3", marker="o", markerfacecolor="white", linestyle="",
                   label="Pareto-dominiert (offen)"),
        plt.Line2D([], [], color="0.2", linewidth=0.9, label="Pareto-Front"),
    ]
    achse.legend(handles=legende, fontsize=7, loc="upper right", frameon=True, framealpha=0.9)
    _ablegen(plt, abbildung, name)


def kostenlinien(pareto, kosten_je_durchgang, preis):
    """Startwert (C_opt) und Steigung (c_exec) der drei Linien der Abb. 6.4, Mittel ueber Seeds."""

    def programm(baseline, modell):
        auswahl = pareto[(pareto["baseline"] == baseline) & (pareto["kandidatenmodell"] == modell)]
        return float(auswahl["kosten"].iloc[0])

    tabelle = kosten_je_durchgang[preis]

    def c_opt(baseline, modell=None):
        auswahl = tabelle[tabelle["baseline"] == baseline]
        if modell is not None:
            auswahl = auswahl[auswahl["kandidatenmodell"] == modell]
        return statistics.mean(auswahl["c_opt"])

    return {
        # B1 startet bei 0 wie beim Ausgleichspunkt (ADR-0022).
        "B1 ohne Optimierung (DeepSeek)": (0.0, programm("ohne-optimierung", DEEPSEEK)),
        "Verfahren (DeepSeek)": (c_opt("verfahren"), programm("verfahren", DEEPSEEK)),
        "B3 gpt-oss (Rechenbeispiel, von keiner Strategie gewählt)": (
            c_opt("ein-modell", GPT_OSS), programm("ein-modell", GPT_OSS)
        ),
    }


def schnittpunkt(linie, referenz):
    """Programmlaeufe, ab denen `linie` billiger ist als `referenz`; None ohne Schnitt."""
    (start, steigung), (start_referenz, steigung_referenz) = linie, referenz
    if steigung >= steigung_referenz:
        return None
    return (start - start_referenz) / (steigung_referenz - steigung) * 1000


def abbildung_6_4(linien, schnitt):
    """Gesamtkosten ueber die Zahl der Programmlaeufe, 0 bis 100.000, linear."""
    plt, komma = _zeichenwerkzeug()
    from matplotlib.ticker import FuncFormatter

    abbildung, achse = plt.subplots(figsize=(6.2, 4.0))
    laeufe = [0, 100_000]
    for (name, (start, steigung)), linienart in zip(linien.items(), ("-", "--", ":")):
        achse.plot(laeufe, [start + steigung * anzahl / 1000 for anzahl in laeufe],
                   color="black", linestyle=linienart, linewidth=1.5, label=name)
    b1_start, b1_steigung = linien["B1 ohne Optimierung (DeepSeek)"]
    hoehe = b1_start + b1_steigung * schnitt / 1000
    achse.plot([schnitt], [hoehe], marker="o", color="black", markersize=5, zorder=4)
    achse.annotate(
        f"Ausgleich B3 gpt-oss gegen B1:\netwa {zahl(round(schnitt, -3), 0)} Programmläufe",
        (schnitt, hoehe), xytext=(12, -32), textcoords="offset points", fontsize=7,
        arrowprops={"arrowstyle": "-", "color": "0.4", "linewidth": 0.7},
    )
    achse.set_xlim(0, 100_000)
    achse.set_ylim(0, None)
    achse.set_xlabel("Programmläufe im Betrieb")
    achse.set_ylabel("Gesamtkosten in $ (abgerechnet)")
    achse.xaxis.set_major_formatter(
        FuncFormatter(lambda wert, _: f"{wert:,.0f}".replace(",", "."))
    )
    achse.yaxis.set_major_formatter(komma)
    achse.grid(color="0.92", linewidth=0.6)
    achse.legend(fontsize=7, loc="upper left", frameon=False)
    _ablegen(plt, abbildung, "abb-6-4-gesamtkosten")


def abbildung_lernkurven(lernkurven_je_seed):
    """Anhang: Lernkurven nur als Punkte - die drei Stufen stammen aus verschiedenen Laeufen."""
    plt, komma = _zeichenwerkzeug()
    stufen = ("Sondierung", "Gleichverteilung", "Einzelmodell")
    beschriftung = ("Sondierung\n0,75 $ (B/8)", "B2 Gleichverteilung\n1,50 $ (B/4)",
                    "B3 Einzelmodell\n6,00 $ (B)")
    abbildung, achse = plt.subplots(figsize=(6.4, 4.0))
    versatz = {modell: (nummer - 1.5) * 0.16 for nummer, modell in enumerate(KANDIDATENMODELLE)}
    for modell in KANDIDATENMODELLE:
        for nummer, stufe in enumerate(stufen):
            werte = lernkurven_je_seed[(lernkurven_je_seed["kandidatenmodell"] == modell)
                                       & (lernkurven_je_seed["stufe"] == stufe)]["bewertung"]
            x = nummer + versatz[modell]
            achse.scatter([x] * len(werte), werte, marker=MARKER_JE_MODELL[modell],
                          facecolors="white", edgecolors="0.35", s=16, linewidths=0.8, zorder=2)
            achse.plot([x - 0.06, x + 0.06], [statistics.mean(werte)] * 2, color="black",
                       linewidth=1.6, zorder=3)
    achse.set_xticks(range(len(stufen)), beschriftung)
    achse.set_ylabel("F1 (Auswahldaten, n = 200)")
    achse.yaxis.set_major_formatter(komma)
    achse.grid(axis="y", color="0.92", linewidth=0.6)
    achse.legend(
        handles=[plt.Line2D([], [], marker=MARKER_JE_MODELL[modell], color="0.35",
                            markerfacecolor="white", linestyle="", label=LANGNAMEN[modell])
                 for modell in KANDIDATENMODELLE]
        + [plt.Line2D([], [], color="black", linewidth=1.6, label="Mittelwert über 3 Seeds")],
        fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False,
    )
    _ablegen(plt, abbildung, "anhang-lernkurven")


# --- Versionen und Pruefsummen --------------------------------------------------------------


def pruefsumme(pfad):
    rechner = hashlib.sha256()
    with pfad.open("rb") as datei:
        for block in iter(lambda: datei.read(1 << 20), b""):
            rechner.update(block)
    return rechner.hexdigest()


def schreibe_versionen_und_pruefsummen():
    pip = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    gesucht = ("dspy", "litellm", "optuna", "scipy")
    pakete = [zeile for zeile in pip
              if re.split(r"[=@ ]", zeile, maxsplit=1)[0].lower() in gesucht]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=WURZEL, capture_output=True,
                            text=True, check=True).stdout.strip()
    offen = subprocess.run(["git", "status", "--porcelain"], cwd=WURZEL, capture_output=True,
                           text=True, check=True).stdout.splitlines()
    dieses_skript = WURZEL / "dspy_optimizer" / "abbildungen_und_tabellen_kapitel_6.py"
    datendateien = (
        [PROTOKOLL]
        + sorted((WURZEL / "ergebnisse" / "programme").rglob("*.json"))
        + sorted(AUSGABE.glob("*.csv"))
        + [AUSGABE / "bericht.md", ZUSATZAUSWERTUNG, dieses_skript]
    )
    zeilen = [
        "# Versionen und Prüfsummen (Kapitel 6, Anhang)",
        "",
        f"Python: {sys.version.split()[0]}",
        "",
        "## Pakete (`pip freeze`)",
        "",
        *[f"- `{paket}`" for paket in pakete],
        "",
        "## Stand des Repositorys",
        "",
        f"- Commit: `{commit}`",
        f"- nicht committete Änderungen beim Erzeugen: {len(offen)} Datei(en)",
        *[f"  - `{zeile}`" for zeile in offen],
        "",
        "## SHA-256 der Datendateien (Python `hashlib`)",
        "",
        "| Datei | SHA-256 |",
        "|---|---|",
        *[f"| `{pfad.relative_to(WURZEL).as_posix()}` | `{pruefsumme(pfad)}` |"
          for pfad in datendateien],
        "",
    ]
    (ZIEL / "anhang-versionen-und-pruefsummen.md").write_text("\n".join(zeilen), encoding="utf-8")


# --- Ablauf ---------------------------------------------------------------------------------


def main():
    ZIEL.mkdir(parents=True, exist_ok=True)
    protokoll = lies_protokoll()
    test = lies_csv("testbewertungen.csv")
    ff1 = lies_csv("ff1-je-seed.csv")
    programme_je_seed = lies_csv("programme-je-seed.csv")
    endqualitaet = lies_csv("endqualitaet-je-seed.csv")
    vergleiche_je_seed = {preis: lies_csv(f"vergleiche-je-seed-{preis}.csv")
                          for preis in PREISFASSUNGEN}
    kosten_je_durchgang = {preis: lies_csv(f"kosten-je-durchgang-{preis}.csv")
                           for preis in PREISFASSUNGEN}
    ff4_je_seed = {preis: lies_csv(f"ff4-je-seed-{preis}.csv") for preis in PREISFASSUNGEN}
    pareto = {preis: lies_csv(f"pareto-{preis}.csv") for preis in PREISFASSUNGEN}

    # Gegenprobe gegen die zusammengefassten Tabellen der Auswertung.
    for preis in PREISFASSUNGEN:
        zusammengefasst = lies_csv(f"kosten-{preis}.csv")
        for zeile in zusammengefasst.itertuples():
            baseline = {wert: schluessel for schluessel, wert
                        in BASELINENAMEN_DER_AUSWERTUNG.items()}[zeile.baseline]
            auswahl = kosten_je_durchgang[preis][kosten_je_durchgang[preis]["baseline"] == baseline]
            if isinstance(zeile.kandidatenmodell, str):
                modell = {wert: schluessel for schluessel, wert in LANGNAMEN.items()}[
                    zeile.kandidatenmodell]
                auswahl = auswahl[auswahl["kandidatenmodell"] == modell]
            pruefe_gleich(statistics.mean(auswahl["c_opt"]), mittel_aus_text(zeile[6]),
                          f"C_opt {zeile.baseline} {zeile.kandidatenmodell} ({preis})", 6e-4)
        vergleiche = lies_csv(f"vergleiche-{preis}.csv")
        for zeile in vergleiche.itertuples():
            je_seed = vergleiche_je_seed[preis][vergleiche_je_seed[preis]["gegen"] == zeile[1]]
            pruefe_gleich(statistics.mean(je_seed["bewertung_gegner"]), mittel_aus_text(zeile[3]),
                          f"F1 Gegner {zeile[1]} ({preis})")
    b1_f1 = statistics.mean(
        endqualitaet.loc[endqualitaet["strategie"] == "Ohne Optimierung", "bewertung_test"])

    schreibe_csv(tabelle_6_3(kosten_je_durchgang["abgerechnet"], protokoll), "tab-6-3-strategien.csv")
    schreibe_csv(tabelle_6_4(ff1, test), "tab-6-4-modelle-vor-und-nach-optimierung.csv")
    schreibe_csv(tabelle_6_5(vergleiche_je_seed, programme_je_seed),
                 "tab-6-5-verfahren-gegen-baselines.csv")
    schreibe_csv(tabelle_6_6(kosten_je_durchgang, ff4_je_seed, test),
                 "tab-6-6-kosten-je-strategie.csv")
    zusammengefasst, je_seed = tabelle_alle_programme(test)
    schreibe_csv(zusammengefasst, "anhang-alle-programme-f1-em-absatztreffer.csv")
    schreibe_csv(je_seed, "anhang-alle-programme-f1-em-absatztreffer-je-seed.csv")
    schreibe_csv(tabelle_ungueltige_durchgaenge(lies_csv("ungueltige-durchgaenge.csv"), protokoll),
                 "anhang-ungueltige-durchgaenge.csv")
    for name, tabelle in wilcoxon_tabellen().items():
        schreibe_csv(tabelle, name)

    abbildung_6_1(ff1)
    abbildung_6_2(endqualitaet, programme_je_seed, b1_f1)
    abbildung_6_3(pareto["abgerechnet"], "abgerechnet", "abb-6-3-pareto")
    abbildung_6_3(pareto["festgehalten"], "festgehalten", "anhang-pareto-stichtagspreis")
    linien = {preis: kostenlinien(pareto[preis], kosten_je_durchgang, preis)
              for preis in PREISFASSUNGEN}
    schnitte = {}
    for preis, linien_der_fassung in linien.items():
        b1, verfahren, gpt_oss = linien_der_fassung.values()
        schnitte[preis] = {"gpt-oss": schnittpunkt(gpt_oss, b1),
                           "Verfahren": schnittpunkt(verfahren, b1)}
    abbildung_6_4(linien["abgerechnet"], schnitte["abgerechnet"]["gpt-oss"])
    abbildung_lernkurven(lies_csv("lernkurven-je-seed.csv"))
    schreibe_versionen_und_pruefsummen()

    for preis, werte in schnitte.items():
        print(f"Abb. 6.4, {preis}: Schnitt mit B1 bei gpt-oss "
              f"{zahl(werte['gpt-oss'], 0)}, Verfahren {zahl(werte['Verfahren'], 0)} Programmläufen")
        for name, (start, steigung) in linien[preis].items():
            print(f"  {name}: Start {start:.4f} $, Steigung {steigung:.5f} $ je 1000")
    print(f"Geschrieben nach {ZIEL}")


if __name__ == "__main__":
    main()
