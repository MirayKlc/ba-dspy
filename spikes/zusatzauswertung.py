"""Zusatzauswertung fuer Kapitel 6: Statistiken, die `auswertung.py` nicht rechnet.

Liest nur das Protokoll (kein API-Aufruf). Die Regeln von `dspy_optimizer/auswertung.py`
(gueltige Durchgaenge, erster gueltiger je Seed/Baseline/Kandidatenmodell, gueltige
Testbewertung, festgehaltener Preis, Zwischenbewertung) sind hier unabhaengig vom Paket
`dspy_optimizer` nachgebaut. Bevor etwas Neues gerechnet wird, prueft `pruefe_gegen_referenz`
die Nachbauten gegen die CSV-Dateien der regulaeren Auswertung und bricht bei einer
Abweichung ab.

Aufruf aus dem Stammordner des Repositorys (vorher `python -m dspy_optimizer.auswertung`,
damit `ergebnisse/auswertung/` existiert):
    python spikes/zusatzauswertung.py > docs/zusatzauswertung-kapitel-6.md
"""

import csv
import datetime
import itertools
import json
import math
import pathlib
import statistics
import sys
import warnings
from typing import NamedTuple

import numpy as np
from scipy import stats

PROTOKOLL = pathlib.Path(
    sys.argv[1] if len(sys.argv) > 1
    else "ergebnisse/protokoll.jsonl"
)
REFERENZ = pathlib.Path(
    sys.argv[2] if len(sys.argv) > 2
    else "ergebnisse/auswertung"
)

# --- Konstanten (aus hauptexperiment.py / verfahren.py) ----------------------------------
B = 6.0
KOSTENOBERGRENZE = 0.2895  # $ je 1000 Programmlaeufe
PROGRAMMLAEUFE_JE_KOSTENOBERGRENZE = 1000
ALPHA = 0.05

GPTOSS = "openai/gpt-oss-120b"
GEMMA = "google/gemma-4-31b-it"
DEEPSEEK = "deepseek/deepseek-v4.1-flash"
LUNA = "openai/gpt-6-luna"
NAMEN = (GPTOSS, GEMMA, DEEPSEEK, LUNA)  # Reihenfolge der Laeufe
KURZ = {GPTOSS: "gpt-oss", GEMMA: "Gemma", DEEPSEEK: "DeepSeek", LUNA: "Luna"}
STRAT = {"ohne-optimierung": "B1", "gleichverteilung": "B2", "ein-modell": "B3",
         "verfahren": "Verfahren"}
# Paare in der Richtung der Rangfolge ohne Optimierung (besser − schlechter)
PAARE = [(DEEPSEEK, GEMMA), (DEEPSEEK, GPTOSS), (DEEPSEEK, LUNA),
         (GEMMA, GPTOSS), (GEMMA, LUNA), (GPTOSS, LUNA)]
ZERO_METHODS = ("wilcox", "pratt", "zsplit")
PREISE = ("abgerechnet", "festgehalten")


# =========================================================================================
# Nachbau von auswertung.py / verfahren.py
# =========================================================================================

class Preis(NamedTuple):
    ab: str
    prompt: float
    antwort: float


# Preisaenderung bei DeepSeek, $ je 1 Mio.
# Tokens (Prompt / Antwort): Stichtagspreis 0,07 / 0,60; danach 0,077 / 0,55 und 0,10 / 0,66.
PREISE_DEEPSEEK = (
    Preis("", 0.07, 0.60),
    Preis("2026-10-13T15:11:07+02:00", 0.077, 0.55),
    Preis("2026-10-14T11:36:16+02:00", 0.10, 0.66),
)


def _preis_zur_zeit(zeitpunkt):
    t = datetime.datetime.fromisoformat(zeitpunkt)
    gueltig = PREISE_DEEPSEEK[0]
    for p in PREISE_DEEPSEEK[1:]:
        if t >= datetime.datetime.fromisoformat(p.ab):
            gueltig = p
    return gueltig


def _listenkosten(z, p):
    return (z["tokens_prompt"] * p.prompt + z["tokens_antwort"] * p.antwort) / 1e6


def modell(z):
    return (z.get("konfiguration") or {}).get("kandidatenmodell")


def dollar_fest(z):
    d = z.get("dollar")
    if not d or modell(z) != DEEPSEEK:
        return d
    damals = _preis_zur_zeit(z["zeitpunkt"])
    if damals == PREISE_DEEPSEEK[0] or not _listenkosten(z, damals):
        return d
    return d * _listenkosten(z, PREISE_DEEPSEEK[0]) / _listenkosten(z, damals)


def faktor_fest(z):
    d = z.get("dollar")
    return dollar_fest(z) / d if d else 1.0


class Stand(NamedTuple):
    name: str
    werte: tuple
    kosten: float  # $ je Programmlauf

    @property
    def bewertung(self):
        return statistics.fmean(self.werte)


def kostenfilter(staende, grenze):
    return [s for s in staende if s.kosten * PROGRAMMLAEUFE_JE_KOSTENOBERGRENZE <= grenze]


def rangfolge(staende):
    # hoechste Bewertung, bei Gleichstand niedrigere Kosten, dann Reihenfolge der Laeufe
    return sorted(staende, key=lambda s: (-s.bewertung, s.kosten))


def p_ttest_less(x, fuehrender):
    p = float(stats.ttest_rel(x, fuehrender, alternative="less").pvalue)
    return None if math.isnan(p) else p


def p_wilcoxon(x, y, zero_method, alternative):
    d = np.asarray(x, float) - np.asarray(y, float)
    if not np.any(d != 0):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = float(stats.wilcoxon(x, y, zero_method=zero_method, alternative=alternative).pvalue)
    return None if math.isnan(p) else p


def p_ttest_zweiseitig(x, y):
    p = float(stats.ttest_rel(x, y).pvalue)
    return None if math.isnan(p) else p


class Entscheidung(NamedTuple):
    weiter: list
    fuehrender: str | None
    p_werte: dict
    ausgeschieden: dict
    vergleiche: int


def zwischenbewertung(staende, grenze, test=p_ttest_less, alpha=ALPHA):
    """Zwischenbewertung: Kostenfilter,
    Fuehrender, einseitiger Test gegen ihn; allein weiter nur, wenn ALLE anderen nachweisbar
    schlechter sind, sonst die besten zwei. Ausscheiden durch den Test von hinten nach vorn."""
    uebrig = kostenfilter(staende, grenze)
    ausgeschieden = {s.name: "kostenfilter" for s in staende if s not in uebrig}
    if not uebrig:
        return Entscheidung([], None, {}, ausgeschieden, 0)
    rang = rangfolge(uebrig)
    fuehrender = rang[0]
    p = {s.name: test(s.werte, fuehrender.werte) for s in rang[1:]}
    durch_test = []
    for s in reversed(rang[1:]):
        if p[s.name] is not None and p[s.name] < alpha:
            durch_test.append(s.name)
        else:
            break
    if len(durch_test) == len(rang) - 1:
        weiter = [fuehrender.name]
    else:
        weiter = [s.name for s in rang[:2]]
    for s in rang[1:]:
        if s.name in durch_test:
            ausgeschieden[s.name] = "test"
        elif s.name not in weiter:
            ausgeschieden[s.name] = "nicht-unter-den-zwei"
    return Entscheidung(weiter, fuehrender.name, p, ausgeschieden, len(rang) - 1)


class Durchgang(NamedTuple):
    nummer: int
    seed: int
    baseline: str
    kandidatenmodell: str | None
    zeilen: tuple
    entscheidung: dict


def _zeilen_je_durchgang(zeilen):
    je = {}
    for z in zeilen:
        if z.get("zweck") != "experiment" or z.get("abschnitt") == "testbewertung":
            continue
        je.setdefault(z["durchgang"], []).append(z)
    return dict(sorted(je.items()))


def _entscheidungszeile(zeilen):
    for z in zeilen:
        if z.get("abschnitt") == "endauswahl":
            return z
        if z.get("abschnitt") == "zwischenbewertung" and z.get("kein_gewinner"):
            return z
    return None


def _ist_gueltig(zeilen):
    return _entscheidungszeile(zeilen) is not None and not any(
        z.get("anbieterfehler") for z in zeilen)


def finde_gueltige_durchgaenge(zeilen):
    gefunden = {}
    for nr, zz in _zeilen_je_durchgang(zeilen).items():
        if not _ist_gueltig(zz):
            continue
        e = _entscheidungszeile(zz)
        km = None
        if e["baseline"] == "ein-modell":
            (km,) = e["kandidatenmodelle"]
        gefunden.setdefault((e["seed"], e["baseline"], km),
                            Durchgang(nr, e["seed"], e["baseline"], km, tuple(zz), e))
    return sorted(gefunden.values(), key=lambda d: d.nummer)


class Test(NamedTuple):
    durchgang: int
    seed: int
    baseline: str
    phase: str | None
    modell: str
    bewertung: float
    dollar: float
    dollar_fest: float
    testbeispiele: int
    je_1000: float
    je_1000_fest: float
    latenz_median: float
    werte: tuple

    def c_exec(self, preis):  # $ je Programmlauf (FF4)
        return (self.dollar if preis == "abgerechnet" else self.dollar_fest) / self.testbeispiele


class Daten:
    def __init__(self, zeilen):
        self.zeilen = zeilen
        self.durchgaenge = finde_gueltige_durchgaenge(zeilen)
        self.nach_nr = {d.nummer: d for d in self.durchgaenge}
        self.seeds = sorted({d.seed for d in self.durchgaenge})
        self.tests = self._tests()

    def _tests(self):
        out, gesehen = [], set()
        for z in self.zeilen:
            if z.get("zweck") != "experiment" or z.get("abschnitt") != "testbewertung":
                continue
            if z.get("anbieterfehler") or z.get("bewertung") is None:
                continue
            d = self.nach_nr.get(z["durchgang"])
            paar = z["konfiguration"].get("paar")
            if d is None or paar in gesehen:
                continue
            gesehen.add(paar)
            f = faktor_fest(z)
            out.append(Test(d.nummer, d.seed, d.baseline, z.get("phase"), modell(z),
                            z["bewertung"], z["dollar"], z["dollar"] * f,
                            z["konfiguration"]["testdaten"],
                            z["dollar_je_programmlauf"] * 1000,
                            z["dollar_je_programmlauf"] * f * 1000,
                            z.get("latenz_median_sekunden"), tuple(z["einzelbewertungen"])))
        return out

    def durchgang(self, seed, baseline, km=None):
        for d in self.durchgaenge:
            if (d.seed, d.baseline, d.kandidatenmodell) == (seed, baseline, km):
                return d
        return None

    def nachbewertungen(self, nr, phase):
        return [z for z in self.nach_nr[nr].zeilen
                if z["abschnitt"] == "nachbewertung" and z.get("phase") == phase]

    def staende(self, nr, phase, preis="abgerechnet"):
        return tuple(
            Stand(modell(z), tuple(z["einzelbewertungen"]),
                  z["dollar_je_programmlauf"] * (1.0 if preis == "abgerechnet" else faktor_fest(z)))
            for z in self.nachbewertungen(nr, phase))

    def tests_von(self, nr, phase=None):
        return {t.modell: t for t in self.tests
                if t.durchgang == nr and (phase is None or t.phase == phase)}

    def waehle(self, d, grenze, preis):
        if d.baseline != "verfahren":
            u = kostenfilter(self.staende(d.nummer, None, preis), grenze)
            return rangfolge(u)[0].name if u else "kein Gewinner"
        sond = zwischenbewertung(self.staende(d.nummer, "sondierung", preis), grenze)
        fokus = self.staende(d.nummer, "fokus", preis)
        if not sond.weiter:
            return "kein Gewinner"
        if set(sond.weiter) != {s.name for s in fokus}:
            return "nicht nachrechenbar"
        return zwischenbewertung(fokus, grenze).fuehrender or "kein Gewinner"

    def gewaehlt(self, d, preis):
        if d.baseline == "ein-modell":
            (t,) = self.tests_von(d.nummer).values()
            return t
        m = self.waehle(d, KOSTENOBERGRENZE, preis)
        return self.tests_von(d.nummer, "fokus" if d.baseline == "verfahren" else None).get(m)

    def auswahl_nachbewertung(self, t):
        (z,) = [z for z in self.nachbewertungen(t.durchgang, t.phase) if modell(z) == t.modell]
        return z

    def kosten_je_durchgang(self, preis):
        out = {}
        for d in self.durchgaenge:
            ml = [z for z in d.zeilen if z["abschnitt"] in ("optimierung", "nachbewertung")]
            wert = (lambda z: z.get("dollar") or 0.0) if preis == "abgerechnet" else (
                lambda z: dollar_fest(z) or 0.0)
            opt = sum(wert(z) for z in ml if z["abschnitt"] == "optimierung")
            nb = sum(wert(z) for z in ml if z["abschnitt"] == "nachbewertung")
            geplant = [z.get("dollar_geplant") for z in ml if z.get("dollar_geplant") is not None]
            c_opt = opt + nb
            out[d.nummer] = dict(
                seed=d.seed, baseline=d.baseline, km=d.kandidatenmodell, optimierung=opt,
                nachbewertung=nb, c_opt=c_opt, geplant=sum(geplant) if geplant else None,
                abw_plan=(c_opt - sum(geplant)) / sum(geplant) if geplant else None)
        return out

    def ff4(self, preis):
        kosten = self.kosten_je_durchgang(preis)
        out = []
        for d in self.durchgaenge:
            if d.baseline == "ohne-optimierung":
                continue
            ref_d = self.durchgang(d.seed, "ohne-optimierung")
            g, r = self.gewaehlt(d, preis), self.gewaehlt(ref_d, preis)
            if g is None or r is None:
                continue
            c_opt = kosten[d.nummer]["c_opt"]
            ce, cr = g.c_exec(preis), r.c_exec(preis)
            out.append(dict(seed=d.seed, durchgang=d.nummer, baseline=d.baseline,
                            gewaehlt=g.modell, referenz=r.modell, c_opt=c_opt,
                            c_exec_1000=ce * 1000, c_ref_1000=cr * 1000,
                            c_opt_laeufe=c_opt / ce,
                            break_even=None if ce >= cr else c_opt / (cr - ce)))
        return out

    def vergleiche(self, preis):
        out = []
        for seed in self.seeds:
            v = self.gewaehlt(self.durchgang(seed, "verfahren"), preis)
            gegner = [("B2", self.durchgang(seed, "gleichverteilung")),
                      ("B1", self.durchgang(seed, "ohne-optimierung"))] + [
                (f"B3 {KURZ[n]}", self.durchgang(seed, "ein-modell", n)) for n in NAMEN]
            for name, d in gegner:
                g = self.gewaehlt(d, preis)
                out.append(dict(seed=seed, gegen=name, v=v, g=g,
                                differenz=v.bewertung - g.bewertung,
                                p=p_ttest_zweiseitig(v.werte, g.werte)))
        return out


# =========================================================================================
# Pruefung gegen die Referenz-CSV
# =========================================================================================

def lies_csv(name):
    with open(REFERENZ / name, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))


def zahl(s):
    return None if s in ("", None) else float(s.replace(",", "."))


def gleich(a, b, tol=1e-9, was=""):
    if a is None or b is None:
        ist_gleich = a is None and b is None
    else:
        ist_gleich = math.isclose(a, b, rel_tol=tol, abs_tol=1e-12)
    if not ist_gleich:
        raise SystemExit(f"Nachbau weicht von der Referenz ab: {was}: {a} != {b}")


def pruefe_gegen_referenz(daten):
    n = 0
    # programme-je-seed: F1, Kosten, Latenz
    progs = {(int(r["durchgang"]), r["kandidatenmodell"]): r for r in lies_csv("programme-je-seed.csv")}
    if len(progs) != len(daten.tests):
        raise SystemExit("Zahl der Testbewertungen weicht ab")
    for t in daten.tests:
        r = progs[(t.durchgang, t.modell)]
        nb = daten.auswahl_nachbewertung(t)
        for a, b, w in [(t.bewertung, zahl(r["bewertung_test"]), "F1 Test"),
                        (nb["bewertung"], zahl(r["bewertung_auswahl"]), "F1 Auswahl"),
                        (statistics.fmean(t.werte), zahl(r["bewertung_test"]), "Mittel Einzelwerte"),
                        (t.je_1000, zahl(r["je_1000_test"]), "$ Test"),
                        (t.je_1000_fest, zahl(r["je_1000_test_festgehalten"]), "$ Test fest"),
                        (nb["dollar_je_programmlauf"] * 1000 * faktor_fest(nb),
                         zahl(r["je_1000_auswahl_festgehalten"]), "$ Auswahl fest"),
                        (t.latenz_median, zahl(r["latenz_median_test"]), "Latenz")]:
            gleich(a, b, 1e-9, f"{w} {t.durchgang} {t.modell}")
            n += 1
    # ff1
    for r in lies_csv("ff1-je-seed.csv"):
        s = int(r["seed"])
        for bl, sp in (("ohne-optimierung", "bewertung_ohne_optimierung"),
                       ("gleichverteilung", "bewertung_gleichverteilung")):
            ts = daten.tests_von(daten.durchgang(s, bl).nummer)
            for m in NAMEN:
                gleich(ts[m].bewertung, zahl(r[f"{sp} {m}"]), was=f"FF1 {s} {bl} {m}")
                n += 1
    # Zwischenbewertungen mit und ohne Obergrenze
    for datei, grenze in (("zwischenbewertungen.csv", KOSTENOBERGRENZE),
                          ("zwischenbewertungen-ohne-obergrenze.csv", math.inf)):
        for r in lies_csv(datei):
            e = zwischenbewertung(daten.staende(int(r["durchgang"]), "sondierung"), grenze)
            if ", ".join(e.weiter) != r["weiter"] or e.vergleiche != int(r["vergleiche"]):
                raise SystemExit(f"Zwischenbewertung weicht ab: {datei} {r['durchgang']}")
            for m in NAMEN:
                gleich(e.p_werte.get(m), zahl(r[f"p_wert {m}"]), was=f"{datei} {r['seed']} {m}")
                if e.ausgeschieden.get(m, "") != r[f"ausgeschieden {m}"]:
                    raise SystemExit(f"Grund weicht ab: {datei} {r['seed']} {m}")
                n += 1
    # Auswahlen
    for r in lies_csv("auswahlen.csv"):
        d = daten.nach_nr[int(r["durchgang"])]
        for sp, g, p in (("nachgerechnet", KOSTENOBERGRENZE, "abgerechnet"),
                         ("ohne_obergrenze", math.inf, "abgerechnet"),
                         ("festgehaltener_preis", KOSTENOBERGRENZE, "festgehalten")):
            if daten.waehle(d, g, p) != r[sp]:
                raise SystemExit(f"Auswahl weicht ab: {r['durchgang']} {sp}")
            n += 1
    # Vergleiche je Seed
    for preis in PREISE:
        ref = lies_csv(f"vergleiche-je-seed-{preis}.csv")
        eig = daten.vergleiche(preis)
        if len(ref) != len(eig):
            raise SystemExit("Zahl der Vergleiche weicht ab")
        for r, e in zip(ref, eig):
            gleich(e["differenz"], zahl(r["differenz"]), was=f"Differenz {preis} {r['seed']} {r['gegen']}")
            gleich(e["p"], zahl(r["p_wert"]), 1e-7, f"p {preis} {r['seed']} {r['gegen']}")
            n += 2
    # FF4 und Kosten je Durchgang
    for preis in PREISE:
        ref = lies_csv(f"ff4-je-seed-{preis}.csv")
        eig = daten.ff4(preis)
        for r, e in zip(ref, eig, strict=True):
            for k_ref, k in (("c_opt", "c_opt"), ("c_exec_je_1000", "c_exec_1000"),
                             ("c_ref_je_1000", "c_ref_1000"),
                             ("c_opt_in_programmlaeufen", "c_opt_laeufe"), ("break_even", "break_even")):
                gleich(e[k], zahl(r[k_ref]), 1e-9, f"FF4 {preis} {r['durchgang']} {k}")
                n += 1
        kref = {int(r["durchgang"]): r for r in lies_csv(f"kosten-je-durchgang-{preis}.csv")}
        for nr, k in daten.kosten_je_durchgang(preis).items():
            gleich(k["c_opt"], zahl(kref[nr]["c_opt"]), was=f"C_opt {preis} {nr}")
            gleich(k["abw_plan"], zahl(kref[nr]["abweichung_vom_plan_anteil"]), was=f"Plan {preis} {nr}")
            n += 2
    # Lernkurven (Sondierung)
    for r in lies_csv("lernkurven-je-seed.csv"):
        if r["stufe"] != "Sondierung":
            continue
        d = daten.durchgang(int(r["seed"]), "verfahren")
        (z,) = [z for z in daten.nachbewertungen(d.nummer, "sondierung") if modell(z) == r["kandidatenmodell"]]
        gleich(z["bewertung"], zahl(r["bewertung"]), was="Lernkurve")
        n += 1
    return n


# =========================================================================================
# Ausgabe-Helfer
# =========================================================================================

def f3(x):
    return "–" if x is None else f"{x:.3f}"


def f4p(p):
    if p is None:
        return "–"
    return "<0.0001" if p < 0.0001 else f"{p:.4f}"


def pct(x):
    return "–" if x is None else f"{100 * x:+.1f} %"


def sig(p, a=ALPHA):
    return p is not None and p < a


def tabelle(kopf, zeilen):
    out = ["| " + " | ".join(kopf) + " |", "|" + "---|" * len(kopf)]
    out += ["| " + " | ".join(str(z) for z in zeile) + " |" for zeile in zeilen]
    return "\n".join(out)


def mittel_sd(werte):
    werte = [w for w in werte if w is not None]
    if not werte:
        return None, None
    return statistics.fmean(werte), (statistics.stdev(werte) if len(werte) > 1 else None)


def msd(werte, f=f3):
    m, s = mittel_sd(werte)
    return f"{f(m)} ± {f(s)}"


def spanne(werte, f=f3):
    werte = [w for w in werte if w is not None]
    return f"{f(min(werte))} … {f(max(werte))}" if werte else "–"


def kendall_tau(a, b):
    paare = list(itertools.combinations(a, 2))
    sg = lambda x: (x > 0) - (x < 0)  # noqa: E731
    return sum(sg(a[x] - a[y]) * sg(b[x] - b[y]) for x, y in paare) / len(paare)


Z_MDE = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)


def mde(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    return Z_MDE * d.std(ddof=1) / math.sqrt(len(d)), d.std(ddof=1)


def nz(x, y):
    return int(np.count_nonzero(np.asarray(x, float) - np.asarray(y, float)))


# =========================================================================================
# Auswertung
# =========================================================================================

def main():
    zeilen = [json.loads(l) for l in PROTOKOLL.read_text(encoding="utf-8").splitlines() if l.strip()]
    daten = Daten(zeilen)
    ausgabezeilen = []
    schreibe = ausgabezeilen.append

    geprueft = pruefe_gegen_referenz(daten)
    schreibe("# Zusatzauswertung Kapitel 6")
    schreibe("")
    schreibe(f"Quelle: `{PROTOKOLL.name}` ({len(zeilen)} Zeilen), kein API-Aufruf. "
             f"{len(daten.durchgaenge)} gültige Durchgänge, {len(daten.tests)} Testbewertungen. "
             f"Nachbau geprüft gegen die Referenz-CSV in `{REFERENZ}`: {geprueft} Werte gleich "
             "(F1 je Programm, FF1, p-Werte der Zwischenbewertungen mit/ohne Obergrenze, Auswahlen, "
             "Vergleiche je Seed beider Preisfassungen, FF4 je Seed, C_opt und Planabweichung je "
             "Durchgang, Lernkurve Sondierung).")
    schreibe("")
    schreibe(f"Konventionen: Differenzen in F1 (0..1). p-Werte 4 Stellen. α = {ALPHA}. "
             f"scipy {__import__('scipy').__version__}; Wilcoxon `scipy.stats.wilcoxon(x, y, zero_method=…)` "
             "mit `method='auto'` (bei n > 50 bzw. Nullen/Bindungen Normalapproximation, ohne "
             "Stetigkeitskorrektur). t-Test `scipy.stats.ttest_rel`. Test = Testdaten (n = 500), "
             "Auswahl = Auswahldaten (n = 200).")
    schreibe("")

    def progs(seed, bl):
        """Je Modell das Testprogramm einer Strategie in einem Seed (B3: je Modell sein Lauf)."""
        if bl == "ein-modell":
            return {m: daten.tests_von(daten.durchgang(seed, bl, m).nummer)[m] for m in NAMEN}
        return daten.tests_von(daten.durchgang(seed, bl).nummer)

    # ---------------------------------------------------------------- 1 + 2 Modellpaare
    schreibe("## 1. Paarweise Modellvergleiche auf den Testdaten (t-Test, zweiseitig)")
    schreibe("")
    schreibe("Differenz = F1(erstes) − F1(zweites). B3 = Einzelmodell-Programm je Modell (verschiedene Läufe).")
    schreibe("")
    paar_ergebnisse = []  # für Abschnitt 2
    for bl in ("ohne-optimierung", "gleichverteilung", "ein-modell"):
        schreibe(f"### {STRAT[bl]} ({bl})")
        schreibe("")
        tabellenzeilen = []
        for a, b in PAARE:
            for s in daten.seeds:
                pr = progs(s, bl)
                x, y = pr[a].werte, pr[b].werte
                pt = p_ttest_zweiseitig(x, y)
                pw = {zm: p_wilcoxon(x, y, zm, "two-sided") for zm in ZERO_METHODS}
                paar_ergebnisse.append(dict(bl=bl, a=a, b=b, seed=s, diff=pr[a].bewertung - pr[b].bewertung,
                                            pt=pt, pw=pw, nz=nz(x, y), fa=pr[a].bewertung, fb=pr[b].bewertung))
                tabellenzeilen.append([f"{KURZ[a]} − {KURZ[b]}", s, f3(pr[a].bewertung), f3(pr[b].bewertung),
                                       f3(pr[a].bewertung - pr[b].bewertung), f4p(pt), "ja" if sig(pt) else "nein"])
        schreibe(tabelle(["Paar", "Seed", "F1 A", "F1 B", "Differenz", "p (t)", "sig."], tabellenzeilen))
        schreibe("")
    schreibe("### Zusammenfassung: signifikant in x von 3 Seeds (t-Test)")
    schreibe("")
    tabellenzeilen = []
    for a, b in PAARE:
        zeile = [f"{KURZ[a]} − {KURZ[b]}"]
        for bl in ("ohne-optimierung", "gleichverteilung", "ein-modell"):
            e = [r for r in paar_ergebnisse if r["bl"] == bl and (r["a"], r["b"]) == (a, b)]
            zeile += [f"{sum(sig(r['pt']) for r in e)}/3", f3(statistics.fmean(r["diff"] for r in e)),
                      ", ".join(f4p(r["pt"]) for r in e)]
        tabellenzeilen.append(zeile)
    schreibe(tabelle(["Paar", "B1 sig.", "B1 Ø Diff.", "B1 p (S1, S2, S3)", "B2 sig.", "B2 Ø Diff.",
                      "B2 p (S1, S2, S3)", "B3 sig.", "B3 Ø Diff.", "B3 p (S1, S2, S3)"], tabellenzeilen))
    schreibe("")
    schreibe("**Hervorgehoben:**")
    schreibe("")
    for a, b in ((DEEPSEEK, GEMMA), (GEMMA, GPTOSS)):
        for bl in ("ohne-optimierung", "gleichverteilung", "ein-modell"):
            e = [r for r in paar_ergebnisse if r["bl"] == bl and (r["a"], r["b"]) == (a, b)]
            schreibe(f"- {KURZ[a]} − {KURZ[b]}, {STRAT[bl]}: Differenz je Seed "
                     + ", ".join(f"S{r['seed']} {f3(r['diff'])} (p {f4p(r['pt'])})" for r in e)
                     + f"; Ø {f3(statistics.fmean(r['diff'] for r in e))}; signifikant in "
                     f"{sum(sig(r['pt']) for r in e)}/3")
    schreibe("")

    schreibe("## 2. Dieselben Paare mit dem Wilcoxon-Vorzeichen-Rang-Test (zweiseitig)")
    schreibe("")
    tabellenzeilen = []
    for r in paar_ergebnisse:
        tabellenzeilen.append([STRAT[r["bl"]], f"{KURZ[r['a']]} − {KURZ[r['b']]}", r["seed"], r["nz"], f4p(r["pt"])]
                              + [f4p(r["pw"][zm]) for zm in ZERO_METHODS])
    schreibe(tabelle(["Strategie", "Paar", "Seed", "Fragen mit Δ≠0", "p t-Test", "p wilcox", "p pratt", "p zsplit"], tabellenzeilen))
    schreibe("")
    schreibe("### Zusammenfassung Wilcoxon: signifikant in x von 3 Seeds (wilcox / pratt / zsplit; t-Test zum Vergleich)")
    schreibe("")
    tabellenzeilen = []
    for a, b in PAARE:
        zeile = [f"{KURZ[a]} − {KURZ[b]}"]
        for bl in ("ohne-optimierung", "gleichverteilung", "ein-modell"):
            e = [r for r in paar_ergebnisse if r["bl"] == bl and (r["a"], r["b"]) == (a, b)]
            zeile.append(" / ".join(str(sum(sig(r["pw"][zm]) for r in e)) for zm in ZERO_METHODS)
                         + f" (t: {sum(sig(r['pt']) for r in e)})")
        tabellenzeilen.append(zeile)
    schreibe(tabelle(["Paar", "B1", "B2", "B3"], tabellenzeilen))
    schreibe("")
    schreibe("### Fälle, in denen ein Wilcoxon-Urteil vom t-Test-Urteil abweicht (Modellpaare)")
    schreibe("")
    abw = []
    for r in paar_ergebnisse:
        for zm in ZERO_METHODS:
            if sig(r["pw"][zm]) != sig(r["pt"]):
                abw.append([STRAT[r["bl"]], f"{KURZ[r['a']]} − {KURZ[r['b']]}", r["seed"], zm,
                            f4p(r["pt"]), f4p(r["pw"][zm]),
                            "t sig., Wilcoxon nicht" if sig(r["pt"]) else "Wilcoxon sig., t nicht"])
    schreibe(tabelle(["Strategie", "Paar", "Seed", "zero_method", "p t", "p Wilcoxon", "Richtung"], abw)
             if abw else "Keine Abweichung.")
    schreibe("")

    # ---------------------------------------------------------------- 3a Verfahren vs Baselines
    schreibe("## 3. Wilcoxon-Gegenprobe für die Programmvergleiche und die Zwischenbewertung")
    schreibe("")
    schreibe("### 3a. Verfahren gegen jede Baseline (Testdaten, zweiseitig), wie `vergleiche-je-seed-*.csv`")
    schreibe("")
    abw3 = []
    for preis in PREISE:
        schreibe(f"**{preis}**")
        schreibe("")
        tabellenzeilen = []
        for v in daten.vergleiche(preis):
            x, y = v["v"].werte, v["g"].werte
            pw = {zm: p_wilcoxon(x, y, zm, "two-sided") for zm in ZERO_METHODS}
            tabellenzeilen.append([v["seed"], v["gegen"], KURZ[v["g"].modell], f3(v["differenz"]), nz(x, y),
                                   f4p(v["p"]), "ja" if sig(v["p"]) else "nein"] + [f4p(pw[zm]) for zm in ZERO_METHODS])
            for zm in ZERO_METHODS:
                if sig(pw[zm]) != sig(v["p"]):
                    abw3.append([preis, v["seed"], v["gegen"], zm, f4p(v["p"]), f4p(pw[zm])])
        schreibe(tabelle(["Seed", "gegen", "Gegner-Modell", "Differenz", "Fragen mit Δ≠0", "p t", "t sig.",
                          "p wilcox", "p pratt", "p zsplit"], tabellenzeilen))
        schreibe("")
    schreibe("Abweichende Urteile (Wilcoxon vs. t-Test):")
    schreibe("")
    schreibe(tabelle(["Preis", "Seed", "gegen", "zero_method", "p t", "p Wilcoxon"], abw3) if abw3 else "Keine Abweichung.")
    schreibe("")

    # ---------------------------------------------------------------- 3b Zwischenbewertung
    schreibe("### 3b. Zwischenbewertung des Verfahrens (Auswahldaten, n = 200, einseitig `less`, gegen den Führenden)")
    schreibe("")
    schreibe("Regel aus ADR-0004 (Nachtrag 08./09.10.2026) unverändert, nur der Test getauscht. Führender = höchster Mittelwert.")
    schreibe("")
    zweiter_rein = []
    for grenze, titel in ((math.inf, "ohne Kostenobergrenze (wie `zwischenbewertungen-ohne-obergrenze.csv`)"),
                          (KOSTENOBERGRENZE, "mit Kostenobergrenze (wie im Experiment)")):
        schreibe(f"**{titel}**")
        schreibe("")
        tabellenzeilen = []
        for s in daten.seeds:
            d = daten.durchgang(s, "verfahren")
            st = daten.staende(d.nummer, "sondierung")
            fuehr = rangfolge(kostenfilter(st, grenze))[0]
            tests = {"t-Test": p_ttest_less}
            tests.update({f"Wilcoxon {zm}": (lambda zm: lambda x, y: p_wilcoxon(x, y, zm, "less"))(zm)
                          for zm in ZERO_METHODS})
            for tn, tf in tests.items():
                e = zwischenbewertung(st, grenze, test=tf)
                zeile = [s, tn]
                for m in NAMEN:
                    if m == e.fuehrender:
                        zeile.append("Führender")
                    elif m in e.p_werte:
                        sm = next(x for x in st if x.name == m)
                        zeile.append(f"{f4p(e.p_werte[m])} (Δ≠0: {nz(sm.werte, fuehr.werte)})")
                    else:
                        zeile.append(e.ausgeschieden.get(m, ""))
                zeile.append(", ".join(KURZ[m] for m in e.weiter))
                tabellenzeilen.append(zeile)
                if len(e.weiter) > 1:
                    zweiter_rein.append((titel, s, tn, e.weiter))
        schreibe(tabelle(["Seed", "Test"] + [f"p {KURZ[m]}" for m in NAMEN] + ["weiter"], tabellenzeilen))
        schreibe("")
    schreibe("Zweites Modell in der Fokusphase unter einer Testvariante: "
             + ("; ".join(f"{t.split(' (')[0]}, Seed {s}, {tn}: {', '.join(KURZ[m] for m in wt)}"
                          for t, s, tn, wt in zweiter_rein) if zweiter_rein else "**in keinem Seed, unter keiner Variante**."))
    schreibe("")
    schreibe("F1 der Sondierung (Auswahldaten) je Seed, zur Einordnung:")
    schreibe("")
    tabellenzeilen = []
    for s in daten.seeds:
        st = daten.staende(daten.durchgang(s, "verfahren").nummer, "sondierung")
        tabellenzeilen.append([s] + [f3(next(x for x in st if x.name == m).bewertung) for m in NAMEN]
                              + [f3(next(x for x in st if x.name == m).kosten * 1000) for m in NAMEN])
    schreibe(tabelle(["Seed"] + [f"F1 {KURZ[m]}" for m in NAMEN] + [f"$/1000 {KURZ[m]}" for m in NAMEN], tabellenzeilen))
    schreibe("")

    # ---------------------------------------------------------------- 4 FF1 mit vollem Budget
    schreibe("## 4. FF1 mit vollem Budget: Rangfolge B1 gegen B3 (Testdaten)")
    schreibe("")
    tabellenzeilen, wechsel = [], []
    for s in daten.seeds:
        b1, b3 = progs(s, "ohne-optimierung"), progs(s, "ein-modell")
        f1 = {m: b1[m].bewertung for m in NAMEN}
        f3_ = {m: b3[m].bewertung for m in NAMEN}
        r1 = sorted(NAMEN, key=lambda m: -f1[m])
        r3 = sorted(NAMEN, key=lambda m: -f3_[m])
        tabellenzeilen.append([s, f"{kendall_tau(f1, f3_):.3f}", " > ".join(KURZ[m] for m in r1),
                               " > ".join(f"{KURZ[m]} {f3(f3_[m])}" for m in r3), KURZ[r1[0]], KURZ[r3[0]]])
        for a, b in itertools.combinations(NAMEN, 2):
            if (f1[a] - f1[b]) * (f3_[a] - f3_[b]) < 0:
                vorne3 = a if f3_[a] > f3_[b] else b
                hinten3 = b if vorne3 == a else a
                wechsel.append(dict(seed=s, paar=f"{KURZ[vorne3]} vor {KURZ[hinten3]}",
                                    d=f3_[vorne3] - f3_[hinten3],
                                    p3=p_ttest_zweiseitig(b3[vorne3].werte, b3[hinten3].werte),
                                    p1=p_ttest_zweiseitig(b1[a].werte, b1[b].werte),
                                    pw={zm: p_wilcoxon(b3[vorne3].werte, b3[hinten3].werte, zm, "two-sided")
                                        for zm in ZERO_METHODS}))
    schreibe(tabelle(["Seed", "Kendalls τ", "Rang B1", "Rang B3 (F1)", "Spitze B1", "Spitze B3"], tabellenzeilen))
    schreibe("")
    taus = [float(r[1]) for r in tabellenzeilen]
    schreibe(f"τ Mittel ± SD: {msd(taus)}")
    schreibe("")
    schreibe("Platzwechsel (B3 gegen B1), t-Test zweiseitig zwischen den beiden B3-Programmen (p_B3) und den beiden B1-Programmen (p_B1):")
    schreibe("")
    schreibe(tabelle(["Seed", "Wechsel (Ordnung in B3)", "Δ F1 in B3", "p_B3 (t)", "p_B1 (t)", "sig.",
                      "p_B3 wilcox", "p_B3 pratt", "p_B3 zsplit"],
                     [[x["seed"], x["paar"], f3(x["d"]), f4p(x["p3"]), f4p(x["p1"]), "ja" if sig(x["p3"]) else "nein"]
                      + [f4p(x["pw"][zm]) for zm in ZERO_METHODS] for x in wechsel]) if wechsel else "Kein Platzwechsel.")
    schreibe("")
    je_paar = {}
    for x in wechsel:
        k = frozenset(x["paar"].split(" vor "))
        je_paar.setdefault(k, []).append(x)
    for k, xs in je_paar.items():
        nsig = sum(sig(x["p3"]) for x in xs)
        schreibe(f"- {' / '.join(sorted(k))}: Wechsel in {len(xs)} von 3 Seeds, davon signifikant {nsig}; "
                 f"belastbar (≥ 2 von 3): {'ja' if nsig >= 2 else 'nein'}")
    schreibe("")

    # ---------------------------------------------------------------- 5 MDE
    schreibe("## 5. Kleinster nachweisbarer Unterschied (MDE), 80 % Power, zweiseitig α = 0,05, n = 500 gepaart")
    schreibe("")
    schreibe(f"MDE = (z_0,975 + z_0,8) · SD_diff / √n = {Z_MDE:.4f} · SD_diff / √500; SD_diff = Stichproben-SD der Differenzen je Frage.")
    schreibe("")
    mde_zeilen, kategorien = [], {}

    def mde_add(kat, label, s, x, y):
        m, sd = mde(x, y)
        mde_zeilen.append([kat, label, s, f"{sd:.3f}", f3(m), nz(x, y)])
        kategorien.setdefault(kat, []).append(m)

    for s in daten.seeds:
        v = daten.gewaehlt(daten.durchgang(s, "verfahren"), "abgerechnet")
        b1 = daten.gewaehlt(daten.durchgang(s, "ohne-optimierung"), "abgerechnet")
        mde_add("Verfahren − B1", f"DeepSeek − {KURZ[b1.modell]}", s, v.werte, b1.werte)
        for preis in PREISE:
            b2 = daten.gewaehlt(daten.durchgang(s, "gleichverteilung"), preis)
            mde_add(f"Verfahren − B2 ({preis})", f"DeepSeek − {KURZ[b2.modell]}", s, v.werte, b2.werte)
        b3 = daten.tests_von(daten.durchgang(s, "ein-modell", DEEPSEEK).nummer)[DEEPSEEK]
        mde_add("Verfahren − B3 DeepSeek", "DeepSeek − DeepSeek", s, v.werte, b3.werte)
        for bl in ("ohne-optimierung", "gleichverteilung"):
            pr = progs(s, bl)
            for a, b in ((DEEPSEEK, GEMMA), (GEMMA, GPTOSS), (GPTOSS, LUNA)):
                mde_add(f"{KURZ[a]} − {KURZ[b]} ({STRAT[bl]})", f"{KURZ[a]} − {KURZ[b]}", s, pr[a].werte, pr[b].werte)
    schreibe(tabelle(["Vergleich", "Programme", "Seed", "SD_diff", "MDE", "Fragen mit Δ≠0"], mde_zeilen))
    schreibe("")
    schreibe("Spannen je Vergleich:")
    schreibe("")
    schreibe(tabelle(["Vergleich", "MDE min … max"], [[k, spanne(v)] for k, v in kategorien.items()]))
    schreibe("")
    # DeepSeek gegen DeepSeek: alle Zeilen, deren Programme beide DeepSeek sind
    dd = [float(r[4]) for r in mde_zeilen if r[1] == "DeepSeek − DeepSeek"]
    versch = [float(r[4]) for r in mde_zeilen if r[1] != "DeepSeek − DeepSeek"]
    schreibe(f"- Beide Programme DeepSeek (Verfahren gegen B1, B2 mit DeepSeek, B3 DeepSeek): {spanne(dd)}")
    schreibe(f"- Verschiedene Modelle: {spanne(versch)}")
    schreibe("")

    # ---------------------------------------------------------------- 6 t-Intervalle
    schreibe("## 6. 95-%-t-Intervall über die Seeds (df = 2) für die mittlere Differenz")
    schreibe("")
    tq = stats.t.ppf(0.975, 2)
    tabellenzeilen = []
    intervalle = {}
    for preis in PREISE:
        vg = daten.vergleiche(preis)
        for gegen in ("B1", "B2", "B3 DeepSeek", "B3 Gemma", "B3 gpt-oss", "B3 Luna"):
            if preis == "festgehalten" and gegen != "B2":
                continue
            ds = [v["differenz"] for v in vg if v["gegen"] == gegen]
            m, sd = statistics.fmean(ds), statistics.stdev(ds)
            h = tq * sd / math.sqrt(len(ds))
            intervalle[(gegen, preis)] = (m - h, m + h)
            tabellenzeilen.append([f"Verfahren − {gegen}", preis if gegen == "B2" else "beide gleich",
                                   ", ".join(f3(d) for d in ds), f3(m), f3(sd), f3(m - h), f3(m + h)])
    schreibe(f"t_0,975;2 = {tq:.4f}")
    schreibe("")
    schreibe(tabelle(["Differenz", "Preis", "je Seed", "Mittel", "SD", "untere Grenze", "obere Grenze"], tabellenzeilen))
    schreibe("")

    # ---------------------------------------------------------------- 7 FF4
    schreibe("## 7. FF4 je Seed, beide Preisfassungen")
    schreibe("")
    ff4_alle = {preis: daten.ff4(preis) for preis in PREISE}
    for preis in PREISE:
        schreibe(f"**{preis}**")
        schreibe("")
        tabellenzeilen = []
        for s in daten.seeds:
            v = next(r for r in ff4_alle[preis] if r["seed"] == s and r["baseline"] == "verfahren")
            g = next(r for r in ff4_alle[preis] if r["seed"] == s and r["baseline"] == "ein-modell"
                     and r["gewaehlt"] == GPTOSS)
            t_g = daten.tests_von(g["durchgang"])[GPTOSS]
            t_b1 = daten.gewaehlt(daten.durchgang(s, "ohne-optimierung"), preis)
            t_v = daten.gewaehlt(daten.durchgang(s, "verfahren"), preis)
            tabellenzeilen.append([s, f3(v["c_opt"]), f3(v["c_exec_1000"]), f3(v["c_ref_1000"]), f"{v['c_opt_laeufe']:,.0f}",
                                   "keiner" if v["break_even"] is None else f"{v['break_even']:,.0f}",
                                   f3(g["c_opt"]), f3(g["c_exec_1000"]), f3(g["c_exec_1000"] / g["c_ref_1000"]),
                                   "keiner" if g["break_even"] is None else f"{g['break_even']:,.0f}",
                                   f3(t_g.bewertung - t_b1.bewertung), f3(t_g.bewertung - t_v.bewertung)])
        schreibe(tabelle(["Seed", "Verf. C_opt $", "Verf. c_exec $/1000", "B1 c_exec $/1000", "Verf. C_opt in Läufen",
                          "Verf. Break-even vs B1", "B3 gpt-oss C_opt $", "B3 gpt-oss c_exec $/1000",
                          "Faktor gpt-oss/B1", "gpt-oss Break-even vs B1", "ΔF1 gpt-oss − B1", "ΔF1 gpt-oss − Verfahren"], tabellenzeilen))
        schreibe("")
    schreibe("Schnittpunkt der mittleren kumulierten Kostenlinien B1 und B3 gpt-oss (Mittel von C_opt und c_exec über die Seeds):")
    schreibe("")
    kosten = {p: daten.kosten_je_durchgang(p) for p in PREISE}
    tabellenzeilen = []
    for preis in PREISE:
        g = [r for r in ff4_alle[preis] if r["baseline"] == "ein-modell" and r["gewaehlt"] == GPTOSS]
        copt_g = statistics.fmean(r["c_opt"] for r in g)
        ce_g = statistics.fmean(r["c_exec_1000"] for r in g) / 1000
        ce_b1 = statistics.fmean(r["c_ref_1000"] for r in g) / 1000
        copt_b1 = statistics.fmean(k["c_opt"] for k in kosten[preis].values() if k["baseline"] == "ohne-optimierung")
        n0 = copt_g / (ce_b1 - ce_g)
        n1 = (copt_g - copt_b1) / (ce_b1 - ce_g)
        tabellenzeilen.append([preis, f3(copt_g), f"{ce_g * 1000:.4f}", f"{ce_b1 * 1000:.4f}", f3(copt_b1),
                               f"{n0:,.0f}", f"{n1:,.0f}"])
    schreibe(tabelle(["Preis", "Ø C_opt gpt-oss B3 $", "Ø c_exec gpt-oss $/1000", "Ø c_exec B1 $/1000",
                      "Ø Bewertungskosten B1 $", "Schnitt (B1 ohne Fixkosten)", "Schnitt (B1 mit 0,167 $ Fixkosten)"], tabellenzeilen))
    schreibe("")

    # ---------------------------------------------------------------- 8 Planabweichung
    schreibe("## 8. Planabweichung je Optimierungslauf")
    schreibe("")
    schreibe("Plan = `dollar_geplant` der Optimierungszeile = Kostenmodell für die geplanten Trials "
             "(Fixkosten + num_trials · Kosten je Trial, ADR-0014/0017), **ohne** Nachbewertung. "
             "Ist = `dollar` der Optimierungszeile (abgerechnet; festgehalten nach ADR-0028). Daneben die "
             "Abweichung vom Budgetanteil (`dollar_budgetanteil`). Nur Läufe gültiger Durchgänge.")
    schreibe("")
    tabellenzeilen, je_gruppe = [], {}
    for d in daten.durchgaenge:
        for r in d.zeilen:
            if r["abschnitt"] != "optimierung":
                continue
            ist, ist_f, plan, anteil = r["dollar"], dollar_fest(r), r["dollar_geplant"], r["dollar_budgetanteil"]
            gruppe = (STRAT[d.baseline] + (f" {r['phase']}" if r.get("phase") else ""), KURZ[modell(r)])
            je_gruppe.setdefault(gruppe, []).append(((ist - plan) / plan, (ist_f - plan) / plan))
            tabellenzeilen.append([d.seed, d.nummer, gruppe[0], gruppe[1], r["num_trials_geplant"], f3(anteil), f3(plan),
                                   f3(ist), pct((ist - plan) / plan), pct((ist_f - plan) / plan), pct((ist - anteil) / anteil)])
    schreibe(tabelle(["Seed", "DG", "Strategie/Phase", "Modell", "Trials", "Budgetanteil $", "Plan $", "Ist $",
                      "Abw. vom Plan (abger.)", "Abw. vom Plan (fest)", "Abw. vom Budgetanteil (abger.)"], tabellenzeilen))
    schreibe("")
    schreibe("Je Strategie/Phase und Modell (Spanne über die Seeds):")
    schreibe("")
    schreibe(tabelle(["Strategie/Phase", "Modell", "Abw. abgerechnet", "Abw. festgehalten"],
                     [[g[0], g[1], spanne([a for a, _ in v], pct), spanne([b for _, b in v], pct)]
                      for g, v in je_gruppe.items()]))
    schreibe("")
    alle_abw = [a for v in je_gruppe.values() for a, _ in v]
    schreibe(f"Alle Optimierungsläufe: {spanne(alle_abw, pct)} (abgerechnet), "
             f"{spanne([b for v in je_gruppe.values() for _, b in v], pct)} (festgehalten).")
    schreibe("")
    schreibe("Je Durchgang (wie `kosten-je-durchgang-*.csv`: C_opt = Optimierung + Nachbewertung gegen die Summe der "
             "geplanten Dollar beider Messläufe):")
    schreibe("")
    tabellenzeilen = []
    for preis in PREISE:
        k = [v for v in kosten[preis].values() if v["abw_plan"] is not None]
        tabellenzeilen.append([preis, spanne([v["abw_plan"] for v in k], pct)])
        gruppen = {}
        for v in k:
            gruppen.setdefault((STRAT[v["baseline"]], KURZ.get(v["km"], "")), []).append(v["abw_plan"])
        tabellenzeilen.append([f"{preis}: Mittel je Strategie",
                               "; ".join(f"{g[0]}{' ' + g[1] if g[1] else ''} {pct(statistics.fmean(x))}" for g, x in gruppen.items())])
        tabellenzeilen.append([f"{preis}: Spanne der Strategie-Mittel", spanne([statistics.fmean(x) for x in gruppen.values()], pct)])
    schreibe(tabelle(["Preis", "Abweichung je Durchgang"], tabellenzeilen))
    schreibe("")

    # ---------------------------------------------------------------- 9 Messrauschen B1
    schreibe("## 9. Messrauschen: B1 je Seed und Modell")
    schreibe("")
    tabellenzeilen = []
    for m in NAMEN:
        werte = []
        for s in daten.seeds:
            t = progs(s, "ohne-optimierung")[m]
            nb = daten.auswahl_nachbewertung(t)
            werte.append((nb["bewertung"], t.bewertung, nb["dollar_je_programmlauf"] * 1000,
                          nb["dollar_je_programmlauf"] * 1000 * faktor_fest(nb), t.je_1000, t.je_1000_fest))
        for i, s in enumerate(daten.seeds):
            tabellenzeilen.append([KURZ[m], s] + [f3(x) if j < 2 else f"{x:.4f}" for j, x in enumerate(werte[i])])
        tabellenzeilen.append([KURZ[m], "Ø ± SD"] + [msd([wv[j] for wv in werte], f3 if j < 2 else (lambda x: "–" if x is None else f"{x:.4f}"))
                                         for j in range(6)])
        tabellenzeilen.append([KURZ[m], "Spanne max−min"] + [f"{max(wv[j] for wv in werte) - min(wv[j] for wv in werte):.4f}"
                                                             for j in range(6)])
    schreibe(tabelle(["Modell", "Seed", "F1 Auswahl", "F1 Test", "$/1000 Auswahl", "$/1000 Auswahl fest",
                      "$/1000 Test", "$/1000 Test fest"], tabellenzeilen))
    schreibe("")

    # ---------------------------------------------------------------- 10 Auswahl-Test-Lücke
    schreibe("## 10. Lücke Auswahl − Test (B1) und Vorsprung DeepSeek vor Gemma")
    schreibe("")
    tabellenzeilen = []
    for m in NAMEN:
        g = []
        for s in daten.seeds:
            t = progs(s, "ohne-optimierung")[m]
            g.append(daten.auswahl_nachbewertung(t)["bewertung"] - t.bewertung)
        tabellenzeilen.append([KURZ[m]] + [f3(x) for x in g] + [msd(g)])
    schreibe(tabelle(["Modell (B1)", "S1", "S2", "S3", "Ø ± SD"], tabellenzeilen))
    schreibe("")
    tabellenzeilen = []
    for titel, bl, phase, auf_testdaten in (("B1 Auswahl", "ohne-optimierung", None, False),
                                   ("B1 Test", "ohne-optimierung", None, True),
                                   ("Sondierung B/8 Auswahl", "verfahren", "sondierung", False),
                                   ("B2 Auswahl", "gleichverteilung", None, False),
                                   ("B2 Test", "gleichverteilung", None, True)):
        ds, fd, fg = [], [], []
        for s in daten.seeds:
            d = daten.durchgang(s, bl)
            if auf_testdaten:
                ts = daten.tests_von(d.nummer)
                a, b = ts[DEEPSEEK].bewertung, ts[GEMMA].bewertung
            else:
                st = {x.name: x for x in daten.staende(d.nummer, phase)}
                a, b = st[DEEPSEEK].bewertung, st[GEMMA].bewertung
            ds.append(a - b); fd.append(a); fg.append(b)
        tabellenzeilen.append([titel] + [f3(x) for x in ds] + [f3(statistics.fmean(ds)), f3(statistics.fmean(fd)), f3(statistics.fmean(fg))])
    schreibe(tabelle(["Datensatz", "Δ S1", "Δ S2", "Δ S3", "Ø Δ", "Ø F1 DeepSeek", "Ø F1 Gemma"], tabellenzeilen))
    schreibe("")
    schreibe("Sondierung hat keine Testbewertung (nur DeepSeeks Fokus-Programm wurde getestet).")
    schreibe("")

    # ---------------------------------------------------------------- 11 Latenz
    schreibe("## 11. Latenz (Median je Testbewertung, s): B1 gegen B2 und B3")
    schreibe("")
    tabellenzeilen = []
    langsamer = {"B2": [], "B3": []}
    for m in NAMEN:
        lat = {bl: [progs(s, bl)[m].latenz_median for s in daten.seeds]
               for bl in ("ohne-optimierung", "gleichverteilung", "ein-modell")}
        mb1 = statistics.fmean(lat["ohne-optimierung"])
        zeile = [KURZ[m], " / ".join(f"{x:.2f}" for x in lat["ohne-optimierung"]), f"{mb1:.2f}"]
        for bl, k in (("gleichverteilung", "B2"), ("ein-modell", "B3")):
            mb = statistics.fmean(lat[bl])
            n_l = sum(x > y for x, y in zip(lat[bl], lat["ohne-optimierung"]))
            zeile += [" / ".join(f"{x:.2f}" for x in lat[bl]), f"{mb:.2f}", f"{mb - mb1:+.2f}",
                      f"{n_l}/3"]
            if mb > mb1:
                langsamer[k].append(KURZ[m])
        tabellenzeilen.append(zeile)
    schreibe(tabelle(["Modell", "B1 S1/S2/S3", "B1 Ø", "B2 S1/S2/S3", "B2 Ø", "Δ B2−B1", "Seeds langsamer",
                      "B3 S1/S2/S3", "B3 Ø", "Δ B3−B1", "Seeds langsamer"], tabellenzeilen))
    schreibe("")
    schreibe(f"Im Mittel langsamer als B1 — mit B2: {', '.join(langsamer['B2'])} ({len(langsamer['B2'])} von 4); "
             f"mit B3: {', '.join(langsamer['B3'])} ({len(langsamer['B3'])} von 4).")
    schreibe("")

    # ---------------------------------------------------------------- 12 Pareto
    schreibe("## 12. Pareto-Front (Mittel über die Seeds, Testdaten, F1 gegen $ je 1000)")
    schreibe("")
    for preis in PREISE:
        punkte = []
        for (bl, m) in dict.fromkeys((t.baseline, t.modell) for t in daten.tests):
            ts = [t for t in daten.tests if (t.baseline, t.modell) == (bl, m)]
            punkte.append((f"{STRAT[bl]} {KURZ[m]}", statistics.fmean(t.bewertung for t in ts),
                           statistics.fmean((t.je_1000 if preis == "abgerechnet" else t.je_1000_fest) for t in ts)))
        tabellenzeilen, front = [], []
        for name, q0, k0 in punkte:
            ist_dominiert = any(k <= k0 and q >= q0 and (k < k0 or q > q0) for _, q, k in punkte)
            tabellenzeilen.append([name, f3(q0), f3(k0), "nein" if not ist_dominiert else "ja"])
            if not ist_dominiert:
                front.append((k0, name))
        schreibe(f"**{preis}**: Front (nach Kosten) = " + ", ".join(n for _, n in sorted(front)))
        schreibe("")
        schreibe(tabelle(["Programm", "Ø F1 Test", "Ø $/1000 Test", "dominiert"], tabellenzeilen))
        schreibe("")

    print("\n".join(ausgabezeilen))


if __name__ == "__main__":
    main()
