"""Das Hauptexperiment: je Seed das Verfahren und alle Baselines, danach als eigene, letzte
Stufe die Testbewertung.

Je Seed laufen sieben Durchgaenge in fester Reihenfolge: ohne Optimierung, Verfahren,
Gleichverteilung, dann je Kandidatenmodell "ein Kandidatenmodell bekommt alles" in der
Reihenfolge von `DIE_VIER`. Jeder Durchgang hat eine eigene, ueber das ganze Experiment
fortlaufende Nummer: die naechste ist die hoechste im Protokoll plus 1, keine wird
wiederverwendet.

Fortgesetzt wird je Durchgang aus dem Protokoll. Fertig ist ein Durchgang, wenn seine
Entscheidungszeile im Protokoll steht und seine Programme gespeichert sind; ein angefangener
wird beim Neustart vollstaendig mit neuer Nummer wiederholt, seine Zeilen bleiben im Protokoll.
Ein Anbieterfehler haelt das Modul an, ohne den Durchgang selbst zu wiederholen; ebenso ein
Restbudget der Fokusphase, das fuer keinen Trial reicht. Nach jedem Seed haelt das Modul an und
stellt die Kosten dem Plan gegenueber; der naechste Seed beginnt erst mit einem neuen Aufruf.

Die Testdaten werden erst bewertet, wenn alle Seeds fertig sind. Bis dahin wird die Naht ohne
Testdaten gebaut. Bewertet werden die gespeicherten Modell-Programm-Paare, jedes genau einmal
gueltig; ein Anbieterfehler wird einmal sofort wiederholt, beim zweiten haelt das Modul an.

Die gespeicherten Programme liegen unter `ergebnisse/programme/`, je Durchgang ein Ordner mit
einer Datei je Paar und dem Manifest `durchgang.json`, das die Paare aufzaehlt. Sie werden nach
jedem Seed zusammen mit dem Protokoll versioniert.

Aufruf (lange Laeufe etwa in `tmux` auf einem Server):
    python -u -m dspy_optimizer.hauptexperiment                 misst, was im aktuellen Seed fehlt
    python -m dspy_optimizer.hauptexperiment stand              Kosten gegen Plan, kostet nichts
    python -u -m dspy_optimizer.hauptexperiment testbewertung   die letzte Stufe
"""

import datetime
import json
import os
import pathlib
import statistics
import sys
from typing import NamedTuple

from dotenv import load_dotenv

from dspy_optimizer.baselines import Baselines
from dspy_optimizer.datensatz import GROESSEN, Aufteilung, berechne_fingerabdruck, lade_aufteilung
from dspy_optimizer.kandidatenmodelle import DIE_VIER, kandidatenmodell_mit_namen
from dspy_optimizer.kostenerfassung import messhygiene
from dspy_optimizer.kostenmodell import BudgetReichtNicht
from dspy_optimizer.optimierungslauf import Optimierungslauf
from dspy_optimizer.programm import ModellProgrammPaar, MultiHopQA
from dspy_optimizer.protokoll import WURZEL, Einordnung, Protokoll
from dspy_optimizer.verfahren import UngueltigerDurchgang, Verfahren
from dspy_optimizer.vorstudie import werte_aus

# Das Optimierungsbudget B je Durchgang in Dollar.
B = 6.0

# Die Kostenobergrenze in Dollar je 1000 Programmlaeufe, vor dem Hauptexperiment festgeschrieben.
# Die Vorstudie ergibt 0,28954 und muss auf diesen
# Wert runden (`lehne_start_ohne_voraussetzungen_ab`).
KOSTENOBERGRENZE = 0.2895

# Fingerabdruecke der vier Teile der festen Aufteilung, am 09.10.2026 auf zwei Rechnern (Server
# und Windows-Rechner) uebereinstimmend gemessen.
FINGERABDRUECKE = Aufteilung(
    trainingsdaten="e910af1a022a",
    validierungsdaten="280b5593cd7e",
    auswahldaten="a015835f06a4",
    testdaten="1d3b4ba933a5",
)

SEEDS = (1, 2, 3)

# Die harte Obergrenze aller API-Kosten und der Wechselkurs, zu dem B festgelegt wurde
# (EZB-Referenzkurs vom 09.10.2026). Der Bericht
# rechnet mit diesem festen Kurs; bei einer Notbremse gilt der Tageskurs.
GRENZE_EURO = 200
DOLLAR_JE_EURO = 1.1206

# Die Kuerzungsreihenfolge. Sie greift nie
# automatisch, sondern nur, wenn die Hochrechnung nach einem Seed ueber 200 EUR liegt und von
# Hand entschieden wird. Dann wird hier `KUERZUNG` gesetzt und versioniert, damit die Stufe fuer
# jeden weiteren Neustart gilt.
#   0  ungekuerzt
#   1  Einzelmodelle von vier auf zwei: DeepSeek V4.1 Flash und Gemma 4 31B, in der Vorstudie die
#      besten unter der Kostenobergrenze und damit die staerksten Gegner des Verfahrens
#   2  dazu die Testdaten von 500 auf die ersten 300 der festen Aufteilung
#   3  dazu die Seeds von drei auf zwei; unter zwei nicht
KUERZUNG = 0
BLEIBEN_BEI_KUERZUNG = ("google/gemma-4-31b-it", "deepseek/deepseek-v4.1-flash")
TESTDATEN_GEKUERZT = 300

PROGRAMME = WURZEL / "ergebnisse" / "programme"
MANIFEST = "durchgang.json"


class HauptexperimentAngehalten(RuntimeError):
    """Das Hauptexperiment hat angehalten, weil ueber das weitere Vorgehen von Hand zu
    entscheiden ist."""


class Umfang(NamedTuple):
    """Was gemessen wird: die Seeds, die Kandidatenmodelle der Einzelmodell-Durchgaenge und die
    Zahl der Testbeispiele."""

    seeds: tuple[int, ...]
    ein_modell: tuple[str, ...]
    testdaten: int


def umfang_nach_kuerzung(kuerzung):
    """Der Umfang nach einer Stufe der Kuerzungsreihenfolge (siehe `KUERZUNG`)."""
    if kuerzung not in range(4):
        raise ValueError(f"Kuerzungsstufen gibt es von 0 bis 3, nicht {kuerzung!r}.")
    return Umfang(
        seeds=SEEDS if kuerzung < 3 else SEEDS[:2],
        ein_modell=tuple(
            modell.name
            for modell in DIE_VIER
            if kuerzung < 1 or modell.name in BLEIBEN_BEI_KUERZUNG
        ),
        testdaten=GROESSEN.testdaten if kuerzung < 2 else TESTDATEN_GEKUERZT,
    )


UMFANG = umfang_nach_kuerzung(KUERZUNG)


class Aufgabe(NamedTuple):
    """Ein Durchgang, wie er im Plan steht: Seed, Baseline und, nur bei `ein-modell`, das
    Kandidatenmodell, das alles bekommt."""

    seed: int
    baseline: str
    kandidatenmodell: str | None = None


def aufgaben_im_seed(seed, umfang):
    """Die Durchgaenge eines Seeds in ihrer festen Reihenfolge."""
    return [
        Aufgabe(seed, "ohne-optimierung"),
        Aufgabe(seed, "verfahren"),
        Aufgabe(seed, "gleichverteilung"),
    ] + [Aufgabe(seed, "ein-modell", name) for name in umfang.ein_modell]


class Durchgang(NamedTuple):
    """Ein fertiger Durchgang: seine Nummer, seine Aufgabe und was von ihm gespeichert ist."""

    nummer: int
    aufgabe: Aufgabe
    manifest: dict


def finde_fertige_durchgaenge(zeilen, programme):
    """Die fertigen Durchgaenge im Protokoll, je Aufgabe der erste.

    Fertig heisst: Zweck Experiment; eine Entscheidungszeile - die Endauswahl, beim Verfahren
    ohne Gewinner nach der Sondierung die Zwischenbewertung -; in keiner Zeile ein
    Anbieterfehler; und die Programme sind gespeichert. Die Zeilen der Testbewertung zaehlen
    dafuer nicht mit: sie entstehen erst in der letzten Stufe.
    """
    nach_nummer = {}
    for zeile in zeilen:
        if zeile.get("zweck") != "experiment" or zeile.get("abschnitt") == "testbewertung":
            continue
        nach_nummer.setdefault(zeile["durchgang"], []).append(zeile)
    fertige_durchgaenge = {}
    for nummer, zeilen_des_durchgangs in sorted(nach_nummer.items()):
        entscheidung = _entscheidungszeile(zeilen_des_durchgangs)
        if entscheidung is None or any(
            zeile.get("anbieterfehler") for zeile in zeilen_des_durchgangs
        ):
            continue
        manifest = _lies_manifest(programme, nummer)
        if manifest is None:
            continue
        aufgabe = _aufgabe_aus_entscheidung(entscheidung)
        fertige_durchgaenge.setdefault(aufgabe, Durchgang(nummer, aufgabe, manifest))
    return fertige_durchgaenge


def _entscheidungszeile(zeilen):
    for zeile in zeilen:
        if zeile.get("abschnitt") == "endauswahl":
            return zeile
        if zeile.get("abschnitt") == "zwischenbewertung" and zeile.get("kein_gewinner"):
            return zeile
    return None


def _aufgabe_aus_entscheidung(entscheidung):
    """Die Aufgabe aus der Entscheidungszeile; beim Einzelmodell steht dort genau eines."""
    kandidatenmodell = None
    if entscheidung["baseline"] == "ein-modell":
        (kandidatenmodell,) = entscheidung["kandidatenmodelle"]
    return Aufgabe(entscheidung["seed"], entscheidung["baseline"], kandidatenmodell)


def _ordner_des_durchgangs(programme, nummer):
    return pathlib.Path(programme) / f"durchgang-{nummer:03d}"


def _lies_manifest(programme, nummer):
    """Das Manifest des Durchgangs, wenn es und jedes Paar darin auf der Platte liegen."""
    pfad = _ordner_des_durchgangs(programme, nummer) / MANIFEST
    if not pfad.exists():
        return None
    manifest = json.loads(pfad.read_text(encoding="utf-8"))
    if not all((pathlib.Path(programme) / paar["datei"]).exists() for paar in manifest["paare"]):
        return None
    return manifest


def finde_naechste_durchgangsnummer(zeilen):
    """Die hoechste Durchgangsnummer im Protokoll plus 1."""
    nummern = [
        zeile["durchgang"]
        for zeile in zeilen
        if zeile.get("zweck") == "experiment" and zeile.get("durchgang")
    ]
    return max(nummern, default=0) + 1


def finde_aktuellen_seed(fertige_durchgaenge, umfang):
    """Der erste Seed, in dem noch ein Durchgang fehlt; `None`, wenn alle fertig sind."""
    for seed in umfang.seeds:
        if any(aufgabe not in fertige_durchgaenge for aufgabe in aufgaben_im_seed(seed, umfang)):
            return seed
    return None


class PaarZurTestbewertung(NamedTuple):
    """Ein Modell-Programm-Paar, das eine Testbewertung bekommt, vor dem Speichern.

    `programm` traegt kein eigenes LM. `phase` nur beim Verfahren. `rolle`: `gewinner`,
    `teilnehmer` (die uebrigen Programme einer Baseline), `fuehrender-ohne-gewinner` oder
    `zusatzangabe` (beide beim Verfahren ohne Gewinner).
    """

    kandidatenmodell: str
    phase: str | None
    rolle: str
    programm: object


def waehle_paare_der_baseline(ergebnis):
    """Jedes Programm einer Baseline, auch wenn sie ohne Gewinner blieb."""
    return [
        PaarZurTestbewertung(
            name, None, "gewinner" if name == ergebnis.gewinner else "teilnehmer", programm
        )
        for name, programm in ergebnis.programme.items()
    ]


def waehle_paare_des_verfahrens(ergebnis):
    """Mit Gewinner sein Programm aus der Fokusphase. Ohne Gewinner nach der Fokusphase der
    Fuehrende dort ohne Kostenfilter und die Zusatzangabe aus der Sondierung; endet das
    Verfahren schon nach der Sondierung, keines."""
    if ergebnis.gewinner is not None:
        gewinner = ergebnis.gewinner
        return [
            PaarZurTestbewertung(
                gewinner,
                "fokus",
                "gewinner",
                ergebnis.ergebnisse_der_fokusphase[gewinner].programm,
            )
        ]
    if not ergebnis.ergebnisse_der_fokusphase:
        return []
    fuehrender = ergebnis.fuehrender_ohne_kostenfilter
    zusatzangabe = ergebnis.zusatzangabe.kandidatenmodell
    return [
        PaarZurTestbewertung(
            fuehrender,
            "fokus",
            "fuehrender-ohne-gewinner",
            ergebnis.ergebnisse_der_fokusphase[fuehrender].programm,
        ),
        PaarZurTestbewertung(
            zusatzangabe,
            "sondierung",
            "zusatzangabe",
            ergebnis.zwischenstand.ergebnisse[zusatzangabe].programm,
        ),
    ]


def speichere_durchgang(programme, nummer, aufgabe, kein_gewinner, paare):
    """Die Paare eines Durchgangs speichern, je eine JSON-Datei, und zuletzt das Manifest.

    Das Manifest kommt zuletzt und auf einen Schlag (`os.replace`): liegt es da, liegen auch
    alle Paare da. Ein Ordner, den es schon gibt, wird nicht ueberschrieben.
    """
    ordner = _ordner_des_durchgangs(programme, nummer)
    ordner.mkdir(parents=True)
    eintraege = []
    for paar in paare:
        datei = f"{ordner.name}/{paar.rolle}-{paar.kandidatenmodell.replace('/', '-')}.json"
        ModellProgrammPaar.aus_programm_und_modell(
            paar.programm, kandidatenmodell_mit_namen(paar.kandidatenmodell)
        ).save(pathlib.Path(programme) / datei)
        eintraege.append(
            {
                "datei": datei,
                "kandidatenmodell": paar.kandidatenmodell,
                "phase": paar.phase,
                "rolle": paar.rolle,
            }
        )
    manifest = {
        "durchgang": nummer,
        "seed": aufgabe.seed,
        "baseline": aufgabe.baseline,
        "kandidatenmodell": aufgabe.kandidatenmodell,
        "kein_gewinner": kein_gewinner,
        "paare": eintraege,
    }
    vorlaeufig = ordner / f"{MANIFEST}.tmp"
    vorlaeufig.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(vorlaeufig, ordner / MANIFEST)


def lade_modell_programm_paar(programme, datei):
    """Ein gespeichertes Paar, so geladen, wie es gespeichert wurde."""
    paar = ModellProgrammPaar()
    paar.load(pathlib.Path(programme) / datei)
    return paar


def finde_gueltig_bewertete_paare(zeilen, testbeispiele):
    """Die Paare (Datei), die schon eine gueltige Testbewertung haben: ohne Anbieterfehler, mit
    Bewertung, ueber genau so viele Testbeispiele, wie jetzt bewertet werden."""
    return {
        zeile["konfiguration"]["paar"]
        for zeile in zeilen
        if zeile.get("zweck") == "experiment"
        and zeile.get("abschnitt") == "testbewertung"
        and not zeile.get("anbieterfehler")
        and zeile.get("bewertung") is not None
        and zeile["konfiguration"].get("testdaten") == testbeispiele
        and zeile["konfiguration"].get("paar")
    }


class Hauptexperiment:
    """Faehrt die Durchgaenge des aktuellen Seeds und, als letzte Stufe, die Testbewertung.

    `kostenmodelle` je Kandidatenmodell in der Reihenfolge von `DIE_VIER`: sie ist die
    Reihenfolge der Laeufe und bricht als letzte Regel einen Gleichstand.
    """

    def __init__(
        self, optimierungslauf, protokoll, kostenmodelle, *, programme=PROGRAMME, umfang=UMFANG
    ):
        self.optimierungslauf = optimierungslauf
        self.protokoll = protokoll
        self.programme = pathlib.Path(programme)
        self.umfang = umfang
        angaben = {
            "optimierungslauf": optimierungslauf,
            "kostenmodelle": kostenmodelle,
            "kostenobergrenze": KOSTENOBERGRENZE,
            "protokoll": protokoll,
        }
        self.verfahren = Verfahren(**angaben)
        self.baselines = Baselines(**angaben)

    def finde_fertige_durchgaenge(self):
        return finde_fertige_durchgaenge(self.protokoll.zeilen(), self.programme)

    def __call__(self):
        """Misst, was im aktuellen Seed fehlt, und haelt danach an. Gibt den Seed zurueck, der
        dabei fertig wurde, oder `None`, wenn schon alle fertig waren."""
        fertige_durchgaenge = self.finde_fertige_durchgaenge()
        seed = finde_aktuellen_seed(fertige_durchgaenge, self.umfang)
        if seed is None:
            print("Alle Seeds sind fertig. Naechste Stufe: testbewertung.", flush=True)
            return None
        for aufgabe in aufgaben_im_seed(seed, self.umfang):
            if aufgabe in fertige_durchgaenge:
                continue
            nummer = finde_naechste_durchgangsnummer(self.protokoll.zeilen())
            print(f"[{_jetzt()}] Durchgang {nummer}: {_beschreibe(aufgabe)} beginnt", flush=True)
            try:
                kein_gewinner, paare = self._fuehre_durchgang_aus(aufgabe, nummer)
            except UngueltigerDurchgang as fehler:
                raise HauptexperimentAngehalten(
                    f"Durchgang {nummer} ({_beschreibe(aufgabe)}) ist ungueltig: {fehler} "
                    "Welcher Fehler es war, steht unter `stoerungen` in seinen letzten "
                    "Protokollzeilen. Nach der Klaerung neu starten: wiederholt wird nur dieser "
                    "Durchgang, mit neuer Nummer (ADR-0027, Punkt 7)."
                ) from fehler
            except BudgetReichtNicht as fehler:
                raise HauptexperimentAngehalten(
                    f"Durchgang {nummer} ({_beschreibe(aufgabe)}) laesst sich nicht planen: "
                    f"{fehler} Miray entscheidet, wie es weitergeht (ADR-0027, "
                    "Bauentscheidungen). Ein Neustart wiederholt den Durchgang mit neuer Nummer."
                ) from fehler
            speichere_durchgang(self.programme, nummer, aufgabe, kein_gewinner, paare)
            ausgegeben = sum(
                zeile.get("dollar") or 0.0
                for zeile in self.protokoll.zeilen()
                if zeile.get("zweck") == "experiment" and zeile.get("durchgang") == nummer
            )
            print(
                f"[{_jetzt()}] Durchgang {nummer} fertig: "
                f"{'kein Gewinner' if kein_gewinner else 'mit Gewinner'}, "
                f"{len(paare)} Programme gespeichert, {ausgegeben:.4f} $",
                flush=True,
            )
        print(f"[{_jetzt()}] Seed {seed} ist fertig. Das Modul haelt an.", flush=True)
        return seed

    def _fuehre_durchgang_aus(self, aufgabe, nummer):
        """Fuehrt einen Durchgang aus; zurueck kommen, ob er ohne Gewinner blieb, und seine
        Paare."""
        startprogramm = MultiHopQA()
        if aufgabe.baseline == "verfahren":
            ergebnis = self.verfahren(startprogramm, B, seed=aufgabe.seed, durchgang=nummer)
            return ergebnis.kein_gewinner, waehle_paare_des_verfahrens(ergebnis)
        if aufgabe.baseline == "ohne-optimierung":
            ergebnis = self.baselines.ohne_optimierung(
                startprogramm, seed=aufgabe.seed, durchgang=nummer
            )
        elif aufgabe.baseline == "gleichverteilung":
            ergebnis = self.baselines.gleichverteilung(
                startprogramm, B, seed=aufgabe.seed, durchgang=nummer
            )
        else:
            ergebnis = self.baselines.ein_modell(
                aufgabe.kandidatenmodell, startprogramm, B, seed=aufgabe.seed, durchgang=nummer
            )
        return ergebnis.kein_gewinner, waehle_paare_der_baseline(ergebnis)

    def testbewertung(self):
        """Die letzte Stufe: jedes gespeicherte Paar jedes fertigen Durchgangs bekommt genau
        eine gueltige Testbewertung. Was schon gueltig bewertet ist, wird nicht noch einmal
        bezahlt."""
        fertige_durchgaenge = self.finde_fertige_durchgaenge()
        seed = finde_aktuellen_seed(fertige_durchgaenge, self.umfang)
        if seed is not None:
            raise HauptexperimentAngehalten(
                f"Seed {seed} ist noch nicht fertig. Die Testdaten werden erst bewertet, wenn "
                "alle Seeds fertig sind (ADR-0027, Punkt 6)."
            )
        testbeispiele = len(self.optimierungslauf.testdaten)
        gueltig_bewertete_paare = finde_gueltig_bewertete_paare(
            self.protokoll.zeilen(), testbeispiele
        )
        alle_paare = [
            (durchgang, paar)
            for durchgang in sorted(
                fertige_durchgaenge.values(), key=lambda durchgang: durchgang.nummer
            )
            for paar in durchgang.manifest["paare"]
        ]
        offene_paare = [
            (durchgang, paar)
            for durchgang, paar in alle_paare
            if paar["datei"] not in gueltig_bewertete_paare
        ]
        print(
            f"[{_jetzt()}] Testbewertung ueber {testbeispiele} Testbeispiele: "
            f"{len(alle_paare)} Paare, davon {len(offene_paare)} noch ohne gueltige Testbewertung",
            flush=True,
        )
        for durchgang, paar in offene_paare:
            for versuch in (1, 2):
                ergebnis = self._bewerte_auf_testdaten(durchgang, paar)
                print(
                    f"[{_jetzt()}] {paar['datei']}: Bewertung {ergebnis.bewertung:.3f}, "
                    f"Formatfehler {ergebnis.formatfehler}, "
                    f"Anbieterfehler {ergebnis.anbieterfehler}",
                    flush=True,
                )
                if not ergebnis.anbieterfehler:
                    break
                if versuch == 2:
                    raise HauptexperimentAngehalten(
                        f"Die Testbewertung von {paar['datei']} meldet zweimal hintereinander "
                        "Anbieterfehler. Die Zeilen zaehlen nicht (ADR-0027, Punkt 12); welcher "
                        "Fehler es war, steht unter `stoerungen`. Nach der Klaerung neu starten: "
                        "gueltig bewertete Paare werden nicht wiederholt."
                    )
                print("Anbieterfehler: einmal sofort wiederholt", flush=True)
        print(f"[{_jetzt()}] Testbewertung fertig.", flush=True)

    def _bewerte_auf_testdaten(self, durchgang, paar):
        """Dasselbe gespeicherte Paar beantwortet die Testfragen, ohne MIPROv2."""
        geladen = lade_modell_programm_paar(self.programme, paar["datei"])
        kandidatenmodell = kandidatenmodell_mit_namen(paar["kandidatenmodell"])
        if geladen.kandidatenmodell != kandidatenmodell.name:
            raise ValueError(
                f"{paar['datei']} laeuft auf {geladen.kandidatenmodell}, laut Manifest auf "
                f"{kandidatenmodell.name}."
            )
        return self.optimierungslauf.nur_bewerten(
            kandidatenmodell,
            geladen.programm_ohne_eigenes_lm(),
            durchgang.aufgabe.seed,
            einordnung=Einordnung.experiment(
                durchgang.aufgabe.baseline, durchgang=durchgang.nummer, phase=paar["phase"]
            ),
            abschnitt="testbewertung",
            zusaetzliche_konfiguration={
                "paar": paar["datei"],
                "rolle": paar["rolle"],
                "kein_gewinner": durchgang.manifest["kein_gewinner"],
            },
        )


def _beschreibe(aufgabe):
    namen = {
        "ohne-optimierung": "ohne Optimierung",
        "verfahren": "Verfahren",
        "gleichverteilung": "Gleichverteilung",
        "ein-modell": f"ein Kandidatenmodell bekommt alles ({aufgabe.kandidatenmodell})",
    }
    return f"Seed {aufgabe.seed}, {namen[aufgabe.baseline]}"


def _jetzt():
    return datetime.datetime.now().strftime("%d.%m. %H:%M")


def lehne_start_ohne_voraussetzungen_ab(aufteilung, auswertung):
    """Prueft vor dem ersten bezahlten Aufruf die Voraussetzungen und bricht sonst ohne Kosten
    ab: dieselbe Aufteilung wie in der Vorstudie, die Vorstudie vollstaendig im Protokoll, und
    sie ergibt die festgeschriebene Kostenobergrenze."""
    for name, teil, festgehalten in zip(Aufteilung._fields, aufteilung, FINGERABDRUECKE):
        if berechne_fingerabdruck(teil) != festgehalten:
            raise HauptexperimentAngehalten(
                f"Die {name.capitalize()} haben den Fingerabdruck {berechne_fingerabdruck(teil)}, "
                f"festgehalten ist {festgehalten} (ADR-0002, Nachtrag 09.10.2026). Die Aufteilung "
                "ist eine andere als in der Vorstudie."
            )
    if auswertung.offen:
        raise HauptexperimentAngehalten(
            f"Die Vorstudie ist im Protokoll nicht vollstaendig: es fehlen "
            f"{len(auswertung.offen)} Laeufe. Ohne sie gibt es weder Kostenmodelle noch "
            "Kostenobergrenze."
        )
    if round(auswertung.kostenobergrenze, 4) != KOSTENOBERGRENZE:
        raise HauptexperimentAngehalten(
            f"Die Vorstudie ergibt eine Kostenobergrenze von {auswertung.kostenobergrenze:.5f} $ "
            f"je 1000 Programmlaeufe, festgeschrieben ist {KOSTENOBERGRENZE} (ADR-0021, Nachtrag "
            "10.10.2026)."
        )


def ordne_kostenmodelle(auswertung):
    """Die Kostenmodelle der Vorstudie in der Reihenfolge von `DIE_VIER` - die Vorstudie lief
    rueckwaerts (`vorstudie.REIHENFOLGE`)."""
    return [auswertung.kostenmodelle[modell.name] for modell in DIE_VIER]


class Kostenplan(NamedTuple):
    """Was das Hauptexperiment nach der Vorstudie kosten soll, in Dollar.

    `durchgaenge_je_seed`: die Durchgaenge mit B und die Bewertung ohne Optimierung auf den
    Auswahldaten. `testbewertung_je_seed`: die Testbewertungen der Programme eines Seeds, die
    erst in der letzten Stufe anfallen. Zusammen ist das der Plan der Vorstudie je Seed
    (`vorstudie.Auswertung.hauptexperiment`, rund 38,60 $).
    """

    durchgaenge_je_seed: float
    testbewertung_je_seed: float
    seeds: tuple[int, ...]

    @property
    def je_seed(self):
        return self.durchgaenge_je_seed + self.testbewertung_je_seed

    @property
    def teststufe(self):
        return len(self.seeds) * self.testbewertung_je_seed

    @property
    def gesamt(self):
        return len(self.seeds) * self.je_seed


def plane_kosten(auswertung, umfang):
    """Derselbe Plan wie `vorstudie.Auswertung.hauptexperiment`, nach dem Umfang: je Seed
    Verfahren, Gleichverteilung und die Einzelmodelle mit B, die vier Bewertungen ohne
    Optimierung; als Testbewertungen die vier unoptimierten Programme, die vier der
    Gleichverteilung, die der Einzelmodelle und das Ergebnis des Verfahrens, gerechnet mit dem
    teuersten Kandidatenmodell."""
    if auswertung.offen:
        raise ValueError("Ohne vollstaendige Vorstudie gibt es keinen Kostenplan.")
    ergebnisse = auswertung.ergebnisse
    ausfuehrungskosten_ohne = sum(
        ergebnis.ausfuehrungskosten_ohne for ergebnis in ergebnisse.values()
    )
    ausfuehrungskosten_optimiert = {
        name: ergebnis.kostenmodell.ausfuehrungskosten.betrag
        for name, ergebnis in ergebnisse.items()
    }
    mit_b = 2 + len(umfang.ein_modell)
    return Kostenplan(
        durchgaenge_je_seed=mit_b * B + GROESSEN.auswahldaten * ausfuehrungskosten_ohne,
        testbewertung_je_seed=umfang.testdaten
        * (
            ausfuehrungskosten_ohne
            + sum(ausfuehrungskosten_optimiert.values())
            + sum(ausfuehrungskosten_optimiert[name] for name in umfang.ein_modell)
            + max(ausfuehrungskosten_optimiert.values())
        ),
        seeds=umfang.seeds,
    )


def rechne_hoch(plan, *, bisher, je_seed, fertige_seeds, test):
    """Was am Ende ausgegeben sein wird: bisher ausgegeben, dazu je offenem Seed der groessere
    Wert von Plan und tatsaechlichem Mittel je fertigem Seed - bei einem angefangenen nur, was
    davon noch fehlt -, und der Plan der Teststufe, soweit er noch nicht ausgegeben ist."""
    gemessen = [je_seed[seed] for seed in fertige_seeds]
    mittel = statistics.fmean(gemessen) if gemessen else 0.0
    je_offenem = max(plan.durchgaenge_je_seed, mittel)
    noch_offen = sum(
        max(0.0, je_offenem - je_seed.get(seed, 0.0))
        for seed in plan.seeds
        if seed not in fertige_seeds
    )
    return bisher + noch_offen + max(0.0, plan.teststufe - test)


def fasse_kosten_zusammen(zeilen, auswertung, programme, umfang):
    """Kosten gegen Plan und Hochrechnung, nur aus dem Protokoll - ohne einen API-Aufruf."""
    zeilen = list(zeilen)
    plan = plane_kosten(auswertung, umfang)
    fertige_durchgaenge = finde_fertige_durchgaenge(zeilen, programme)
    experiment = [zeile for zeile in zeilen if zeile.get("zweck") == "experiment"]
    tests = [zeile for zeile in experiment if zeile.get("abschnitt") == "testbewertung"]
    durchgaenge = [zeile for zeile in experiment if zeile.get("abschnitt") != "testbewertung"]

    je_seed, nummern = {}, {}
    for zeile in durchgaenge:
        je_seed[zeile["seed"]] = je_seed.get(zeile["seed"], 0.0) + (zeile.get("dollar") or 0.0)
        nummern.setdefault(zeile["seed"], set()).add(zeile["durchgang"])
    fertige_nummern = {durchgang.nummer for durchgang in fertige_durchgaenge.values()}
    fertige_seeds = tuple(
        seed
        for seed in umfang.seeds
        if all(aufgabe in fertige_durchgaenge for aufgabe in aufgaben_im_seed(seed, umfang))
    )
    test = sum(zeile.get("dollar") or 0.0 for zeile in tests)
    bisher = sum(zeile.get("dollar") or 0.0 for zeile in zeilen)
    im_experiment = sum(zeile.get("dollar") or 0.0 for zeile in experiment)
    hochgerechnet = rechne_hoch(
        plan, bisher=bisher, je_seed=je_seed, fertige_seeds=fertige_seeds, test=test
    )
    grenze = GRENZE_EURO * DOLLAR_JE_EURO

    je_seed_aufgaben = len(aufgaben_im_seed(umfang.seeds[0], umfang))
    text = [
        "# Hauptexperiment (Issue #12): Kosten gegen Plan",
        "",
        f"Plan je Seed: {plan.je_seed:.2f} $ = {plan.durchgaenge_je_seed:.2f} $ fuer die "
        f"{je_seed_aufgaben} Durchgaenge + {plan.testbewertung_je_seed:.2f} $ fuer ihre "
        "Testbewertungen, die erst in der letzten Stufe anfallen (ADR-0027, Punkt 6).",
        "",
        "| Seed | Durchgaenge fertig | wiederholt | ausgegeben $ | Plan $ | Abweichung |",
        "|---|---|---|---|---|---|",
    ]
    for seed in umfang.seeds:
        fertig_im_seed = sum(
            aufgabe in fertige_durchgaenge for aufgabe in aufgaben_im_seed(seed, umfang)
        )
        wiederholt = len(nummern.get(seed, set()) - fertige_nummern)
        ausgegeben = je_seed.get(seed, 0.0)
        abweichung = (ausgegeben - plan.durchgaenge_je_seed) / plan.durchgaenge_je_seed * 100
        text.append(
            f"| {seed} | {fertig_im_seed} von {je_seed_aufgaben} | {wiederholt} "
            f"| {ausgegeben:.2f} | {plan.durchgaenge_je_seed:.2f} | "
            + (f"{abweichung:+.1f} % |" if seed in fertige_seeds else "- |")
        )
    ungueltige_tests = sum(1 for zeile in tests if zeile.get("anbieterfehler"))
    gueltige_tests = len(finde_gueltig_bewertete_paare(zeilen, umfang.testdaten))
    paare = sum(len(durchgang.manifest["paare"]) for durchgang in fertige_durchgaenge.values())
    text += [
        "",
        f"Teststufe: {gueltige_tests} von {paare} gespeicherten Paaren gueltig bewertet "
        f"({ungueltige_tests} Wiederholungen nach Anbieterfehlern), ausgegeben {test:.2f} $, "
        f"Plan {plan.teststufe:.2f} $.",
        "",
        f"Ausgegeben bisher, alle Zeilen des Protokolls: {bisher:.2f} $ (Vorstudie "
        f"{auswertung.ausgegeben:.2f} $, Hauptexperiment {im_experiment:.2f} $, sonst "
        f"{bisher - auswertung.ausgegeben - im_experiment:.2f} $).",
        f"Hochrechnung bis zum Ende: {hochgerechnet:.2f} $ = "
        f"{hochgerechnet / DOLLAR_JE_EURO:.2f} EUR "
        f"(Grenze {GRENZE_EURO} EUR = {grenze:.2f} $ zum Kurs {DOLLAR_JE_EURO} vom 09.10.2026; "
        "massgeblich ist der Tageskurs).",
    ]
    if hochgerechnet > grenze:
        text += [
            "",
            f"**Notbremse:** die Hochrechnung liegt ueber {GRENZE_EURO} EUR. Nach der "
            "Kuerzungsreihenfolge (ADR-0002) kuerzen, nur so weit wie noetig und nur fuer die "
            "offenen Seeds - nie automatisch, Miray entscheidet (ADR-0027, Punkt 5). Dafuer "
            "`KUERZUNG` in `dspy_optimizer/hauptexperiment.py` setzen und committen.",
        ]
    return "\n".join(text)


def main():
    # HotPotQA enthaelt Umlaute und Akzente; die Windows-Konsole kann die sonst nicht ausgeben.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    protokoll = Protokoll()
    argumente = sys.argv[1:]
    if argumente == ["stand"]:
        _drucke_bericht(protokoll)
        return
    if argumente not in ([], ["testbewertung"]):
        sys.exit("Aufruf: python -m dspy_optimizer.hauptexperiment [stand | testbewertung]")

    load_dotenv()
    messhygiene()
    aufteilung = lade_aufteilung()
    auswertung = werte_aus(protokoll.zeilen())
    lehne_start_ohne_voraussetzungen_ab(aufteilung, auswertung)
    ist_teststufe = argumente == ["testbewertung"]
    naht = Optimierungslauf(
        trainingsdaten=aufteilung.trainingsdaten,
        validierungsdaten=aufteilung.validierungsdaten,
        auswahldaten=aufteilung.auswahldaten,
        protokoll=protokoll,
        # Waehrend der Seeds wird die Naht ohne Testdaten gebaut, damit kein Durchgang sie sieht.
        testdaten=aufteilung.testdaten[: UMFANG.testdaten] if ist_teststufe else (),
    )
    experiment = Hauptexperiment(naht, protokoll, ordne_kostenmodelle(auswertung))
    try:
        if ist_teststufe:
            experiment.testbewertung()
            _drucke_rueckweg("nach der Testbewertung")
        elif (seed := experiment()) is not None:
            _drucke_rueckweg(f"nach Seed {seed}")
    finally:
        _drucke_bericht(protokoll)


def _drucke_rueckweg(stufe):
    """Gibt aus, was nach einer Stufe zu versionieren ist. Das Modul committet nicht selbst."""
    print(
        "\nJetzt Protokoll und Programme sichern:\n"
        "  git add ergebnisse/protokoll.jsonl ergebnisse/programme\n"
        f'  git commit -m "Protokoll und Programme {stufe}" -- '
        "ergebnisse/protokoll.jsonl ergebnisse/programme\n"
        "  git push",
        flush=True,
    )


def _drucke_bericht(protokoll):
    """Der Bericht nach dem Lauf; scheitert er, steht das im Log, ohne die Meldung zu
    verdecken, mit der das Modul angehalten hat."""
    zeilen = protokoll.zeilen()
    try:
        print(
            "\n" + fasse_kosten_zusammen(zeilen, werte_aus(zeilen), PROGRAMME, UMFANG), flush=True
        )
    except ValueError as fehler:
        print(f"\nKein Bericht moeglich: {fehler}", flush=True)


if __name__ == "__main__":
    main()
