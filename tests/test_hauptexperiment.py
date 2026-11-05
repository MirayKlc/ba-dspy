"""Tests zum Hauptexperiment.

Geprueft wird der **Ablauf**: welche Durchgaenge in welcher Reihenfolge laufen, wann das Modul
anhaelt, was nach einem Abbruch noch fehlt, welche Programme gespeichert werden und wann die
Testdaten bewertet werden. Dazu die **Rechnung** von Plan und Hochrechnung und die
**Pruefungen vor dem ersten bezahlten Aufruf**.

Alles gegen eine Attrappe der Naht, die keinen API-Aufruf macht, aber echte Zeilen ins
Protokoll schreibt: das Hauptexperiment erkennt fertige Durchgaenge nur am Protokoll und an
den gespeicherten Programmen, und genau so soll es geprueft werden. Verfahren und Baselines
laufen echt.
"""

import json
from typing import NamedTuple

import pytest

from dspy_optimizer import hauptexperiment as modul
from dspy_optimizer.datensatz import Aufteilung
from dspy_optimizer.hauptexperiment import (
    B,
    KOSTENOBERGRENZE,
    Aufgabe,
    Hauptexperiment,
    HauptexperimentAngehalten,
    aufgaben_im_seed,
    fasse_kosten_zusammen,
    finde_fertige_durchgaenge,
    rechne_hoch,
    plane_kosten,
    lade_modell_programm_paar,
    lehne_start_ohne_voraussetzungen_ab,
    umfang_nach_kuerzung,
)
from dspy_optimizer.kandidatenmodelle import DIE_VIER
from dspy_optimizer.kostenerfassung import Messwerte, Tokens
from dspy_optimizer.kostenmodell import Kostenmodell, Parameter
from dspy_optimizer.optimierungslauf import Bewertungsergebnis, Optimierungsergebnis
from dspy_optimizer.programm import lehne_eigenes_lm_ab
from dspy_optimizer.protokoll import Einordnung, Protokoll
from dspy_optimizer.vorstudie import Auswertung, Lauf, Messpunkt, Modellergebnis
from tests.attrappe import BASIS, N, herkunft, optimiert

GPT_OSS, GEMMA, DEEPSEEK, LUNA = (modell.name for modell in DIE_VIER)
TESTBEISPIELE = 50

# Dollar je Programmlauf. Luna reisst die Kostenobergrenze von 0,2895 $ je 1000, die anderen
# halten sie ein; gpt-oss ist das guenstigste.
AUSFUEHRUNGSKOSTEN = {GPT_OSS: 0.00016, GEMMA: 0.00028, DEEPSEEK: 0.00022, LUNA: 0.0009}
ZU_TEUER = 0.0009

# Die feste Reihenfolge der Durchgaenge innerhalb eines Seeds.
EIN_SEED = [
    ("ohne-optimierung", None),
    ("verfahren", None),
    ("gleichverteilung", None),
    ("ein-modell", GPT_OSS),
    ("ein-modell", GEMMA),
    ("ein-modell", DEEPSEEK),
    ("ein-modell", LUNA),
]


def kostenmodell(name):
    return Kostenmodell(
        kandidatenmodell=name,
        fixkosten=Parameter(0.04, "erfunden"),
        kosten_je_trial=Parameter(0.02, "erfunden"),
        ausfuehrungskosten=Parameter(AUSFUEHRUNGSKOSTEN[name], "erfunden"),
    )


KOSTENMODELLE = tuple(kostenmodell(modell.name) for modell in DIE_VIER)


def messwerte(dollar, programmlaeufe, anbieterfehler=0):
    return Messwerte(
        dollar=dollar,
        lm_aufrufe=2 * programmlaeufe,
        programmlaeufe=programmlaeufe,
        fehlgeschlagene_programmlaeufe=anbieterfehler,
        auswertungen=1,
        tokens=Tokens(100, 10),
        tokens_aus_evaluationen=Tokens(100, 10),
        anbieter={},
        stoerungen=(),
        dauer_sekunden=60.0,
        latenzen=(1.0,),
        anbieterfehler=anbieterfehler,
    )


class Aufruf(NamedTuple):
    art: str
    kandidatenmodell: str
    einordnung: object
    seed: int
    abschnitt: str | None
    zusaetzliche_konfiguration: dict | None


class AttrappeNaht:
    """Die Naht ohne API: schreibt die Zeilen eines Optimierungslaufs (Optimierung und
    Nachbewertung) bzw. die eine Zeile von `nur_bewerten` ins Protokoll und gibt Zahlen zurueck.

    Ausgegeben wird genau der Plan. Alle bewerten gleich (`BASIS`), die Ausfuehrungskosten
    stehen in `AUSFUEHRUNGSKOSTEN`; `fokuskosten` ueberschreibt sie in der Fokusphase.
    `stoerung` bekommt jeden `Aufruf` und sagt `"anbieterfehler"`, `"absturz"` oder `None`.
    `mehrkosten` gibt jeder Optimierung der Sondierung so viele Dollar ueber dem Plan.
    """

    def __init__(self, protokoll, stoerung=None, fokuskosten=None, mehrkosten=0.0):
        self.protokoll = protokoll
        self.stoerung = stoerung or (lambda aufruf: None)
        self.fokuskosten = fokuskosten or {}
        self.mehrkosten = mehrkosten
        self.auswahldaten = list(range(N))
        self.testdaten = list(range(TESTBEISPIELE))
        self.aufrufe = []
        self.programme = []

    def _kosten(self, name, einordnung):
        if einordnung.phase == "fokus":
            return self.fokuskosten.get(name, AUSFUEHRUNGSKOSTEN[name])
        return AUSFUEHRUNGSKOSTEN[name]

    def _stoere(self, aufruf):
        self.aufrufe.append(aufruf)
        stoerung = self.stoerung(aufruf)
        if stoerung == "absturz":
            raise RuntimeError("Absturz mitten im Lauf")
        return 1 if stoerung == "anbieterfehler" else 0

    def __call__(
        self, kandidatenmodell, startprogramm, num_trials, seed, *, einordnung, plan=None
    ):
        name = kandidatenmodell.name
        fehler = self._stoere(Aufruf("optimierung", name, einordnung, seed, None, None))
        konfiguration = {"kandidatenmodell": name, "num_trials": num_trials}
        optimierungskosten = plan.dollar_optimierung + (
            self.mehrkosten if einordnung.phase == "sondierung" else 0.0
        )
        self.protokoll.schreibe(
            abschnitt="optimierung",
            einordnung=einordnung,
            konfiguration=konfiguration,
            seed=seed,
            bewertung=0.8,
            messwerte=messwerte(optimierungskosten, 35),
        )
        kosten = self._kosten(name, einordnung)
        self.protokoll.schreibe(
            abschnitt="nachbewertung",
            einordnung=einordnung,
            konfiguration=konfiguration,
            seed=seed,
            bewertung=0.8,
            messwerte=messwerte(N * kosten, N, anbieterfehler=fehler),
        )
        return Optimierungsergebnis(
            programm=optimiert(startprogramm, f"{herkunft(einordnung)}: {name}"),
            bewertung=0.8,
            bewertungen_je_beispiel=BASIS,
            optimierungskosten=optimierungskosten,
            nachbewertungskosten=N * kosten,
            ausfuehrungskosten=kosten,
            programmlaeufe=35,
            formatfehler=0,
            anbieterfehler=fehler,
        )

    def nur_bewerten(
        self,
        kandidatenmodell,
        programm,
        seed,
        *,
        einordnung,
        abschnitt,
        zusaetzliche_konfiguration=None,
    ):
        name = kandidatenmodell.name
        lehne_eigenes_lm_ab(programm)
        self.programme.append(programm)
        fehler = self._stoere(
            Aufruf("bewerten", name, einordnung, seed, abschnitt, zusaetzliche_konfiguration)
        )
        daten = self.testdaten if abschnitt == "testbewertung" else self.auswahldaten
        kosten = AUSFUEHRUNGSKOSTEN[name]
        teil = "testdaten" if abschnitt == "testbewertung" else "auswahldaten"
        self.protokoll.schreibe(
            abschnitt=abschnitt,
            einordnung=einordnung,
            konfiguration={
                "kandidatenmodell": name,
                teil: len(daten),
                **(zusaetzliche_konfiguration or {}),
            },
            seed=seed,
            bewertung=0.7,
            messwerte=messwerte(len(daten) * kosten, len(daten), anbieterfehler=fehler),
        )
        return Bewertungsergebnis(
            bewertung=0.7,
            bewertungen_je_beispiel=BASIS,
            bewertungskosten=len(daten) * kosten,
            ausfuehrungskosten=kosten,
            formatfehler=0,
            anbieterfehler=fehler,
        )

    def durchgaenge(self):
        """Je Aufruf `(baseline, durchgang)`, ohne Wiederholungen, in der Reihenfolge."""
        gesehen = []
        for aufruf in self.aufrufe:
            schluessel = (aufruf.einordnung.baseline, aufruf.einordnung.durchgang)
            if schluessel not in gesehen:
                gesehen.append(schluessel)
        return gesehen

    def testbewertungen(self):
        return [aufruf for aufruf in self.aufrufe if aufruf.abschnitt == "testbewertung"]


@pytest.fixture
def protokoll(tmp_path):
    return Protokoll(tmp_path / "protokoll.jsonl")


@pytest.fixture
def programme(tmp_path):
    return tmp_path / "programme"


def experiment(naht, protokoll, programme, kuerzung=0):
    return Hauptexperiment(
        naht, protokoll, KOSTENMODELLE, programme=programme, umfang=umfang_nach_kuerzung(kuerzung)
    )


def fahre(protokoll, programme, seeds=1, **naht):
    """`seeds` Seeds nacheinander, je mit eigenem Neustart, mit einer frischen Attrappe."""
    lauf = AttrappeNaht(protokoll, **naht)
    for _ in range(seeds):
        experiment(lauf, protokoll, programme)()
    return lauf


def durchgaenge_im_protokoll(protokoll):
    return sorted({zeile["durchgang"] for zeile in protokoll.zeilen() if zeile.get("durchgang")})


# --- Ablauf eines Seeds -----------------------------------------------------------------


def test_ein_seed_faehrt_die_sieben_durchgaenge_in_fester_reihenfolge(protokoll, programme):
    """Ohne Optimierung, Verfahren, Gleichverteilung, Ein-Modell in der Reihenfolge von
    `DIE_VIER` - jeder Durchgang mit eigener, fortlaufender Nummer."""
    lauf = fahre(protokoll, programme)

    assert lauf.durchgaenge() == [
        ("ohne-optimierung", 1),
        ("verfahren", 2),
        ("gleichverteilung", 3),
        ("ein-modell", 4),
        ("ein-modell", 5),
        ("ein-modell", 6),
        ("ein-modell", 7),
    ]
    ein_modell = [
        aufruf.kandidatenmodell
        for aufruf in lauf.aufrufe
        if aufruf.einordnung.baseline == "ein-modell"
    ]
    assert ein_modell == [GPT_OSS, GEMMA, DEEPSEEK, LUNA]
    assert {aufruf.seed for aufruf in lauf.aufrufe} == {1}


def test_nach_einem_seed_haelt_das_modul_an_der_naechste_braucht_einen_neustart(
    protokoll, programme
):
    """Ein Aufruf faehrt genau einen Seed; erst der Neustart den naechsten, mit den naechsten
    Nummern."""
    lauf = AttrappeNaht(protokoll)
    experiment(lauf, protokoll, programme)()
    erster = len(lauf.aufrufe)

    experiment(lauf, protokoll, programme)()

    zweiter = lauf.aufrufe[erster:]
    assert {aufruf.seed for aufruf in zweiter} == {2}
    assert {aufruf.einordnung.durchgang for aufruf in zweiter} == set(range(8, 15))


def test_sind_alle_seeds_fertig_misst_ein_neustart_nichts_mehr(protokoll, programme):
    lauf = fahre(protokoll, programme, seeds=3)
    vorher = len(lauf.aufrufe)

    experiment(lauf, protokoll, programme)()

    assert len(lauf.aufrufe) == vorher
    assert {aufruf.seed for aufruf in lauf.aufrufe} == {1, 2, 3}


def test_waehrend_der_seeds_wird_keine_einzige_testfrage_bewertet(protokoll, programme):
    """Die Testbewertung ist blind: bis alle Seeds fertig sind, gibt es keine Testzahl."""
    lauf = fahre(protokoll, programme, seeds=3)

    assert lauf.testbewertungen() == []
    assert all(zeile["abschnitt"] != "testbewertung" for zeile in protokoll.zeilen())


# --- Gespeicherte Programme ---------------------------------------------------------------


def manifest(programme, nummer):
    return json.loads((programme / f"durchgang-{nummer:03d}" / "durchgang.json").read_text())


def test_jeder_durchgang_speichert_die_programme_die_eine_testbewertung_bekommen(
    protokoll, programme
):
    """Alle vier ohne Optimierung, das Ergebnis des Verfahrens, alle vier der
    Gleichverteilung und je eines der Einzelmodelle."""
    fahre(protokoll, programme)

    paare = {nummer: manifest(programme, nummer)["paare"] for nummer in range(1, 8)}
    assert [paar["kandidatenmodell"] for paar in paare[1]] == [GPT_OSS, GEMMA, DEEPSEEK, LUNA]
    assert [(paar["kandidatenmodell"], paar["rolle"], paar["phase"]) for paar in paare[2]] == [
        (GPT_OSS, "gewinner", "fokus")
    ]
    assert [paar["kandidatenmodell"] for paar in paare[3]] == [GPT_OSS, GEMMA, DEEPSEEK, LUNA]
    assert [[paar["kandidatenmodell"] for paar in paare[n]] for n in range(4, 8)] == [
        [GPT_OSS],
        [GEMMA],
        [DEEPSEEK],
        [LUNA],
    ]
    # Luna reisst die Kostenobergrenze: kein Gewinner, aber trotzdem gespeichert.
    assert manifest(programme, 7)["kein_gewinner"] is True
    assert paare[7][0]["rolle"] == "teilnehmer"
    assert manifest(programme, 4)["kein_gewinner"] is False


def test_ein_gespeichertes_paar_ist_das_programm_des_durchgangs_auf_seinem_modell(
    protokoll, programme
):
    """Speichern und Laden ergeben dasselbe Programm."""
    fahre(protokoll, programme)

    (paar,) = manifest(programme, 2)["paare"]
    geladen = lade_modell_programm_paar(programme, paar["datei"])

    assert geladen.kandidatenmodell == GPT_OSS
    for _, praediktor in geladen.programm.named_predictors():
        assert praediktor.signature.instructions == f"fokus: {GPT_OSS}"
        assert len(praediktor.demos) == 1


def test_kein_gespeichertes_programm_enthaelt_den_api_schluessel(
    protokoll, programme, monkeypatch
):
    """Die Programme werden versioniert; der Schluessel bleibt in `.env`."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-nur-zum-testen")
    fahre(protokoll, programme)

    dateien = list(programme.rglob("*.json"))
    assert len(dateien) == 7 + 4 + 1 + 4 + 4
    assert all("sk-or-nur-zum-testen" not in datei.read_text() for datei in dateien)


def ohne_gewinner_nach_der_fokusphase():
    """In der Fokusphase reissen beide Ueberlebenden die Kostenobergrenze."""
    return {GPT_OSS: ZU_TEUER, DEEPSEEK: ZU_TEUER}


def test_verfahren_ohne_gewinner_speichert_den_fuehrenden_der_fokusphase_und_die_zusatzangabe(
    protokoll, programme
):
    """Beide gekennzeichnet als "kein Gewinner"."""
    fahre(protokoll, programme, fokuskosten=ohne_gewinner_nach_der_fokusphase())

    eintrag = manifest(programme, 2)
    assert eintrag["kein_gewinner"] is True
    gespeichert = [
        (paar["kandidatenmodell"], paar["rolle"], paar["phase"]) for paar in eintrag["paare"]
    ]
    assert gespeichert == [
        (GPT_OSS, "fuehrender-ohne-gewinner", "fokus"),
        (GPT_OSS, "zusatzangabe", "sondierung"),
    ]
    fuehrender, zusatz = (
        lade_modell_programm_paar(programme, paar["datei"]) for paar in eintrag["paare"]
    )
    for _, praediktor in fuehrender.programm.named_predictors():
        assert praediktor.signature.instructions == f"fokus: {GPT_OSS}"
    for _, praediktor in zusatz.programm.named_predictors():
        assert praediktor.signature.instructions == f"sondierung: {GPT_OSS}"


def test_endet_das_verfahren_nach_der_sondierung_ist_der_durchgang_fertig_ohne_programm(
    protokoll, programme, monkeypatch
):
    """Niemand haelt die Obergrenze: keine Testbewertung, aber der Durchgang ist fertig."""
    monkeypatch.setattr(modul, "KOSTENOBERGRENZE", 0.1)
    lauf = AttrappeNaht(protokoll)
    experiment(lauf, protokoll, programme)()

    assert manifest(programme, 2)["paare"] == []
    assert manifest(programme, 2)["kein_gewinner"] is True
    assert Aufgabe(1, "verfahren") in finde_fertige_durchgaenge(protokoll.zeilen(), programme)


# --- Anhalten und Fortsetzen --------------------------------------------------------------


def anbieterfehler_bei(baseline, kandidatenmodell, wie_oft=1):
    """Eine Stoerung: die ersten `wie_oft` Laeufe von `kandidatenmodell` in `baseline` melden
    einen Anbieterfehler."""
    gezaehlt = []

    def stoerung(aufruf):
        if (aufruf.einordnung.baseline, aufruf.kandidatenmodell) == (baseline, kandidatenmodell):
            gezaehlt.append(aufruf)
            if len(gezaehlt) <= wie_oft:
                return "anbieterfehler"
        return None

    return stoerung


def test_ein_anbieterfehler_haelt_an_ein_neustart_wiederholt_nur_diesen_durchgang(
    protokoll, programme
):
    """Angehalten statt wiederholt; der Neustart faehrt den Durchgang mit neuer Nummer, die
    fertigen davor nicht noch einmal. Die Zeilen des ungueltigen Durchgangs bleiben stehen,
    weil das Protokoll nur angehaengt wird."""
    lauf = AttrappeNaht(protokoll, stoerung=anbieterfehler_bei("gleichverteilung", GEMMA))

    with pytest.raises(HauptexperimentAngehalten, match="Anbieterfehler"):
        experiment(lauf, protokoll, programme)()
    angehalten = len(lauf.aufrufe)
    experiment(lauf, protokoll, programme)()

    assert lauf.durchgaenge()[:4] == [
        ("ohne-optimierung", 1),
        ("verfahren", 2),
        ("gleichverteilung", 3),
        ("gleichverteilung", 4),
    ]
    assert lauf.aufrufe[angehalten].einordnung.durchgang == 4
    assert durchgaenge_im_protokoll(protokoll) == list(range(1, 9))
    assert not (programme / "durchgang-003").exists()
    fertig = finde_fertige_durchgaenge(protokoll.zeilen(), programme)
    assert fertig[Aufgabe(1, "gleichverteilung")].nummer == 4
    assert len(fertig) == 7


def test_ein_absturz_mitten_im_durchgang_kostet_nur_diesen_durchgang(protokoll, programme):
    abgestuerzt = []

    def stoerung(aufruf):
        if aufruf.einordnung.baseline == "verfahren" and aufruf.einordnung.phase == "fokus":
            if not abgestuerzt:
                abgestuerzt.append(aufruf)
                return "absturz"
        return None

    lauf = AttrappeNaht(protokoll, stoerung=stoerung)
    with pytest.raises(RuntimeError, match="Absturz"):
        experiment(lauf, protokoll, programme)()
    experiment(lauf, protokoll, programme)()

    assert lauf.durchgaenge()[:3] == [("ohne-optimierung", 1), ("verfahren", 2), ("verfahren", 3)]
    assert (
        finde_fertige_durchgaenge(protokoll.zeilen(), programme)[Aufgabe(1, "verfahren")].nummer
        == 3
    )


def test_fehlen_die_gespeicherten_programme_wird_der_durchgang_wiederholt(protokoll, programme):
    """Fertig ist ein Durchgang nur mit Entscheidungszeile **und** gespeicherten Programmen."""
    lauf = fahre(protokoll, programme)
    (programme / "durchgang-003" / "durchgang.json").unlink()

    assert Aufgabe(1, "gleichverteilung") not in finde_fertige_durchgaenge(
        protokoll.zeilen(), programme
    )
    experiment(lauf, protokoll, programme)()

    assert lauf.durchgaenge()[7] == ("gleichverteilung", 8)
    assert {aufruf.seed for aufruf in lauf.aufrufe} == {1}


def test_reicht_das_restbudget_der_fokusphase_fuer_keinen_trial_haelt_das_modul_an(
    protokoll, programme, monkeypatch
):
    """Das Modul haelt an, statt selbst zu entscheiden. Hier hat die Sondierung ihr geplantes
    Geld mehr als verdoppelt."""
    lauf = AttrappeNaht(protokoll, mehrkosten=B / 8)

    with pytest.raises(HauptexperimentAngehalten, match="keinen einzigen Trial"):
        experiment(lauf, protokoll, programme)()

    assert lauf.durchgaenge() == [("ohne-optimierung", 1), ("verfahren", 2)]


# --- Testbewertung ------------------------------------------------------------------------


def test_vor_dem_ende_aller_seeds_verweigert_die_testbewertung_jede_messung(protokoll, programme):
    lauf = fahre(protokoll, programme, seeds=2)
    vorher = len(lauf.aufrufe)

    with pytest.raises(HauptexperimentAngehalten, match="Seed 3"):
        experiment(lauf, protokoll, programme).testbewertung()

    assert len(lauf.aufrufe) == vorher


def gueltige_testbewertungen(protokoll):
    return [
        zeile["konfiguration"]["paar"]
        for zeile in protokoll.zeilen()
        if zeile["abschnitt"] == "testbewertung" and not zeile["anbieterfehler"]
    ]


def test_die_testbewertung_bewertet_jedes_gespeicherte_paar_genau_einmal(protokoll, programme):
    """Genau eine gueltige Testbewertung je gespeichertem Paar; ein Neustart bewertet nichts
    doppelt."""
    lauf = fahre(protokoll, programme, seeds=3)
    hauptexperiment = experiment(lauf, protokoll, programme)

    hauptexperiment.testbewertung()
    hauptexperiment.testbewertung()

    bewertet = gueltige_testbewertungen(protokoll)
    gespeichert = [
        paar["datei"] for nummer in range(1, 22) for paar in manifest(programme, nummer)["paare"]
    ]
    assert sorted(bewertet) == sorted(gespeichert)
    assert len(bewertet) == 3 * (4 + 1 + 4 + 4)
    assert len(lauf.testbewertungen()) == len(bewertet)
    for programm in lauf.programme:
        lehne_eigenes_lm_ab(programm)


def test_eine_testbewertungszeile_nennt_paar_durchgang_seed_und_kennzeichnung(
    protokoll, programme
):
    lauf = fahre(protokoll, programme, seeds=3, fokuskosten=ohne_gewinner_nach_der_fokusphase())
    experiment(lauf, protokoll, programme).testbewertung()

    zeilen = [
        zeile
        for zeile in protokoll.zeilen()
        if zeile["abschnitt"] == "testbewertung" and zeile["durchgang"] == 2
    ]
    assert [
        (
            zeile["baseline"],
            zeile["phase"],
            zeile["seed"],
            zeile["konfiguration"]["rolle"],
            zeile["konfiguration"]["kein_gewinner"],
        )
        for zeile in zeilen
    ] == [
        ("verfahren", "fokus", 1, "fuehrender-ohne-gewinner", True),
        ("verfahren", "sondierung", 1, "zusatzangabe", True),
    ]
    assert all(zeile["konfiguration"]["testdaten"] == TESTBEISPIELE for zeile in zeilen)


def test_ein_anbieterfehler_in_der_testbewertung_wird_einmal_sofort_wiederholt(
    protokoll, programme
):
    """Die ungueltige Zeile bleibt stehen, die Wiederholung zaehlt."""
    lauf = fahre(protokoll, programme, seeds=3)
    lauf.stoerung = lambda aufruf: (
        "anbieterfehler"
        if aufruf.abschnitt == "testbewertung"
        and aufruf.einordnung.durchgang == 5
        and len([a for a in lauf.testbewertungen() if a.einordnung.durchgang == 5]) == 1
        else None
    )

    experiment(lauf, protokoll, programme).testbewertung()

    zeilen = [
        zeile
        for zeile in protokoll.zeilen()
        if zeile["abschnitt"] == "testbewertung" and zeile["durchgang"] == 5
    ]
    assert [zeile["anbieterfehler"] for zeile in zeilen] == [1, 0]
    assert len(gueltige_testbewertungen(protokoll)) == 39


def test_beim_zweiten_anbieterfehler_haelt_die_testbewertung_an(protokoll, programme):
    lauf = fahre(protokoll, programme, seeds=3)
    lauf.stoerung = lambda aufruf: (
        "anbieterfehler"
        if aufruf.abschnitt == "testbewertung" and aufruf.einordnung.durchgang == 5
        else None
    )

    with pytest.raises(HauptexperimentAngehalten, match="zweimal"):
        experiment(lauf, protokoll, programme).testbewertung()

    in_durchgang_5 = [a for a in lauf.testbewertungen() if a.einordnung.durchgang == 5]
    assert len(in_durchgang_5) == 2
    # Nichts nach dem angehaltenen Paar.
    assert lauf.testbewertungen()[-1].einordnung.durchgang == 5

    lauf.stoerung = lambda aufruf: None
    experiment(lauf, protokoll, programme).testbewertung()
    assert len(gueltige_testbewertungen(protokoll)) == 39


# --- Kuerzung -----------------------------------------------------------------------------


def test_die_kuerzungsreihenfolge(protokoll):
    """Erst die Einzelmodelle von vier auf zwei (DeepSeek und Gemma), dann die Testdaten auf
    die ersten 300, zuletzt die Seeds auf zwei."""
    assert umfang_nach_kuerzung(0) == ((1, 2, 3), (GPT_OSS, GEMMA, DEEPSEEK, LUNA), 500)
    assert umfang_nach_kuerzung(1) == ((1, 2, 3), (GEMMA, DEEPSEEK), 500)
    assert umfang_nach_kuerzung(2) == ((1, 2, 3), (GEMMA, DEEPSEEK), 300)
    assert umfang_nach_kuerzung(3) == ((1, 2), (GEMMA, DEEPSEEK), 300)
    with pytest.raises(ValueError):
        umfang_nach_kuerzung(4)
    assert aufgaben_im_seed(1, umfang_nach_kuerzung(1))[3:] == [
        Aufgabe(1, "ein-modell", GEMMA),
        Aufgabe(1, "ein-modell", DEEPSEEK),
    ]


# --- Plan und Hochrechnung ----------------------------------------------------------------

OHNE = {GPT_OSS: 0.000112, GEMMA: 0.000207, DEEPSEEK: 0.000219, LUNA: 0.000287}
OPTIMIERT = {GPT_OSS: 0.000163, GEMMA: 0.0002895, DEEPSEEK: 0.000227, LUNA: 0.000898}


def auswertung(kostenobergrenze_von_gemma=0.0002895, offen=(), ausgegeben=11.5):
    ergebnisse = {}
    for modell in reversed(DIE_VIER):
        optimiert_kosten = OPTIMIERT[modell.name]
        if modell.name == GEMMA:
            optimiert_kosten = kostenobergrenze_von_gemma
        ergebnisse[modell.name] = Modellergebnis(
            kandidatenmodell=modell.name,
            ohne_optimierung=Lauf(
                Messpunkt(modell.name, None), None, {"dollar_je_programmlauf": OHNE[modell.name]}
            ),
            optimiert=(),
            fixkosten_messungen=(),
            kostenmodell=Kostenmodell(
                modell.name,
                Parameter(0.04, "erfunden"),
                Parameter(0.02, "erfunden"),
                Parameter(optimiert_kosten, "erfunden"),
            ),
        )
    return Auswertung(ergebnisse, tuple(offen), ausgegeben, 0, "erfunden")


def test_der_plan_je_seed_ist_der_plan_der_vorstudie_geteilt_durch_die_seeds():
    """Der Plan aus `docs/vorstudie-ergebnisse.md` (rund 38,60 $ je Seed), getrennt in die
    Durchgaenge und ihre Testbewertungen, die erst in der letzten Stufe anfallen."""
    plan = plane_kosten(auswertung(), umfang_nach_kuerzung(0))

    assert plan.je_seed == pytest.approx(auswertung().hauptexperiment(B) / 3)
    assert plan.durchgaenge_je_seed == pytest.approx(6 * B + 200 * sum(OHNE.values()))
    assert plan.teststufe == pytest.approx(3 * plan.testbewertung_je_seed)
    assert plan.gesamt == pytest.approx(auswertung().hauptexperiment(B))


def test_der_plan_folgt_der_kuerzung():
    plan = plane_kosten(auswertung(), umfang_nach_kuerzung(3))

    assert plan.durchgaenge_je_seed == pytest.approx(4 * B + 200 * sum(OHNE.values()))
    assert plan.testbewertung_je_seed == pytest.approx(
        300
        * (
            sum(OHNE.values())
            + sum(OPTIMIERT.values())
            + OPTIMIERT[GEMMA]
            + OPTIMIERT[DEEPSEEK]
            + max(OPTIMIERT.values())
        )
    )
    assert plan.seeds == (1, 2)


def test_die_hochrechnung_nimmt_fuer_offene_seeds_den_groesseren_wert_von_plan_und_mittel():
    """Bisher ausgegeben + offene Seeds x max(Plan, Mittel je Seed), dazu der Plan der
    Teststufe."""
    plan = plane_kosten(auswertung(), umfang_nach_kuerzung(0))

    unter_plan = rechne_hoch(plan, bisher=60.0, je_seed={1: 30.0}, fertige_seeds=(1,), test=0.0)
    ueber_plan = rechne_hoch(plan, bisher=80.0, je_seed={1: 50.0}, fertige_seeds=(1,), test=0.0)

    assert unter_plan == pytest.approx(60.0 + 2 * plan.durchgaenge_je_seed + plan.teststufe)
    assert ueber_plan == pytest.approx(80.0 + 2 * 50.0 + plan.teststufe)


def test_ein_angefangener_seed_zaehlt_nur_mit_dem_was_noch_fehlt():
    plan = plane_kosten(auswertung(), umfang_nach_kuerzung(0))

    wert = rechne_hoch(plan, bisher=20.0, je_seed={1: 10.0}, fertige_seeds=(), test=0.0)

    assert wert == pytest.approx(
        20.0 + (plan.durchgaenge_je_seed - 10.0) + 2 * plan.durchgaenge_je_seed + plan.teststufe
    )


def test_der_bericht_nennt_kosten_gegen_plan_und_die_hochrechnung(protokoll, programme):
    fahre(protokoll, programme)

    text = fasse_kosten_zusammen(
        protokoll.zeilen(), auswertung(), programme, umfang_nach_kuerzung(0)
    )

    assert "| 1 | 7 von 7 | 0 |" in text
    assert "| 2 | 0 von 7 | 0 |" in text
    assert "Hochrechnung" in text
    assert "Notbremse" not in text


def test_ueber_200_euro_nennt_der_bericht_die_notbremse(protokoll, programme):
    """Die Notbremse wird nur genannt, das Modul kuerzt nie selbst. Seed 1 hat das Doppelte
    des Plans gekostet: hochgerechnet wird mit dem tatsaechlichen Mittel."""
    fahre(protokoll, programme)
    protokoll.schreibe(
        abschnitt="nachbewertung",
        einordnung=Einordnung.experiment("gleichverteilung", durchgang=99),
        konfiguration={},
        seed=1,
        bewertung=None,
        messwerte=messwerte(70.0, 0),
    )

    text = fasse_kosten_zusammen(
        protokoll.zeilen(), auswertung(), programme, umfang_nach_kuerzung(0)
    )

    assert "Notbremse" in text
    assert "Kuerzungsreihenfolge" in text


# --- Pruefungen vor dem ersten bezahlten Aufruf -------------------------------------------


class Beispiel(NamedTuple):
    id: str


def aufteilung_mit(abdruecke):
    """Eine Aufteilung, deren Teile die gegebenen Fingerabdruecke haetten: die Pruefung wird
    hier ueber `berechne_fingerabdruck` gesteuert."""
    return Aufteilung(*([Beispiel(abdruck)] for abdruck in abdruecke))


@pytest.fixture
def abdruck_ist_kennung(monkeypatch):
    monkeypatch.setattr(modul, "berechne_fingerabdruck", lambda teil: teil[0].id)


def test_die_richtige_aufteilung_und_eine_vollstaendige_vorstudie_kommen_durch(
    abdruck_ist_kennung,
):
    lehne_start_ohne_voraussetzungen_ab(aufteilung_mit(modul.FINGERABDRUECKE), auswertung())


def test_eine_andere_aufteilung_bricht_ab(abdruck_ist_kennung):
    abdruecke = modul.FINGERABDRUECKE._replace(testdaten="000000000000")

    with pytest.raises(HauptexperimentAngehalten, match="Testdaten"):
        lehne_start_ohne_voraussetzungen_ab(aufteilung_mit(abdruecke), auswertung())


def test_eine_unvollstaendige_vorstudie_bricht_ab(abdruck_ist_kennung):
    with pytest.raises(HauptexperimentAngehalten, match="Vorstudie"):
        lehne_start_ohne_voraussetzungen_ab(
            aufteilung_mit(modul.FINGERABDRUECKE), auswertung(offen=(Messpunkt(LUNA, 10),))
        )


def test_eine_andere_kostenobergrenze_bricht_ab(abdruck_ist_kennung):
    with pytest.raises(HauptexperimentAngehalten, match="Kostenobergrenze"):
        lehne_start_ohne_voraussetzungen_ab(
            aufteilung_mit(modul.FINGERABDRUECKE), auswertung(kostenobergrenze_von_gemma=0.0003)
        )


def test_die_festgeschriebenen_werte():
    """Budget B und Kostenobergrenze, wie vor dem Hauptexperiment festgeschrieben."""
    assert B == 6.0
    assert KOSTENOBERGRENZE == 0.2895
    assert modul.UMFANG == umfang_nach_kuerzung(0)
