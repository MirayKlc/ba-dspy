"""Tests zu den Baselines.

Wie beim Verfahren alles gegen die Attrappe des Optimierungslaufs (`tests/attrappe.py`):
kein einziger API-Aufruf. Die Baselines benutzen dieselbe Naht wie das Verfahren und
bekommen dasselbe Optimierungsbudget B, damit die Verteilung der einzige Unterschied ist.
"""

import statistics

import pytest

from dspy_optimizer.baselines import Baselines
from dspy_optimizer.programm import ModellProgrammPaar, MultiHopQA
from dspy_optimizer.protokoll import Einordnung, Protokoll
from dspy_optimizer.verfahren import UngueltigerDurchgang
from tests.attrappe import (
    A,
    B,
    BASIS,
    C,
    D,
    GUENSTIG,
    KOSTENMODELLE,
    KOSTENOBERGRENZE,
    N,
    OPTIMIERUNGSBUDGET,
    TEURER,
    WEIT_HINTEN,
    ZU_TEUER,
    AttrappeLauf,
    Vorgabe,
    kostenmodell,
    schlechter,
    vertauscht,
)


@pytest.fixture
def protokoll(tmp_path):
    return Protokoll(tmp_path / "protokoll.jsonl")


def baselines_mit(lauf, protokoll, kostenmodelle=KOSTENMODELLE):
    return Baselines(
        optimierungslauf=lauf,
        kostenmodelle=kostenmodelle,
        kostenobergrenze=KOSTENOBERGRENZE,
        protokoll=protokoll,
    )


def dicht_beieinander():
    """Keiner ist nachweisbar schlechter als A; B liegt im Mittel vor C, C vor D."""
    return {
        A: Vorgabe(BASIS),
        B: Vorgabe(schlechter(an={5}, um=0.1)),
        C: Vorgabe(schlechter(an={5, 12}, um=0.1)),
        D: Vorgabe(schlechter(an={5}, um=0.3)),
    }


# --- Gleichverteilung -------------------------------------------------------------------


def gleichverteilung(vorgaben, protokoll, *, seed=0, durchgang=1):
    lauf = AttrappeLauf(vorgaben)
    ergebnis = baselines_mit(lauf, protokoll).gleichverteilung(
        MultiHopQA(), OPTIMIERUNGSBUDGET, seed=seed, durchgang=durchgang
    )
    return ergebnis, lauf


def test_die_gleichverteilung_gibt_jedem_kandidatenmodell_ein_viertel_des_budgets(protokoll):
    """Der Hauptvergleich der Arbeit: B zu
    gleichen Dollar-Teilen auf alle vier, in einer einzigen Phase. Jeder Anteil wird ueber das
    Kostenmodell in `num_trials` uebersetzt und bezahlt auch die Nachbewertung."""
    _, lauf = gleichverteilung(dicht_beieinander(), protokoll, seed=9, durchgang=2)

    assert [aufruf.kandidatenmodell for aufruf in lauf.aufrufe] == [A, B, C, D]
    for aufruf, modell in zip(lauf.aufrufe, KOSTENMODELLE):
        assert aufruf.plan.budgetanteil == pytest.approx(OPTIMIERUNGSBUDGET / 4)
        assert aufruf.plan.kostenmodell == modell
        assert aufruf.num_trials == modell.num_trials_fuer(OPTIMIERUNGSBUDGET / 4, N)
        assert aufruf.seed == 9
        assert aufruf.einordnung == Einordnung.experiment("gleichverteilung", durchgang=2)


def test_die_gleichverteilung_waehlt_nach_der_regel_der_endauswahl(protokoll):
    """Erst die Kostenobergrenze, dann die hoechste Bewertung aus der Nachbewertung.
    A ist am besten, reisst aber die Obergrenze - gewonnen hat B, als Modell-Programm-Paar
    mit seinem optimierten Programm."""
    vorgaben = dicht_beieinander()
    vorgaben[A] = Vorgabe(BASIS, ausfuehrungskosten=ZU_TEUER)

    ergebnis, _ = gleichverteilung(vorgaben, protokoll)

    assert ergebnis.gewinner == B
    assert not ergebnis.kein_gewinner
    assert ergebnis.ausgeschieden[A] == "kostenfilter"
    assert isinstance(ergebnis.programm, ModellProgrammPaar)
    assert ergebnis.programm.kandidatenmodell == B
    for _, praediktor in ergebnis.programm.programm.named_predictors():
        assert praediktor.signature.instructions == f"gleichverteilung: {B}"


def test_bei_gleichstand_gewinnen_die_niedrigeren_ausfuehrungskosten(protokoll):
    """Wie in der Endauswahl des Verfahrens: bei gleicher Bewertung das billigere."""
    vorgaben = dicht_beieinander()
    vorgaben[A] = Vorgabe(BASIS, ausfuehrungskosten=TEURER)
    vorgaben[B] = Vorgabe(vertauscht(0, 1), ausfuehrungskosten=GUENSTIG)

    ergebnis, _ = gleichverteilung(vorgaben, protokoll)

    assert ergebnis.gewinner == B


def test_die_gleichverteilung_haelt_alle_vier_programme_fuer_die_testbewertung(protokoll):
    """FF1 vergleicht die Rangfolge aller vier Programme der Gleichverteilung auf den
    Testdaten - nicht nur das des Gewinners. Die Programme tragen kein eigenes LM: so nimmt
    sie `nur_bewerten` entgegen."""
    ergebnis, _ = gleichverteilung(dicht_beieinander(), protokoll)

    assert list(ergebnis.programme) == [A, B, C, D]
    for name, programm in ergebnis.programme.items():
        for _, praediktor in programm.named_predictors():
            assert praediktor.signature.instructions == f"gleichverteilung: {name}"
            assert praediktor.lm is None


def test_die_gleichverteilung_schreibt_ihre_auswahl_als_entscheidungszeile(protokoll):
    """Gewinner und Ausgeschiedene, nachrechenbar, dazu Optimierungsbudget und tatsaechlicher
    Verbrauch des Durchgangs. Die Zeile hat keine Phase - nur das Verfahren hat Phasen."""
    vorgaben = dicht_beieinander()
    vorgaben[A] = Vorgabe(BASIS, ausfuehrungskosten=ZU_TEUER)
    vorgaben[C] = vorgaben[C]._replace(mehrkosten=0.02)

    ergebnis, lauf = gleichverteilung(vorgaben, protokoll, seed=3, durchgang=7)

    (zeile,) = protokoll.zeilen()
    assert zeile["abschnitt"] == "endauswahl"
    assert (zeile["zweck"], zeile["baseline"], zeile["phase"], zeile["durchgang"]) == (
        "experiment",
        "gleichverteilung",
        None,
        7,
    )
    assert zeile["seed"] == 3
    assert zeile["gewinner"] == B
    assert zeile["kein_gewinner"] is False
    assert zeile["ausgeschieden"][A] == "kostenfilter"
    assert zeile["kostenobergrenze"] == KOSTENOBERGRENZE
    assert set(zeile["kandidatenmodelle"]) == {A, B, C, D}
    assert zeile["kandidatenmodelle"][A]["ausfuehrungskosten_je_1000"] == pytest.approx(
        ZU_TEUER * 1000
    )
    geplant = sum(aufruf.plan.dollar_gesamt for aufruf in lauf.aufrufe)
    assert zeile["optimierungsbudget"] == OPTIMIERUNGSBUDGET
    assert zeile["ausgegeben"] == pytest.approx(geplant + 0.02)
    assert zeile["uebrig"] == pytest.approx(OPTIMIERUNGSBUDGET - geplant - 0.02)
    assert ergebnis.budgetverbrauch.ausgegeben == pytest.approx(zeile["ausgegeben"])
    assert ergebnis.budgetverbrauch.uebrig == pytest.approx(zeile["uebrig"])
    assert "dollar" not in zeile


def test_eine_baseline_siebt_nichts_aus_ausser_ueber_die_kostenobergrenze(protokoll):
    """Die Baselines kennen nur Kostenobergrenze und Bewertung - keinen Test und
    kein "hoechstens zwei". In ihrer Zeile darf deshalb kein Ausscheiden durch einen Test
    stehen: eine Auswertung, die zaehlt, wie oft der Test etwas fand (FF2), zaehlte es sonst
    mit."""
    vorgaben = {
        A: Vorgabe(BASIS, ausfuehrungskosten=ZU_TEUER),
        B: Vorgabe(BASIS),
        C: Vorgabe(WEIT_HINTEN),
        D: Vorgabe(schlechter(an={3}, um=0.1)),
    }

    ergebnis, _ = gleichverteilung(vorgaben, protokoll)

    assert ergebnis.gewinner == B
    assert ergebnis.ausgeschieden == {A: "kostenfilter"}
    (zeile,) = protokoll.zeilen()
    assert zeile["ausgeschieden"] == {A: "kostenfilter"}
    for feld in ("weiter", "test_hat_gefunden", "vergleiche", "alpha"):
        assert feld not in zeile
    assert zeile["kandidatenmodelle"][C] == {
        "bewertung": pytest.approx(statistics.fmean(WEIT_HINTEN)),
        "ausfuehrungskosten_je_1000": pytest.approx(GUENSTIG * 1000),
    }


def test_ein_anbieterfehler_bricht_auch_die_gleichverteilung_ab(protokoll):
    """Fuer die Baselines gilt dasselbe wie fuer das Verfahren: keine Auswahl auf dieser
    Grundlage, und keine weiteren bezahlten Laeufe."""
    vorgaben = dicht_beieinander()
    vorgaben[B] = vorgaben[B]._replace(anbieterfehler=1)

    lauf = AttrappeLauf(vorgaben)

    with pytest.raises(UngueltigerDurchgang, match="Anbieterfehler") as abbruch:
        baselines_mit(lauf, protokoll).gleichverteilung(
            MultiHopQA(), OPTIMIERUNGSBUDGET, seed=0, durchgang=4
        )

    assert [aufruf.kandidatenmodell for aufruf in lauf.aufrufe] == [A, B]
    assert B in str(abbruch.value)
    assert "gleichverteilung" in str(abbruch.value)
    assert "Durchgang 4" in str(abbruch.value)
    assert protokoll.zeilen() == []


def test_ohne_gemessene_ausfuehrungskosten_waehlt_die_gleichverteilung_nicht(protokoll):
    vorgaben = dicht_beieinander()
    vorgaben[D] = Vorgabe(BASIS, ausfuehrungskosten=None)

    with pytest.raises(UngueltigerDurchgang, match="Ausfuehrungskosten"):
        gleichverteilung(vorgaben, protokoll)

    assert protokoll.zeilen() == []


def test_reicht_ein_viertel_fuer_keinen_trial_startet_gar_kein_lauf(protokoll):
    """Geplant wird fuer alle vier, bevor der erste Lauf Geld kostet."""
    lauf = AttrappeLauf(dicht_beieinander())
    teuer = KOSTENMODELLE[:3] + (kostenmodell(D, 10.0),)

    with pytest.raises(ValueError, match="keinen einzigen Trial"):
        baselines_mit(lauf, protokoll, kostenmodelle=teuer).gleichverteilung(
            MultiHopQA(), OPTIMIERUNGSBUDGET, seed=0, durchgang=1
        )

    assert lauf.aufrufe == []
    assert protokoll.zeilen() == []


# --- Ein Kandidatenmodell bekommt alles -------------------------------------------------


def ein_modell(name, vorgaben, protokoll, *, seed=0, durchgang=1):
    lauf = AttrappeLauf(vorgaben)
    ergebnis = baselines_mit(lauf, protokoll).ein_modell(
        name, MultiHopQA(), OPTIMIERUNGSBUDGET, seed=seed, durchgang=durchgang
    )
    return ergebnis, lauf


@pytest.mark.parametrize("name", [A, B, C, D])
def test_jedes_kandidatenmodell_bekommt_einmal_das_ganze_budget(name, protokoll):
    """Einmal je Kandidatenmodell, sonst hinge das Ergebnis daran, welches man als Vergleich
    gewaehlt hat. Ein Lauf mit ganz B, ueber das Kostenmodell in `num_trials` uebersetzt, samt
    Nachbewertung."""
    ergebnis, lauf = ein_modell(name, dicht_beieinander(), protokoll, seed=5, durchgang=3)

    (aufruf,) = lauf.aufrufe
    modell = {modell.kandidatenmodell: modell for modell in KOSTENMODELLE}[name]
    assert aufruf.kandidatenmodell == name
    assert aufruf.plan.budgetanteil == pytest.approx(OPTIMIERUNGSBUDGET)
    assert aufruf.plan.kostenmodell == modell
    assert aufruf.num_trials == modell.num_trials_fuer(OPTIMIERUNGSBUDGET, N)
    assert aufruf.seed == 5
    assert aufruf.einordnung == Einordnung.experiment("ein-modell", durchgang=3)
    assert ergebnis.gewinner == name
    assert ergebnis.programm.kandidatenmodell == name
    assert list(ergebnis.programme) == [name]


def test_mit_dem_ganzen_budget_bekommt_ein_modell_mehr_trials_als_mit_einem_viertel(protokoll):
    """Gleiche Dollar, anders verteilt: der einzige Unterschied zur Gleichverteilung."""
    _, viertel = gleichverteilung(dicht_beieinander(), protokoll)
    _, alles = ein_modell(A, dicht_beieinander(), protokoll)

    assert alles.aufrufe[0].num_trials > viertel.aufrufe[0].num_trials


def test_reisst_das_einzige_programm_die_obergrenze_gibt_es_kein_gewinner(protokoll):
    """Das Ergebnis dieser Baseline ist "kein Gewinner", nicht das zu teure
    Programm. Das Programm bleibt fuer seine gekennzeichnete Testbewertung erhalten."""
    vorgaben = dicht_beieinander()
    vorgaben[C] = Vorgabe(BASIS, ausfuehrungskosten=ZU_TEUER)

    ergebnis, _ = ein_modell(C, vorgaben, protokoll, durchgang=2)

    assert ergebnis.kein_gewinner
    assert ergebnis.gewinner is None
    assert ergebnis.programm is None
    assert "Kostenobergrenze" in ergebnis.meldung
    assert C in ergebnis.meldung
    assert list(ergebnis.programme) == [C]
    (zeile,) = protokoll.zeilen()
    assert zeile["abschnitt"] == "endauswahl"
    assert zeile["baseline"] == "ein-modell"
    assert zeile["gewinner"] is None
    assert zeile["kein_gewinner"] is True
    assert zeile["ausgeschieden"] == {C: "kostenfilter"}
    assert list(zeile["kandidatenmodelle"]) == [C]
    assert zeile["optimierungsbudget"] == OPTIMIERUNGSBUDGET


def test_ein_unbekanntes_kandidatenmodell_wird_vor_dem_ersten_lauf_abgelehnt(protokoll):
    lauf = AttrappeLauf(dicht_beieinander())

    with pytest.raises(ValueError, match="kein Kostenmodell"):
        baselines_mit(lauf, protokoll).ein_modell(
            "gibt-es-nicht", MultiHopQA(), OPTIMIERUNGSBUDGET, seed=0, durchgang=1
        )

    assert lauf.aufrufe == []


# --- Ohne Optimierung -------------------------------------------------------------------


def ohne_optimierung(vorgaben, protokoll, *, seed=0, durchgang=1):
    lauf = AttrappeLauf(vorgaben)
    startprogramm = MultiHopQA()
    ergebnis = baselines_mit(lauf, protokoll).ohne_optimierung(
        startprogramm, seed=seed, durchgang=durchgang
    )
    return ergebnis, lauf, startprogramm


def test_ohne_optimierung_wird_jedes_kandidatenmodell_nur_bewertet(protokoll):
    """Ueber den Eingang "nur bewerten", nicht ueber `num_trials = 0` - das zahlte die
    Fixkosten und suchte nicht; so laeuft nur die Fixkosten-Messung der Vorstudie
    (`nur_fixkosten`). Bewertet wird auf den Auswahldaten, als Nachbewertung, wie bei den
    optimierten Programmen."""
    ergebnis, lauf, startprogramm = ohne_optimierung(
        dicht_beieinander(), protokoll, seed=4, durchgang=6
    )

    assert lauf.aufrufe == []
    assert [bewertung.kandidatenmodell for bewertung in lauf.bewertungsaufrufe] == [A, B, C, D]
    for bewertung in lauf.bewertungsaufrufe:
        assert bewertung.programm is startprogramm
        assert bewertung.seed == 4
        assert bewertung.abschnitt == "nachbewertung"
        assert bewertung.einordnung == Einordnung.experiment("ohne-optimierung", durchgang=6)
    assert list(ergebnis.programme) == [A, B, C, D]
    assert all(programm is startprogramm for programm in ergebnis.programme.values())


def test_ohne_optimierung_waehlt_nach_derselben_regel(protokoll):
    """Unter den vier unoptimierten Programmen erst Kostenobergrenze, dann Bewertung. Das Paar
    ist das, was man ohne Verfahren genommen haette (FF4)."""
    vorgaben = dicht_beieinander()
    vorgaben[A] = Vorgabe(BASIS, ausfuehrungskosten=ZU_TEUER)

    ergebnis, _, _ = ohne_optimierung(vorgaben, protokoll)

    assert ergebnis.gewinner == B
    assert ergebnis.ausgeschieden[A] == "kostenfilter"
    assert ergebnis.programm.kandidatenmodell == B


def test_ohne_optimierung_hat_kein_optimierungsbudget(protokoll):
    """Die Baseline ohne Optimierung optimiert nicht und zaehlt nicht zu
    B. Ihre Zeile traegt, was die Bewertungen gekostet haben; Budget und Rest stehen leer
    statt bei einer erfundenen Zahl."""
    vorgaben = dicht_beieinander()
    vorgaben[B] = vorgaben[B]._replace(mehrkosten=0.01)

    ergebnis, _, _ = ohne_optimierung(vorgaben, protokoll, seed=1, durchgang=2)

    (zeile,) = protokoll.zeilen()
    assert zeile["abschnitt"] == "endauswahl"
    assert (zeile["baseline"], zeile["phase"], zeile["durchgang"], zeile["seed"]) == (
        "ohne-optimierung",
        None,
        2,
        1,
    )
    assert zeile["gewinner"] == A
    assert zeile["optimierungsbudget"] is None
    assert zeile["ausgegeben"] == pytest.approx(4 * N * GUENSTIG + 0.01)
    assert zeile["uebrig"] is None
    assert ergebnis.budgetverbrauch.uebrig is None


def test_ein_anbieterfehler_bricht_auch_die_baseline_ohne_optimierung_ab(protokoll):
    vorgaben = dicht_beieinander()
    vorgaben[C] = vorgaben[C]._replace(anbieterfehler=2)

    with pytest.raises(UngueltigerDurchgang, match="Anbieterfehler") as abbruch:
        ohne_optimierung(vorgaben, protokoll, durchgang=5)

    assert "ohne-optimierung" in str(abbruch.value)
    assert protokoll.zeilen() == []


# --- Alle zusammen ----------------------------------------------------------------------


def test_jede_baseline_schreibt_eine_entscheidungszeile_und_die_teure_referenz_laeuft_nicht(
    protokoll,
):
    """Ein Seed ueber alle Baselines, wie im Hauptexperiment: je Durchgang genau eine
    Entscheidungszeile; bei der Einzelmodell-Baseline hat jedes Kandidatenmodell seinen
    eigenen Durchgang. Die teure Referenz ist nicht Teil des Versuchsaufbaus - keine Zeile
    traegt ihren Namen."""
    lauf = AttrappeLauf(dicht_beieinander())
    baselines = baselines_mit(lauf, protokoll)
    programm = MultiHopQA()

    baselines.ohne_optimierung(programm, seed=0, durchgang=1)
    baselines.gleichverteilung(programm, OPTIMIERUNGSBUDGET, seed=0, durchgang=1)
    for durchgang, name in enumerate([A, B, C, D], start=1):
        baselines.ein_modell(name, programm, OPTIMIERUNGSBUDGET, seed=0, durchgang=durchgang)

    zeilen = protokoll.zeilen()
    assert [(zeile["baseline"], zeile["durchgang"]) for zeile in zeilen] == [
        ("ohne-optimierung", 1),
        ("gleichverteilung", 1),
        ("ein-modell", 1),
        ("ein-modell", 2),
        ("ein-modell", 3),
        ("ein-modell", 4),
    ]
    assert {zeile["abschnitt"] for zeile in zeilen} == {"endauswahl"}
    assert "teure-referenz" not in {zeile["baseline"] for zeile in zeilen}
    assert [zeile["gewinner"] for zeile in zeilen[2:]] == [A, B, C, D]
