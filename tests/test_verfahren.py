"""Tests zum Verfahren: Sondierungsphase und Zwischenbewertung, Fokusphase, Endauswahl und
Rueckgabe.

Alles laeuft gegen eine **Attrappe des Optimierungslaufs** (`tests/attrappe.py`), die
vorgegebene Zahlen zurueckgibt - kein einziger API-Aufruf. Das ist der Grund fuer genau diese
Naht: die Logik des Verfahrens haengt nur an dem, was ein Optimierungslauf zurueckgibt, nicht
daran, wie er es misst.
"""

import math
import statistics

import dspy
import pytest

from dspy_optimizer import verfahren as modul
from dspy_optimizer.programm import ModellProgrammPaar, MultiHopQA, lehne_eigenes_lm_ab
from dspy_optimizer.protokoll import Einordnung, Protokoll
from dspy_optimizer.verfahren import (
    Stand,
    UngueltigerDurchgang,
    Verfahren,
    kostenfilter,
    p_wert,
    rangfolge,
    zwischenbewertung,
)
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


def verfahren_mit(lauf, protokoll, kostenmodelle=KOSTENMODELLE):
    return Verfahren(
        optimierungslauf=lauf,
        kostenmodelle=kostenmodelle,
        kostenobergrenze=KOSTENOBERGRENZE,
        protokoll=protokoll,
    )


def sondiere(vorgaben, protokoll, *, seed=0, durchgang=1):
    lauf = AttrappeLauf(vorgaben)
    zwischenstand = verfahren_mit(lauf, protokoll).sondiere(
        MultiHopQA(), OPTIMIERUNGSBUDGET, seed=seed, durchgang=durchgang
    )
    return zwischenstand, lauf


def dicht_beieinander():
    """Keiner ist nachweisbar schlechter als A; B liegt im Mittel vor C, C vor D."""
    return {
        A: Vorgabe(BASIS),
        B: Vorgabe(schlechter(an={5}, um=0.1)),
        C: Vorgabe(schlechter(an={5, 12}, um=0.1)),
        D: Vorgabe(schlechter(an={5}, um=0.3)),
    }


# --- Sondierungsphase ------------------------------------------------------------------


def test_jedes_kandidatenmodell_bekommt_ein_achtel_des_optimierungsbudgets(protokoll):
    """Die Sondierung bekommt die Haelfte von B, zu gleichen Dollar-Teilen."""
    _, lauf = sondiere(dicht_beieinander(), protokoll)

    assert [aufruf.kandidatenmodell for aufruf in lauf.aufrufe] == [A, B, C, D]
    for aufruf in lauf.aufrufe:
        assert aufruf.plan.budgetanteil == pytest.approx(OPTIMIERUNGSBUDGET / 8)


def test_der_budgetanteil_wird_ueber_das_kostenmodell_in_num_trials_uebersetzt(protokoll):
    """Gleiche Dollar heisst gerade nicht gleich viel Suche: wessen Trial billiger ist,
    bekommt mehr davon. Jeder Anteil bezahlt auch die Nachbewertung."""
    _, lauf = sondiere(dicht_beieinander(), protokoll)

    for aufruf, modell in zip(lauf.aufrufe, KOSTENMODELLE):
        assert aufruf.plan.kostenmodell == modell
        assert aufruf.num_trials == modell.num_trials_fuer(OPTIMIERUNGSBUDGET / 8, N)
        assert aufruf.plan.num_trials == aufruf.num_trials
        assert aufruf.plan.dollar_gesamt <= OPTIMIERUNGSBUDGET / 8
    trials = {aufruf.kandidatenmodell: aufruf.num_trials for aufruf in lauf.aufrufe}
    assert trials[A] > trials[B] == trials[C] > trials[D]


def test_alle_laufen_mit_demselben_seed_als_sondierung_ihres_durchgangs(protokoll):
    """Gleicher Seed ueber alle Kandidatenmodelle (Messhygiene); jede Protokollzeile sagt,
    wozu sie gehoert."""
    _, lauf = sondiere(dicht_beieinander(), protokoll, seed=11, durchgang=4)

    erwartet = Einordnung.experiment("verfahren", durchgang=4, phase="sondierung")
    assert {aufruf.seed for aufruf in lauf.aufrufe} == {11}
    assert {aufruf.einordnung for aufruf in lauf.aufrufe} == {erwartet}


def test_reicht_der_budgetanteil_fuer_keinen_trial_startet_gar_kein_lauf(protokoll):
    """Geplant wird fuer alle vier, bevor der erste Lauf Geld kostet."""
    lauf = AttrappeLauf(dicht_beieinander())
    teuer = KOSTENMODELLE[:3] + (kostenmodell(D, 10.0),)
    verfahren = verfahren_mit(lauf, protokoll, kostenmodelle=teuer)

    with pytest.raises(ValueError, match="keinen einzigen Trial"):
        verfahren.sondiere(MultiHopQA(), OPTIMIERUNGSBUDGET, seed=0, durchgang=1)

    assert lauf.aufrufe == []
    assert protokoll.zeilen() == []


# --- Zwischenbewertung: die vier Grundfaelle --------------------------------------------


def test_liegt_ein_kandidatenmodell_klar_vorn_ueberlebt_es_allein(protokoll):
    vorgaben = {
        A: Vorgabe(BASIS),
        B: Vorgabe(WEIT_HINTEN),
        C: Vorgabe(schlechter(an=range(1, N, 2), um=0.3)),
        D: Vorgabe(schlechter(an=range(0, N, 3), um=0.3)),
    }

    zwischenstand, _ = sondiere(vorgaben, protokoll)

    entscheidung = zwischenstand.entscheidung
    assert entscheidung.weiter == (A,)
    assert entscheidung.fuehrender == A
    assert entscheidung.ausgeschieden == {B: "test", C: "test", D: "test"}
    assert entscheidung.test_hat_gefunden


def test_liegen_alle_dicht_beieinander_findet_der_test_nichts_und_die_obergrenze_greift(
    protokoll,
):
    zwischenstand, _ = sondiere(dicht_beieinander(), protokoll)

    entscheidung = zwischenstand.entscheidung
    assert entscheidung.weiter == (A, B)
    assert entscheidung.ausgeschieden == {C: "nicht-unter-den-zwei", D: "nicht-unter-den-zwei"}
    assert not entscheidung.test_hat_gefunden
    assert entscheidung.vergleiche == 3


def test_reisst_das_beste_die_kostenobergrenze_scheidet_es_trotz_hoechster_qualitaet_aus(
    protokoll,
):
    vorgaben = dicht_beieinander()
    vorgaben[A] = Vorgabe(BASIS, ausfuehrungskosten=ZU_TEUER)

    zwischenstand, _ = sondiere(vorgaben, protokoll)

    entscheidung = zwischenstand.entscheidung
    assert entscheidung.ausgeschieden[A] == "kostenfilter"
    assert A not in entscheidung.weiter
    assert entscheidung.fuehrender == B
    assert entscheidung.weiter == (B, C)


def test_die_reihenfolge_ist_bindend_vertauscht_wird_die_menge_der_ueberlebenden_leer():
    """Erst der Kostenfilter, dann der Fuehrende, dann der Test.

    A ist das beste Kandidatenmodell, reisst aber die Kostenobergrenze; B, C und D sind
    nachweisbar schlechter als A, untereinander aber nicht zu unterscheiden. Zieht man den
    Test vor, wirft er B, C und D gegen A hinaus - und danach der Kostenfilter A selbst.
    """
    staende = [
        Stand(A, BASIS, ZU_TEUER),
        Stand(B, WEIT_HINTEN, GUENSTIG),
        Stand(C, schlechter(an=range(0, N, 2), um=0.3, basis=vertauscht(0, 2)), TEURER),
        Stand(D, schlechter(an=set(range(0, N, 2)) | {1}, um=0.3), GUENSTIG),
    ]

    # Vertauscht: Fuehrender und Test vor dem Kostenfilter.
    fuehrender = rangfolge(staende)[0]
    nach_dem_test = [
        stand
        for stand in staende
        if stand is fuehrender or not p_wert(stand, fuehrender) < modul.ALPHA
    ]
    assert kostenfilter(nach_dem_test, KOSTENOBERGRENZE) == []

    # In der bindenden Reihenfolge bleibt jemand uebrig.
    entscheidung = modul.zwischenbewertung(staende, KOSTENOBERGRENZE)
    assert entscheidung.ausgeschieden[A] == "kostenfilter"
    assert entscheidung.weiter == (B, C)


# --- Schritt 3: wer durch den Test ausscheidet ------------------------------------------


def staende_zweitbester_gegen_drittbester():
    """Ein Beispiel mit 200 Fragen, das die Regel fuer Schritt 3 begruendet.

    Der Fuehrende hat 150 Fragen richtig (1,0) und 50 falsch (0,0): 0,750. A weicht nur bei 6
    Fragen ab, dort um 0,5 nach unten: 0,735 - ein kleiner, ganz gleichmaessiger Rueckstand.
    C ist bei 40 Fragen um 0,5 besser und bei 50 um 0,5 schlechter: 0,725 - ein groesserer,
    aber unregelmaessiger Rueckstand.
    """
    fuehrender = (1.0,) * 150 + (0.0,) * 50
    a = tuple(0.5 if i < 6 else wert for i, wert in enumerate(fuehrender))
    c = tuple(
        0.5 if 6 <= i < 56 or 150 <= i < 190 else wert for i, wert in enumerate(fuehrender)
    )
    return [
        Stand("Fuehrender", fuehrender, GUENSTIG),
        Stand("A", a, GUENSTIG),
        Stand("C", c, GUENSTIG),
    ]


def test_der_zweitbeste_fliegt_nicht_hinaus_waehrend_der_drittbeste_bleibt():
    """A ist nachweisbar schlechter als der Fuehrende (p = 0,007), C nicht (p = 0,146).
    Schiede jeder nachweisbar Schlechtere sofort aus, ginge C weiter und A nicht - ein
    schlechter bewertetes Kandidatenmodell vor einem besser bewerteten. Deshalb scheidet A
    durch den Test nur aus, wenn auch C es tut; C tut es nicht, also gehen die zwei Besten
    weiter: der Fuehrende und A."""
    staende = staende_zweitbester_gegen_drittbester()
    assert [stand.bewertung for stand in staende] == pytest.approx([0.750, 0.735, 0.725])

    entscheidung = zwischenbewertung(staende, KOSTENOBERGRENZE)

    assert entscheidung.p_werte == {
        "A": pytest.approx(0.007, abs=5e-4),
        "C": pytest.approx(0.146, abs=5e-4),
    }
    assert entscheidung.weiter == ("Fuehrender", "A")
    assert entscheidung.ausgeschieden == {"C": "nicht-unter-den-zwei"}


def test_ein_nachweisbar_schlechteres_bleibt_wenn_ein_schlechter_bewertetes_bleibt():
    """B ist nachweisbar schlechter (k = 3, p = 0,042) und bleibt trotzdem, weil C hinter ihm
    nicht nachweisbar schlechter ist (k = 2, p = 0,080). D ganz hinten ist nachweisbar
    schlechter, und hinter D steht niemand mehr: D scheidet durch den Test aus. B ist unter den
    zwei Besten und geht weiter."""
    staende = [
        Stand(A, BASIS, GUENSTIG),
        Stand(B, schlechter(an={0, 7, 14}, um=0.2), GUENSTIG),
        Stand(C, schlechter(an={0, 7}, um=0.4), GUENSTIG),
        Stand(D, WEIT_HINTEN, GUENSTIG),
    ]
    assert [stand.name for stand in rangfolge(staende)] == [A, B, C, D]

    entscheidung = zwischenbewertung(staende, KOSTENOBERGRENZE)

    assert entscheidung.p_werte[B] < modul.ALPHA < entscheidung.p_werte[C]
    assert entscheidung.weiter == (A, B)
    assert entscheidung.ausgeschieden == {C: "nicht-unter-den-zwei", D: "test"}


def test_ein_nachweisbar_schlechteres_jenseits_der_zwei_scheidet_als_nicht_unter_den_zwei_aus():
    """C ist nachweisbar schlechter (k = 3), D hinter ihm nicht (k = 2). C bleibt also nach
    dem Test, ist aber nicht unter den besten zwei - und so steht es auch in der Zeile."""
    staende = [
        Stand(A, BASIS, GUENSTIG),
        Stand(B, schlechter(an={0}, um=0.2), GUENSTIG),
        Stand(C, schlechter(an={0, 7, 14}, um=0.2), GUENSTIG),
        Stand(D, schlechter(an={0, 7}, um=0.4), GUENSTIG),
    ]
    assert [stand.name for stand in rangfolge(staende)] == [A, B, C, D]

    entscheidung = zwischenbewertung(staende, KOSTENOBERGRENZE)

    assert entscheidung.p_werte[C] < modul.ALPHA
    assert entscheidung.weiter == (A, B)
    assert entscheidung.ausgeschieden == {C: "nicht-unter-den-zwei", D: "nicht-unter-den-zwei"}


def test_der_handlauf_vom_dritten_oktober_entscheidet_unveraendert():
    """Die Nachbewertungen des Lasttests vom 03.10.2026 (im Protokoll ein Handlauf) aus dem
    echten Protokoll: nach Schritt 3 gehen DeepSeek als Fuehrender und Gemma weiter, wie im
    Lauf selbst. Der Lauf hatte keine Kostenobergrenze.

    Braucht `ergebnisse/protokoll.jsonl` mit diesen Zeilen; sonst uebersprungen."""
    zeilen = [
        zeile
        for zeile in Protokoll().zeilen()
        if zeile.get("zweck") == "handlauf"
        and zeile.get("abschnitt") == "nachbewertung"
        and zeile["zeitpunkt"].startswith("2026-10-03")
    ]
    if len(zeilen) != 4:
        pytest.skip(f"Die vier Nachbewertungen vom 03.10.2026 fehlen ({len(zeilen)} gefunden).")
    staende = [
        Stand(
            zeile["konfiguration"]["kandidatenmodell"],
            tuple(zeile["einzelbewertungen"]),
            zeile["dollar_je_programmlauf"],
        )
        for zeile in zeilen
    ]

    entscheidung = zwischenbewertung(staende, math.inf)

    assert entscheidung.weiter == ("deepseek/deepseek-v4.1-flash", "google/gemma-4-31b-it")
    assert entscheidung.ausgeschieden == {
        "openai/gpt-oss-120b": "test",
        "openai/gpt-6-luna": "test",
    }


def test_der_test_hat_gefunden_auch_wenn_er_niemanden_hinauswirft():
    """FF2 fragt, wie oft die Sondierung ueberhaupt signifikant unterscheiden konnte. Nach
    Schritt 3 kann ein Kandidatenmodell nachweisbar schlechter sein und trotzdem bleiben -
    der Test hat dann trotzdem etwas gefunden."""
    entscheidung = zwischenbewertung(staende_zweitbester_gegen_drittbester(), KOSTENOBERGRENZE)

    assert "test" not in entscheidung.ausgeschieden.values()
    assert entscheidung.test_hat_gefunden


# --- Zwischenbewertung: Gruende, Zaehlung, Test und Randfaelle ---------------------------


def test_je_kandidatenmodell_steht_woran_es_gescheitert_ist(protokoll):
    """Kostenfilter, Test oder nicht unter den besten zwei - und wer weiter ist, steht nicht
    unter den Ausgeschiedenen."""
    vorgaben = {
        A: Vorgabe(BASIS, ausfuehrungskosten=ZU_TEUER),
        B: Vorgabe(BASIS),
        C: Vorgabe(WEIT_HINTEN),
        D: Vorgabe(schlechter(an={3}, um=0.1)),
    }

    zwischenstand, _ = sondiere(vorgaben, protokoll)

    entscheidung = zwischenstand.entscheidung
    assert entscheidung.weiter == (B, D)
    assert entscheidung.ausgeschieden == {A: "kostenfilter", C: "test"}


def test_gezaehlt_wird_ob_der_test_etwas_gefunden_hat_und_wie_oft_er_verglichen_hat(protokoll):
    """FF2: wie oft konnte die Sondierung ueberhaupt unterscheiden. Laesst der Kostenfilter
    nur einen durch, hat der Test nichts zu vergleichen."""
    vorgaben = {
        A: Vorgabe(BASIS, ausfuehrungskosten=ZU_TEUER),
        B: Vorgabe(BASIS),
        C: Vorgabe(WEIT_HINTEN, ausfuehrungskosten=ZU_TEUER),
        D: Vorgabe(WEIT_HINTEN, ausfuehrungskosten=ZU_TEUER),
    }

    zwischenstand, _ = sondiere(vorgaben, protokoll)

    entscheidung = zwischenstand.entscheidung
    assert entscheidung.weiter == (B,)
    assert entscheidung.vergleiche == 0
    assert not entscheidung.test_hat_gefunden


def test_der_test_ist_einseitig_mit_alpha_fuenf_prozent():
    """Einseitig: gefragt ist nur, ob ein Kandidatenmodell *schlechter* ist als der
    Fuehrende. Bei k = 3 abweichenden Beispielen liegt p einseitig bei 0,042 und
    fuehrt zum Ausscheiden; zweiseitig waeren es 0,083, und es bliebe."""
    fuehrender = Stand(A, BASIS, GUENSTIG)
    drei = Stand(B, schlechter(an={0, 7, 14}), GUENSTIG)
    zwei = Stand(C, schlechter(an={0, 7}), GUENSTIG)

    assert modul.ALPHA == 0.05
    assert p_wert(drei, fuehrender) == pytest.approx(0.0416, abs=1e-4)
    assert p_wert(zwei, fuehrender) == pytest.approx(0.0800, abs=1e-4)

    entscheidung = modul.zwischenbewertung([fuehrender, drei, zwei], KOSTENOBERGRENZE)
    assert entscheidung.ausgeschieden == {B: "test"}
    assert entscheidung.weiter == (A, C)


# Ein in jedem Beispiel exakt gleicher Abstand hat die Streuung null; scipy warnt dann vor
# Rundungsfehlern. In echten Einzelbewertungen kommt das nicht vor, hier ist es der Punkt.
@pytest.mark.filterwarnings("ignore:Precision loss occurred")
def test_der_test_vergleicht_frage_fuer_frage_und_nicht_die_mittelwerte():
    """Gepaart: dieselbe Abweichung in den Mittelwerten ist nachweisbar, wenn sie in jedem
    Beispiel gleich ausfaellt, und nicht nachweisbar, wenn sie in einem einzigen steckt."""
    fuehrender = Stand(A, BASIS, GUENSTIG)
    ueberall = Stand(B, schlechter(an=range(N), um=0.01), GUENSTIG)
    an_einem = Stand(C, schlechter(an={0}, um=0.4), GUENSTIG)

    assert p_wert(ueberall, fuehrender) < 0.001
    assert p_wert(an_einem, fuehrender) > 0.1


def test_bei_gleichstand_fuehrt_wer_die_niedrigeren_ausfuehrungskosten_hat():
    """Randfall Gleichstand: es entscheiden die Ausfuehrungskosten - beim Bestimmen des
    Fuehrenden."""
    staende = [
        Stand(A, BASIS, TEURER),
        Stand(B, vertauscht(0, 1), GUENSTIG),
        Stand(C, WEIT_HINTEN, GUENSTIG),
    ]

    entscheidung = modul.zwischenbewertung(staende, KOSTENOBERGRENZE)

    assert entscheidung.fuehrender == B
    assert entscheidung.weiter == (B, A)


def test_bei_gleichstand_um_den_zweiten_platz_entscheiden_die_ausfuehrungskosten():
    """Randfall Gleichstand - beim Schritt "hoechstens zwei"."""
    staende = [
        Stand(A, BASIS, GUENSTIG),
        Stand(B, schlechter(an={5}, um=0.1), TEURER),
        Stand(C, schlechter(an={5}, um=0.1, basis=vertauscht(0, 1)), GUENSTIG),
    ]

    entscheidung = modul.zwischenbewertung(staende, KOSTENOBERGRENZE)

    assert entscheidung.weiter == (A, C)
    assert entscheidung.ausgeschieden == {B: "nicht-unter-den-zwei"}


def test_wer_genau_auf_der_kostenobergrenze_liegt_haelt_sie_ein():
    """"Hoechstens" schliesst den Wert ein."""
    genau = Stand(A, BASIS, KOSTENOBERGRENZE / 1000)

    assert kostenfilter([genau], KOSTENOBERGRENZE) == [genau]


def test_identische_einzelbewertungen_sind_nicht_nachweisbar_schlechter():
    """Weichen zwei Kandidatenmodelle in keinem Beispiel ab, ist der t-Wert 0/0. Das ist
    kein Unterschied, also auch kein Ausscheiden; der p-Wert bleibt leer statt erfunden."""
    fuehrender = Stand(A, BASIS, GUENSTIG)
    gleich = Stand(B, BASIS, TEURER)

    assert p_wert(gleich, fuehrender) is None
    assert modul.zwischenbewertung([fuehrender, gleich], KOSTENOBERGRENZE).weiter == (A, B)


def test_ohne_gemessene_ausfuehrungskosten_faellt_keine_entscheidung(protokoll):
    """Die Naht meldet `None`, wenn kein Programmlauf gemessen wurde. Keine Messung ist keine
    Messung von null; der Kostenfilter haette nichts, woran er pruefen koennte."""
    vorgaben = dicht_beieinander()
    vorgaben[C] = Vorgabe(BASIS, ausfuehrungskosten=None)

    with pytest.raises(UngueltigerDurchgang, match="Ausfuehrungskosten"):
        sondiere(vorgaben, protokoll)

    assert protokoll.zeilen() == []


def test_einzelbewertungen_ueber_verschiedene_beispiele_lassen_sich_nicht_paaren():
    with pytest.raises(ValueError, match="paaren"):
        p_wert(Stand(B, BASIS[:-1], GUENSTIG), Stand(A, BASIS, GUENSTIG))


# --- Kein Gewinner, Restbudget, Protokoll, Abbruch, Wiederholbarkeit ---------------------


def test_scheitern_alle_am_kostenfilter_gibt_es_keinen_gewinner_und_das_restbudget_bleibt(
    protokoll,
):
    """Randfall kein Gewinner: das Verfahren endet nach der Sondierung, ohne die Fokusphase
    zu starten; das Restbudget wird nicht ausgegeben und steht in der Zeile."""
    vorgaben = {
        name: Vorgabe(BASIS, ausfuehrungskosten=ZU_TEUER) for name in (A, B, C, D)
    }

    zwischenstand, lauf = sondiere(vorgaben, protokoll)

    assert zwischenstand.kein_gewinner
    assert zwischenstand.entscheidung.weiter == ()
    assert zwischenstand.entscheidung.fuehrender is None
    assert set(zwischenstand.entscheidung.ausgeschieden.values()) == {"kostenfilter"}
    assert len(lauf.aufrufe) == 4
    (zeile,) = protokoll.zeilen()
    assert zeile["kein_gewinner"] is True
    assert zeile["restbudget"] == pytest.approx(zwischenstand.restbudget)
    assert zwischenstand.restbudget > 0


def test_das_restbudget_ist_das_budget_minus_die_tatsaechlich_ausgegebenen_dollar(protokoll):
    """Nicht minus die geplanten: Optimierung und Nachbewertung, wie sie wirklich gekostet
    haben."""
    vorgaben = dicht_beieinander()
    vorgaben[B] = vorgaben[B]._replace(mehrkosten=0.03)

    zwischenstand, lauf = sondiere(vorgaben, protokoll)

    geplant = sum(aufruf.plan.dollar_gesamt for aufruf in lauf.aufrufe)
    assert zwischenstand.ausgegeben == pytest.approx(geplant + 0.03)
    assert zwischenstand.restbudget == pytest.approx(OPTIMIERUNGSBUDGET - geplant - 0.03)
    assert not zwischenstand.kein_gewinner


def test_die_entscheidungszeile_steht_im_protokoll(protokoll):
    """Die Auswertung liest nur das Protokoll: wer weiter ist, wer
    ausscheidet, warum, und ob der Test etwas gefunden hat - nachrechenbar."""
    vorgaben = dicht_beieinander()
    vorgaben[A] = Vorgabe(BASIS, ausfuehrungskosten=ZU_TEUER)
    vorgaben[C] = Vorgabe(WEIT_HINTEN)

    zwischenstand, _ = sondiere(vorgaben, protokoll, seed=2, durchgang=5)

    (zeile,) = protokoll.zeilen()
    assert zeile["abschnitt"] == "zwischenbewertung"
    assert (zeile["zweck"], zeile["baseline"], zeile["phase"], zeile["durchgang"]) == (
        "experiment",
        "verfahren",
        "sondierung",
        5,
    )
    assert zeile["seed"] == 2
    assert zeile["weiter"] == [B, D]
    assert zeile["fuehrender"] == B
    assert zeile["ausgeschieden"] == {A: "kostenfilter", C: "test"}
    assert zeile["test_hat_gefunden"] is True
    assert zeile["vergleiche"] == 2
    assert zeile["kein_gewinner"] is False
    assert zeile["kostenobergrenze"] == KOSTENOBERGRENZE
    assert zeile["alpha"] == 0.05
    assert zeile["optimierungsbudget"] == OPTIMIERUNGSBUDGET
    assert zeile["ausgegeben"] == pytest.approx(zwischenstand.ausgegeben)
    assert zeile["restbudget"] == pytest.approx(zwischenstand.restbudget)
    assert zeile["kandidatenmodelle"][A] == {
        "bewertung": pytest.approx(statistics.fmean(BASIS)),
        "ausfuehrungskosten_je_1000": pytest.approx(ZU_TEUER * 1000),
        "p_wert": None,
    }
    assert zeile["kandidatenmodelle"][B]["p_wert"] is None
    assert zeile["kandidatenmodelle"][C]["p_wert"] < 0.05
    assert zeile["kandidatenmodelle"][D]["p_wert"] > 0.05
    assert "dollar" not in zeile


def test_ein_anbieterfehler_bricht_den_durchgang_ab_statt_zu_entscheiden(protokoll):
    """Ein gedrosseltes Modell saehe sonst aus wie ein schlechtes. Abgebrochen wird
    sofort - der Durchgang ist ohnehin ungueltig, jeder weitere Lauf waere verlorenes Geld."""
    vorgaben = dicht_beieinander()
    vorgaben[B] = vorgaben[B]._replace(anbieterfehler=1)

    with pytest.raises(UngueltigerDurchgang, match="Anbieterfehler") as abbruch:
        sondiere(vorgaben, protokoll, durchgang=3)

    assert B in str(abbruch.value)
    assert "Durchgang 3" in str(abbruch.value)
    assert protokoll.zeilen() == []


def test_ein_formatfehler_bricht_nichts_ab_er_steckt_schon_als_null_in_der_bewertung(protokoll):
    """Ein Formatfehler ist eine Schwaeche des Modells und zaehlt als
    Einzelbewertung 0,0. Er macht den Durchgang nicht ungueltig."""
    vorgaben = dicht_beieinander()
    vorgaben[B] = vorgaben[B]._replace(formatfehler=5)
    ohne, _ = sondiere(dicht_beieinander(), Protokoll(protokoll.pfad.with_name("ohne.jsonl")))

    mit, lauf = sondiere(vorgaben, protokoll)

    assert len(lauf.aufrufe) == 4
    assert mit.entscheidung == ohne.entscheidung
    assert len(protokoll.zeilen()) == 1


def test_ein_anbieterfehler_haelt_auch_die_folgenden_laeufe_an(protokoll):
    lauf = AttrappeLauf({**dicht_beieinander(), B: Vorgabe(BASIS, anbieterfehler=2)})

    with pytest.raises(UngueltigerDurchgang):
        verfahren_mit(lauf, protokoll).sondiere(
            MultiHopQA(), OPTIMIERUNGSBUDGET, seed=0, durchgang=1
        )

    assert [aufruf.kandidatenmodell for aufruf in lauf.aufrufe] == [A, B]


def test_gleiche_eingaben_und_gleicher_seed_fuehren_zu_gleichen_entscheidungen(tmp_path):
    erstes, erster_lauf = sondiere(dicht_beieinander(), Protokoll(tmp_path / "1.jsonl"), seed=3)
    zweites, zweiter_lauf = sondiere(dicht_beieinander(), Protokoll(tmp_path / "2.jsonl"), seed=3)

    assert erstes.entscheidung == zweites.entscheidung
    assert erster_lauf.aufrufe == zweiter_lauf.aufrufe
    (erste_zeile,) = Protokoll(tmp_path / "1.jsonl").zeilen()
    (zweite_zeile,) = Protokoll(tmp_path / "2.jsonl").zeilen()
    erste_zeile.pop("zeitpunkt")
    zweite_zeile.pop("zeitpunkt")
    assert erste_zeile == zweite_zeile


def test_der_zwischenstand_haelt_die_ergebnisse_der_sondierung_fuer_die_fokusphase(protokoll):
    """Der Warm-Start der Fokusphase braucht die Programme der Sondierung."""
    zwischenstand, _ = sondiere(dicht_beieinander(), protokoll)

    assert list(zwischenstand.ergebnisse) == [A, B, C, D]
    assert zwischenstand.ergebnisse[A].bewertungen_je_beispiel == BASIS


# --- Fokusphase -------------------------------------------------------------------------


def durchlaufe(vorgaben, protokoll, *, fokus=None, seed=0, durchgang=1):
    """Ein ganzer Durchgang: Sondierung, Zwischenbewertung, Fokusphase, Endauswahl."""
    lauf = AttrappeLauf(vorgaben, fokus=fokus)
    ergebnis = verfahren_mit(lauf, protokoll)(
        MultiHopQA(), OPTIMIERUNGSBUDGET, seed=seed, durchgang=durchgang
    )
    return ergebnis, lauf


def in_der_fokusphase(lauf):
    return [aufruf for aufruf in lauf.aufrufe if aufruf.einordnung.phase == "fokus"]


def klar_vorn():
    """A liegt klar vorn: alle anderen sind nachweisbar schlechter."""
    return {
        A: Vorgabe(BASIS),
        B: Vorgabe(WEIT_HINTEN),
        C: Vorgabe(schlechter(an=range(1, N, 2), um=0.3)),
        D: Vorgabe(schlechter(an=range(0, N, 3), um=0.3)),
    }


def test_das_restbudget_geht_zu_gleichen_dollar_teilen_an_die_ueberlebenden(protokoll):
    """Gleiche Dollar, nicht gleich viel Suche: wessen Trial billiger ist, bekommt mehr
    davon. Jeder Anteil bezahlt auch die Nachbewertung."""
    ergebnis, lauf = durchlaufe(dicht_beieinander(), protokoll, seed=7, durchgang=2)

    restbudget = ergebnis.zwischenstand.restbudget
    fokus = in_der_fokusphase(lauf)
    assert [aufruf.kandidatenmodell for aufruf in fokus] == [A, B]
    kostenmodelle = {modell.kandidatenmodell: modell for modell in KOSTENMODELLE}
    for aufruf in fokus:
        modell = kostenmodelle[aufruf.kandidatenmodell]
        assert aufruf.plan.budgetanteil == pytest.approx(restbudget / 2)
        assert aufruf.plan.kostenmodell == modell
        assert aufruf.num_trials == modell.num_trials_fuer(restbudget / 2, N)
        assert aufruf.seed == 7
        assert aufruf.einordnung == Einordnung.experiment(
            "verfahren", durchgang=2, phase="fokus"
        )
    trials = {aufruf.kandidatenmodell: aufruf.num_trials for aufruf in fokus}
    assert trials[A] > trials[B]


def test_ein_einziger_ueberlebender_bekommt_das_ganze_restbudget(protokoll):
    ergebnis, lauf = durchlaufe(klar_vorn(), protokoll)

    (aufruf,) = in_der_fokusphase(lauf)
    assert aufruf.kandidatenmodell == A
    assert aufruf.plan.budgetanteil == pytest.approx(ergebnis.zwischenstand.restbudget)


def test_das_restbudget_kommt_aus_den_tatsaechlichen_kosten_der_sondierung(protokoll):
    """Auch was ein spaeter Ausgeschiedener ueberzogen hat, fehlt der Fokusphase."""
    vorgaben = dicht_beieinander()
    vorgaben[C] = vorgaben[C]._replace(mehrkosten=0.4)

    _, lauf = durchlaufe(vorgaben, protokoll)

    geplant = sum(
        aufruf.plan.dollar_gesamt
        for aufruf in lauf.aufrufe
        if aufruf.einordnung.phase == "sondierung"
    )
    for aufruf in in_der_fokusphase(lauf):
        assert aufruf.plan.budgetanteil == pytest.approx(
            (OPTIMIERUNGSBUDGET - geplant - 0.4) / 2
        )


def test_reicht_der_anteil_eines_ueberlebenden_fuer_keinen_trial_bricht_der_durchgang_ab(
    protokoll,
):
    """Das Kostenmodell lehnt den Anteil mit dem Mindestbudget ab - bevor die Fokusphase
    Geld kostet."""
    vorgaben = dicht_beieinander()
    vorgaben[D] = vorgaben[D]._replace(mehrkosten=1.9)
    lauf = AttrappeLauf(vorgaben)

    with pytest.raises(ValueError, match="keinen einzigen Trial"):
        verfahren_mit(lauf, protokoll)(
            MultiHopQA(), OPTIMIERUNGSBUDGET, seed=0, durchgang=1
        )

    assert in_der_fokusphase(lauf) == []
    assert [zeile["abschnitt"] for zeile in protokoll.zeilen()] == ["zwischenbewertung"]


def test_der_warm_start_traegt_die_instruktion_aber_keine_few_shot_beispiele(protokoll):
    """MIPROv2 misst den Startwert mit den mitgegebenen Beispielen, bucht ihn aber als "ohne
    Beispiele"."""
    ergebnis, lauf = durchlaufe(dicht_beieinander(), protokoll)

    startprogramme = {
        aufruf.kandidatenmodell: programm
        for aufruf, programm in zip(lauf.aufrufe, lauf.startprogramme)
        if aufruf.einordnung.phase == "fokus"
    }
    assert set(startprogramme) == {A, B}
    for name, startprogramm in startprogramme.items():
        for _, praediktor in startprogramm.named_predictors():
            assert praediktor.signature.instructions == f"sondierung: {name}"
            assert praediktor.demos == []
        # Das Programm der Sondierung selbst bleibt, wie es war.
        sondiert = ergebnis.zwischenstand.ergebnisse[name].programm
        assert all(len(praediktor.demos) == 1 for _, praediktor in sondiert.named_predictors())


def test_scheitern_schon_alle_am_kostenfilter_startet_die_fokusphase_nicht(protokoll):
    """Randfall kein Gewinner: keine Zusatzangabe - kein Programm lag
    unter der Obergrenze -, und das Restbudget bleibt unausgegeben."""
    vorgaben = {name: Vorgabe(BASIS, ausfuehrungskosten=ZU_TEUER) for name in (A, B, C, D)}

    ergebnis, lauf = durchlaufe(vorgaben, protokoll)

    assert len(lauf.aufrufe) == 4
    assert ergebnis.kein_gewinner
    assert ergebnis.gewinner is None
    assert ergebnis.programm is None
    assert ergebnis.zusatzangabe is None
    assert ergebnis.endauswahl is None
    assert "Kostenobergrenze" in ergebnis.meldung
    assert "Sondierung" in ergebnis.meldung
    assert ergebnis.budgetverbrauch.fokus == 0
    assert ergebnis.budgetverbrauch.uebrig == pytest.approx(ergebnis.zwischenstand.restbudget)
    assert [platz.kandidatenmodell for platz in ergebnis.rangliste] == [A, B, C, D]
    assert [zeile["abschnitt"] for zeile in protokoll.zeilen()] == ["zwischenbewertung"]


# --- Endauswahl -------------------------------------------------------------------------


def staende_der_fokusphase(lauf):
    return [
        Stand(
            aufruf.kandidatenmodell,
            lauf.vorgabe(aufruf.kandidatenmodell, "fokus").bewertungen,
            lauf.vorgabe(aufruf.kandidatenmodell, "fokus").ausfuehrungskosten,
        )
        for aufruf in in_der_fokusphase(lauf)
    ]


def test_die_endauswahl_wendet_dieselben_vier_schritte_an_wie_die_zwischenbewertung(protokoll):
    """Das beste Programm der Fokusphase reisst die Kostenobergrenze: es scheidet trotz
    hoechster Qualitaet aus, gewonnen hat das andere."""
    ergebnis, lauf = durchlaufe(
        dicht_beieinander(),
        protokoll,
        fokus={A: Vorgabe(BASIS, ausfuehrungskosten=ZU_TEUER), B: Vorgabe(WEIT_HINTEN)},
    )

    staende = staende_der_fokusphase(lauf)
    assert ergebnis.endauswahl == zwischenbewertung(staende, KOSTENOBERGRENZE)
    assert ergebnis.endauswahl.ausgeschieden == {A: "kostenfilter"}
    assert ergebnis.gewinner == B
    assert not ergebnis.kein_gewinner
    assert ergebnis.meldung is None
    assert ergebnis.zusatzangabe is None


def test_in_der_endauswahl_scheidet_aus_wer_nachweisbar_schlechter_ist(protokoll):
    """In der Fokusphase kann der Zweite der Sondierung den Ersten ueberholen."""
    ergebnis, _ = durchlaufe(
        dicht_beieinander(), protokoll, fokus={A: Vorgabe(WEIT_HINTEN), B: Vorgabe(BASIS)}
    )

    assert ergebnis.gewinner == B
    assert ergebnis.endauswahl.fuehrender == B
    assert ergebnis.endauswahl.ausgeschieden == {A: "test"}
    assert ergebnis.endauswahl.test_hat_gefunden


def test_bei_gleichstand_in_der_endauswahl_gewinnen_die_niedrigeren_ausfuehrungskosten(
    protokoll,
):
    """Randfall Gleichstand - in der Endauswahl wie in der Zwischenbewertung."""
    ergebnis, _ = durchlaufe(
        dicht_beieinander(),
        protokoll,
        fokus={
            A: Vorgabe(BASIS, ausfuehrungskosten=TEURER),
            B: Vorgabe(vertauscht(0, 1), ausfuehrungskosten=GUENSTIG),
        },
    )

    assert ergebnis.gewinner == B


def test_reisst_am_ende_jedes_paar_die_obergrenze_kommt_kein_gewinner_aber_eine_zusatzangabe(
    protokoll,
):
    """Die Ausfuehrungskosten sind in der Fokusphase gestiegen. Statt des
    guenstigsten zu teuren Paares gibt es eine klare Meldung - und das guenstigere Programm
    der Sondierung, das die Obergrenze noch einhielt, als Zusatzangabe, nicht als Ergebnis."""
    vorgaben = dicht_beieinander()
    vorgaben[A] = vorgaben[A]._replace(ausfuehrungskosten=TEURER)
    vorgaben[B] = vorgaben[B]._replace(ausfuehrungskosten=GUENSTIG)

    ergebnis, _ = durchlaufe(
        vorgaben,
        protokoll,
        fokus={
            A: Vorgabe(BASIS, ausfuehrungskosten=ZU_TEUER),
            B: Vorgabe(BASIS, ausfuehrungskosten=ZU_TEUER),
        },
    )

    assert ergebnis.kein_gewinner
    assert ergebnis.gewinner is None
    assert ergebnis.programm is None
    assert ergebnis.endauswahl.ausgeschieden == {A: "kostenfilter", B: "kostenfilter"}
    assert "Kostenobergrenze" in ergebnis.meldung
    assert "Zusatzangabe" in ergebnis.meldung
    assert B in ergebnis.meldung
    zusatz = ergebnis.zusatzangabe
    assert isinstance(zusatz, ModellProgrammPaar)
    assert zusatz.kandidatenmodell == B
    # Das Programm der Sondierung, so wie es gemessen wurde - samt Few-Shot-Beispielen.
    for _, praediktor in zusatz.programm.named_predictors():
        assert praediktor.signature.instructions == f"sondierung: {B}"
        assert len(praediktor.demos) == 1


def test_das_ergebnis_ist_ein_modell_programm_paar_auf_dem_gewinnermodell(protokoll):
    """Ein gewoehnliches DSPy-Programm, das Gewinnermodell fest in den
    Praediktoren - das optimierte Programm der Fokusphase."""
    ergebnis, _ = durchlaufe(dicht_beieinander(), protokoll)

    assert ergebnis.gewinner == A
    assert isinstance(ergebnis.programm, ModellProgrammPaar)
    assert ergebnis.programm.kandidatenmodell == A
    for _, praediktor in ergebnis.programm.programm.named_predictors():
        assert praediktor.signature.instructions == f"fokus: {A}"


def test_rangliste_aller_kandidatenmodelle_und_budgetverbrauch_haengen_am_ergebnis(protokoll):
    """Vorn, wer in die Fokusphase kam, geordnet nach der Endauswahl; dahinter die in der
    Sondierung Ausgeschiedenen. Wer an der Kostenobergrenze scheiterte, steht hinter allen,
    die sie einhalten - wie bei der Wahl des Fuehrenden."""
    vorgaben = {
        A: Vorgabe(BASIS, ausfuehrungskosten=ZU_TEUER),
        B: Vorgabe(BASIS),
        C: Vorgabe(WEIT_HINTEN),
        D: Vorgabe(schlechter(an={3}, um=0.1)),
    }
    ergebnis, lauf = durchlaufe(
        vorgaben,
        protokoll,
        fokus={B: Vorgabe(schlechter(an={5}, um=0.1)), D: Vorgabe(BASIS)},
    )

    assert [platz.kandidatenmodell for platz in ergebnis.rangliste] == [D, B, C, A]
    assert [platz.phase for platz in ergebnis.rangliste] == [
        "fokus", "fokus", "sondierung", "sondierung"
    ]
    assert [platz.ausgeschieden for platz in ergebnis.rangliste] == [
        None, None, "test", "kostenfilter"
    ]
    assert ergebnis.rangliste[0].bewertung == pytest.approx(statistics.fmean(BASIS))
    assert ergebnis.rangliste[-1].ausfuehrungskosten_je_1000 == pytest.approx(ZU_TEUER * 1000)

    verbrauch = ergebnis.budgetverbrauch
    fokus = sum(aufruf.plan.dollar_gesamt for aufruf in in_der_fokusphase(lauf))
    assert verbrauch.optimierungsbudget == OPTIMIERUNGSBUDGET
    assert verbrauch.sondierung == pytest.approx(ergebnis.zwischenstand.ausgegeben)
    assert verbrauch.fokus == pytest.approx(fokus)
    assert verbrauch.gesamt == pytest.approx(verbrauch.sondierung + fokus)
    assert verbrauch.uebrig == pytest.approx(OPTIMIERUNGSBUDGET - verbrauch.gesamt)


def test_der_budgetverbrauch_umfasst_die_nachbewertungen_und_jede_planabweichung(protokoll):
    """Die Nachbewertung zahlt der Budgetanteil. Und ueberzieht ein Lauf der Fokusphase,
    steht auch das im Verbrauch - das Verfahren plant das Budget, es regelt es nicht."""
    ergebnis, lauf = durchlaufe(
        dicht_beieinander(), protokoll, fokus={B: Vorgabe(BASIS, mehrkosten=0.05)}
    )

    alles = sum(aufruf.plan.dollar_gesamt for aufruf in lauf.aufrufe) + 0.05
    nachbewertungen = sum(aufruf.plan.dollar_nachbewertung for aufruf in lauf.aufrufe)
    assert nachbewertungen > 0
    assert ergebnis.budgetverbrauch.gesamt == pytest.approx(alles)


def test_die_entscheidungszeile_der_endauswahl_steht_im_protokoll(protokoll):
    """Wie die Zwischenbewertung, dazu Gewinner, Zusatzangabe, Rangliste
    und der Verbrauch des ganzen Durchgangs."""
    ergebnis, _ = durchlaufe(
        dicht_beieinander(),
        protokoll,
        fokus={A: Vorgabe(BASIS, ausfuehrungskosten=ZU_TEUER), B: Vorgabe(WEIT_HINTEN)},
        seed=4,
        durchgang=6,
    )

    zwischen, zeile = protokoll.zeilen()
    assert zwischen["abschnitt"] == "zwischenbewertung"
    assert zeile["abschnitt"] == "endauswahl"
    assert (zeile["zweck"], zeile["baseline"], zeile["phase"], zeile["durchgang"]) == (
        "experiment",
        "verfahren",
        "fokus",
        6,
    )
    assert zeile["seed"] == 4
    assert zeile["gewinner"] == B
    assert zeile["kein_gewinner"] is False
    assert zeile["zusatzangabe"] is None
    assert zeile["weiter"] == [B]
    assert zeile["fuehrender"] == B
    assert zeile["ausgeschieden"] == {A: "kostenfilter"}
    assert zeile["test_hat_gefunden"] is False
    assert zeile["vergleiche"] == 0
    assert zeile["kostenobergrenze"] == KOSTENOBERGRENZE
    assert zeile["alpha"] == 0.05
    assert set(zeile["kandidatenmodelle"]) == {A, B}
    assert zeile["kandidatenmodelle"][A]["ausfuehrungskosten_je_1000"] == pytest.approx(
        ZU_TEUER * 1000
    )
    assert zeile["rangliste"] == [B, A, C, D]
    verbrauch = ergebnis.budgetverbrauch
    assert zeile["optimierungsbudget"] == OPTIMIERUNGSBUDGET
    assert zeile["ausgegeben_sondierung"] == pytest.approx(verbrauch.sondierung)
    assert zeile["ausgegeben_fokus"] == pytest.approx(verbrauch.fokus)
    assert zeile["ausgegeben"] == pytest.approx(verbrauch.gesamt)
    assert zeile["uebrig"] == pytest.approx(verbrauch.uebrig)
    assert "dollar" not in zeile


def test_ohne_gewinner_nennt_die_entscheidungszeile_die_zusatzangabe(protokoll):
    vorgaben = dicht_beieinander()
    vorgaben[A] = vorgaben[A]._replace(ausfuehrungskosten=TEURER)
    ergebnis, _ = durchlaufe(
        vorgaben,
        protokoll,
        fokus={name: Vorgabe(BASIS, ausfuehrungskosten=ZU_TEUER) for name in (A, B)},
    )

    *_, zeile = protokoll.zeilen()
    assert zeile["gewinner"] is None
    assert zeile["kein_gewinner"] is True
    assert zeile["zusatzangabe"] == B == ergebnis.zusatzangabe.kandidatenmodell


def test_ein_anbieterfehler_in_der_fokusphase_bricht_den_durchgang_ab(protokoll):
    """Wie in der Sondierung: keine Endauswahl auf dieser Grundlage."""
    with pytest.raises(UngueltigerDurchgang, match="Anbieterfehler") as abbruch:
        durchlaufe(
            dicht_beieinander(),
            protokoll,
            fokus={B: Vorgabe(BASIS, anbieterfehler=1)},
            durchgang=8,
        )

    assert B in str(abbruch.value)
    assert "Fokusphase" in str(abbruch.value)
    assert "Durchgang 8" in str(abbruch.value)
    assert [zeile["abschnitt"] for zeile in protokoll.zeilen()] == ["zwischenbewertung"]


def test_fehlen_in_der_fokusphase_die_ausfuehrungskosten_faellt_keine_endauswahl(protokoll):
    with pytest.raises(UngueltigerDurchgang, match="Ausfuehrungskosten"):
        durchlaufe(
            dicht_beieinander(), protokoll, fokus={A: Vorgabe(BASIS, ausfuehrungskosten=None)}
        )

    assert [zeile["abschnitt"] for zeile in protokoll.zeilen()] == ["zwischenbewertung"]


def test_ein_vollstaendiger_durchgang_ende_zu_ende(tmp_path):
    """Sondierung, Zwischenbewertung, Fokusphase, Endauswahl - gegen die Attrappe, und mit
    gleichem Seed zweimal gleich. Am Ende steht ein Programm, das sich speichern und ohne
    weitere Angaben wieder laden laesst (ausgefuehrt wird es in `tests/test_programm.py`)."""

    def einmal(datei):
        protokoll = Protokoll(tmp_path / datei)
        ergebnis, lauf = durchlaufe(
            dicht_beieinander(), protokoll, fokus={B: Vorgabe(BASIS)}, seed=5, durchgang=3
        )
        zeilen = protokoll.zeilen()
        for zeile in zeilen:
            zeile.pop("zeitpunkt")
        return ergebnis, lauf, zeilen

    ergebnis, lauf, zeilen = einmal("1.jsonl")
    _, zweiter_lauf, zweite_zeilen = einmal("2.jsonl")

    assert [(aufruf.kandidatenmodell, aufruf.einordnung.phase) for aufruf in lauf.aufrufe] == [
        (A, "sondierung"), (B, "sondierung"), (C, "sondierung"), (D, "sondierung"),
        (A, "fokus"), (B, "fokus"),
    ]
    assert [zeile["abschnitt"] for zeile in zeilen] == ["zwischenbewertung", "endauswahl"]
    # Gleichstand in Bewertung und Kosten: die feste Reihenfolge der Laeufe entscheidet.
    assert ergebnis.gewinner == A
    assert ergebnis.programm.kandidatenmodell == A
    assert [platz.kandidatenmodell for platz in ergebnis.rangliste] == [A, B, C, D]
    assert ergebnis.budgetverbrauch.gesamt <= OPTIMIERUNGSBUDGET
    assert lauf.aufrufe == zweiter_lauf.aufrufe
    assert zeilen == zweite_zeilen

    ergebnis.programm.save(tmp_path / "gewinner", save_program=True)
    geladen = dspy.load(tmp_path / "gewinner", allow_pickle=True)
    assert geladen.kandidatenmodell == A
    for _, praediktor in geladen.programm.named_predictors():
        assert praediktor.signature.instructions == f"fokus: {A}"


def test_ohne_gewinner_nach_der_fokusphase_gibt_es_den_fuehrenden_ohne_kostenfilter(protokoll):
    """Reisst am Ende jedes Paar die Kostenobergrenze, bekommt neben der
    Zusatzangabe auch der Fuehrende der Fokusphase eine Testbewertung - der mit der hoechsten
    Bewertung, als waere der Kostenfilter nicht da. Dafuer gibt das Ergebnis die Programme der
    Fokusphase heraus, ohne eigenes LM."""
    ergebnis, _ = durchlaufe(
        dicht_beieinander(),
        protokoll,
        fokus={
            A: Vorgabe(WEIT_HINTEN, ausfuehrungskosten=ZU_TEUER),
            B: Vorgabe(BASIS, ausfuehrungskosten=ZU_TEUER),
        },
    )

    assert ergebnis.kein_gewinner
    assert ergebnis.fuehrender_ohne_kostenfilter == B
    assert set(ergebnis.ergebnisse_der_fokusphase) == {A, B}
    programm = ergebnis.ergebnisse_der_fokusphase[B].programm
    lehne_eigenes_lm_ab(programm)
    for _, praediktor in programm.named_predictors():
        assert praediktor.signature.instructions == f"fokus: {B}"


def test_der_fuehrende_ohne_kostenfilter_kann_ein_anderer_sein_als_der_gewinner(protokoll):
    """Reisst das beste Programm der Fokusphase die Obergrenze, gewinnt das andere; ohne
    Kostenfilter fuehrt trotzdem das beste."""
    ergebnis, _ = durchlaufe(
        dicht_beieinander(),
        protokoll,
        fokus={A: Vorgabe(BASIS, ausfuehrungskosten=ZU_TEUER), B: Vorgabe(WEIT_HINTEN)},
    )

    assert ergebnis.gewinner == B
    assert ergebnis.fuehrender_ohne_kostenfilter == A
    assert ergebnis.ergebnisse_der_fokusphase[A].programm is not None


def test_endet_das_verfahren_nach_der_sondierung_gibt_es_keine_fokusphase(protokoll):
    vorgaben = {name: Vorgabe(BASIS, ausfuehrungskosten=ZU_TEUER) for name in (A, B, C, D)}

    ergebnis, _ = durchlaufe(vorgaben, protokoll)

    assert ergebnis.ergebnisse_der_fokusphase == {}
    assert ergebnis.fuehrender_ohne_kostenfilter is None
