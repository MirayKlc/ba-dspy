"""Tests zum Kostenmodell.

Kein einziger API-Aufruf: das Kostenmodell rechnet, es misst nicht. Gemessen wird in
`kostenerfassung`; hier wird nur geprueft, dass aus einem Dollar-Betrag eine Zahl
`num_trials` wird, aus ihr wieder derselbe Betrag, und dass die Planabweichung hinterher in
der Protokollzeile steht.

Die Zahlen der Attrappe sind rund gewaehlt und nicht die des Experiments: das Kostenmodell
traegt keine eigenen Werte, es bekommt sie.
"""

import pytest

from dspy_optimizer.kostenerfassung import messhygiene, messlauf
from dspy_optimizer.kostenmodell import (
    BudgetReichtNicht,
    Kostenmodell,
    Messung,
    Parameter,
    Plan,
    kalibriere,
)
from dspy_optimizer.protokoll import HANDLAUF, Einordnung, Protokoll

HERKUNFT = "Vorstudie vom 20.09.2026, drei Laeufe mit openai/gpt-oss-120b"
KANDIDATENMODELL = "openai/gpt-oss-120b"

# Ein Trial kostet 0,01 $, die Fixkosten 0,40 $. Die Nachbewertung ueber 20
# Auswahlbeispiele zu je 0,0005 $ kostet 0,01 $. 0,51 $ reichen damit fuer genau zehn
# Trials samt Nachbewertung.
FIXKOSTEN = 0.40
KOSTEN_JE_TRIAL = 0.01
AUSFUEHRUNGSKOSTEN = 0.0005
AUSWAHLBEISPIELE = 20
NACHBEWERTUNG = 0.01


def kostenmodell_mit(
    fixkosten=FIXKOSTEN, kosten_je_trial=KOSTEN_JE_TRIAL, ausfuehrungskosten=AUSFUEHRUNGSKOSTEN
):
    return Kostenmodell(
        kandidatenmodell=KANDIDATENMODELL,
        fixkosten=Parameter(fixkosten, HERKUNFT),
        kosten_je_trial=Parameter(kosten_je_trial, HERKUNFT),
        ausfuehrungskosten=Parameter(ausfuehrungskosten, HERKUNFT),
    )


@pytest.fixture
def kostenmodell():
    return kostenmodell_mit()


def test_ein_dollar_betrag_wird_zu_einer_zahl_num_trials(kostenmodell):
    """Der Hinweg: der Budgetanteil geht in den einen Stellhebel ueber."""
    assert kostenmodell.num_trials_fuer(0.51, AUSWAHLBEISPIELE) == 10


def test_eine_zahl_num_trials_wird_zu_einem_dollar_betrag(kostenmodell):
    """Der Rueckweg: Fixkosten + num_trials * Kosten je Trial + Nachbewertung."""
    assert kostenmodell.dollar_fuer(10, AUSWAHLBEISPIELE) == pytest.approx(0.51)
    assert kostenmodell.dollar_fuer(1, AUSWAHLBEISPIELE) == pytest.approx(0.42)


def test_die_nachbewertung_wird_vom_budgetanteil_abgezogen(kostenmodell):
    """Der Budgetanteil bezahlt die Nachbewertung.

    Ohne Abzug reichten 0,50 $ fuer zehn Trials; weil die Nachbewertung ueber 20 Beispiele
    0,01 $ kostet, bleiben nur neun - der zehnte ginge ueber den Budgetanteil hinaus.
    """
    assert kostenmodell.dollar_nachbewertung(AUSWAHLBEISPIELE) == pytest.approx(
        NACHBEWERTUNG
    )
    assert kostenmodell.num_trials_fuer(0.50, AUSWAHLBEISPIELE) == 9


def test_mehr_auswahlbeispiele_kosten_trials(kostenmodell):
    """Die Nachbewertung waechst mit den Auswahlbeispielen, und was sie mehr kostet,
    fehlt der Suche: 200 Beispiele kosten 0,10 $, das sind neun Trials weniger als 20."""
    assert kostenmodell.num_trials_fuer(0.60, 20) == 19
    assert kostenmodell.num_trials_fuer(0.60, 200) == 10


def test_der_plan_rechnet_ohne_sicherheitsaufschlag(kostenmodell):
    """Geschaetzt wird nach bestem Wissen, nicht mit Puffer: genau Ausfuehrungskosten mal
    Zahl der Auswahlbeispiele, 0,0005 $ mal 20."""
    plan = kostenmodell.plane(0.51, AUSWAHLBEISPIELE)

    assert plan.dollar_nachbewertung == pytest.approx(0.01)
    assert plan.dollar_optimierung == pytest.approx(0.50)
    assert plan.dollar_gesamt == pytest.approx(0.51)


@pytest.mark.parametrize("num_trials", [1, 2, 7, 13, 50, 137])
def test_der_rundlauf_schliesst_sich_von_den_trials_aus(kostenmodell, num_trials):
    """Hin und zurueck ergibt dieselbe Zahl - auch wenn die Division nicht aufgeht.

    Ohne diese Zusicherung landete ein Betrag, der rechnerisch genau fuer n Trials reicht,
    wegen eines Rundungsfehlers in der letzten Nachkommastelle bei n-1: das Verfahren
    verschenkte dann bei jeder Umrechnung einen Trial. Die Nachbewertung steckt in beiden
    Richtungen mit drin.
    """
    dollar = kostenmodell.dollar_fuer(num_trials, AUSWAHLBEISPIELE)

    assert kostenmodell.num_trials_fuer(dollar, AUSWAHLBEISPIELE) == num_trials


@pytest.mark.parametrize("budgetanteil", [0.42, 0.51, 0.519, 1.00, 2.37, 13.0])
def test_der_rundlauf_schliesst_sich_von_den_dollar_aus(kostenmodell, budgetanteil):
    """Die Richtung, die das Verfahren geht: Betrag rein, Trials raus, Kosten zurueck.

    Der Plan bleibt samt Nachbewertung im zugeteilten Betrag, und was uebrig bleibt, reicht
    fuer keinen weiteren Trial - sonst waere entweder Geld verschenkt oder mehr verplant als
    da ist.
    """
    plan = kostenmodell.plane(budgetanteil, AUSWAHLBEISPIELE)

    assert (
        kostenmodell.num_trials_fuer(plan.dollar_gesamt, AUSWAHLBEISPIELE)
        == plan.num_trials
    )
    assert plan.dollar_gesamt <= budgetanteil * (1 + 1e-9)
    assert budgetanteil - plan.dollar_gesamt < KOSTEN_JE_TRIAL


def test_ein_rest_der_nicht_fuer_einen_trial_reicht_wird_abgerundet(kostenmodell):
    """Geplant wird, was bezahlbar ist: 0,519 $ reichen fuer zehn Trials, nicht fuer elf."""
    assert kostenmodell.num_trials_fuer(0.519, AUSWAHLBEISPIELE) == 10
    assert kostenmodell.plane(0.519, AUSWAHLBEISPIELE).dollar_gesamt == pytest.approx(
        0.51
    )


def test_das_knappste_ausreichende_budget_ergibt_genau_einen_trial(kostenmodell):
    """Die Untergrenze selbst geht noch durch - sie ist erreichbar, nicht verboten."""
    mindestbudget = kostenmodell.mindestbudget(AUSWAHLBEISPIELE)

    assert mindestbudget == pytest.approx(0.42)  # Fixkosten + ein Trial + Nachbewertung
    assert kostenmodell.num_trials_fuer(mindestbudget, AUSWAHLBEISPIELE) == 1


def test_ein_zu_kleiner_budgetanteil_ist_ein_fehler_und_nicht_null_trials(kostenmodell):
    """Ein Lauf mit num_trials = 0 waere ein bezahlter Lauf ohne jede Suche.

    Er kostet die Fixkosten - Beispiele sammeln, Instruktionen vorschlagen, eine erste
    vollstaendige Bewertung - und probiert danach nichts mehr aus. Deshalb bricht das
    Kostenmodell ab, statt eine Null zurueckzugeben, die weiter unten niemandem auffiele.
    """
    # Eine eigene Klasse, damit das Hauptexperiment genau diesen Fall erkennt.
    with pytest.raises(BudgetReichtNicht) as fehler:
        kostenmodell.num_trials_fuer(0.405, AUSWAHLBEISPIELE)

    assert "0.405000" in str(fehler.value)
    assert "0.420000" in str(fehler.value)  # das noetige Mindestbudget


def test_das_genannte_mindestbudget_enthaelt_die_nachbewertung(kostenmodell):
    """Fuer die Optimierung allein reichten 0,415 $ - fuer einen Trial samt Nachbewertung
    nicht. Die Meldung nennt die Nachbewertung, damit niemand den Budgetanteil nur um die
    Optimierung erhoeht und am naechsten Fehler haengen bleibt."""
    with pytest.raises(ValueError) as fehler:
        kostenmodell.plane(0.415, AUSWAHLBEISPIELE)

    assert "0.420000" in str(fehler.value)
    assert "Nachbewertung" in str(fehler.value)
    assert "0.010000" in str(fehler.value)


def test_ein_budgetanteil_unter_den_fixkosten_ist_derselbe_fehler(kostenmodell):
    with pytest.raises(ValueError):
        kostenmodell.plane(0.01, AUSWAHLBEISPIELE)


def test_ein_lauf_ohne_trials_hat_keine_kosten_sondern_gibt_es_nicht(kostenmodell):
    """Dieselbe Grenze in beide Richtungen, sonst schloesse sich der Rundlauf an ihrem Rand
    nicht: `dollar_fuer(0)` haette eine Antwort, die `num_trials_fuer` verweigert."""
    with pytest.raises(ValueError):
        kostenmodell.dollar_fuer(0, AUSWAHLBEISPIELE)


def test_eine_nachbewertung_ueber_keine_beispiele_gibt_es_nicht(kostenmodell):
    """Ohne Auswahlbeispiele gibt es keine Einzelbewertungen und keinen gepaarten Test -
    geplant werden kann so ein Lauf nicht."""
    with pytest.raises(ValueError):
        kostenmodell.plane(0.51, 0)


def test_ein_kostenmodell_ohne_kosten_je_trial_sagt_das_statt_durch_null_zu_teilen():
    """Die Parameter sind von aussen setzbar - also auch falsch setzbar.

    Ohne diese Pruefung faende der Fehler sich als `ZeroDivisionError` mitten in der
    Budgetplanung wieder, statt als Satz ueber das Kostenmodell.
    """
    ohne_kosten = kostenmodell_mit()._replace(kosten_je_trial=Parameter(0.0, "vergessen"))

    with pytest.raises(ValueError) as fehler:
        ohne_kosten.num_trials_fuer(1.00, AUSWAHLBEISPIELE)

    assert "vergessen" in str(fehler.value)


def test_ein_kostenmodell_mit_kostenloser_nachbewertung_ist_ein_fehler():
    """Eine vergessene Null bei den Ausfuehrungskosten waere kein Rechenfehler, sondern ein
    still zu grosser Plan: jeder Lauf ginge um die ganze Nachbewertung ueber seinen Budgetanteil."""
    ohne_nachbewertung = kostenmodell_mit()._replace(ausfuehrungskosten=Parameter(0.0, "vergessen"))

    with pytest.raises(ValueError, match="vergessen"):
        ohne_nachbewertung.plane(1.00, AUSWAHLBEISPIELE)


def test_die_parameter_sind_von_aussen_setzbar_und_tragen_ihre_herkunft(kostenmodell):
    """Jede gemessene Zahl im Modell muss sagen koennen, woher sie kommt.

    Die im DSPy-Quelltext hinterlegte Formel ist toter Code und rechnet falsch; was hier
    steht, stammt aus Messungen - und das steht an der Zahl selbst, nicht in einem
    Kommentar daneben. Das Kostenmodell gilt je Kandidatenmodell und sagt, fuer welches.
    """
    assert kostenmodell.fixkosten.herkunft == HERKUNFT
    assert kostenmodell.kosten_je_trial.herkunft == HERKUNFT
    assert kostenmodell.ausfuehrungskosten.herkunft == HERKUNFT
    assert kostenmodell.als_felder() == {
        "kandidatenmodell": KANDIDATENMODELL,
        "fixkosten": FIXKOSTEN,
        "fixkosten_herkunft": HERKUNFT,
        "kosten_je_trial": KOSTEN_JE_TRIAL,
        "kosten_je_trial_herkunft": HERKUNFT,
        "ausfuehrungskosten": AUSFUEHRUNGSKOSTEN,
        "ausfuehrungskosten_herkunft": HERKUNFT,
    }


def test_der_plan_weiss_fuer_welches_kandidatenmodell_und_welche_daten_er_gilt(
    kostenmodell,
):
    """Sonst plant man fuer ein Kandidatenmodell und bezahlt den Lauf eines anderen, und die
    Planabweichung im Protokoll ist Unsinn, ohne dass es auffaellt."""
    plan = kostenmodell.plane(0.51, AUSWAHLBEISPIELE)

    assert plan.kandidatenmodell == KANDIDATENMODELL
    assert plan.auswahlbeispiele == AUSWAHLBEISPIELE


def test_fuer_denselben_betrag_bekommt_das_billigere_kandidatenmodell_mehr_trials():
    """Der beim Budget in US-Dollar bewusst in Kauf genommene Effekt, hier festgehalten.

    Gleiche Dollar, nicht gleiche Suche: ein zehnmal billigeres Kandidatenmodell kauft von
    demselben Betrag deutlich mehr Trials. Genau das soll die Sondierungsphase tun.
    """
    billig = kostenmodell_mit(fixkosten=0.10, kosten_je_trial=0.001, ausfuehrungskosten=0.00005)
    teuer = kostenmodell_mit(fixkosten=0.10, kosten_je_trial=0.01, ausfuehrungskosten=0.0005)

    # (1,00 - 0,10 - 0,001) / 0,001 und (1,00 - 0,10 - 0,01) / 0,01
    assert billig.num_trials_fuer(1.00, AUSWAHLBEISPIELE) == 899
    assert teuer.num_trials_fuer(1.00, AUSWAHLBEISPIELE) == 89


def test_die_abweichung_zwischen_plan_und_ist_wird_beziffert(kostenmodell):
    """Das Verfahren ist budget-planend, nicht budget-regelnd.

    Es kann einen laufenden Lauf nicht anhalten. Die Planabweichung ist deshalb kein
    Fehler, den es zu vermeiden gilt, sondern selbst ein Messergebnis. Sie gilt fuer
    jeden der beiden Messlaeufe gegen seinen eigenen Teil des Plans.
    """
    plan = kostenmodell.plane(0.51, AUSWAHLBEISPIELE)
    optimierung = plan.abweichung("optimierung", 0.55)
    nachbewertung = plan.abweichung("nachbewertung", 0.012)

    assert plan.num_trials == 10
    assert optimierung.geplant == pytest.approx(0.50)
    assert optimierung.differenz == pytest.approx(0.05)
    assert optimierung.anteil == pytest.approx(0.1)
    assert nachbewertung.geplant == pytest.approx(0.01)
    assert nachbewertung.differenz == pytest.approx(0.002)
    assert nachbewertung.anteil == pytest.approx(0.2)


def test_ein_lauf_unter_plan_weist_eine_negative_abweichung_aus(kostenmodell):
    plan = kostenmodell.plane(0.51, AUSWAHLBEISPIELE)

    abweichung = plan.abweichung("optimierung", 0.45)

    assert abweichung.differenz == pytest.approx(-0.05)
    assert abweichung.anteil == pytest.approx(-0.1)


def test_fuer_einen_ungeplanten_abschnitt_gibt_es_keine_abweichung(kostenmodell):
    """Eine Testbewertung bezahlt kein Budgetanteil - gegen welchen Plan sollte sie rechnen?"""
    plan = kostenmodell.plane(0.51, AUSWAHLBEISPIELE)

    with pytest.raises(ValueError, match="testbewertung"):
        plan.abweichung("testbewertung", 0.01)


def test_ein_plan_ueber_null_dollar_reisst_die_protokollzeile_nicht_mit(kostenmodell):
    """`plane()` erzeugt so einen Plan nie - von Hand gebaut ist er trotzdem moeglich.

    Die Zeile entsteht am Ende eines bezahlten Laufs: ein `ZeroDivisionError` beim
    Ausrechnen des Prozentwerts naehme die einzige Aufzeichnung mit, die der Lauf
    hinterlaesst.
    """
    leer = Plan(
        num_trials=0,
        auswahlbeispiele=AUSWAHLBEISPIELE,
        budgetanteil=0.0,
        dollar_optimierung=0.0,
        dollar_nachbewertung=0.0,
        kostenmodell=kostenmodell,
    )

    abweichung = leer.abweichung("optimierung", 0.02)

    assert abweichung.differenz == pytest.approx(0.02)
    assert abweichung.anteil is None
    assert abweichung.als_felder()["dollar_abweichung_anteil"] is None


def test_die_abweichung_und_die_herkunft_stehen_in_der_protokollzeile(tmp_path, kostenmodell):
    """Je Lauf ausgewiesen heisst: in der einen Zeile, die der Lauf hinterlaesst.

    Mitgeschrieben wird auch, mit welchen Parametern geplant wurde. Sonst liesse sich
    spaeter nicht mehr sagen, ob eine Abweichung am Lauf lag oder an einem inzwischen neu
    kalibrierten Modell.
    """
    messhygiene()
    protokoll = Protokoll(tmp_path / "protokoll.jsonl")

    with messlauf(
        protokollverzeichnis=tmp_path,
        abschnitt="optimierung",
        einordnung=Einordnung.experiment("verfahren", durchgang=1, phase="sondierung"),
        konfiguration={"kandidatenmodell": "attrappe", "num_trials": 10},
        plan=kostenmodell.plane(0.51, AUSWAHLBEISPIELE),
        protokoll=protokoll,
    ):
        pass  # kein Aufruf, kein Geld - die Zeile entsteht trotzdem

    (zeile,) = protokoll.zeilen()

    assert zeile["num_trials_geplant"] == 10
    assert zeile["konfiguration"]["num_trials"] == 10  # dieselbe Zahl, zwei Bedeutungen
    assert zeile["dollar_budgetanteil"] == pytest.approx(0.51)
    assert zeile["dollar_geplant"] == pytest.approx(0.50)  # nur die Optimierung
    assert zeile["dollar"] == 0.0
    assert zeile["dollar_abweichung"] == pytest.approx(-0.50)
    assert zeile["dollar_abweichung_anteil"] == pytest.approx(-1.0)
    assert zeile["kostenmodell"]["fixkosten_herkunft"] == HERKUNFT
    assert zeile["kostenmodell"]["kandidatenmodell"] == KANDIDATENMODELL


def test_die_nachbewertungszeile_traegt_ihren_teil_des_plans(tmp_path, kostenmodell):
    """Plan gegen Ist gilt fuer beide Messlaeufe eines Optimierungslaufs: die
    Nachbewertung wird geplant, also steht ihre Abweichung in ihrer eigenen Zeile."""
    messhygiene()
    protokoll = Protokoll(tmp_path / "protokoll.jsonl")

    with messlauf(
        protokollverzeichnis=tmp_path,
        abschnitt="nachbewertung",
        einordnung=Einordnung.experiment("verfahren", durchgang=1, phase="sondierung"),
        konfiguration={"kandidatenmodell": "attrappe", "num_trials": 10},
        plan=kostenmodell.plane(0.51, AUSWAHLBEISPIELE),
        protokoll=protokoll,
    ):
        pass

    (zeile,) = protokoll.zeilen()

    assert zeile["dollar_budgetanteil"] == pytest.approx(0.51)
    assert zeile["dollar_geplant"] == pytest.approx(0.01)
    assert zeile["dollar_abweichung"] == pytest.approx(-0.01)


def test_ein_plan_an_einem_ungeplanten_abschnitt_faellt_vor_dem_lauf_auf(
    tmp_path, kostenmodell
):
    """Beim Schreiben der Zeile waere es zu spaet: der Fehler risse die einzige Aufzeichnung
    eines bezahlten Laufs mit. Also bricht der Messlauf ab, bevor er beginnt."""
    messhygiene()
    protokoll = Protokoll(tmp_path / "protokoll.jsonl")
    begonnen = False

    with pytest.raises(ValueError, match="testbewertung"):
        with messlauf(
            protokollverzeichnis=tmp_path,
            abschnitt="testbewertung",
            einordnung=Einordnung.experiment("verfahren", durchgang=1, phase="fokus"),
            konfiguration={"kandidatenmodell": "attrappe"},
            plan=kostenmodell.plane(0.51, AUSWAHLBEISPIELE),
            protokoll=protokoll,
        ):
            begonnen = True

    assert not begonnen
    assert protokoll.zeilen() == []


def test_ohne_plan_bleibt_die_protokollzeile_wie_sie_war(tmp_path, kostenmodell):
    """Handlaeufe werden nicht geplant - sie duerfen keine Planfelder tragen."""
    messhygiene()
    protokoll = Protokoll(tmp_path / "protokoll.jsonl")

    with messlauf(
        protokollverzeichnis=tmp_path,
        abschnitt=None,
        einordnung=HANDLAUF,
        konfiguration={"kandidatenmodell": "attrappe"},
        protokoll=protokoll,
    ):
        pass

    (zeile,) = protokoll.zeilen()

    assert "dollar_geplant" not in zeile
    assert "num_trials_geplant" not in zeile


GEMESSENE_FIXKOSTEN = Parameter(0.40, "Vorstudie, Mittel aus drei Fixkosten-Messungen")


def aus_messungen(messungen, fixkosten=GEMESSENE_FIXKOSTEN):
    return kalibriere(
        messungen,
        herkunft=HERKUNFT,
        kandidatenmodell=KANDIDATENMODELL,
        fixkosten=fixkosten,
        ausfuehrungskosten=Parameter(AUSFUEHRUNGSKOSTEN, HERKUNFT),
    )


def test_die_kalibrierung_nimmt_die_gemessenen_fixkosten_und_findet_die_kosten_je_trial():
    """Die Fixkosten sind gemessen und kommen von aussen; aus den Optimierungslaeufen kommt nur
    noch die Steigung. Gemessen sind nur Optimierungskosten;
    die Nachbewertung kommt beim Planen dazu. Von Hand: ueber den Fixkosten liegen 0,10 $ bei
    10 und 0,20 $ bei 20 Trials, also 0,01 $ je Trial."""
    kostenmodell = aus_messungen([Messung(10, 0.50), Messung(20, 0.60)])

    assert kostenmodell.kandidatenmodell == KANDIDATENMODELL
    assert kostenmodell.fixkosten == GEMESSENE_FIXKOSTEN
    assert kostenmodell.kosten_je_trial.betrag == pytest.approx(0.01)
    assert kostenmodell.dollar_optimierung(30) == pytest.approx(0.70)
    assert kostenmodell.dollar_fuer(30, AUSWAHLBEISPIELE) == pytest.approx(0.71)


def test_die_steigung_ist_die_regression_durch_den_nullpunkt_ueber_den_fixkosten():
    """`linear_regression(T, d - F, proportional=True)`, ohne Aufschlag. Von Hand: ueber den
    Fixkosten 0,08 $ bei 10 und 0,22 $ bei 20 Trials;
    Steigung = (10 * 0,08 + 20 * 0,22) / (10^2 + 20^2) = 5,2 / 500 = 0,0104.
    Eine freie Regression legte die Gerade anders: Steigung 0,014, Achsenabschnitt 0,34 -
    die gemessenen Fixkosten gingen verloren."""
    kostenmodell = aus_messungen([Messung(10, 0.48), Messung(20, 0.62)])

    assert kostenmodell.fixkosten.betrag == pytest.approx(0.40)
    assert kostenmodell.kosten_je_trial.betrag == pytest.approx(0.0104)


def test_die_kalibrierung_nennt_ihre_grundlage_als_herkunft():
    """Die Herkunft wandert in jede Protokollzeile eines geplanten Laufs."""
    kostenmodell = aus_messungen([Messung(10, 0.50), Messung(20, 0.60)])

    assert kostenmodell.fixkosten.herkunft == GEMESSENE_FIXKOSTEN.herkunft
    assert HERKUNFT in kostenmodell.kosten_je_trial.herkunft
    assert "2 Optimierungslaeufe" in kostenmodell.kosten_je_trial.herkunft
    assert "Nullpunkt" in kostenmodell.kosten_je_trial.herkunft
    assert kostenmodell.ausfuehrungskosten.herkunft == HERKUNFT


def test_die_kalibrierung_braucht_zwei_verschiedene_trial_zahlen():
    """Rechnerisch reichte ein Lauf, aber erst mehrere zeigen, wie stark die Steigung
    streut."""
    with pytest.raises(ValueError) as fehler:
        aus_messungen([Messung(10, 0.50), Messung(10, 0.52)])

    assert "verschiedene" in str(fehler.value)


@pytest.mark.parametrize(
    "fixkosten",
    [None, Parameter(None, "alte Zeile ohne Feld"), Parameter(0.0, "x"), Parameter(-0.01, "x")],
)
def test_ohne_gemessene_fixkosten_ueber_null_wird_nicht_kalibriert(fixkosten):
    """Fixkosten vorhanden und groesser als 0. Alte Zeilen haben das Feld nicht, und eine
    Luecke darf nicht still als 0 durchgehen."""
    with pytest.raises(ValueError, match="Fixkosten"):
        aus_messungen([Messung(10, 0.50), Messung(20, 0.60)], fixkosten=fixkosten)


def test_die_kalibrierung_lehnt_eine_messreihe_ohne_kosten_je_trial_ab():
    """Steigung groesser als 0. Liegen die Laeufe unter den Fixkosten, kaeme eine negative
    Steigung heraus, und jeder Budgetanteil liesse sich in beliebig viele Trials uebersetzen."""
    with pytest.raises(ValueError, match="je Trial"):
        aus_messungen([Messung(10, 0.30), Messung(20, 0.35)])


def test_jeder_optimierungslauf_muss_mehr_kosten_als_die_fixkosten():
    """Ein Lauf mit Trials, der weniger kostet als sein fester Teil allein, passt nicht zu
    den gemessenen Fixkosten. Hier ist die Steigung trotzdem positiv - (10 * -0,01 + 20 * 0,30)
    / 500 -, die Pruefung muss also je Lauf greifen."""
    with pytest.raises(ValueError, match="10 Trials"):
        aus_messungen([Messung(10, 0.39), Messung(20, 0.70)])
