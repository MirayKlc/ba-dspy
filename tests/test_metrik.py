"""Tests zur Metrik.

Exact Match und F1 sind reine Funktionen: kein Netz, kein Sprachmodell, keine Zufallszahlen.
Deshalb sind sie hier als Tabellentests abgedeckt - je Zeile ein Fall mit von Hand
ausgerechnetem Sollwert, damit der Test nicht dieselbe Rechnung wie der Code anstellt.
"""

import dspy
import pytest

from dspy_optimizer.metrik import (
    Bewertung,
    absatztreffer,
    bewerte,
    bewertung_von,
    exakte_uebereinstimmung,
    f1,
    zusammenfassung,
)


@pytest.mark.parametrize(
    "vorhergesagt,richtig,erwartet",
    [
        ("Paris", "Paris", 1.0),
        ("paris", "PARIS", 1.0),
        ("Paris.", "Paris", 1.0),
        ("the Eiffel Tower", "Eiffel Tower", 1.0),
        ("  Paris  ", "Paris", 1.0),
        ("Paris", "London", 0.0),
        ("Paris France", "Paris", 0.0),
        ("", "Paris", 0.0),
        ("...", "Paris", 0.0),
        ("yes", "no", 0.0),
    ],
)
def test_exakte_uebereinstimmung_vergleicht_auf_der_normalisierten_form(
    vorhergesagt, richtig, erwartet
):
    assert exakte_uebereinstimmung(vorhergesagt, richtig) == erwartet


@pytest.mark.parametrize(
    "vorhergesagt,richtig,erwartet",
    [
        # Volltreffer und Totalausfall
        ("Paris", "Paris", 1.0),
        ("the Eiffel Tower", "Eiffel Tower.", 1.0),
        ("Paris", "London", 0.0),
        # Teiltreffer, von Hand gerechnet:
        # "cat sat" gegen "cat sat on mat": 2 gemeinsame Woerter,
        # Genauigkeit 2/2, Trefferquote 2/4, F1 = 2*1*0.5/1.5
        ("the cat sat", "cat sat on the mat", 2 / 3),
        # "Steve Jobs" gegen "Jobs": 1 gemeinsames Wort, 1/2 und 1/1
        ("Steve Jobs", "Jobs", 2 / 3),
        # Woerter zaehlen einzeln: "cat cat" gegen "cat" hat *ein* gemeinsames Wort,
        # nicht zwei - sonst kaeme eine Genauigkeit ueber 1 heraus.
        ("cat cat", "cat", 2 / 3),
        # Leere und satzzeichenleere Antworten geben nichts her, statt zu krachen
        ("", "Paris", 0.0),
        ("...", "Paris", 0.0),
        ("Paris", "", 0.0),
        ("", "", 0.0),
    ],
)
def test_f1_ist_der_wortweise_ueberlapp_der_normalisierten_antworten(
    vorhergesagt, richtig, erwartet
):
    assert f1(vorhergesagt, richtig) == pytest.approx(erwartet)


def beispiel_mit(goldene=("Absatz A", "Absatz B")):
    return dspy.Example(
        question="Wer?",
        answer="Paris",
        paragraphs=[{"title": t, "sentences": ["Satz."]} for t in ["Absatz A", "Absatz B", "C"]],
        gold_titles=list(goldene),
    ).with_inputs("question", "paragraphs")


@pytest.mark.parametrize(
    "gewaehlte_titel,erwartet",
    [
        (["Absatz A", "Absatz B"], 1.0),
        (["Absatz B", "Absatz A"], 1.0),
        (["Absatz A", "C"], 0.5),
        (["C"], 0.0),
        ([], 0.0),
        # Beide goldenen dabei, plus ein Ablenker: die Trefferquote bleibt voll. Dass Modul 1
        # zu viel mitnimmt, ist ein Kostenproblem, kein Treffer-Problem.
        (["Absatz A", "Absatz B", "C"], 1.0),
        # Zweimal derselbe Treffer bleibt ein Treffer
        (["Absatz A", "Absatz A"], 0.5),
        # Gross-/Kleinschreibung und Satzzeichen duerfen nicht ueber einen Treffer entscheiden
        (["absatz a.", "ABSATZ B"], 1.0),
    ],
)
def test_absatztreffer_ist_der_anteil_gefundener_goldener_absaetze(gewaehlte_titel, erwartet):
    vorhersage = dspy.Prediction(answer="Paris", relevant_titles=gewaehlte_titel, relevant_text="")

    assert absatztreffer(beispiel_mit(), vorhersage) == pytest.approx(erwartet)


def test_absatztreffer_ist_unbekannt_wenn_die_vorhersage_keine_titel_nennt():
    """Ohne Titel gibt es nichts zu protokollieren - und das darf die Bewertung nicht stoppen."""
    assert absatztreffer(beispiel_mit(), dspy.Prediction(answer="Paris")) is None


def test_die_optimierung_bewertet_die_antwort_und_nicht_die_absatzwahl():
    """Der Rueckgabewert der Metrik ist der F1 der Antwort - sonst nichts."""
    richtige_antwort_falsche_absaetze = dspy.Prediction(answer="Paris", relevant_titles=["C"])
    falsche_antwort_richtige_absaetze = dspy.Prediction(
        answer="London", relevant_titles=["Absatz A", "Absatz B"]
    )

    assert bewerte(beispiel_mit(), richtige_antwort_falsche_absaetze) == 1.0
    assert bewerte(beispiel_mit(), falsche_antwort_richtige_absaetze) == 0.0


def test_die_ausfuehrliche_bewertung_haelt_exakt_f1_und_absatztreffer_nebeneinander():
    vorhersage = dspy.Prediction(answer="Paris, France", relevant_titles=["Absatz A"])

    einzeln = bewertung_von(beispiel_mit(), vorhersage)

    assert einzeln.exakt == 0.0
    assert einzeln.f1 == pytest.approx(2 / 3)
    assert einzeln.absatztreffer == 0.5


def test_eine_vorhersage_ohne_antwort_zaehlt_als_null_statt_zu_krachen():
    """Ein Formatfehler eines Kandidatenmodells kostet Punkte, aber nicht den Lauf."""
    assert bewerte(beispiel_mit(), dspy.Prediction()) == 0.0
    assert bewertung_von(beispiel_mit(), dspy.Prediction(answer=None)).f1 == 0.0


def test_die_zusammenfassung_mittelt_jede_groesse_fuer_sich():
    einzelne = [
        Bewertung(exakt=1.0, f1=1.0, absatztreffer=1.0),
        Bewertung(exakt=0.0, f1=0.5, absatztreffer=0.0),
    ]

    assert zusammenfassung(einzelne) == Bewertung(exakt=0.5, f1=0.75, absatztreffer=0.5)


def test_die_zusammenfassung_mittelt_den_absatztreffer_nur_ueber_bekannte_werte():
    """Ein unbekannter Treffer ist keine Null - sonst redete man die Protokollgroesse klein."""
    einzelne = [
        Bewertung(exakt=1.0, f1=1.0, absatztreffer=1.0),
        Bewertung(exakt=1.0, f1=1.0, absatztreffer=None),
    ]

    assert zusammenfassung(einzelne).absatztreffer == 1.0


def test_die_zusammenfassung_von_gar_nichts_ist_null_und_kein_fehler():
    assert zusammenfassung([]) == Bewertung(exakt=0.0, f1=0.0, absatztreffer=None)


def test_beim_sammeln_von_few_shot_beispielen_zaehlt_nur_die_exakte_antwort():
    """Mit `trace` sammelt DSPy Few-Shot-Beispiele - und nimmt dort jeden Wert ueber 0 als
    gelungen an (`dspy/teleprompt/bootstrap.py` Z. 206-209, ohne `metric_threshold`). Ein F1
    von 2/3 wuerde so eine *falsche* Antwort als Vorbild in den Prompt schreiben.
    """
    fast_richtig = dspy.Prediction(answer="Paris, France")
    genau_richtig = dspy.Prediction(answer="Paris")
    spur = [("ein Praediktor", {}, {})]

    assert bewerte(beispiel_mit(), genau_richtig, trace=spur) is True
    assert bewerte(beispiel_mit(), fast_richtig, trace=spur) is False
    # Ohne Spur wird bewertet, nicht gesammelt: dann bleibt es der stetige F1.
    assert bewerte(beispiel_mit(), fast_richtig) == pytest.approx(2 / 3)
