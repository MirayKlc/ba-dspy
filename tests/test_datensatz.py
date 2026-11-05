"""Tests zum Datensatz-Lader.

Geprueft wird `aufteilung_aus`: die reine Funktion, die aus zwei Rohdaten-Teilen die feste
Aufteilung erzeugt. Sie braucht kein Netz, deshalb laufen diese Tests ohne Download. Nur der
letzte Test fasst die echten Daten an und wird uebersprungen, wenn sie nicht greifbar sind.
"""

import hashlib
import random

import pytest

from dspy_optimizer.datensatz import (
    GROESSEN,
    MISCHUNG_SEED,
    aufteilung_aus,
    lade_aufteilung,
    zeige,
)


def rohbeispiel(kennung):
    """Ein Beispiel in genau der Form, in der Hugging Face HotPotQA/distractor liefert."""
    titel = [f"{kennung}-Absatz-{i}" for i in range(10)]
    return {
        "id": kennung,
        "question": f"Frage zu {kennung}?",
        "answer": f"Antwort {kennung}",
        "type": "bridge",
        "level": "hard",
        "supporting_facts": {"title": titel[:2], "sent_id": [0, 0]},
        "context": {
            "title": titel,
            "sentences": [[f"Erster Satz von {t}.", f"Zweiter Satz von {t}."] for t in titel],
        },
    }


def rohteil(praefix, anzahl=1000):
    return [rohbeispiel(f"{praefix}-{i}") for i in range(anzahl)]


def kennungen(teil):
    return [beispiel.id for beispiel in teil]


def fingerabdruck(teil):
    """Die Kennungen eines Teils, in ihrer Reihenfolge, zu einer kurzen Pruefsumme verdichtet."""
    return hashlib.sha256(",".join(kennungen(teil)).encode()).hexdigest()[:16]


@pytest.fixture
def aufteilung():
    return aufteilung_aus(rohteil("train"), rohteil("dev"))


def test_aufteilung_hat_die_festgelegten_groessen(aufteilung):
    assert [len(teil) for teil in aufteilung] == [200, 200, 200, 500]
    assert GROESSEN == (200, 200, 200, 500)


def test_die_auswahldaten_aendern_keinen_der_anderen_teile(aufteilung):
    """Trainings-, Validierungs- und Testdaten sind genau dieselben Fragen in derselben
    Reihenfolge wie in der Aufteilung ohne Auswahldaten - sonst waeren Handlaeufe und Lasttest
    nicht vergleichbar. Die Pruefsummen stammen aus der Aufteilung ohne Auswahldaten."""
    assert fingerabdruck(aufteilung.trainingsdaten) == "6d110ac4d41f29d3"
    assert fingerabdruck(aufteilung.validierungsdaten) == "751f6ae0fbe94cb8"
    assert fingerabdruck(aufteilung.testdaten) == "14e63628154d93b8"


def test_die_auswahldaten_sind_das_dritte_stueck_derselben_ziehung():
    """Nach den ersten 200 (Training) und den zweiten 200 (Validierung) die naechsten 200 aus
    dem Trainingsteil - nicht aus dem Entwicklungsteil, aus dem die Testdaten kommen."""
    aufteilung = aufteilung_aus(rohteil("train"), rohteil("dev"))

    # Hier taugt jedes Rohbeispiel, die Ziehung ist also genau die gemischte Reihenfolge.
    reihenfolge = list(range(1000))
    random.Random(MISCHUNG_SEED).shuffle(reihenfolge)
    ziehung = [f"train-{i}" for i in reihenfolge]
    assert kennungen(aufteilung.trainingsdaten) == ziehung[:200]
    assert kennungen(aufteilung.validierungsdaten) == ziehung[200:400]
    assert kennungen(aufteilung.auswahldaten) == ziehung[400:600]


def test_aufteilung_ist_bei_jedem_aufruf_dieselbe(aufteilung):
    zweite = aufteilung_aus(rohteil("train"), rohteil("dev"))

    for links, rechts in zip(aufteilung, zweite):
        assert kennungen(links) == kennungen(rechts)


def test_aufteilung_haengt_nicht_vom_seed_des_verfahrens_ab():
    random.seed(1)
    erste = aufteilung_aus(rohteil("train"), rohteil("dev"))
    random.seed(4711)
    zweite = aufteilung_aus(rohteil("train"), rohteil("dev"))

    for links, rechts in zip(erste, zweite):
        assert kennungen(links) == kennungen(rechts)


def test_gezogen_wird_aus_dem_ganzen_teil_nicht_nur_vom_anfang():
    roh_trainingsteil = rohteil("train")
    aufteilung = aufteilung_aus(roh_trainingsteil, rohteil("dev"))

    anfang = [roh["id"] for roh in roh_trainingsteil[:400]]
    assert kennungen(aufteilung.trainingsdaten) + kennungen(aufteilung.validierungsdaten) != anfang


def test_die_vier_teile_ueberschneiden_sich_nicht(aufteilung):
    assert len({beispiel.id for teil in aufteilung for beispiel in teil}) == 1100


def test_kein_beispiel_taucht_in_beiden_rohdaten_teilen_doppelt_auf():
    # Beide Teile tragen hier dieselben Kennungen: die Aufteilung muss trotzdem
    # ueberschneidungsfrei bleiben und darf sich nicht auf die Herkunft verlassen.
    gleicher_teil = rohteil("beide", 1100)

    aufteilung = aufteilung_aus(gleicher_teil, gleicher_teil)

    assert len({beispiel.id for teil in aufteilung for beispiel in teil}) == 1100


def test_jedes_beispiel_traegt_frage_antwort_absaetze_und_goldene_titel(aufteilung):
    for teil in aufteilung:
        for beispiel in teil:
            assert beispiel.question and beispiel.answer
            assert len(beispiel.paragraphs) == 10
            assert all(absatz["sentences"] for absatz in beispiel.paragraphs)
            assert len(beispiel.gold_titles) == 2
            titel = [absatz["title"] for absatz in beispiel.paragraphs]
            assert set(beispiel.gold_titles) <= set(titel)


def test_frage_und_absaetze_sind_die_eingaben_die_antwort_nicht(aufteilung):
    beispiel = aufteilung.trainingsdaten[0]

    assert set(beispiel.inputs().keys()) == {"question", "paragraphs"}
    assert "answer" not in beispiel.inputs()


def ohne_zehnten_absatz(roh):
    roh["context"]["title"].pop()
    roh["context"]["sentences"].pop()


def nur_ein_goldener_absatz(roh):
    roh["supporting_facts"] = {"title": roh["supporting_facts"]["title"][:1], "sent_id": [0]}


def goldener_absatz_fehlt_im_kontext(roh):
    roh["supporting_facts"]["title"][0] = "Absatz, den es gar nicht gibt"


def mit_leerem_absatz(roh):
    roh["context"]["sentences"][3] = []


def ohne_antwort(roh):
    roh["answer"] = "  "


def leichte_frage(roh):
    roh["level"] = "easy"


@pytest.mark.parametrize(
    "kaputt_machen",
    [
        ohne_zehnten_absatz,
        nur_ein_goldener_absatz,
        goldener_absatz_fehlt_im_kontext,
        mit_leerem_absatz,
        ohne_antwort,
        leichte_frage,
    ],
    ids=lambda mangel: mangel.__name__,
)
def test_untaugliche_beispiele_werden_uebersprungen(kaputt_machen):
    untauglich = rohteil("untauglich", 500)
    for roh in untauglich:
        kaputt_machen(roh)
    tauglich = rohteil("tauglich", 700)

    aufteilung = aufteilung_aus(untauglich + tauglich, rohteil("dev"))

    gezogen = (
        kennungen(aufteilung.trainingsdaten)
        + kennungen(aufteilung.validierungsdaten)
        + kennungen(aufteilung.auswahldaten)
    )
    assert all(kennung.startswith("tauglich") for kennung in gezogen)


def test_zu_wenige_brauchbare_beispiele_sind_ein_fehler():
    with pytest.raises(ValueError, match="599"):
        aufteilung_aus(rohteil("train", 599), rohteil("dev"))


def test_die_ansicht_zeigt_frage_antwort_und_markiert_die_goldenen_absaetze(aufteilung):
    beispiel = aufteilung.trainingsdaten[0]

    text = zeige(beispiel)

    assert beispiel.question in text and beispiel.answer in text
    markiert = {
        zeile.split(":")[0].removeprefix("  * ")
        for zeile in text.splitlines()
        if zeile.startswith("  * ")
    }
    assert markiert == set(beispiel.gold_titles)
    assert all(absatz["title"] in text for absatz in beispiel.paragraphs)


def test_der_echte_datensatz_passt_zum_lader():
    """Faengt ab, dass Hugging Face Felder oder Teile umbenennt.

    Braucht den Datensatz im Cache oder eine Netzverbindung; sonst uebersprungen.
    """
    try:
        aufteilung = lade_aufteilung()
    except Exception as fehler:  # noqa: BLE001 - jeder Grund ist hier ein Grund zu ueberspringen
        pytest.skip(f"HotPotQA nicht ladbar: {type(fehler).__name__}: {fehler}")

    assert [len(teil) for teil in aufteilung] == [200, 200, 200, 500]
    assert len({beispiel.id for teil in aufteilung for beispiel in teil}) == 1100
    # Dieselben Fragen wie in der Aufteilung ohne Auswahldaten; deren Pruefsummen.
    assert fingerabdruck(aufteilung.trainingsdaten) == "e910af1a022a229f"
    assert fingerabdruck(aufteilung.validierungsdaten) == "280b5593cd7edf0b"
    assert fingerabdruck(aufteilung.testdaten) == "1d3b4ba933a51533"
    beispiel = aufteilung.trainingsdaten[0]
    assert len(beispiel.paragraphs) == 10
    assert len(beispiel.gold_titles) == 2
