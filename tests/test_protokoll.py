"""Tests zum fortlaufenden Protokoll.

Das Protokoll ist die einzige Aufzeichnung, die einen Absturz ueberlebt. Nachtraeglich
laesst es sich nicht rekonstruieren: was nicht in dem Moment geschrieben wurde, in dem der
Lauf lief, ist weg. Entsprechend hart sind die Tests: nur anhaengen, sofort auf die Platte,
und eine halbe letzte Zeile darf die frueheren nicht mitreissen.
"""

import json

import pytest

from dspy_optimizer.kostenerfassung import Messwerte, Tokens
from dspy_optimizer.protokoll import (
    ABSCHNITTE,
    HANDLAUF,
    VORSTUDIE,
    Einordnung,
    Protokoll,
    verzeichnis_fuer_lauf,
)


def messwerte(dollar=0.25, programmlaeufe=20, fixkosten=None):
    return Messwerte(
        dollar=dollar,
        lm_aufrufe=2 * programmlaeufe,
        programmlaeufe=programmlaeufe,
        fehlgeschlagene_programmlaeufe=0,
        auswertungen=1,
        tokens=Tokens(prompt=1000, antwort=200),
        tokens_aus_evaluationen=Tokens(prompt=900, antwort=180),
        anbieter={"novita": 2 * programmlaeufe},
        stoerungen=(),
        fixkosten=fixkosten,
    )


@pytest.fixture
def protokoll(tmp_path):
    return Protokoll(tmp_path / "protokoll.jsonl")


def schreibe(protokoll, abschnitt="optimierung", einordnung=VORSTUDIE, **abweichend):
    felder = {
        "abschnitt": abschnitt,
        "einordnung": einordnung,
        "konfiguration": {"kandidatenmodell": "google/gemma-4-31b-it", "num_trials": 6},
        "seed": 0,
        "bewertung": 0.62,
        "messwerte": messwerte(),
    }
    return protokoll.schreibe(**{**felder, **abweichend})


def test_eine_zeile_je_lauf_mit_allen_geforderten_groessen(protokoll):
    """Konfiguration, Seed, Tokens, Dollar, Programmlaeufe, Bewertung, Anbieter, Zeitstempel."""
    schreibe(protokoll)

    (zeile,) = protokoll.zeilen()

    assert zeile["abschnitt"] == "optimierung"
    assert zeile["zweck"] == "vorstudie"
    assert zeile["konfiguration"]["kandidatenmodell"] == "google/gemma-4-31b-it"
    assert zeile["seed"] == 0
    assert zeile["dollar"] == 0.25
    assert zeile["tokens_prompt"] == 1000
    assert zeile["tokens_antwort"] == 200
    assert zeile["programmlaeufe"] == 20
    assert zeile["bewertung"] == 0.62
    assert zeile["anbieter"] == {"novita": 40}
    assert zeile["zeitpunkt"]


def test_optimierungskosten_und_ausfuehrungskosten_stehen_getrennt(protokoll):
    """Die beiden duerfen nie in einer Zahl zusammenfallen."""
    schreibe(
        protokoll, abschnitt="optimierung", messwerte=messwerte(dollar=1.20, programmlaeufe=600)
    )
    schreibe(
        protokoll, abschnitt="nachbewertung", messwerte=messwerte(dollar=0.04, programmlaeufe=200)
    )

    optimierung, nachbewertung = protokoll.zeilen()

    assert optimierung["dollar"] == 1.20
    assert nachbewertung["dollar"] == 0.04
    # Die Ausfuehrungskosten sind die Kosten *eines* Programmlaufs, nicht des Abschnitts.
    assert nachbewertung["dollar_je_programmlauf"] == pytest.approx(0.04 / 200)


def test_eine_optimierungszeile_fuehrt_keine_dollar_je_programmlauf(protokoll):
    """In einem Optimierungslauf liefen die gezaehlten Programmlaeufe auf Zwischenkandidaten,
    und die Laeufe des Bootstrappings fehlen in der Zahl ganz. Die Division waere
    weder eine Ausfuehrungskosten-Schaetzung noch sonst etwas Deutbares - und sie saehe der
    echten Messung der Nachbewertung zum Verwechseln aehnlich.

    Die Spalte bleibt trotzdem in jeder Zeile: die Auswertung soll eine Spalte lesen koennen,
    und was dort nicht steht, kann sie nicht missverstehen.
    """
    schreibe(
        protokoll, abschnitt="optimierung", messwerte=messwerte(dollar=1.20, programmlaeufe=600)
    )

    zeile = protokoll.zeilen()[0]

    assert "dollar_je_programmlauf" in zeile
    assert zeile["dollar_je_programmlauf"] is None
    # Die Summe des Abschnitts bleibt unangetastet - nur die Division entfaellt.
    assert zeile["dollar"] == 1.20
    assert zeile["programmlaeufe"] == 600


def test_exact_match_und_absatztreffer_stehen_neben_der_bewertung(protokoll):
    """Je Lauf drei Zahlen: F1, Exact Match und die Absatztreffer.

    Die Bewertung bleibt der F1 - an ihr haengt die Kalibrierregel des Kostenmodells. Die beiden
    anderen stehen daneben, in eigenen Spalten.
    """
    schreibe(protokoll, abschnitt="nachbewertung", exakt=0.4, absatztreffer=0.85)

    (zeile,) = protokoll.zeilen()
    assert zeile["bewertung"] == 0.62
    assert zeile["exakt"] == 0.4
    assert zeile["absatztreffer"] == 0.85


def test_ohne_exact_match_und_absatztreffer_bleiben_die_spalten_leer(protokoll):
    """Eine Optimierungszeile kennt nur MIPROv2s eigene Zahl, den F1.

    Die Spalten stehen trotzdem in jeder Zeile, wie `dollar_je_programmlauf`: die Auswertung
    soll eine Spalte lesen koennen.
    """
    schreibe(protokoll)

    (zeile,) = protokoll.zeilen()
    assert "exakt" in zeile and zeile["exakt"] is None
    assert "absatztreffer" in zeile and zeile["absatztreffer"] is None


def test_eine_zeile_traegt_abschnitt_und_zweck_statt_der_art(protokoll):
    """Aeltere Zeilen trugen ein einziges Feld `art`, das zwei Fragen vermischte: welcher Teil
    eines Laufs, und wozu er lief."""
    schreibe(protokoll, abschnitt="nachbewertung", einordnung=HANDLAUF)

    (zeile,) = protokoll.zeilen()
    assert zeile["abschnitt"] == "nachbewertung"
    assert zeile["zweck"] == "handlauf"
    assert "art" not in zeile
    # Die Felder des Experiments stehen in jeder Zeile, leer, wo sie nichts bedeuten - wie
    # `dollar_je_programmlauf`: die Auswertung soll eine Spalte lesen koennen.
    assert zeile["baseline"] is None
    assert zeile["phase"] is None
    assert zeile["durchgang"] is None


def test_die_abschnitte_tragen_die_festgelegten_namen():
    """Der zweite Messlauf eines Optimierungslaufs heisst Nachbewertung, nicht Ausfuehrung.
    Die Fixkosten-Messung ist ein eigener Abschnitt."""
    assert ABSCHNITTE == ("optimierung", "nachbewertung", "testbewertung", "fixkosten")


def test_die_fixkosten_stehen_nur_in_optimierung_und_fixkosten_messung(protokoll):
    """In einer Bewertung des fertigen Programms gibt es keinen festen Teil; dort steht
    `null`, auch wenn die Messwerte etwas anderes hergaeben."""
    for abschnitt in ("optimierung", "fixkosten", "nachbewertung"):
        schreibe(protokoll, abschnitt=abschnitt, messwerte=messwerte(fixkosten=0.08))

    optimierung, fixkosten, nachbewertung = protokoll.zeilen()
    assert optimierung["fixkosten"] == 0.08
    assert fixkosten["fixkosten"] == 0.08
    assert nachbewertung["fixkosten"] is None


def test_eine_fixkosten_messung_fuehrt_keine_dollar_je_programmlauf(protokoll):
    """Sonst saehe ihr Betrag geteilt durch die 200 Programmlaeufe der ersten Bewertung wie
    Ausfuehrungskosten aus."""
    schreibe(protokoll, abschnitt="fixkosten", messwerte=messwerte(fixkosten=0.08))

    (zeile,) = protokoll.zeilen()
    assert zeile["dollar_je_programmlauf"] is None


@pytest.mark.parametrize("einordnung", [HANDLAUF, Einordnung.experiment("ein-modell", durchgang=1)])
def test_eine_fixkosten_messung_gibt_es_nur_in_der_vorstudie(protokoll, einordnung):
    """Sie misst die Fixkosten fuer das Kostenmodell; ausserhalb der Vorstudie misst sie
    nichts, was jemand braucht."""
    with pytest.raises(ValueError, match="Vorstudie"):
        schreibe(protokoll, abschnitt="fixkosten", einordnung=einordnung)

    assert protokoll.zeilen() == []


def test_eine_fixkosten_messung_kalibriert_die_steigung_nicht(protokoll):
    """Sie traegt eine Bewertung - MIPROv2s eigene des unveraenderten Programms -, ist aber
    kein Messpunkt der Geraden: dort laege sie bei 0 Trials."""
    schreibe(protokoll, abschnitt="fixkosten", messwerte=messwerte(fixkosten=0.08))

    assert protokoll.kalibrierbare_zeilen() == []


def test_eine_zeile_des_verfahrens_traegt_baseline_phase_und_durchgang(protokoll):
    """Ohne sie weder Rangfolge zwischen den Phasen (FF1) noch Kosten je Durchgang (FF3)."""
    schreibe(
        protokoll,
        einordnung=Einordnung.experiment("verfahren", durchgang=2, phase="sondierung"),
    )

    (zeile,) = protokoll.zeilen()
    assert zeile["zweck"] == "experiment"
    assert zeile["baseline"] == "verfahren"
    assert zeile["phase"] == "sondierung"
    assert zeile["durchgang"] == 2


def test_eine_zeile_einer_baseline_traegt_keine_phase(protokoll):
    """Phasen hat nur das Verfahren."""
    schreibe(protokoll, einordnung=Einordnung.experiment("gleichverteilung", durchgang=1))

    (zeile,) = protokoll.zeilen()
    assert zeile["baseline"] == "gleichverteilung"
    assert zeile["phase"] is None
    assert zeile["durchgang"] == 1


@pytest.mark.parametrize(
    "angaben, gesucht",
    [
        ({"zweck": "probelauf"}, "probelauf"),  # ausdruecklich kein zulaessiger Zweck
        ({"zweck": "experiment", "durchgang": 1}, "Baseline"),
        ({"zweck": "experiment", "baseline": "gleichverteilung"}, "Durchgang"),
        ({"zweck": "experiment", "baseline": "verfahren", "durchgang": 1}, "Phase"),
        ({"zweck": "experiment", "baseline": "zufall", "durchgang": 1}, "zufall"),
        (
            {"zweck": "experiment", "baseline": "verfahren", "durchgang": 1, "phase": "finale"},
            "finale",
        ),
        (
            {"zweck": "experiment", "baseline": "ein-modell", "durchgang": 1, "phase": "fokus"},
            "Phase",
        ),
        ({"zweck": "experiment", "baseline": "ein-modell", "durchgang": 0}, "Durchgang"),
        ({"zweck": "experiment", "baseline": "ein-modell", "durchgang": "1"}, "Durchgang"),
        ({"zweck": "experiment", "baseline": "ein-modell", "durchgang": True}, "Durchgang"),
        ({"zweck": "vorstudie", "durchgang": 1}, "Durchgang"),
        ({"zweck": "handlauf", "baseline": "verfahren"}, "Baseline"),
        ({"zweck": "vorstudie", "phase": "fokus"}, "Phase"),
    ],
)
def test_ungueltige_oder_fehlende_angaben_fallen_beim_bau_auf(angaben, gesucht):
    """Frueh, bevor irgendetwas bezahlt ist: eine halbe Zeile waere spaeter nicht zu ergaenzen."""
    with pytest.raises(ValueError, match=gesucht):
        Einordnung(**angaben)


def test_ein_unbekannter_abschnitt_ist_ein_fehler_und_keine_neue_spalte(protokoll):
    """`ausfuehrung` ist kein gueltiger Abschnitt (aelterer Name der Nachbewertung)."""
    with pytest.raises(ValueError, match="ausfuehrung"):
        schreibe(protokoll, abschnitt="ausfuehrung")

    assert protokoll.zeilen() == []


def test_ohne_abschnitt_nur_beim_handlauf_und_in_der_vorstudie(protokoll):
    """Der kleine Handlauf und der Kostenabgleich ueber Trainingsdaten gehoeren zu keinem
    Abschnitt. Kalibrieren darf so eine Zeile nicht.

    Im Experiment aber ist jeder Messlauf einer: sonst liessen sich die Kosten eines
    Durchgangs nicht nach Optimierungs- und Ausfuehrungskosten trennen.
    """
    schreibe(protokoll, abschnitt=None, einordnung=HANDLAUF)
    schreibe(protokoll, abschnitt=None, einordnung=VORSTUDIE)
    assert [zeile["abschnitt"] for zeile in protokoll.zeilen()] == [None, None]
    assert protokoll.kalibrierbare_zeilen() == []

    with pytest.raises(ValueError, match="Abschnitt"):
        schreibe(
            protokoll, abschnitt=None, einordnung=Einordnung.experiment("ein-modell", durchgang=1)
        )


@pytest.mark.parametrize("einordnung", [VORSTUDIE, HANDLAUF])
def test_eine_testbewertung_gibt_es_nur_im_experiment(protokoll, einordnung):
    """Die Testdaten bleiben bis zur abschliessenden Auswertung unberuehrt; eine
    Testbewertungszeile der Vorstudie oder eines Handlaufs hiesse, sie wurden vorher gesehen."""
    schreibe(
        protokoll,
        abschnitt="testbewertung",
        einordnung=Einordnung.experiment("ohne-optimierung", durchgang=1),
    )

    with pytest.raises(ValueError, match="ADR-0008"):
        schreibe(protokoll, abschnitt="testbewertung", einordnung=einordnung)

    assert [zeile["zweck"] for zeile in protokoll.zeilen()] == ["experiment"]


def test_ohne_einordnung_entsteht_keine_zeile(protokoll):
    with pytest.raises(ValueError, match="Einordnung"):
        schreibe(protokoll, einordnung="vorstudie")

    assert protokoll.zeilen() == []


# Eine Zeile im alten Format, geschrieben von einem Handlauf am 20.09.2026 - gekuerzt, aber
# mit allen Feldern, die fuer die fruehere Kalibrierregel zaehlten. Sie haette diese Regel
# erfuellt, obwohl sie ein Handlauf war.
ALTE_ZEILE_VOM_20_09 = {
    "zeitpunkt": "2026-09-20T14:18:01+02:00",
    "art": "optimierung",
    "konfiguration": {"kandidatenmodell": "openai/gpt-oss-120b", "num_trials": 2},
    "seed": 0,
    "bewertung": 0.7837999999999999,
    "dollar": 0.045094,
}


def test_kalibriert_wird_nur_aus_gelungenen_optimierungen_der_vorstudie(protokoll):
    """Zweck Vorstudie, Abschnitt Optimierung, Bewertung nicht leer.

    Ein Handlauf ist kein Messpunkt der Vorstudie, ein abgebrochener Lauf zoege die Gerade
    nach unten, und eine alte Zeile ohne Zweck laesst sich nicht mehr zuordnen.
    """
    with open(protokoll.pfad, "w", encoding="utf-8") as datei:
        datei.write(json.dumps(ALTE_ZEILE_VOM_20_09) + "\n")
    schreibe(protokoll, seed=1)  # die eine, die zaehlt
    schreibe(protokoll, seed=2, bewertung=None)  # abgebrochen
    schreibe(protokoll, seed=3, einordnung=HANDLAUF)
    schreibe(protokoll, seed=4, abschnitt="nachbewertung")
    schreibe(
        protokoll,
        seed=5,
        einordnung=Einordnung.experiment("verfahren", durchgang=1, phase="sondierung"),
    )

    assert [zeile["seed"] for zeile in protokoll.kalibrierbare_zeilen()] == [1]


def test_alte_zeilen_bleiben_wie_sie_sind(protokoll):
    """Das Protokoll wird nie umgeschrieben, auch nicht fuer ein neues Format."""
    with open(protokoll.pfad, "w", encoding="utf-8") as datei:
        datei.write(json.dumps(ALTE_ZEILE_VOM_20_09) + "\n")
    vorher = protokoll.pfad.read_text(encoding="utf-8")

    schreibe(protokoll)

    assert protokoll.pfad.read_text(encoding="utf-8").startswith(vorher)
    assert protokoll.zeilen()[0] == ALTE_ZEILE_VOM_20_09


def test_das_protokoll_haengt_nur_an_und_schreibt_nie_um(protokoll):
    schreibe(protokoll, seed=0)
    vorher = protokoll.pfad.read_text(encoding="utf-8")

    schreibe(protokoll, seed=1)

    nachher = protokoll.pfad.read_text(encoding="utf-8")
    assert nachher.startswith(vorher)
    assert [zeile["seed"] for zeile in protokoll.zeilen()] == [0, 1]


def test_jede_zeile_steht_sofort_auf_der_platte(protokoll):
    """Der Lauf danach kann abstuerzen; die Zeile davor muss trotzdem da sein."""
    schreibe(protokoll)

    # Gelesen wird ohne Zutun des Protokoll-Objekts: so saehe es ein neuer Prozess.
    roh = protokoll.pfad.read_text(encoding="utf-8")

    assert roh.endswith("\n")
    assert json.loads(roh.splitlines()[0])["dollar"] == 0.25


def test_eine_abgeschnittene_letzte_zeile_reisst_die_frueheren_nicht_mit(protokoll):
    """Stuerzt der Prozess mitten im Schreiben ab, bleibt eine halbe Zeile stehen."""
    schreibe(protokoll, seed=0)
    schreibe(protokoll, seed=1)
    with open(protokoll.pfad, "a", encoding="utf-8") as datei:
        datei.write('{"abschnitt": "optimierung", "seed": 2')

    zeilen = protokoll.zeilen()

    assert [zeile["seed"] for zeile in zeilen] == [0, 1]


def test_ein_protokoll_legt_sein_verzeichnis_an_statt_zu_scheitern(tmp_path):
    """Es laeuft ab dem allerersten API-Aufruf - da darf kein fehlender Ordner im Weg sein."""
    protokoll = Protokoll(tmp_path / "neu" / "tiefer" / "protokoll.jsonl")

    schreibe(protokoll)

    assert len(protokoll.zeilen()) == 1


def test_ein_leeres_protokoll_hat_keine_zeilen_und_ist_kein_fehler(protokoll):
    assert protokoll.zeilen() == []


def test_stoerungen_stehen_in_der_zeile(protokoll):
    """Ein Lauf mit verschluckten Fehlern muss auch spaeter noch als solcher erkennbar sein."""
    gestoert = messwerte()._replace(
        fehlgeschlagene_programmlaeufe=3, stoerungen=("3 von 20 Programmlaeufen ...",)
    )

    schreibe(protokoll, messwerte=gestoert)

    (zeile,) = protokoll.zeilen()
    assert zeile["fehlgeschlagene_programmlaeufe"] == 3
    assert zeile["stoerungen"] == ["3 von 20 Programmlaeufen ..."]


def test_das_verzeichnis_eines_laufs_ist_angelegt_und_traegt_den_namen(tmp_path):
    """Dieses Verzeichnis bekommt MIPROv2 spaeter als `log_dir`."""
    verzeichnis = verzeichnis_fuer_lauf("gemma-4-31b-it", wurzel=tmp_path)

    assert verzeichnis.is_dir()
    assert "gemma-4-31b-it" in verzeichnis.name
    assert verzeichnis_fuer_lauf("gemma-4-31b-it", wurzel=tmp_path) != verzeichnis


SONDIERUNG = Einordnung.experiment("verfahren", durchgang=3, phase="sondierung")


def test_eine_entscheidung_steht_als_eigene_zeile_im_protokoll(protokoll):
    """Zwischenbewertung und Endauswahl kosten kein Geld und sind trotzdem ein Messergebnis.
    Die Zeile traegt, wozu sie gehoert, wie jede andere auch."""
    zeile = protokoll.schreibe_entscheidung(
        abschnitt="zwischenbewertung",
        einordnung=SONDIERUNG,
        seed=7,
        entscheidung={"weiter": ["a"], "ausgeschieden": {"b": "test"}},
    )

    (gelesen,) = protokoll.zeilen()
    assert gelesen == zeile
    assert gelesen["abschnitt"] == "zwischenbewertung"
    assert (gelesen["zweck"], gelesen["baseline"], gelesen["phase"], gelesen["durchgang"]) == (
        "experiment",
        "verfahren",
        "sondierung",
        3,
    )
    assert gelesen["seed"] == 7
    assert gelesen["weiter"] == ["a"]
    assert gelesen["ausgeschieden"] == {"b": "test"}
    assert "zeitpunkt" in gelesen
    assert protokoll.kalibrierbare_zeilen() == []


def test_eine_entscheidung_ist_kein_messlauf_und_umgekehrt(protokoll):
    """Ein Messlauf kann nicht als Entscheidung geschrieben werden und eine Entscheidung
    nicht als Messlauf: die Auswertung trennt die beiden an ihrem Abschnitt."""
    with pytest.raises(ValueError, match="keine Entscheidung"):
        protokoll.schreibe_entscheidung(
            abschnitt="optimierung", einordnung=SONDIERUNG, seed=0, entscheidung={}
        )
    with pytest.raises(ValueError, match="kein Abschnitt"):
        schreibe(protokoll, abschnitt="zwischenbewertung")

    assert protokoll.zeilen() == []


def test_eine_entscheidung_ohne_einordnung_entsteht_nicht(protokoll):
    with pytest.raises(ValueError, match="keine Einordnung"):
        protokoll.schreibe_entscheidung(
            abschnitt="zwischenbewertung", einordnung="experiment", seed=0, entscheidung={}
        )

    assert protokoll.zeilen() == []


def test_eine_entscheidung_kann_die_einordnung_ihrer_zeile_nicht_ueberschreiben(protokoll):
    """Die Felder der Entscheidung kommen nach den geprueften Feldern in die Zeile; ein
    gleichnamiges Feld ersetzte sie still."""
    with pytest.raises(ValueError, match="durchgang"):
        protokoll.schreibe_entscheidung(
            abschnitt="zwischenbewertung",
            einordnung=SONDIERUNG,
            seed=0,
            entscheidung={"durchgang": 1},
        )

    assert protokoll.zeilen() == []
