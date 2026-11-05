"""Tests zur Vorstudie.

Zwei Pruefstellen: der **Ablauf** - was in welcher Reihenfolge gemessen wird, was nach einem
Abbruch noch fehlt, wann angehalten wird - und die **Auswertung**, die aus Protokollzeilen
Kostenmodell, Kostenobergrenze und die Untergrenze von B rechnet.

Beide gegen eine Attrappe der Naht, die keinen API-Aufruf macht, aber echte Zeilen ins
Protokoll schreibt: die Vorstudie liest nur das Protokoll, und genau so soll sie
geprueft werden.
"""

import json
import statistics

import pytest

from dspy_optimizer.kandidatenmodelle import DIE_VIER, kandidatenmodell_mit_namen
from dspy_optimizer.kostenerfassung import Messwerte, Tokens
from dspy_optimizer.optimierungslauf import (
    Bewertungsergebnis,
    Fixkostenergebnis,
    Optimierungsergebnis,
)
from dspy_optimizer.protokoll import VORSTUDIE, Protokoll
from dspy_optimizer.vorstudie import (
    Messpunkt,
    Vorstudie,
    VorstudieAngehalten,
    bericht,
    werte_aus,
)

GPT_OSS, GEMMA, DEEPSEEK, LUNA = (modell.name for modell in DIE_VIER)
AUSWAHLBEISPIELE = 200


class Erfunden:
    """Die erfundenen Zahlen eines Kandidatenmodells: Kosten auf einer Geraden.

    Die drei Fixkosten-Messungen streuen um die Fixkosten, mit dem Faktor 0,9, 1,0 und 1,1
    (`STREUUNG`); ihr Mittelwert sind genau die Fixkosten.
    """

    def __init__(self, fixkosten, kosten_je_trial, ohne, optimiert, bewertung_ohne=0.6):
        self.fixkosten = fixkosten
        self.kosten_je_trial = kosten_je_trial
        # Ausfuehrungskosten je Programmlauf, ohne Optimierung und je Trial-Zahl optimiert.
        self.ohne = ohne
        self.optimiert = optimiert
        self.bewertung_ohne = bewertung_ohne


ERFUNDEN = {
    GPT_OSS: Erfunden(0.05, 0.004, 0.00011, {10: 0.00020, 20: 0.00030, 40: 0.00040}, 0.70),
    GEMMA: Erfunden(0.06, 0.005, 0.00015, {10: 0.00030, 20: 0.00030, 40: 0.00030}, 0.75),
    DEEPSEEK: Erfunden(0.07, 0.006, 0.00020, {10: 0.00050, 20: 0.00060, 40: 0.00070}, 0.80),
    LUNA: Erfunden(0.10, 0.010, 0.00026, {10: 0.00080, 20: 0.00090, 40: 0.00100}, 0.65),
}


# Je Fixkosten-Messung 1, 2, 3 der Faktor auf die erfundenen Fixkosten.
STREUUNG = {1: 0.9, 2: 1.0, 3: 1.1}


def messwerte(dollar, programmlaeufe, anbieterfehler=0, dauer=60.0, fixkosten=None):
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
        dauer_sekunden=dauer,
        latenzen=(1.0, 2.0, 3.0),
        anbieterfehler=anbieterfehler,
        fixkosten=fixkosten,
    )


class AttrappeNaht:
    """Die Naht ohne API: schreibt die beiden Zeilen eines Optimierungslaufs bzw. die eine
    Zeile von `nur_bewerten` oder `nur_fixkosten` ins Protokoll, mit den erfundenen Zahlen.

    `anbieterfehler` nennt Messpunkte `(name, num_trials)`, deren Lauf einen Anbieterfehler
    meldet - `(name, 0)` fuer jede Fixkosten-Messung des Kandidatenmodells; `absturz` einen
    Messpunkt, dessen Lauf mitten in der Optimierung abbricht.
    """

    def __init__(self, protokoll, anbieterfehler=(), absturz=None):
        self.protokoll = protokoll
        self.anbieterfehler = set(anbieterfehler)
        self.absturz = absturz
        self.auswahldaten = list(range(AUSWAHLBEISPIELE))
        self.testdaten = []
        self.gerufen = []

    def __call__(self, kandidatenmodell, startprogramm, num_trials, seed, *, einordnung, plan=None):
        name = kandidatenmodell.name
        self.gerufen.append((name, num_trials, seed, einordnung, None))
        erfunden = ERFUNDEN[name]
        konfiguration = {
            "kandidatenmodell": name,
            "anbieter": kandidatenmodell.anbieter,
            "num_trials": num_trials,
        }
        if self.absturz == (name, num_trials):
            self.protokoll.schreibe(
                abschnitt="optimierung", einordnung=einordnung, konfiguration=konfiguration,
                seed=seed, bewertung=None, messwerte=messwerte(0.01, 0),
            )
            raise RuntimeError("Absturz mitten in der Optimierung")
        fehler = 1 if (name, num_trials) in self.anbieterfehler else 0
        optimierungskosten = erfunden.fixkosten + num_trials * erfunden.kosten_je_trial
        self.protokoll.schreibe(
            abschnitt="optimierung", einordnung=einordnung, konfiguration=konfiguration,
            seed=seed, bewertung=0.8,
            messwerte=messwerte(
                optimierungskosten, 35 * num_trials, fixkosten=erfunden.fixkosten
            ),
        )
        ausfuehrungskosten = erfunden.optimiert[num_trials]
        self.protokoll.schreibe(
            abschnitt="nachbewertung", einordnung=einordnung, konfiguration=konfiguration,
            seed=seed, bewertung=0.8, exakt=0.7,
            messwerte=messwerte(
                AUSWAHLBEISPIELE * ausfuehrungskosten, AUSWAHLBEISPIELE, anbieterfehler=fehler
            ),
        )
        return Optimierungsergebnis(
            programm=startprogramm,
            bewertung=0.8,
            bewertungen_je_beispiel=(0.8,) * AUSWAHLBEISPIELE,
            optimierungskosten=optimierungskosten,
            nachbewertungskosten=AUSWAHLBEISPIELE * ausfuehrungskosten,
            ausfuehrungskosten=ausfuehrungskosten,
            programmlaeufe=35 * num_trials,
            formatfehler=0,
            anbieterfehler=fehler,
        )

    def nur_bewerten(self, kandidatenmodell, programm, seed, *, einordnung, abschnitt):
        name = kandidatenmodell.name
        self.gerufen.append((name, None, seed, einordnung, abschnitt))
        erfunden = ERFUNDEN[name]
        fehler = 1 if (name, None) in self.anbieterfehler else 0
        self.protokoll.schreibe(
            abschnitt=abschnitt, einordnung=einordnung,
            konfiguration={
                "kandidatenmodell": name, "anbieter": kandidatenmodell.anbieter, "num_trials": None
            },
            seed=seed, bewertung=erfunden.bewertung_ohne, exakt=0.5,
            messwerte=messwerte(
                AUSWAHLBEISPIELE * erfunden.ohne, AUSWAHLBEISPIELE, anbieterfehler=fehler
            ),
        )
        return Bewertungsergebnis(
            bewertung=erfunden.bewertung_ohne,
            bewertungen_je_beispiel=(erfunden.bewertung_ohne,) * AUSWAHLBEISPIELE,
            bewertungskosten=AUSWAHLBEISPIELE * erfunden.ohne,
            ausfuehrungskosten=erfunden.ohne,
            formatfehler=0,
            anbieterfehler=fehler,
        )


    def nur_fixkosten(self, kandidatenmodell, startprogramm, seed, *, einordnung, wiederholung):
        name = kandidatenmodell.name
        self.gerufen.append((name, 0, seed, einordnung, wiederholung))
        fixkosten = ERFUNDEN[name].fixkosten * STREUUNG[wiederholung]
        fehler = 1 if (name, 0) in self.anbieterfehler else 0
        self.protokoll.schreibe(
            abschnitt="fixkosten", einordnung=einordnung,
            konfiguration={
                "kandidatenmodell": name, "anbieter": kandidatenmodell.anbieter,
                "num_trials": 0, "wiederholung": wiederholung,
            },
            seed=seed, bewertung=0.6,
            messwerte=messwerte(
                fixkosten, AUSWAHLBEISPIELE, anbieterfehler=fehler, fixkosten=fixkosten
            ),
        )
        return Fixkostenergebnis(
            fixkosten=fixkosten, kosten=fixkosten, bewertung=0.6, formatfehler=0,
            anbieterfehler=fehler,
        )


@pytest.fixture
def protokoll(tmp_path):
    return Protokoll(tmp_path / "protokoll.jsonl")


def messpunkte(naht):
    return [(name, num_trials) for name, num_trials, *_ in naht.gerufen]


# Bewertungen und Optimierungslaeufe (16 Messpunkte) und Fixkosten-Messungen (zwoelf).
ALTE_MESSPUNKTE = [
    (name, num_trials)
    for name in (LUNA, DEEPSEEK, GEMMA, GPT_OSS)
    for num_trials in (None, 10, 20, 40)
]
FIXKOSTEN_MESSUNGEN = [(name, 0) for name in (LUNA, DEEPSEEK, GEMMA, GPT_OSS) for _ in range(3)]


# --- Ablauf ---------------------------------------------------------------------------


def test_misst_den_messplan_teuerstes_kandidatenmodell_zuerst(protokoll):
    """Je Kandidatenmodell eine Bewertung ohne Optimierung und drei Optimierungslaeufe mit
    10, 20 und 40 Trials; das teuerste Kandidatenmodell (Luna) zuerst. Danach, in derselben
    Reihenfolge, je drei Fixkosten-Messungen."""
    naht = AttrappeNaht(protokoll)

    Vorstudie(naht, protokoll)()

    assert messpunkte(naht) == ALTE_MESSPUNKTE + FIXKOSTEN_MESSUNGEN


def test_die_fixkosten_messungen_laufen_mit_den_seeds_0_4_5(protokoll):
    """Nicht mit Seeds des Hauptexperiments (1, 2, 3). Jede Messung traegt ihre Nummer,
    sonst saehen die drei Zeilen im Protokoll gleich aus."""
    naht = AttrappeNaht(protokoll)

    Vorstudie(naht, protokoll)()

    luna = [
        (seed, wiederholung)
        for name, num_trials, seed, _, wiederholung in naht.gerufen
        if name == LUNA and num_trials == 0
    ]
    assert luna == [(0, 1), (4, 2), (5, 3)]
    alle_uebrigen = {seed for _, num_trials, seed, *_ in naht.gerufen if num_trials != 0}
    assert alle_uebrigen == {0}


def test_im_protokoll_ohne_fixkostenfeld_fehlen_genau_die_zwoelf_fixkosten_messungen(protokoll):
    """Die 16 Messpunkte aus Bewertungen und Optimierungslaeufen bleiben fertig, auch wenn
    ihren Zeilen das Feld `fixkosten` fehlt: das Protokoll wird nur angehaengt, bestehende
    Zeilen werden nicht umgeschrieben. Gaelten diese Laeufe als offen, bezahlte der naechste
    Start sie neu. Der normale Startbefehl misst genau die fehlenden Fixkosten."""
    for name, num_trials in ALTE_MESSPUNKTE:
        modell = kandidatenmodell_mit_namen(name)
        if num_trials is None:
            AttrappeNaht(protokoll).nur_bewerten(
                modell, None, 0, einordnung=VORSTUDIE, abschnitt="nachbewertung"
            )
        else:
            AttrappeNaht(protokoll)(modell, None, num_trials, 0, einordnung=VORSTUDIE)
    alte_zeilen = protokoll.zeilen()
    for zeile in alte_zeilen:
        del zeile["fixkosten"]
    protokoll.pfad.write_text(
        "".join(json.dumps(zeile) + "\n" for zeile in alte_zeilen), encoding="utf-8"
    )

    assert werte_aus(protokoll.zeilen()).offen == tuple(
        Messpunkt(name, 0, wiederholung)
        for name in (LUNA, DEEPSEEK, GEMMA, GPT_OSS)
        for wiederholung in (1, 2, 3)
    )
    naht = AttrappeNaht(protokoll)
    Vorstudie(naht, protokoll)()
    assert messpunkte(naht) == FIXKOSTEN_MESSUNGEN


def test_haelt_nach_einem_anbieterfehler_in_der_fixkosten_messung_an(protokoll):
    """Wie bei jedem Lauf der Vorstudie: ein Anbieterfehler macht den Lauf ungueltig."""
    naht = AttrappeNaht(protokoll, anbieterfehler={(DEEPSEEK, 0)})

    with pytest.raises(VorstudieAngehalten, match="Fixkosten-Messung"):
        Vorstudie(naht, protokoll)()

    assert messpunkte(naht) == ALTE_MESSPUNKTE + FIXKOSTEN_MESSUNGEN[:4]


def test_eine_fixkosten_messung_ohne_gemessene_fixkosten_ist_nicht_fertig(protokoll):
    """Kam die erste vollstaendige Bewertung nicht zu Ende, ist das Feld leer - dann wird die
    Messung wiederholt. Ebenso eine Zeile mit einem anderen Seed als dem ihrer Nummer: sie
    ist nicht die geplante Messung."""
    luna = kandidatenmodell_mit_namen(LUNA)
    for wiederholung, seed, fixkosten in ((1, 0, None), (2, 1, 0.1)):
        protokoll.schreibe(
            abschnitt="fixkosten", einordnung=VORSTUDIE,
            konfiguration={
                "kandidatenmodell": LUNA, "anbieter": luna.anbieter, "num_trials": 0,
                "wiederholung": wiederholung,
            },
            seed=seed, bewertung=0.6, messwerte=messwerte(0.1, 200, fixkosten=fixkosten),
        )

    assert werte_aus(protokoll.zeilen()).offen[16:18] == (
        Messpunkt(LUNA, 0, 1), Messpunkt(LUNA, 0, 2)
    )


def test_haelt_nach_einem_anbieterfehler_an(protokoll):
    """Auf einem Lauf mit Anbieterfehler wird nichts entschieden. Die Vorstudie haelt an,
    damit vor dem naechsten Lauf geklaert werden kann, ob es ein Rate Limit war."""
    naht = AttrappeNaht(protokoll, anbieterfehler={(LUNA, 20)})

    with pytest.raises(VorstudieAngehalten, match="Anbieterfehler"):
        Vorstudie(naht, protokoll)()

    assert messpunkte(naht) == [(LUNA, None), (LUNA, 10), (LUNA, 20)]


def test_haelt_auch_nach_einem_anbieterfehler_ohne_optimierung_an(protokoll):
    naht = AttrappeNaht(protokoll, anbieterfehler={(LUNA, None)})

    with pytest.raises(VorstudieAngehalten):
        Vorstudie(naht, protokoll)()

    assert messpunkte(naht) == [(LUNA, None)]


def test_setzt_nach_dem_anhalten_fort_und_wiederholt_nur_den_ungueltigen_lauf(protokoll):
    """Bezahlte Laeufe werden nicht doppelt bezahlt; der Lauf mit Anbieterfehler zaehlt nicht
    und wird wiederholt."""
    with pytest.raises(VorstudieAngehalten):
        Vorstudie(AttrappeNaht(protokoll, anbieterfehler={(LUNA, 20)}), protokoll)()

    naht = AttrappeNaht(protokoll)
    Vorstudie(naht, protokoll)()

    assert messpunkte(naht)[:3] == [(LUNA, 20), (LUNA, 40), (DEEPSEEK, None)]
    assert len(messpunkte(naht)) == 28 - 2


def test_setzt_nach_einem_absturz_fort(protokoll):
    """Ein abgestuerzter Lauf hinterlaesst eine Optimierungszeile ohne Bewertung und keine
    Nachbewertung - er ist nicht fertig und wird wiederholt."""
    with pytest.raises(RuntimeError, match="Absturz"):
        Vorstudie(AttrappeNaht(protokoll, absturz=(LUNA, 10)), protokoll)()

    naht = AttrappeNaht(protokoll)
    Vorstudie(naht, protokoll)()

    assert messpunkte(naht)[0] == (LUNA, 10)
    assert len(messpunkte(naht)) == 28 - 1


def test_zaehlt_nur_zeilen_der_vorstudie_als_fertig(protokoll):
    """Ein Handlauf mit denselben Zahlen ist kein Lauf der Vorstudie."""
    from dspy_optimizer.protokoll import HANDLAUF

    AttrappeNaht(protokoll).nur_bewerten(
        DIE_VIER[3], None, 0, einordnung=HANDLAUF, abschnitt="nachbewertung"
    )
    naht = AttrappeNaht(protokoll)
    Vorstudie(naht, protokoll)()

    assert messpunkte(naht)[0] == (LUNA, None)


def test_ein_lauf_bei_einem_frueheren_anbieter_wird_neu_gemessen(protokoll):
    """Gemessen wird das Kandidatenmodell bei seinem festgelegten Anbieter. Wechselt der
    Anbieter, sind die Laeufe beim alten eine andere Messung - am 10.10.2026 wechselte
    DeepSeek V4.1 Flash von OpenInference zu inference.net."""
    luna, deepseek = (kandidatenmodell_mit_namen(name) for name in (LUNA, DEEPSEEK))
    deepseek_beim_frueheren_anbieter = deepseek._replace(anbieter="frueherer/anbieter")
    naht_der_ersten_laeufe = AttrappeNaht(protokoll)
    for modell in (luna, deepseek_beim_frueheren_anbieter):
        naht_der_ersten_laeufe.nur_bewerten(
            modell, None, 0, einordnung=VORSTUDIE, abschnitt="nachbewertung"
        )
        for num_trials in (10, 20, 40):
            naht_der_ersten_laeufe(modell, None, num_trials, 0, einordnung=VORSTUDIE)

    naht = AttrappeNaht(protokoll)
    Vorstudie(naht, protokoll)()

    assert messpunkte(naht)[:4] == [(DEEPSEEK, n) for n in (None, 10, 20, 40)]
    assert len(messpunkte(naht)) == 28 - 4


# --- Auswertung -----------------------------------------------------------------------


@pytest.fixture
def ausgewertet(protokoll):
    """Die ganze Vorstudie gegen die Attrappe, dann ausgewertet."""
    Vorstudie(AttrappeNaht(protokoll), protokoll)()
    return werte_aus(protokoll.zeilen())


def test_kalibriert_das_kostenmodell_je_kandidatenmodell(ausgewertet):
    """Messfrage 1: die Fixkosten sind der Mittelwert der drei Fixkosten-Messungen,
    (0,045 + 0,05 + 0,055) / 3 = 0,05; die Kosten je Trial die Steigung durch die
    Optimierungslaeufe darueber. Die Ausfuehrungskosten sind der Mittelwert der drei
    Nachbewertungen: (0,0002 + 0,0003 + 0,0004) / 3 = 0,0003."""
    modell = ausgewertet.kostenmodelle[GPT_OSS]

    assert modell.kandidatenmodell == GPT_OSS
    assert modell.fixkosten.betrag == pytest.approx(0.05)
    assert modell.kosten_je_trial.betrag == pytest.approx(0.004)
    assert modell.ausfuehrungskosten.betrag == pytest.approx(0.0003)
    assert "Vorstudie" in modell.fixkosten.herkunft
    assert "3 Fixkosten-Messungen" in modell.fixkosten.herkunft
    assert "Vorstudie" in modell.ausfuehrungskosten.herkunft


def test_ohne_fixkosten_meldet_die_auswertung_was_fehlt_statt_abzubrechen(protokoll):
    """Ein Protokoll ohne gueltige Fixkosten-Messung. Ohne Fixkosten gibt es kein Kostenmodell; der
    Bericht nennt die naechste Fixkosten-Messung, statt mit einem Fehler abzubrechen."""
    with pytest.raises(VorstudieAngehalten):
        Vorstudie(AttrappeNaht(protokoll, anbieterfehler={(LUNA, 0)}), protokoll)()

    auswertung = werte_aus(protokoll.zeilen())
    text = bericht(auswertung)

    assert auswertung.ergebnisse == {}
    assert "Fixkosten fehlen noch: 12 Fixkosten-Messungen sind offen" in text
    assert "Es fehlen noch 12 Laeufe, als naechster Fixkosten-Messung 1 von 3" in text


def test_kostenobergrenze_ist_der_mittelwert_des_zweitteuersten(ausgewertet):
    """Die Kostenobergrenze ist der optimierte Mittelwert des zweitteuersten
    Kandidatenmodells. Die Mittelwerte optimiert: gpt-oss 0,0003,
    Gemma 0,0003, DeepSeek 0,0006, Luna 0,0009 $ je Programmlauf. Zweitteuerstes ist
    DeepSeek: 0,6 $ je 1000 Programmlaeufe - genau Luna reisst sie."""
    assert ausgewertet.zweitteuerstes == DEEPSEEK
    assert ausgewertet.kostenobergrenze == pytest.approx(0.6)


def test_ohne_vollstaendigen_messplan_gibt_es_keine_kostenobergrenze(protokoll):
    """Erst wenn alle vier gemessen sind, steht fest, welches das zweitteuerste ist."""
    with pytest.raises(VorstudieAngehalten):
        Vorstudie(AttrappeNaht(protokoll, anbieterfehler={(GEMMA, 0)}), protokoll)()
    auswertung = werte_aus(protokoll.zeilen())

    assert list(auswertung.ergebnisse) == [LUNA, DEEPSEEK]
    assert auswertung.offen[0] == Messpunkt(GEMMA, 0, 1)
    with pytest.raises(ValueError, match="noch nicht vollstaendig"):
        auswertung.kostenobergrenze


def test_b_reicht_jedem_kandidatenmodell_fuer_neun_trials_in_der_sondierung(ausgewertet):
    """B/8 je Kandidatenmodell, mindestens rund 9 Trials, massgeblich das teuerste. Von
    Hand: Luna 0,10 + 9 * 0,010 + 200 * 0,0009 = 0,37 $, also B = 8 * 0,37 = 2,96 $
    (gpt-oss braeuchte nur 0,146 $, Gemma 0,165 $, DeepSeek 0,244 $)."""
    assert ausgewertet.b_mindestens == pytest.approx(2.96)

    for modell in ausgewertet.kostenmodelle.values():
        assert modell.num_trials_fuer(ausgewertet.b_mindestens / 8, AUSWAHLBEISPIELE) >= 9
    luna = ausgewertet.kostenmodelle[LUNA]
    assert luna.num_trials_fuer(ausgewertet.b_mindestens / 8, AUSWAHLBEISPIELE) == 9


def test_schaetzt_das_hauptexperiment_fuer_ein_b(ausgewertet):
    """Messfrage 4: passt das Experiment in die verfuegbaren Mittel? Je Seed 6 * B
    (Verfahren, Gleichverteilung, vier Einzelmodelle) und die Bewertungen ausserhalb von B,
    drei Seeds. Von Hand, je Seed:
    Baseline ohne Optimierung 200 * 0,00072 = 0,144; Testbewertung der vier unoptimierten
    500 * 0,00072 = 0,36; der vier Gleichverteilten und der vier Einzelmodelle je
    500 * 0,0021 = 1,05; des Verfahrensergebnisses hoechstens 500 * 0,0009 = 0,45.
    Zusammen 3,054 je Seed; bei B = 3: 3 * (18 + 3,054) = 63,162 $."""
    assert ausgewertet.hauptexperiment(3.0) == pytest.approx(63.162)


def test_zaehlt_als_ausgegeben_auch_was_nicht_gewertet_wird(protokoll):
    """Ein abgebrochener Lauf zaehlt fuer nichts, bezahlt ist er trotzdem. Die
    ganze Vorstudie kostet mit den erfundenen Zahlen 3,994 $ und die Fixkosten-Messungen
    3 * (0,05 + 0,06 + 0,07 + 0,10) = 0,84 $, der Absturz 0,01 $ dazu."""
    with pytest.raises(RuntimeError):
        Vorstudie(AttrappeNaht(protokoll, absturz=(LUNA, 10)), protokoll)()
    Vorstudie(AttrappeNaht(protokoll), protokoll)()

    assert werte_aus(protokoll.zeilen()).ausgegeben == pytest.approx(4.844)


def test_spannweite_ohne_optimierung(ausgewertet):
    """Messfrage 2 (Spannweite der Kandidatenmodelle ohne Optimierung): DeepSeek 0,80 gegen
    Luna 0,65."""
    assert ausgewertet.spannweite == pytest.approx(0.15)


def test_anstieg_der_ausfuehrungskosten_durch_die_optimierung(ausgewertet):
    """Messfrage 3 (Anstieg der Ausfuehrungskosten durch die Optimierung): gpt-oss 0,0003
    optimiert gegen 0,00011 ohne."""
    assert ausgewertet.ergebnisse[GPT_OSS].anstieg == pytest.approx(0.0003 / 0.00011)


def test_latenz_und_laufzeit_je_kandidatenmodell(ausgewertet):
    """Latenz ohne und mit Optimierung und Stunden je Dollar. Jede Zeile der Attrappe:
    Latenzen 1, 2, 3 s, also Median 2 s, und 60 s Dauer. gpt-oss hat sieben gueltige
    Zeilen, 420 s fuer 0,632 $: 0,1167 h / 0,632 $ = 0,1846 h je Dollar. Die
    Fixkosten-Messungen zaehlen nicht mit."""
    gpt_oss = ausgewertet.ergebnisse[GPT_OSS]

    assert gpt_oss.latenz_ohne == pytest.approx(2.0)
    assert gpt_oss.latenz_optimiert == pytest.approx(2.0)
    assert gpt_oss.stunden_je_dollar == pytest.approx(420 / 3600 / 0.632)


def test_bericht_nennt_jedes_kandidatenmodell_und_die_kostenobergrenze(ausgewertet):
    text = bericht(ausgewertet)

    for name in (GPT_OSS, GEMMA, DEEPSEEK, LUNA):
        assert name in text
    assert "Kostenobergrenze: 0.600 $ je 1000 Programmlaeufe" in text
    assert "B mindestens: 2.96 $" in text


def test_bericht_einer_unvollstaendigen_vorstudie_nennt_was_fehlt(protokoll):
    """Angehalten bei der ersten Fixkosten-Messung von Gemma: Luna und DeepSeek sind fertig,
    es fehlen je drei Fixkosten-Messungen von Gemma und gpt-oss."""
    with pytest.raises(VorstudieAngehalten):
        Vorstudie(AttrappeNaht(protokoll, anbieterfehler={(GEMMA, 0)}), protokoll)()

    text = bericht(werte_aus(protokoll.zeilen()))

    assert LUNA in text
    assert "Es fehlen noch 6 Laeufe" in text
    assert "Kostenobergrenze" not in text


def test_eine_nachbewertung_ohne_bewertung_ist_nicht_fertig(protokoll):
    """Die Bewertung ist der Marker fuer einen zu Ende gekommenen Messlauf. Bricht
    die Nachbewertung nach den Programmlaeufen ab, stehen Ausfuehrungskosten in der Zeile, aber
    keine Bewertung - der Lauf wird wiederholt."""
    protokoll.schreibe(
        abschnitt="nachbewertung", einordnung=VORSTUDIE,
        konfiguration={"kandidatenmodell": LUNA, "num_trials": None},
        seed=0, bewertung=None, messwerte=messwerte(0.05, AUSWAHLBEISPIELE),
    )
    naht = AttrappeNaht(protokoll)
    Vorstudie(naht, protokoll)()

    assert messpunkte(naht)[0] == (LUNA, None)


def test_bericht_ohne_gemessene_latenz(protokoll):
    """Scheitern alle Programmlaeufe einer Bewertung, ist keine Latenz gestoppt: das Feld ist
    leer, und der Bericht zeigt einen Strich statt abzustuerzen."""
    Vorstudie(AttrappeNaht(protokoll), protokoll)()
    zeilen = protokoll.zeilen()
    for zeile in zeilen:
        zeile["latenz_median_sekunden"] = None

    text = bericht(werte_aus(zeilen))

    assert LUNA in text


def test_bericht_nennt_anbieterfehler_und_gesamtsumme(protokoll):
    """Die Zahl der Anbieterfehler der Vorstudie muss im Bericht stehen. Und Messfrage 4
    rechnet Vorstudie und Hauptexperiment zusammen gegen die verfuegbaren Mittel."""
    with pytest.raises(VorstudieAngehalten):
        Vorstudie(AttrappeNaht(protokoll, anbieterfehler={(LUNA, 10)}), protokoll)()
    Vorstudie(AttrappeNaht(protokoll), protokoll)()
    auswertung = werte_aus(protokoll.zeilen())

    text = bericht(auswertung)

    assert auswertung.anbieterfehler == 1
    assert "Anbieterfehler in der Vorstudie: 1" in text
    luna_10_noch_einmal = 0.10 + 10 * 0.010 + AUSWAHLBEISPIELE * 0.0008
    assert auswertung.ausgegeben == pytest.approx(3.994 + 0.84 + luna_10_noch_einmal)
    zusammen = auswertung.ausgegeben + auswertung.hauptexperiment(2.96)
    assert f"Vorstudie und Hauptexperiment zusammen: {zusammen:.2f} $" in text


def test_der_bericht_zeigt_jede_fixkosten_messung_und_die_abweichung_der_geraden(ausgewertet):
    """Die Fixkosten als gemessen - jede Messung mit Seed und Betrag - und je
    Optimierungslauf, wie weit die Gerade danebenliegt. Mit den erfundenen Zahlen liegen die
    Laeufe genau auf der Geraden: 0,05 + 10 * 0,004 = 0,09 $, Abweichung 0 %."""
    text = bericht(ausgewertet)

    assert "## Je Fixkosten-Messung" in text
    assert f"| {GPT_OSS} | 1 | 0 | 0.0450 |" in text
    assert f"| {GPT_OSS} | 2 | 4 | 0.0500 |" in text
    assert f"| {GPT_OSS} | 3 | 5 | 0.0550 |" in text
    (zeile,) = [zeile for zeile in text.splitlines() if zeile.startswith(f"| {GPT_OSS} | 10 |")]
    assert zeile.startswith(f"| {GPT_OSS} | 10 | 0.0900 | 0.0900 |")
    assert "0.0 % |" in zeile


def test_die_auswertung_allein_misst_nichts(protokoll, monkeypatch, capsys):
    """`python -m dspy_optimizer.vorstudie auswertung` kostet nichts, auch wenn
    Fixkosten-Messungen fehlen; gemessen wird nur beim Aufruf ohne Zusatz."""
    from dspy_optimizer import vorstudie

    def keine_naht(**_):
        raise AssertionError("Die Auswertung hat eine Naht gebaut und haette gemessen.")

    monkeypatch.setattr(vorstudie, "Protokoll", lambda: protokoll)
    monkeypatch.setattr(vorstudie, "Optimierungslauf", keine_naht)
    monkeypatch.setattr(vorstudie.sys, "argv", ["vorstudie", "auswertung"])

    vorstudie.main()

    assert "Es fehlen noch 28 Laeufe" in capsys.readouterr().out
