"""Tests zur Auswertung.

Die Auswertung liest nur das Protokoll und ruft keine API. Getestet wird an zwei Stellen:
an kleinen, von Hand gebauten Zeilen fuer die Regeln (wann ein Durchgang zaehlt, wie der Preis
umgerechnet wird, Kendalls Tau, Break-even) und am echten Protokoll des Hauptexperiments.
"""

import math

import pytest
from scipy import stats

from dspy_optimizer.auswertung import (
    DEEPSEEK,
    NAMEN,
    Auswertung,
    break_even,
    dollar_zum_festgehaltenen_preis,
    formatiere,
    gepaarter_test,
    finde_gueltige_durchgaenge,
    kendall_tau,
    mittel_sd,
    pareto_dominiert,
    waehle_nach_der_regel,
)
from dspy_optimizer.protokoll import PROTOKOLLDATEI, Protokoll
from dspy_optimizer.verfahren import Stand

A, B, C, D = "anbieter/a", "anbieter/b", "anbieter/c", "anbieter/d"


def messlauf(abschnitt, baseline, durchgang, *, modell=A, seed=1, phase=None, **felder):
    """Eine Zeile eines Messlaufs im Experiment, nur mit den Feldern, die hier zaehlen."""
    return {
        "zeitpunkt": "2026-10-11T10:00:00+02:00",
        "abschnitt": abschnitt,
        "zweck": "experiment",
        "baseline": baseline,
        "phase": phase,
        "durchgang": durchgang,
        "konfiguration": {"kandidatenmodell": modell},
        "seed": seed,
        "bewertung": 0.8,
        "einzelbewertungen": None,
        "dollar": 1.0,
        "dollar_je_programmlauf": None,
        "tokens_prompt": 0,
        "tokens_antwort": 0,
        "anbieterfehler": 0,
        "stoerungen": [],
        **felder,
    }


def entscheidung(abschnitt, baseline, durchgang, *, seed=1, phase=None, **felder):
    return {
        "zeitpunkt": "2026-10-11T10:00:00+02:00",
        "abschnitt": abschnitt,
        "zweck": "experiment",
        "baseline": baseline,
        "phase": phase,
        "durchgang": durchgang,
        "seed": seed,
        **felder,
    }


# --- Wann ein Durchgang zaehlt -------------------------------------------------------------


def test_ein_durchgang_mit_anbieterfehler_zaehlt_nicht_seine_wiederholung_schon():
    zeilen = [
        messlauf(
            "optimierung", "ein-modell", 1, anbieterfehler=1,
            stoerungen=["Anbieterfehler: LMRateLimitError"],
        ),
        messlauf("optimierung", "ein-modell", 2),
        messlauf("nachbewertung", "ein-modell", 2),
        entscheidung("endauswahl", "ein-modell", 2, kandidatenmodelle={A: {}}),
    ]
    assert [d.nummer for d in finde_gueltige_durchgaenge(zeilen)] == [2]
    ungueltig = Auswertung(zeilen).ungueltige_durchgaenge()
    assert list(ungueltig["durchgang"]) == [1]
    assert list(ungueltig["ursache"]) == ["LMRateLimitError"]
    assert list(ungueltig["dollar"]) == [1.0]


def test_ein_abgebrochener_durchgang_ohne_entscheidung_zaehlt_nicht():
    zeilen = [
        messlauf("optimierung", "verfahren", 1, phase="sondierung"),
        entscheidung("zwischenbewertung", "verfahren", 1, phase="sondierung", kein_gewinner=False),
    ]
    assert finde_gueltige_durchgaenge(zeilen) == []
    assert list(Auswertung(zeilen).ungueltige_durchgaenge()["ursache"]) == [
        "ohne Entscheidung abgebrochen"
    ]


def test_ein_verfahren_das_nach_der_sondierung_endet_ist_gueltig():
    """Ohne Gewinner nach der Sondierung ist die Zwischenbewertung die letzte Zeile."""
    zeilen = [
        messlauf("optimierung", "verfahren", 1, phase="sondierung"),
        entscheidung("zwischenbewertung", "verfahren", 1, phase="sondierung", kein_gewinner=True),
    ]
    assert [d.nummer for d in finde_gueltige_durchgaenge(zeilen)] == [1]


def test_vorstudie_handlauf_und_testbewertungen_sind_keine_durchgaenge():
    zeilen = [
        {**messlauf("optimierung", None, None), "zweck": "vorstudie"},
        {**messlauf("optimierung", None, None), "zweck": "handlauf"},
        messlauf("testbewertung", "ein-modell", 3),
    ]
    assert finde_gueltige_durchgaenge(zeilen) == []
    assert Auswertung(zeilen).ungueltige_durchgaenge().empty


def test_eine_testbewertung_mit_anbieterfehler_zaehlt_als_wiederholt():
    test = {
        "konfiguration": {"kandidatenmodell": A, "paar": "durchgang-001/gewinner-a.json"},
        "dollar_je_programmlauf": 0.001,
        "einzelbewertungen": [0.8],
    }
    zeilen = [
        messlauf("optimierung", "ein-modell", 1),
        entscheidung("endauswahl", "ein-modell", 1, kandidatenmodelle={A: {}}),
        messlauf("testbewertung", "ein-modell", 1, anbieterfehler=2, **test),
        messlauf("testbewertung", "ein-modell", 1, **test),
    ]
    auswertung = Auswertung(zeilen)
    assert auswertung.wiederholte_testbewertungen() == 1
    assert len(auswertung.testbewertungen()) == 1


def zwei_rangfolgen(bewertungen_ohne, bewertungen_mit):
    """Ein Seed mit Baseline ohne Optimierung (Durchgang 1) und Gleichverteilung (Durchgang 2),
    je eine Testbewertung der vier Kandidatenmodelle des Experiments mit den gegebenen
    Einzelbewertungen."""
    zeilen = []
    for nummer, baseline, bewertungen in (
        (1, "ohne-optimierung", bewertungen_ohne),
        (2, "gleichverteilung", bewertungen_mit),
    ):
        zeilen.append(entscheidung("endauswahl", baseline, nummer, kandidatenmodelle={}))
        for modell, einzelbewertungen in zip(NAMEN, bewertungen):
            zeilen.append(
                messlauf(
                    "testbewertung", baseline, nummer, modell=modell,
                    konfiguration={"kandidatenmodell": modell, "paar": f"{nummer}-{modell}"},
                    bewertung=sum(einzelbewertungen) / len(einzelbewertungen),
                    einzelbewertungen=einzelbewertungen, dollar_je_programmlauf=0.0001,
                )
            )
    return zeilen


def test_ein_gleichstand_ist_kein_platzwechsel():
    oben, mitte, unten = [0.9, 0.8, 0.9], [0.5, 0.6, 0.5], [0.1, 0.2, 0.1]
    gleich = [0.3, 0.3, 0.3]
    zeilen = zwei_rangfolgen([oben, mitte, gleich, unten], [oben, mitte, gleich, gleich])
    assert Auswertung(zeilen).platzwechsel().empty


def test_ein_platzwechsel_wird_mit_dem_test_der_optimierten_programme_berichtet():
    oben, mitte, unten = [0.9, 0.8, 0.9], [0.5, 0.6, 0.5], [0.1, 0.2, 0.1]
    zeilen = zwei_rangfolgen([oben, mitte, unten, [0.0] * 3], [mitte, oben, unten, [0.0] * 3])
    wechsel = Auswertung(zeilen).platzwechsel()
    assert list(wechsel["paar"]) == ["gpt-oss-120b / Gemma 4 31B"]
    assert wechsel["p_mit"].iloc[0] == pytest.approx(stats.ttest_rel(mitte, oben).pvalue)


# --- Kosten zum festgehaltenen Preis -------------------------------------------------------


def test_andere_kandidatenmodelle_behalten_ihre_abgerechneten_kosten():
    zeile = messlauf("nachbewertung", "ein-modell", 1, zeitpunkt="2026-10-15T08:00:00+02:00")
    assert dollar_zum_festgehaltenen_preis(zeile) == 1.0


def test_deepseek_vor_der_ersten_erhoehung_bleibt_wie_abgerechnet():
    zeile = messlauf(
        "nachbewertung", "ein-modell", 1, modell=DEEPSEEK,
        zeitpunkt="2026-10-12T05:41:00+02:00", tokens_prompt=10**6, tokens_antwort=10**6,
    )
    assert dollar_zum_festgehaltenen_preis(zeile) == 1.0


@pytest.mark.parametrize(
    ("zeitpunkt", "listenpreis_damals"),
    [
        ("2026-10-13T17:02:00+02:00", 0.077 + 0.55),
        ("2026-10-15T08:17:00+02:00", 0.10 + 0.66),
    ],
)
def test_deepseek_nach_einer_erhoehung_im_verhaeltnis_der_listenpreise(
    zeitpunkt, listenpreis_damals
):
    """Gleicher Rabattanteil angenommen: abgerechnet mal festgehaltener durch damaligen
    Listenpreis, beides fuer die Tokens dieser Zeile."""
    zeile = messlauf(
        "nachbewertung", "ein-modell", 1, modell=DEEPSEEK, zeitpunkt=zeitpunkt,
        tokens_prompt=10**6, tokens_antwort=10**6, dollar=0.5,
    )
    assert dollar_zum_festgehaltenen_preis(zeile) == pytest.approx(
        0.5 * (0.07 + 0.60) / listenpreis_damals
    )


# --- Rechenregeln ------------------------------------------------------------------------


def test_kendalls_tau_zaehlt_einige_und_uneinige_paare():
    ohne = {A: 0.4, B: 0.3, C: 0.2, D: 0.1}
    assert kendall_tau(ohne, ohne) == 1
    assert kendall_tau(ohne, {A: 0.1, B: 0.2, C: 0.3, D: 0.4}) == -1
    # Ein Paar vertauscht: (5 einig - 1 uneinig) / 6
    assert kendall_tau(ohne, {A: 0.4, B: 0.3, C: 0.1, D: 0.2}) == pytest.approx(4 / 6)


def test_der_gepaarte_test_ist_zweiseitig():
    x = [0.1, 0.5, 0.9, 0.4, 0.7]
    y = [0.2, 0.4, 0.6, 0.5, 0.3]
    assert gepaarter_test(x, y) == pytest.approx(stats.ttest_rel(x, y).pvalue)
    assert gepaarter_test(x, x) is None


def test_mittelwert_und_standardabweichung_ueber_die_seeds():
    assert mittel_sd([1.0, 2.0, 3.0]) == (2.0, 1.0)
    assert formatiere(*mittel_sd([0.81, 0.82, 0.83])) == "0,820 ± 0,010"
    # Ein einziger Wert hat keine Streuung: angezeigt, nicht als 0 ausgegeben.
    assert formatiere(*mittel_sd([0.81])) == "0,810 ± –"


def test_wahl_mit_und_ohne_kostenobergrenze():
    teuer_und_gut = Stand(A, (0.9, 0.9), 0.0005)
    billig = Stand(B, (0.8, 0.8), 0.0001)
    assert waehle_nach_der_regel([teuer_und_gut, billig], 0.2895) == B
    assert waehle_nach_der_regel([teuer_und_gut, billig], math.inf) == A
    assert waehle_nach_der_regel([teuer_und_gut], 0.2895) is None


def test_break_even_nur_wenn_das_gewaehlte_paar_billiger_ist():
    assert break_even(6.0, 0.0001, 0.0003) == pytest.approx(30000)
    assert break_even(6.0, 0.0003, 0.0003) is None
    assert break_even(6.0, 0.0004, 0.0003) is None


def test_pareto_dominiert_wer_teurer_und_nicht_besser_ist():
    kosten = [0.1, 0.2, 0.3, 0.1, 0.2]
    qualitaet = [0.80, 0.85, 0.84, 0.79, 0.85]
    assert pareto_dominiert(kosten, qualitaet) == [False, False, True, True, False]


# --- Am echten Protokoll -------------------------------------------------------------------

echtes_protokoll = pytest.mark.skipif(
    not PROTOKOLLDATEI.exists(), reason="ohne ergebnisse/protokoll.jsonl"
)


@pytest.fixture(scope="module")
def echte_auswertung():
    return Auswertung(Protokoll().zeilen())


@echtes_protokoll
def test_echt_die_gueltigen_und_ungueltigen_durchgaenge_des_hauptexperiments(echte_auswertung):
    assert [d.nummer for d in echte_auswertung.durchgaenge] == [
        1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 15, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27
    ]
    ungueltig = echte_auswertung.ungueltige_durchgaenge()
    assert list(ungueltig["durchgang"]) == [9, 10, 11, 14, 16, 22]
    je_seed = ungueltig.groupby("seed")["dollar"].sum()
    # Genau 15,358 $, auf Cent gerundet 15,36 $.
    assert je_seed[2] == pytest.approx(15.358, abs=0.001)
    assert je_seed[3] == pytest.approx(0.77, abs=0.005)


@echtes_protokoll
def test_echt_jedes_paar_hat_genau_eine_gueltige_testbewertung(echte_auswertung):
    tests = echte_auswertung.testbewertungen()
    assert len(tests) == 39
    assert tests["paar"].is_unique
    assert set(tests["testbeispiele"]) == {500}
    assert echte_auswertung.wiederholte_testbewertungen() == 0


@echtes_protokoll
def test_echt_nachgerechnet_entscheidet_jede_auswahl_wie_im_protokoll(echte_auswertung):
    auswahlen = echte_auswertung.auswahlen()
    assert (auswahlen["protokolliert"] == auswahlen["nachgerechnet"]).all()


@echtes_protokoll
def test_echt_unter_der_obergrenze_heisst_beim_einzelmodell_mit_gewinner(echte_auswertung):
    """Die Spalte ersetzt das Etikett "kein Gewinner" und muss deshalb dasselbe sagen wie die
    Endauswahl im Protokoll."""
    programme = echte_auswertung.programme()
    einzelmodelle = programme[programme["baseline"] == "ein-modell"]
    gewinner = {
        d.nummer: d.entscheidung["gewinner"] is not None
        for d in echte_auswertung.durchgaenge
        if d.baseline == "ein-modell"
    }
    assert len(einzelmodelle) == 12
    assert list(einzelmodelle["unter_grenze"]) == [
        gewinner[nummer] for nummer in einzelmodelle["durchgang"]
    ]


@echtes_protokoll
def test_echt_ohne_kostenfilter_ginge_jedes_mal_deepseek_weiter(echte_auswertung):
    ohne = echte_auswertung.zwischenbewertungen(kostenobergrenze=math.inf)
    assert list(ohne["weiter"]) == [DEEPSEEK] * 3
    assert [round(p, 4) for p in ohne["p_wert openai/gpt-oss-120b"]] == [0.0026, 0.0004, 0.0006]
    assert [round(p, 4) for p in ohne["p_wert google/gemma-4-31b-it"]] == [0.0351, 0.0143, 0.0011]
    assert (ohne["p_wert openai/gpt-6-luna"] < 0.0001).all()


@echtes_protokoll
def test_echt_die_gleichverteilung_in_seed_3_zum_festgehaltenen_preis(echte_auswertung):
    """Zum Preis vom Stichtag laege DeepSeek dort bei rund 0,23 $ je 1000 und waere gewaehlt
    worden."""
    auswahlen = echte_auswertung.auswahlen().set_index("durchgang")
    assert auswahlen.loc[23, "protokolliert"] == "openai/gpt-oss-120b"
    assert auswahlen.loc[23, "festgehaltener_preis"] == DEEPSEEK
    staende = echte_auswertung.staende(23, phase=None, preis="festgehalten")
    (deepseek,) = [stand for stand in staende if stand.name == DEEPSEEK]
    assert deepseek.ausfuehrungskosten_je_1000 == pytest.approx(0.23, abs=0.005)


@echtes_protokoll
def test_echt_ff1_je_seed_ein_tau(echte_auswertung):
    ff1 = echte_auswertung.ff1()
    assert list(ff1["seed"]) == [1, 2, 3]
    assert ff1["tau"].between(-1, 1).all()


@echtes_protokoll
def test_echt_endqualitaet_je_strategie(echte_auswertung):
    """Je Seed eine Bewertung auf den Testdaten je Strategie: ohne Optimierung, die vier
    Einzelmodelle, die Gleichverteilung in beiden Preisfassungen und das Verfahren."""
    endqualitaet = echte_auswertung.endqualitaet()
    assert len(endqualitaet) == 8 * 3
    mittel = endqualitaet.groupby("strategie", sort=False)["bewertung_test"].mean()
    assert list(mittel.index) == [
        "Ohne Optimierung",
        "Einzelmodell: gpt-oss-120b",
        "Einzelmodell: Gemma 4 31B",
        "Einzelmodell: DeepSeek V4.1 Flash",
        "Einzelmodell: GPT-6 Luna",
        "Gleichverteilung (abgerechnet)",
        "Gleichverteilung (Preis vom Stichtag)",
        "Verfahren",
    ]
    assert mittel["Verfahren"] == pytest.approx(0.805, abs=0.0005)
    assert mittel["Gleichverteilung (abgerechnet)"] == pytest.approx(0.783, abs=0.0005)
    assert mittel["Gleichverteilung (Preis vom Stichtag)"] == pytest.approx(0.803, abs=0.0005)


@echtes_protokoll
def test_echt_schreibt_tabellen_abbildungen_und_bericht(echte_auswertung, tmp_path):
    echte_auswertung.schreibe(tmp_path)
    assert (tmp_path / "bericht.md").exists()
    for abbildung in (
        "endqualitaet", "pareto-abgerechnet", "pareto-festgehalten", "lernkurven", "ff1-rangfolge"
    ):
        assert (tmp_path / f"{abbildung}.pdf").exists()
        assert (tmp_path / f"{abbildung}.png").exists()
    assert list(tmp_path.glob("*.csv"))
