"""Das fortlaufende Protokoll: je Messlauf eine Zeile, ausschliesslich angehaengt.

Das Protokoll wird ab dem ersten API-Aufruf gefuehrt und laesst sich nachtraeglich nicht
rekonstruieren: die Dollar-Betraege je Aufruf liegen nur im Arbeitsspeicher des laufenden
Prozesses, und MIPROv2 verliert bei einem Abbruch alle bis dahin berechneten Bewertungen.

Das Format ist JSON Lines: je Zeile ein vollstaendiges JSON-Objekt. Angehaengt wird mit
`flush` und `fsync`, damit die Zeile auch dann auf der Platte steht, wenn der Prozess
unmittelbar danach beendet wird. Eine unvollstaendige letzte Zeile, etwa nach einem Absturz
waehrend des Schreibens, laesst die frueheren Zeilen lesbar.

Geschrieben wird die Zeile von `dspy_optimizer.kostenerfassung.messlauf`, auch wenn der
Lauf mit einem Fehler abbricht, da die angefallenen Kosten trotzdem erfasst werden muessen.
"""

import dataclasses
import datetime
import json
import os
import pathlib
import re

WURZEL = pathlib.Path(__file__).resolve().parent.parent
PROTOKOLLDATEI = WURZEL / "ergebnisse" / "protokoll.jsonl"
LAEUFE = WURZEL / "ergebnisse" / "laeufe"

# Jede Zeile haelt zwei Angaben getrennt fest: welcher Teil eines Laufs sie ist (der
# Abschnitt) und wozu der Lauf diente (der Zweck). So kann etwa eine Optimierung zugleich
# zur Vorstudie gehoeren.
#
# Die Abschnitte; jeder entspricht einem Messlauf:
#   optimierung   - der MIPROv2-Lauf; seine Kosten sind Optimierungskosten.
#   nachbewertung - die vollstaendige Bewertung direkt nach der Optimierung; daraus ergeben
#                   sich die Ausfuehrungskosten je Programmlauf.
#   testbewertung - die einmalige Bewertung eines fertigen Paars auf den Testdaten.
#   fixkosten     - die Fixkosten-Messung der Vorstudie: nur der feste Teil eines
#                   MIPROv2-Laufs, ohne Suche und ohne Nachbewertung. Ein eigener
#                   Abschnitt, damit sie nicht in die Steigung des Kostenmodells eingeht.
# Optimierungs- und Ausfuehrungskosten werden getrennt erfasst und nicht miteinander
# verrechnet.
ABSCHNITTE = ("optimierung", "nachbewertung", "testbewertung", "fixkosten")

# Die Abschnitte, in denen MIPROv2 laeuft und deshalb ein fester Kostenanteil anfaellt: nur
# in ihren Zeilen stehen gemessene Fixkosten.
MIT_FIXKOSTEN = ("optimierung", "fixkosten")

# Die Entscheidungen des Verfahrens und der Baselines. Sie sind kein Messlauf und verursachen
# keine Kosten, stehen aber als eigene Zeile im Protokoll (wer weiterkommt, wer ausscheidet
# und warum), weil die Auswertung ausschliesslich das Protokoll liest.
ENTSCHEIDUNGEN = ("zwischenbewertung", "endauswahl")

# Die Zwecke:
#   vorstudie  - Messlaeufe vor dem Hauptexperiment; nur aus ihnen wird kalibriert.
#   handlauf   - ein kleiner Lauf zum Hinsehen, der weder zur Vorstudie noch zum
#                Hauptexperiment zaehlt.
#   experiment - ein Messlauf des Hauptexperiments.
ZWECKE = ("vorstudie", "handlauf", "experiment")

# Wer im Experiment antritt: das Verfahren selbst oder eine der Baselines.
# Der Wert `teure-referenz` ist im Format
# vorgesehen; die teure Referenz wird im Experiment aber nicht ausgefuehrt.
BASELINES = ("verfahren", "gleichverteilung", "ein-modell", "ohne-optimierung", "teure-referenz")

# Phasen hat nur das Verfahren.
PHASEN = ("sondierung", "fokus")


@dataclasses.dataclass(frozen=True)
class Einordnung:
    """Wozu ein Messlauf gehoert: Zweck und, im Experiment, Baseline, Phase und Durchgang.

    Geprueft wird bei der Erzeugung, nicht erst beim Schreiben: die Einordnung entsteht vor
    dem ersten bezahlten Aufruf, die Zeile erst danach, und eine unvollstaendige Zeile liesse
    sich spaeter nicht mehr ergaenzen. Die Klasse ist unveraenderlich, damit eine gepruefte
    Einordnung nicht nachtraeglich ungueltig wird. Erzeugt wird sie ueblicherweise ueber
    `VORSTUDIE`, `HANDLAUF` und `Einordnung.experiment(...)`.
    """

    zweck: str
    baseline: str | None = None
    phase: str | None = None
    durchgang: int | None = None

    def __post_init__(self):
        if self.zweck not in ZWECKE:
            raise ValueError(_unbekannt(self.zweck, "kein Zweck", ZWECKE))
        if self.zweck == "experiment":
            _pruefe_experiment(self.baseline, self.phase, self.durchgang)
            return
        for name in ("baseline", "phase", "durchgang"):
            wert = getattr(self, name)
            if wert is not None:
                raise ValueError(
                    f"{name.capitalize()} {wert!r} gehoert nur zum Zweck experiment, nicht zu "
                    f"{self.zweck}: eine Vorstudie oder ein Handlauf ist Teil keines Durchgangs."
                )

    @classmethod
    def experiment(cls, baseline, *, durchgang, phase=None):
        """Ein Messlauf des Hauptexperiments; `phase` nur beim Verfahren."""
        return cls("experiment", baseline=baseline, phase=phase, durchgang=durchgang)

    def als_felder(self):
        """Die Felder der Einordnung; sie stehen in jeder Zeile und bleiben leer, wo sie nicht
        zutreffen."""
        return dataclasses.asdict(self)


def _unbekannt(wert, was, moeglich):
    return f"{wert!r} ist {was}. Moeglich sind: {', '.join(moeglich)}"


def _pruefe_experiment(baseline, phase, durchgang):
    """Prueft Baseline, Durchgang und Phase eines Messlaufs im Experiment.

    Ohne diese Angaben koennte die Auswertung weder die Phasen vergleichen (FF1) noch die
    Kosten eines Durchgangs summieren (FF3)."""
    if baseline is None:
        raise ValueError(
            f"Im Experiment fehlt die Baseline. Moeglich sind: {', '.join(BASELINES)}"
        )
    if baseline not in BASELINES:
        raise ValueError(_unbekannt(baseline, "keine Baseline", BASELINES))
    # `bool` ist in Python ein `int`; `True` als Durchgang 1 waere ein stiller Tippfehler.
    if not isinstance(durchgang, int) or isinstance(durchgang, bool) or durchgang < 1:
        raise ValueError(
            f"Der Durchgang muss eine ganze Zahl ab 1 sein, nicht {durchgang!r}: ohne ihn "
            "lassen sich die Kosten eines Durchgangs nicht summieren (ADR-0019)."
        )
    if baseline == "verfahren":
        if phase not in PHASEN:
            raise ValueError(
                f"Phase {phase!r}: ein Messlauf des Verfahrens gehoert zu einer Phase. "
                f"Moeglich sind: {', '.join(PHASEN)}"
            )
    elif phase is not None:
        raise ValueError(
            f"Phase {phase!r}: nur das Verfahren hat Phasen, die Baseline {baseline} nicht."
        )


VORSTUDIE = Einordnung("vorstudie")
HANDLAUF = Einordnung("handlauf")


def _pruefe_ist_einordnung(einordnung):
    if not isinstance(einordnung, Einordnung):
        raise ValueError(
            f"{einordnung!r} ist keine Einordnung. Gebraucht wird ein `Einordnung`-Objekt, "
            "etwa `VORSTUDIE`, `HANDLAUF` oder `Einordnung.experiment(...)` (ADR-0019)."
        )


def pruefe_einordnung(abschnitt, einordnung):
    """Prueft, ob Abschnitt und Einordnung zusammenpassen.

    Wird vor dem ersten bezahlten Aufruf aufgerufen und beim Schreiben der Zeile erneut.

    Ohne Abschnitt ist ein Messlauf nur als Handlauf oder als Vorstudie zulaessig, die keinem
    der Abschnitte entspricht, etwa der kleine Lauf ueber fuenf Trainingsbeispiele oder der
    Kostenabgleich, der Trainingsdaten bewertet und deshalb keine Nachbewertung ist. Zur
    Kalibrierung taugt eine solche Vorstudie nicht, dafuer ist der Abschnitt Optimierung
    noetig. Im Experiment gehoert jeder Messlauf zu einem Abschnitt, sonst liessen sich
    Optimierungs- und Ausfuehrungskosten eines Durchgangs nicht trennen.

    Eine Testbewertung gibt es nur im Experiment, damit die Testdaten bis zur abschliessenden
    Auswertung unberuehrt bleiben. Eine
    Fixkosten-Messung gibt es nur in der Vorstudie: sie liefert die Fixkosten fuer das
    Kostenmodell und ist kein Optimierungslauf.
    """
    _pruefe_ist_einordnung(einordnung)
    if abschnitt is None:
        if einordnung.zweck == "experiment":
            raise ValueError(
                "Ein Messlauf des Experiments braucht einen Abschnitt. Ohne geht es nur beim "
                "Handlauf oder bei einer Vorstudie."
            )
    elif abschnitt not in ABSCHNITTE:
        raise ValueError(_unbekannt(abschnitt, "kein Abschnitt", ABSCHNITTE))
    elif abschnitt == "testbewertung" and einordnung.zweck != "experiment":
        raise ValueError(
            f"Eine Testbewertung mit Zweck {einordnung.zweck}: die Testdaten bleiben bis zur "
            "abschliessenden Auswertung unberuehrt und werden nur im Hauptexperiment bewertet "
            "(ADR-0008)."
        )
    elif abschnitt == "fixkosten" and einordnung.zweck != "vorstudie":
        raise ValueError(
            f"Eine Fixkosten-Messung mit Zweck {einordnung.zweck}: sie misst die Fixkosten "
            "fuer das Kostenmodell und gehoert nur in die Vorstudie (ADR-0014, Nachtrag "
            "10.10.2026)."
        )


def kalibrierbar(zeile):
    """Ob eine Protokollzeile zur Kalibrierung des Kostenmodells verwendet werden darf.

    Bedingungen: Zweck Vorstudie, Abschnitt Optimierung, Bewertung nicht leer. Die Bewertung
    kennzeichnet einen vollstaendig abgeschlossenen Lauf; ein abgebrochener Lauf hinterlaesst
    ebenfalls eine Zeile und wuerde die Ausgleichsgerade nach unten ziehen. Zeilen ohne Zweck
    stammen aus einem aelteren Protokollformat und zaehlen nicht, auch nicht der fruehe
    Handlauf vom 20.09.2026, der als `optimierung` im Protokoll steht.
    """
    return (
        zeile.get("zweck") == "vorstudie"
        and zeile.get("abschnitt") == "optimierung"
        and zeile.get("bewertung") is not None
    )


def _jetzt():
    """Der Zeitstempel einer Zeile: lokale Zeit mit Zeitzone, sekundengenau."""
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


class Protokoll:
    """Eine Protokolldatei, die nur waechst."""

    def __init__(self, pfad=PROTOKOLLDATEI):
        self.pfad = pathlib.Path(pfad)

    def schreibe(
        self,
        *,
        abschnitt,
        einordnung,
        konfiguration,
        seed,
        bewertung,
        messwerte,
        plan=None,
        exakt=None,
        absatztreffer=None,
        einzelbewertungen=None,
    ):
        """Eine Zeile anhaengen und zurueckgeben.

        Alle Felder sind Pflichtargumente, weil sich eine unvollstaendige Zeile spaeter nicht
        mehr ergaenzen liesse. `messwerte` ist ein `dspy_optimizer.kostenerfassung.Messwerte`.

        `plan` ist ein `dspy_optimizer.kostenmodell.Plan`, sofern der Lauf geplant war. Dann
        enthaelt die Zeile zusaetzlich die geplante Trial-Zahl, die geplanten Kosten dieses
        Messlaufs, die Planabweichung und die Planungsparameter. Geplant werden beide
        Messlaeufe eines Optimierungslaufs, Optimierung und Nachbewertung, jeder gegen seinen
        eigenen Teil des Plans. Ein Handlauf
        wird nicht geplant und enthaelt diese Felder nicht.

        `bewertung` ist der F1-Score, das Optimierungsziel. `exakt` (Exact Match) und
        `absatztreffer` werden daneben protokolliert, damit je Lauf alle drei Kennzahlen der
        Metrik vorliegen. Sie bleiben leer, wo sie nicht gemessen wurden, etwa
        in einer Optimierungszeile, fuer die nur der F1-Score von MIPROv2 vorliegt.

        `einzelbewertungen` sind die F1-Werte je Beispiel in der Reihenfolge der Daten. Sie
        stehen bei Nachbewertung und Testbewertung, weil die Auswertung nur das Protokoll
        liest und der gepaarte Test ohne sie nicht berechnet werden kann.

        `dollar_je_programmlauf` bleibt bei einem Optimierungslauf und einer
        Fixkosten-Messung leer, weil der Wert dort keine Ausfuehrungskosten darstellt.
        Umgekehrt stehen `fixkosten` nur in Zeilen, in denen MIPROv2 lief,
        und bleiben in jeder Bewertung leer. Die Feldnamen sind in allen Zeilen gleich, damit
        die Auswertung jede Groesse aus einer einzigen Spalte lesen kann.
        """
        pruefe_einordnung(abschnitt, einordnung)

        felder = messwerte.als_felder()
        if abschnitt not in MIT_FIXKOSTEN:
            felder["fixkosten"] = None
        else:
            # Wo MIPROv2 lief (Optimierung oder Fixkosten-Messung), beziehen sich die
            # gezaehlten Programmlaeufe auf Zwischenkandidaten, und die Laeufe des
            # Bootstrappings fehlen in der Zahl ganz. Die Dollar je Programmlauf waeren hier
            # keine Ausfuehrungskosten, liegen aber nahe am Messwert der Nachbewertung. Damit
            # die beiden Werte beim Auswerten nicht verwechselt werden, bleibt das Feld leer.
            felder["dollar_je_programmlauf"] = None

        zeile = {
            "zeitpunkt": _jetzt(),
            "abschnitt": abschnitt,
            **einordnung.als_felder(),
            "konfiguration": konfiguration,
            "seed": seed,
            "bewertung": bewertung,
            "exakt": exakt,
            "absatztreffer": absatztreffer,
            "einzelbewertungen": None if einzelbewertungen is None else list(einzelbewertungen),
            **felder,
            **(
                plan.abweichung(abschnitt, messwerte.dollar).als_felder()
                if plan is not None
                else {}
            ),
        }
        self._haenge_an(zeile)
        return zeile

    def schreibe_entscheidung(self, *, abschnitt, einordnung, seed, entscheidung):
        """Eine Entscheidungszeile anhaengen und zurueckgeben.

        `entscheidung` ist ein Woerterbuch aus reinen JSON-Werten; den Inhalt legt die
        aufrufende Entscheidungsregel fest. Die Zeile enthaelt keine Dollar-Angabe: eine
        Entscheidung verursacht keine Kosten, und ein Feld `dollar` mit 0,0 saehe aus wie ein
        kostenloser Messlauf.
        """
        if abschnitt not in ENTSCHEIDUNGEN:
            raise ValueError(_unbekannt(abschnitt, "keine Entscheidung", ENTSCHEIDUNGEN))
        _pruefe_ist_einordnung(einordnung)

        kopf = {
            "zeitpunkt": _jetzt(),
            "abschnitt": abschnitt,
            **einordnung.als_felder(),
            "seed": seed,
        }
        doppelt = sorted(set(kopf) & set(entscheidung))
        if doppelt:
            raise ValueError(
                f"Die Entscheidung bringt die Felder {', '.join(doppelt)} mit. Sie ersetzten "
                "still die gepruefte Einordnung der Zeile (ADR-0019)."
            )
        zeile = {**kopf, **entscheidung}
        self._haenge_an(zeile)
        return zeile

    def _haenge_an(self, zeile):
        """Die Zeile anhaengen und sofort auf die Platte schreiben (`flush`, `fsync`).

        `default=str` faengt Werte ab, die sich nicht als JSON schreiben lassen: ein Messlauf
        soll nicht daran scheitern, dass die Konfiguration ein nicht serialisierbares Objekt
        enthaelt.
        """
        self.pfad.parent.mkdir(parents=True, exist_ok=True)
        with open(self.pfad, "a", encoding="utf-8") as datei:
            datei.write(json.dumps(zeile, ensure_ascii=False, default=str) + "\n")
            datei.flush()
            os.fsync(datei.fileno())

    def zeilen(self):
        """Alle lesbaren Zeilen, in der Reihenfolge, in der sie geschrieben wurden.

        Eine unvollstaendige Zeile wird uebersprungen statt einen Fehler auszuloesen: sie ist
        der Rest eines abgestuerzten Laufs, die Zeilen davor sind vollstaendig und gueltig.
        """
        if not self.pfad.exists():
            return []
        gelesen = []
        for zeile in self.pfad.read_text(encoding="utf-8").splitlines():
            try:
                gelesen.append(json.loads(zeile))
            except json.JSONDecodeError:
                continue
        return gelesen

    def kalibrierbare_zeilen(self):
        """Genau die Zeilen, aus denen das Kostenmodell kalibriert werden darf (`kalibrierbar`).

        Aus ihnen stammt nur die Steigung, also die Kosten je Trial; die Fixkosten stammen aus
        den Zeilen mit Abschnitt `fixkosten`.

        Gefiltert wird nur nach dieser Regel, nicht nach Kandidatenmodell: das Kostenmodell
        gilt je Kandidatenmodell, die Gruppierung nimmt die Vorstudie selbst vor.
        """
        return [zeile for zeile in self.zeilen() if kalibrierbar(zeile)]


def verzeichnis_fuer_lauf(name, wurzel=LAEUFE):
    """Legt ein neues Verzeichnis fuer einen Lauf an; es dient MIPROv2 als `log_dir`.

    MIPROv2 legt dort je Trial das Kandidatenprogramm ab. Das ist die einzige Moeglichkeit,
    nach einem Abbruch Zwischenergebnisse zu sichern, allerdings ohne die Bewertungen, die DSPy
    nicht speichert.
    """
    sauber = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")
    stempel = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    verzeichnis = pathlib.Path(wurzel) / f"{stempel}-{sauber}"
    nummer = 2
    while verzeichnis.exists():
        verzeichnis = pathlib.Path(wurzel) / f"{stempel}-{sauber}-{nummer}"
        nummer += 1
    verzeichnis.mkdir(parents=True)
    return verzeichnis
